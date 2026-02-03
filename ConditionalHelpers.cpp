/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ConditionalHelpers.hpp"
#include "ModelImporter.hpp"
#include "importerUtils.hpp"
#include "toposort.hpp"

namespace onnx2trt
{

// Search for a network Layer name in a SubgraphPortsMap.
SubgraphPortsMap::const_iterator findLayer(const SubgraphPortsMap& inputs, const std::string layerName)
{
    return std::find_if(
        inputs.begin(), inputs.end(), [&](const auto& item) { return layerName == item.first->getName(); });
}

// Add an ConditionalInputLayer between `layer` and its inputs.
// I.e. input[inIdx] -> layer ==> input[inIdx] -> ConditionalInputLayer -> layer.
void addConditionalInputLayer(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    nvinfer1::ILayer& layer, int32_t inIdx, ::ONNX_NAMESPACE::NodeProto const* node)
{
    auto input = layer.getInput(inIdx);
    if (input == nullptr)
    {
        // Phantom input (an input that is really constant weights).
        return;
    }

    if (layer.getType() == nvinfer1::LayerType::kCONDITIONAL_OUTPUT)
    {
        return;
    }

    auto const name = input->getName();
    auto it = inputsMap.find(name);
    nvinfer1::IIfConditionalInputLayer* inputLayer = nullptr;
    if (it == inputsMap.end())
    {
        inputLayer = N_CHECK(conditional->addInput(*input));
        inputsMap[name] = inputLayer;
        const std::string inputLayerName(name);
        ctx->registerLayer(inputLayer, inputLayerName + "_InputLayer", node);
        // Note: Since multiple conditionals may use the same external tensor, check unique names for output tensors of
        // IfConditionalInputLayers to avoid tensor name duplication.
        ctx->registerTensor(
            TensorOrWeights{N_CHECK(inputLayer->getOutput(0))}, inputLayerName + "_InputLayer_output", /*checkUniqueName*/ true);
    }
    else
    {
        // An InputLayer may in the inputsMap if it has several consumers.
        inputLayer = it->second;
    }
    auto ifOutput = N_CHECK(inputLayer->getOutput(0));
    layer.setInput(inIdx, *ifOutput);
}

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
// subgraphParentIdx is the index of the parent node in the main graph, used for error reporting.
void importSubgraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, std::vector<TensorOrWeights>& subgraphTensors, int32_t subgraphParentIdx)
{
    auto net = ctx->network();
    int32_t beforeSubgraph = net->getNbLayers();

    // Establish scope for names local to the subgraph.
    NameScope nameScope(*ctx);

    std::vector<Status> errors{};
    onnx2trt::parseGraph(
        ctx, subgraph, errors, /*deserializingINetwork=*/false, /*currentNode=*/nullptr, subgraphParentIdx);

    for (int32_t i = 0; i < subgraph.output_size(); ++i)
    {
        std::string name = subgraph.output(i).name();
        subgraphTensors.push_back(ctx->tensors().at(name));
    }

    for (int32_t i = beforeSubgraph; i < net->getNbLayers(); i++)
    {
        newLayers.push_back(net->getLayer(i));
    }
}

// Add an IConditionalInputLayer to `layer`'s inputs, if they don't already exist.
void addConditionalInputIfNeeded(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    nvinfer1::ILayer& layer, SubgraphPortsMap subgraphInputsMap, ::ONNX_NAMESPACE::NodeProto const* node)
{
    // Return all of the layer's inputs that are external to the subgraph that
    // that the layer belongs to.
    auto getLayerExternalInputs = [&](std::string const& layerName) {
        std::set<int32_t> inIndices;
        auto iter = findLayer(subgraphInputsMap, layerName);
        if (iter != subgraphInputsMap.end())
        {
            const auto& indicesSet = iter->second;
            inIndices.insert(indicesSet.begin(), indicesSet.end());
        }

        return inIndices;
    };

    const auto inIndices = getLayerExternalInputs(layer.getName());
    for (auto inIdx : inIndices)
    {
        LOG_VERBOSE("Adding Input layer for " << layer.getName());
        addConditionalInputLayer(ctx, conditional, inputsMap, layer, inIdx, node);
    }
}

// Add IConditionalInputLayers to `layer`'s inputs.
void addIfInputLayers(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers, ::ONNX_NAMESPACE::NodeProto const* node)
{
    // Find all of the tensors entering the subgraph.
    SubgraphPortsMap externalInputs;
    getSubgraphInputs(newLayers, externalInputs);

    // Add a ConditionalInputLayer in front of each input that is external to the subgraph.
    for (const auto& layer : newLayers)
    {
        addConditionalInputIfNeeded(ctx, conditional, inputsMap, *layer, externalInputs, node);
    }
}

// Given a subgraph, find all of its external inputs (tensors entering the subgraph).
void getSubgraphInputs(const std::vector<nvinfer1::ILayer*>& newLayers, SubgraphPortsMap& externalInputs)
{
    using PortIndex = int32_t;
    using TensorsSet = std::unordered_set<nvinfer1::ITensor*>;
    TensorsSet outputTensors;
    TensorsSet inputTensors;

    // To determine which tensors are entering or exiting the given graph, we first collect the sets of all input and
    // output tensors. Then we categorize the tensors according to this logic:
    //  Entering tensors := {inputs} - {outputs}
    //  Exiting tensors := {outputs} - {inputs}

    // Collect all input and output tensors belonging to nodes in the graph.

    auto getTensors = [](nvinfer1::ILayer const* l, bool const input, auto inserter) {
        auto const count = input ? l->getNbInputs() : l->getNbOutputs();
        for (int32_t i = 0; i < count; i++)
        {
            inserter(input ? l->getInput(i) : l->getOutput(i));
        }
    };

    for (const auto& l : newLayers)
    {
        getTensors(l, false, [&](nvinfer1::ITensor* t) { outputTensors.insert(t); });
        getTensors(l, true, [&](nvinfer1::ITensor* t) { inputTensors.insert(t); });
    }

    using TensorsVec = std::vector<nvinfer1::ITensor*>;
    auto getInputs = [&](nvinfer1::ILayer const* l, TensorsVec& res) {
        getTensors(l, true, [&](nvinfer1::ITensor* t) { res.emplace_back(t); });
    };

    // Retrieve the list of tensors either exiting or entering the subgraph.
    auto filterTensors = [&](TensorsSet const& tensors, auto getNodeAccessor) {
        for (nvinfer1::ILayer const* l : newLayers)
        {
            PortIndex i = 0;

            TensorsVec nodeAccessor;
            getNodeAccessor(l, nodeAccessor);
            for (const auto& tensor : nodeAccessor)
            {
                if (tensor == nullptr)
                {
                    continue;
                }
                if (tensors.count(tensor) == 0)
                {
                    externalInputs[l].insert(i);
                }
                i++;
            }
        }
    };

    filterTensors(outputTensors, getInputs);
}

} // namespace onnx2trt
