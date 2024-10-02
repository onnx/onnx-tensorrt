/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ConditionalHelpers.hpp"
#include "ModelImporter.hpp"
#include "importerUtils.hpp"
#include "toposort.hpp"

namespace onnx2trt
{

using NodeName = std::string;
using LayerName = std::string;
using InputIndex = int32_t;

// A SubgraphPortsMap maps either the inputs or outputs ports of each node in an ONNX graph.
using SubgraphPortsMap = std::unordered_map<nvinfer1::ITensor*, std::set<InputIndex>>;

// An InputsMap tracks which IIfConditionalInputLayer we've added to a layer's inputs,
// so that we can reuse them if needed.
using InputsMap = std::unordered_map<LayerName, nvinfer1::IIfConditionalInputLayer*>;

// Search for a network Layer name in a SubgraphPortsMap using partial (prefix) name matching.
// ONNX nodes are matched to network layers using prefix-matching because an ONNX node may have
// several network layers associcated with it.
SubgraphPortsMap::const_iterator findLayer(const SubgraphPortsMap& inputs, const std::string layerName)
{
    return std::find_if(inputs.begin(), inputs.end(), [&](const auto& item) {
        std::string const key = item.first->getName();
        return layerName.compare(0, key.size(), key) == 0;
    });
}

// Add an ConditionalInputLayer between `layer` and its inputs.
// I.e. input[inIdx] -> layer ==> input[inIdx] -> ConditionalInputLayer -> layer.
void addConditionalInputLayer(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    nvinfer1::ILayer& layer, int32_t inIdx)
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
        ctx->registerLayer(inputLayer, inputLayerName + "_InputLayer", nullptr);
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
};

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
void importSubgraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, std::vector<TensorOrWeights>& subgraphTensors)
{
    auto net = ctx->network();
    int32_t beforeSubgraph = net->getNbLayers();

    // Establish scope for names local to the subgraph.
    NameScope nameScope(*ctx);

    std::vector<Status> errors{};
    onnx2trt::parseGraph(ctx, subgraph, errors);

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
    nvinfer1::ILayer& layer, SubgraphPortsMap subgraphInputsMap)
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
        addConditionalInputLayer(ctx, conditional, inputsMap, layer, inIdx);
    }
}

// Add IConditionalInputLayers to `layer`'s inputs.
void addIfInputLayers(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers)
{
    // Find all of the tensors entering the subgraph.
    // The node-names are from the ONNX context.
    using InputIndex = int32_t;
    std::unordered_map<nvinfer1::ITensor*, std::set<InputIndex>> subgraphInputsMap;
    getSubgraphInputs(newLayers, subgraphInputsMap);

    // Add a ConditionalInputLayer in front of each input that is external to the subgraph.
    for (const auto& layer : newLayers)
    {
        addConditionalInputIfNeeded(ctx, conditional, inputsMap, *layer, subgraphInputsMap);
    }
}

// Given a subgraph, find all of its external inputs/outputs (tensors entering/exiting the subgraph).
void getSubgraphTensors(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<nvinfer1::ITensor*, std::set<int32_t>>& externalOutputs, bool extractOutputs,
    const std::vector<std::string>* reportedOutputs = nullptr)
{
    using NodeName = std::string;
    using TensorName = std::string;
    using PortIndex = int32_t;
    using Port = std::pair<NodeName, PortIndex>;
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
    auto getOutputs = [&](nvinfer1::ILayer const* l, TensorsVec& res) {
        getTensors(l, false, [&](nvinfer1::ITensor* t) { res.emplace_back(t); });
    };

    auto getInputs = [&](nvinfer1::ILayer const* l, TensorsVec& res) {
        getTensors(l, true, [&](nvinfer1::ITensor* t) { res.emplace_back(t); });
    };

    // Retrieve the list of tensors either exiting or entering the subgraph.
    std::unordered_map<nvinfer1::ITensor*, std::vector<Port>> externalPortsMap;
    auto filterTensors = [&](TensorsSet const& tensors, auto getNodeAccessor) {
        for (nvinfer1::ILayer const* l : newLayers)
        {
            const auto& nodeName = l->getName();
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
                    TensorName tensorName = tensor->getName();
                    auto prefixFound = false;
                    if (reportedOutputs)
                    {
                        // reportedOutputs are the names of the outputs as reported by the ONNX parser and help
                        // us further filter the output tensors.
                        //      Exiting tensors := {outputs} - {inputs} - {unreported tensors}
                        // An example: a Split node is internal to a subgraph and has 4 outputs, but only two are
                        // connected to the rest of the graph.  To prevent mistaking the 2 unused outputs as subgraph
                        // outputs, we look for them in reportedOutputs which leads us to ignore the 2 tensors.
                        const auto iter = std::find_if(
                            reportedOutputs->begin(), reportedOutputs->end(), [&](const auto& outputName) {
                                // Prefix name matching.
                                return tensorName.compare(0, outputName.size(), outputName) == 0;
                            });
                        prefixFound = iter != reportedOutputs->end();
                    }
                    if (!reportedOutputs || prefixFound)
                    {
                        externalPortsMap[tensor].push_back(std::make_pair(nodeName, i));
                    }
                }
                i++;
            }
        }
    };

    if (extractOutputs)
    {
        filterTensors(inputTensors, getOutputs);
    }
    else
    {
        filterTensors(outputTensors, getInputs);
    }

    // Create the user's view of the external inputs, which uses the node-name as the key for
    // looking up input/output port index.
    for (auto const& input : externalPortsMap)
    {
        for (const Port& inPort : input.second)
        {
            auto* tensor = input.first;
            auto const portIndex = inPort.second;
            externalOutputs[tensor].insert(portIndex);
        }
    }
}

void getSubgraphOutputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<nvinfer1::ITensor*, std::set<int32_t>>& externalOutputs,
    const std::vector<std::string>& reportedOutputs)
{
    getSubgraphTensors(newLayers, externalOutputs, true, &reportedOutputs);
}

void getSubgraphInputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<nvinfer1::ITensor*, std::set<int32_t>>& externalInputs)
{
    getSubgraphTensors(newLayers, externalInputs, false);
}

} // namespace onnx2trt
