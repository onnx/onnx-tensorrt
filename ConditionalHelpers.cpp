/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ConditionalHelpers.hpp"
#include "ModelImporter.hpp"
#include "onnx2trt_utils.hpp"
#include "toposort.hpp"

namespace onnx2trt
{

using NodeName = std::string;
using LayerName = std::string;
using InputIndex = int32_t;

// A SubgraphPortsMap maps either the inputs or outputs ports of each node in an ONNX graph.
using SubgraphPortsMap = std::unordered_map<NodeName, std::set<InputIndex>>;

// An InputsMap tracks which IIfConditionalInputLayer we've added to a layer's inputs,
// so that we can reuse them if needed.
using InputsMap = std::unordered_map<LayerName, nvinfer1::IIfConditionalInputLayer*>;

// Search for a network Layer name in a SubgraphPortsMap using partial (prefix) name matching.
// ONNX nodes are matched to network layers using prefix-matching because an ONNX node may have
// several network layers associcated with it.
SubgraphPortsMap::const_iterator findLayer(const SubgraphPortsMap& inputs, const std::string layerName)
{
    return std::find_if(inputs.begin(), inputs.end(), [&](const auto& item) {
        const auto& key = item.first;
        return layerName.compare(0, key.size(), key) == 0;
    });
}

// Add an ConditionalInputLayer between `layer` and its inputs.
// I.e. input[inIdx] -> layer ==> input[inIdx] -> ConditionalInputLayer -> layer.
Status addConditionalInputLayer(IImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    nvinfer1::ILayer& layer, int32_t inIdx)
{
    auto input = layer.getInput(inIdx);
    if (input == nullptr)
    {
        // Phantom input (an input that is really constant weights).
        return Status::success();
    }

    if (layer.getType() == nvinfer1::LayerType::kCONDITIONAL_OUTPUT)
    {
        return Status::success();
    }

    auto const name = input->getName();
    auto it = inputsMap.find(name);
    nvinfer1::IIfConditionalInputLayer* inputLayer = nullptr;
    if (it == inputsMap.end())
    {
        inputLayer = conditional->addInput(*input);
        inputsMap[name] = inputLayer;
        const std::string inputLayerName(name);
        ctx->registerLayer(inputLayer, inputLayerName + "_InputLayer");
        // Note: Since multiple conditionals may use the same external tensor, check unique names for output tensors of
        // IfConditionalInputLayers to avoid tensor name duplication.
        ctx->registerTensor(
            TensorOrWeights{inputLayer->getOutput(0)}, inputLayerName + "_InputLayer_output", /*checkUniqueName*/ true);
    }
    else
    {
        // An InputLayer may in the inputsMap if it has several consumers.
        inputLayer = it->second;
    }
    layer.setInput(inIdx, *(inputLayer->getOutput(0)));
    return Status::success();
};

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
Status importSubgraph(IImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, StringMap<TensorOrWeights>& subgraphTensors)
{
    auto net = ctx->network();
    int32_t beforeSubgraph = net->getNbLayers();

    // Establish scope for names local to the subgraph.
    NameScope nameScope(*ctx);

    CHECK(onnx2trt::parseGraph(ctx, subgraph));

    for (int32_t i = 0; i < subgraph.output_size(); ++i)
    {
        std::string name = subgraph.output(i).name();
        subgraphTensors.emplace(std::make_pair(name, ctx->tensors().at(name)));
    }

    for (int32_t i = beforeSubgraph; i < net->getNbLayers(); i++)
    {
        newLayers.push_back(net->getLayer(i));
    }

    return Status::success();
}

// Add an IConditionalInputLayer to `layer`'s inputs, if they don't already exist.
Status addConditionalInputIfNeeded(IImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
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
    return Status::success();
}

// Add IConditionalInputLayers to `layer`'s inputs.
Status addIfInputLayers(IImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers)
{
    // Find all of the tensors entering the subgraph.
    // The node-names are from the ONNX context.
    using NodeName = std::string;
    using InputIndex = int32_t;
    std::unordered_map<NodeName, std::set<InputIndex>> subgraphInputsMap;
    getSubgraphInputs(newLayers, subgraphInputsMap);

    // Add a ConditionalInputLayer in front of each input that is external to the subgraph.
    for (const auto& layer : newLayers)
    {
        addConditionalInputIfNeeded(ctx, conditional, inputsMap, *layer, subgraphInputsMap);
    }

    return Status::success();
}

// Add an IConditionalOutputLayer to `layer`'s outputs.
Status addIfOutputLayers(IImporterContext* ctx, nvinfer1::IIfConditional* conditional,
    ::ONNX_NAMESPACE::GraphProto const& thenGraph, std::vector<nvinfer1::ILayer*> const& thenLayers,
    StringMap<TensorOrWeights> const& thenSubgraphTensors, ::ONNX_NAMESPACE::GraphProto const& elseGraph,
    std::vector<nvinfer1::ILayer*> const& elseLayers, StringMap<TensorOrWeights> const& elseSubgraphTensors,
    std::vector<TensorOrWeights>& graphOutputs)
{
    // Reported outputs are outputs that the ONNX model reports as subgraph outputs.  This list is
    // not sufficient because it may produce names that are not fully compatible with TensorRT's naming.
    // We use this list to help find the subgraph (SG) output tensors.
    auto getReportedOutputs
        = [&ctx](const ::ONNX_NAMESPACE::GraphProto& body, std::vector<std::string>& reportedOutputs) {
              // Assuming that the subgraph was imported already, we can iterate on its output tensors.
              const auto nbOutputs = body.output_size();
              for (auto i = 0; i < nbOutputs; i++)
              {
                  reportedOutputs.emplace_back(body.output(i).name());
              }
          };

    using NodeName = std::string;
    std::unordered_map<NodeName, std::set<int32_t>> thenOutputs;
    std::unordered_map<NodeName, std::set<int32_t>> elseOutputs;

    std::vector<std::string> thenReportedOutputs;
    getReportedOutputs(thenGraph, thenReportedOutputs);
    getSubgraphOutputs(thenLayers, thenOutputs, thenReportedOutputs);
    std::vector<std::string> elseReportedOutputs;
    getReportedOutputs(elseGraph, elseReportedOutputs);
    getSubgraphOutputs(elseLayers, elseOutputs, elseReportedOutputs);

    // Retrieve the output tensors of a subgraph (tensors exiting the subgraph).
    auto getSubgraphOutputTensors
        = [](IImporterContext* ctx, std::vector<nvinfer1::ITensor*>& sgOutputs, SubgraphPortsMap& subgraphOutputs,
              ::ONNX_NAMESPACE::GraphProto const& subgraph, std::vector<nvinfer1::ILayer*> subgraphLayers,
              StringMap<TensorOrWeights> const& subgraphTensors) {
              for (const auto& layer : subgraphLayers)
              {
                  const auto layerName = layer->getName();
                  auto iter = findLayer(subgraphOutputs, layerName);
                  if (iter != subgraphOutputs.end())
                  {
                      sgOutputs.push_back(layer->getOutput(0));
                  }
              }

              if (sgOutputs.empty())
              {
                  // No new layers, so we can't deduce the outputs and have to use what ONNX tells us.
                  const int32_t nbOutputs = subgraph.output_size();
                  for (int32_t outIdx = 0; outIdx < nbOutputs; outIdx++)
                  {
                      const auto thenName = subgraph.output(outIdx).name();
                      TensorOrWeights tw = subgraphTensors.at(thenName);
                      auto* thenTensor = &convertToTensor(tw, ctx);
                      sgOutputs.push_back(thenTensor);
                  }
              }
          };

    std::vector<nvinfer1::ITensor*> thenOutputTensors;
    getSubgraphOutputTensors(ctx, thenOutputTensors, thenOutputs, thenGraph, thenLayers, thenSubgraphTensors);

    std::vector<nvinfer1::ITensor*> elseSGOutputTensors;
    getSubgraphOutputTensors(ctx, elseSGOutputTensors, elseOutputs, elseGraph, elseLayers, elseSubgraphTensors);

    ASSERT(thenOutputTensors.size() == elseSGOutputTensors.size()
            && "The then/else branches of an If operator must have the same number of outputs.",
        ErrorCode::kINVALID_NODE);

    // Add an ConditionalOutputLayer with one output and two inputs
    // (one from the thenGraph and another from the elseGraph).
    for (size_t i = 0; i < elseSGOutputTensors.size(); i++)
    {
        auto* outputLayer = conditional->addOutput(*thenOutputTensors[i], *elseSGOutputTensors[i]);
        ctx->registerLayer(outputLayer, std::string(conditional->getName()) + "_OutputLayer");
        graphOutputs.emplace_back(outputLayer->getOutput(0));
    }
    return Status::success();
}

// Given a subgraph, find all of its external inputs/outputs (tensors entering/exiting the subgraph).
Status getSubgraphTensors(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<std::string, std::set<int32_t>>& externalOutputs, bool extractOutputs,
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
    std::unordered_map<TensorName, std::vector<Port>> externalPortsMap;
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
                        externalPortsMap[tensorName].push_back(std::make_pair(nodeName, i));
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
            auto const nodeName = inPort.first;
            auto const portIndex = inPort.second;
            externalOutputs[nodeName].insert(portIndex);
        }
    }
    return Status::success();
}

Status getSubgraphOutputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<std::string, std::set<int32_t>>& externalOutputs,
    const std::vector<std::string>& reportedOutputs)
{
    return getSubgraphTensors(newLayers, externalOutputs, true, &reportedOutputs);
}

Status getSubgraphInputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<std::string, std::set<int32_t>>& externalInputs)
{
    return getSubgraphTensors(newLayers, externalInputs, false);
}

} // namespace onnx2trt
