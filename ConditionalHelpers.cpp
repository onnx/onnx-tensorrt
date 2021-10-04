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
        ctx->registerTensor(TensorOrWeights{inputLayer->getOutput(0)}, inputLayerName + "_InputLayer_output");
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
Status importSubgraph(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::GraphProto& subgraph, std::vector<nvinfer1::ILayer*>& newLayers)
{
    auto net = ctx->network();
    int32_t beforeSubgraph = net->getNbLayers();
    CHECK(onnx2trt::parseGraph(ctx, subgraph));

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
    const ::ONNX_NAMESPACE::GraphProto& subgraph, const std::vector<nvinfer1::ILayer*>& newLayers)
{
    // Find all of the tensors entering the subgraph.
    // The node-names are from the ONNX context.
    using NodeName = std::string;
    using InputIndex = int32_t;
    std::unordered_map<NodeName, std::set<InputIndex>> subgraphInputsMap;
    getSubgraphInputs(subgraph, subgraphInputsMap);

    // Add a ConditionalInputLayer in front of each input that is external to the subgraph.
    for (const auto& layer : newLayers)
    {
        addConditionalInputIfNeeded(ctx, conditional, inputsMap, *layer, subgraphInputsMap);
    }

    return Status::success();
}

// Add an IConditionalOutputLayer to `layer`'s outputs.
Status addIfOutputLayers(IImporterContext* ctx, nvinfer1::IIfConditional* conditional,
    const ::ONNX_NAMESPACE::GraphProto& thenGraph, const std::vector<nvinfer1::ILayer*>& thenLayers,
    const ::ONNX_NAMESPACE::GraphProto& elseGraph, const std::vector<nvinfer1::ILayer*>& elseLayers,
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
    getSubgraphOutputs(thenGraph, thenOutputs, thenReportedOutputs);
    std::vector<std::string> elseReportedOutputs;
    getReportedOutputs(thenGraph, elseReportedOutputs);
    getSubgraphOutputs(elseGraph, elseOutputs, elseReportedOutputs);

    // Retrieve the output tensors of a subgraph (tensors exiting the subgraph).
    auto getSubgraphOutputTensors
        = [](IImporterContext* ctx, std::vector<nvinfer1::ITensor*>& sgOutputs, SubgraphPortsMap& subgraphOutputs,
              const ::ONNX_NAMESPACE::GraphProto& subgraph, std::vector<nvinfer1::ILayer*> subgraphLayers) {
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
                      auto* thenTensor = &convertToTensor(ctx->tensors().at(thenName), ctx);
                      sgOutputs.push_back(thenTensor);
                  }
              }
          };

    std::vector<nvinfer1::ITensor*> thenOutputTensors;
    getSubgraphOutputTensors(ctx, thenOutputTensors, thenOutputs, thenGraph, thenLayers);

    std::vector<nvinfer1::ITensor*> elseSGOutputTensors;
    getSubgraphOutputTensors(ctx, elseSGOutputTensors, elseOutputs, elseGraph, elseLayers);

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
Status getSubgraphTensors(const ::ONNX_NAMESPACE::GraphProto& graph,
    std::unordered_map<std::string, std::set<int32_t>>& externalOutputs, bool extractOutputs,
    const std::vector<std::string>* reportedOutputs = nullptr)
{
    std::vector<size_t> topoOrder;
    ASSERT(toposort(graph.node(), &topoOrder) && "Failed to sort the model topologically.", ErrorCode::kINVALID_GRAPH);
    using NodeName = std::string;
    using TensorName = std::string;
    using PortIndex = int32_t;
    using Port = std::pair<NodeName, PortIndex>;
    std::unordered_set<TensorName> outputTensors;
    std::unordered_set<TensorName> inputTensors;

    // To determine which tensors are entering or exiting the given graph, we first collect the sets of all input and
    // output tensors. Then we categorize the tensors according to this logic:
    //  Entering tensors := {inputs} - {outputs}
    //  Exiting tensors := {outputs} - {inputs}

    // Collect all input and output tensors belonging to nodes in the graph.
    for (const auto& nodeIndex : topoOrder)
    {
        const auto& node = graph.node(nodeIndex);
        for (const auto& outputName : node.output())
        {
            outputTensors.insert(outputName);
        }
        for (const auto& inputName : node.input())
        {
            inputTensors.insert(inputName);
        }
    }

    using NodeProto = const ::ONNX_NAMESPACE::NodeProto;
    auto getOutputs = [](NodeProto& node) { return node.output(); };
    auto getInputs = [](NodeProto& node) { return node.input(); };

    // Retrieve the list of tensors either exiting or entering the subgraph.
    std::unordered_map<TensorName, std::vector<Port>> externalPortsMap;
    auto filterTensors = [&](std::unordered_set<TensorName> tensors, auto nodeAccessor) {
        for (const auto& nodeIndex : topoOrder)
        {
            const auto& node = graph.node(nodeIndex);
            const auto& nodeName = getNodeName(node);
            PortIndex i = 0;

            for (const auto& tensorName : nodeAccessor(node))
            {
                if (tensorName.empty())
                {
                    continue;
                }
                if (tensors.count(tensorName) == 0)
                {
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

Status getSubgraphOutputs(const ::ONNX_NAMESPACE::GraphProto& graph,
    std::unordered_map<std::string, std::set<int32_t>>& externalOutputs,
    const std::vector<std::string>& reportedOutputs)
{
    return getSubgraphTensors(graph, externalOutputs, true, &reportedOutputs);
}

Status getSubgraphInputs(
    const ::ONNX_NAMESPACE::GraphProto& graph, std::unordered_map<std::string, std::set<int32_t>>& externalInputs)
{
    return getSubgraphTensors(graph, externalInputs, false);
}

} // namespace onnx2trt
