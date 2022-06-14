/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Helper functions used for importing the ONNX If-operator follow below.
 *
 */

#pragma once

#include "ImporterContext.hpp"
#include "Status.hpp"
#include <NvInfer.h>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx2trt
{

// Given a subgraph, find all of its external inputs (tensors entering the subgraph).
// The result is returned in `subgraphInputs`, which is a map indexed by layer-name and with values indicating a set
// of external input indices.
Status getSubgraphInputs(
    const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<std::string, std::set<int32_t>>& subgraphInputs);

// Given a subgraph, find all of its external outputs (tensors exiting the subgraph).
// The result is returned in `subgraphInputs`, which is a map indexed by layer-name and with values indicating a set
// of external outputs indices.
Status getSubgraphOutputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<std::string, std::set<int32_t>>& subgraphOutputs,
    const std::vector<std::string>& reportedOutputs);

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
Status importSubgraph(IImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, StringMap<TensorOrWeights>& subgraphTensors);

using InputsMap = std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer*>;

// Add IIfConditionalInputLayers to the inputs of the subgraph indicated by `subgraph`.
onnx2trt::Status addIfInputLayers(IImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers);

// Add IIfConditionalOutputLayers to the outputs of the subgraph indicated by `subgraph`.
onnx2trt::Status addIfOutputLayers(IImporterContext* ctx, nvinfer1::IIfConditional* conditional,
    ::ONNX_NAMESPACE::GraphProto const& thenGraph, std::vector<nvinfer1::ILayer*> const& thenLayers,
    StringMap<TensorOrWeights> const& thenSubgraphTensors, ::ONNX_NAMESPACE::GraphProto const& elseGraph,
    std::vector<nvinfer1::ILayer*> const& elseLayers, StringMap<TensorOrWeights> const& elseSubgraphTensors,
    std::vector<TensorOrWeights>& graphOutputs);

} // namespace onnx2trt
