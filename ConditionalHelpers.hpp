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
// The result is returned in `subgraphInputs`, which is a map indexed by ITensor (a tensor entering the subgraph) and
// with values indicating a set of external input indices.
void getSubgraphInputs(std::vector<nvinfer1::ILayer*> const& newLayers,
    std::unordered_map<nvinfer1::ITensor*, std::set<int32_t>>& subgraphInputs);

// Given a subgraph, find all of its external outputs (tensors exiting the subgraph).
// The result is returned in `subgraphInputs`, which is a map indexed by ITensor (a tensor exiting the subgraph) and
// with values indicating a set of external outputs indices.
void getSubgraphOutputs(const std::vector<nvinfer1::ILayer*>& newLayers,
    std::unordered_map<nvinfer1::ITensor*, std::set<int32_t>>& subgraphOutputs,
    const std::vector<std::string>& reportedOutputs);

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
void importSubgraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, std::vector<TensorOrWeights>& subgraphTensors);

using InputsMap = std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer*>;

// Add IIfConditionalInputLayers to the inputs of the subgraph indicated by `subgraph`.
void addIfInputLayers(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers);

} // namespace onnx2trt
