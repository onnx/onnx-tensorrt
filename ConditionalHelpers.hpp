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

using NodeName = std::string;
using LayerName = std::string;
using InputIndex = int32_t;

// A SubgraphPortsMap maps inputs' ports of each layer in an ONNX graph.
using SubgraphPortsMap = std::unordered_map<const nvinfer1::ILayer*, std::unordered_set<InputIndex>>;

// Given a subgraph, find all of its external inputs (tensors entering the subgraph).
void getSubgraphInputs(const std::vector<nvinfer1::ILayer*>& newLayers, SubgraphPortsMap& externalInputs);

// Take a snapshot of the network before and after parsing the subgraph and return a list
// of newly added network layers.
// subgraphParentIdx is the index of the parent node in the main graph, used for error reporting.
void importSubgraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& subgraph,
    std::vector<nvinfer1::ILayer*>& newLayers, std::vector<TensorOrWeights>& subgraphTensors,
    int32_t subgraphParentIdx = -1);

// An InputsMap tracks which IIfConditionalInputLayer we've added to a layer's inputs,
// so that we can reuse them if needed.
using InputsMap = std::unordered_map<LayerName, nvinfer1::IIfConditionalInputLayer*>;

// Add IIfConditionalInputLayers to the inputs of the subgraph indicated by `subgraph`.
void addIfInputLayers(ImporterContext* ctx, nvinfer1::IIfConditional* conditional, InputsMap& inputsMap,
    const std::vector<nvinfer1::ILayer*>& newLayers, ::ONNX_NAMESPACE::NodeProto const* node);

} // namespace onnx2trt
