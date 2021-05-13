/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>

#include "TensorOrWeights.hpp"
#include "ImporterContext.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addRNNInput(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, std::vector<TensorOrWeights>& inputs, const std::string& direction);

// Zeros out invalid timesteps in toMask. maxLen must be provided if reverse is true
nvinfer1::ITensor* clearMissingSequenceElements(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens, nvinfer1::ITensor* toMask, nvinfer1::ITensor* maxLen, bool reverse = false, nvinfer1::ITensor* counter = nullptr);

// Returns a bool tensor which is true during valid timesteps
nvinfer1::ITensor* getRaggedMask(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens, nvinfer1::ITensor* maxLen = nullptr, bool reverse = false, nvinfer1::ITensor* counter = nullptr);

// Selects between prevH and Ht to forward previous hidden state through invalid timesteps
nvinfer1::ITensor* maskRNNHidden(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens, nvinfer1::ITensor* prevH, nvinfer1::ITensor* Ht, nvinfer1::ITensor* maxLen = nullptr, bool reverse = false, nvinfer1::ITensor* counter = nullptr);

// Splits a bidirectional hidden state into forward and reverse passes, masks each using maskRNNHidden, then concatenates
nvinfer1::ITensor* maskBidirRNNHidden(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens, nvinfer1::ITensor* maxLen, nvinfer1::ITensor* Ht1, nvinfer1::ITensor* Ht, nvinfer1::ITensor* singlePassShape);

} // namespace onnx2trt
