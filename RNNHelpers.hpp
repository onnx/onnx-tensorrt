/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
