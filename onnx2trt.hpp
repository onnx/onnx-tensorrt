/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "NvOnnxParser.h"
#include "ShapedWeights.hpp"
#include "TensorOrWeights.hpp"
#include "Status.hpp"

#include <onnx/onnx_pb.h>
#include <NvInfer.h>
#include <vector>
#include <functional>

namespace onnx2trt {

class IImporterContext;

// TODO: Find ABI-safe alternative approach for this:
//         Can't use std::vector
//         Can't use ::onnx::NodeProto
//         Can't use std::function
typedef ValueOrStatus<std::vector<TensorOrWeights>> NodeImportResult;
typedef std::function<NodeImportResult(IImporterContext *ctx,
                                       ::ONNX_NAMESPACE::NodeProto const &node,
                                       std::vector<TensorOrWeights> &inputs)>
    NodeImporter;

struct IImporterContext {
  virtual nvinfer1::INetworkDefinition* network() = 0;
  virtual ShapedWeights createTempWeights(ShapedWeights::DataType type,
                                          nvinfer1::Dims shape) = 0;
  virtual int64_t getOpsetVersion(const char* domain="") const = 0;
protected:
  virtual ~IImporterContext() {}
};

} // namespace onnx2trt
