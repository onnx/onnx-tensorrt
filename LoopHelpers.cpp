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

#include "LoopHelpers.hpp"
#include "onnx2trt_utils.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addLoopCounter(IImporterContext* ctx, nvinfer1::ILoop* loop, int32_t initial)
{
    nvinfer1::ITensor* initialTensor = addConstantScalar(ctx, initial, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* one = addConstantScalar(ctx, 1, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 1})->getOutput(0);

    auto counter = loop->addRecurrence(*initialTensor);
    nvinfer1::ITensor* addOne = ctx->network()->addElementWise(*counter->getOutput(0), *one, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    counter->setInput(1, *addOne);
    return counter->getOutput(0);
}

} // namespace onnx2trt
