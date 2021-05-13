/*
 * SPDX-License-Identifier: Apache-2.0
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
