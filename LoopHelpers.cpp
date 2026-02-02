/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "LoopHelpers.hpp"
#include "importerUtils.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addLoopCounter(ImporterContext* ctx, nvinfer1::ILoop* loop, int64_t initial)
{
    nvinfer1::ITensor* initialTensor
        = addConstantScalar(ctx, initial, ::ONNX_NAMESPACE::TensorProto::INT64, 1)->getOutput(0);
    nvinfer1::ITensor* one
        = addConstantScalar(ctx, static_cast<int64_t>(1), ::ONNX_NAMESPACE::TensorProto::INT64, 1)->getOutput(0);

    auto counter = N_CHECK(loop->addRecurrence(*initialTensor));
    nvinfer1::ITensor* addOne
        = getElementWiseResult(ctx, *N_CHECK(counter->getOutput(0)), *one, nvinfer1::ElementWiseOperation::kSUM);
    counter->setInput(1, *addOne);
    return N_CHECK(counter->getOutput(0));
}

} // namespace onnx2trt
