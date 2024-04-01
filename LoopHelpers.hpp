/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>

#include "ImporterContext.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addLoopCounter(ImporterContext* ctx, nvinfer1::ILoop* loop, int64_t initial = 0);

} // namespace onnx2trt
