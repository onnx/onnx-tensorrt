/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>

#include "ImporterContext.hpp"

namespace onnx2trt
{

nvinfer1::ITensor* addLoopCounter(IImporterContext* ctx, nvinfer1::ILoop* loop, int32_t initial = 0);

} // namespace onnx2trt
