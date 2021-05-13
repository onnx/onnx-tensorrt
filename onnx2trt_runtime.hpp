/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx2trt_common.hpp"

namespace onnx2trt
{

typedef Plugin* (*plugin_deserializer)(const void* serialData, size_t serialLength);

} // namespace onnx2trt
