/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx2trt.hpp"

namespace onnx2trt
{

StringMap<NodeImporter>& getBuiltinOpImporterMap();

} // namespace onnx2trt
