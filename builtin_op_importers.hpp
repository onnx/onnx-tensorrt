/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx2trt.hpp"
#include "utils.hpp"

namespace onnx2trt
{

string_map<NodeImporter>& getBuiltinOpImporterMap();

} // namespace onnx2trt
