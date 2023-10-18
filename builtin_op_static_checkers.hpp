/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx2trt.hpp"

namespace onnx2trt
{

StringMap<OpStaticErrorChecker>& getOpStaticErrorCheckerMap();

} // namespace onnx2trt
