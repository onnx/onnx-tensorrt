/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"

namespace onnx2trt
{

StringMap<OpStaticErrorChecker>& getOpStaticErrorCheckerMap();

} // namespace onnx2trt
