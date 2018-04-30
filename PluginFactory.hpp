/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "NvOnnxParserRuntime.h"
#include "onnx2trt_runtime.hpp"
#include "utils.hpp"
#include "builtin_plugins.hpp"

#include <NvInfer.h>

#include <list>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cassert>

namespace onnx2trt {

class PluginFactory final : public nvonnxparser::IPluginFactory {
  nvinfer1::ILogger* _logger;
  string_map<plugin_deserializer> _plugin_registry;
  std::list<UniqueOwnable> _owned_plugin_instances;
public:
  PluginFactory(nvinfer1::ILogger* logger)
    : _logger(logger),
      _plugin_registry(getBuiltinPluginMap()) {}
  void registerPlugin(const char* plugin_type, plugin_deserializer func) {
    // Note: This allows existing importers to be replaced
    _plugin_registry[plugin_type] = func;
  }
  // This is used by TRT during engine deserialization
  virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
                                          const void* serialData,
                                          size_t serialLength) override;
  virtual void destroy() override { delete this; }
};

} // namespace onnx2trt
