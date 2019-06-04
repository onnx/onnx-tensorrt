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

#include "PluginFactory.hpp"
#include "plugin.hpp"
#include "serialize.hpp"
#include "builtin_plugins.hpp"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace onnx2trt {

nvinfer1::IPlugin*
PluginFactory::createPlugin(const char* layerName,
                            const void* serialData,
                            size_t serialLength) {
  const char* magic_string;
  deserialize_value(&serialData, &serialLength, &magic_string);
  if( magic_string != std::string(REGISTERABLE_PLUGIN_MAGIC_STRING) ) {
    std::cerr << "ERROR: Not a valid serialized plugin" << std::endl;
    return nullptr;
  }
  const char* plugin_type;
  deserialize_value(&serialData, &serialLength, &plugin_type);
  if( !_plugin_registry.count(plugin_type) ) {
    std::cerr << "ERROR: No plugin registered for op: " << plugin_type << std::endl;
    return nullptr;
  }
  auto create_plugin_func = _plugin_registry.at(plugin_type);
  Plugin* plugin = create_plugin_func(serialData, serialLength);
  _owned_plugin_instances.emplace_back(plugin);
  return plugin;
}

} // namespace onnx2trt
