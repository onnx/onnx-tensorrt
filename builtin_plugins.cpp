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

#include "builtin_plugins.hpp"
#include "plugin.hpp"
#include "PluginFactory.hpp"
#include "FancyActivation.hpp"
#include "ResizeNearest.hpp"
#include "Split.hpp"
#include "InstanceNormalization.hpp"

namespace onnx2trt {

string_map<plugin_deserializer>& getBuiltinPluginMap() {
  static string_map<plugin_deserializer> builtin_plugins;
  return builtin_plugins;
}

namespace {

static bool registerBuiltinPlugin(const char* plugin_type,
                                  plugin_deserializer func) {
  bool inserted = getBuiltinPluginMap().insert({plugin_type, func}).second;
  assert(inserted);
  return inserted;
}

#define IGNORE_UNUSED_GLOBAL(x) \
  static void _ignore_unused2_##x(); \
  static void _ignore_unused1_##x() { (void)_ignore_unused2_##x; (void)x; } \
  static void _ignore_unused2_##x() { (void)_ignore_unused1_##x; } \
  struct SwallowSemicolon##x {}

#define REGISTER_BUILTIN_PLUGIN(plugin_type_string, PluginClass) \
  Plugin* _build_##PluginClass(const void* serialData, \
                               size_t serialLength) {  \
    return new PluginClass(serialData, serialLength);  \
  }                                                    \
  static const bool _registered_##PluginClass = \
      registerBuiltinPlugin(plugin_type_string, _build_##PluginClass); \
  IGNORE_UNUSED_GLOBAL(_registered_##PluginClass)

#define REGISTER_BUILTIN_NVPLUGIN(plugin_type_string, PluginClass) \
  Plugin* _build_##PluginClass(const void* serialData, \
                                      size_t serialLength) { \
    return new NvPlugin( \
        nvinfer1::plugin::create##PluginClass(serialData, serialLength)); \
  } \
  static const bool _registered_##PluginClass = \
      registerBuiltinPlugin(plugin_type_string, _build_##PluginClass); \
  IGNORE_UNUSED_GLOBAL(_registered_##PluginClass)

REGISTER_BUILTIN_PLUGIN("FancyActivation",       FancyActivationPlugin);
REGISTER_BUILTIN_PLUGIN("ResizeNearest",         ResizeNearestPlugin);
REGISTER_BUILTIN_PLUGIN("Split"        ,         SplitPlugin);
REGISTER_BUILTIN_PLUGIN("InstanceNormalization", InstanceNormalizationPlugin);

REGISTER_BUILTIN_NVPLUGIN("Concat", ConcatPlugin);

} // namespace

} // namespace onnx2trt
