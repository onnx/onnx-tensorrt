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

#include "plugin_common.hpp"
#include "serialize.hpp"
#include "onnx2trt_common.hpp"

#include <NvInferPlugin.h>

#include <memory>
#include <vector>

namespace onnx2trt {

// A convenient base class for plugins. Provides default implementations of
// some methods.
// Adapts a plugin so that its type is automatically serialized, enabling it
// to be identified when deserializing.
class Plugin : public nvinfer1::IPluginExt, public IOwnable {

public:
  virtual const char* getPluginType() const = 0;

#if NV_TENSORRT_MAJOR > 4
  virtual const char* getPluginVersion() const = 0;
#endif

  nvinfer1::Dims const&  getInputDims(int index) const { return _input_dims.at(index); }
  size_t                 getMaxBatchSize()       const { return _max_batch_size; }
  nvinfer1::DataType     getDataType()           const { return _data_type; }
  nvinfer1::PluginFormat getDataFormat()         const { return _data_format; }

  size_t getWorkspaceSize(int) const override { return 0; }

  int         initialize()           override { return 0;}
  void        terminate()            override {}

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;

  void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) override;
  void destroy() override { delete this; }
protected:
  void   deserializeBase(void const*& serialData, size_t& serialLength);
  size_t getBaseSerializationSize();
  void   serializeBase(void*& buffer);

  std::vector<nvinfer1::Dims> _input_dims;
  size_t                      _max_batch_size;
  nvinfer1::DataType          _data_type;
  nvinfer1::PluginFormat      _data_format;
  virtual ~Plugin() {}
};

class PluginV2 : public nvinfer1::IPluginV2, public IOwnable {

public:
  virtual const char* getPluginType() const = 0;

  virtual const char* getPluginVersion() const = 0;

  nvinfer1::Dims const&  getInputDims(int index) const { return _input_dims.at(index); }
  size_t                 getMaxBatchSize()       const { return _max_batch_size; }
  nvinfer1::DataType     getDataType()           const { return _data_type; }
  nvinfer1::PluginFormat getDataFormat()         const { return _data_format; }

  size_t getWorkspaceSize(int) const override { return 0; }

  int         initialize()           override { return 0;}
  void        terminate()            override {}

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;

  void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) override;
  void destroy() override { delete this; }
protected:
  void   deserializeBase(void const*& serialData, size_t& serialLength);
  size_t getBaseSerializationSize() const;
  void   serializeBase(void*& buffer) const;

  std::vector<nvinfer1::Dims> _input_dims;
  size_t                      _max_batch_size;
  nvinfer1::DataType          _data_type;
  nvinfer1::PluginFormat      _data_format;
  virtual ~PluginV2() {}
};

class PluginAdapter : public PluginV2 {
protected:
  nvinfer1::IPluginV2*     _pluginV2;
public:
  PluginAdapter(nvinfer1::IPluginV2* pluginV2) :
    _pluginV2(pluginV2) {}
  virtual int getNbOutputs() const override;
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputs) override ;
  virtual void serialize(void* buffer) const override;
  virtual size_t getSerializationSize() const override;

  virtual int  initialize() override;
  virtual void terminate() override;

  virtual void destroy() override;
  virtual nvinfer1::IPluginV2* clone() const override;
  virtual const char* getPluginVersion() const override;
  virtual const char* getPluginNamespace() const override;
  virtual void setPluginNamespace(const char* pluginNamespace) override;

  virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
  virtual void configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs,
                                   const nvinfer1::Dims *outputDims, int nbOutputs,
                                   nvinfer1::DataType type,
                                   nvinfer1::PluginFormat format,
                                   int maxBatchSize);
  virtual size_t getWorkspaceSize(int maxBatchSize) const override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;
};

// This makes a plugin compatible with onnx2trt::PluginFactory by serializing
// its plugin type.
class TypeSerializingPlugin : public PluginAdapter {
  UniqueOwnable _owned_plugin;
  PluginV2* _plugin;
public:
  TypeSerializingPlugin(PluginV2* plugin)
    : PluginAdapter(plugin), _owned_plugin(plugin), _plugin(plugin) {}
  void serialize(void* buffer) const override {
    const char* plugin_type = _plugin->getPluginType();
    serialize_value(&buffer, (const char*)REGISTERABLE_PLUGIN_MAGIC_STRING);
    serialize_value(&buffer, plugin_type);
    return _plugin->serialize(buffer);
  }
  size_t getSerializationSize() const override {
    const char* plugin_type = _plugin->getPluginType();
    // Note: +1 for NULL-terminated string
    return (sizeof(REGISTERABLE_PLUGIN_MAGIC_STRING) + 1 +
            strlen(plugin_type) +
            _plugin->getSerializationSize());
  }
  const char* getPluginType() const override {
    return _plugin->getPluginType();
  }
#if NV_TENSORRT_MAJOR > 4
  const char* getPluginVersion() const override { return _plugin->getPluginType(); }
#endif

  void destroy() override { delete this; }
};

} // namespace onnx2trt
