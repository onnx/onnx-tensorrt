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
#include <NvInfer.h>

#include "plugin.hpp"
#include "serialize.hpp"

#include <cassert>

namespace {
    constexpr const char* RESIZE_PLUGIN_VERSION{"001"};
    constexpr const char* RESIZE_PLUGIN_NAME{"ResizeNearest"};
}

class ResizeNearestPlugin final : public onnx2trt::PluginV2 {
  int   _ndims;
  float _scale[nvinfer1::Dims::MAX_DIMS];
  nvinfer1::Dims _output_dims;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_ndims);
    deserialize_value(&serialData, &serialLength, &_scale);
  }
  size_t getSerializationSize() const override {
    return serialized_size(_ndims) + serialized_size(_scale) + getBaseSerializationSize();
  }
  void serialize(void *buffer) const override {
    serializeBase(buffer);
    serialize_value(&buffer, _ndims);
    serialize_value(&buffer, _scale);
  }
public:
  ResizeNearestPlugin(std::vector<float> const& scale)
    : _ndims(scale.size()) {
    assert(scale.size() <= nvinfer1::Dims::MAX_DIMS);
    std::copy(scale.begin(), scale.end(), _scale);
  }
  ResizeNearestPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return RESIZE_PLUGIN_NAME; }

  virtual void destroy() override { delete this; }

  virtual nvinfer1::IPluginV2* clone() const override { return new ResizeNearestPlugin{std::vector<float>(_scale, _scale + _ndims)}; }

  virtual const char* getPluginVersion() const override { return RESIZE_PLUGIN_VERSION; }

  virtual void setPluginNamespace(const char* pluginNamespace) override {}

  virtual const char* getPluginNamespace() const override { return ""; }

  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

class ResizeNearestPluginCreator : public nvinfer1::IPluginCreator
{
public:
  ResizeNearestPluginCreator() {}

  ~ResizeNearestPluginCreator() {}

  const char* getPluginName() const { return RESIZE_PLUGIN_NAME; }

  const char* getPluginVersion() const { return RESIZE_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new ResizeNearestPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(ResizeNearestPluginCreator);
