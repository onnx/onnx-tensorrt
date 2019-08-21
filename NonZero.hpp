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

#include <thrust/device_vector.h>
#include <cassert>

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)


namespace {
    constexpr const char* NONZERO_PLUGIN_VERSION{"001"};
    constexpr const char* NONZERO_PLUGIN_NAME{"NonZero"};
}
enum class NonZeroType : int {
    NonZero,
    MAX_VALUE //no mean
};
class NonZeroPlugin final : public onnx2trt::PluginV2 {
private:
  int _value;
  unsigned long long _numbers;
  int* _lensOfDim;
  int* _mulOfSon;
  unsigned int _rows;
  //int _nx, _ny, _nz;
  //int _x_stride, _y_stride, _z_stride;
  //thrust::device_vector<int> _d_segment_offsets;
  //thrust::device_vector<float*> _d_output_ptrs;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    //deserialize_value(&serialData, &serialLength, &_value);
  }
  virtual size_t getSerializationSize() const override {
    return  getBaseSerializationSize();
  }
  virtual void serialize(void *buffer) const override {
    serializeBase(buffer);
    //serialize_value(&buffer, _value);
  }
public:
  NonZeroPlugin(){
    //assert(value <= NonZeroType::MAX_VALUE);
  }
  NonZeroPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return NONZERO_PLUGIN_NAME; }

  virtual void destroy() override {if(_lensOfDim!=nullptr)cudaFree(_lensOfDim);  if(_mulOfSon!=nullptr)cudaFree(_mulOfSon);delete this; }

  virtual nvinfer1::IPluginV2* clone() const override { return new NonZeroPlugin{}; }

  virtual const char* getPluginVersion() const override { return NONZERO_PLUGIN_VERSION; }

  virtual void setPluginNamespace(const char* pluginNamespace) override {}

  virtual const char* getPluginNamespace() const override { return ""; }
  //sds:The number of the output tensor. 分割成几分，就是介个tensor.
  virtual int getNbOutputs() const override { return 1; };
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;
};

//sds:最新的添加plugin的方式。
class NonZeroPluginCreator : public nvinfer1::IPluginCreator
{
public:
  NonZeroPluginCreator() {}

  ~NonZeroPluginCreator() {}

  const char* getPluginName() const { return NONZERO_PLUGIN_NAME; }

  const char* getPluginVersion() const { return NONZERO_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() { std::cerr<< "Function not implemented" << std::endl; return nullptr; }
  //不支持
  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) { std::cerr<< "Function not implemented" << std::endl; return nullptr; }
  //sds:支持反序列化(新版本的添加plugin方式。如果是老版本的，需要在pluginFactory中注册deserialize)。
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) { return new NonZeroPlugin{serialData, serialLength}; }

  void setPluginNamespace(const char* libNamespace) { mNamespace = libNamespace; }

  const char* getPluginNamespace() const { return mNamespace.c_str(); }
private:
    std::string mNamespace;
};

REGISTER_TENSORRT_PLUGIN(NonZeroPluginCreator);
