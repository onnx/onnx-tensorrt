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

namespace {
    constexpr const char* MYCAST_PLUGIN_VERSION{"001"};
    constexpr const char* MYCAST_PLUGIN_NAME{"MyCast"};
}
enum class MyCastType : int {
    MyCast,
    MAX_VALUE //no mean
};
class MyCastPlugin : public onnx2trt::PluginV2Ext {
 public:
  MyCastPlugin(nvinfer1::DataType src_type, nvinfer1::DataType dst_type)
      : src_type_(src_type), dst_type_(dst_type) {}

  MyCastPlugin(const void* serialized_data, size_t length)
      : onnx2trt::PluginV2Ext(serialized_data, length) {
    const char* buffer = static_cast<const char*>(serialized_data);
    src_type_ = ReadFromBuffer<nvinfer1::DataType>(&buffer);
    dst_type_ = ReadFromBuffer<nvinfer1::DataType>(&buffer);
    src_dims_ = ReadFromBuffer<nvinfer1::Dims>(&buffer);
  }

  MyCastPlugin(const MyCastPlugin& rhs)
      : onnx2trt::PluginV2Ext(rhs),
        src_type_(rhs.src_type_),
        dst_type_(rhs.dst_type_),
        src_dims_(rhs.src_dims_) {}

  // Methods from IPluginV2Ext.

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types,int num_inputs) const override;

  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcasted,
                                    int num_inputs) const override {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int input_index) const override {
    return false;
  }

  void configurePlugin(const nvinfer1::Dims* input_dims, int num_inputs,
                       const nvinfer1::Dims* output_dims, int num_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format, int max_batch_size);

  nvinfer1::IPluginV2Ext* clone() const override { return new MyCastPlugin(*this); }

  // Methods from IPluginV2.

  const char* getPluginType() const override { return MYCAST_PLUGIN_NAME; };

  const char* getPluginVersion() const override { return MYCAST_PLUGIN_VERSION; };

  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                           int num_input_dims) override ;

  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

  size_t getWorkspaceSize(int max_batch_size) const override { return 0; }

  int enqueue(int batch_size, const void* const* inputs, void** outputs, void*,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override {
    return 2 * sizeof(nvinfer1::DataType) + sizeof(nvinfer1::Dims);
  }

  void serialize(void* serialized_data) const override {
    char* buffer = static_cast<char*>(serialized_data);
    WriteToBuffer(src_type_, &buffer);
    WriteToBuffer(dst_type_, &buffer);
    WriteToBuffer(src_dims_, &buffer);
  }

 private:
  nvinfer1::DataType src_type_;
  nvinfer1::DataType dst_type_;
  nvinfer1::Dims src_dims_;
};

//sds:最新的添加plugin的方式。
class MyCastPluginCreator : public nvinfer1::IPluginCreator {
 public:
  MyCastPluginCreator() {}
  ~MyCastPluginCreator() {}
  const char* getPluginName() const override { return MYCAST_PLUGIN_NAME; }

  const char* getPluginVersion() const override { return MYCAST_PLUGIN_VERSION; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    std::cerr<< "Function not implemented" << std::endl; return nullptr;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* field_collection) override {
    std::cerr<< "Function not implemented" << std::endl; return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serial_data,
                               size_t serial_len) override {
    return new MyCastPlugin(serial_data, serial_len);
  }

  void setPluginNamespace(const char* plugin_namespace) override {
    namespace_ = plugin_namespace;
  }

  const char* getPluginNamespace() const override { return namespace_.c_str(); }

 private:
  std::string namespace_;
};

REGISTER_TENSORRT_PLUGIN(MyCastPluginCreator);
