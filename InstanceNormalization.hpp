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

#include "plugin.hpp"
#include "serialize.hpp"
#include <cudnn.h>
#include <vector>

typedef unsigned short half_type;

class InstanceNormalizationPlugin final : public onnx2trt::Plugin {
  float _epsilon;
  int   _nchan;
  std::vector<float> _h_scale;
  std::vector<float> _h_bias;
  float* _d_scale;
  float* _d_bias;
  bool _initialized;
  cudnnHandle_t _cudnn_handle;
  cudnnTensorDescriptor_t _x_desc, _y_desc, _b_desc;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_epsilon);
    deserialize_value(&serialData, &serialLength, &_nchan);
    deserialize_value(&serialData, &serialLength, &_h_scale);
    deserialize_value(&serialData, &serialLength, &_h_bias);
  }
  size_t getSerializationSize() override {
    return (serialized_size(_epsilon) +
            serialized_size(_nchan) +
            serialized_size(_h_scale) +
            serialized_size(_h_bias)) + getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _epsilon);
    serialize_value(&buffer, _nchan);
    serialize_value(&buffer, _h_scale);
    serialize_value(&buffer, _h_bias);
  }
public:
  InstanceNormalizationPlugin(float epsilon,
                              nvinfer1::Weights const& scale,
                              nvinfer1::Weights const& bias);
  InstanceNormalizationPlugin(void const* serialData, size_t serialLength) : _initialized(false) {
    this->deserialize(serialData, serialLength);
  }
  const char* getPluginType() const override { return "InstanceNormalization"; }
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *inputDims,
                                     int nbInputs) override {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 1);
    return *inputDims;
  }
  int initialize() override;
  void terminate() override;
  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  size_t getWorkspaceSize(int maxBatchSize) const override;
  ~InstanceNormalizationPlugin();
};
