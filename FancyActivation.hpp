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

#include <cassert>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

class FancyActivationPlugin final : public onnx2trt::Plugin {
public:
  enum ActivationType : int {
    LEAKY_RELU,
    ELU,
    SELU,
    SOFTPLUS,
    SOFTSIGN,
    HARD_SIGMOID,
    HARD_TANH,
    CLIP,
    FLOOR,
    CEIL,
    THRESHOLDED_RELU,
  };
private:
  ActivationType _activation_type;
  float _alpha;
  float _gamma;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_activation_type);
    deserialize_value(&serialData, &serialLength, &_alpha);
    deserialize_value(&serialData, &serialLength, &_gamma);
  }
  size_t getSerializationSize() override {
    return (serialized_size(_activation_type) +
            serialized_size(_alpha) +
            serialized_size(_gamma)) + getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, (int)_activation_type);
    serialize_value(&buffer, _alpha);
    serialize_value(&buffer, _gamma);
  }
public:
  FancyActivationPlugin(ActivationType activation_type, float alpha=0., float gamma=0.)
    : _activation_type(activation_type), _alpha(alpha), _gamma(gamma) {}
  FancyActivationPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "FancyActivation"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputs) override {
    assert(index == 0);
    assert(inputDims);
    assert(nbInputs == 1);
    return *inputDims;
  }
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  template <typename Data>
    int doEnqueue(int batchSize,
                  const void *const *inputs, void **outputs,
                  void *workspace, cudaStream_t stream);
};
