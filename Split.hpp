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

#include <thrust/device_vector.h>
#include <cassert>

class SplitPlugin final : public onnx2trt::Plugin {
  int _axis;
  std::vector<int> _output_lengths;
  int _nx, _ny, _nz;
  int _x_stride, _y_stride, _z_stride;
  thrust::device_vector<int> _d_segment_offsets;
  thrust::device_vector<float*> _d_output_ptrs;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_axis);
    deserialize_value(&serialData, &serialLength, &_output_lengths);
  }
  virtual size_t getSerializationSize() override {
    return serialized_size(_axis) + serialized_size(_output_lengths)
      + getBaseSerializationSize();
  }
  virtual void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _axis);
    serialize_value(&buffer, _output_lengths);
  }
public:
  SplitPlugin(int axis, std::vector<int> const& output_lengths)
    : _axis(axis), _output_lengths(output_lengths) {
    assert(axis <= nvinfer1::Dims::MAX_DIMS);
  }
  SplitPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "Split"; }
  virtual int getNbOutputs() const override { return _output_lengths.size(); }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;
};
