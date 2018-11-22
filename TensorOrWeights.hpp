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

#include "ShapedWeights.hpp"

#include <NvInfer.h>
#include <cassert>

namespace onnx2trt {

class TensorOrWeights {
  union {
    nvinfer1::ITensor* _tensor;
    ShapedWeights      _weights;
  };
  enum { NODE_TENSOR, NODE_WEIGHTS } _variant;
public:
  inline TensorOrWeights() : _tensor(nullptr), _variant(NODE_TENSOR) {}
  inline TensorOrWeights(nvinfer1::ITensor* tensor)
    : _tensor(tensor), _variant(NODE_TENSOR) {}
  inline TensorOrWeights(ShapedWeights const& weights)
    : _weights(weights), _variant(NODE_WEIGHTS) {}
  inline bool is_tensor()  const { return _variant == NODE_TENSOR; }
  inline bool is_weights() const { return _variant == NODE_WEIGHTS; }
  inline nvinfer1::ITensor& tensor() {
    assert(this->is_tensor());
    return *_tensor;
  }
  inline nvinfer1::ITensor const& tensor() const {
    assert(this->is_tensor());
    return *_tensor;
  }
  inline ShapedWeights& weights() {
    assert(this->is_weights());
    return _weights;
  }
  inline ShapedWeights const& weights() const {
    assert(this->is_weights());
    return _weights;
  }
  inline nvinfer1::Dims shape() const {
    return this->is_tensor() ? _tensor->getDimensions() : _weights.shape;
  }
  inline operator bool() const {
    return this->is_tensor() ? (bool)_tensor : (bool)_weights;
  }
  nvinfer1::ITensor* reset_tensor(nvinfer1::ITensor* tensor) {
    assert(this->is_tensor());
    return _tensor = tensor;
  }
};

} // namespace onnx2trt
