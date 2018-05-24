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

#include "ShapedWeights.hpp"
#include "trt_utils.hpp"
#include "onnx2trt_utils.hpp"

namespace onnx2trt {

size_t ShapedWeights::count() const {
  if( this->shape.nbDims == 0 ) {
    return 0;
  } else {
    size_t c = 1;
    for( int i=0; i<this->shape.nbDims; ++i ) {
      c *= this->shape.d[i];
    }
    return c;
  }
}

ShapedWeights ShapedWeights::empty(DataType type) {
  return ShapedWeights(type, nullptr, nvinfer1::Dims{0});
}

ShapedWeights::ShapedWeights() : values(nullptr), shape{0} {}

ShapedWeights::ShapedWeights(DataType type_, void* values_, nvinfer1::Dims shape_)
  : type(type_), values(values_), shape(shape_) {
  // Note: this->shape.type[] is not used
}

size_t ShapedWeights::size_bytes() const {
  return this->count() * get_dtype_size(this->type);
}

ShapedWeights::operator bool() const {
  return (bool)this->values;
}

ShapedWeights::operator nvinfer1::Weights() const {
  nvinfer1::Weights w;
  bool supported_type = convert_dtype(this->type, &w.type);
  (void)supported_type;
  assert(supported_type);
  w.values = this->values;
  w.count = this->count();
  return w;
}

bool transposeWeights(ShapedWeights const& weights,
                      nvinfer1::Permutation const& perm,
                      ShapedWeights* result) {
  nvinfer1::Dims shape = weights.shape;
  nvinfer1::Dims new_shape;
  new_shape.nbDims = shape.nbDims;
  for( int d=0; d<shape.nbDims; ++d ) {
    new_shape.d[d] = shape.d[perm.order[d]];
    result->shape.d[d] = new_shape.d[d];
  }
  // TODO: Need to generalize this transpose implementation
  assert(perm.order[0] == 1 && perm.order[1] == 0);
  if( shape.nbDims == 2 &&
      weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT ) {
    for( int i=0; i<new_shape.d[0]; ++i ) {
      for( int j=0; j<new_shape.d[1]; ++j ) {
        float const* src = (float*)weights.values;
        float*       dst = (float*)result->values;
        int src_stride = weights.shape.d[1];
        int dst_stride = result->shape.d[1];
        dst[i * dst_stride + j] = src[j * src_stride + i];
      }
    }
  } else {
    // TODO: Implement general transposes and multiple data types
    // Unsupported weights transpose
    return false;
  }
  return true;
}

} // namespace onnx2trt
