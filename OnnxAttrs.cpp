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

#include "OnnxAttrs.hpp"
#include "ShapedWeights.hpp"
#include "onnx2trt_utils.hpp"

template<> float OnnxAttrs::get<float>(std::string key) const {
  return this->at(key)->f();
}
template<> int OnnxAttrs::get<int>(std::string key) const {
  return this->at(key)->i();
}
template<> bool OnnxAttrs::get<bool>(std::string key) const {
  int value = this->at(key)->i();
  assert(value == bool(value));
  return bool(value);
}
template<> std::string OnnxAttrs::get<std::string>(std::string key) const {
  return this->at(key)->s();
}
template<> std::vector<int> OnnxAttrs::get<std::vector<int>>(std::string key) const {
  auto attr = this->at(key)->ints();
  return std::vector<int>(attr.begin(), attr.end());
}

template<> std::vector<int64_t> OnnxAttrs::get<std::vector<int64_t>>(std::string key) const {
  auto attr = this->at(key)->ints();
  return std::vector<int64_t>(attr.begin(), attr.end());
}

template<> std::vector<float> OnnxAttrs::get<std::vector<float>>(std::string key) const {
  auto attr = this->at(key)->floats();
  return std::vector<float>(attr.begin(), attr.end());
}
template<> nvinfer1::Dims OnnxAttrs::get<nvinfer1::Dims>(std::string key) const {
  auto values = this->get<std::vector<int>>(key);
  nvinfer1::Dims dims;
  dims.nbDims = values.size();
  std::copy(values.begin(), values.end(), dims.d);
  // Note: No dimension type information is included
  return dims;
}
template<> nvinfer1::DimsHW OnnxAttrs::get<nvinfer1::DimsHW>(std::string key) const {
  nvinfer1::Dims dims = this->get<nvinfer1::Dims>(key);
  assert(dims.nbDims == 2);
  return nvinfer1::DimsHW(dims.d[0], dims.d[1]);
}
template<> nvinfer1::Permutation OnnxAttrs::get<nvinfer1::Permutation>(std::string key) const {
  auto values = this->get<std::vector<int>>(key);
  nvinfer1::Permutation perm;
  std::copy(values.begin(), values.end(), perm.order);
  // Fill unused values with identity permutation
  for( int i=values.size(); i<nvinfer1::Dims::MAX_DIMS; ++i ) {
    perm.order[i] = i;
  }
  return perm;
}
template<> onnx2trt::ShapedWeights OnnxAttrs::get<onnx2trt::ShapedWeights>(std::string key) const {
  ::ONNX_NAMESPACE::TensorProto const& onnx_weights_tensor = this->at(key)->t();
  onnx2trt::ShapedWeights weights;
  convert_onnx_weights(onnx_weights_tensor, &weights);
  return weights;
}
