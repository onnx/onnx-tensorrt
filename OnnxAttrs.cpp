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

template<> float OnnxAttrs::get<float>(const std::string& key) const {
  return this->at(key)->f();
}

template<> int OnnxAttrs::get<int>(const std::string& key) const {
  return this->at(key)->i();
}

template<> bool OnnxAttrs::get<bool>(const std::string& key) const {
  int value = this->at(key)->i();
  assert(value == bool(value));
  return bool(value);
}

template<> std::string OnnxAttrs::get<std::string>(const std::string& key) const {
  return this->at(key)->s();
}

template<> std::vector<int> OnnxAttrs::get<std::vector<int>>(const std::string& key) const {
  auto attr = this->at(key)->ints();
  return std::vector<int>(attr.begin(), attr.end());
}

template<> std::vector<int64_t> OnnxAttrs::get<std::vector<int64_t>>(const std::string& key) const {
  auto attr = this->at(key)->ints();
  return std::vector<int64_t>(attr.begin(), attr.end());
}

template<> std::vector<float> OnnxAttrs::get<std::vector<float>>(const std::string& key) const {
  auto attr = this->at(key)->floats();
  return std::vector<float>(attr.begin(), attr.end());
}

template<> nvinfer1::Dims OnnxAttrs::get<nvinfer1::Dims>(const std::string& key) const {
  auto values = this->get<std::vector<int>>(key);
  nvinfer1::Dims dims;
  dims.nbDims = values.size();
  std::copy(values.begin(), values.end(), dims.d);
  // Note: No dimension type information is included
  return dims;
}

template<> nvinfer1::DimsHW OnnxAttrs::get<nvinfer1::DimsHW>(const std::string& key) const {
  nvinfer1::Dims dims = this->get<nvinfer1::Dims>(key);
  assert(dims.nbDims == 2);
  return nvinfer1::DimsHW(dims.d[0], dims.d[1]);
}

template<> nvinfer1::Permutation OnnxAttrs::get<nvinfer1::Permutation>(const std::string& key) const {
  auto values = this->get<std::vector<int>>(key);
  nvinfer1::Permutation perm;
  std::copy(values.begin(), values.end(), perm.order);
  // Fill unused values with identity permutation
  for( int i=values.size(); i<nvinfer1::Dims::MAX_DIMS; ++i ) {
    perm.order[i] = i;
  }
  return perm;
}

template<> onnx2trt::ShapedWeights OnnxAttrs::get<onnx2trt::ShapedWeights>(const std::string& key) const {
  ::ONNX_NAMESPACE::TensorProto const& onnx_weights_tensor = this->at(key)->t();
  onnx2trt::ShapedWeights weights;
  convert_onnx_weights(onnx_weights_tensor, &weights);
  return weights;
}

template<> nvinfer1::DataType OnnxAttrs::get<nvinfer1::DataType>(const std::string& key) const {
    ::ONNX_NAMESPACE::TensorProto::DataType onnx_dtype = static_cast<::ONNX_NAMESPACE::TensorProto::DataType>(this->at(key)->i());
    nvinfer1::DataType dtype{};
    if (!onnx2trt::convert_dtype(onnx_dtype, &dtype)) {
        dtype = static_cast<nvinfer1::DataType>(-1);
    }
    return dtype;
}

template <>
std::vector<nvinfer1::DataType> OnnxAttrs::get<std::vector<nvinfer1::DataType>>(const std::string& key) const {
    auto attr = this->at(key)->ints();
    auto onnx_dtypes = std::vector<int64_t>(attr.begin(), attr.end());
    std::vector<nvinfer1::DataType> dtypes{};
    for (auto onnx_dtype : onnx_dtypes)
    {
        nvinfer1::DataType dtype{};
        if (!onnx2trt::convert_dtype(static_cast<int32_t>(onnx_dtype), &dtype))
        {
            dtype = static_cast<nvinfer1::DataType>(-1);
        }
        dtypes.push_back(dtype);
    }
    return dtypes;
}

template <>
::ONNX_NAMESPACE::GraphProto OnnxAttrs::get<::ONNX_NAMESPACE::GraphProto>(const std::string& key) const
{
    return this->at(key)->g();
}

template <>
nvinfer1::RNNOperation OnnxAttrs::get<nvinfer1::RNNOperation>(const std::string& key) const
{
    std::string op = this->get<std::string>(key);
    if (op == std::string("relu"))
    {
      return nvinfer1::RNNOperation::kRELU;
    }
    if (op == std::string("tanh"))
    {
      return nvinfer1::RNNOperation::kTANH;
    }
    if (op == std::string("lstm"))
    {
      return nvinfer1::RNNOperation::kLSTM;
    }
    if (op == std::string("gru"))
    {
      return nvinfer1::RNNOperation::kGRU;
    }
    throw std::runtime_error("Unknown RNNOperation: " + op);
}

template <>
nvinfer1::RNNInputMode OnnxAttrs::get<nvinfer1::RNNInputMode>(const std::string& key) const
{
    std::string mode = this->get<std::string>(key);
    if (mode == std::string("skip"))
    {
      return nvinfer1::RNNInputMode::kSKIP;
    }
    if (mode == std::string("linear"))
    {
      return nvinfer1::RNNInputMode::kLINEAR;
    }
    throw std::runtime_error("Unknown RNNInputMode: " + mode);
}

template <>
nvinfer1::RNNDirection OnnxAttrs::get<nvinfer1::RNNDirection>(const std::string& key) const
{
    std::string direction = this->get<std::string>(key);
    if (direction == std::string("unidirection"))
    {
      return nvinfer1::RNNDirection::kUNIDIRECTION;
    }
    if (direction == std::string("bidirection"))
    {
      return nvinfer1::RNNDirection::kBIDIRECTION;
    }
    throw std::runtime_error("Unknown RNNDirection: " + direction);
}

template <>
std::vector<std::string> OnnxAttrs::get<std::vector<std::string>>(const std::string& key) const
{
    auto attr = this->at(key)->strings();
    return std::vector<std::string>(attr.begin(), attr.end());
}

template <>
nvinfer1::ScaleMode OnnxAttrs::get<nvinfer1::ScaleMode>(const std::string& key) const
{
    std::string s = this->get<std::string>(key);
    if (s == "uniform")
    {
      return nvinfer1::ScaleMode::kUNIFORM;
    }
    if (s == "channel")
    {
      return nvinfer1::ScaleMode::kCHANNEL;
    }
    if (s == "elementwise")
    {
      return nvinfer1::ScaleMode::kELEMENTWISE;
    }
    throw std::runtime_error("Unknown ScaleMode: " + s);
}

template <>
nvinfer1::MatrixOperation OnnxAttrs::get<nvinfer1::MatrixOperation>(const std::string& key) const
{
    std::string s = this->get<std::string>(key);
    if (s == "none")
    {
      return nvinfer1::MatrixOperation::kNONE;
    }
    if (s == "transpose")
    {
      return nvinfer1::MatrixOperation::kTRANSPOSE;
    }
    if (s == "vector")
    {
      return nvinfer1::MatrixOperation::kVECTOR;
    }
    throw std::runtime_error("Unknown MatrixOperation: " + s);
}

template <>
nvinfer1::ResizeMode OnnxAttrs::get<nvinfer1::ResizeMode>(const std::string& key) const
{
    std::string mode = this->get<std::string>(key);
    if (mode == std::string("nearest"))
    {
      return nvinfer1::ResizeMode::kNEAREST;
    }
    if (mode == std::string("linear"))
    {
      return nvinfer1::ResizeMode::kLINEAR;
    }
    throw std::runtime_error("Unknown ResizeMode: " + mode);
}
