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

#include <onnx/onnx_pb.h>
#include <NvInfer.h>
#include <vector>
#include <unordered_map>

class OnnxAttrs {
  template<typename T>
  using string_map = std::unordered_map<std::string, T>;
  typedef string_map<::ONNX_NAMESPACE::AttributeProto const*> AttrMap;
  AttrMap _attrs;
public:
  explicit OnnxAttrs(::ONNX_NAMESPACE::NodeProto const& onnx_node) {
    for( auto const& attr : onnx_node.attribute() ) {
      _attrs.insert({attr.name(), &attr});
    }
  }
  bool count(std::string key) const { return _attrs.count(key); }
  ::ONNX_NAMESPACE::AttributeProto const* at(std::string key) const {
    if( !_attrs.count(key) ) {
      throw std::out_of_range("Attribute not found: " + key);
    }
    return _attrs.at(key);
  }
  template<typename T> T get(const std::string& key) const;
  template<typename T> T get(const std::string& key, T const& default_value) const {
    return _attrs.count(key) ? this->get<T>(key) : default_value;
  }
};
