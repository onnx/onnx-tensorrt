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

#include "NvOnnxParser.h"

#include <string>
#include <cassert>

#define MAKE_ERROR(desc, code) \
  Status((code), (desc), __FILE__, __LINE__, __func__)

#define ASSERT(condition, error_code) do { \
    if( !(condition) ) { \
      return MAKE_ERROR("Assertion failed: " #condition, (error_code)); \
    } \
  } while(0)

#define MAKE_INPUT_ERROR(desc, code, name) \
  Status((code), (desc), name, __LINE__, __func__)

#define ASSERT_INPUT(condition, error_code, name) do {\
    if ( !(condition) ) {\
      return MAKE_INPUT_ERROR("Assertion failed: " #condition, (error_code), (name)); \
    } \
  } while(0)

#define ASSERT_C(condition, error_code) do { \
    if( !(condition) ) { \
      return error_code; \
    } \
  } while(0)

#define GET_VALUE(value_or_error_, result_ptr) do { \
    auto const& value_or_error = value_or_error_; \
    if( value_or_error.is_error() ) { \
      return value_or_error.error(); \
    } else { \
      *result_ptr = value_or_error.value(); \
    } \
  } while(0)

#define TRT_CHECK(call) do { \
    Status status = call; \
    if( !status.is_success() ) { \
      return status; \
    } \
  } while(0)

namespace onnx2trt {

using nvonnxparser::ErrorCode;

class Status : public nvonnxparser::IParserError {
  ErrorCode   _code;
  std::string _desc;
  std::string _file;
  int         _line;
  std::string _func;
  int         _node;
public:
  static Status success() { return Status(ErrorCode::kSUCCESS); }
  Status() {}
  explicit Status(ErrorCode code, std::string desc="",
                  std::string file="", int line=0, std::string func="",
                  int node=-1)
    : _code(code), _desc(desc), _file(file), _line(line), _func(func),
      _node(node) {}
  ErrorCode   code() const override { return _code; }
  const char* desc() const override { return _desc.c_str(); }
  const char* file() const override { return _file.c_str(); }
  int         line() const override { return _line; }
  const char* func() const override { return _func.c_str(); }
  int         node() const override { return _node; }
  bool is_error()   const { return _code != ErrorCode::kSUCCESS; }
  bool is_success() const { return _code == ErrorCode::kSUCCESS; }
  void setNode(int node) { _node = node; }
};

template<typename T>
class ValueOrStatus {
  bool   _is_error;
  T      _value;
  Status _error;
public:
  ValueOrStatus(T const& value) : _is_error(false), _value(value), _error(Status::success()) {}
  ValueOrStatus(T&& value)      : _is_error(false), _value(value), _error(Status::success()) {}
  ValueOrStatus(Status const& error) : _is_error(true), _error(error) {}
  ValueOrStatus(Status&& error)      : _is_error(true), _error(error) {}
  bool is_error() const { return _is_error; }
  T     const& value() const { assert(!_is_error); return _value; }
  T&           value()       { assert(!_is_error); return _value; }
  Status const& error() const { assert( _is_error); return _error; }
};

} // namespace onnx2trt