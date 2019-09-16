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

#include <onnx/onnx_pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <sstream>
#include <iostream>
#include <fstream>

using std::cerr;
using std::endl;

#pragma once

namespace {

// Helper function to convert ONNX dims to TRT dims
template<typename OnnxDims>
inline bool convert_dims(OnnxDims const& onnx_dims, nvinfer1::Dims& trt_dims)
{
  std::vector<int> onnx_dims_vector;
  std::vector<nvinfer1::DimensionType> onnx_type_vector;
  for( auto const& onnx_dim : onnx_dims ) {
    onnx_dims_vector.push_back((onnx_dim.dim_param() == "" ? onnx_dim.dim_value() : -1));
    onnx_type_vector.push_back(static_cast<nvinfer1::DimensionType>(0));
  }

  trt_dims.nbDims = onnx_dims_vector.size();
  if (trt_dims.nbDims > nvinfer1::Dims::MAX_DIMS){
    return false;
  }
  std::copy(onnx_dims_vector.begin(), onnx_dims_vector.end(), trt_dims.d);
  std::copy(onnx_type_vector.begin(), onnx_type_vector.end(), trt_dims.type);
  return true;
}

// Removes raw data from the text representation of an ONNX model
inline void remove_raw_data_strings(std::string& s) {
  std::string::size_type beg = 0;
  const std::string key = "raw_data: \"";
  const std::string sub = "...";
  while( (beg = s.find(key, beg)) != std::string::npos ) {
    beg += key.length();
    std::string::size_type end = beg - 1;
    // Note: Must skip over escaped end-quotes
    while( s[(end = s.find("\"", ++end)) - 1] == '\\' ) {}
    if( end - beg > 128 ) { // Only remove large data strings
      s.replace(beg, end - beg, "...");
    }
    beg += sub.length();
  }
}

// Removes float_data, int32_data etc. from the text representation of an ONNX model
inline std::string remove_repeated_data_strings(std::string& s) {
  std::istringstream iss(s);
  std::ostringstream oss;
  bool is_repeat = false;
  for( std::string line; std::getline(iss, line); ) {
    if(  line.find("float_data:") != std::string::npos ||
         line.find("int32_data:") != std::string::npos ||
         line.find("int64_data:") != std::string::npos ) {
      if( !is_repeat ) {
        is_repeat = true;
        oss << line.substr(0, line.find(":") + 1) << " ...\n";
      }
    } else {
      is_repeat = false;
      oss << line << "\n";
    }
  }
  return oss.str();
}

} // anonymous namespace

inline std::string pretty_print_onnx_to_string(::google::protobuf::Message const& message) {
  std::string s;
  ::google::protobuf::TextFormat::PrintToString(message, &s);
  remove_raw_data_strings(s);
  s = remove_repeated_data_strings(s);
  return s;
}

inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::ModelProto const& message) {
  stream << pretty_print_onnx_to_string(message);
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::NodeProto const& message) {
  stream << pretty_print_onnx_to_string(message);
  return stream;
}


//...
//...Consider moving all of the below functions into a stand alone
//...

inline bool ParseFromFile_WAR(google::protobuf::Message* msg,
                       const char*                filename) {
  
  std::ifstream stream(filename, std::ios::in | std::ios::binary);
  if (!stream) {
      cerr <<  "Could not open file " << std::string(filename) <<endl;
      return false;
  }
  google::protobuf::io::IstreamInputStream rawInput(&stream);
  
  google::protobuf::io::CodedInputStream coded_input(&rawInput);
  // Note: This WARs the very low default size limit (64MB)
  coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::max()/4);
  return msg->ParseFromCodedStream(&coded_input);
}

inline bool ParseFromTextFile(google::protobuf::Message* msg,
                       const char*                filename) {
  std::ifstream stream(filename, std::ios::in );
  if (!stream) {
      cerr <<  "Could not open file " << std::string(filename) <<endl;
      return false;
  }
      
  google::protobuf::io::IstreamInputStream rawInput(&stream);
  
  return google::protobuf::TextFormat::Parse(&rawInput, msg);
}

inline std::string onnx_ir_version_string(int64_t ir_version=::ONNX_NAMESPACE::IR_VERSION) {
int onnx_ir_major = ir_version / 1000000;
int onnx_ir_minor = ir_version % 1000000 / 10000;
int onnx_ir_patch = ir_version % 10000;
return (std::to_string(onnx_ir_major) + "." +
std::to_string(onnx_ir_minor) + "." +
std::to_string(onnx_ir_patch));
}

