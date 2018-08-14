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
#include "trt_utils.hpp"
#include "OnnxAttrs.hpp"

#include <onnx/onnx.pb.h>
#include <onnx/onnxifi.h>
#include <NvInfer.h>

#include <iostream>
using std::cerr;
using std::endl;

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape) {
  if( shape.nbDims == 0 ) {
    return stream;
  }
  stream << "(" << shape.d[0];
  for( int i=1; i<shape.nbDims; ++i ) {
    stream << ", " << shape.d[i];
  }
  stream << ")";
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype) {
  switch( dtype ) {
  case nvinfer1::DataType::kFLOAT: return stream << "float32";
  case nvinfer1::DataType::kHALF:  return stream << "float16";
  case nvinfer1::DataType::kINT8:  return stream << "int8";
#if NV_TENSORRT_MAJOR >= 4
  case nvinfer1::DataType::kINT32: return stream << "int32";
#endif
  default: throw std::runtime_error("Unknown dtype");
  }
}

// TODO: Remove this when finished debugging
inline std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm) {
  int ndims = nvinfer1::Dims::MAX_DIMS;
  stream << "(" << perm.order[0];
  for( int i=1; i<ndims; ++i ) {
    stream << ", " << perm.order[i];
  }
  stream << ")";
  return stream;
}
/*
// TODO: Remove this when finished debugging
inline std::ostream& operator<<(std::ostream& stream, google::protobuf::Message const& message) {
  stream << print_onnx_to_string(message);
  return stream;
}
*/
namespace onnx2trt {

inline int get_dtype_size(::ONNX_NAMESPACE::TensorProto::DataType onnx_dtype) {
  switch( onnx_dtype ) {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:   return 4;
  case ::ONNX_NAMESPACE::TensorProto::INT8:    return 1;
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return 2;
  case ::ONNX_NAMESPACE::TensorProto::INT32:   return 4;
  case ::ONNX_NAMESPACE::TensorProto::INT64:   return 8;
  // TODO: Add remaining ONNX types...
    // TODO: Some sort of error handling
  default: return -1;//throw std::invalid_argument("Unsupported TRT data type: " +
                     //                  std::to_string((int)trt_dtype));
  }
}

inline const char* get_dtype_name(::ONNX_NAMESPACE::TensorProto::DataType onnx_dtype) {
  switch( onnx_dtype ) {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:      return "FLOAT";
  case ::ONNX_NAMESPACE::TensorProto::UINT8:      return "UINT8";
  case ::ONNX_NAMESPACE::TensorProto::INT8:       return "INT8";
  case ::ONNX_NAMESPACE::TensorProto::UINT16:     return "UINT16";
  case ::ONNX_NAMESPACE::TensorProto::INT16:      return "INT16";
  case ::ONNX_NAMESPACE::TensorProto::INT32:      return "INT32";
  case ::ONNX_NAMESPACE::TensorProto::INT64:      return "INT64";
  case ::ONNX_NAMESPACE::TensorProto::STRING:     return "STRING";
  case ::ONNX_NAMESPACE::TensorProto::BOOL:       return "BOOL";
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16:    return "FLOAT16";
  case ::ONNX_NAMESPACE::TensorProto::DOUBLE:     return "DOUBLE";
  case ::ONNX_NAMESPACE::TensorProto::UINT32:     return "UINT32";
  case ::ONNX_NAMESPACE::TensorProto::UINT64:     return "UINT64";
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:  return "COMPLEX64";
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return "COMPLEX128";
  default: return "<UNKNOWN>";
  }
}

inline bool convert_dtype(::ONNX_NAMESPACE::TensorProto::DataType onnx_dtype,
                          nvinfer1::DataType* trt_dtype) {
  switch( onnx_dtype ) {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:   *trt_dtype = nvinfer1::DataType::kFLOAT; break;
  case ::ONNX_NAMESPACE::TensorProto::INT8:    *trt_dtype = nvinfer1::DataType::kINT8;  break;
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16: *trt_dtype = nvinfer1::DataType::kHALF;  break;
#if NV_TENSORRT_MAJOR >= 4
  case ::ONNX_NAMESPACE::TensorProto::INT32:   *trt_dtype = nvinfer1::DataType::kINT32; break;
#endif
  default:
    cerr << "Unsupported ONNX data type: " << get_dtype_name(onnx_dtype)
         << " (" << std::to_string(onnx_dtype) << ")" << endl;
    return false;
  }
  return true;
}

template<typename OnnxDims>
inline nvinfer1::Dims convert_dims(OnnxDims const& onnx_dims) {
  enum { BATCH_DIM = 0 };
  std::vector<int> onnx_dims_vector;
  for( auto const& onnx_dim : onnx_dims ) {
    // TODO: Unknown dimensions are represented using onnx_dim.dim_param
    onnx_dims_vector.push_back(onnx_dim.dim_value());
  }
  nvinfer1::Dims trt_dims;
  trt_dims.nbDims = onnx_dims_vector.size();
  assert(trt_dims.nbDims <= nvinfer1::Dims::MAX_DIMS);
  std::copy(onnx_dims_vector.begin(), onnx_dims_vector.end(), trt_dims.d);
  trt_dims = set_dims_CHW(remove_dim(trt_dims, BATCH_DIM));
  return trt_dims;
}

inline bool convert_weight_descriptor(onnxTensorDescriptorV1 const &desc,
                                      onnx2trt::ShapedWeights *weights) {
  nvinfer1::Dims shape;
  shape.nbDims = desc.dimensions;
  // Special case for scalars
  if( shape.nbDims == 0 ) {
    shape.nbDims = 1;
    shape.d[0] = 1;
    shape.type[0] = nvinfer1::DimensionType::kCHANNEL;
  } else {
    std::copy(desc.shape, desc.shape + desc.dimensions, shape.d);
  }

  size_t element_count = 1;
  for (int i = 0; i < shape.nbDims; ++i) {
    element_count *= shape.d[i];
  }

  void* data_ptr;
  size_t nbytes;
  ::ONNX_NAMESPACE::TensorProto::DataType dtype;
  data_ptr = (void*)(desc. buffer);
  if (desc.dataType == ONNXIFI_DATATYPE_FLOAT32) {
    dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    nbytes = element_count * sizeof(float);
  } else if (desc.dataType == ONNXIFI_DATATYPE_FLOAT16) {
    dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    nbytes = element_count * sizeof(float) / 2;
  } else if (desc.dataType == ONNXIFI_DATATYPE_INT32) {
    dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    nbytes = element_count * sizeof(int32_t);
  } else if (desc.dataType == ONNXIFI_DATATYPE_INT64) {
    dtype = ::ONNX_NAMESPACE::TensorProto::INT64;
    nbytes = element_count * sizeof(int64_t);
  } else {
    // Unsupported format
    return false;
  }

  onnx2trt::ShapedWeights trt_weights(dtype, data_ptr, shape);
  (void)nbytes;
  assert(trt_weights.size_bytes() == nbytes);
  *weights = trt_weights;
  return true;
}

inline bool convert_onnx_weights(::ONNX_NAMESPACE::TensorProto const& onnx_tensor,
                                 onnx2trt::ShapedWeights* weights) {
  nvinfer1::Dims shape;
  shape.nbDims = onnx_tensor.dims().size();
  std::copy(onnx_tensor.dims().begin(), onnx_tensor.dims().end(),
            shape.d);
  // Special case for scalars
  if( shape.nbDims == 0 ) {
    shape.nbDims = 1;
    shape.d[0] = 1;
    shape.type[0] = nvinfer1::DimensionType::kCHANNEL;
  }
  auto dtype = onnx_tensor.data_type();
  void* data_ptr; // TODO: See if can make const*
  size_t nbytes;
  if( onnx_tensor.raw_data().size() > 0 ) {
    data_ptr = (void*)onnx_tensor.raw_data().data();
    nbytes = onnx_tensor.raw_data().size();
  } else if( onnx_tensor.float_data().size() > 0 ) {
    assert(onnx_tensor.data_type() == ::ONNX_NAMESPACE::TensorProto::FLOAT);
    data_ptr = (void*)onnx_tensor.float_data().data();
    nbytes = onnx_tensor.float_data().size() * sizeof(float);
  } else if( onnx_tensor.int32_data().size() > 0 ) {
    // TODO: Need special handling for int8 or float16 stored as int32_data
    assert(get_dtype_size(dtype) == 4);
    data_ptr = (void*)onnx_tensor.int32_data().data();
    nbytes = onnx_tensor.int32_data().size() * sizeof(int32_t);
  } else if( onnx_tensor.int64_data().size() > 0 ) {
    assert(onnx_tensor.data_type() == ::ONNX_NAMESPACE::TensorProto::INT64);
    data_ptr = (void*)onnx_tensor.int64_data().data();
    nbytes = onnx_tensor.int64_data().size() * sizeof(int64_t);
  } else {
    // Unsupported ONNX tensor format!
    return false;
  }

  onnx2trt::ShapedWeights trt_weights(dtype, data_ptr, shape);
  (void)nbytes;
  assert(trt_weights.size_bytes() == nbytes);
  *weights = trt_weights;
  return true;
}

inline int div_ceil(int n, int d) {
  return (n - 1) / d + 1;
}

inline int get_conv_output_size(int input_size, int filter_size,
                                int stride, int dilation_rate,
                                int total_padding) {
  // This is based on the CUDNN formula
  int effective_input_size  = input_size + total_padding;
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  return div_ceil(effective_input_size - (effective_filter_size - 1), stride);
}

void get_kernel_params(::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::DimsHW const& input_shape,
                       nvinfer1::DimsHW* kernel_size,
                       nvinfer1::DimsHW* strides,
                       nvinfer1::DimsHW* beg_padding,
                       nvinfer1::DimsHW* end_padding,
                       nvinfer1::DimsHW* dilations=nullptr,
                       nvinfer1::DimsHW const* output_shape=nullptr,
                       bool enable_padding_trick=true);

inline nvinfer1::ScaleMode get_scale_mode(nvinfer1::Dims const& weights_shape) {
  if( weights_shape.nbDims == 1 ) {
    if( weights_shape.d[0] == 1 ) {
      return nvinfer1::ScaleMode::kUNIFORM;
    } else {
      return nvinfer1::ScaleMode::kCHANNEL;
    }
  } else {
    return nvinfer1::ScaleMode::kELEMENTWISE;
  }
}

} // namespace onnx2trt
