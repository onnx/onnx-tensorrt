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

#include "InstanceNormalization.hpp"
#include <cassert>

#define CHECK_CUDA(call) do {    \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    return status; \
  } \
} while(0)

#define CHECK_CUDNN(call) do {    \
  cudnnStatus_t status = call; \
  if( status != CUDNN_STATUS_SUCCESS ) { \
    return status; \
  } \
} while(0)

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value) {
  union F32 { unsigned int u; float f; };
  static const F32 magic      = { (254 - 15) << 23 };
  static const F32 was_infnan = { (127 + 16) << 23 };
  F32 result;
  result.u = (value & 0x7fff) << 13;  // exponent/mantissa bits
  result.f *= magic.f;                // exponent adjust
  if (result.f >= was_infnan.f) {     // make sure Inf/NaN survive
    result.u |= 255 << 23;
  }
  result.u |= (value & 0x8000) << 16; // sign bit
  return result.f;
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(float epsilon,
                                                         nvinfer1::Weights const& scale,
                                                         nvinfer1::Weights const& bias)
  : _epsilon(epsilon), _nchan(scale.count),  _initialized(false) {
  assert(scale.count == bias.count);
  if( scale.type == nvinfer1::DataType::kFLOAT ) {
    _h_scale.assign((float*)scale.values, (float*)scale.values + scale.count);
  } else if( scale.type == nvinfer1::DataType::kHALF ) {
    _h_scale.reserve(_nchan);
    for( int c=0; c<_nchan; ++c ) {
      unsigned short value = ((unsigned short*)scale.values)[c];
      _h_scale.push_back(half_to_float_fast(value));
    }
  } else {
    throw std::runtime_error("Unsupported scale dtype");
  }
  if( bias.type == nvinfer1::DataType::kFLOAT ) {
    _h_bias.assign((float*)bias.values, (float*)bias.values + bias.count);
  } else if( bias.type == nvinfer1::DataType::kHALF ) {
    _h_bias.reserve(_nchan);
    for( int c=0; c<_nchan; ++c ) {
      unsigned short value = ((unsigned short*)bias.values)[c];
      _h_bias.push_back(half_to_float_fast(value));
    }
  } else {
    throw std::runtime_error("Unsupported bias dtype");
  }
}

int InstanceNormalizationPlugin::initialize() {
  if (_initialized) {
    return 0;
  }
  nvinfer1::Dims input_dims = this->getInputDims(0);
  (void)input_dims;
  assert(is_CHW(input_dims));
  assert(input_dims.d[0] == _nchan);
  int nbatch = this->getMaxBatchSize();
  size_t nchan_bytes = _nchan * sizeof(float);
  // Note: We repeat the data for each batch entry so that we can do the full
  //       computation in a single CUDNN call in enqueue().
  CHECK_CUDA(cudaMalloc((void**)&_d_scale, nbatch * nchan_bytes));
  CHECK_CUDA(cudaMalloc((void**)&_d_bias,  nbatch * nchan_bytes));
  for( int i=0; i<nbatch; ++i ) {
    CHECK_CUDA(cudaMemcpy(_d_scale + i * _nchan, _h_scale.data(), nchan_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(_d_bias  + i * _nchan,  _h_bias.data(), nchan_bytes, cudaMemcpyHostToDevice));
  }
  CHECK_CUDNN(cudnnCreate(&_cudnn_handle));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_b_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_y_desc));
  _initialized = true;
  return 0;
}

void InstanceNormalizationPlugin::terminate() {
  if (!_initialized) {
    return;
  }
  cudnnDestroyTensorDescriptor(_y_desc);
  cudnnDestroyTensorDescriptor(_x_desc);
  cudnnDestroyTensorDescriptor(_b_desc);
  cudaFree(_d_bias);
  cudaFree(_d_scale);
  cudnnDestroy(_cudnn_handle);
  _initialized = false;
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin() {
  terminate();
}

bool InstanceNormalizationPlugin::supportsFormat(nvinfer1::DataType type,
                                                 nvinfer1::PluginFormat format) const {
  return ((type == nvinfer1::DataType::kFLOAT ||
           type == nvinfer1::DataType::kHALF) &&
          format == nvinfer1::PluginFormat::kNCHW);
}

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype,
                                      cudnnDataType_t* cudnn_dtype) {
  switch( trt_dtype ) {
  case nvinfer1::DataType::kFLOAT: *cudnn_dtype = CUDNN_DATA_FLOAT; break;
  case nvinfer1::DataType::kHALF:  *cudnn_dtype = CUDNN_DATA_HALF;  break;
  default: return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

int InstanceNormalizationPlugin::enqueue(int batchSize,
                                         const void *const *inputs, void **outputs,
                                         void *workspace, cudaStream_t stream) {
  assert(_initialized);
  nvinfer1::Dims input_dims = this->getInputDims(0);
  int n = batchSize;
  int c = input_dims.d[0];
  int h = input_dims.d[1];
  int w = input_dims.d[2];
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n*c, 1, 1));
  cudnnDataType_t cudnn_dtype;
  CHECK_CUDNN(convert_trt2cudnn_dtype(this->getDataType(), &cudnn_dtype));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n*c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n*c, h, w));
  float alpha = 1;
  float beta  = 0;
  void const* x_ptr = inputs[0];
  void*       y_ptr = outputs[0];
  CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));
  // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
  //       overflows (NaNs) for fp32 data in some circumstances. The lower-
  //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
  //       acceptable.
  CHECK_CUDNN(
      cudnnBatchNormalizationForwardTraining(
          _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
          _x_desc, x_ptr, _y_desc, y_ptr, _b_desc, _d_scale, _d_bias,
          1., nullptr, nullptr, _epsilon, nullptr, nullptr));
  return 0;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(int maxBatchSize) const {
  return 0;
}
