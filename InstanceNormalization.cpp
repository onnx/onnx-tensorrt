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
  if (_initialized)
    return 0;
  
  nvinfer1::Dims input_dims = this->getInputDims(0);
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
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_x_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_y_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_b_desc));

  int c = _nchan;
  int h = input_dims.d[1];
  int w = input_dims.d[2];
  if (getDataType() == nvinfer1::DataType::kFLOAT) {   
    CHECK_CUDA(cudaMalloc((void**)&_x_fp16_buf, nbatch * c * h * w * sizeof(half_type)));
    CHECK_CUDA(cudaMalloc((void**)&_y_fp16_buf, nbatch * c * h * w * sizeof(half_type)));
  }
  
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_x_fp16_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&_y_fp16_desc));
  _initialized = true;
  return 0;
}

void InstanceNormalizationPlugin::terminate() {
  if (!_initialized)
    return;
  
  cudnnDestroyTensorDescriptor(_y_fp16_desc);
  cudnnDestroyTensorDescriptor(_x_fp16_desc);
  if (getDataType() == nvinfer1::DataType::kFLOAT) {  
    cudaFree(_y_fp16_buf);
    cudaFree(_x_fp16_buf);
  }
  cudnnDestroyTensorDescriptor(_b_desc);
  cudnnDestroyTensorDescriptor(_y_desc);
  cudnnDestroyTensorDescriptor(_x_desc);
  cudaFree(_d_bias);
  cudaFree(_d_scale);
  cudnnDestroy(_cudnn_handle);
  _initialized = false;
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin()
 {
   terminate();
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
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n*c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n*c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n*c, 1, 1));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_x_fp16_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, n*c, h, w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(_y_fp16_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, n*c, h, w));

  float alpha = 1;
  float beta  = 0;
  float const* x_ptr = static_cast<float const*>(inputs[0]);
  float*       y_ptr = static_cast<float*      >(outputs[0]);
  
  CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));

  if (getDataType()==nvinfer1::DataType::kFLOAT) {
    CHECK_CUDNN(cudnnTransformTensor(_cudnn_handle,
                                     &alpha, _x_desc, x_ptr,
                                     &beta, _x_fp16_desc, _x_fp16_buf));
  } else {
    _x_fp16_buf = const_cast<half_type*>(reinterpret_cast<const half_type*>(x_ptr));
    _y_fp16_buf = reinterpret_cast<half_type*>(y_ptr);
  }
    // TODO: Need to check for failure if using the persistent algo?
  CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
                _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
                _x_fp16_desc, _x_fp16_buf, _y_fp16_desc, _y_fp16_buf, _b_desc, _d_scale, _d_bias,
                1., nullptr, nullptr, _epsilon, nullptr, nullptr));
  
  if (getDataType()==nvinfer1::DataType::kFLOAT) {
    CHECK_CUDNN(cudnnTransformTensor(_cudnn_handle,
                                     &alpha, _y_fp16_desc, _y_fp16_buf,
                                     &beta, _y_desc, y_ptr));
  }
  
  return 0;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(int maxBatchSize) const 
{
  int nbatch = this->getMaxBatchSize();
  int typesize = sizeof(float);
  
  nvinfer1::Dims input_dims = this->getInputDims(0);
  size_t ws =  _nchan * sizeof(float) *  nbatch * 2 ; // scale and
                                                      // bias
  #if NV_TENSORRT_MAJOR >= 4
  typesize = (getDataType()==nvinfer1::DataType::kFLOAT?sizeof(float): sizeof(half_type));
  int c = _nchan;
  int h = input_dims.d[1];
  int w = input_dims.d[2];
  ws += (nbatch * c * h * w * typesize)*2; // in/out
  # endif
  return ws;
}

