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

#include "FancyActivation.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>

// TODO: Move this into a common header
size_t get_num_elements(nvinfer1::Dims dims) {
  size_t num_elements = 1;
  for( int d=0; d<dims.nbDims; ++d ) {
    num_elements *= dims.d[d];
  }
  return num_elements;
}

#define CAPTURE(...) [__VA_ARGS__]
#define UNARY_TRANSFORM(capture, func) \
  thrust::transform(thrust::cuda::par.on(stream), \
                    idata, idata + num_elements, odata, \
                    capture __device__ (float x) { return func; })


template <typename Data>
int FancyActivationPlugin::doEnqueue(int batchSize,
                                         const void *const *inputs, void **outputs,
                                         void *workspace, cudaStream_t stream) {
  size_t num_elements = batchSize * get_num_elements(this->getInputDims(0));

  Data const* idata = static_cast<Data const*>(inputs[0]);
  Data*       odata = static_cast<Data*      >(outputs[0]);
  // Note: These local-scope copies are needed for lambda capture
  float alpha = _alpha;
  float beta = _beta;

  switch( _activation_type ) {
  case LEAKY_RELU:   UNARY_TRANSFORM(CAPTURE(alpha), max(x, alpha * x)); break;
  case ELU:          UNARY_TRANSFORM(CAPTURE(alpha),
                                          x < 0 ?
                                          alpha * (expf(x) - 1) :
                                          x); break;
  case SELU:         UNARY_TRANSFORM(CAPTURE(alpha, beta),
                                          x <= 0 ?
                                          beta * (alpha * (expf(x) - 1.f)) :
                                          beta * x); break;
  case SOFTPLUS:     UNARY_TRANSFORM(CAPTURE(), logf(expf(x) + 1)); break;
  case SOFTSIGN:     UNARY_TRANSFORM(CAPTURE(), x / (1.f + fabs(x))); break;
  case HARD_SIGMOID: UNARY_TRANSFORM(CAPTURE(alpha, beta),
                                     max(0.f, min(1.f, alpha * x + beta))); break;
  case HARD_TANH:    UNARY_TRANSFORM(CAPTURE(), max(-1.f, min(1.f, x))); break;
  case CLIP:         UNARY_TRANSFORM(CAPTURE(alpha, beta),
                                     max(alpha, min(beta, x))); break;
  case FLOOR:        UNARY_TRANSFORM(CAPTURE(), floorf(x)); break;
  case CEIL:         UNARY_TRANSFORM(CAPTURE(), ceilf(x)); break;
  case THRESHOLDED_RELU: UNARY_TRANSFORM(CAPTURE(alpha), x > alpha ? x : 0); break;
  default: return -1;
  }
  return cudaGetLastError() != cudaSuccess;
}

int FancyActivationPlugin::enqueue(int batchSize,
                                   const void *const *inputs, void **outputs,
                                   void *workspace, cudaStream_t stream) {
  if (getDataType()==nvinfer1::DataType::kFLOAT) {        
    return doEnqueue<float>(batchSize, inputs, outputs, workspace, stream);
  } else {
#if CUDART_VERSION < 9000
    throw std::runtime_error("FP16 plugin is not support for CUDA < 9.0");
#else    
    return doEnqueue<__half>(batchSize, inputs, outputs, workspace, stream);
#endif  
  }
}

bool FancyActivationPlugin::supportsFormat(nvinfer1::DataType type,
                                           nvinfer1::PluginFormat format) const {
  return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF);
}
