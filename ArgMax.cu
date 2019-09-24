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

#include "ArgMax.hpp"
#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ArgMaxPlugin::getOutputDimensions(int index,
                                                 const nvinfer1::Dims *inputDims,
                                                 int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(_ndims == 2);
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  for( int d=0; d<input.nbDims; ++d ) {
    output.type[d] = input.type[d];
    if( input.type[d] == nvinfer1::DimensionType::kCHANNEL ) {
      output.d[d] = 1;
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
}

int ArgMaxPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);

  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  assert(_ndims == 2);
  return 0;
}

__global__
void argmax_kernel(int channels, int osize, float const * idata, unsigned char * odata) 
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  float max = -1e6;
  unsigned char am = 0;
  
  for (unsigned char i = 0; i < channels; ++i) {
    float v = idata[x + i * osize]; 
    if (v >= max) {
      max = v;
      am = i;
    } 
  }
  
  odata[x] = am;
}

__global__
void argmax_kernel(int channels, int osize, __half const * idata, unsigned char * odata) 
{

}

int ArgMaxPlugin::enqueue(int batchSize,
                          const void *const *inputs, 
                          void **outputs,
                          void *workspace,
                          cudaStream_t stream) {
  
  auto const & input_dims = this->getInputDims(0);
  int channels = input_dims.d[0];
  int osize = _output_dims.d[2] * _output_dims.d[1];
  unsigned char * output = static_cast<unsigned char *>(outputs[0]);

  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties(&dev_prop, 0);
  
  dim3 block(dev_prop.maxThreadsPerBlock, 1);
  dim3 grid(std::ceil(osize / block.x));

  if (getDataType() == nvinfer1::DataType::kFLOAT) {
    argmax_kernel<<<grid, block, 0, stream>>>(
      channels,
      osize,
      static_cast<float const *>(inputs[0]), 
      output
    );
  } else if (getDataType() == nvinfer1::DataType::kHALF) {
    argmax_kernel<<<grid, block, 0, stream>>>(
      channels,
      osize,
      static_cast<__half const *>(inputs[0]), 
      output
    );
  }

  return cudaGetLastError() != cudaSuccess;
}
