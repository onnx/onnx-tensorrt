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

#include <algorithm>
#include <cuda_fp16.h>
#include <cassert>

#include "Split.hpp"

nvinfer1::Dims SplitPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const& input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[_axis] = _output_lengths.at(index);
  return output_dims;
}

int SplitPlugin::initialize() {
  std::vector<int> segment_offsets(1, 0);
  for( int i=0; i<this->getNbOutputs(); ++i ) {
    segment_offsets.push_back(segment_offsets.back() + _output_lengths[i]);
  }
  _d_segment_offsets = segment_offsets;
  nvinfer1::Dims dims = this->getInputDims(0);
  _nx = 1;
  for( int i=dims.nbDims-1; i>_axis; --i ) {
    _nx *= dims.d[i];
  }
  _ny = dims.d[_axis];
  _nz = 1;
  for( int i=_axis-1; i>=0; --i ) {
    _nz *= dims.d[i];
  }
  _d_output_ptrs.resize(this->getNbOutputs(), nullptr);
  return 0;
}

template<typename T>
__device__
int upper_bound(T const* vals, int n, T const& key) {
  int i = 0;
  while( n > 0 ) {
    int m = n / 2;
    int j = i + m;
    if( !(key < vals[j]) ) {
      i  = j + 1;
      n -= m + 1;
    } else {
      n = m;
    }
  }
  return i;
}

template<typename T>
__global__
void split_kernel(int nsegment,
                  int const* __restrict__ segment_offsets,
                  T   const* __restrict__ idata,
                  T*  const* odatas,
                  int nx,
                  int src_ny,
                  int nz) {
  int x0     = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0     = threadIdx.z + blockIdx.z * blockDim.z;
  for( int z=z0; z<nz; z+=blockDim.z*gridDim.z ) {
    for( int src_y=src_y0; src_y<src_ny; src_y+=blockDim.y*gridDim.y ) {
      for( int x=x0; x<nx; x+=blockDim.x*gridDim.x ) {
  int segment = upper_bound(segment_offsets, nsegment, src_y) - 1;
  int dst_y = src_y - segment_offsets[segment];
  int dst_ny = segment_offsets[segment + 1] - segment_offsets[segment];
  odatas[segment][x + nx*(dst_y + dst_ny*z)] =
            idata[x + nx*(src_y + src_ny*z)];
      }
    }
  }
}

int SplitPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int const* d_segment_offsets_ptr =
    thrust::raw_pointer_cast(&_d_segment_offsets[0]);
  float  const* idata    = reinterpret_cast<float  const*>(inputs[0]);
  float* const* h_odatas = reinterpret_cast<float* const*>(outputs);
  float** odatas = thrust::raw_pointer_cast(&_d_output_ptrs[0]);
  cudaError_t cuda_status =
    cudaMemcpyAsync(odatas, h_odatas,
                    _d_output_ptrs.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
  if( cuda_status != cudaSuccess ) {
    return 1;
  }
  int nz = _nz * batchSize;
  dim3 block(32, 16);
  dim3 grid(std::min((_nx - 1) / block.x + 1, 65535u),
            std::min((_ny - 1) / block.y + 1, 65535u),
            std::min((_nz - 1) / block.z + 1, 65535u));
  if (getDataType()==nvinfer1::DataType::kFLOAT) {      
    split_kernel<<<grid, block, 0, stream>>>
      (_d_segment_offsets.size(), d_segment_offsets_ptr, idata, odatas,
       _nx, _ny, nz);
  } else {
    split_kernel<<<grid, block, 0, stream>>>
      (_d_segment_offsets.size(), d_segment_offsets_ptr, (__half const*)idata, (__half**)odatas,
       _nx, _ny, nz);
  }
  return cudaGetLastError() != cudaSuccess;
}
