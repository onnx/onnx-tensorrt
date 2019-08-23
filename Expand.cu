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

#include "Expand.hpp"

//sds:对于这个split来说，这里的index必须是0，表示只有一个输出
//sds: index,The index of the output tensor.
nvinfer1::Dims ExpandPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  
  assert(index == 0);// only one output
  // input and shape
  assert(nbInputs == 1);
  
  return output_dims;
}

int ExpandPlugin::initialize() {
  nvinfer1::Dims const& input_dims1 =this->getInputDims(0);
  input_dims = input_dims1;
  return 0;
}


//sds，以下写法只满足是二维的输入
template<typename T>
__global__ void expand_kernel(const int rows, const int columns, const int dim_src_y, const int dim_src_x,T const* __restrict__ x, T * __restrict__ y) {
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    int temp_x, temp_y;
    if(x_index < columns && y_index < rows)
    {
    temp_x = x_index%dim_src_x;
    temp_y = y_index%dim_src_y;
        y[y_index*columns + x_index] = x[temp_y*dim_src_x + temp_x];
    }
    
  }


int ExpandPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  float * odatas = reinterpret_cast<float *>(outputs[0]);

  dim3 block(32, 16);
  dim3 grid((output_dims.d[0]+32-1)/32, (output_dims.d[1]+16-1) / 16);
  expand_kernel<<<grid, block, 0, stream>>>(output_dims.d[0],output_dims.d[1], input_dims.d[0], input_dims.d[1], idata1, odatas);

  return cudaGetLastError() != cudaSuccess;
}
