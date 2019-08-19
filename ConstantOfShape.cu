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

#include "ConstantOfShape.hpp"

//sds:对于这个split来说，这里的index必须是0，表示只有一个输出
//sds: index,The index of the output tensor.
nvinfer1::Dims ConstantOfShapePlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  // 'equal' and 'less' is 2, 'where' is 3
  assert(nbInputs == 1);
  assert(index == 0);// only one output
  nvinfer1::Dims const& input_dims = inputDims[0];
  //output_dims == input_dims[0]
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int ConstantOfShapePlugin::initialize() {
  nvinfer1::Dims dims = this->getInputDims(0);
  _numbers = 1;
  for( int i=dims.nbDims-1; i>=0; i-- ) {
    _numbers *= dims.d[i];
  }
  return 0;
}  



template<typename T>
__global__ void constant_shape_kernel(const int n, T value, T const* __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        y[index] = value;
    }
  }
}

  
//sds:这里的inputs是在显存，outputs是在内存。
//sds:每个plugin进入enqueue，带过来的inputs有可能在内存或者显存，由addPluginV2时传入的inputs决定
//    inputs是一个指针，维度信息需要自己初始化，比如在initialize中或者enqueue中。
//    outputs? outputs已经按照getOutputDims指定的初始化，应该都是gpu指针吧?
int ConstantOfShapePlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  //float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  float const* odatas = reinterpret_cast<float const*>(outputs[0]);

  dim3 block(512);
  dim3 grid((_numbers + 512 - 1) / 512);
      
  constant_shape_kernel<<<_numbers, grid, block, 0, stream>>>(_value, odatas);

  return cudaGetLastError() != cudaSuccess;
}
