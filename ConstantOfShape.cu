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
  // input shape and input value is weights
  assert(nbInputs == 0);
  assert(index == 0);// only one output
  return output_dims;
}

int ConstantOfShapePlugin::initialize() {
 // nvinfer1::Dims dims = this->getInputDims(0);
  _numbers = 1;
  for( int i=output_dims.nbDims-1; i>=0; i-- ) {
    _numbers *= output_dims.d[i];
  }
  return 0;
}  



template<typename T>
__global__ void constant_shape_kernel(unsigned long long n, T value, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        y[index] = value;
    }
  }




//sds-temp,仅支持input[0] 是shape, input[1]是scalar的输入
int ConstantOfShapePlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  //float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  float * odatas = reinterpret_cast<float *>(outputs[0]);

  dim3 block(512);
  dim3 grid((_numbers + 512 - 1) / 512);
  constant_shape_kernel<<<grid, block, 0, stream>>>(_numbers, _value, odatas);

  gdb_copy_to_cpu("constantOfShape", odatas, _numbers);

  return cudaGetLastError() != cudaSuccess;
}
