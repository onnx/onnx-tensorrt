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

#include "NonZero.hpp"

//sds:对于这个split来说，这里的index必须是0，表示只有一个输出
//sds: index,The index of the output tensor.
nvinfer1::Dims NonZeroPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  // 'equal' and 'less' is 2, 'where' is 3
  assert(nbInputs == 1);
  assert(index == 0);// only one output
  nvinfer1::Dims const& input_dims = inputDims[0];
  //output_dims == input_dims[0]
  //nvinfer1::Dims output_dims(input_dims.nbDims, _numbers);
  //sds 输入的每个lement必须都是非0(实际上前层是constantOfShape,每个element都是1).满足这个条件，则输出必是rows*numbers
  nvinfer1::Dims output_dims;
  output_dims.nbDims=2;
  output_dims.d[0]=_rows;
  output_dims.d[1]=_numbers;
  return output_dims;
}

int NonZeroPlugin::initialize() {
  nvinfer1::Dims dims = this->getInputDims(0);
  _rows = dims.nbDims;
  _numbers = 1;
  
  int* hLensOfDim = new int[_rows];
  int* hmulOfSon = new int[_rows];
  for( int i=0; i<=dims.nbDims-1; i++ ) {
    _numbers *= dims.d[i];
    hLensOfDim[i]= dims.d[i];
  }


  hmulOfSon[_rows-1]=1;
  for( int i=_rows-2; i>=0; i-- ) {
    hmulOfSon[i]=hmulOfSon[i+1] * hLensOfDim[i+1];
  }
  
  int length= sizeof(int) * _rows;
  CHECK_CUDA(cudaMalloc((void**)&_lensOfDim, length));
  CHECK_CUDA(cudaMemcpy(_lensOfDim, hLensOfDim, length, cudaMemcpyHostToDevice));

  
  CHECK_CUDA(cudaMalloc((void**)&_mulOfSon, length));
  CHECK_CUDA(cudaMemcpy(_mulOfSon, hmulOfSon, length, cudaMemcpyHostToDevice));

  delete []hLensOfDim;
  delete []hmulOfSon;
  return 0;
}


template<typename T>
__global__ void non_zero_kernel(const int columns, int* _lenOfDim, int* _mulOfSon,T * __restrict__ y) {
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(x_index < columns)
    {
        int y_index = threadIdx.y;
        //sds,每个element赋值满足下面的规律
        y[y_index*columns + x_index] = (T)((x_index/_mulOfSon[y_index]) % _lenOfDim[y_index]);
    }
    
  }


int NonZeroPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  //float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  float * odatas = reinterpret_cast<float *>(outputs[0]);

  dim3 block(_rows, 512);
  dim3 grid(1, (_numbers + 512 - 1) / 512);
  extern __shared__ int lensOfDim[];
  non_zero_kernel<<<grid, block, 0, stream>>>(_numbers, _lensOfDim, _mulOfSon, odatas);

  return cudaGetLastError() != cudaSuccess;
}
