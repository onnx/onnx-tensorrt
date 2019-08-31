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


  //需要保证所有input的rank一样
  rank =  this->getInputDims(0).nbDims;
  h_dims_a.resize(rank);
  h_dims_a_mul.resize(rank);

  h_dims_y.resize(rank);
  h_dims_y_mul.resize(rank);

  
  for(int i = 0 ; i<rank; i++)
  {
    h_dims_a[i] = this->getInputDims(0).d[i];
   
    h_dims_y[i] = output_dims.d[i];

  }
  h_dims_a_mul[rank-1]=1;
  h_dims_y_mul[rank-1]=1;
  for(int i = rank-2; i>=0 ; i--)
  {
      h_dims_a_mul[i]=h_dims_a_mul[i+1]*h_dims_a[i+1];
      h_dims_y_mul[i]=h_dims_y_mul[i+1]*h_dims_y[i+1];
  }

      
  _numbers = 1;
  for( int i=output_dims.nbDims-1; i>=0; i-- ) {
    _numbers *= output_dims.d[i];
  }

  //赋值给device vetor

    dims_a     =  h_dims_a        ;
    dims_a_mul =  h_dims_a_mul   ;
    dims_y     =  h_dims_y       ;
    dims_y_mul =  h_dims_y_mul   ;


  
  return 0;
}


//sds-temp，以下写法只验证了二维的输入，其它维度应该也是支持的。
template<typename T>
__global__ void expand_kernel(const int n, T const* __restrict__ a, T * __restrict__ y,
                                 int rank, int* aDim, int* aDimMul, int* yDim, int* yDimMul) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int a_index = 0;
        int tmp_index = index;
        for(int j = 0 ; j < rank; j++)
        {
            int temp1 = tmp_index / yDimMul[j];
            tmp_index = tmp_index % yDimMul[j];

            {
                a_index += (temp1 % aDim[j])* aDimMul[j];
            }
      
        }

        a_index += tmp_index;
        
        y[index] = a[a_index];
    }
  }



//sds-temp,当前仅测试了一种场景，1*128扩展到10*128
int ExpandPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  float const * idata1 = reinterpret_cast<float const *>(inputs[0]);
  float * odatas = reinterpret_cast<float *>(outputs[0]);

  //dim3 block(32, 16);
  //sds-temp, trt里的0对应最高维度
  //dim3 grid((output_dims.d[1]+32-1)/32, (output_dims.d[0]+16-1) / 16);

  dim3 block(512);
  dim3 grid((_numbers + 512 - 1) / 512);

  
  int * p_dims_a =thrust::raw_pointer_cast(&dims_a[0]);
  int * p_dims_a_mul =thrust::raw_pointer_cast(&dims_a_mul[0]);
  int * p_dims_y =thrust::raw_pointer_cast(&dims_y[0]);
  int * p_dims_y_mul =thrust::raw_pointer_cast(&dims_y_mul[0]);

  gdb_copy_to_cpu("Expand input", (float *)idata1,  h_dims_a[0]*h_dims_a_mul[0]);
    
  expand_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, odatas, rank, p_dims_a, p_dims_a_mul, p_dims_y, p_dims_y_mul);


  gdb_copy_to_cpu("Expand output", odatas, h_dims_y[0]*h_dims_y_mul[0]);

  return cudaGetLastError() != cudaSuccess;
}
