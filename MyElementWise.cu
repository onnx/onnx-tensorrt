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

#include "MyElementWise.hpp"

//sds:对于这个split来说，这里的index必须是0
//sds: index,The index of the output tensor.
nvinfer1::Dims MyElementWisePlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  // 'not' is 1, 'equal' and 'less' is 2, 'where' is 3
  assert(nbInputs == 1 || nbInputs == 2 || nbInputs == 3);
  assert(index == 0);// only one output
  nvinfer1::Dims const& input_dims = inputDims[0];
  //output_dims == input_dims[0]
  nvinfer1::Dims output_dims = input_dims;
  return output_dims;
}

int MyElementWisePlugin::initialize() {
  nvinfer1::Dims dims = this->getInputDims(0);
  _numbers = 1;
  for( int i=dims.nbDims-1; i>=0; i-- ) {
    _numbers *= dims.d[i];
  }
  return 0;
}


/*
Returns the tensor resulted from performing the equal logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).

This operator supports multidirectional (i.e., Numpy-style) broadcasting;

*/
//sds-temp,以下kernel暂时还不支持broadcasting. 比如[1,2,3] equal [2], 需要把2扩展成[2,2,2]
template<typename T>
__global__ void equal_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        T temp=0;
        if(a[index] == b[index])
        //if(fabs(a[index] - b[index]) < 1e-6)
            temp = 1;
        y[index] = temp;
    }
  }

template<typename T>
__global__ void less_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        T temp=0;
        if(a[index] < b[index])
            temp = 1;
        y[index] = temp;
    }
  }

template<typename T>
__global__ void where_kernel(const int n,T const* __restrict__ cond, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        if(cond[index])
            y[index] = a[index];
        else
            y[index] = b[index];
    }
  }

template<typename T>
__global__ void not_kernel(const int n, T const* __restrict__ a, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        T temp=0;
        if(a[index] > 0)
            temp = 1;
        y[index] = temp;
    }
  }

template<typename T>
__global__ void add_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] + b[index];
    }
  }

template<typename T>
__global__ void mul_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] * b[index];
    }
  }

template<typename T>
__global__ void pow_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] + b[index];
    }
  }




//sds:这里的inputs是在显存，outputs是在内存。
//sds:每个plugin进入enqueue，带过来的inputs有可能在内存或者显存，由addPluginV2时传入的inputs决定
//    inputs是一个指针，维度信息需要自己初始化，比如在initialize中或者enqueue中。
//    outputs? outputs已经按照getOutputDims指定的初始化，应该都是gpu指针吧?
int MyElementWisePlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  

  float * odatas = reinterpret_cast<float *>(outputs[0]);

  /*
  cudaError_t cuda_status =
    cudaMemcpyAsync(odatas, h_odatas,
                    _d_output_ptrs.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
  if( cuda_status != cudaSuccess ) {
    return 1;
  }
  */

  dim3 block(512);
  dim3 grid((_numbers + 512 - 1) / 512);
  
 switch(_type){
     
  case(MyElementWiseType::Equal): 
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    equal_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas);
    break;
  }
  case(MyElementWiseType::Less):
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    less_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas);
    break;
  }
  case(MyElementWiseType::Where):
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    float  const* idata3    = reinterpret_cast<float  const*>(inputs[2]);
    where_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2,idata3, odatas);
    break;
  }
  case(MyElementWiseType::Not):
    not_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, odatas);
    break;
  case(MyElementWiseType::Add): 
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    add_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas);
    break;
  }
  case(MyElementWiseType::Mul): 
    {
      float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
      mul_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas);
      break;
    }
  case(MyElementWiseType::Pow): 
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    pow_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas);
    break;
  }

  default:
    break;
  }
  return cudaGetLastError() != cudaSuccess;
}
