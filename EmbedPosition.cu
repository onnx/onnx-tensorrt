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
#include <cub/cub.cuh>
#include "EmbedPosition.hpp"

//sds:对于这个split来说，这里的index必须是0，表示只有一个输出
//sds: index,The index of the output tensor.
//sds,仅有此函数是创建引擎时执行。不要依赖其他函数
nvinfer1::Dims EmbedPositionPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  // 'equal' and 'less' is 2, 'where' is 3
  assert(nbInputs == 1);
  assert(index == 0);// only one output
  nvinfer1::Dims const& input_dims = inputDims[0];
  
  return input_dims;
}

//sds,此函数是运行时执行.创建引擎时不会执行。
int EmbedPositionPlugin::initialize() {
  nvinfer1::Dims dims = this->getInputDims(0);
 
  sentence_num = dims.d[1];
  batch_size = dims.d[0];
  return 0;
}


template <typename T, unsigned TPB>
__global__ void embed_position_kernel(T const* __restrict__ a, T * __restrict__ y) {
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	 using BlockReduce = cub::BlockReduce<float, TPB>;

    __shared__ typename BlockReduce::TempStorage tmpStorage;
    __shared__ float sum_temp;
    cub::Sum sum;
    float threadData(0);

    threadData = a[x_index];

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);
	if (threadIdx.x == 0)
    {
        sum_temp = Z;
    }
	
	if(fabs(threadData - 0.0) <= 1e-6) 
	{
	    y[x_index] = threadIdx.x - sum_temp +2;
    }
	else
	{
	    y[x_index] = 1;
    }
  }

//sds-temp,仅支持特例 : 所有element 都是1. output dim在初始化引擎时确定. 此情况下输出满足特定规律。
int EmbedPositionPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  float * odatas = reinterpret_cast<float *>(outputs[0]);

  //dims(x,y,z)
  dim3 block(sentence_num);
  dim3 grid(batch_size);
  extern __shared__ int lensOfDim[];
  
  const int snum_ = 128;
  assert( snum_ == sentence_num);
  embed_position_kernel<float, snum_><<<grid, block, 0, stream>>>(idata1, odatas);

  gdb_copy_to_cpu("EmbedPosition output", odatas, sentence_num* batch_size);

  return cudaGetLastError() != cudaSuccess;
}
