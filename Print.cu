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

#include "Print.hpp"

//sds:输出其实没啥用
//sds-temp,多少个输出在哪确认的?
nvinfer1::Dims PrintPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  
  assert(index == 0);// only one output
  // input and shape
  
  return inputDims[0];
}

int PrintPlugin::initialize() {
  nvinfer1::Dims const& input_dims1 =this->getInputDims(0);
  
  return 0;
}

//sds-temp,当前仅测试了一种场景，1*128扩展到10*128
int PrintPlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {


  float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  

  float * odatas = reinterpret_cast<float *>(outputs[0]);

  nvinfer1::Dims const& input_dims1 = this->getInputDims(0);
  int number_ = 1;
  for(int i = 0 ; i <input_dims1.nbDims;i++)
  {
       number_ *= input_dims1.d[i];
  }

  
  cudaError_t status = cudaMemcpy(odatas, idata1, sizeof(float)*number_, cudaMemcpyDeviceToDevice);
  
   if( status != cudaSuccess ) { 
      printf("PrintPlugin enqueue cuda error! \n");
      return false;
   } 

   

  //int num= this->getInputDimsNum();
  //for(int i = 0 ; i< num; i++)
  {    
      
      gdb_copy_to_cpu(std::to_string(this->names[0]), (float *)idata1, number_);
  }
  return cudaGetLastError() != cudaSuccess;
}
