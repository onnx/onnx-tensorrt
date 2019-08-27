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

#include "MyCast.hpp"



template <typename SrcT, typename DstT>
__global__ void Cast(const SrcT* input, int num_elements, DstT* output) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (num_elements); i += blockDim.x * gridDim.x)
  {
    output[i] = static_cast<DstT>(input[i]);
  }
}

template <typename SrcT, typename DstT>
void RunCast(const SrcT* d_input, int num_elements, DstT* d_output,
             cudaStream_t stream) {


  dim3 block(512);
  dim3 grid((num_elements + 512 - 1) / 512);
  //TF_CHECK_OK(cudaLaunchKernel(Cast<SrcT, DstT>, threads_per_block,
            //                   blocks_per_grid, 0, stream, d_input,
                //               num_elements, d_output));
  Cast<<<grid, block, 0, stream>>>(d_input, num_elements, d_output);

}



nvinfer1::Dims MyCastPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index == 0);
  return inputDims[0];
}


void MyCastPlugin::configurePlugin(const nvinfer1::Dims* input_dims, int num_inputs,
                       const nvinfer1::Dims* output_dims, int num_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format, int max_batch_size) {
    assert(1 == num_inputs);
    assert(1 == num_outputs);
    assert(src_type_ == input_types[0]);
    assert(dst_type_ == output_types[0]);

    src_dims_ = input_dims[0];
  }



nvinfer1::DataType MyCastPlugin::getOutputDataType(int index, const nvinfer1::DataType* input_types,int num_inputs) const{

    assert(num_inputs == 1);
    assert(index == 0);
    return dst_type_;
  }

bool MyCastPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const{
    return type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32;
  }
  

int MyCastPlugin::enqueue(int batch_size, const void* const* inputs, void** outputs, void*,
              cudaStream_t stream){
    int num_elements = batch_size;
    for (int i = 0; i < src_dims_.nbDims; i++) {
      num_elements *= src_dims_.d[i];
    }
    const void* input = inputs[0];
    void* output = outputs[0];
    //DCHECK_NE(static_cast<int>(src_type_), static_cast<int>(dst_type_));

    switch (src_type_) {
      case nvinfer1::DataType::kFLOAT:
        RunCast(reinterpret_cast<const float*>(input), num_elements,
                reinterpret_cast<int32_t*>(output), stream);
        break;
      case nvinfer1::DataType::kINT32:
        RunCast(reinterpret_cast<const int32_t*>(input), num_elements,
                reinterpret_cast<float*>(output), stream);
        break;
      default:
        return 1;  // Indicates a failure.
    }
    return 0;
  }
