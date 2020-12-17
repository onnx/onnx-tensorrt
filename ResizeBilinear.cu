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

#include <cassert>
#include <algorithm>
#include <cuda_fp16.h>
#include "ResizeBilinear.hpp"

//==================================================================

// TODO: Move this to a common header
inline bool 
is_CHW(
    nvinfer1::Dims const& dims) 
{
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

//==================================================================

nvinfer1::Dims 
ResizeBilinearPlugin::getOutputDimensions(
    int index,
    const nvinfer1::Dims *inputDims,
    int nbInputs) 
{
    assert(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    assert(is_CHW(input));
    assert(_ndims == 2);
    assert(index == 0);
    nvinfer1::Dims output = {0};
    output.nbDims = input.nbDims;
    
    output.d[0] = input.d[0];
    output.d[1] = input.d[1] * _scale[0];
    output.d[2] = input.d[2] * _scale[1];

    output.type[0] = input.type[0];
    output.type[1] = input.type[1];
    output.type[2] = input.type[2];

    return output;
}

//==================================================================

int 
ResizeBilinearPlugin::initialize() 
{
    _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
    assert(is_CHW(this->getInputDims(0)));
    assert(is_CHW(_output_dims));
    assert(_ndims == 2);
    return 0;
}

//==================================================================

template <typename Data> __global__
void 
resize_bilinear_kernel_2d(
    int nbatch,
    float2 scale,
    int2 osize,
    Data const* idata, int istride, int ibatchstride,
    Data*       odata, int ostride, int obatchstride) 
{
    const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int z0 = blockIdx.z;
    const int src_cols = int(osize.x / scale.x);
    const int src_rows = int(osize.y / scale.y);

    for( int batch=z0; batch<nbatch; batch+=gridDim.z ) 
    {
        for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) 
        {
            for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) 
            {
                const float src_x = float(ox) / float(scale.x);
                const float src_y = float(oy) / float(scale.y);
                const int x1 = int(src_x);
                const int y1 = int(src_y);
                const int x2 = x1 + 1;
                const int y2 = y1 + 1;
                const float x1f = float(x1);
                const float y1f = float(y1);
                const float x2f = float(x2);
                const float y2f = float(y2);
                const int x2_read = ::min(x2, src_cols - 1);
                const int y2_read = ::min(y2, src_rows - 1);
                const int index_output = batch * obatchstride + oy * ostride + ox;

                odata[index_output] = 
                    float(idata[batch * ibatchstride + y1 * istride + x1]) *           float((x2f - src_x) * (y2f - src_y)) +
                    float(idata[batch * ibatchstride + y1 * istride + x2_read]) *      float((src_x - x1f) * (y2f - src_y)) + 
                    float(idata[batch * ibatchstride + y2_read * istride + x1]) *      float((x2f - src_x) * (src_y - y1f)) + 
                    float(idata[batch * ibatchstride + y2_read * istride + x2_read]) * float((src_x - x1f) * (src_y - y1f));
            }
        }
    }
}

//==================================================================

int
ResizeBilinearPlugin::enqueue(
    int batchSize,
    const void *const *inputs, void **outputs,
    void *workspace, cudaStream_t stream)
{
    auto const& input_dims = this->getInputDims(0);
    int nchan = input_dims.d[0];
    
    if (_ndims != 2)
    {
        return -1;
    }

    const float2 scale = {_scale[1], _scale[0]};
    const int2 osize = {_output_dims.d[2], _output_dims.d[1]};
    const int istride =   input_dims.d[2];
    const int ostride = _output_dims.d[2];
    const int ibatchstride =   input_dims.d[1] * istride;
    const int obatchstride = _output_dims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1,
        (osize.y - 1) / block.y + 1,
        std::min(batchSize * nchan, 65535));

    if (getDataType() == nvinfer1::DataType::kFLOAT) 
    {
        resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>(
            batchSize * nchan, scale, osize,
            static_cast<float const*>( inputs[0]), istride, ibatchstride,
            static_cast<float*>(outputs[0]), ostride, obatchstride);
    } 
    else 
    {
        resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>(
            batchSize * nchan, scale, osize,
            static_cast<__half const*>( inputs[0]), istride, ibatchstride,
            static_cast<__half*      >(outputs[0]), ostride, obatchstride);
    }

    return cudaGetLastError() != cudaSuccess;
}

//==================================================================
