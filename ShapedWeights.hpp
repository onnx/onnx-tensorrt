/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <onnx/onnx_pb.h>

namespace onnx2trt
{

class ShapedWeights
{
public:
    using DataType = int32_t;
    DataType type;
    void* values;
    nvinfer1::Dims shape;
    const char* name = nullptr;
    static ShapedWeights empty(DataType type);
    ShapedWeights();
    explicit ShapedWeights(DataType type, void* values, nvinfer1::Dims shape_);
    size_t count() const;
    size_t size_bytes() const;
    const char* getName() const;
    void setName(const char* name);
    explicit operator bool() const;
    operator nvinfer1::Weights() const;
};

bool transposeWeights(ShapedWeights const& weights, nvinfer1::Permutation const& perm, ShapedWeights* result);

} // namespace onnx2trt
