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

    static ShapedWeights empty(DataType type);

    ShapedWeights();

    explicit ShapedWeights(DataType type, void* values, nvinfer1::Dims shape_);

    size_t count() const;

    size_t size_bytes() const;

    const char* getName() const;

    void setName(const char* name);

    explicit operator bool() const;

    operator nvinfer1::Weights() const;

    template <typename T>
    T& at(size_t index)
    {
        assert(index >= 0 && (index * sizeof(T)) < size_bytes());
        return static_cast<T*>(values)[index];
    }

    template <typename T>
    const T& at(size_t index) const
    {
        assert(index >= 0 && (index * sizeof(T)) < size_bytes());
        return static_cast<const T*>(values)[index];
    }

public:
    DataType type;
    void* values;
    nvinfer1::Dims shape;
    const char* name{};
};

class IImporterContext;
bool transposeWeights(ShapedWeights const& weights, nvinfer1::Permutation const& perm, ShapedWeights* result, IImporterContext* ctx);

} // namespace onnx2trt
