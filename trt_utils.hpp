/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Status.hpp"
#include "TensorOrWeights.hpp"
#include "onnx2trt.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace onnx2trt
{

inline int getDtypeSize(nvinfer1::DataType trtDtype)
{
    switch (trtDtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    // TRT does not support booleans as a native type, so we treat them like int32 values.
    case nvinfer1::DataType::kBOOL:
        return 4;
    // TODO: Some sort of error handling
    default: return -1;
    }
}

inline nvinfer1::Dims insert_dim(nvinfer1::Dims const& dims, int idx, int value)
{
    assert(idx < dims.nbDims + 1);
    nvinfer1::Dims new_dims;
    new_dims.nbDims = dims.nbDims + 1;
    for (int i = 0; i < idx; ++i)
    {
        new_dims.d[i] = dims.d[i];
    }
    new_dims.d[idx] = value;
    for (int i = idx + 1; i < new_dims.nbDims; ++i)
    {
        new_dims.d[i] = dims.d[i - 1];
    }
    return new_dims;
}

inline nvinfer1::Dims remove_dim(nvinfer1::Dims const& dims, int idx)
{
    assert(idx < dims.nbDims);
    nvinfer1::Dims new_dims;
    new_dims.nbDims = dims.nbDims - 1;
    for (int i = 0; i < idx; ++i)
    {
        new_dims.d[i] = dims.d[i];
    }
    for (int i = idx; i < new_dims.nbDims; ++i)
    {
        new_dims.d[i] = dims.d[i + 1];
    }
    // Special case for scalar result (i.e., there was only one dim originally)
    if (new_dims.nbDims == 0)
    {
        new_dims.nbDims = 1;
        new_dims.d[0] = 1;
    }
    return new_dims;
}

// Adds unitary dimensions on the left
inline nvinfer1::Dims expand_dims(nvinfer1::Dims const& dims, int ndim_new)
{
    assert(dims.nbDims <= ndim_new);
    nvinfer1::Dims new_dims;
    new_dims.nbDims = ndim_new;
    int j = 0;
    for (; j < ndim_new - dims.nbDims; ++j)
    {
        new_dims.d[j] = 1;
    }
    for (int i = 0; i < dims.nbDims; ++i, ++j)
    {
        new_dims.d[j] = dims.d[i];
    }
    return new_dims;
}

inline nvinfer1::Permutation remove_first_dim(nvinfer1::Permutation const& perm)
{
    assert(perm.order[0] == 0);
    nvinfer1::Permutation new_perm;
    int ndim = nvinfer1::Dims::MAX_DIMS;
    for (int i = 0; i < ndim - 1; ++i)
    {
        new_perm.order[i] = perm.order[i + 1] - 1;
    }
    return new_perm;
}

inline nvinfer1::DimsHW operator-(nvinfer1::DimsHW dims)
{
    return nvinfer1::DimsHW(-dims.h(), -dims.w());
}

// Note: These are used for checking beg_padding == end_padding
inline bool operator==(nvinfer1::Dims const& a, nvinfer1::Dims const& b)
{
    if (a.nbDims != b.nbDims)
    {
        return false;
    }
    for (int i = 0; i < a.nbDims; ++i)
    {
        if (a.d[i] != b.d[i])
        {
            return false;
        }
    }
    return true;
}
inline bool operator!=(nvinfer1::Dims const& a, nvinfer1::Dims const& b)
{
    return !(a == b);
}

inline TensorOrWeights identity(IImporterContext* ctx, TensorOrWeights input)
{
    if (input.is_weights())
    {
        return input;
    }
    else
    {
        auto* layer = ctx->network()->addIdentity(input.tensor());
        if (!layer)
        {
            return nullptr;
        }
        return layer->getOutput(0);
    }
}

inline ::ONNX_NAMESPACE::TensorProto_DataType trtDataTypeToONNX(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT: return ::ONNX_NAMESPACE::TensorProto::FLOAT;
    case nvinfer1::DataType::kHALF: return ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    case nvinfer1::DataType::kINT32: return ::ONNX_NAMESPACE::TensorProto::INT32;
    case nvinfer1::DataType::kINT8: return ::ONNX_NAMESPACE::TensorProto::INT8;
    case nvinfer1::DataType::kBOOL: return ::ONNX_NAMESPACE::TensorProto::BOOL;
    case nvinfer1::DataType::kUINT8: return ::ONNX_NAMESPACE::TensorProto::UINT8;
    default: return ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
    }
    throw std::runtime_error{"Unreachable"};
}

} // namespace onnx2trt
