/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ShapedWeights.hpp"
#include "bfloat16.hpp"
#include "half.h"
#include <NvInfer.h>
#include <typeindex>
#include <unordered_map>

// Subset of helper functions that deal exclusively with weights to be shared across IParser and IParserRefitter classes.
// Define weightLog Macros here to ensure that an ImporterCtx class is not needed to log.

namespace onnx2trt
{

// Return the name of an ONNX data enum.
char const* getDtypeName(int32_t onnxDtype);

// Return the size in bits of an ONNX data type.
int32_t getDtypeSizeBits(int32_t onnxDtype);

// Return the size in bytes of an tensor/weights object, handle sub-byte padding.
size_t getTensorOrWeightsSizeBytes(int64_t count, int32_t onnxDtype);

// Find the corresponding ONNX data type of a built-in data type.
template <typename T>
[[nodiscard]] constexpr ShapedWeights::DataType getShapedWeightsDataType()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return ::ONNX_NAMESPACE::TensorProto::BOOL;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::INT8;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::UINT8;
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::INT16;
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::UINT16;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::INT32;
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::UINT32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::INT64;
    }
    else if constexpr (std::is_same_v<T, uint64_t>)
    {
        return ::ONNX_NAMESPACE::TensorProto::UINT64;
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return ::ONNX_NAMESPACE::TensorProto::FLOAT;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return ::ONNX_NAMESPACE::TensorProto::DOUBLE;
    }
    else if constexpr (std::is_same_v<T, half_float::half>)
    {
        return ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    }
    else if constexpr (std::is_same_v<T, BFloat16>)
    {
        return ::ONNX_NAMESPACE::TensorProto::BFLOAT16;
    }
    else
    {
        static_assert(!std::is_same_v<T, T>, "Unsupported type"); //< C++23: `static_assert(false, "Unsupported type")`
    }
}

// Return the volume of a Dims object
int64_t volume(nvinfer1::Dims const& dims);

// Normalize the slashes in a string representing a filepath.
std::string normalizePath(std::string const& path);

// Generate a unique name for a given weight or tensor name (passed as the |basename|)
std::string const& generateUniqueName(
    std::set<std::string>& namesSet, int64_t& suffixCounter, std::string const& basename);

} // namespace onnx2trt
