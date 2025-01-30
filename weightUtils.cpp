/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "weightUtils.hpp"
#include "bfloat16.hpp"
#include "half.h"
#include <cstring> // For std::memcpy
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <typeindex>
#include <unordered_map>
#include <iterator>

namespace onnx2trt
{

char const* getDtypeName(int32_t onnxDtype)
{
    switch (onnxDtype)
    {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: return "FLOAT";
    case ::ONNX_NAMESPACE::TensorProto::UINT8: return "UINT8";
    case ::ONNX_NAMESPACE::TensorProto::INT8: return "INT8";
    case ::ONNX_NAMESPACE::TensorProto::UINT16: return "UINT16";
    case ::ONNX_NAMESPACE::TensorProto::INT16: return "INT16";
    case ::ONNX_NAMESPACE::TensorProto::INT32: return "INT32";
    case ::ONNX_NAMESPACE::TensorProto::INT64: return "INT64";
    case ::ONNX_NAMESPACE::TensorProto::STRING: return "STRING";
    case ::ONNX_NAMESPACE::TensorProto::BOOL: return "BOOL";
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return "FLOAT16";
    case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: return "BFLOAT16";
    case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return "DOUBLE";
    case ::ONNX_NAMESPACE::TensorProto::UINT32: return "UINT32";
    case ::ONNX_NAMESPACE::TensorProto::UINT64: return "UINT64";
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX64: return "COMPLEX64";
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return "COMPLEX128";
    case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1: return "FLOAT4E2M1";
    default: return "<UNKNOWN>";
    }
}

int32_t getDtypeSizeBits(int32_t onnxDtype)
{
    switch (onnxDtype)
    {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return 16;
    case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: return 16;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: return 32;
    case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return 64;
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX64: return 64;
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return 128;
    case ::ONNX_NAMESPACE::TensorProto::UINT8: return 8;
    case ::ONNX_NAMESPACE::TensorProto::INT8: return 8;
    case ::ONNX_NAMESPACE::TensorProto::UINT16: return 16;
    case ::ONNX_NAMESPACE::TensorProto::INT16: return 16;
    case ::ONNX_NAMESPACE::TensorProto::UINT32: return 32;
    // Booleans are stored in int32 tensors in ONNX
    case ::ONNX_NAMESPACE::TensorProto::BOOL: return 8;
    case ::ONNX_NAMESPACE::TensorProto::INT32: return 32;
    case ::ONNX_NAMESPACE::TensorProto::UINT64: return 64;
    case ::ONNX_NAMESPACE::TensorProto::INT64: return 64;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN: return 8;
    case ::ONNX_NAMESPACE::TensorProto::INT4: return 4;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1: return 4;
    default: return -1;
    }
}

size_t getTensorOrWeightsSizeBytes(int64_t count, int32_t onnxDtype)
{

    int32_t dTypeSize = getDtypeSizeBits(onnxDtype);

    if (dTypeSize == -1
        || static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / static_cast<size_t>(dTypeSize))
    {
        throw std::runtime_error("Size of weights exceeds maximum!");
    }

    int64_t sizeInBits = count * dTypeSize;
    if (sizeInBits % 8 != 0)
    {
        // This is a specific implementation to INT4, since this is currently the only sub-byte data type
        // we're supporting. Different data-types may have different padding.
        assert(
            onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT4 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1);
        sizeInBits += 4;
    }
    assert(sizeInBits % 8 == 0);
    return static_cast<size_t>(sizeInBits / 8);
}

int64_t volume(nvinfer1::Dims const& dims)
{
    std::for_each(
        dims.d, dims.d + dims.nbDims, [](int32_t d) { assert(d >= 0 && "volume makes no sense for dynamic shapes"); });
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

std::string normalizePath(std::string const& path)
{
    std::vector<std::string> normPath;
    auto addToPath = [&normPath](std::string s) {
        // Ignore all extra slashes, and current directory paths
        if (s == "/" || s == "./")
        {
            return;
        }
        // Push back to normPath under the following circumstances
        // 1. Current string is not "../" or
        // 2. "../" if it's the first string or
        // 3. "../" is the previous string in normPath
        if (s != "../" || normPath.empty() || (!normPath.empty() && normPath.back() == "../"))
        {
            normPath.push_back(s);
        }
        // Remove previous entry since "../" was encountered.
        else
        {
            normPath.pop_back();
        }
    };

    size_t i = 0;
    size_t n = path.size();
    std::string sep = "/";

    // Loop through path, split on all path seperator tokens, and append to normPath if applicable.
    while (i < n)
    {
        auto slashPos = path.find(sep, i);
        if (slashPos == std::string::npos)
        {
            addToPath(path.substr(i, n - i));
            break;
        }
        else
        {
            addToPath(path.substr(i, slashPos - i + 1));
            i = slashPos + 1;
        }
    }

    // Build final output string
    std::string out;
    for (auto s : normPath)
    {
        out += s;
    }
    return out;
}

std::string const& generateUniqueName(
    std::set<std::string>& namesSet, int64_t& suffixCounter, std::string const& basename)
{
    std::string candidate = basename;

    while (namesSet.find(candidate) != namesSet.end())
    {
        candidate = basename + "_" + std::to_string(suffixCounter);
        ++suffixCounter;
    }

    namesSet.insert(candidate);
    // Return reference to newly inserted string to avoid any c_str()'s going out of scope
    return *namesSet.find(candidate);
}

} // namespace onnx2trt
