/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TensorOrWeights.hpp"
#include <cassert>

namespace onnx2trt
{

std::string TensorOrWeights::getType() const
{
    if (is_tensor())
    {
        switch (tensor().getType())
        {
        case nvinfer1::DataType::kFLOAT: return "FLOAT";
        case nvinfer1::DataType::kHALF: return "HALF";
        case nvinfer1::DataType::kBF16: return "BF16";
        case nvinfer1::DataType::kINT8: return "INT8";
        case nvinfer1::DataType::kUINT8: return "UINT8";
        case nvinfer1::DataType::kINT32: return "INT32";
        case nvinfer1::DataType::kINT64: return "INT64";
        case nvinfer1::DataType::kBOOL: return "BOOL";
        case nvinfer1::DataType::kFP8: return "FP8";
        case nvinfer1::DataType::kINT4: return "INT4";
        case nvinfer1::DataType::kFP4: return "FP4";
        case nvinfer1::DataType::kE8M0: return "E8M0";
        }
    }
    else
    {
        switch (weights().type)
        {
        // Demote double to float.
        case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return "FLOAT";
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: return "FLOAT";
        case ::ONNX_NAMESPACE::TensorProto::INT8: return "INT8";
        case ::ONNX_NAMESPACE::TensorProto::UINT8: return "UINT8";
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return "HALF";
        case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: return "BF16";
        case ::ONNX_NAMESPACE::TensorProto::BOOL: return "BOOL";
        case ::ONNX_NAMESPACE::TensorProto::INT32: return "INT32";
        case ::ONNX_NAMESPACE::TensorProto::INT64: return "INT64";
        case ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN: return "FP8";
        case ::ONNX_NAMESPACE::TensorProto::INT4: return "INT4";
        case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1: return "FP4";
        }
    }
    return "UNKNOWN TYPE";
}

nvinfer1::DataType TensorOrWeights::convertONNXDataType(ShapedWeights::DataType datatype) const
{
    switch (datatype)
    {
        case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return nvinfer1::DataType::kFLOAT;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: return nvinfer1::DataType::kFLOAT;
        case ::ONNX_NAMESPACE::TensorProto::INT8: return nvinfer1::DataType::kINT8;
        case ::ONNX_NAMESPACE::TensorProto::UINT8: return nvinfer1::DataType::kUINT8;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return nvinfer1::DataType::kHALF;
        case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: return nvinfer1::DataType::kBF16;
        case ::ONNX_NAMESPACE::TensorProto::BOOL: return nvinfer1::DataType::kBOOL;
        case ::ONNX_NAMESPACE::TensorProto::INT32: return nvinfer1::DataType::kINT32;
        case ::ONNX_NAMESPACE::TensorProto::INT64: return nvinfer1::DataType::kINT64;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN: return nvinfer1::DataType::kFP8;
        case ::ONNX_NAMESPACE::TensorProto::INT4: return nvinfer1::DataType::kINT4;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1: return nvinfer1::DataType::kFP4;
        }
        assert(false && "Unknown datatype");
        return nvinfer1::DataType::kFLOAT;
}

ShapedWeights::DataType TensorOrWeights::convertTRTDataType(nvinfer1::DataType datatype) const
{
    switch (datatype)
    {
        case nvinfer1::DataType::kFLOAT: return ::ONNX_NAMESPACE::TensorProto::FLOAT;
        case nvinfer1::DataType::kINT8: return ::ONNX_NAMESPACE::TensorProto::INT8;
        case nvinfer1::DataType::kUINT8: return ::ONNX_NAMESPACE::TensorProto::UINT8;
        case nvinfer1::DataType::kHALF: return ::ONNX_NAMESPACE::TensorProto::FLOAT16;
        case nvinfer1::DataType::kBF16: return ::ONNX_NAMESPACE::TensorProto::BFLOAT16;
        case nvinfer1::DataType::kBOOL: return ::ONNX_NAMESPACE::TensorProto::BOOL;
        case nvinfer1::DataType::kINT32: return ::ONNX_NAMESPACE::TensorProto::INT32;
        case nvinfer1::DataType::kINT64: return ::ONNX_NAMESPACE::TensorProto::INT64;
        case nvinfer1::DataType::kFP8: return ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN;
        case nvinfer1::DataType::kINT4: return ::ONNX_NAMESPACE::TensorProto::INT4;
        case nvinfer1::DataType::kFP4: return ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1;
        case nvinfer1::DataType::kE8M0: break;
        }
        assert(false && "Unknown datatype");
        return ::ONNX_NAMESPACE::TensorProto::FLOAT;
}

} // namespace onnx2trt
