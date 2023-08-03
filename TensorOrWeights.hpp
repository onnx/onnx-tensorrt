/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"

#include <NvInfer.h>
#include <cassert>

namespace onnx2trt
{

class TensorOrWeights
{
    union
    {
        nvinfer1::ITensor* _tensor;
        ShapedWeights _weights;
    };
    enum
    {
        NODE_TENSOR,
        NODE_WEIGHTS
    } _variant;

public:
    TensorOrWeights()
        : _tensor(nullptr)
        , _variant(NODE_TENSOR)
    {
    }
    TensorOrWeights(nvinfer1::ITensor* tensor)
        : _tensor(tensor)
        , _variant(NODE_TENSOR)
    {
    }
    TensorOrWeights(ShapedWeights const& weights)
        : _weights(weights)
        , _variant(NODE_WEIGHTS)
    {
    }
    bool is_tensor() const
    {
        return _variant == NODE_TENSOR;
    }
    bool is_weights() const
    {
        return _variant == NODE_WEIGHTS;
    }
    bool isNullTensor() const
    {
        return is_tensor() && _tensor == nullptr;
    }
    nvinfer1::ITensor& tensor()
    {
        assert(!isNullTensor());
        return *_tensor;
    }
    nvinfer1::ITensor const& tensor() const
    {
        assert(!isNullTensor());
        return *_tensor;
    }
    ShapedWeights& weights()
    {
        assert(is_weights());
        return _weights;
    }
    ShapedWeights const& weights() const
    {
        assert(is_weights());
        return _weights;
    }
    nvinfer1::Dims shape() const
    {
        return is_tensor() ? _tensor->getDimensions() : _weights.shape;
    }
    explicit operator bool() const
    {
        return is_tensor() ? _tensor != nullptr : static_cast<bool>(_weights);
    }
    bool isFp32() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kFLOAT
                           : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
    bool isFp16() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kHALF
                    : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    }
    bool isBFp16() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kBF16
                    : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    }
    bool isInt32() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kINT32
                           : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }
    bool isInt64() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kINT64
                           : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_INT64;
    }
    bool isInt8() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kINT8
                           : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }
    bool isBool() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kBOOL : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    }
    bool isFp8() const
    {
        return is_tensor() ? _tensor->getType() == nvinfer1::DataType::kFP8 : _weights.type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN;
    }
    std::string getName() const
    {
        return is_tensor() ? _tensor->getName() : _weights.getName();
    }
    std::string getType() const
    {
        if (is_tensor())
        {
            switch (_tensor->getType())
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
            }
        }
        else
        {
            switch (_weights.type)
            {
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
            }
        }
        return "UNKNOWN TYPE";
    }

    nvinfer1::DataType convertONNXDataType(ShapedWeights::DataType datatype) const
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
        }
        assert(false && "Unknown datatype");
        return nvinfer1::DataType::kFLOAT;
    }

    ShapedWeights::DataType convertTRTDataType(nvinfer1::DataType datatype) const
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
        }
        assert(false && "Unknown datatype");
        return ::ONNX_NAMESPACE::TensorProto::FLOAT;
    }

    nvinfer1::DataType getDataType() const
    {
        if (is_tensor()) return _tensor->getType();
        else return convertONNXDataType(_weights.type);
    }
    
    ShapedWeights::DataType getONNXDataType() const
    {
        if (is_tensor()) return convertTRTDataType(_tensor->getType());
        else return _weights.type;
    }
};

} // namespace onnx2trt
