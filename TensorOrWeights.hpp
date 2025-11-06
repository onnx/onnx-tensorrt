/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include <NvInfer.h>
#include <cassert>
#include <stdexcept>

namespace onnx2trt
{

//! Abstract representation of a tensor, which might be a nvinfer1::ITensor or ShapedWeights.
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
    //! Represents "null tensor", which is used to denote "missing tensor".
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
        if (is_weights() || isNullTensor())
        {
            throw std::runtime_error("Trying to access weights or a null tensor!");
        }
        return *_tensor;
    }
    nvinfer1::ITensor const& tensor() const
    {
        if (is_weights() || isNullTensor())
        {
            throw std::runtime_error("Trying to access weights or a null tensor!");
        }
        return *_tensor;
    }
    ShapedWeights& weights()
    {
        if (is_tensor())
        {
            throw std::runtime_error("Trying to access a null weights!");
        }
        return _weights;
    }
    ShapedWeights const& weights() const
    {
        if (is_tensor())
        {
            throw std::runtime_error("Trying to access a null weights!");
        }
        return _weights;
    }
    nvinfer1::Dims shape() const
    {
        return is_tensor() ? tensor().getDimensions() : weights().shape;
    }
    explicit operator bool() const
    {
        return is_tensor() ? _tensor != nullptr : static_cast<bool>(_weights);
    }
    bool isFp32() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kFLOAT
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
    bool isFp16() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kHALF
                    : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    }
    bool isBFp16() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kBF16
                    : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    }
    bool isInt32() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kINT32
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }
    bool isInt64() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kINT64
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_INT64;
    }
    bool isInt8() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kINT8
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }
    bool isUint8() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kUINT8
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    }
    bool isInt4() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kINT4
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_INT4;
    }
    bool isBool() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kBOOL
                           : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    }
    bool isFp8() const
    {
        return is_tensor() ? tensor().getType() == nvinfer1::DataType::kFP8 : weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN;
    }
    void setName(char const* name)
    {
        if (is_tensor())
        {
            tensor().setName(name);
        }
        else
        {
            weights().setName(name);
        }
    }
    std::string getName() const
    {
        return is_tensor() ? tensor().getName() : weights().getName();
    }
    std::string getType() const;

    nvinfer1::DataType convertONNXDataType(ShapedWeights::DataType datatype) const;

    ShapedWeights::DataType convertTRTDataType(nvinfer1::DataType datatype) const;

    nvinfer1::DataType getDataType() const
    {
        if (is_tensor())
        {
            return tensor().getType();
        }
        else
        {
            return convertONNXDataType(weights().type);
        }
    }

    ShapedWeights::DataType getONNXDataType() const
    {
        if (is_tensor())
        {
            return convertTRTDataType(tensor().getType());
        }
        else
        {
            return weights().type;
        }
    }
};

} // namespace onnx2trt
