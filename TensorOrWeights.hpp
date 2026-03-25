/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include <NvInfer.h>
#include <cassert>
#include <stdexcept>
#include <variant>

namespace onnx2trt
{

//! Abstract representation of a tensor, which might be a nvinfer1::ITensor or ShapedWeights.
//! Essentially a `std::variant<nvinfer1::ITensor*, ShapedWeights>`, but really more like a
//! `std::variant<std::monostate, nvinfer1::ITensor&, ShapedWeights>` in that it treats the null-`ITensor` state as
//! special.
class TensorOrWeights
{
    using ITensor = nvinfer1::ITensor;
    using VariantType = std::variant<ITensor*, ShapedWeights>;
    VariantType mVariant{nullptr};

public:
    //! Represents "null tensor", which is used to denote "missing tensor".
    TensorOrWeights() = default;
    TensorOrWeights(nvinfer1::ITensor* tensor)
        : mVariant(tensor)
    {
    }
    TensorOrWeights(ShapedWeights const& weights)
        : mVariant(weights)
    {
    }
    bool is_tensor() const
    {
        return !this->is_weights();
    }
    bool is_weights() const
    {
        return std::holds_alternative<ShapedWeights>(mVariant);
    }
    bool isNullTensor() const
    {
        ITensor* const* ptrPtr = std::get_if<ITensor*>(&mVariant);
        return ptrPtr != nullptr && *ptrPtr == nullptr;
    }
    nvinfer1::ITensor& tensor()
    {
        if (ITensor** ptrPtr = std::get_if<ITensor*>(&mVariant); ptrPtr != nullptr && *ptrPtr != nullptr)
        {
            return **ptrPtr;
        }
        throw std::runtime_error("Trying to access weights or a null tensor!");
    }
    nvinfer1::ITensor const& tensor() const
    {
        return const_cast<TensorOrWeights&>(*this).tensor();
    }
    ShapedWeights& weights()
    {
        if (auto* ptr = std::get_if<ShapedWeights>(&mVariant))
        {
            return *ptr;
        }
        throw std::runtime_error("Trying to access a null weights!");
    }
    ShapedWeights const& weights() const
    {
        return const_cast<TensorOrWeights&>(*this).weights();
    }
    nvinfer1::Dims shape() const
    {
        return is_tensor() ? tensor().getDimensions() : weights().shape;
    }
    explicit operator bool() const
    {
        return std::visit([](auto&& ptrOrWeights) { return static_cast<bool>(ptrOrWeights); }, mVariant);
    }
    bool isFp32() const
    {
        return this->getDataType() == nvinfer1::DataType::kFLOAT;
    }
    bool isFp16() const
    {
        return this->getDataType() == nvinfer1::DataType::kHALF;
    }
    bool isBFp16() const
    {
        return this->getDataType() == nvinfer1::DataType::kBF16;
    }
    bool isInt32() const
    {
        return this->getDataType() == nvinfer1::DataType::kINT32;
    }
    bool isInt64() const
    {
        return this->getDataType() == nvinfer1::DataType::kINT64;
    }
    bool isInt8() const
    {
        return this->getDataType() == nvinfer1::DataType::kINT8;
    }
    bool isUint8() const
    {
        return this->getDataType() == nvinfer1::DataType::kUINT8;
    }
    bool isInt4() const
    {
        return this->getDataType() == nvinfer1::DataType::kINT4;
    }
    bool isBool() const
    {
        return this->getDataType() == nvinfer1::DataType::kBOOL;
    }
    bool isFp8() const
    {
        return this->getDataType() == nvinfer1::DataType::kFP8;
    }
    //! Set the name. The caller must keep the buffer alive.
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

    //! \return the type as a string, e.g. "FLOAT", "HALF", "BF16", "INT8", "UINT8", "INT32", "INT64", "BOOL", "FP8",
    //! "INT4", "FP4", "E8M0".
    std::string getType() const;

    static nvinfer1::DataType convertONNXDataType(ShapedWeights::DataType datatype);

    //! Convert a TensorRT data type to an ONNX data type.
    //! \note DataType::kE8M0 is not supported and throws.
    static ShapedWeights::DataType convertTRTDataType(nvinfer1::DataType datatype);

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
