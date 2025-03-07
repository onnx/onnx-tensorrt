/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "errorHelpers.hpp"
#include "weightUtils.hpp"
#include <string>
#include <vector>

namespace onnx2trt
{

// Class reponsible for reading, casting, and converting weight values from an ONNX model and into ShapedWeights
// objects. All temporary weights are stored in a buffer owned by the class so they do not go out of scope.

class WeightsContext
{
    struct BufferDeleter
    {
        void operator()(void* ptr)
        {
            operator delete(ptr);
        }
    };

    using BufferPtr = std::unique_ptr<void, BufferDeleter>;

    nvinfer1::ILogger* mLogger;

    // Vector of hunks to maintain ownership of weights.
    std::vector<BufferPtr> mWeightBuffers;

    // Keeps track of the absolute location of the file in order to read external weights.
    std::string mOnnxFileLocation;

public:
    WeightsContext(nvinfer1::ILogger* logger)
        : mLogger(logger){};

    int32_t* convertUINT8(uint8_t const* weightValues, nvinfer1::Dims const& shape);

    float* convertDouble(double const* weightValues, nvinfer1::Dims const& shape);

    template <typename DataType>
    DataType* convertInt32Data(int32_t const* weightValues, nvinfer1::Dims const& shape, int32_t onnxdtype);

    uint8_t* convertPackedInt32Data(
        int32_t const* weightValues, nvinfer1::Dims const& shape, size_t nbytes, int32_t onnxdtype);

    // Function to create an internal buffer to own the weights without any type conversions.
    void* ownWeights(void const* weightValues, ShapedWeights::DataType const dataType, nvinfer1::Dims const& shape,
        size_t const nBytes);

    // Function to read bytes from an external file and return the data in a buffer.
    bool parseExternalWeights(

        std::string const& file, int64_t offset, int64_t length, std::vector<char>& weightsBuf, size_t& size);

    // Function to read data from an ONNX Tensor and move it into a ShapedWeights object.
    // Handles external weights as well.
    bool convertOnnxWeights(
        ::ONNX_NAMESPACE::TensorProto const& onnxTensor, ShapedWeights* weights, bool ownAllWeights = false);

    // Helper function to convert weightValues' type from fp16/bf16 to fp32.
    template <typename DataType>
    [[nodiscard]] float* convertToFp32(ShapedWeights const& w);

    // Helper function to get fp32 representation of fp16, bf16, or fp32 weights.
    float* getFP32Values(ShapedWeights const& w);

    // Register an unique name for the created weights.
    ShapedWeights createNamedTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape,
        std::set<std::string>& namesSet, int64_t& suffixCounter, bool batchNormNode = false);

    // Create weights with a given name.
    ShapedWeights createNamedWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape, std::string const& name,
        std::set<std::string>* bufferedNames = nullptr);

    // Creates a ShapedWeights object class of a given type and shape.
    ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape);

    // Sets the absolute filepath of the loaded ONNX model in order to read external weights.
    void setOnnxFileLocation(std::string location)
    {
        mOnnxFileLocation = location;
    }

    // Returns the absolutate filepath of the loaded ONNX model.
    std::string getOnnxFileLocation()
    {
        return mOnnxFileLocation;
    }

    // Returns the logger object.
    nvinfer1::ILogger& logger()
    {
        return *mLogger;
    }
};

template <typename DataType>
DataType* WeightsContext::convertInt32Data(int32_t const* weightValues, nvinfer1::Dims const& shape, int32_t onnxdtype)
{
    size_t const nbWeights = volume(shape);
    DataType* newWeights{static_cast<DataType*>(createTempWeights(onnxdtype, shape).values)};

    for (size_t i = 0; i < nbWeights; i++)
    {
        newWeights[i] = static_cast<DataType>(weightValues[i]);
    }
    return newWeights;
}
template <typename DataType>
[[nodiscard]] float* WeightsContext::convertToFp32(ShapedWeights const& w)
{
    int64_t const nbWeights = volume(w.shape);
    auto result = static_cast<float*>(createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, w.shape).values);
    std::copy_n(static_cast<DataType const*>(w.values), nbWeights, result);

    return result;
}

} // namespace onnx2trt
