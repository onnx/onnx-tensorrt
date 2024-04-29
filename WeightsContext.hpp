/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "weightUtils.hpp"
#include <string>
#include <vector>

namespace onnx2trt
{

// Class reponsible for reading, casting, and converting weight values from an ONNX model and into ShapedWeights
// objects. All temporary weights are stored in a buffer owned by the class so they do not go out of scope.

class WeightsContext
{
    nvinfer1::ILogger* mLogger;
    // Vector of bytes to maintain ownership of weights.
    std::vector<std::vector<uint8_t>> mWeightBuffers;
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

    // Helper function to convert weightValues' type from fp16 to fp32.
    float* convertFP16Data(void* weightValues, nvinfer1::Dims const& shape);

    // Helper function to get fp32 representation of fp16 or fp32 weights.
    float* getFP32Values(ShapedWeights const& w);

    // Register an unique name for the created weights.
    ShapedWeights createNamedTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape,
        std::set<std::string>& namesSet, int64_t& suffixCounter, bool batchNormNode = false);

    // Create weights with a given name.
    ShapedWeights createNamedWeights(
        ShapedWeights::DataType type, nvinfer1::Dims const& shape, std::string const& name);

    // Creates a ShapedWeights object class of a given type and shape.
    ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape, uint8_t value = 0)
    {
        ShapedWeights weights(type, nullptr, shape);
        allocateTempWeights(weights);
        return weights;
    }

    // Allocate the data buffer
    void allocateTempWeights(ShapedWeights& weightsRef)
    {
        mWeightBuffers.push_back(std::vector<uint8_t>(weightsRef.size_bytes(), 0));
        weightsRef.values = mWeightBuffers.back().data();
    }

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

} // namespace onnx2trt
