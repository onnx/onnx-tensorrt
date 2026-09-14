/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "errorHelpers.hpp"
#include "weightUtils.hpp"
#include <map>
#include <span>
#include <string>
#include <vector>

namespace onnx2trt
{
using FileHandle =
#ifdef _WIN32
    void*;
#else
    int;
#endif

// Class responsible for reading, casting, and converting weight values from an ONNX model and into ShapedWeights
// objects. All temporary weights are stored in a buffer owned by the class so they do not go out of scope.

class WeightsContext
{
public:
    //! A memory-mapped external weights file: a pointer to the mapping and its size in bytes.
    struct MemoryMapping
    {
        void* data{nullptr};
        int64_t size{0};
    };

private:
    nvinfer1::ILogger* mLogger{};

    // Vector of chunks to maintain ownership of weights.
    std::vector<std::unique_ptr<std::byte[]>> mWeightBuffers;

    // Keeps track of the absolute location of the file in order to read external weights.
    std::string mOnnxFileLocation;

    std::map<std::string, FileHandle> mMappedFiles;
#ifdef _WIN32
    std::map<std::string, FileHandle> mFileMappingHandles;
#endif
    std::map<std::string, MemoryMapping> mMemoryMappings;

    template <typename T>
    using StringMap = std::unordered_map<std::string, T>;

    StringMap<::ONNX_NAMESPACE::TensorProto const*> mInitializers;

    StringMap<std::pair<void const*, size_t>> mExternalInits;

public:
    explicit WeightsContext(nvinfer1::ILogger* logger)
        : mLogger(logger)
    {
    }

    ~WeightsContext();

    WeightsContext(WeightsContext const& other) = delete;
    WeightsContext& operator=(WeightsContext const& other) = delete;
    WeightsContext(WeightsContext&& other) = delete;
    WeightsContext& operator=(WeightsContext&& other) = delete;

    int32_t* convertUINT8(uint8_t const* weightValues, nvinfer1::Dims const& shape);

    //! Convert the DOUBLE weights in \p weightValues to FLOAT, clamping values outside the FLOAT range.
    //! The result is backed by an internal buffer owned by this WeightsContext (see createTempWeights) and
    //! remains valid for the context's lifetime.
    //! \return A span over the converted FLOAT weights, or an empty span when \p weightValues holds fewer
    //! than volume(shape) elements (which would otherwise be read out of bounds).
    [[nodiscard]] std::span<float> convertDouble(std::span<double const> weightValues, nvinfer1::Dims const& shape);

    //! Numerically cast the ONNX int32-backed \p weightValues to the destination type T, sized by \p shape.
    //! \return A pointer to the converted weights, or nullptr when \p weightValues holds fewer than
    //! volume(shape) elements (which would otherwise be read out of bounds).
    template <typename T>
    T* convertInt32Data(std::span<int32_t const> weightValues, nvinfer1::Dims const& shape, int32_t onnxdtype);

    uint8_t* convertPackedInt32Data(
        int32_t const* weightValues, nvinfer1::Dims const& shape, size_t nbytes, int32_t onnxdtype);

    //! Create an internal buffer that takes ownership of \p weightValues without any type conversion.
    void* ownWeights(void const* weightValues, ShapedWeights::DataType const dataType, nvinfer1::Dims const& shape,
        size_t const nBytes);

    //! Map \p file, then point \p weightsRef at the [offset, offset + length) window within it. Rejects
    //! an offset/length that fall outside the mapped file. A length of 0 spans to the end of the file.
    bool parseExternalWeights(std::string const& file, int64_t offset, int64_t length, MemoryMapping& weightsRef);

    //! Import an initializer whose data lives in an external file into \p weights, validating the declared
    //! offset/length against the mapped file and the mapped size against \p shape before any conversion.
    //! \return false on any inconsistency.
    bool importExternalWeights(
        ::ONNX_NAMESPACE::TensorProto const& onnxTensor, nvinfer1::Dims const& shape, ShapedWeights* weights);
    // Function to read data from an ONNX Tensor and move it into a ShapedWeights object.
    // Handles external weights as well.
    bool convertOnnxWeights(
        ::ONNX_NAMESPACE::TensorProto const& onnxTensor, ShapedWeights* weights, bool ownAllWeights = false);

    //! Convert the fp16/bf16 weights in \p w to a newly allocated fp32 buffer. The template parameter is the
    //! source element type (half or BFloat16); the result is always fp32, backed by an internal buffer owned
    //! by this WeightsContext.
    //! \return A pointer to the fp32 weights.
    template <typename T>
    [[nodiscard]] float* convertToFp32(ShapedWeights const& w);

    // Helper function to get fp32 representation of fp16, bf16, or fp32 weights.
    float* getFP32Values(ShapedWeights const& w);

    // Register an unique name for the created weights. If the weights are expected to be refittable by the IParserRefitter,
    // use a different identifier.
    ShapedWeights createNamedTempWeights(ShapedWeights::DataType type, nvinfer1::Dims const& shape,
        std::set<std::string>& namesSet, int64_t& suffixCounter, bool refittable = false);

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

    // Returns the absolute filepath of the loaded ONNX model.
    std::string getOnnxFileLocation()
    {
        return mOnnxFileLocation;
    }

    // Returns the logger object.
    nvinfer1::ILogger& logger()
    {
        return *mLogger;
    }

    MemoryMapping mmap(std::string const& file);

    void clearMemoryMappings();

    StringMap<::ONNX_NAMESPACE::TensorProto const*>& initializerMap()
    {
        return mInitializers;
    }

    bool loadExternalInit(char const* name, void const* data, size_t size);
};

template <typename T>
T* WeightsContext::convertInt32Data(
    std::span<int32_t const> weightValues, nvinfer1::Dims const& shape, int32_t onnxdtype)
{
    auto* ctx = this; // For logging macros.
    size_t const nbWeights = volume(shape);
    if (nbWeights > weightValues.size())
    {
        LOG_ERROR("int32 weights source holds " << weightValues.size() << " elements but shape " << shape
                                                << " requires " << nbWeights << ". Rejecting malformed weights.");
        return nullptr;
    }
    T* newWeights{static_cast<T*>(createTempWeights(onnxdtype, shape).values)};

    for (size_t i = 0; i < nbWeights; i++)
    {
        newWeights[i] = static_cast<T>(weightValues[i]);
    }
    return newWeights;
}
template <typename T>
[[nodiscard]] float* WeightsContext::convertToFp32(ShapedWeights const& w)
{
    int64_t const nbWeights = volume(w.shape);
    auto result = static_cast<float*>(createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, w.shape).values);
    std::copy_n(static_cast<T const*>(w.values), nbWeights, result);

    return result;
}

} // namespace onnx2trt
