/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvInferRuntime.h"
#include "Status.hpp"
#include "WeightsContext.hpp"
#include "errorHelpers.hpp"
#include <onnx/onnx_pb.h>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

// Logging macros
#define LOG_REFITTER(msg, severity)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        std::ostringstream ss{};                                                                                       \
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)                                                         \
            ss << ONNX2TRT_FILENAME << ":" << __LINE__ << ": ";                                                        \
        ss << msg;                                                                                                     \
        mLogger->log(severity, ss.str().c_str());                                                                      \
    } while (0)

#define LOG_REFITTER_VERBOSE(msg) LOG_REFITTER(msg, nvinfer1::ILogger::Severity::kVERBOSE)
#define LOG_REFITTER_WARNING(msg) LOG_REFITTER(msg, nvinfer1::ILogger::Severity::kWARNING)
#define LOG_REFITTER_ERROR(msg) LOG_REFITTER(msg, nvinfer1::ILogger::Severity::kERROR)

namespace onnx2trt
{
class ModelRefitter : public nvonnxparser::IParserRefitter
{
private:
    nvinfer1::IRefitter* mRefitter;
    nvinfer1::ILogger* mLogger;

    //! WeightsContext object to hold ownership of ONNX weights and any temporary weights created by the refitter.
    WeightsContext mWeightsContext;

    //! ONNX ModelProto object to hold ownership of ONNX weights whenever a data type conversion is not needed.
    ::ONNX_NAMESPACE::ModelProto mOnnxModel;

    //! Counter to limit the recursion depth to a set amount for nodes containing subgraphs.
    size_t nestedDepth{0};

    //! Set to keep track of how many times a refittable name created by the parser shows up, to avoid duplicate naming in TRT.
    //! Currently tracks the following nodes:
    //!     1. BatchNorm - Parser pre-combines scales and bias weights for the IScaleLayer.
    //!     2. ConstantOfShape - The value of the ConstantOfShape does not have a name, so the parser needs to create one for it.
    std::set<std::string> mTempRefittableWeights;
    //! An increasing suffix counter used to uniquify refittable weight names created by the parser.
    int64_t mTempRefittableWeightsSuffixCounter{0};

    size_t successfullyRefittedWeights{};
    std::unordered_set<std::string> mRefittableWeights;
    std::unordered_set<std::string> mRefittedWeights;

    mutable std::vector<Status> mErrors;

    std::unordered_set<std::string> getRefittableWeights();

    //! T is the working type.
    //! TConvertFunc is a functor for converting ShapedWeights to an array of type T.
    //! It should return a T*.
    template <typename T, typename TConvertFunc>
    size_t batchnormWeightRefitter(
        ::ONNX_NAMESPACE::NodeProto const& node, std::vector<ShapedWeights>& inputs, TConvertFunc&& f);

    void refitOnnxWeights();
    void refitOnnxGraph(::ONNX_NAMESPACE::GraphProto const& graph);
    void refitOnnxNode(::ONNX_NAMESPACE::NodeProto const& node, ::ONNX_NAMESPACE::GraphProto const& graph);
    void refitOnnxConstantNode(::ONNX_NAMESPACE::NodeProto const& node, std::string const& graphName);
    void refitOnnxConstantOfShapeNode(::ONNX_NAMESPACE::NodeProto const& node, std::string const& graphName);
    void refitOnnxBatchNormNode(::ONNX_NAMESPACE::NodeProto const& node, ::ONNX_NAMESPACE::GraphProto const& graph);
    void refitOnnxIfNode(::ONNX_NAMESPACE::NodeProto const& node);
    void refitOnnxLoopNode(::ONNX_NAMESPACE::NodeProto const& node);
    void refitOnnxScanNode(::ONNX_NAMESPACE::NodeProto const& node);

public:
    ModelRefitter(nvinfer1::IRefitter* refitter, nvinfer1::ILogger* logger)
        : mRefitter{refitter}
        , mLogger{logger}
        , mWeightsContext{WeightsContext{logger}}
    {
    }

    bool refitFromBytes(void const* serializedOnnxModel, size_t serializedOnnxModelSize,
        char const* modelPath = nullptr) noexcept override;
    bool refitFromFile(char const* onnxModelFile) noexcept override;

    int32_t getNbErrors() const noexcept override
    {
        return mErrors.size();
    }

    nvonnxparser::IParserError const* getError(int32_t index) const noexcept override
    {
        ONNXTRT_TRY
        {
            return (index >= 0 && static_cast<size_t>(index) < mErrors.size()) ? &mErrors.at(index) : nullptr;
        }
        ONNXTRT_CATCH_LOG(mLogger)
        return nullptr;
    }

    void clearErrors() noexcept override
    {
        mErrors.clear();
    }

    bool loadModelProto(void const* serializedOnnxModel, size_t serializedOnnxModelSize,
        char const* modelPath = nullptr) noexcept override;

    bool loadInitializer(char const* name, void const* data, size_t size) noexcept override;

    bool refitModelProto() noexcept override;
};

} // namespace onnx2trt
