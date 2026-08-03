/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvInferRuntime.h"
#include "Status.hpp"
#include "WeightsContext.hpp"
#include "errorHelpers.hpp"
#include <onnx/onnx-ml.pb.h>
#include <cstddef>
#include <set>
#include <span>
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
    size_t mNestedDepth{0};

    //! Set to keep track of how many times a refittable name created by the parser shows up, to avoid duplicate naming
    //! in TRT. Currently tracks the following nodes:
    //!     1. BatchNorm - Parser pre-combines scales and bias weights for the IScaleLayer.
    //!     2. ConstantOfShape - The value of the ConstantOfShape does not have a name, so the parser needs to create
    //!     one for it.
    std::set<std::string> mTempRefittableWeights;
    //! An increasing suffix counter used to uniquify refittable weight names created by the parser.
    int64_t mTempRefittableWeightsSuffixCounter{0};

    size_t mSuccessfullyRefittedWeights{};
    std::unordered_set<std::string> mRefittableWeights;
    std::unordered_set<std::string> mRefittedWeights;

    //! Optional observer receiving one callback per refittable weight during refit. Owned externally.
    nvonnxparser::IRefitterObserver* mObserver{nullptr};

    mutable std::vector<Status> mErrors;

    std::unordered_set<std::string> getRefittableWeights();

    //! Emit one record to the attached observer, if any. No-op when no observer is set.
    //! \param trtName Name of the TensorRT refittable engine weight.
    //! \param kind Transform that produces the refit data from the sources.
    //! \param onnxDtype ONNX TensorProto::DataType of the source data before any transformation
    //!   (e.g. DOUBLE for a kDOUBLE_TO_FLOAT initializer).
    //! \param trtDtype TensorRT data type of the post-transform refit data passed to the
    //!   refitter.
    //! \param count Element count of the emitted refit data.
    //! \param sources Parser-owned ONNX source names. Valid only during the callback.
    //! \param epsilon Epsilon used by kBATCH_NORM_FOLD_* transforms; ignored otherwise.
    //! \param fixedData Parser-owned attribute payload for kCONSTANT_NODE / kCONSTANT_OF_SHAPE.
    //!   Empty span for the other kinds. Valid only during the callback.
    void notifyObserver(char const* trtName, nvonnxparser::RefitTransformKind kind, int32_t onnxDtype,
        nvinfer1::DataType trtDtype, int64_t count, std::span<char const* const> sources,
        float epsilon = 0.0F, std::span<std::byte const> fixedData = {}) noexcept;

    //! T is the working type.
    //! TConvertFunc is a functor for converting ShapedWeights to an array of type T.
    //! It should return a T*.
    //! \p sourceOnnxDtype is the original ONNX TensorProto::DataType of the BN input initializers
    //! (before any DOUBLE-to-FLOAT promotion), reported verbatim through IRefitterObserver so a
    //! consumer can replay the fold from the source-dtype bytes.
    template <typename T, typename TConvertFunc>
    size_t batchnormWeightRefitter(::ONNX_NAMESPACE::NodeProto const& node, std::vector<ShapedWeights>& inputs,
        int32_t sourceOnnxDtype, TConvertFunc&& f);

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

    //! Set or clear the optional refit observer. Ownership remains with the caller.
    void setRefitObserver(nvonnxparser::IRefitterObserver* observer) noexcept override
    {
        mObserver = observer;
    }
};

} // namespace onnx2trt
