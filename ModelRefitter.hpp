/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvInferRuntime.h"
#include "Status.hpp"
#include "WeightsContext.hpp"
#include <onnx/onnx_pb.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace onnx2trt
{
class ModelRefitter : public nvonnxparser::IParserRefitter
{
private:
    nvinfer1::IRefitter* mRefitter;
    nvinfer1::ILogger* mLogger;

    // WeightsContext object to hold ownership of ONNX weights and any temporary weights created by the refitter.
    WeightsContext mWeightsContext;

    // ONNX ModelProto object to hold ownership of ONNX weights whenever a data type conversion is not needed
    ::ONNX_NAMESPACE::ModelProto onnx_model;

    std::unordered_set<std::string> refittable_weights;

    std::vector<Status> mErrors;

    std::unordered_set<std::string> getRefittableWeights();

    //! T is the working type.
    //! TConvertFunc is a functor for converting ShapedWeights to an array of type T.
    //! It should return a T*.
    template <typename T, typename TConvertFunc>
    ValueOrStatus<size_t> batchnormWeightRefitter(::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
        std::vector<ShapedWeights>& inputs, TConvertFunc&& f);

    Status refitOnnxWeights(::ONNX_NAMESPACE::ModelProto const& onnx_model);

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
        assert(0 <= index && index < (int32_t) mErrors.size());
        return &mErrors[index];
    }

    void clearErrors() noexcept override
    {
        mErrors.clear();
    }
};

} // namespace onnx2trt
