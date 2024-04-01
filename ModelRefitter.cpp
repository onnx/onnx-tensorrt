/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ModelRefitter.hpp"
#include "ShapedWeights.hpp"
#include "onnxProtoUtils.hpp"
#include "toposort.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <algorithm>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace onnx2trt
{
namespace
{
Status deserializeOnnxModelFile(char const* onnxModelFile, ::ONNX_NAMESPACE::ModelProto& onnx_model)
{
    // Define S_ISREG macro for Windows
#if !defined(S_ISREG)
#define S_ISREG(mode) (((mode) &S_IFMT) == S_IFREG)
#endif

    struct stat sb;
    ASSERT(!(stat(onnxModelFile, &sb) == 0 && !S_ISREG(sb.st_mode))
            && "Failed to parse the ONNX model; input is not a regular file.",
        ErrorCode::kMODEL_DESERIALIZE_FAILED);

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    bool const fileLoadSuccess = ParseFromFileAsBinary(&onnx_model, onnxModelFile);
    ASSERT(fileLoadSuccess && "Failed to parse the ONNX model!", ErrorCode::kMODEL_DESERIALIZE_FAILED);
    return Status::success();
}
} // anonymous namespace

std::unordered_set<std::string> ModelRefitter::getRefittableWeights()
{
    int32_t numWeights = mRefitter->getAllWeights(0, nullptr);
    std::vector<char const*> weightNames{static_cast<size_t>(numWeights)};
    mRefitter->getAllWeights(numWeights, weightNames.data());
    return std::unordered_set<std::string>{weightNames.begin(), weightNames.end()};
}

template <typename T, typename TConvertFunc>
ValueOrStatus<size_t> ModelRefitter::batchnormWeightRefitter(
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx, std::vector<ShapedWeights>& inputs, TConvertFunc&& f)
{
    auto const& scale = inputs.at(0);
    auto const& bias = inputs.at(1);
    auto const& mean = inputs.at(2);
    auto const& variance = inputs.at(3);

    T const* const scaleValues = f(scale);
    T const* const biasValues = f(bias);
    T const* const meanValues = f(mean);
    T const* const varianceValues = f(variance);

    T eps = static_cast<T>(1e-5f);

    for (auto const& attr : node.attribute())
    {
        if (attr.name() == "epsilon")
        {
            eps = static_cast<T>(attr.f());
            break;
        }
    }

    // Fold the weights together into a single bias and scale
    int32_t const nbChannels = scale.shape.d[0];
    ShapedWeights::DataType weightType = typeid(T).hash_code() == typeid(BFloat16).hash_code()
        ? ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        : (typeid(T).hash_code() == typeid(half_float::half).hash_code() ? ::ONNX_NAMESPACE::TensorProto::FLOAT16
                                                                         : ::ONNX_NAMESPACE::TensorProto::FLOAT);

    ShapedWeights combinedScale = mWeightsContext.createTempWeights(weightType, scale.shape);
    ShapedWeights combinedBias = mWeightsContext.createTempWeights(weightType, bias.shape);

    // Validate that all the weights have the same amount of values
    bool allSame = scale.count() == bias.count() && mean.count() == scale.count() && variance.count() == scale.count()
        && combinedScale.count() == scale.count() && combinedBias.count() == scale.count();
    ASSERT(allSame && "Inputs to BatchNormalization must have the same shape!", ErrorCode::kREFIT_FAILED);

    for (int32_t i = 0; i < nbChannels; ++i)
    {
        combinedScale.at<T>(i) = scaleValues[i] / sqrtf(varianceValues[i] + eps);
        combinedBias.at<T>(i) = biasValues[i] - meanValues[i] * combinedScale.at<T>(i);
    }
    size_t successfullyRefittedWeights = 0;
    const std::string node_id = "node_trt_batch_norm_" + std::to_string(nodeIdx);
    const std::string scale_weight = node_id + "_scale_weight";
    const std::string bias_weight = node_id + "_bias_weight";
    if (refittable_weights.count(scale_weight))
    {
        ASSERT(
            mRefitter->setNamedWeights(scale_weight.c_str(), std::move(combinedScale)) && "Failed to set named weights",
            ErrorCode::kREFIT_FAILED);
        ++successfullyRefittedWeights;
    }
    if (refittable_weights.count(bias_weight))
    {
        ASSERT(
            mRefitter->setNamedWeights(bias_weight.c_str(), std::move(combinedBias)) && "Failed to set named weights",
            ErrorCode::kREFIT_FAILED);
        ++successfullyRefittedWeights;
    }
    return successfullyRefittedWeights;
}

//! Functor for extracting weights from ShapedWeights via cheap pointer cast to T*.
template <typename T>
class QuickCast
{
public:
    T const* operator()(ShapedWeights const& w) const
    {
        return static_cast<T const*>(w.values);
    };
};

Status ModelRefitter::refitOnnxWeights(::ONNX_NAMESPACE::ModelProto const& onnx_model)
{
    size_t successfullyRefittedWeights = 0;
    for (::ONNX_NAMESPACE::TensorProto const& initializer : onnx_model.graph().initializer())
    {
        if (!refittable_weights.count(initializer.name()))
        {
            continue;
        }
        ShapedWeights weights;
        ASSERT(mWeightsContext.convertOnnxWeights(initializer, &weights, /*ownAllWeights=*/true)
                && "Failed to import initializer.",
            ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(
            mRefitter->setNamedWeights(initializer.name().c_str(), std::move(weights)) && "Failed to set named weights",
            ErrorCode::kREFIT_FAILED);
        ++successfullyRefittedWeights;
    }

    std::vector<size_t> topoOrder;
    ASSERT(toposort(onnx_model.graph().node(), &topoOrder) && "Failed to sort the model topologically.",
        ErrorCode::kINVALID_GRAPH);

    // Import Constant nodes (mapping their node output names to engine weight names) and BatchNormalization nodes
    for (auto const& nodeIdx : topoOrder)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = onnx_model.graph().node(nodeIdx);
        if (node.op_type() == "Constant")
        {
            if (!refittable_weights.count(node.output(0)))
            {
                continue;
            }
            ::ONNX_NAMESPACE::AttributeProto const& node_attribute = node.attribute(0);
            if (node_attribute.name() == "value_float")
            {
                ShapedWeights convertedWeights
                    = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
                float value = node_attribute.f();
                std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(float));
                ASSERT(mRefitter->setNamedWeights(node.output(0).c_str(), std::move(convertedWeights))
                        && "Failed to set named weights",
                    ErrorCode::kREFIT_FAILED);
            }
            else if (node_attribute.name() == "value_floats")
            {
                std::vector<float> values{node_attribute.floats().begin(), node_attribute.floats().end()};
                int32_t valueSize = values.size();
                ShapedWeights convertedWeights
                    = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {1, {valueSize}});
                std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(float));
                ASSERT(mRefitter->setNamedWeights(node.output(0).c_str(), std::move(convertedWeights))
                        && "Failed to set named weights",
                    ErrorCode::kREFIT_FAILED);
            }
            else if (node_attribute.name() == "value_int")
            {
                ShapedWeights convertedWeights
                    = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {0, {}});
                int64_t value = node_attribute.i();
                std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(int64_t));
                ASSERT(mRefitter->setNamedWeights(node.output(0).c_str(), std::move(convertedWeights))
                        && "Failed to set named weights",
                    ErrorCode::kREFIT_FAILED);
            }
            else if (node_attribute.name() == "value_ints")
            {
                std::vector<int64_t> values{node_attribute.ints().begin(), node_attribute.ints().end()};
                int32_t valueSize = values.size();
                ShapedWeights convertedWeights
                    = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {1, {valueSize}});
                std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(int64_t));
                ASSERT(mRefitter->setNamedWeights(node.output(0).c_str(), std::move(convertedWeights))
                        && "Failed to set named weights",
                    ErrorCode::kREFIT_FAILED);
            }
            else
            {
                ::ONNX_NAMESPACE::TensorProto const& onnx_weights_tensor = node_attribute.t();
                ShapedWeights weights;
                ASSERT(mWeightsContext.convertOnnxWeights(onnx_weights_tensor, &weights)
                        && "Failed to import Constant node.",
                    ErrorCode::kUNSUPPORTED_NODE);
                ASSERT(mRefitter->setNamedWeights(node.output(0).c_str(), std::move(weights))
                        && "Failed to set named weights",
                    ErrorCode::kREFIT_FAILED);
            }
            ++successfullyRefittedWeights;
        }
        else if (node.op_type() == "BatchNormalization")
        {
            ASSERT(node.input().size() == 5 && "BatchNorm node does not have five required inputs.",
                ErrorCode::kINVALID_NODE);

            std::vector<ShapedWeights> batch_norm_inputs(4);
            // The following looping construct is due to the fact that some tensors might be shared among the
            // BatchNorm's inputs
            const std::vector<std::string> input_names(node.input().begin() + 1, node.input().end());
            for (size_t inputIdx = 0; inputIdx < input_names.size(); ++inputIdx)
            {
                for (::ONNX_NAMESPACE::TensorProto const& initializer : onnx_model.graph().initializer())
                {
                    if (input_names.at(inputIdx) == initializer.name())
                    {
                        ShapedWeights weights;
                        ASSERT(mWeightsContext.convertOnnxWeights(initializer, &weights)
                                && "Failed to import initializer.",
                            ErrorCode::kUNSUPPORTED_NODE);
                        weights.name = initializer.name().c_str();
                        batch_norm_inputs.at(inputIdx) = std::move(weights);
                        break;
                    }
                }
            }

            ValueOrStatus<size_t> batchnorm_refitted_weights{0};
            auto const scaleType = batch_norm_inputs.at(0).type;
            bool const typesEqual = scaleType == batch_norm_inputs.at(1).type
                && scaleType == batch_norm_inputs.at(2).type && scaleType == batch_norm_inputs.at(3).type;
            if (typesEqual && scaleType == ::ONNX_NAMESPACE::TensorProto::FLOAT16)
            {
                batchnorm_refitted_weights = batchnormWeightRefitter<half_float::half>(
                    node, nodeIdx, batch_norm_inputs, QuickCast<half_float::half>());
                if (batchnorm_refitted_weights.is_error())
                {
                    return batchnorm_refitted_weights.error();
                }
            }
            else if (typesEqual && scaleType == ::ONNX_NAMESPACE::TensorProto::BFLOAT16)
            {
                batchnorm_refitted_weights
                    = batchnormWeightRefitter<BFloat16>(node, nodeIdx, batch_norm_inputs, QuickCast<BFloat16>());
                if (batchnorm_refitted_weights.is_error())
                {
                    return batchnorm_refitted_weights.error();
                }
            }
            else
            {
                // Do calculations in FP32, possibly promoting/demoting arithmetic types of some operands.
                batchnorm_refitted_weights = batchnormWeightRefitter<float>(node, nodeIdx, batch_norm_inputs,
                    [this](ShapedWeights const& w) { return mWeightsContext.getFP32Values(w); });
                if (batchnorm_refitted_weights.is_error())
                {
                    return batchnorm_refitted_weights.error();
                }
            }
            successfullyRefittedWeights += batchnorm_refitted_weights.value();
        }
    }
    ASSERT(successfullyRefittedWeights == refittable_weights.size() && "Failed to refit all the weights.",
        ErrorCode::kREFIT_FAILED);
    return Status::success();
}

bool ModelRefitter::refitFromBytes(
    void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath) noexcept
{
    if (modelPath)
    {
        // Keep track of the absolute path to the ONNX file.
        mWeightsContext.setOnnxFileLocation(modelPath);
    }

    Status status
        = deserializeOnnxModel(serializedOnnxModel, serializedOnnxModelSize, &onnx_model);
    if (status.is_error())
    {
        mErrors.push_back(status);
        return false;
    }

    refittable_weights = getRefittableWeights();
    status = refitOnnxWeights(onnx_model);
    if (status.is_error())
    {
        mErrors.push_back(status);
        return false;
    }
    return true;
}

bool ModelRefitter::refitFromFile(char const* onnxModelFile) noexcept
{
    // Keep track of the absolute path to the ONNX file.
    mWeightsContext.setOnnxFileLocation(onnxModelFile);

    Status status = deserializeOnnxModelFile(onnxModelFile, onnx_model);
    if (status.is_error())
    {
        mErrors.push_back(status);
        return false;
    }

    refittable_weights = getRefittableWeights();
    if (!refittable_weights.empty())
    {
        status = refitOnnxWeights(onnx_model);
        if (status.is_error())
        {
            mErrors.push_back(status);
            return false;
        }
    }
    return true;
}
} // namespace onnx2trt
