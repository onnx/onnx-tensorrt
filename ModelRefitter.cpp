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
void deserializeOnnxModelFile(char const* onnxModelFile, ::ONNX_NAMESPACE::ModelProto& onnx_model)
{
    // Define S_ISREG macro for Windows
#if !defined(S_ISREG)
#define S_ISREG(mode) (((mode) & S_IFMT) == S_IFREG)
#endif

    struct stat sb;
    ONNXTRT_CHECK(!(stat(onnxModelFile, &sb) == 0 && !S_ISREG(sb.st_mode))
            && "Failed to parse the ONNX model; input is not a regular file.",
        ErrorCode::kMODEL_DESERIALIZE_FAILED);

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    bool const fileLoadSuccess = ParseFromFileAsBinary(&onnx_model, onnxModelFile);
    ONNXTRT_CHECK(fileLoadSuccess && "Failed to parse the ONNX model!", ErrorCode::kMODEL_DESERIALIZE_FAILED);
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
size_t ModelRefitter::batchnormWeightRefitter(
    ::ONNX_NAMESPACE::NodeProto const& node, std::vector<ShapedWeights>& inputs, TConvertFunc&& f)
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

    ShapedWeights combinedScale = mWeightsContext.createNamedTempWeights(
        weightType, scale.shape, mBatchNormWeightNames, mBatchNormWeightSuffixCounter, /*batchNormNode=*/true);
    ShapedWeights combinedBias = mWeightsContext.createNamedTempWeights(
        weightType, bias.shape, mBatchNormWeightNames, mBatchNormWeightSuffixCounter, /*batchNormNode=*/true);

    // Validate that all the weights have the same amount of values
    bool allSame = scale.count() == bias.count() && mean.count() == scale.count() && variance.count() == scale.count()
        && combinedScale.count() == scale.count() && combinedBias.count() == scale.count();
    ONNXTRT_CHECK(allSame && "Inputs to BatchNormalization must have the same shape!", ErrorCode::kREFIT_FAILED);

    for (int32_t i = 0; i < nbChannels; ++i)
    {
        combinedScale.at<T>(i) = scaleValues[i] / sqrtf(varianceValues[i] + eps);
        combinedBias.at<T>(i) = biasValues[i] - meanValues[i] * combinedScale.at<T>(i);
    }
    size_t successfullyRefittedWeights = 0;
    if (refittableWeights.count(combinedScale.name))
    {
        refittableWeights.erase(combinedScale.name);
        ONNXTRT_CHECK(
            mRefitter->setNamedWeights(combinedScale.name, std::move(combinedScale)) && "Failed to set named weights",
            ErrorCode::kREFIT_FAILED);
        ++successfullyRefittedWeights;
    }
    if (refittableWeights.count(combinedBias.name))
    {
        refittableWeights.erase(combinedBias.name);
        ONNXTRT_CHECK(
            mRefitter->setNamedWeights(combinedBias.name, std::move(combinedBias)) && "Failed to set named weights",
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

void ModelRefitter::refitOnnxWeights(::ONNX_NAMESPACE::ModelProto const& onnx_model)
{
    nestedDepth = 0;
    successfullyRefittedWeights = 0;
    size_t const numberOfWeightsToRefit = refittableWeights.size();
    refitOnnxGraph(onnx_model.graph());
    ONNXTRT_CHECK(successfullyRefittedWeights == numberOfWeightsToRefit && "Failed to refit all the weights.",
        ErrorCode::kREFIT_FAILED);
}

void ModelRefitter::refitOnnxGraph(::ONNX_NAMESPACE::GraphProto const& graph)
{
    for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
    {
        if (!refittableWeights.count(initializer.name()))
        {
            continue;
        }
        // Remove the weight name from the set as some initializers
        // might have the same name across different nested constructs (e.g. IF nodes);
        // the assumption is that those weights would have the same value
        refittableWeights.erase(initializer.name());
        if (refittedWeights.count(initializer.name()))
        {
            LOG_REFITTER_WARNING("Duplicate initializer name ("
                << initializer.name() << ") was found when processing the graph (" << graph.name()
                << "). The refit process would only work properly if both initializers have the same values.");
        }
        else
        {
            refittedWeights.insert(initializer.name());
        }
        ShapedWeights weights;
        ONNXTRT_CHECK(mWeightsContext.convertOnnxWeights(initializer, &weights, /*ownAllWeights=*/true)
                && "Failed to import initializer.",
            ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK(
            mRefitter->setNamedWeights(initializer.name().c_str(), std::move(weights)) && "Failed to set named weights",
            ErrorCode::kREFIT_FAILED);
        ++successfullyRefittedWeights;
    }

    std::vector<size_t> topoOrder;
    ONNXTRT_CHECK(
        toposort(graph.node(), &topoOrder) && "Failed to sort the model topologically.", ErrorCode::kINVALID_GRAPH);

    for (auto const& nodeIdx : topoOrder)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = graph.node(nodeIdx);
        refitOnnxNode(node, graph);
    }
}

void ModelRefitter::refitOnnxNode(::ONNX_NAMESPACE::NodeProto const& node, ::ONNX_NAMESPACE::GraphProto const& graph)
{
    // For nodes that contain subgraphs (Ifs, Loops, Scans),
    // ensure that the recursion depth is limited to a set amount.
    ++nestedDepth;
    static size_t const MAX_NESTED_SUBGRAPHS = 24;
    ONNXTRT_CHECK((nestedDepth <= MAX_NESTED_SUBGRAPHS)
            && "ONNX graph contains nested structures that exceed the maximum allowed by TensorRT!",
        ErrorCode::kUNSUPPORTED_GRAPH);

    if (node.op_type() == "Constant")
    {
        refitOnnxConstantNode(node, graph.name());
    }
    else if (node.op_type() == "BatchNormalization")
    {
        refitOnnxBatchNormNode(node, graph);
    }
    else if (node.op_type() == "If")
    {
        refitOnnxIfNode(node);
    }
    else if (node.op_type() == "Loop")
    {
        refitOnnxLoopNode(node);
    }
    else if (node.op_type() == "Scan")
    {
        refitOnnxScanNode(node);
    }
    --nestedDepth;
}

void ModelRefitter::refitOnnxConstantNode(::ONNX_NAMESPACE::NodeProto const& node, std::string const& graphName)
{
    if (!refittableWeights.count(node.output(0)))
    {
        return;
    }
    refittableWeights.erase(node.output(0));
    if (refittedWeights.count(node.output(0)))
    {
        LOG_REFITTER_WARNING("Duplicate weight name name ("
            << node.output(0) << ") was found when processing the graph (" << graphName
            << "). The refit process would only work properly if both weights have the same values.");
    }
    else
    {
        refittedWeights.insert(node.output(0));
    }
    ShapedWeights weights;
    ::ONNX_NAMESPACE::AttributeProto const& nodeAttribute = node.attribute(0);
    if (nodeAttribute.name() == "value_float")
    {
        weights = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
        float value = nodeAttribute.f();
        ONNXTRT_CHECK(weights.count() == 1 && "Failed to import Constant node.", ErrorCode::kUNSUPPORTED_NODE);
        std::memcpy(weights.values, &value, sizeof(float));
    }
    else if (nodeAttribute.name() == "value_floats")
    {
        std::vector<float> values{nodeAttribute.floats().begin(), nodeAttribute.floats().end()};
        int64_t valueSize = values.size();
        weights = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {1, {valueSize}});
        ONNXTRT_CHECK(
            weights.count() == values.size() && "Failed to import Constant node.", ErrorCode::kUNSUPPORTED_NODE);
        std::memcpy(weights.values, values.data(), weights.count() * sizeof(float));
    }
    else if (nodeAttribute.name() == "value_int")
    {
        weights = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {0, {}});
        int64_t value = nodeAttribute.i();
        ONNXTRT_CHECK(weights.count() == 1 && "Failed to import Constant node.", ErrorCode::kUNSUPPORTED_NODE);
        std::memcpy(weights.values, &value, sizeof(int64_t));
    }
    else if (nodeAttribute.name() == "value_ints")
    {
        std::vector<int64_t> values{nodeAttribute.ints().begin(), nodeAttribute.ints().end()};
        int64_t valueSize = values.size();
        weights = mWeightsContext.createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {1, {valueSize}});
        ONNXTRT_CHECK(
            weights.count() == values.size() && "Failed to import Constant node.", ErrorCode::kUNSUPPORTED_NODE);
        std::memcpy(weights.values, values.data(), weights.count() * sizeof(int64_t));
    }
    else
    {
        ::ONNX_NAMESPACE::TensorProto const& onnx_weights_tensor = nodeAttribute.t();
        ONNXTRT_CHECK(
            mWeightsContext.convertOnnxWeights(onnx_weights_tensor, &weights) && "Failed to import Constant node.",
            ErrorCode::kUNSUPPORTED_NODE);
    }
    ONNXTRT_CHECK(
        mRefitter->setNamedWeights(node.output(0).c_str(), std::move(weights)) && "Failed to set named weights",
        ErrorCode::kREFIT_FAILED);
    ++successfullyRefittedWeights;
}

void ModelRefitter::refitOnnxBatchNormNode(
    ::ONNX_NAMESPACE::NodeProto const& node, ::ONNX_NAMESPACE::GraphProto const& graph)
{
    ONNXTRT_CHECK(
        node.input().size() == 5 && "BatchNorm node does not have five required inputs.", ErrorCode::kINVALID_NODE);
    std::vector<ShapedWeights> batchNormInputs;
    // The following looping construct is due to the fact that some tensors
    // might be shared among the BatchNorm's inputs
    std::vector<std::string> const inputNames(node.input().begin() + 1, node.input().end());
    for (size_t inputIdx = 0; inputIdx < inputNames.size(); ++inputIdx)
    {
        for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
        {
            if (inputNames.at(inputIdx) == initializer.name())
            {
                ShapedWeights weights;
                ONNXTRT_CHECK(
                    mWeightsContext.convertOnnxWeights(initializer, &weights) && "Failed to import initializer.",
                    ErrorCode::kUNSUPPORTED_NODE);
                weights.name = initializer.name().c_str();
                batchNormInputs.push_back(std::move(weights));
                break;
            }
        }
    }

    // If some of the inputs to the BN node were not actual initializers,
    // the weight folding logic from Parser is no longer applicable and
    // we must have already refitted the weights directly in refitOnnxGraph()
    if (batchNormInputs.size() < 4)
    {
        return;
    }
    size_t batchnormRefittedWeights{0};
    auto const scaleType = batchNormInputs.at(0).type;
    bool const typesEqual = scaleType == batchNormInputs.at(1).type && scaleType == batchNormInputs.at(2).type
        && scaleType == batchNormInputs.at(3).type;
    if (typesEqual && scaleType == ::ONNX_NAMESPACE::TensorProto::FLOAT16)
    {
        batchnormRefittedWeights
            = batchnormWeightRefitter<half_float::half>(node, batchNormInputs, QuickCast<half_float::half>());
    }
    else if (typesEqual && scaleType == ::ONNX_NAMESPACE::TensorProto::BFLOAT16)
    {
        batchnormRefittedWeights = batchnormWeightRefitter<BFloat16>(node, batchNormInputs, QuickCast<BFloat16>());
    }
    else
    {
        // Do calculations in FP32, possibly promoting/demoting arithmetic types of some operands.
        batchnormRefittedWeights = batchnormWeightRefitter<float>(
            node, batchNormInputs, [this](ShapedWeights const& w) { return mWeightsContext.getFP32Values(w); });
    }
    successfullyRefittedWeights += batchnormRefittedWeights;
}

void ModelRefitter::refitOnnxIfNode(::ONNX_NAMESPACE::NodeProto const& node)
{
    size_t thenGraphOutputSize{};
    size_t elseGraphOutputSize{};
    for (auto const& attr : node.attribute())
    {
        if (attr.name() == "then_branch")
        {
            ::ONNX_NAMESPACE::GraphProto const& thenGraph = static_cast<::ONNX_NAMESPACE::GraphProto const&>(attr.g());
            refitOnnxGraph(thenGraph);
            thenGraphOutputSize = thenGraph.output_size();
        }
        else if (attr.name() == "else_branch")
        {
            ::ONNX_NAMESPACE::GraphProto const& elseGraph = static_cast<::ONNX_NAMESPACE::GraphProto const&>(attr.g());
            refitOnnxGraph(elseGraph);
            elseGraphOutputSize = elseGraph.output_size();
        }
    }

    // Number of outputs are the same between the two branches.
    ONNXTRT_CHECK(thenGraphOutputSize == elseGraphOutputSize
            && "then/else subgraphs within the IF node should have the same number of outputs",
        ErrorCode::kREFIT_FAILED);
}

void ModelRefitter::refitOnnxLoopNode(::ONNX_NAMESPACE::NodeProto const& node)
{
    ::ONNX_NAMESPACE::GraphProto const& body = static_cast<::ONNX_NAMESPACE::GraphProto const&>(node.attribute(0).g());
    refitOnnxGraph(body);
}

void ModelRefitter::refitOnnxScanNode(::ONNX_NAMESPACE::NodeProto const& node)
{
    for (auto const& attr : node.attribute())
    {
        if (attr.name() == "body")
        {
            ::ONNX_NAMESPACE::GraphProto const& body = static_cast<::ONNX_NAMESPACE::GraphProto const&>(attr.g());
            refitOnnxGraph(body);
            break;
        }
    }
}

bool ModelRefitter::refitFromBytes(
    void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath) noexcept
{
    ONNXTRT_TRY
    {
        if (modelPath)
        {
            // Keep track of the absolute path to the ONNX file.
            mWeightsContext.setOnnxFileLocation(modelPath);
        }

        deserializeOnnxModel(serializedOnnxModel, serializedOnnxModelSize, &onnx_model);

        refittableWeights = getRefittableWeights();
        refitOnnxWeights(onnx_model);
        return true;
    }
    ONNXTRT_CATCH_LOG(mLogger)
    return false;
}

bool ModelRefitter::refitFromFile(char const* onnxModelFile) noexcept
{
    ONNXTRT_TRY
    {
        // Keep track of the absolute path to the ONNX file.
        mWeightsContext.setOnnxFileLocation(onnxModelFile);

        deserializeOnnxModelFile(onnxModelFile, onnx_model);
        refittableWeights = getRefittableWeights();
        if (!refittableWeights.empty())
        {
            refitOnnxWeights(onnx_model);
        }
        return true;
    }
    ONNXTRT_CATCH_LOG(mLogger)

    return false;
}
} // namespace onnx2trt
