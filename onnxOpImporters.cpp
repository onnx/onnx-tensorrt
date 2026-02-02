/*
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include "AttentionHelpers.hpp"
#include "ConditionalHelpers.hpp"
#include "LoopHelpers.hpp"
#include "ModelImporter.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "OnnxAttrs.hpp"
#include "RNNHelpers.hpp"
#include "ShapeTensor.hpp"
#include "bfloat16.hpp"
#include "errorHelpers.hpp"
#include "half.h"
#include "importerUtils.hpp"
#include "onnxOpImporters.hpp"

#include <algorithm> // For std::min, std::max
#include <array>
#include <cstring> // For std::memcpy, std::memset
#include <iostream>
#include <iterator>
#include <numeric> // For std::iota
#include <sstream>
#include <tuple>
#include <unordered_set>


namespace onnx2trt
{

StringMap<NodeImporter>& getBuiltinOpImporterMap()
{
    static StringMap<NodeImporter> builtin_op_importers;
    return builtin_op_importers;
}

namespace
{

using nvinfer1::DataType;

#define IGNORE_UNUSED_GLOBAL(x)                                                                                        \
    static void _ignore_unused2_##x();                                                                                 \
    static void _ignore_unused1_##x()                                                                                  \
    {                                                                                                                  \
        (void) _ignore_unused2_##x;                                                                                    \
        (void) x;                                                                                                      \
    }                                                                                                                  \
    static void _ignore_unused2_##x()                                                                                  \
    {                                                                                                                  \
        (void) _ignore_unused1_##x;                                                                                    \
    }                                                                                                                  \
    struct SwallowSemicolon##x                                                                                         \
    {                                                                                                                  \
    }

#define DECLARE_BUILTIN_OP_IMPORTER(op)                                                                                \
    NodeOutputs import##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,        \
        std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op)                                                                                 \
    NodeOutputs import##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,        \
        std::vector<TensorOrWeights>& inputs);                                                                         \
    static bool const op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeOutputs import##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,        \
        std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer, node, nodeIdx)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ONNXTRT_CHECK_NODE(layer_ptr, "Input layer is null.", node, nodeIdx, ErrorCode::kINVALID_NODE);                \
        auto* output = N_CHECK(layer->getOutput(0));                                                                   \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_IDENTITY(input, node, nodeIdx)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        TensorOrWeights output = identity(ctx, input);                                                                 \
        ONNXTRT_CHECK_NODE(output, "Failed to add an identity layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);   \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_ALL_OUTPUTS(layer, node, nodeIdx)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ONNXTRT_CHECK_NODE(layer_ptr, "The input layer is null.", node, nodeIdx, ErrorCode::kINVALID_NODE);            \
        std::vector<TensorOrWeights> outputs;                                                                          \
        for (int i = 0; i < layer_ptr->getNbOutputs(); ++i)                                                            \
            outputs.push_back(N_CHECK(layer_ptr->getOutput(i)));                                                       \
        return {outputs};                                                                                              \
    } while (0)

void assertIsWeights(TensorOrWeights const& input, std::string const& specificMsg)
{
    if (!input.is_weights())
    {
        std::ostringstream msg;
        msg << specificMsg;
        msg << " Try applying constant folding on the model using Polygraphy: "
               "https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/examples/cli/surgeon/"
               "02_folding_constants";
        throw std::runtime_error(msg.str());
    }
}

bool registerBuiltinOpImporter(std::string op, NodeImporter const& importer)
{
    bool inserted = getBuiltinOpImporterMap().insert({op, importer}).second;
    assert(inserted);
    return inserted;
}

bool onlySupportInt32TRTPlugin(std::string const& pluginName)
{
    // TRT plugins that doesn't support INT64 as inputs, but support INT32.
    static std::vector<std::string> const names = {
        "CustomQKVToContextPluginDynamic",
        "EfficientNMS_TRT",
        "EfficientNMS_ONNX_TRT",
        "EfficientNMS_Implicit_TF_TRT",
        "EfficientNMS_Explicit_TF_TRT",
        "VoxelGeneratorPlugin",
        "ScatterND",
        "ROIAlign_TRT",
        "PillarScatterPlugin",
        "MultiscaleDeformableAttnPlugin_TRT",
        "CustomEmbLayerNormPluginDynamic",
    };
    return std::find(names.begin(), names.end(), pluginName) != names.end();
}

DEFINE_BUILTIN_OP_IMPORTER(Abs)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acos)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kACOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acosh)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kACOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(And)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kAND);
}

DEFINE_BUILTIN_OP_IMPORTER(Asin)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kASIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asinh)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kASINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Atan)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kATAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Atanh)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kATANH);
}

DEFINE_BUILTIN_OP_IMPORTER(Attention)
{
    ONNXTRT_CHECK_NODE(node.output().size() == 1, "TensorRT only supports Attention nodes with one output.", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);

    ONNXTRT_CHECK_NODE(inputs.at(0).shape().nbDims == inputs.at(1).shape().nbDims
            && inputs.at(0).shape().nbDims == inputs.at(2).shape().nbDims,
        "Q, K, and V inputs must have the same number of dimensions.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    // 3D case needs to be reshaped to 4D.
    bool const needsReshape = inputs.at(0).shape().nbDims == 3;

    // Get inputs.
    nvinfer1::ITensor& query = convertToQTensor(inputs.at(0), attrs, ctx, needsReshape);
    nvinfer1::ITensor& key = convertToKTensor(inputs.at(1), attrs, ctx, needsReshape);
    nvinfer1::ITensor& value = convertToVTensor(inputs.at(2), attrs, ctx, needsReshape);
    bool const hasAttnMask = inputs.size() > 3 && !inputs.at(3).isNullTensor();
    bool const hasPastKey = inputs.size() > 4 && !inputs.at(4).isNullTensor();
    bool const hasPastValue = inputs.size() > 5 && !inputs.at(5).isNullTensor();

    ONNXTRT_CHECK_NODE(!hasPastKey, "Setting past_key for the Attention node is not supported in TensorRT.", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE(!hasPastValue, "Setting past_value for the Attention node is not supported in TensorRT.", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Get attributes.
    bool const isCausal = static_cast<bool>(attrs.get<int64_t>("is_causal", 0));
    nvinfer1::AttentionNormalizationOp const normOp = parseNormalizationOp(attrs);
    bool const decomposable = static_cast<bool>(attrs.get<int64_t>("TRT_decomposable", 0));

    // Add the Attention layer.
    nvinfer1::IAttention* attention = N_CHECK(ctx->network()->addAttention(query, key, value, normOp, isCausal));
    ctx->registerAttention(attention, node);
    attention->setDecomposable(decomposable);

    if (hasAttnMask)
    {
        attention->setMask(convertToMaskTensor(inputs.at(3), ctx));
    }

    return {{&reshapeOutputTensor(*attention->getOutput(0), ctx, needsReshape)}};
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_QuantizedAttention)
{
    OnnxAttrs attrs(node, ctx);

    auto normQuantizeToTypeOnnx
        = attrs.get<int32_t>("normalization_quantize_to_type", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    DataType normQuantizeToType;
    if (normQuantizeToTypeOnnx == ::ONNX_NAMESPACE::TensorProto::UNDEFINED)
    {
        LOG_WARNING("normalization_quantize_to_type is not set. Defaulting to FP8.");
        normQuantizeToType = DataType::kFP8;
    }
    else
    {
        ONNXTRT_CHECK_NODE(convertDtype(normQuantizeToTypeOnnx, &normQuantizeToType),
            "Attribute normalization_quantize_to_type specifies an unsupported data type " << normQuantizeToType << ".",
            node, nodeIdx, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(normQuantizeToType == DataType::kFP8 || normQuantizeToType == DataType::kINT8,
            "normalization_quantize_to_type must be FP8 or INT8.", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }

    bool const isCausal = static_cast<bool>(attrs.get<int64_t>("is_causal", 0));
    nvinfer1::AttentionNormalizationOp const normOp = parseNormalizationOp(attrs);
    bool const decomposable = static_cast<bool>(attrs.get<int64_t>("TRT_decomposable", 0));

    ONNXTRT_CHECK_NODE(inputs.size() == 5, "Quantized Attention requires 5 inputs.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);

    ONNXTRT_CHECK_NODE(inputs.at(0).shape().nbDims == inputs.at(1).shape().nbDims
            && inputs.at(0).shape().nbDims == inputs.at(2).shape().nbDims,
        "Q, K, and V inputs must have the same number of dimensions.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    ONNXTRT_CHECK_NODE(inputs.at(0).shape().nbDims == 4, "Quantized Attention only supports 4D inputs.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);

    // Get inputs.
    nvinfer1::ITensor& query = convertToQTensor(inputs.at(0), attrs, ctx);
    nvinfer1::ITensor& key = convertToKTensor(inputs.at(1), attrs, ctx);
    nvinfer1::ITensor& value = convertToVTensor(inputs.at(2), attrs, ctx);
    bool const hasAttnMask = !inputs.at(3).isNullTensor();
    nvinfer1::ITensor& normalizationQuantizeScale = convertToTensor(inputs.at(4), ctx);

    // Add the Attention layer.
    nvinfer1::IAttention* attention = N_CHECK(ctx->network()->addAttention(query, key, value, normOp, isCausal));
    ctx->registerAttention(attention, node);
    attention->setDecomposable(decomposable);
    attention->setNormalizationQuantizeToType(normQuantizeToType);
    attention->setNormalizationQuantizeScale(normalizationQuantizeScale);

    if (hasAttnMask)
    {
        attention->setMask(convertToMaskTensor(inputs.at(3), ctx));
    }

    return {{attention->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Add)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMax)
{
    return argMinMaxHelper(ctx, node, nodeIdx, inputs, nvinfer1::TopKOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMin)
{
    return argMinMaxHelper(ctx, node, nodeIdx, inputs, nvinfer1::TopKOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool)
{
    return poolingHelper(ctx, node, nodeIdx, inputs, nvinfer1::PoolingType::kAVERAGE);
}

NodeOutputs batchnormFallback(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx, std::vector<TensorOrWeights>& inputs)
{
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;

    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const rank = getNbDims(&input);

    std::array<nvinfer1::ITensor*, 4> tensors = {
        &convertToTensor(inputs.at(1), ctx),
        &convertToTensor(inputs.at(2), ctx),
        &convertToTensor(inputs.at(3), ctx),
        &convertToTensor(inputs.at(4), ctx),
    };

    // Alias tensors for convenience
    auto& [scale, bias, mean, variance] = tensors;

    // Reshape batchnorm weights from [C] to [N, C, ...] for elementwise operations.
    bool const needsExpandDims = rank > 1;
    if (needsExpandDims)
    {
        std::vector<int32_t> axes(rank - 1);
        axes[0] = 0;
        std::iota(axes.begin() + 1, axes.end(), 2);
        for (auto*& t : tensors)
        {
            t = unsqueezeTensor(ctx, *t, axes);
        }
    }

    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5F);

    auto varType = variance->getType();
    nvinfer1::IConstantLayer* epsLayer;
    if (varType == DataType::kHALF)
    {
        epsLayer
            = addConstantScalar(ctx, static_cast<half_float::half>(eps), ::ONNX_NAMESPACE::TensorProto::FLOAT16, rank);
    }
    else if (varType == DataType::kBF16)
    {
        epsLayer = addConstantScalar(ctx, static_cast<BFloat16>(eps), ::ONNX_NAMESPACE::TensorProto::BFLOAT16, rank);
    }
    else
    {
        epsLayer = addConstantScalar(ctx, eps, ::ONNX_NAMESPACE::TensorProto::FLOAT, rank);
    }
    nvinfer1::ITensor* epsilon = N_CHECK(epsLayer->getOutput(0));

    // For stronglyTyped networks, cast BatchNormalization parameters to the same type as the input type.
    if (ctx->isStronglyTyped())
    {
        LOG_VERBOSE("Casting BatchNormalization parameters to the same type as input for StronglyTyped networks.");
        for (auto*& t : tensors)
        {
            t = castHelper(ctx, t, input.getType());
        }
        epsilon = castHelper(ctx, epsilon, input.getType());
    }

    // batchnorm = scale * (input - mean) / sqrt(variance + epsilon) + bias
    // The WAR is split the single c++ code line into 3 to avoid the sequence swap by compiler.
    nvinfer1::ITensor* divisor
        = getUnaryResult(ctx, *getElementWiseResult(ctx, *variance, *epsilon, eOp::kSUM), uOp::kSQRT);
    nvinfer1::ITensor* dividend = getElementWiseResult(ctx, input, *mean, eOp::kSUB);
    auto intermediateResult
        = getElementWiseResult(ctx, *scale, *getElementWiseResult(ctx, *dividend, *divisor, eOp::kDIV), eOp::kPROD);
    nvinfer1::IElementWiseLayer* layer = N_CHECK(ctx->network()->addElementWise(*intermediateResult, *bias, eOp::kSUM));

    ctx->registerLayer(layer, node);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

template <typename T>
T const* type_convert(ImporterContext* ctx, ShapedWeights const& weights)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return ctx->getWeightsContext().getFP32Values(weights);
    }
    return static_cast<T const*>(weights.values);
}

template <typename T>
NodeOutputs batchnormWeightHelper(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx, std::vector<TensorOrWeights>& inputs)
{
    auto const scale = inputs.at(1).weights();
    auto const bias = inputs.at(2).weights();
    auto const mean = inputs.at(3).weights();
    auto const variance = inputs.at(4).weights();

    T const* scaleValues = type_convert<T>(ctx, scale);
    T const* biasValues = type_convert<T>(ctx, bias);
    T const* meanValues = type_convert<T>(ctx, mean);
    T const* varianceValues = type_convert<T>(ctx, variance);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    OnnxAttrs attrs(node, ctx);
    T eps = static_cast<T>(attrs.get<float>("epsilon", 1e-5f));

    // Fold the weights together into a single bias and scale
    int32_t const nbChannels = scale.shape.d[0];
    ShapedWeights::DataType weightType = typeid(T).hash_code() == typeid(BFloat16).hash_code()
        ? ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        : (typeid(T).hash_code() == typeid(half_float::half).hash_code() ? ::ONNX_NAMESPACE::TensorProto::FLOAT16
                                                                         : ::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto combinedScale = ctx->createNamedTempWeights(weightType, scale.shape, /*refittable=*/true);
    auto combinedBias = ctx->createNamedTempWeights(weightType, bias.shape, /*refittable=*/true);

    // Validate that all the weights have the same amount of values
    bool allSame = scale.count() == bias.count() && mean.count() == scale.count() && variance.count() == scale.count()
        && combinedScale.count() == scale.count() && combinedBias.count() == scale.count();
    ONNXTRT_CHECK_NODE(
        allSame, "Inputs to BatchNormalization must have the same shape!", node, nodeIdx, ErrorCode::kINVALID_NODE);

    for (int32_t i = 0; i < nbChannels; ++i)
    {
        combinedScale.at<T>(i) = scaleValues[i] / sqrtf(varianceValues[i] + eps);
        combinedBias.at<T>(i) = biasValues[i] - meanValues[i] * combinedScale.at<T>(i);
    }

    return scaleHelper(ctx, node, nodeIdx, *tensorPtr, nvinfer1::ScaleMode::kCHANNEL, combinedBias, combinedScale,
        ShapedWeights::empty(weightType), combinedBias.getName(), combinedScale.getName());
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
    ONNXTRT_CHECK_NODE((inputs.at(1).shape().nbDims == 1), "The shape of the scale input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((inputs.at(2).shape().nbDims == 1), "The shape of the bias input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((inputs.at(3).shape().nbDims == 1), "The shape of the mean input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((inputs.at(4).shape().nbDims == 1), "The shape of the var input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);

    bool const allInputsWeights = inputs.at(1).is_weights() && inputs.at(2).is_weights() && inputs.at(3).is_weights()
        && inputs.at(4).is_weights();

    // If any input is not an initializer, use fallback method implementing batchnorm as a combination of elementwise
    // layers.
    if (!allInputsWeights)
    {
        LOG_VERBOSE("Found BatchNormalization node with non-initializer inputs, using Elementwise fallback");
        return batchnormFallback(ctx, node, nodeIdx, inputs);
    }

    // If all inputs are weights of the same type, then combine the weights into a single scale layer.
    auto tensorType = inputs.at(0).getType();
    bool allWeightsSameType = tensorType == inputs.at(1).getType() && tensorType == inputs.at(2).getType()
        && tensorType == inputs.at(3).getType() && tensorType == inputs.at(4).getType();
    if (allWeightsSameType)
    {
        LOG_VERBOSE(
            "Found BatchNormalization node with conforming initializer types. Combining into a single scale node.");
        if (tensorType == "FLOAT")
        {
            return batchnormWeightHelper<float>(ctx, node, nodeIdx, inputs);
        }
        if (tensorType == "HALF")
        {
            return batchnormWeightHelper<half_float::half>(ctx, node, nodeIdx, inputs);
        }
        if (tensorType == "BF16")
        {
            return batchnormWeightHelper<BFloat16>(ctx, node, nodeIdx, inputs);
        }
        ONNXTRT_CHECK_NODE(false, "Invalid data type provided for BatchNormalization", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE_DATATYPE);
    }
    // With weights of different types, use fallback method to cast weights to the input type for stronglyTyped
    // networks.
    else if (ctx->isStronglyTyped())
    {
        return batchnormFallback(ctx, node, nodeIdx, inputs);
    }

    // For weakly-typed networks, cast everything to FP32 for consistency.
    LOG_VERBOSE(
        "Found BatchNormalization node with non-conforming initializer types. Casting parameters to FP32 and combining "
        "into a single scale node.");
    return batchnormWeightHelper<float>(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(BlackmanWindow)
{

    /***

    Operation returns a window vector, where

    Y[n] = 0.42 - 0.5cos(2pi*n / N) + 0.08cos(4pi*n / N)

    Where N is the window length, and n is each element in the window.

    Note that if `periodic == 0`, the denominator becomes N - 1.

    This can be represented by creating a range 'n' from 0 -> N, and performing the operations elementwise.

    ***/

    OnnxAttrs attrs(node, ctx);
    int32_t outputDtype = attrs.get<int32_t>("output_datatype", 1);
    int32_t periodic = attrs.get<int32_t>("periodic", 1);
    ONNXTRT_CHECK_NODE(outputDtype == 1, "Output must be float32-type!", node, nodeIdx, ErrorCode::kINVALID_NODE);

    constexpr float alpha = 0.42F;
    constexpr float beta = 0.5F;
    constexpr float gamma = 0.08F;

    auto* N = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE(
        N->getDimensions().nbDims == 0, "Window length must be a scalar!", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto* window = generateWindow(ctx, N);

    auto lhsCosOutput = windowHelper(ctx, 2.F * M_PI, window, N, nvinfer1::UnaryOperation::kCOS, periodic);

    auto betaTensor
        = N_CHECK(addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto betaLayer
        = N_CHECK(ctx->network()->addElementWise(*betaTensor, *lhsCosOutput, nvinfer1::ElementWiseOperation::kPROD));
    auto betaOutput = N_CHECK(betaLayer->getOutput(0));

    auto rhsCosOutput = windowHelper(ctx, 4.F * M_PI, window, N, nvinfer1::UnaryOperation::kCOS, periodic);
    auto gammaTensor
        = N_CHECK(addConstantScalar(ctx, gamma, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto gammaLayer
        = N_CHECK(ctx->network()->addElementWise(*gammaTensor, *rhsCosOutput, nvinfer1::ElementWiseOperation::kPROD));
    auto gammaOutput = N_CHECK(gammaLayer->getOutput(0));

    auto alphaTensor
        = N_CHECK(addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto alphaMinusBeta
        = N_CHECK(ctx->network()->addElementWise(*alphaTensor, *betaOutput, nvinfer1::ElementWiseOperation::kSUB));
    auto alphaMinusBetaTensor = N_CHECK(alphaMinusBeta->getOutput(0));

    auto plusGamma = N_CHECK(
        ctx->network()->addElementWise(*alphaMinusBetaTensor, *gammaOutput, nvinfer1::ElementWiseOperation::kSUM));
    RETURN_FIRST_OUTPUT(plusGamma, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Cast)
{
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    // Get data type to cast to. Ignore "saturate" attribute as TRT will reject casts to FP8.
    auto onnxType = attrs.get<int32_t>("to");
    DataType newType{DataType::kFLOAT};
    LOG_VERBOSE("Casting to type: " << newType);
    ONNXTRT_CHECK_NODE(convertDtype(onnxType, &newType), "Unsupported cast!", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Add the layer.
    nvinfer1::ICastLayer* layer = N_CHECK(ctx->network()->addCast(tensor, newType));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(CastLike)
{
    // Get input tensor to cast
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    // Get datatype to cast to, extracted from the second input tensor. Ignore "saturate" attribute as TRT will reject
    // casts to FP8.
    auto type = convertToTensor(inputs.at(1), ctx).getType();
    nvinfer1::ICastLayer* layer = N_CHECK(ctx->network()->addCast(tensor, type));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kCEIL);
}

DEFINE_BUILTIN_OP_IMPORTER(Celu)
{

    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    using eOpInstuctor = std::tuple<int, int, const nvinfer1::ElementWiseOperation>;

    ONNXTRT_CHECK_NODE((!inputs.empty()), "Inputs vector is empty.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    TensorOrWeights input = inputs.at(0);
    float alpha = attrs.get<float>("alpha", 1.0);

    TensorOrWeights weightsOfZero = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
    ShapedWeights weightsOfOnes = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
    std::vector<float> ones{1};
    std::memcpy(weightsOfOnes.values, ones.data(), weightsOfOnes.count() * sizeof(float));
    ShapedWeights weightsOfAlpha = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
    std::vector<float> alphas{alpha};
    std::memcpy(weightsOfAlpha.values, alphas.data(), weightsOfAlpha.count() * sizeof(float));

    // Variable name -> index in inputTensors
    // x -> 0
    // 0 -> 1
    // 1 -> 2
    // alpha -> 3
    std::vector<TensorOrWeights> newInputs{input, weightsOfZero, weightsOfOnes, weightsOfAlpha};

    std::vector<nvinfer1::ITensor*> inputTensors;
    int32_t maxNbDims = -1;
    for (auto i : newInputs)
    {
        maxNbDims = std::max(maxNbDims, i.shape().nbDims);
    }

    for (auto i : newInputs)
    {
        auto* tensor_ptr = &convertToTensor(i, ctx);

        // Broadcast all input tensors to size of maxNbDims
        broadcastTensor(ctx, tensor_ptr, maxNbDims);
        ONNXTRT_CHECK_NODE(tensor_ptr->getDimensions().nbDims == maxNbDims, "Failed to broadcast tensors elementwise!",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }

    // Calculate (x/alpha)
    std::vector<TensorOrWeights> tempInputs{newInputs[0], newInputs[3]};
    elementwiseCheck(tempInputs, eOp::kDIV, node, nodeIdx);
    nvinfer1::ITensor* combined = inputTensors.at(0);
    auto* divLayer = N_CHECK(ctx->network()->addElementWise(*combined, *inputTensors.at(3), eOp::kDIV));
    ctx->registerLayer(divLayer, node);
    combined = N_CHECK(divLayer->getOutput(0));

    // Calculate exp(x/alpha) -> 4
    nvinfer1::IUnaryLayer* uLayer = N_CHECK(ctx->network()->addUnary(*combined, uOp::kEXP));
    ctx->registerLayer(uLayer, node);
    combined = N_CHECK(uLayer->getOutput(0));
    inputTensors.push_back(combined);

    std::vector<eOpInstuctor> operations{
        // max(0,x) -> 5
        eOpInstuctor(0, 1, eOp::kMAX),
        // (exp(x/alpha)-1)) -> 6
        eOpInstuctor(4, 2, eOp::kSUB),
        // alpha*(exp(x/alpha)-1) -> 7
        eOpInstuctor(3, 6, eOp::kPROD),
        // min(0,alpha*(exp(x/alpha)-1)) -> 8
        eOpInstuctor(1, 7, eOp::kMIN),
        // max(0,x) + min(0,alpha*(exp(x/alpha)-1)) -> 9
        eOpInstuctor(5, 8, eOp::kSUM),
    };

    for (auto it : operations)
    {
        nvinfer1::ITensor* firstTensor = inputTensors.at(std::get<0>(it));
        nvinfer1::ITensor* secondTensor = inputTensors.at(std::get<1>(it));
        eOp const op = std::get<2>(it);
        tempInputs = {firstTensor, secondTensor};
        elementwiseCheck(tempInputs, op, node, nodeIdx);
        ONNXTRT_CHECK_NODE((firstTensor->getDimensions().nbDims == secondTensor->getDimensions().nbDims),
            "The rank of operands should be the same adding inputs. First tensor rank is "
                << firstTensor->getDimensions().nbDims << ", but second tensor rank is "
                << secondTensor->getDimensions().nbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = N_CHECK(ctx->network()->addElementWise(*firstTensor, *secondTensor, op));
        ctx->registerLayer(layer, node);
        inputTensors.push_back(N_CHECK(layer->getOutput(0)));
    }
    return {{inputTensors.back()}};
}

// Helper function to perform clip through elementwise operations
template <typename ScalarType>
NodeOutputs elementwiseClipHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, size_t numInputs, int32_t onnxType)
{
    OnnxAttrs attrs(node, ctx);
    auto* input = &convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = getNbDims(input);
    nvinfer1::ITensor* alphaT{nullptr};
    nvinfer1::ITensor* betaT{nullptr};
    ScalarType alpha = std::numeric_limits<ScalarType>::lowest();
    ScalarType beta = std::numeric_limits<ScalarType>::max();
    if (numInputs == 1)
    {
        alphaT = N_CHECK(addConstantScalar(ctx, alpha, onnxType, nbDims)->getOutput(0));
        betaT = N_CHECK(addConstantScalar(ctx, beta, onnxType, nbDims)->getOutput(0));
    }
    else if (numInputs == 2)
    {
        alphaT = &convertToTensor(inputs.at(1), ctx);
        broadcastTensors(ctx, input, alphaT);
        betaT = N_CHECK(addConstantScalar(ctx, beta, onnxType, nbDims)->getOutput(0));
    }
    else if (numInputs == 3)
    {
        // "min" can be optional if "max" is specified. Check for this case here
        if (!inputs.at(1).isNullTensor())
        {
            alphaT = &convertToTensor(inputs.at(1), ctx);
            broadcastTensors(ctx, input, alphaT);
        }
        else
        {
            alphaT = N_CHECK(addConstantScalar(ctx, alpha, onnxType, nbDims)->getOutput(0));
        }
        if (!inputs.at(2).isNullTensor())
        {
            betaT = &convertToTensor(inputs.at(2), ctx);
            broadcastTensors(ctx, input, betaT);
        }
        else
        {
            betaT = N_CHECK(addConstantScalar(ctx, beta, onnxType, nbDims)->getOutput(0));
        }
    }

    // Now that we have alphaT and betaT, do the elementwise calculation
    using eOp = nvinfer1::ElementWiseOperation;
    auto* lowerClipLayer = N_CHECK(ctx->network()->addElementWise(*input, *alphaT, eOp::kMAX));
    auto* lowerClip = N_CHECK(lowerClipLayer->getOutput(0));
    auto* upperClipLayer = N_CHECK(ctx->network()->addElementWise(*lowerClip, *betaT, eOp::kMIN));
    auto* upperClip = N_CHECK(upperClipLayer->getOutput(0));
    return {{upperClip}};
}

DEFINE_BUILTIN_OP_IMPORTER(Clip)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    // For INT32 and multi-input clips, use elementwise operators instead.
    size_t numInputs = inputs.size();
    bool elementwiseClip = inputs.at(0).isInt32() || inputs.at(0).isInt64();
    for (size_t i = 1; i < numInputs; i++)
    {
        elementwiseClip |= inputs.at(i).is_tensor();
    }
    if (elementwiseClip)
    {
        auto type = convertToTensor(inputs.at(0), ctx).getType();
        ONNXTRT_CHECK_NODE((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16
                               || type == DataType::kINT32 || type == DataType::kINT64),
            "This version of TensorRT only supports floating-point, INT32, or INT64 inputs for Clip! The current input "
            "type is "
                + getTrtDtypeName(type) + ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        if (type == DataType::kHALF)
        {
            return elementwiseClipHelper<half_float::half>(
                ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::FLOAT16);
        }
        if (type == DataType::kBF16)
        {
            return elementwiseClipHelper<BFloat16>(
                ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::BFLOAT16);
        }
        if (type == DataType::kFLOAT)
        {
            return elementwiseClipHelper<float>(ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::FLOAT);
        }
        if (type == DataType::kINT64)
        {
            return elementwiseClipHelper<int64_t>(ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::INT64);
        }
        return elementwiseClipHelper<int32_t>(ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::INT32);
    }

    // Activation path only supports float/half initializers
    OnnxAttrs attrs(node, ctx);
    // beta is the upper bound
    float alpha = std::numeric_limits<float>::lowest();
    float beta = std::numeric_limits<float>::max();

    if (ctx->getOpsetVersion() >= 11)
    {
        // Handle "min" node input.
        if (numInputs == 2)
        {
            ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "Clip min value must be an initializer!", node, nodeIdx,
                ErrorCode::kINVALID_NODE);
            auto min = inputs.at(1).weights();
            alpha = getSingleValueAsFloat(min);
        }
        // Handle both "min" and "max" node inputs
        else if (numInputs == 3)
        {
            // "min" can be optional if "max" is specified. Check for this case here
            if (!inputs.at(1).isNullTensor())
            {
                ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "Clip min value must be an initializer!", node, nodeIdx,
                    ErrorCode::kINVALID_NODE);
                auto min = inputs.at(1).weights();
                alpha = getSingleValueAsFloat(min);
            }

            if (!inputs.at(2).isNullTensor())
            {
                ONNXTRT_CHECK_NODE(inputs.at(2).is_weights(), "Clip max value must be an initializer!", node, nodeIdx,
                    ErrorCode::kINVALID_NODE);
                auto max = inputs.at(2).weights();
                beta = getSingleValueAsFloat(max);
            }
        }
    }
    else
    {
        alpha = attrs.get("min", std::numeric_limits<float>::lowest());
        beta = attrs.get("max", std::numeric_limits<float>::max());
    }

    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kCLIP, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Concat)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        auto* tensorPtr = &convertToTensor(input, ctx);
        tensors.push_back(tensorPtr);
    }
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis");
    int32_t nbDims = inputs.at(0).shape().nbDims;
    convertAxis(axis, nbDims, node, nodeIdx);
    auto* layer = N_CHECK(ctx->network()->addConcatenation(tensors.data(), tensors.size()));
    ctx->registerLayer(layer, node);
    layer->setAxis(axis);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Constant)
{
    OnnxAttrs attrs(node, ctx);

    // Having the trt_outputs_range_min attributes means it's from
    // serialized iNetworkDefinition.
    if (!attrs.get<std::vector<float>>("trt_outputs_range_min", {}).empty())
    {
        // just create a constant layer here for 1-1 mapping during network deserialization
        auto weights = attrs.get<ShapedWeights>("value");
        auto* layer = N_CHECK(ctx->network()->addConstant(weights.shape, weights));
        ctx->network()->setWeightsName(weights, weights.getName());
        RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
    }

    if (ctx->getOpsetVersion() >= 12)
    {
        if (attrs.count("value_float"))
        {
            ShapedWeights convertedWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
            float value = attrs.get<float>("value_float");
            std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(float));
            return {{convertedWeights}};
        }

        if (attrs.count("value_floats"))
        {
            std::vector<float> values = attrs.get<std::vector<float>>("value_floats");
            int32_t valueSize = values.size();
            ShapedWeights convertedWeights
                = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {1, {valueSize}});
            std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(float));
            return {{convertedWeights}};
        }
        if (attrs.count("value_int"))
        {
            ShapedWeights convertedWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {0, {}});
            int64_t value = attrs.get<int64_t>("value_int");
            std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(int64_t));
            return {{convertedWeights}};
        }

        if (attrs.count("value_ints"))
        {
            std::vector<int64_t> values = attrs.get<std::vector<int64_t>>("value_ints");
            int32_t valueSize = values.size();
            ShapedWeights convertedWeights
                = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {1, {valueSize}});
            std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(int64_t));
            return {{convertedWeights}};
        }
    }
    attrs.get<ShapedWeights>("value");

    return {{attrs.get<ShapedWeights>("value")}};
}

DEFINE_BUILTIN_OP_IMPORTER(ConstantOfShape)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* shape = &convertToTensor(inputs.at(0), ctx);

    // For refit, create named temp weights for ConstantOfShape value tensors since they're not named by default.
    ShapedWeights zeroWeights
        = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, {1}},  /*refittable=*/ true);
    static_cast<float*>(zeroWeights.values)[0] = 0.f;
    auto valueWeights = TensorOrWeights{attrs.get<ShapedWeights>("value", zeroWeights)};
    valueWeights.setName(zeroWeights.getName());
    nvinfer1::ITensor* value = &convertToTensor(valueWeights, ctx);
    return {{constantOfShape(ctx, value, shape)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv)
{
    if (inputs.at(1).is_tensor() || (inputs.size() > 2 && inputs.at(2).is_tensor()))
    {
        // Handle dynamic weights convolution
        return convMultiInput(ctx, node, nodeIdx, inputs);
    }

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    auto kernelWeights = inputs.at(1).weights();

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    LOG_VERBOSE("Convolution input dimensions: " << dims);
    ONNXTRT_CHECK_NODE(dims.nbDims >= 0, "TensorRT could not compute output dimensions of Conv", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    bool const needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> axes{3};
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK_NODE(tensorPtr, "Failed to unsqueeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }
    if (kernelWeights.shape.nbDims == 3)
    {
        kernelWeights.shape.nbDims = 4;
        kernelWeights.shape.d[3] = 1;
    }

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ONNXTRT_CHECK_NODE((nbSpatialDims == kernelWeights.shape.nbDims - 2),
        "The number of spatial dimensions and the kernel shape doesn't match up for the Conv operator. Number of "
        "spatial dimensions = "
            << nbSpatialDims << ", number of kernel dimensions = " << kernelWeights.shape.nbDims << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Weights biasWeights;
    if (inputs.size() == 3)
    {
        assertIsWeights(inputs.at(2), "The bias tensor is required to be an initializer for the Conv operator.");
        auto shapedBiasWeights = inputs.at(2).weights();
        // Unsqueeze scalar weights to 1D
        if (shapedBiasWeights.shape.nbDims == 0)
        {
            shapedBiasWeights.shape = {1, {1}};
        }
        ONNXTRT_CHECK_NODE((shapedBiasWeights.shape.nbDims == 1), "The bias tensor is required to be 1D.", node,
            nodeIdx, ErrorCode::kINVALID_NODE);
        ONNXTRT_CHECK_NODE((shapedBiasWeights.shape.d[0] == kernelWeights.shape.d[0]),
            "The shape of the bias tensor misaligns with the weight tensor. Shape of bias weights = "
                << shapedBiasWeights.shape.d[0] << ", shape of kernel weights = " << kernelWeights.shape.d[0] << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        biasWeights = shapedBiasWeights;
    }
    else
    {
        biasWeights = ShapedWeights::empty(kernelWeights.type);
    }
    nvinfer1::Dims kernelSize;
    kernelSize.nbDims = nbSpatialDims;
    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        kernelSize.d[nbSpatialDims - i] = kernelWeights.shape.d[kernelWeights.shape.nbDims - i];
    }
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool excludePadding;
    getKernelParams(
        ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, excludePadding, &dilations);

    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        ONNXTRT_CHECK_NODE((kernelSize.d[nbSpatialDims - i] == kernelWeights.shape.d[kernelWeights.shape.nbDims - i]),
            "The size of spatial dimension and the size of kernel shape are not equal for the Conv operator. "
            "Size of spatial dimensions = "
                << kernelSize.d[nbSpatialDims - i]
                << ", size of kernel dimensions = " << kernelWeights.shape.d[kernelWeights.shape.nbDims - i] << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    int32_t nchan = dims.d[1];
    int32_t noutput = kernelWeights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = N_CHECK(ctx->network()->addConvolutionNd(*tensorPtr, noutput, kernelSize, kernelWeights, biasWeights));

    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int32_t ngroup = attrs.get("group", 1);
    ONNXTRT_CHECK_NODE((nchan == -1 || kernelWeights.shape.d[1] * ngroup == nchan),
        "Kernel weight dimension failed to broadcast to input.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    // Register layer name as well as kernel weights and bias weights (if any)
    ctx->registerLayer(layer, node);
    ctx->network()->setWeightsName(kernelWeights, inputs.at(1).weights().getName());
    if (inputs.size() == 3)
    {
        ctx->network()->setWeightsName(biasWeights, inputs.at(2).weights().getName());
    }
    tensorPtr = N_CHECK(layer->getOutput(0));
    dims = tensorPtr->getDimensions();

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> axes{3};
        tensorPtr = squeezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK_NODE(tensorPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    LOG_VERBOSE("Using kernel: " << kernelSize << ", strides: " << strides << ", prepadding: " << begPadding
                                 << ", postpadding: " << endPadding << ", dilations: " << dilations
                                 << ", numOutputs: " << noutput << ", nbGroups: " << ngroup);
    LOG_VERBOSE("Convolution output dimensions: " << dims);

    return {{tensorPtr}};
}

// TRT only supports 2D or 3D deconvolutions (Layout: [N,C,D1,D2,(D3)])
// Inputs should be of dimension 4 or 5.
// When input.nbDims = 3, we expand it to 4D
DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose)
{
    // Expand spatial dims from 1D to 2D, return true if reshaped activation
    auto const NCWtoNCHW = [&ctx](nvinfer1::ITensor*& tensor, nvinfer1::Dims& tensorShape) {
        if (tensor && tensor->getDimensions().nbDims == 3)
        {
            std::vector<int32_t> const axes{3};
            tensor = unsqueezeTensor(ctx, *tensor, axes);
            tensorShape = tensor->getDimensions();
            return true;
        }
        // for initializer, just change the shape by appending 1
        if (tensorShape.nbDims == 3)
        {
            tensorShape.nbDims = 4;
            tensorShape.d[3] = 1;
        }
        return false;
    };

    ONNXTRT_CHECK_NODE(
        inputs.size() >= 2, "deconvolution require at least 2 inputs.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    auto inputType = tensorPtr->getType();
    nvinfer1::ITensor* kernelTensorPtr = inputs.at(1).is_tensor() ? &convertToTensor(inputs.at(1), ctx) : nullptr;

    bool const validBias = inputs.size() > 2 && !inputs.at(2).isNullTensor();
    nvinfer1::ITensor* biasTensorPtr
        = validBias && inputs.at(2).is_tensor() ? &convertToTensor(inputs.at(2), ctx) : nullptr;

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    // Deconvolution input must be at least 3D and at most 5D.
    ONNXTRT_CHECK_NODE(dims.nbDims >= 3 && dims.nbDims <= 5,
        "Deconvolution input must be at least 3D and at most 5D! The current input is rank " << dims.nbDims << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernelShape = inputs.at(1).shape();

    bool needReshapeBack = NCWtoNCHW(tensorPtr, dims);
    NCWtoNCHW(kernelTensorPtr, kernelShape);

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ONNXTRT_CHECK_NODE((nbSpatialDims == kernelShape.nbDims - 2),
        "The number of spatial dimensions and the kernel shape doesn't match up. Number of spatial dimensions = "
            << nbSpatialDims << ", number of kernel dimensions = " << kernelShape.nbDims << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Get all attributes
    OnnxAttrs attrs(node, ctx);
    nvinfer1::Dims outputShape;
    nvinfer1::Dims outputPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims kernelSize;
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool excludePadding = false;

    int32_t ngroup = attrs.get("group", 1);
    int32_t noutput = kernelShape.d[1] * ngroup; // Note: Weights order is CKRS

    // Get static bias weights
    nvinfer1::Weights staticBiasWeights;
    if (validBias && biasTensorPtr == nullptr)
    {
        auto shapedBiasWeights = inputs.at(2).weights();
        // ONNX requires shapedBiasWeights to be 1D
        ONNXTRT_CHECK_NODE(shapedBiasWeights.shape.nbDims == 1,
            "The bias tensor is required to be 1D. Provided bias has rank " << shapedBiasWeights.shape.nbDims << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        ONNXTRT_CHECK_NODE((shapedBiasWeights.shape.d[0] == noutput),
            "The number of the bias weights does not align with the number of output maps. Number of bias weights = "
                << shapedBiasWeights.shape.d[0] << ", number of output maps = " << noutput << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        staticBiasWeights = shapedBiasWeights;
    }
    else
    {
        staticBiasWeights = ShapedWeights::empty(trtDataTypeToONNX(inputType));
    }

    // Kernel shape either comes from the attributes or extracted from the kernel weights shape
    kernelSize.nbDims = nbSpatialDims;
    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        kernelSize.d[nbSpatialDims - i] = kernelShape.d[kernelShape.nbDims - i];
    }

    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, excludePadding, &dilations,
        &outputPadding);

    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        ONNXTRT_CHECK_NODE((kernelSize.d[nbSpatialDims - i] == kernelShape.d[kernelShape.nbDims - i]),
            "Attribute kernel_shape misaligns with the dimensions of the weight tensor. Number of spatial dimensions = "
                << kernelSize.d[nbSpatialDims - i]
                << ", number of kernel dimensions = " << kernelShape.d[kernelShape.nbDims - i] << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    // Set padding. ONNX ConvTranspose supports many different padding modes. Order of priority for padding:
    // 1. Output shape is specified - calculate expected pre and post padding.
    // 2. AUTO_PAD != NOTSET: ignore all other padding values and set padding mode with layer->setPaddingMode.
    //    Pad the resulting output vector with values from output_padding.
    // 3. Use specified "pads" values from the node. Pad the resulting output vector with values from output_padding.

    auto autoPadMode = attrs.get("auto_pad", std::string("NOTSET"));
    if (attrs.count("output_shape") && autoPadMode == std::string("NOTSET"))
    {
        outputShape = attrs.get<nvinfer1::Dims>("output_shape");

        // This function takes references to begPadding, endPadding and outputPadding and will update them with correct
        // values.
        generatePadding(dims, outputShape, kernelSize, strides, dilations, nbSpatialDims, begPadding, endPadding,
            outputPadding, paddingMode);

        // NOTE: it is possible for generatePadding to produce negative values for pre and post padding, which usually
        // happens when output_shape is provided but output_padding is not. Any negative values generated for
        // post-padding can be translated into outputPadding to pad the output tensor post deconvolution. Any negative
        // values for pre-padding are unsupported.

        for (int32_t i = 0; i < nbSpatialDims; i++)
        {
            ONNXTRT_CHECK_NODE(begPadding.d[i] >= 0,
                "TensorRT does not support negative pre-padding in the ConvTranspose operator!", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
            // Update outputPadding with any negative values in endPadding, and set the corresponding value to 0.
            if (endPadding.d[i] < 0)
            {
                outputPadding.d[i] = endPadding.d[i] * -1;
                endPadding.d[i] = 0;
            }
        }
    }

    // When there is output_padding, if postPadding is larger than outputPadding, just adjust postPadding
    // Or reduce outputPadding as minimum as possible.
    bool hasOutputPadding = false;
    if (outputPadding != makeDims(nbSpatialDims, 0) && autoPadMode == std::string("NOTSET"))
    {
        for (int32_t i = 0; i < nbSpatialDims; ++i)
        {
            if (endPadding.d[i] - outputPadding.d[i] >= 0)
            {
                endPadding.d[i] -= outputPadding.d[i];
                outputPadding.d[i] = 0;
            }
            else
            {
                // Reduce outputPadding as possible.
                outputPadding.d[i] -= endPadding.d[i];
                endPadding.d[i] = 0;
                hasOutputPadding = true;
            }
        }
    }

    auto const emptyBiasWeights = ShapedWeights::empty(trtDataTypeToONNX(inputType));
    auto const kernelWeights = kernelTensorPtr ? nvinfer1::Weights{inputType, nullptr, 0} : inputs.at(1).weights();
    auto const biasWeights = biasTensorPtr ? nvinfer1::Weights{inputType, nullptr, 0} : staticBiasWeights;
    // Create a deconvolution layer and set known attributes - strides,ngroups, and dilations
    // If there is still output padding, remove the bias weights. Bias will be added below.
    auto* layer = N_CHECK(ctx->network()->addDeconvolutionNd(
        *tensorPtr, noutput, kernelSize, kernelWeights, hasOutputPadding ? emptyBiasWeights : biasWeights));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setNbGroups(ngroup);
    layer->setDilationNd(dilations);
    if (kernelTensorPtr)
    {
        layer->setInput(1, *kernelTensorPtr);
    }
    else
    {
        ctx->network()->setWeightsName(kernelWeights, inputs.at(1).weights().getName());
    }
    if (biasTensorPtr)
    {
        layer->setInput(2, *biasTensorPtr);
    }

    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    LOG_VERBOSE("Running deconvolution with: " << "\n"
                                               << "Padding mode: " << autoPadMode << "\n"
                                               << "Pre-padding: " << begPadding << "\n"
                                               << "Post-padding: " << endPadding);

    // Register layer, along with refittable kernel weights and bias weights (if any)
    ctx->registerLayer(layer, node);
    tensorPtr = N_CHECK(layer->getOutput(0));
    dims = tensorPtr->getDimensions();

    // There is still output padding. Add a padding layer to handle it.
    if (hasOutputPadding)
    {
        LOG_VERBOSE("Padding output deconvolution tensor with: " << outputPadding);

        // Add padding layer
        nvinfer1::ITensor* start{};
        nvinfer1::ITensor* totalPadding{};
        std::vector<int64_t> combinePadding{};
        for (int32_t i = 0; i < outputPadding.nbDims; ++i)
        {
            combinePadding.insert(combinePadding.begin(), 0);
            combinePadding.push_back(outputPadding.d[i]);
        }
        ONNXTRT_CHECK_NODE(convertOnnxPadding(ctx, dims.nbDims, combinePadding, start, totalPadding),
            "Failed to convert padding!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        auto const size = getElementWiseResult(
            ctx, shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM);
        auto const stride = makeDims(dims.nbDims, 1);
        auto const& dummy = stride;
        auto* sliceLayer = N_CHECK(ctx->network()->addSlice(*tensorPtr, dummy, dummy, stride));
        sliceLayer->setInput(1, *start);
        sliceLayer->setInput(2, *size);
        sliceLayer->setMode(nvinfer1::SampleMode::kFILL);
        tensorPtr = N_CHECK(sliceLayer->getOutput(0));

        // This bias is not handled by deconv. Use an elementwise to handle it.
        if (biasWeights.count != 0)
        {
            // Set C dimension to weights count and set other dimensions to 1 to enable broadcast
            auto constantDims = makeDims(dims.nbDims, 1);
            constantDims.d[dims.nbDims - nbSpatialDims - 1] = biasWeights.count;
            auto biasConstant = N_CHECK(ctx->network()->addConstant(constantDims, biasWeights));
            tensorPtr = getElementWiseResult(
                ctx, *tensorPtr, *N_CHECK(biasConstant->getOutput(0)), nvinfer1::ElementWiseOperation::kSUM);
        }
    }

    if (validBias && biasTensorPtr == nullptr)
    {
        ctx->network()->setWeightsName(biasWeights, inputs.at(2).weights().getName());
    }

    if (needReshapeBack)
    {
        std::vector<int32_t> axes{3};
        tensorPtr = squeezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK_NODE(tensorPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Cos)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kCOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Cosh)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kCOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(CumSum)
{
    OnnxAttrs attrs(node, ctx);
    int32_t const exclusive = attrs.get<int32_t>("exclusive", 0);
    int32_t const reverse = attrs.get<int32_t>("reverse", 0);

    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* axis = &convertToTensor(inputs.at(1), ctx);

    // TensorRT requires axis to be a scalar.
    if (axis->getDimensions().nbDims == 1)
    {
        std::vector<int32_t> const squeezeAxes{0};
        axis = squeezeTensor(ctx, *axis, squeezeAxes);
    }

    auto layer = N_CHECK(ctx->network()->addCumulative(*input, *axis, nvinfer1::CumulativeOperation::kSUM, exclusive, reverse));
    ctx->registerLayer(layer, node);

    layer->setOperation(nvinfer1::CumulativeOperation::kSUM);
    layer->setExclusive(exclusive);
    layer->setReverse(reverse);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(DeformConv)
{
    auto inputDataType = inputs.at(0).getDataType();
    auto weightDataType = inputs.at(1).getDataType();
    auto offsetDataType = inputs.at(2).getDataType();

    ONNXTRT_CHECK_NODE((inputDataType == DataType::kFLOAT || inputDataType == DataType::kHALF),
        "Inputs must be either FLOAT or FLOAT16. Input type is " + getTrtDtypeName(inputDataType) + ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);

    ONNXTRT_CHECK_NODE((inputDataType == weightDataType && inputDataType == offsetDataType),
        "Inputs must be either all FLOAT or all FLOAT16. Input type = " + getTrtDtypeName(inputDataType)
            + ", weight type = " + getTrtDtypeName(weightDataType)
            + ", offset type = " + getTrtDtypeName(offsetDataType) + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    if (inputs.size() > 3)
    {
        auto biasDataType = inputs.at(3).getDataType();
        ONNXTRT_CHECK_NODE((inputDataType == biasDataType),
            "Inputs must be either all FLOAT or all FLOAT16. Input type = " + getTrtDtypeName(inputDataType)
                + ", bias type = " + getTrtDtypeName(biasDataType) + ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
    }

    if (inputs.size() > 4)
    {
        auto maskDataType = inputs.at(4).getDataType();
        ONNXTRT_CHECK_NODE((inputDataType == maskDataType),
            "Inputs must be either all FLOAT or all FLOAT16. Input type = " + getTrtDtypeName(inputDataType)
                + ", mask type = " + getTrtDtypeName(maskDataType) + ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
    }

    return modulatedDeformableConvPluginHelper(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    checkNotInvalidType(inputs.at(0), {"BOOL", "UINT8"}, node, nodeIdx);
    // Input tensor is in NCHW format
    ONNXTRT_CHECK_NODE((inputs.at(0).shape().nbDims == 4), "The input tensor must be in NCHW format.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    // Extract attributes
    OnnxAttrs attrs(node, ctx);
    auto blockSize = attrs.get<int>("blocksize");
    auto mode = attrs.get<std::string>("mode", "DCR");

    // Useful constants
    auto const inputShape = shapeOf(*tensorPtr);
    auto const N = gather(ctx, inputShape, shapeVector(0));
    auto const C = gather(ctx, inputShape, shapeVector(1));
    auto const H = gather(ctx, inputShape, shapeVector(2));
    auto const W = gather(ctx, inputShape, shapeVector(3));
    auto const blockSizeTensor = shapeVector(blockSize);
    auto const C_2 = floorDiv(ctx, C, mul(ctx, blockSizeTensor, blockSizeTensor));
    auto const H_2 = mul(ctx, H, blockSizeTensor);
    auto const W_2 = mul(ctx, W, blockSizeTensor);
    int32_t const DCRPerm[6] = {0, 3, 4, 1, 5, 2};
    int32_t const CRDPerm[6] = {0, 1, 4, 2, 5, 3};

    ShapeTensor firstShape;
    nvinfer1::Permutation perm{};

    if (mode == "DCR")
    {
        // First reshape to {N, blockSize, blockSize, C / (blockSize * blockSize), H, W}
        firstShape = concat(
            ctx, N, concat(ctx, blockSizeTensor, concat(ctx, blockSizeTensor, concat(ctx, C_2, concat(ctx, H, W)))));
        std::copy(std::begin(DCRPerm), std::end(DCRPerm), std::begin(perm.order));
    }
    else
    {
        // First reshape to {N, C / (blockSize * blockSize), blockSize, blockSize, H, W}
        firstShape = concat(
            ctx, N, concat(ctx, C_2, concat(ctx, blockSizeTensor, concat(ctx, blockSizeTensor, concat(ctx, H, W)))));
        std::copy(std::begin(CRDPerm), std::end(CRDPerm), std::begin(perm.order));
    }

    auto* firstShuffle = addShuffle(ctx, *tensorPtr, firstShape);
    firstShuffle->setSecondTranspose(perm);
    ctx->registerLayer(firstShuffle, node);
    tensorPtr = N_CHECK(firstShuffle->getOutput(0));

    // Finally reshape to {N, C / (blockSize * blockSize), H * blockSize, W * blockSize};
    auto secondShape = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShape);
    tensorPtr = N_CHECK(secondShuffle->getOutput(0));

    return {{tensorPtr}};
}


// Backward traverse the graph to retrieve the input weights from the constant node. We allow skipping all cast/identity
// nodes until reaching the constant node.
ShapedWeights getWeightsFromIdentityOrConstant(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* input)
{
    // Const node output -> const node mapping.
    std::unordered_map<nvinfer1::ITensor*, nvinfer1::IConstantLayer*> constNodeToOutputMap;
    // Identity node output -> identity/cast node mapping.
    std::unordered_map<nvinfer1::ITensor*, nvinfer1::ILayer*> identityCastNodeToOutputMap;

    // Collect all the constant, identity nodes from network.
    int32_t nbLayers = network.getNbLayers();
    for (int32_t i = 0; i < nbLayers; ++i)
    {
        nvinfer1::ILayer* layer = N_CHECK(network.getLayer(i));
        if (layer->getType() == nvinfer1::LayerType::kCONSTANT)
        {
            constNodeToOutputMap[layer->getOutput(0)] = static_cast<nvinfer1::IConstantLayer*>(layer);
        }
        else if ((layer->getType() == nvinfer1::LayerType::kIDENTITY)
            || (layer->getType() == nvinfer1::LayerType::kCAST))
        {
            identityCastNodeToOutputMap[layer->getOutput(0)] = layer;
        }
    }
    // Skip all the cast/identity nodes before current node.
    auto findIdenityIter = identityCastNodeToOutputMap.find(input);
    while (findIdenityIter != identityCastNodeToOutputMap.end())
    {
        input = findIdenityIter->second->getInput(0);
        findIdenityIter = identityCastNodeToOutputMap.find(input);
    }
    // Find out the weights from constant node.
    auto findConstIter = constNodeToOutputMap.find(input);
    if (findConstIter != constNodeToOutputMap.end())
    {
        auto weights = findConstIter->second->getWeights();
        return ShapedWeights(
            trtDataTypeToONNX(weights.type), const_cast<void*>(weights.values), findConstIter->second->getDimensions());
    }
    // Return empty weights when not found.
    return ShapedWeights{};
}

// This is a helper function for QuantizeLinear/DequantizeLinear
NodeOutputs QuantDequantLinearHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    std::vector<TensorOrWeights>& inputs, bool isDQ, bool isCustomOp, DataType customOpType = DataType::kFP8,
    bool isMX = false)
{
    auto getFlagBit = [](nvonnxparser::OnnxParserFlag const flag) { return 1U << static_cast<uint32_t>(flag); };
    bool const enableUInt8AsymmetricQuantization
        = ctx->getFlags() & getFlagBit(nvonnxparser::OnnxParserFlag::kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA);

    if (isDQ && !enableUInt8AsymmetricQuantization)
    {
        checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    }

    auto addConstantLayer = [ctx, node](nvinfer1::INetworkDefinition& network, ShapedWeights const& weights,
                                bool const isE8M0 = false) -> nvinfer1::ITensor* {
        nvinfer1::Weights convertedWeights = weights;
        if (isE8M0)
        {
            convertedWeights.type = nvinfer1::DataType::kE8M0;
        }
        nvinfer1::IConstantLayer* constLayer = N_CHECK(network.addConstant(weights.shape, convertedWeights));
        ctx->registerLayer(constLayer, weights.getName(), &node);
        network.setWeightsName(convertedWeights, weights.getName());
        return N_CHECK(constLayer->getOutput(0));
    };

    auto newConstantInput = [&](int32_t i) {
        return inputs.at(i).is_weights() && (ctx->getConstantLayer(inputs.at(i).weights().getName()) == nullptr);
    };

    // Quantization scales must be provided.
    ONNXTRT_CHECK_NODE((inputs.size() >= 2),
        "This version of TensorRT requires at least 2 inputs for the QuantizeLinear/DequantizeLinear operator.", node,
        nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);

    std::string nodeName = getNodeName(node);
    // Input 0 is the data to quantize or dequantize.
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);

    // Input 1 initializes the layer's scale weights.
    nvinfer1::ITensor* scaleInput = nullptr;
    if (newConstantInput(1))
    {
        // Scale is concrete so verify it now.
        auto scale = inputs.at(1).weights();
        ONNXTRT_CHECK_NODE(
            scale.count() > 0, "Cannot have scale with no coefficients.", node, nodeIdx, ErrorCode::kINVALID_NODE);

        bool scaleAllPositive = false;
        bool isE8M0 = false;
        if (inputs.at(1).isFp32())
        {
            auto const* scaleVal = static_cast<float const*>(scale.values);
            scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](float x) { return x > 0; });
        }
        else if (inputs.at(1).isFp16())
        {
            auto const* scaleVal = static_cast<half_float::half const*>(scale.values);
            scaleAllPositive
                = std::all_of(scaleVal, scaleVal + scale.count(), [](half_float::half x) { return x > 0; });
        }
        else if (inputs.at(1).isBFp16())
        {
            auto const* scaleVal = static_cast<BFloat16 const*>(scale.values);
            scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](BFloat16 x) { return x > 0; });
        }
        else if (inputs.at(1).isUint8() && isCustomOp && isMX)
        {
            scaleAllPositive = true;
            isE8M0 = true;
        }

        bool allowNegativeScale = false;

        ONNXTRT_CHECK_NODE(
            (scaleAllPositive || allowNegativeScale), "Scale coefficients must all be positive", node, nodeIdx,
            ErrorCode::kINVALID_NODE);

        // If the scale is concrete weights, then add a ConstantLayer that will be an input which
        // will initialize the scale weights.
        scaleInput = addConstantLayer(*ctx->network(), scale, isE8M0);
    }
    else
    {
        scaleInput = &convertToTensor(inputs.at(1), ctx);
    }
    auto const& inputDims = dataInput->getDimensions();
    auto const& scaleDims = scaleInput->getDimensions();
    auto const& inputType = dataInput->getType();
    auto const& scaleType = scaleInput->getType();

    auto const& scaleSize = isDynamic(scaleDims) ? 0 : volume(scaleDims);

    // Input 2 initializes the layer's zero-point.
    nvinfer1::ITensor* zeroPointInput = nullptr;
    // ONNX default is UINT8, TRT will default to INT8 as TRT allows UINT8 quantization only on DLA.
    // When importing CustomOp FP8/INT4 Q/DQ, default to FP8/INT4
    DataType chosenDataType = isCustomOp ? customOpType : DataType::kINT8;
    ONNXTRT_CHECK_NODE(!isCustomOp || customOpType == DataType::kFP8 || customOpType == DataType::kINT4,
        "Custom QDQ ops are available only for FP8 and INT4", node, nodeIdx, ErrorCode::kINTERNAL_ERROR);

    OnnxAttrs attrs(node, ctx);
    DataType outputDtype;
    auto const outputDTypeOnnx = attrs.get<int32_t>("output_dtype", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    bool isOutputDtypeSet = (outputDTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isOutputDtypeSet)
    {
        isOutputDtypeSet = convertDtype(outputDTypeOnnx, &outputDtype);
        ONNXTRT_CHECK_NODE(isOutputDtypeSet,
            "Attribute output_dtype specifies an unsupported data type " << outputDtype << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    DataType precision;
    auto const precisionOnnx = attrs.get<int32_t>("precision", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    bool isPrecisionSet = (precisionOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isPrecisionSet)
    {
        ONNXTRT_CHECK_NODE(!isDQ,
            "Attribute precision can only be used with QuantizeLienar. For DequantizeLinear output_dtype determines "
            "precision.",
            node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        isPrecisionSet = convertDtype(precisionOnnx, &precision);
        ONNXTRT_CHECK_NODE(isPrecisionSet,
            "Attribute precision specifies an unsupported data type " << precision << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(
            precision == DataType::kFLOAT || precision == DataType::kHALF || precision == DataType::kBF16,
            "Attribute precision specifies an invalid data type for QuantizeLinear " << precision << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);

        DataType trtPrecisionType = isMX ? inputType : scaleType;
        if (precision != trtPrecisionType)
        {
            LOG_WARNING("TensorRT does not support setting quantization precision, the precision will be set to "
                << trtPrecisionType << ".");
        }
    }

    auto checkQuantizationDatatype = [&node, &nodeIdx, enableUInt8AsymmetricQuantization](DataType dtype) {
        ONNXTRT_CHECK_NODE(dtype == DataType::kFP8 || dtype == DataType::kINT8 || dtype == DataType::kINT4
                || dtype == DataType::kFP4 || (dtype == DataType::kUINT8 && enableUInt8AsymmetricQuantization),
            "TensorRT only allows FP8, INT8, INT4, "
                << (enableUInt8AsymmetricQuantization ? "UINT8, " : "")
                << "and FP4 quantization. The specified quantization type is " + getTrtDtypeName(dtype)
                    + ((dtype == DataType::kUINT8 && !enableUInt8AsymmetricQuantization)
                            ? ". Set the kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA flag to allow importing UINT8."
                            : "."),
            node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
    };

    if (inputs.size() > 2)
    {
        // ONNX spec definition is that when zero point is set, use its datatype for quantization
        DataType zeroPointDataType = inputs.at(2).getDataType();
        ONNXTRT_CHECK_NODE(!isOutputDtypeSet || outputDtype == zeroPointDataType,
            "Mismatch between attribute output_dtype " << outputDtype << " and zero-point data type "
                                                       << zeroPointDataType << ".",
            node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);

        // Validate and set quantization type.
        if (zeroPointDataType == DataType::kFP8 || zeroPointDataType == DataType::kINT8
            || zeroPointDataType == DataType::kINT4
            || zeroPointDataType == DataType::kFP4
            // Use UINT8 faithfully only if the flag is set
            || (zeroPointDataType == DataType::kUINT8 && enableUInt8AsymmetricQuantization))
        {
            chosenDataType = zeroPointDataType;
        }
        // If zero point is set to UINT8 and the flag has not been set, default to INT8.
        else if (zeroPointDataType == DataType::kUINT8 && !enableUInt8AsymmetricQuantization)
        {
            LOG_WARNING(
                "TensorRT supports QuantizeLinear/DequantizeLinear with UINT8 zero_point only on DLA (version >= "
                "3.16). "
                "Defaulting to INT8 instead. To import as UINT8, set the kIMPORT_UINT8_QUANTIZATION flag.");
            chosenDataType = DataType::kINT8;
        }
        else
        {
            LOG_WARNING("For zero_point with type " << zeroPointDataType << " TensorRT will use INT8 instead.");
            chosenDataType = DataType::kINT8;
        }

        if (chosenDataType != DataType::kFP8)
        {
            // For patterns "const" -> Q/DQ, the zero point constant can be shared between different Q/DQ. For pattern
            // like "const" -> "identity" -> Q/DQ. We have to create new constant because its type is INT8 in ONNX but
            // TRT expect FP32 zero point. To handle both case, we always create new constant for zero point.
            auto& zeroPtInput = inputs.at(2);
            ShapedWeights zeroPoint{};

            if (zeroPtInput.is_tensor())
            {
                // Look backward to find out the original weights in "Constant" node.
                zeroPoint = getWeightsFromIdentityOrConstant(*ctx->network(), &zeroPtInput.tensor());
            }
            else
            {
                zeroPoint = zeroPtInput.weights();
                ONNXTRT_CHECK_NODE(zeroPoint.values,
                    "QuantizeLinear/DequantizeLinear zero-point must have non-null values.", node, nodeIdx,
                    nvonnxparser::ErrorCode::kINVALID_NODE);
            }
            if (!zeroPoint.values)
            {
                // Cannot static analysis the zero point values from Q/DQ, fallback to use the activation input.
                zeroPointInput = &convertToTensor(inputs.at(2), ctx);
            }
            else if (enableUInt8AsymmetricQuantization)
            {
                zeroPointInput = addConstantLayer(*ctx->network(), zeroPoint);
            }
            else
            {
                ONNXTRT_CHECK_NODE(shiftIsAllZeros(zeroPoint),
                    "Non-zero zero point is not supported. Please set kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA"
                    "to enable asymmetric quantization if it is on DLA.",
                    node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
                // Convert the zero-point to float because TRT uses float for zero-point. Note this zero-point is
                // not refittable because refit need the same data type as builder time.
                auto fpZeroPoint = createZeroShifts(zeroPoint, ::ONNX_NAMESPACE::TensorProto::FLOAT, ctx);
                zeroPointInput = addConstantLayer(*ctx->network(), fpZeroPoint);
            }

            if (zeroPointInput && !isDynamic(scaleDims))
            {
                auto const zeroPointSize = volume(zeroPointInput->getDimensions());
                // ONNX may represent a scalar using either 0-D or 1-D, so compare sizes instead of shapes.
                ONNXTRT_CHECK_NODE(zeroPointSize == scaleSize,
                    "The scale and zero point must have the same volume. Size of zero point = "
                        << zeroPointSize << ", size of the scale = " << scaleSize << ".",
                    node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
            }
        }
    }
    else if (isOutputDtypeSet)
    {
        if (isDQ)
        {
            ONNXTRT_CHECK_NODE(
                outputDtype == DataType::kFLOAT || outputDtype == DataType::kHALF || outputDtype == DataType::kBF16,
                "Attribute output_dtype specifies an invalid data type for DequantizeLienar" << outputDtype << ".",
                node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        else
        {
            checkQuantizationDatatype(outputDtype);
            chosenDataType = outputDtype;
        }
    }

    // Read the optional quantization axis attribute.
    int32_t axis = attrs.get<int32_t>("axis", 1);

    // Axis attribute was first introduced in opset 13. Use a default of 0 for older models.
    if (ctx->getOpsetVersion() < 13 && !attrs.count("axis"))
    {
        axis = 0;
    }

    convertAxis(axis, inputDims.nbDims, node, nodeIdx);

    if (scaleSize != 1)
    {
        if (scaleDims.nbDims == 1 && !isDynamic(scaleDims))
        {
            // Ensure that number of scale-coefficients is equal to the number of output channels.
            int64_t const K = dataInput->getDimensions().d[axis];
            ONNXTRT_CHECK_NODE(K == scaleSize,
                "The number of scales is not equal to the number of output channels. Number of output channels = "
                    << K << ", number of scales = " << scaleSize << ".",
                node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        else if (scaleDims.nbDims == inputDims.nbDims)
        {
            // Exactly one dimension is blocked, other should have the same dimension as the input
            if (!isDynamic(inputDims) && !isDynamic(scaleDims))
            {
                int32_t rank = inputDims.nbDims;
                std::vector<int32_t> blockDims(rank);
                std::transform(
                    inputDims.d, inputDims.d + rank, scaleDims.d, blockDims.begin(), std::divides<int32_t>());

                auto equals_one = [](int32_t i) { return i == 1; };
                ONNXTRT_CHECK_NODE(std::count_if(blockDims.begin(), blockDims.end(), equals_one) == rank - 1,
                    "Only a single blocking dimension is allowed.", node, nodeIdx,
                    nvonnxparser::ErrorCode::kINVALID_NODE);

                auto const inputSize = volume(inputDims);
                ONNXTRT_CHECK_NODE(inputSize % scaleSize == 0,
                    "Inferred block size is not an integer. Input volume = " << inputSize
                                                                             << ", scale volume = " << scaleSize << ".",
                    node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
            }
        }
        else
        {
            ONNXTRT_CHECK_NODE(false, "Invalid rank for the scale tensor. Rank = " << scaleDims.nbDims << ".", node,
                nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        }
    }
    else
    {
        // Per-Tensor Quantization.
        // Currently axis is ignored by TRT, but it is required here by addScaleNd (for computing nbSpatialDims). Set to
        // a sane default depending on rank the input tensor.
        axis = inputDims.nbDims <= 1 ? 0 : 1;
    }

    // TRT does not support scalar data input for Q/DQ layers, convert 0-D tensor to 1-D first.
    if (inputDims.nbDims == 0)
    {
        dataInput = reshapeTensor(ctx, *dataInput, nvinfer1::Dims{1, {1}});
    }

    // INT4 requires an even last-dimension due to packing restrictions
    if (!isDynamic(inputDims) && (chosenDataType == DataType::kINT4 || chosenDataType == DataType::kFP4))
    {
        auto const inputSize = volume(inputDims);
        ONNXTRT_CHECK_NODE((inputSize % 2 == 0), "4-bit quantization requies an even number of elements.", node,
            nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
    }

    nvinfer1::ILayer* layer = nullptr;
    checkQuantizationDatatype(chosenDataType);

    bool stronglyTyped = ctx->isStronglyTyped();
    if (!stronglyTyped && chosenDataType != DataType::kINT8)
    {
        LOG_WARNING(
            "A strongly typed network is recommended for networks with QuantizedLinear/DequantizedLinear nodes using "
            "precisions other than int8.");
    }
    if (isDQ)
    {
        // Add and configure a DequantizeLayer.
        outputDtype = isOutputDtypeSet ? outputDtype : (isMX ? DataType::kFLOAT : scaleType);
        if (stronglyTyped)
        {
            // Input type is inferred. Layer output type is specified with scaleType.
            nvinfer1::IDequantizeLayer* dq
                = N_CHECK(ctx->network()->addDequantize(*dataInput, *scaleInput, outputDtype));
            dq->setAxis(axis);
            layer = dq;
        }
        else
        {
            // Use legacy API for weakly typed network.
            nvinfer1::IDequantizeLayer* dq = N_CHECK(ctx->network()->addDequantize(*dataInput, *scaleInput));
            dq->setAxis(axis);
            layer = dq;
            // Type constraint for layer output type.
            layer->setOutputType(0, outputDtype);
        }
    }
    else
    {
        // Add and configure a QuantizeLayer.
        if (stronglyTyped)
        {
            if (ctx->getOpsetVersion() < 19 && scaleInput->getType() != inputType)
            {
                // Ensure that Q scale type matches input type.
                auto* scaleCastLayer = N_CHECK(ctx->network()->addCast(*scaleInput, inputType));
                scaleInput = N_CHECK(scaleCastLayer->getOutput(0));
            }
            // Input type is inferred. Layer output type is specified with chosenDataType.
            nvinfer1::IQuantizeLayer* q = N_CHECK(ctx->network()->addQuantize(*dataInput, *scaleInput, chosenDataType));
            q->setAxis(axis);
            layer = q;
        }
        else
        {
            // Use legacy API for weakly typed network.
            nvinfer1::IQuantizeLayer* q = N_CHECK(ctx->network()->addQuantize(*dataInput, *scaleInput));
            q->setAxis(axis);
            layer = q;
            // This implicitly sets layer input type.
            layer->setPrecision(isPrecisionSet ? precision : (isMX ? DataType::kFLOAT : scaleType));
            // Type constraint for layer output type.
            layer->setOutputType(0, chosenDataType);
        }
    }

    layer->setName(nodeName.c_str());
    if (zeroPointInput)
    {
        layer->setInput(2, *zeroPointInput);
    }

    // Register the Q/DQ layer.
    ctx->registerLayer(layer, node);

    // Return layer output
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FP4DynamicQuantize)
{
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* scaleInput = &convertToTensor(inputs.at(1), ctx);
    int32_t const rank = dataInput->getDimensions().nbDims;

    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 1);
    convertAxis(axis, rank, node, nodeIdx);

    auto const blockSizeOnnx = attrs.get<int32_t>("block_size", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    auto const scaleTypeOnnx = attrs.get<int32_t>("scale_type", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);

    DataType scaleType;
    bool isScaleTypeSet = (scaleTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isScaleTypeSet)
    {
        isScaleTypeSet = convertDtype(scaleTypeOnnx, &scaleType);
        ONNXTRT_CHECK_NODE(isScaleTypeSet,
            "Attribute scale_type specifies an unsupported data type " << scaleType << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(scaleType == DataType::kFP8, "scale_type must be FP8.", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        LOG_WARNING("scale_type is not set. Defaulting to FP8.");
        scaleType = DataType::kFP8;
    }

    int32_t blockSize = -1;
    bool isBlockSizeSet = (blockSizeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isBlockSizeSet)
    {
        ONNXTRT_CHECK_NODE(blockSizeOnnx == 16, "Only block_size == 16 is supported for now.", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        blockSize = blockSizeOnnx;
    }
    else
    {
        LOG_WARNING("block_size is not set. Defaulting to 16.");
        blockSize = 16;
    }

    ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "Scale input must be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE(inputs.at(1).weights().count() == 1, "Scale input must be a scalar.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::IDynamicQuantizeLayer* dq
        = N_CHECK(ctx->network()->addDynamicQuantize(*dataInput, axis, blockSize, DataType::kFP4, scaleType));
    dq->setInput(1, *scaleInput);
    ctx->registerLayer(dq, node);

    RETURN_ALL_OUTPUTS(dq, node, nodeIdx);
}

static nvinfer1::Dims getBlockShape(OnnxAttrs attrs, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx)
{
    auto const blockShapeOnnx = attrs.get<std::vector<int32_t>>("block_shape");
    ONNXTRT_CHECK_NODE(!blockShapeOnnx.empty() && blockShapeOnnx.size() <= nvinfer1::Dims::MAX_DIMS,
        "block_shape must be non-empty and not exceed " << nvinfer1::Dims::MAX_DIMS << " dimensions.", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);
    nvinfer1::Dims blockShape;
    blockShape.nbDims = static_cast<int32_t>(blockShapeOnnx.size());
    std::copy_n(blockShapeOnnx.begin(), blockShape.nbDims, blockShape.d);
    return blockShape;
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_BlockQuantize)
{
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* scaleInput = &convertToTensor(inputs.at(1), ctx);

    OnnxAttrs attrs(node, ctx);

    auto const outputTypeOnnx = attrs.get<int32_t>("output_dtype", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);

    DataType outputType;
    bool isOutputTypeSet = (outputTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isOutputTypeSet)
    {
        isOutputTypeSet = convertDtype(outputTypeOnnx, &outputType);
        ONNXTRT_CHECK_NODE(isOutputTypeSet,
            "Attribute output_dtype specifies an unsupported data type " << outputTypeOnnx << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        LOG_WARNING("output_dtype is not set. Defaulting to FP8.");
        outputType = DataType::kFP8;
    }

    nvinfer1::Dims blockShape = getBlockShape(attrs, node, nodeIdx);
    nvinfer1::IQuantizeLayer* q = N_CHECK(ctx->network()->addQuantize(*dataInput, *scaleInput, outputType));
    q->setBlockShape(blockShape);
    ctx->registerLayer(q, node);

    RETURN_ALL_OUTPUTS(q, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_BlockDequantize)
{
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* scaleInput = &convertToTensor(inputs.at(1), ctx);

    OnnxAttrs attrs(node, ctx);

    auto const outputTypeOnnx = attrs.get<int32_t>("output_dtype", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);

    DataType outputType;
    bool isOutputTypeSet = (outputTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isOutputTypeSet)
    {
        isOutputTypeSet = convertDtype(outputTypeOnnx, &outputType);
        ONNXTRT_CHECK_NODE(isOutputTypeSet,
            "Attribute output_dtype specifies an unsupported data type " << outputTypeOnnx << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        LOG_WARNING("output_dtype is not set. Defaulting to scale type.");
        outputType = scaleInput->getType();
    }

    nvinfer1::Dims blockShape = getBlockShape(attrs, node, nodeIdx);
    nvinfer1::IDequantizeLayer* dq = N_CHECK(ctx->network()->addDequantize(*dataInput, *scaleInput, outputType));
    dq->setBlockShape(blockShape);
    ctx->registerLayer(dq, node);

    RETURN_ALL_OUTPUTS(dq, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_DynamicQuantize)
{
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* scaleInput = inputs.size() > 1 ? &convertToTensor(inputs.at(1), ctx) : nullptr;

    OnnxAttrs attrs(node, ctx);

    auto const blockShapeOnnx = attrs.get<std::vector<int32_t>>("block_shape");
    auto const scaleTypeOnnx = attrs.get<int32_t>("scale_type", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    auto const outputTypeOnnx = attrs.get<int32_t>("output_type", ::ONNX_NAMESPACE::TensorProto::UNDEFINED);

    DataType scaleType;
    bool isScaleTypeSet = (scaleTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isScaleTypeSet)
    {
        isScaleTypeSet = convertDtype(scaleTypeOnnx, &scaleType);
        ONNXTRT_CHECK_NODE(isScaleTypeSet,
            "Attribute scale_type specifies an unsupported data type " << scaleType << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        LOG_WARNING("scale_type is not set. Defaulting to Float.");
        scaleType = DataType::kFLOAT;
    }

    DataType outputType;
    bool isOutputTypeSet = (outputTypeOnnx != ::ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (isOutputTypeSet)
    {
        isOutputTypeSet = convertDtype(outputTypeOnnx, &outputType);
        ONNXTRT_CHECK_NODE(isOutputTypeSet,
            "Attribute output_type specifies an unsupported data type " << outputType << ".", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        LOG_WARNING("output_type is not set. Defaulting to FLOAT.");
        outputType = DataType::kFLOAT;
    }

    nvinfer1::Dims blockSizeDims;
    ONNXTRT_CHECK_NODE(!blockShapeOnnx.empty() && blockShapeOnnx.size() <= nvinfer1::Dims::MAX_DIMS,
        "block_shape must be non-empty and not exceed " << nvinfer1::Dims::MAX_DIMS << " dimensions.", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);

    blockSizeDims.nbDims = static_cast<int32_t>(blockShapeOnnx.size());
    for (int32_t i = 0; i < blockSizeDims.nbDims; ++i)
    {
        ONNXTRT_CHECK_NODE(blockShapeOnnx[i] >= -1, "block_shape values must be >= -1.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        blockSizeDims.d[i] = blockShapeOnnx[i];
    }

    if (inputs.size() > 1)
    {
        ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "Scale input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(inputs.at(1).weights().count() == 1, "Scale input must be a scalar.", node, nodeIdx,
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::IDynamicQuantizeLayer* dynq
        = N_CHECK(ctx->network()->addDynamicQuantizeV2(*dataInput, blockSizeDims, outputType, scaleType));
    if (scaleInput)
    {
        dynq->setInput(1, *scaleInput);
    }
    ctx->registerLayer(dynq, node);

    RETURN_ALL_OUTPUTS(dynq, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(QuantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, false /*isDQ*/, false /*isCustomOp*/);
}

DEFINE_BUILTIN_OP_IMPORTER(DequantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, true /*isDQ*/, false /*isCustomOp*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FP8QuantizeLinear)
{
    return QuantDequantLinearHelper(
        ctx, node, nodeIdx, inputs, false /*isDQ*/, true /*isCustomOp*/, DataType::kFP8 /*customOpType*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FP8DequantizeLinear)
{
    return QuantDequantLinearHelper(
        ctx, node, nodeIdx, inputs, true /*isDQ*/, true /*isCustomOp*/, DataType::kFP8 /*customOpType*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_INT4QuantizeLinear)
{
    return QuantDequantLinearHelper(
        ctx, node, nodeIdx, inputs, false /*isDQ*/, true /*isCustomOp*/, DataType::kINT4 /*customOpType*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_INT4DequantizeLinear)
{
    return QuantDequantLinearHelper(
        ctx, node, nodeIdx, inputs, true /*isDQ*/, true /*isCustomOp*/, DataType::kINT4 /*customOpType*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MXFP8QuantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, false /*isDQ*/, true /*isCustomOp*/,
        DataType::kFP8 /*customOpType*/, true /*isMX*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MXFP8DequantizeLinear)
{
    return QuantDequantLinearHelper(
        ctx, node, nodeIdx, inputs, true /*isDQ*/, true /*isCustomOp*/, DataType::kFP8 /*customOpType*/, true /*isMX*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MXFP8DynamicQuantize)
{
    ONNXTRT_CHECK_NODE((inputs.size() == 1),
        "Single dynamic quantization requires 1 input. " << inputs.size() << " are given.", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* dataInput = &convertToTensor(inputs.at(0), ctx);
    auto dtype = inputs.at(0).getDataType();
    ONNXTRT_CHECK_NODE((dtype == DataType::kFLOAT || dtype == DataType::kHALF || dtype == DataType::kBF16),
        "Input type must be FLOAT, FLOAT16, or BFLOAT16. Input type is " + getTrtDtypeName(dtype) + ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    int32_t const rank = dataInput->getDimensions().nbDims;

    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", -1);
    convertAxis(axis, rank, node, nodeIdx);

    int32_t const blockSize = attrs.get<int32_t>("block_size", 32);
    ONNXTRT_CHECK_NODE(
        blockSize == 32, "Only block_size 32 is supported.", node, nodeIdx, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::IDynamicQuantizeLayer* dq
        = N_CHECK(ctx->network()->addDynamicQuantize(*dataInput, axis, blockSize, DataType::kFP8, DataType::kE8M0));
    ctx->registerLayer(dq, node);

    RETURN_ALL_OUTPUTS(dq, node, nodeIdx);
}

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kDIV);
}

DEFINE_BUILTIN_OP_IMPORTER(Dropout)
{
    // TensorRT does not support the Dropout operator with training mode.
    // The source of training mode information comes from :
    // 1. Pre-opset 6: attribute is_test = 0
    // 2. Post-opset 12: input[2] training_mode = true.
    //      We can deal with the cases where training_mode is an initializer.
    if (ctx->getOpsetVersion() >= 12 && node.input().size() == 3)
    {
        ONNXTRT_CHECK_NODE(inputs.at(2).is_weights(),
            "This Version of TensorRT only supports the training_mode input as an initializer.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
        std::vector<int64_t> trainingMode;
        weightsToVector<int64_t>(inputs.at(2).weights(), &trainingMode);
        ONNXTRT_CHECK_NODE(!trainingMode[0], "TensorRT does not support the Dropout operator in training mode.", node,
            nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    size_t noutputs = node.output().size();
    if (noutputs == 1)
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }
    else
    {
        // Add identity layer twice for both Dropout outputs: (output + mask)
        std::vector<TensorOrWeights> outputs;
        outputs.push_back(identity(ctx, inputs.at(0)));

        // Add mask tensor, which is the same shape as the input tensor
        auto& inputTensor = inputs.at(0).tensor();
        nvinfer1::ITensor* maskTensor{nullptr};
        // Post opset 12 the mask tensor contains all 1s. Prior to opset 12 the mask tensor contains all 0s.
        if (ctx->getOpsetVersion() >= 12)
        {
            maskTensor = getElementWiseResult(ctx, inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kEQUAL);
        }
        else
        {
            maskTensor = getElementWiseResult(ctx, inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kLESS);
        }
        outputs.push_back(TensorOrWeights(maskTensor));
        return outputs;
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Einsum)
{
    checkNotInvalidType(inputs.at(0), {"UINT8", "INT32", "INT64"}, node, nodeIdx);
    OnnxAttrs attrs(node, ctx);
    std::string equation = attrs.get<std::string>("equation");

    ONNXTRT_CHECK_NODE((!inputs.empty()), "Inputs vector is empty.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    std::vector<nvinfer1::ITensor*> inputTensors;

    for (auto input : inputs)
    {
        auto* tensor_ptr = &convertToTensor(input, ctx);
        inputTensors.push_back(tensor_ptr);
    }
    auto nbInputs = static_cast<int64_t>(inputTensors.size());

    nvinfer1::IEinsumLayer* einsumLayer{nullptr};
    einsumLayer = N_CHECK(ctx->network()->addEinsum(inputTensors.data(), nbInputs, equation.c_str()));
    ctx->registerLayer(einsumLayer, node);

    RETURN_FIRST_OUTPUT(einsumLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Elu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Equal)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kEQUAL);
}

DEFINE_BUILTIN_OP_IMPORTER(Erf)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kERF);
}

DEFINE_BUILTIN_OP_IMPORTER(Exp)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Expand)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    // "Broadcast the input tensor following the given shape and the broadcast rule."
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(inputTensor);
    auto const inputRank = shapeOf(inputDims);

    // "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
    ONNXTRT_CHECK_NODE((inputs.at(1).shape().nbDims == 1), "The shape tensor is required to be 1D.", node, nodeIdx,
        ErrorCode::kINVALID_VALUE);
    ShapeTensor shape{ctx, inputs.at(1)};
    auto const shapeLength = shapeOf(shape);

    ShapeTensor const newRank = max(ctx, shapeLength, inputRank);
    // "Dimensions are right alignment;..."
    ShapeTensor const newDims = concat(ctx, fillShapeVector(ctx, 1, sub(ctx, newRank, inputRank)), inputDims);
    nvinfer1::ITensor& newInputTensor = reshape(ctx, inputTensor, newDims);

    // ", or the shape.ndim < input.shape.ndim"
    ShapeTensor newShape = concat(ctx, fillShapeVector(ctx, 1, sub(ctx, newRank, shapeLength)), shape);

    ShapeTensor const starts = similar(ctx, newDims, 0);
    // Do the broadcast rule.
    ShapeTensor const sizes = broadcast(ctx, newDims, newShape);
    // Compute (x > 1 ? 1 : 0) for x in newDims, using only TensorRT operations.
    ShapeTensor const zero = shapeVector(0);
    ShapeTensor const one = shapeVector(1);
    ShapeTensor const strides = min(ctx, one, max(ctx, sub(ctx, newDims, one), zero));

    nvinfer1::ISliceLayer* sliceLayer = addSlice(ctx, newInputTensor, starts, sizes, strides);
    ctx->registerLayer(sliceLayer, node);

    RETURN_FIRST_OUTPUT(sliceLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(EyeLike)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t k = attrs.get("k", 0);

    // "Only 2D tensors are supported, i.e. input T1 must be of rank 2..."
    nvinfer1::Dims dims = tensor.getDimensions();
    ONNXTRT_CHECK_NODE(dims.nbDims == 2, "Only 2D tensors are supported. Input must be of rank 2.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // The data type can be specified by the 'dtype' argument
    DataType dtype = tensor.getType();
    if (attrs.count("dtype"))
    {
        auto onnxType = attrs.get<int32_t>("dtype");
        ONNXTRT_CHECK_NODE(
            convertDtype(onnxType, &dtype), "Unsupported cast!", node, nodeIdx, ErrorCode::kINVALID_NODE);
        LOG_VERBOSE("Casting to type: " << dtype);
    }

    // Create weights and constant layer
    ONNXTRT_CHECK_NODE(!isDynamic(dims), "Eyelike does not work for dynamically shaped tensors.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
    int totalWeights = dims.d[0] * dims.d[1];
    std::vector<int> values(totalWeights);
    for (int32_t r = 0; r < dims.d[0]; ++r)
    {
        for (int32_t c = 0; c < dims.d[1]; ++c)
        {
            values[r * dims.d[1] + c] = 0;
            if (c - r == k)
            {
                values[r * dims.d[1] + c] = 1;
            }
        }
    }

    ShapedWeights tempWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, dims);
    std::memcpy(tempWeights.values, values.data(), values.size() * sizeof(int));
    auto* layer = N_CHECK(ctx->network()->addConstant(dims, tempWeights));
    ctx->registerLayer(layer, node);
    auto* layerOutput = N_CHECK(layer->getOutput(0));

    if (dtype != DataType::kINT32)
    {
        return {{castHelper(ctx, layerOutput, dtype)}};
    }
    return {{layerOutput}};
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    int32_t axis = attrs.get("axis", 1);
    convertAxis(axis, nbDims, node, nodeIdx);

    // No-op Flatten: (a, b) => Flatten(axis = 1) => (a, b)
    // Add identity layer to avoid name mangling of engine bindings
    // For rest of configurations, we must flatten.
    if (nbDims == 2 && axis == 1)
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    tensorPtr = flattenTensor(ctx, node, *tensorPtr, axis, true);
    ONNXTRT_CHECK_NODE(tensorPtr, "Failed to flatten the tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    // If UINT8 quantization is supported, Gather nodes can have UINT8 input if they feed into a Dequantize node.
    auto getFlagBit = [](nvonnxparser::OnnxParserFlag const flag) { return 1U << static_cast<uint32_t>(flag); };
    bool const enableUInt8AsymmetricQuantization
        = ctx->getFlags() & getFlagBit(nvonnxparser::OnnxParserFlag::kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA);

    // note that all nodes except Dequantize will also reject Uint8, so that is what ensures this can only feed into a
    // Dequantize node
    if (!enableUInt8AsymmetricQuantization)
    {
        checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    }

    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    convertAxis(axis, nbDims, node, nodeIdx);
    LOG_VERBOSE("Using Gather axis: " << axis);

    if (inputs.at(0).getType() != "INT64" && inputs.at(1).getType() == "INT64")
    {
        // Int64 indices only supported for Int64 data
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    auto* layer = N_CHECK(ctx->network()->addGather(data, *indices, axis));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherElements)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::Dims const& dataDims = data.getDimensions();

    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t const dataNbDims = dataDims.nbDims;
    convertAxis(axis, dataNbDims, node, nodeIdx);
    LOG_VERBOSE("Using Gather axis: " << axis);

    if (inputs.at(0).getType() != "INT64")
    {
        // Int64 indices only supported for Int64 data
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    auto* layer = N_CHECK(ctx->network()->addGatherV2(data, *indices, nvinfer1::GatherMode::kELEMENT));
    layer->setGatherAxis(axis);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherND)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);

    OnnxAttrs attrs(node, ctx);
    auto const nbElementWiseDims = attrs.get<int32_t>("batch_dims", 0);

    if (inputs.at(0).getType() != "INT64")
    {
        // Int64 indices only supported for Int64 data
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    auto* layer = ctx->network()->addGatherV2(data, *indices, nvinfer1::GatherMode::kND);
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setNbElementWiseDims(nbElementWiseDims);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Gelu)
{
    OnnxAttrs attrs(node, ctx);
    auto const approximate = attrs.get("approximate", std::string("none"));
    if (approximate == std::string("none"))
    {
        return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kGELU_ERF);
    }
    else if (approximate == std::string("tanh"))
    {
        return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kGELU_TANH);
    }
    else
    {
        ONNXTRT_CHECK_NODE(false, "Invalid value provided for the Gelu \'approximate\' attribute: " << approximate,
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_ATTR);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    checkNotInvalidType(inputs.at(0), {"INT32", "INT64"}, node, nodeIdx);
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& inputB = convertToTensor(inputs.at(1), ctx);
    // Validate inputs
    ONNXTRT_CHECK_NODE(inputA.getDimensions().nbDims == 2 && inputB.getDimensions().nbDims == 2,
        "GEMM must have 2D inputs! inputA has rank " << inputA.getDimensions().nbDims << ", inputB has rank "
                                                     << inputB.getDimensions().nbDims << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto const getMatrixOp = [](nvinfer1::ITensor const& input, bool transpose) {
        if (input.getDimensions().nbDims == 1)
        {
            return nvinfer1::MatrixOperation::kVECTOR;
        }
        if (transpose)
        {
            return nvinfer1::MatrixOperation::kTRANSPOSE;
        }
        return nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(inputA, transA);
    nvinfer1::MatrixOperation opB = getMatrixOp(inputB, transB);

    LOG_VERBOSE("Using opA: " << static_cast<int>(opA) << " opB: " << static_cast<int>(opB));

    nvinfer1::IMatrixMultiplyLayer* matmul = N_CHECK(ctx->network()->addMatrixMultiply(inputA, opA, inputB, opB));
    ctx->registerLayer(matmul, node);

    nvinfer1::ITensor* matmulTensor = N_CHECK(matmul->getOutput(0));

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, getNbDims(matmulTensor));
        nvinfer1::ITensor* alphaConstantTensor = N_CHECK(alphaConstant->getOutput(0));
        nvinfer1::IElementWiseLayer* scaledMatmul = N_CHECK(
            ctx->network()->addElementWise(*alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD));
        matmulTensor = N_CHECK(scaledMatmul->getOutput(0));
    }

    // In opset 11, the bias tensor is an optional input
    if (inputs.size() > 2)
    {
        nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);

        // Scale C if needed
        if (beta != 1.f)
        {
            nvinfer1::IConstantLayer* betaConstant
                = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, getNbDims(biasTensor));
            nvinfer1::ITensor* betaConstantTensor = N_CHECK(betaConstant->getOutput(0));
            nvinfer1::IElementWiseLayer* scaledBias = N_CHECK(ctx->network()->addElementWise(
                *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD));
            biasTensor = N_CHECK(scaledBias->getOutput(0));
        }
        broadcastTensors(ctx, matmulTensor, biasTensor);
        nvinfer1::IElementWiseLayer* biasAdd
            = N_CHECK(ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM));
        auto output = N_CHECK(biasAdd->getOutput(0));
        return {{output}};
    }

    return {{matmulTensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool)
{
    LOG_VERBOSE("GlobalAveragePool operators are implemented via Reduce layers rather than Pooling layers");
    return {{globalPoolingHelper(ctx, node, convertToTensor(inputs.at(0), ctx), nvinfer1::ReduceOperation::kAVG)}};
}

// GlobalLpPool: pow(reduce_sum(pow(x, p)), 1./p)
DEFINE_BUILTIN_OP_IMPORTER(GlobalLpPool)
{
    auto& tensor = convertToTensor(inputs.at(0), ctx);
    auto inputType = tensor.getType();
    ONNXTRT_CHECK_NODE((inputType == DataType::kFLOAT || inputType == DataType::kHALF),
        "Only FLOAT and HALF are supported in GlobalLpPool. The current type = " + getTrtDtypeName(inputType) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    int32_t const nbDims = getNbDims(&tensor);

    OnnxAttrs attrs{node, ctx};
    float p = static_cast<float>(attrs.get("p", 2));

    // Add constants for p and 1/p
    nvinfer1::IConstantLayer* pLayer;
    nvinfer1::IConstantLayer* pInvLayer;
    if (inputType == DataType::kHALF)
    {
        pLayer
            = addConstantScalar(ctx, static_cast<half_float::half>(p), ::ONNX_NAMESPACE::TensorProto::FLOAT16, nbDims);
        pInvLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(1.F / p), ::ONNX_NAMESPACE::TensorProto::FLOAT16, nbDims);
    }
    else
    {
        pLayer = addConstantScalar(ctx, p, ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims);
        pInvLayer = addConstantScalar(ctx, 1.F / p, ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims);
    }

    // firstPow = pow(x, p)
    auto* firstPow
        = getElementWiseResult(ctx, tensor, *N_CHECK(pLayer->getOutput(0)), nvinfer1::ElementWiseOperation::kPOW);
    // reduced = reduce_sum(firstPow)
    auto* reduced = globalPoolingHelper(ctx, node, *firstPow, nvinfer1::ReduceOperation::kSUM);
    // finalPow = pow(reduced, 1./p)
    auto* finalPow
        = getElementWiseResult(ctx, *reduced, *N_CHECK(pInvLayer->getOutput(0)), nvinfer1::ElementWiseOperation::kPOW);
    return {{finalPow}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool)
{
    LOG_VERBOSE("GlobalMaxPool operators are implemented via Reduce layers rather than Pooling layers");
    return {{globalPoolingHelper(ctx, node, convertToTensor(inputs.at(0), ctx), nvinfer1::ReduceOperation::kMAX)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Greater)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kGREATER);
}

DEFINE_BUILTIN_OP_IMPORTER(GreaterOrEqual)
{
    return greaterLessOrEqual(ctx, node, nodeIdx, &convertToTensor(inputs.at(0), ctx),
        &convertToTensor(inputs.at(1), ctx),
        /*greater*/ true);
}

DEFINE_BUILTIN_OP_IMPORTER(GroupNormalization)
{
    bool const useV2 = ctx->getOpsetVersion() >= 21;
    return normalizationHelper(ctx, node, nodeIdx, inputs, useV2);
}

// singlePassShape is the shape of the output from a single pass.
nvinfer1::ITensor* concatenateRNNOutputs(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ILoop* loop, nvinfer1::ITensor* singlePassShape, nvinfer1::ITensor* sequenceLength,
    nvinfer1::ITensor* concatenatedOutput, int numDirections, std::vector<TensorOrWeights>& inputs,
    bool reverse = false)
{
    nvinfer1::ITensor* yOutput{nullptr};
    if (numDirections == 2)
    {
        nvinfer1::ITensor* forwardStart = addConstant(ctx, std::vector<int32_t>{0, 0, 0},
            ::ONNX_NAMESPACE::TensorProto::INT32,
            nvinfer1::Dims{1, {3}})->getOutput(0);
        nvinfer1::ITensor* reverseStart = addConstant(ctx, std::vector<int32_t>{1, 0, 0},
            ::ONNX_NAMESPACE::TensorProto::INT32,
            nvinfer1::Dims{1, {3}})->getOutput(0);

        LOG_VERBOSE("Concatenated output shape: " << concatenatedOutput->getDimensions());
        nvinfer1::ISliceLayer* HtForwardLayer = N_CHECK(ctx->network()->addSlice(
            *concatenatedOutput, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1}));
        auto forwardHt = N_CHECK(HtForwardLayer->getOutput(0));
        LOG_VERBOSE("Forward pass shape: " << forwardHt->getDimensions());
        HtForwardLayer->setInput(1, *forwardStart);
        HtForwardLayer->setInput(2, *singlePassShape);

        nvinfer1::ISliceLayer* HtBackwardLayer = N_CHECK(ctx->network()->addSlice(
            *concatenatedOutput, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1}));
        auto backwardHt = N_CHECK(HtBackwardLayer->getOutput(0));
        LOG_VERBOSE("Reverse pass shape: " << backwardHt->getDimensions());
        HtBackwardLayer->setInput(1, *reverseStart);
        HtBackwardLayer->setInput(2, *singlePassShape);

        if (inputs.size() > 4 && inputs.at(4))
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
            forwardHt = clearMissingSequenceElements(ctx, node, loop, seqLens, forwardHt, sequenceLength);
            backwardHt
                = clearMissingSequenceElements(ctx, node, loop, seqLens, backwardHt, sequenceLength, /*reverse=*/true);
        }

        nvinfer1::ILoopOutputLayer* forwardOutput
            = N_CHECK(loop->addLoopOutput(*forwardHt, nvinfer1::LoopOutput::kCONCATENATE, 0));
        forwardOutput->setInput(1, *sequenceLength);
        nvinfer1::ILoopOutputLayer* reverseOutput
            = N_CHECK(loop->addLoopOutput(*backwardHt, nvinfer1::LoopOutput::kREVERSE, 0));
        reverseOutput->setInput(1, *sequenceLength);

        auto fTensor = N_CHECK(forwardOutput->getOutput(0));
        auto rTensor = N_CHECK(reverseOutput->getOutput(0));

        std::array<nvinfer1::ITensor*, 2> passes{{fTensor, rTensor}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(passes.data(), passes.size());
        concat->setAxis(1);
        yOutput = N_CHECK(concat->getOutput(0));
    }
    else
    {
        if (inputs.size() > 4 && inputs.at(4))
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
            concatenatedOutput
                = clearMissingSequenceElements(ctx, node, loop, seqLens, concatenatedOutput, sequenceLength, reverse);
        }
        nvinfer1::ILoopOutputLayer* scanOut = N_CHECK(loop->addLoopOutput(
            *concatenatedOutput, (reverse ? nvinfer1::LoopOutput::kREVERSE : nvinfer1::LoopOutput::kCONCATENATE), 0));
        scanOut->setInput(1, *sequenceLength);
        yOutput = N_CHECK(scanOut->getOutput(0));
    }
    return yOutput;
}

DEFINE_BUILTIN_OP_IMPORTER(GRU)
{
    using nvinfer1::Dims;
    using nvinfer1::Dims3;
    using mOp = nvinfer1::MatrixOperation;
    using eOp = nvinfer1::ElementWiseOperation;
    using trtAct = nvinfer1::ActivationType;
    nvinfer1::INetworkDefinition* net = ctx->network();

    OnnxAttrs attrs{node, ctx};
    constexpr int32_t NUM_GATES = 3;
    std::string const direction = attrs.get<std::string>("direction", "forward");
    int32_t const numDirections = (direction == "bidirectional") ? 2 : 1;
    int32_t const hiddenSize = attrs.get<int32_t>("hidden_size");
    int32_t const linearBeforeReset = attrs.get<int32_t>("linear_before_reset", 0);
    float const clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& weights = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& recurrenceWeights = convertToTensor(inputs.at(2), ctx);

    std::vector<trtAct> defaultActs{trtAct::kSIGMOID, trtAct::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {trtAct::kSIGMOID, trtAct::kTANH});
    }
    std::vector<trtAct> activations = attrs.get<std::vector<trtAct>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    std::transform(activations.begin() + activationAlphas.size(), activations.end(),
        std::back_inserter(activationAlphas), &getActivationDefaultAlpha);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    std::transform(activations.begin() + activationBetas.size(), activations.end(), std::back_inserter(activationBetas),
        &getActivationDefaultBeta);

    // Need to split weights/biases into ZR gates and H gate, because h(t) computations depend on z(t) and r(t).
    nvinfer1::ITensor* numDirectionsTensor
        = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto::INT32, 1)->getOutput(0);
    nvinfer1::ITensor* hiddenSizeTensor
        = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, 1)->getOutput(0);
    nvinfer1::ITensor* hiddenSizeDoubledTensor
        = addConstantScalar(ctx, 2 * hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, 1)->getOutput(0);
    nvinfer1::ITensor* eDimTensor = getAxisLength(ctx, input, 2, Dims{1, {1}});

    nvinfer1::ITensor* weightsZRStart = addConstant(ctx, std::vector<int32_t>{0, 0, 0},
        ::ONNX_NAMESPACE::TensorProto::INT32,
        Dims{1, {3}})->getOutput(0);
    auto* weightsZRSizeLayer = N_CHECK(net->addConcatenation(
        std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, hiddenSizeDoubledTensor, eDimTensor}}.data(), 3));
    nvinfer1::ITensor* weightsZRSize = N_CHECK(weightsZRSizeLayer->getOutput(0));
    nvinfer1::ISliceLayer* weightsZRLayer = N_CHECK(net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1}));
    ONNXTRT_CHECK_NODE(weightsZRLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    weightsZRLayer->setInput(1, *weightsZRStart);
    weightsZRLayer->setInput(2, *weightsZRSize);
    nvinfer1::ITensor* weightsZR = N_CHECK(weightsZRLayer->getOutput(0));
    LOG_VERBOSE("Weights for ZR gates shape is: " << weightsZR->getDimensions());

    nvinfer1::ITensor* weightsHStart = addConstant(ctx, std::vector<int32_t>{0, 2 * hiddenSize, 0},
        ::ONNX_NAMESPACE::TensorProto::INT32,
        Dims{1, {3}})->getOutput(0);
    auto weightsHSizeLayer = N_CHECK(net->addConcatenation(
        std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, hiddenSizeTensor, eDimTensor}}.data(), 3));
    nvinfer1::ITensor* weightsHSize = N_CHECK(weightsHSizeLayer->getOutput(0));
    nvinfer1::ISliceLayer* weightsHLayer = N_CHECK(net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1}));
    weightsHLayer->setInput(1, *weightsHStart);
    weightsHLayer->setInput(2, *weightsHSize);
    nvinfer1::ITensor* weightsH = N_CHECK(weightsHLayer->getOutput(0));
    LOG_VERBOSE("Weights for H gate shape is: " << weightsH->getDimensions());

    auto recurrenceWeightsZRLayer = N_CHECK(net->addSlice(
        recurrenceWeights, Dims3{0, 0, 0}, Dims3{numDirections, 2 * hiddenSize, hiddenSize}, Dims3{1, 1, 1}));
    nvinfer1::ITensor* recurrenceWeightsZR = N_CHECK(recurrenceWeightsZRLayer->getOutput(0));
    LOG_VERBOSE("Recurrence weights for ZR gates shape is: " << recurrenceWeightsZR->getDimensions());
    auto recurrenceWeightsHLayer = N_CHECK(net->addSlice(
        recurrenceWeights, Dims3{0, 2 * hiddenSize, 0}, Dims3{numDirections, hiddenSize, hiddenSize}, Dims3{1, 1, 1}));
    nvinfer1::ITensor* recurrenceWeightsH = N_CHECK(recurrenceWeightsHLayer->getOutput(0));
    LOG_VERBOSE("Recurrence weights for H gate shape is: " << recurrenceWeightsH->getDimensions());

    // bias/recurrenceBias will have shape (numDirections, NUM_GATES * hiddenSize)
    nvinfer1::ITensor* biasZR{nullptr};
    nvinfer1::ITensor* biasH{nullptr};
    nvinfer1::ITensor* recurrenceBiasZR{nullptr};
    nvinfer1::ITensor* recurrenceBiasH{nullptr};
    if (inputs.size() > 3 && inputs.at(3))
    {
        // ONNX bias is a concatenation of Wb and Rb on the second axis, so has shape (numDirections, 2 * NUM_GATES *
        // hiddenSize)
        // Unsqueeze to (numDirections, 1, 2 * NUM_GATES * hiddenSize) so we can broadcast later
        nvinfer1::ITensor* concatenatedBias = &convertToTensor(inputs.at(3), ctx);
        nvinfer1::IShuffleLayer* unsqueeze = N_CHECK(net->addShuffle(*concatenatedBias));
        unsqueeze->setReshapeDimensions(Dims3{numDirections, 1, 2 * NUM_GATES * hiddenSize});
        unsqueeze->setZeroIsPlaceholder(false);
        concatenatedBias = N_CHECK(unsqueeze->getOutput(0));

        auto biasZRLayer = N_CHECK(
            net->addSlice(*concatenatedBias, Dims3{0, 0, 0}, Dims3{numDirections, 1, 2 * hiddenSize}, Dims3{1, 1, 1}));
        biasZR = N_CHECK(biasZRLayer->getOutput(0));
        LOG_VERBOSE("Bias for ZR gates shape is: " << biasZR->getDimensions());
        auto biasHLayer = N_CHECK(net->addSlice(
            *concatenatedBias, Dims3{0, 0, 2 * hiddenSize}, Dims3{numDirections, 1, hiddenSize}, Dims3{1, 1, 1}));
        biasH = N_CHECK(biasHLayer->getOutput(0));
        LOG_VERBOSE("Bias for H gate shape is: " << biasH->getDimensions());

        auto recurrenceBiasZRLayer = N_CHECK(net->addSlice(*concatenatedBias, Dims3{0, 0, NUM_GATES * hiddenSize},
            Dims3{numDirections, 1, 2 * hiddenSize}, Dims3{1, 1, 1}));
        recurrenceBiasZR = N_CHECK(recurrenceBiasZRLayer->getOutput(0));
        LOG_VERBOSE("Recurrence bias for ZR gates shape is: " << recurrenceBiasZR->getDimensions());
        auto recurrenceBiasHLayer = N_CHECK(net->addSlice(*concatenatedBias, Dims3{0, 0, (NUM_GATES + 2) * hiddenSize},
            Dims3{numDirections, 1, hiddenSize}, Dims3{1, 1, 1}));
        recurrenceBiasH = N_CHECK(recurrenceBiasHLayer->getOutput(0));
        LOG_VERBOSE("Recurrence bias for H gate shape is: " << recurrenceBiasH->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input, &net]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor
            = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0);
        LOG_VERBOSE("numDirections is: " << numDirections
                                         << ", numDirections Tensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor
            = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0);
        LOG_VERBOSE(
            "hiddenSize is: " << hiddenSize << ", hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, Dims{1, {1}});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        nvinfer1::IConcatenationLayer* concatenatedShape = N_CHECK(net->addConcatenation(
            std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}}.data(), 3));
        return N_CHECK(concatenatedShape->getOutput(0));
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = N_CHECK(net->addLoop());
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);

    // H(t-1)
    auto const getInitialInputValue = [&ctx, &gateOutputShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0),
            gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::IRecurrenceLayer* Ht1 = N_CHECK(loop->addRecurrence(*initialHidden));
    ctx->registerLayer(Ht1, node);
    auto Ht1Output = N_CHECK(Ht1->getOutput(0));
    LOG_VERBOSE("Hidden state shape: " << Ht1Output->getDimensions());

    // Compute stackedZR(t) = f(X(t) * W[zr]^T + H(t-1) * R[zr]^T + (Wb[zr] + Rb[zr])). stackedZR(t) has shape
    // (numDirections, batchSize, 2 * hiddenSize)
    auto xtWTZRLayer = N_CHECK(net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsZR, mOp::kTRANSPOSE));
    nvinfer1::ITensor* xtWTZR = N_CHECK(xtWTZRLayer->getOutput(0));
    LOG_VERBOSE("X(t) * W[zr]^T -> " << xtWTZR->getDimensions());

    auto ht1RTLayer = N_CHECK(net->addMatrixMultiply(*Ht1Output, mOp::kNONE, *recurrenceWeightsZR, mOp::kTRANSPOSE));
    nvinfer1::ITensor* ht1RT = N_CHECK(ht1RTLayer->getOutput(0));
    LOG_VERBOSE("H(t-1) * R[zr]^T -> " << ht1RT->getDimensions());

    auto stackedZRtSumLayer = N_CHECK(net->addElementWise(*xtWTZR, *ht1RT, eOp::kSUM));
    nvinfer1::ITensor* stackedZRt = N_CHECK(stackedZRtSumLayer->getOutput(0));
    if (biasZR && recurrenceBiasZR)
    {
        auto biasSumLayer = N_CHECK(net->addElementWise(*stackedZRt, *biasZR, eOp::kSUM));
        stackedZRt = N_CHECK(biasSumLayer->getOutput(0));
        auto recSumLayer = N_CHECK(net->addElementWise(*stackedZRt, *recurrenceBiasZR, eOp::kSUM));
        stackedZRt = N_CHECK(recSumLayer->getOutput(0));
    }

    nvinfer1::IActivationLayer* stackedZRtLayer
        = N_CHECK(net->addActivation(*addClip(ctx, stackedZRt, clip), activations.at(0)));
    stackedZRtLayer->setAlpha(activationAlphas.at(0));
    stackedZRtLayer->setBeta(activationBetas.at(0));
    stackedZRt = N_CHECK(stackedZRtLayer->getOutput(0));
    LOG_VERBOSE("stackedZR(t) -> " << stackedZRt->getDimensions());

    auto const isolateGate = [&ctx, &hiddenSize, &gateOutputShape, &net](
                                 nvinfer1::ITensor* gates, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolateGate
            = N_CHECK(net->addSlice(*gates, Dims3{0, 0, 0}, Dims3{0, 0, 0}, Dims3{1, 1, 1}));
        isolateGate->setInput(1,
            *addConstant(ctx, std::vector<int32_t>{0, 0, gateIndex * hiddenSize},
                ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, {3}})
                ->getOutput(0));                    // Start
        isolateGate->setInput(2, *gateOutputShape); // Size
        return N_CHECK(isolateGate->getOutput(0));
    };

    // zt = stackedZRt[:, :. 0:H]
    nvinfer1::ITensor* zt = isolateGate(stackedZRt, 0);
    LOG_VERBOSE("z(t) -> " << zt->getDimensions());

    // rt = stackedZRt[:, :. H:2H]
    nvinfer1::ITensor* rt = isolateGate(stackedZRt, 1);
    LOG_VERBOSE("r(t) -> " << rt->getDimensions());

    // Compute h(t)
    nvinfer1::ITensor* ht{nullptr};
    // xtWTH = X(t) * (W[h]^T)
    auto xtWTHLayer = N_CHECK(net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsH, mOp::kTRANSPOSE));
    nvinfer1::ITensor* xtWTH = N_CHECK(xtWTHLayer->getOutput(0));
    if (linearBeforeReset == 0)
    {
        // h(t) = g(xtWTH + (r(t) . H(t-1)) * (R[h]^T) + Rb[h] + Wb[h])
        // rtHt1 = (r(t) . H(t-1))
        nvinfer1::ITensor* rtHt1 = getElementWiseResult(ctx, *rt, *Ht1Output, eOp::kPROD);
        // rtHt1Rh = (r(t) . H(t-1)) * (R[h]^T)
        auto rtHt1RhLayer = N_CHECK(net->addMatrixMultiply(*rtHt1, mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE));
        nvinfer1::ITensor* rtHt1Rh = N_CHECK(rtHt1RhLayer->getOutput(0));

        // (xtWTH + rtHt1Rh) + (Rb[h] + Wb[h])
        nvinfer1::ITensor* actInput = getElementWiseResult(ctx, *xtWTH, *rtHt1Rh, eOp::kSUM);

        // If bias is defines, both recurrence and normal bias must be present
        if (recurrenceBiasH && biasH)
        {
            nvinfer1::ITensor* secondSum = getElementWiseResult(ctx, *recurrenceBiasH, *biasH, eOp::kSUM);
            actInput = getElementWiseResult(ctx, *actInput, *secondSum, eOp::kSUM);
        }

        nvinfer1::IActivationLayer* htLayer
            = N_CHECK(net->addActivation(*addClip(ctx, actInput, clip), activations.at(1)));
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = N_CHECK(htLayer->getOutput(0));
    }
    else
    {
        // h(t) = g(xtWTH + (r(t) . (H(t-1) * (R[h]^T) + Rb[h])) + Wb[h])
        // ht1Rh = H(t-1) * (R[h]^T)
        auto ht1RhLayer = N_CHECK(net->addMatrixMultiply(*Ht1Output, mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE));
        nvinfer1::ITensor* ht1Rh = N_CHECK(ht1RhLayer->getOutput(0));

        // rtHtRhRbh = r(t) . (ht1Rh + Rb[h])
        if (recurrenceBiasH)
        {
            ht1Rh = getElementWiseResult(ctx, *ht1Rh, *recurrenceBiasH, eOp::kSUM);
        }
        nvinfer1::ITensor* rtHtRhRbh = getElementWiseResult(ctx, *rt, *ht1Rh, eOp::kPROD);

        // h(t) = g(xtWTH + rtHtRhRbh + Wb[h])
        if (biasH)
        {
            rtHtRhRbh = getElementWiseResult(ctx, *rtHtRhRbh, *biasH, eOp::kSUM);
        }
        nvinfer1::IActivationLayer* htLayer = N_CHECK(net->addActivation(
            *addClip(ctx, getElementWiseResult(ctx, *xtWTH, *rtHtRhRbh, eOp::kSUM), clip), activations.at(1)));
        ONNXTRT_CHECK_NODE(htLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = htLayer->getOutput(0);
    }
    LOG_VERBOSE("h(t) -> " << ht->getDimensions());

    // H(t) = (1 - z(t)) . h(t) + (z(t) . H(t-1))
    // Constant `1` needs to be the same type as the inputs, either FP16 or FP32.
    auto* constOne = zt->getType() == nvinfer1::DataType::kHALF
        ? N_CHECK(addConstantScalar(ctx, static_cast<half_float::half>(1), ::ONNX_NAMESPACE::TensorProto::FLOAT16, 3)
                      ->getOutput(0))
        : N_CHECK(addConstantScalar(ctx, 1.f, ::ONNX_NAMESPACE::TensorProto::FLOAT, 3)->getOutput(0));
    nvinfer1::ITensor* Ht = getElementWiseResult(ctx,
        *getElementWiseResult(ctx, *getElementWiseResult(ctx, *constOne, *zt, eOp::kSUB), *ht, eOp::kPROD),
        *getElementWiseResult(ctx, *zt, *Ht1Output, eOp::kPROD), eOp::kSUM);

    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape = getElementWiseResult(ctx, *gateOutputShape,
        *addConstant(ctx, std::vector<int32_t>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
            nvinfer1::Dims{1, {3}})
             ->getOutput(0),
        nvinfer1::ElementWiseOperation::kDIV);
    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1Output, Ht, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ht1Output, Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};

    // Add outputs. In ONNX, all GRU outputs are optional, so check for them here.
    auto shouldAddOutput = [&node](int32_t index) { return index < node.output().size() && !node.output(index).empty(); };

    // Y = concatenation of all H(t) for each element of the sequence
    if (shouldAddOutput(0))
    {
        outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
            numDirections, inputs, direction == "reverse"));
    }
    // Yh = last value of H(t)
    if (shouldAddOutput(1))
    {
        outputs.emplace_back(N_CHECK(loop->addLoopOutput(*Ht1Output, nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0)));
    }
    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(HammingWindow)
{
    /***

    Operation returns a window vector, where:

    Y[n] = alpha - beta * cos(2*pi*n / N)

    Where N is the window length, and n is each element in the window.

    Note that if `periodic == 0`, the denominator becomes N - 1.

    This can be represented by creating a range 'n' from 0 -> N, and performing the operations elementwise.

    Note that in the ONNX op definition alpha and beta are not provided. We will use the default values defined in ONNX:

        alpha = 25/46
        beta = 1 - alpha

    ***/

    OnnxAttrs attrs(node, ctx);
    int32_t outputDtype = attrs.get<int32_t>("output_datatype", 1);
    int32_t periodic = attrs.get<int32_t>("periodic", 1);
    ONNXTRT_CHECK_NODE(outputDtype == 1, "Output must be float32-type!", node, nodeIdx, ErrorCode::kINVALID_NODE);

    constexpr float alpha = 25.F / 46.F;
    constexpr float beta = 1.F - alpha;

    auto* N = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE(
        N->getDimensions().nbDims == 0, "Window length must be a scalar!", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto* window = generateWindow(ctx, N);

    auto* cosOutput = windowHelper(ctx, 2.F * M_PI, window, N, nvinfer1::UnaryOperation::kCOS, periodic);

    auto betaTensor
        = N_CHECK(addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto betaLayer
        = N_CHECK(ctx->network()->addElementWise(*betaTensor, *cosOutput, nvinfer1::ElementWiseOperation::kPROD));
    auto betaOutput = N_CHECK(betaLayer->getOutput(0));

    auto alphaTensor
        = N_CHECK(addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto alphaLayer
        = N_CHECK(ctx->network()->addElementWise(*alphaTensor, *betaOutput, nvinfer1::ElementWiseOperation::kSUB));

    RETURN_FIRST_OUTPUT(alphaLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(HannWindow)
{
    /***

    Operation returns a window vector, where:

    Y[n] = sin^2(pi*n / N)

    Where N is the window length, and n is each element in the window.

    Note that if `periodic == 0`, the denominator becomes N - 1.

    This can be represented by creating a range 'n' from 0 -> N, and performing the operations elementwise.

    ***/

    OnnxAttrs attrs(node, ctx);
    int32_t outputDtype = attrs.get<int32_t>("output_datatype", 1);
    int32_t periodic = attrs.get<int32_t>("periodic", 1);
    ONNXTRT_CHECK_NODE(outputDtype == 1, "Output must be float32-type!", node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto* N = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE(
        N->getDimensions().nbDims == 0, "Window length must be a scalar!", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto* window = generateWindow(ctx, N);

    auto sinOutput = windowHelper(ctx, M_PI, window, N, nvinfer1::UnaryOperation::kSIN, periodic);

    auto sinSquaredLayer
        = N_CHECK(ctx->network()->addElementWise(*sinOutput, *sinOutput, nvinfer1::ElementWiseOperation::kPROD));

    RETURN_FIRST_OUTPUT(sinSquaredLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Hardmax)
{
    checkNotInvalidType(inputs.at(0), {"INT64", "INT32", "INT8", "UINT8", "BOOL"}, node, nodeIdx);
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* values = &convertToTensor(inputs.at(0), ctx);
    auto originalDims = values->getDimensions();

    int32_t axis = attrs.get("axis", ctx->getOpsetVersion() < 13 ? 1 : -1);
    convertAxis(axis, originalDims.nbDims, node, nodeIdx);
    if (ctx->getOpsetVersion() < 13)
    {
        // Reshape into 2D tensor
        values = flattenTensor(ctx, node, *values, axis, true);
        axis = 1;
    }
    uint32_t axisMask = 1 << axis;

    auto* topKLayer = N_CHECK(ctx->network()->addTopK(*values, nvinfer1::TopKOperation::kMAX, /* k */ 1, axisMask));

    auto* squeezedIndices = squeezeTensor(ctx, *topKLayer->getOutput(1), {axis});
    auto* zeroOneTensor = N_CHECK(addConstant(ctx, std::vector<int32_t>{0, 1},
        ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
        nvinfer1::Dims{1, {2}})->getOutput(0));
    auto* depth = getAxisLength(ctx, values, axis, nvinfer1::Dims{0});
    auto* oneHotLayer = N_CHECK(ctx->network()->addOneHot(*squeezedIndices, *zeroOneTensor, *depth, axis));
    auto* oneHotOutput = N_CHECK(oneHotLayer->getOutput(0));

    if (ctx->getOpsetVersion() < 13)
    {
        oneHotOutput = reshapeTensor(ctx, *oneHotOutput, originalDims);
    }
    auto* output = castHelper(ctx, oneHotOutput, values->getType());
    return {{output}};
}

DEFINE_BUILTIN_OP_IMPORTER(HardSigmoid)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.2f);
    float beta = attrs.get<float>("beta", 0.5f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Identity)
{
    auto* layer = N_CHECK(ctx->network()->addIdentity(convertToTensor(inputs.at(0), ctx)));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(If)
{
    OnnxAttrs attrs(node, ctx);
    auto cond = inputs.at(0);

    // Push current scope name to top of stack in order to properly set layer metadata and print error message.
    ctx->localFunctionStack().push_back({node.op_type(), getNodeName(node), {}});

    ::ONNX_NAMESPACE::GraphProto const& thenGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("then_branch");
    ::ONNX_NAMESPACE::GraphProto const& elseGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("else_branch");

    // Number of outputs are the same between the two branches.
    ONNXTRT_CHECK_NODE(thenGraph.output_size() == elseGraph.output_size(),
        "then/else subgraphs should have the same number of outputs: then outputs = "
            << thenGraph.output_size() << ", else outputs = " << elseGraph.output_size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    int32_t const nbOutputs = thenGraph.output_size();
    std::vector<TensorOrWeights> graphOutputs;

    // For constant conditions, parse only the selected subgraph
    if (cond.is_weights() && cond.weights().count() == 1)
    {
        // Boolean weights are stored as uint8_t
        auto const value = *(static_cast<uint8_t*>(cond.weights().values));
        ::ONNX_NAMESPACE::GraphProto const& body = value == 1 ? thenGraph : elseGraph;

        // Establish scope for names local to the subgraph.
        NameScope nameScope(*ctx);

        std::vector<Status> errors{};
        onnx2trt::parseGraph(
            ctx, body, errors, /*deserializingINetwork=*/false, /*currentNode=*/nullptr, /*subgraphParentIdx=*/nodeIdx);
        for (int32_t i = 0; i < nbOutputs; i++)
        {
            graphOutputs.emplace_back(ctx->tensors().at(body.output(i).name()));
        }
        return {graphOutputs};
    }

    //
    // The condition is not a build-time constant. Construct an if-conditional construct.
    //

    // The `condition` tensor must be a scalar boolean.
    auto* condTensor = convertToScalar(ctx, &convertToTensor(cond, ctx));
    ONNXTRT_CHECK_NODE(
        condTensor, "Failed to convert the input cond to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto conditional = N_CHECK(ctx->network()->addIfConditional());
    conditional->setName(getNodeName(node).c_str());
    conditional->setCondition(*condTensor);

    std::vector<nvinfer1::ILayer*> thenLayers, elseLayers;
    std::vector<TensorOrWeights> thenSubgraphTensors;
    std::vector<TensorOrWeights> elseSubgraphTensors;

    ctx->localFunctionStack().push_back({"then_branch", thenGraph.name(), {}});
    importSubgraph(ctx, thenGraph, thenLayers, thenSubgraphTensors, /*subgraphParentIdx=*/nodeIdx);
    ctx->localFunctionStack().pop_back();

    ctx->localFunctionStack().push_back({"else_branch", elseGraph.name(), {}});
    importSubgraph(ctx, elseGraph, elseLayers, elseSubgraphTensors, /*subgraphParentIdx=*/nodeIdx);
    ctx->localFunctionStack().pop_back();

    using InputsMap = std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer*>;
    InputsMap inputsMap;
    ctx->localFunctionStack().push_back({"then_branch", thenGraph.name(), {}});
    addIfInputLayers(ctx, conditional, inputsMap, thenLayers, &node);
    ctx->localFunctionStack().pop_back();
    ctx->localFunctionStack().push_back({"else_branch", elseGraph.name(), {}});
    addIfInputLayers(ctx, conditional, inputsMap, elseLayers, &node);
    ctx->localFunctionStack().pop_back();

    ONNXTRT_CHECK_NODE(thenSubgraphTensors.size() == elseSubgraphTensors.size(),
        "Found different number of output tensors in If conditional subgraphs! then outputs = "
            << thenSubgraphTensors.size() << ", else outputs = " << elseSubgraphTensors.size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    ctx->localFunctionStack().pop_back();

    for (size_t i = 0; i < thenSubgraphTensors.size(); i++)
    {
        auto* thenOut = &convertToTensor(thenSubgraphTensors[i], ctx);
        auto* elseOut = &convertToTensor(elseSubgraphTensors[i], ctx);
        auto* outputLayer = N_CHECK(conditional->addOutput(*thenOut, *elseOut));
        ctx->registerLayer(outputLayer, std::string(conditional->getName()) + "_OutputLayer", &node);
        graphOutputs.emplace_back(N_CHECK(outputLayer->getOutput(0)));
    }

    return {graphOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(ImageScaler)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs{node, ctx};
    // Shift the input by a per-channel bias value.
    std::vector<float> biases = attrs.get<std::vector<float>>("bias");
    nvinfer1::Dims dims{1, {static_cast<int32_t>(biases.size())}};
    ShapedWeights shiftWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::copy(biases.begin(), biases.end(), static_cast<float*>(shiftWeights.values));
    // Scale is applied to every element of the input, but we need to duplicate it over every channel.
    float scale = attrs.get<float>("scale", 1.0f);
    ShapedWeights scaleWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::fill(static_cast<float*>(scaleWeights.values), static_cast<float*>(scaleWeights.values) + scaleWeights.count(),
        scale);
    // Finally add the scale layer.
    auto layer = N_CHECK(ctx->network()->addScale(
        tensor, nvinfer1::ScaleMode::kCHANNEL, shiftWeights, scaleWeights, nvinfer1::Weights{}));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    auto inputDataType = inputs.at(0).getDataType();
    auto scaleDataType = inputs.at(1).getDataType();
    auto biasDataType = inputs.at(2).getDataType();

    ONNXTRT_CHECK_NODE((inputDataType == scaleDataType && scaleDataType == biasDataType),
        "Inputs must be either all FLOAT or all FLOAT16. Input type = " + getTrtDtypeName(inputDataType)
            + ", scale type = " + getTrtDtypeName(scaleDataType) + ", bias type = " + getTrtDtypeName(biasDataType)
            + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Choose plugin implementation if requested by the user. kNATIVE_INSTANCENORM is ON by default.
    auto flags = ctx->getFlags();
    uint32_t nativeInstanceNormFlag = 1U << static_cast<uint32_t>(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);
    if (flags & nativeInstanceNormFlag)
    {
        return normalizationHelper(ctx, node, nodeIdx, inputs, /*useV2*/ true);
    }
    ONNXTRT_CHECK_NODE((inputDataType == DataType::kFLOAT || inputDataType == DataType::kHALF),
        "Inputs to InstanceNorm plugin must be either FLOAT or FLOAT16. Input type is " + getTrtDtypeName(inputDataType)
            + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    return instanceNormPluginHelper(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(IsInf)
{
    OnnxAttrs attrs{node, ctx};
    int32_t const detectNegative = attrs.get<int32_t>("detect_negative", 1);
    int32_t const detectPositive = attrs.get<int32_t>("detect_positive", 1);

    if (detectNegative && detectPositive)
    {
        return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kISINF);
    }

    auto& input = convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = getNbDims(&input);
    auto& zeroTensor = *addConstantScalar(ctx, 0.F, ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims)->getOutput(0);

    if (detectNegative)
    {
        auto* isNegLayer
            = N_CHECK(ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kLESS));
        auto* isNeg = N_CHECK(isNegLayer->getOutput(0));
        auto* isInfLayer = N_CHECK(ctx->network()->addUnary(input, nvinfer1::UnaryOperation::kISINF));
        auto* isInf = N_CHECK(isInfLayer->getOutput(0));
        auto* finalLayer = ctx->network()->addElementWise(*isNeg, *isInf, nvinfer1::ElementWiseOperation::kAND);
        RETURN_FIRST_OUTPUT(finalLayer, node, nodeIdx);
    }
    if (detectPositive)
    {
        auto* isPosLayer
            = N_CHECK(ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kGREATER));
        auto* isPos = N_CHECK(isPosLayer->getOutput(0));
        auto* isInfLayer = N_CHECK(ctx->network()->addUnary(input, nvinfer1::UnaryOperation::kISINF));
        auto* isInf = N_CHECK(isInfLayer->getOutput(0));
        auto* finalLayer = ctx->network()->addElementWise(*isPos, *isInf, nvinfer1::ElementWiseOperation::kAND);
        RETURN_FIRST_OUTPUT(finalLayer, node, nodeIdx);
    }
    // In this case, always return false.
    auto* isPosLayer
        = N_CHECK(ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kGREATER));
    auto* isPos = N_CHECK(isPosLayer->getOutput(0));
    auto* isNegLayer
        = N_CHECK(ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kLESS));
    auto* isNeg = N_CHECK(isNegLayer->getOutput(0));
    auto* finalLayer = ctx->network()->addElementWise(*isPos, *isNeg, nvinfer1::ElementWiseOperation::kAND);
    RETURN_FIRST_OUTPUT(finalLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(IsNaN)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kISNAN);
}

DEFINE_BUILTIN_OP_IMPORTER(LayerNormalization)
{
    auto* input = &convertToTensor(inputs.at(0), ctx);

    auto dt = input->getType();
    int32_t nbDims = getNbDims(input);

    nvinfer1::ITensor* scale{nullptr};
    if (inputs.at(1).isNullTensor())
    {
        if (dt == DataType::kHALF)
        {
            scale = addConstantScalar(
                ctx, static_cast<half_float::half>(1), ::ONNX_NAMESPACE::TensorProto::FLOAT16, nbDims)
                        ->getOutput(0);
        }
        else if (dt == DataType::kBF16)
        {
            scale = addConstantScalar(ctx, static_cast<BFloat16>(1), ::ONNX_NAMESPACE::TensorProto::BFLOAT16, nbDims)
                        ->getOutput(0);
        }
        else
        {
            scale = addConstantScalar(ctx, static_cast<float>(1), ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims)
                        ->getOutput(0);
        }
    }
    else
    {
        scale = &convertToTensor(inputs.at(1), ctx);
        broadcastTensors(ctx, input, scale);
    }
    N_CHECK(scale);

    nvinfer1::ITensor* bias{nullptr};
    if (inputs.size() < 3 || inputs.at(2).isNullTensor())
    {
        if (dt == DataType::kHALF)
        {
            bias = addConstantScalar(
                ctx, static_cast<half_float::half>(0), ::ONNX_NAMESPACE::TensorProto::FLOAT16, nbDims)
                       ->getOutput(0);
        }
        else if (dt == DataType::kBF16)
        {
            bias = addConstantScalar(ctx, static_cast<BFloat16>(0), ::ONNX_NAMESPACE::TensorProto::BFLOAT16, nbDims)
                       ->getOutput(0);
        }
        else
        {
            bias = addConstantScalar(ctx, static_cast<float>(0), ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims)
                       ->getOutput(0);
        }
    }
    else
    {
        bias = &convertToTensor(inputs.at(2), ctx);
        broadcastTensors(ctx, input, bias);
    }
    N_CHECK(bias);

    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    int32_t axis = attrs.get("axis", -1);
    nvinfer1::DataType computeType = nvinfer1::DataType::kFLOAT;
    convertDtype(attrs.get<int32_t>("stash_type", 1), &computeType);

    convertAxis(axis, nbDims, node, nodeIdx);
    uint32_t axesMask{0};

    // Populate axesMask with axis values
    for (int32_t i = axis; i < nbDims; i++)
    {
        axesMask |= 1 << i;
    }

    auto* layer = N_CHECK(ctx->network()->addNormalizationV2(*input, *scale, *bias, axesMask));
    layer->setEpsilon(epsilon);
    auto const stronglyTyped = ctx->isStronglyTyped();
    if (!stronglyTyped)
    {
        layer->setComputePrecision(computeType);
    }
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(LeakyRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.01f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kLEAKY_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Less)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kLESS);
}

DEFINE_BUILTIN_OP_IMPORTER(LessOrEqual)
{
    return greaterLessOrEqual(ctx, node, nodeIdx, &convertToTensor(inputs.at(0), ctx),
        &convertToTensor(inputs.at(1), ctx),
        /*greater*/ false);
}

DEFINE_BUILTIN_OP_IMPORTER(Log)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    auto& input = convertToTensor(inputs.at(0), ctx);
    // Don't use softmax converter since it adds a shuffle layer
    // which prevents the builder to fuse softmax and log operations.
    auto* softmax = addSoftmax(ctx, node, nodeIdx, input);
    nvinfer1::IUnaryLayer* unaryLayer = N_CHECK(ctx->network()->addUnary(*softmax, nvinfer1::UnaryOperation::kLOG));
    auto unaryOutput = N_CHECK(unaryLayer->getOutput(0));
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *unaryOutput, shapeOf(input));
    RETURN_FIRST_OUTPUT(reshapeLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Loop)
{
    constexpr int32_t NB_NON_STATE_INPUTS = 2; // First 2 inputs are trip count and condition respectively.
    constexpr int32_t NB_DISCARDED_OUTPUTS
        = 1; // First output is the updated value of the condition, and is ignored by the outer loop node.
    ONNXTRT_CHECK_NODE((inputs.size() >= 2),
        "The Loop operator requires at least 2 inputs. The current number of inputs = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    int32_t const nbInputs = node.input().size();
    // The number of state variables on the input and output is the same.
    int32_t const nbStateVars = nbInputs - NB_NON_STATE_INPUTS;

    ::ONNX_NAMESPACE::GraphProto const& body = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("body");

    auto loop = N_CHECK(ctx->network()->addLoop());
    loop->setName(getNodeName(node).c_str());

    // Establish scope for names local to the subgraph.
    NameScope nameScope(*ctx);

    // Trip count and condition are optional inputs.
    nvinfer1::ITensor* tripLimit{nullptr};
    if (inputs[0])
    {
        // Some convertors will use INT_MAX to signify "use cond input as loop termination". From TRT's perspective,
        // we can just treat these cases as an empty tripLimit.
        bool const isMaxTripCount = inputs[0].is_weights() && inputs[0].isInt64()
            && static_cast<int64_t*>(inputs[0].weights().values)[0]
                >= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        if (!isMaxTripCount)
        {
            tripLimit = convertToScalar(ctx, &convertToTensor(inputs[0], ctx));
            tripLimit = castHelper(ctx, tripLimit, DataType::kINT32);
            ONNXTRT_CHECK_NODE(tripLimit, "Failed to convert the trip-count input to a scalar.", node, nodeIdx,
                ErrorCode::kINVALID_NODE);
            ctx->loopTensors()[body.input(0).name()] = node.input(0);
            loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
            // First graph input is iteration_num, so create a loop counter
            auto counter = convertToScalar(ctx, addLoopCounter(ctx, loop, 0));
            ctx->registerTensor(counter, body.input(0).name());
        }
    }
    nvinfer1::ITensor* cond{nullptr};
    if (inputs[1])
    {
        cond = convertToScalar(ctx, &convertToTensor(inputs[1], ctx));
        ONNXTRT_CHECK_NODE(
            cond, "Failed to convert the input cond to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        ctx->loopTensors()[body.input(1).name()] = node.input(1);
        ctx->registerTensor(cond, body.input(1).name());
    }
    // Add initial state inputs using recurrent layers.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (size_t i = 2; i < inputs.size(); ++i)
    {
        stateVars.emplace_back(N_CHECK(loop->addRecurrence(convertToTensor(inputs[i], ctx))));
        ctx->loopTensors()[body.input(i).name()] = node.input(i);
        ctx->registerTensor(TensorOrWeights{N_CHECK(stateVars.back()->getOutput(0))}, body.input(i).name());
        LOG_VERBOSE("Mapped Loop node input " << node.input(i) << " to loop body input " << body.input(i).name());
    }

    // Loop body
    std::vector<Status> errors{};
    onnx2trt::parseGraph(
        ctx, body, errors, /*deserializingINetwork=*/false, /*currentNode=*/nullptr, /*subgraphParentIdx=*/nodeIdx);

    if (cond)
    {
        // Add recurrence for loop condition
        auto recurrence = N_CHECK(loop->addRecurrence(*cond));
        auto const& bodyOutputName = body.output(0).name();
        auto condOutput = convertToScalar(ctx, &convertToTensor(ctx->tensors().at(bodyOutputName), ctx));
        recurrence->setInput(1, *condOutput);
        auto recurrenceOutput = N_CHECK(recurrence->getOutput(0));
        loop->addTripLimit(*recurrenceOutput, nvinfer1::TripLimit::kWHILE);
    }

    // Set final values of state variables.
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int32_t i = 0; i < nbStateVars; ++i)
    {
        // The first output of the body graph is the updated condition, which is ignored by the Loop node.
        int32_t const index = i + NB_DISCARDED_OUTPUTS;
        auto const& bodyOutputName = body.output(index).name();
        auto& stateOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For state variable output: " << bodyOutputName
                                                  << ", found matching tensor: " << stateOutput.getName()
                                                  << ", with shape: " << stateOutput.getDimensions());
        stateVars.at(i)->setInput(1, stateOutput);
        // Each state variable is also a loop output
        auto stateVarOutput = N_CHECK(stateVars.at(i)->getOutput(0));
        auto stateOutputLayer = N_CHECK(loop->addLoopOutput(*stateVarOutput, nvinfer1::LoopOutput::kLAST_VALUE));
        auto stateOutputTensor = N_CHECK(stateOutputLayer->getOutput(0));
        nodeOutputs.emplace_back(stateOutputTensor);
    }
    int32_t const nbOutputs = body.output_size();
    // Finally, set up scan outputs if there are any
    for (int32_t i = nbStateVars + NB_DISCARDED_OUTPUTS; i < nbOutputs; ++i)
    {
        auto const& bodyOutputName = body.output(i).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
                                        << ", with shape: " << scanOutput.getDimensions());
        nvinfer1::ILoopOutputLayer* trtScanOut
            = N_CHECK(loop->addLoopOutput(scanOutput, nvinfer1::LoopOutput::kCONCATENATE, 0));
        // If trip limit is set, we can set the loop output to the tripLimit, otherwise, set to some dummy constant
        // value.
        // In the latter case, the scan outputs must not be used in the rest of the model.
        if (tripLimit)
        {
            if (cond)
            {
                LOG_WARNING("For loop node "
                    << node.name()
                    << " both a trip limit and a loop termination condition was set! If the number of iterations of "
                       "the loop is less than the trip limit during runtime, scan output "
                    << scanOutput.getName()
                    << " may have the wrong shape. For consistent results it's recommended to export loops with only a "
                       "trip limit.");
            }
            trtScanOut->setInput(1, *tripLimit);
        }
        else
        {
            ONNXTRT_CHECK_NODE(false,
                "TensorRT cannot infer the shape of scan output "
                    << scanOutput.getName()
                    << " since no trip limit was set. For better compatibility, it's recommended to export loops with "
                       "only a trip limit.",
                node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        }
        nodeOutputs.emplace_back(N_CHECK(trtScanOut->getOutput(0)));
    }

    return {nodeOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(LRN)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int size = attrs.get<int>("size");
    float alpha = attrs.get<float>("alpha", 0.0001f);
    float beta = attrs.get<float>("beta", 0.75f);
    float bias = attrs.get<float>("bias", 1.0f);
    auto* layer = N_CHECK(ctx->network()->addLRN(tensor, size, alpha, beta, bias));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(LSTM)
{
    using trtAct = nvinfer1::ActivationType;
    using eOp = nvinfer1::ElementWiseOperation;

    OnnxAttrs attrs{node, ctx};
    constexpr int32_t NUM_GATES = 4;
    std::string const direction = attrs.get<std::string>("direction", "forward");
    int32_t const numDirections = (direction == "bidirectional") ? 2 : 1;
    int32_t const hiddenSize = attrs.get<int>("hidden_size");
    float const clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);

    std::vector<trtAct> defaultActs{trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH});
    }
    std::vector<trtAct> activations = attrs.get<std::vector<trtAct>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    activationAlphas = parseLSTMActivationValues(activations, activationAlphas, true);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    activationBetas = parseLSTMActivationValues(activations, activationBetas, false);

    // Roll Rb into Wb (and RBb into WBb). Bias is in the form  [Wb[iofc], Rb[iofc], WBb[iofc], RBb[iofc]].
    // So reshape such that we can perform a reduction to add Wb and Rb.
    nvinfer1::ITensor* combinedBias{nullptr};
    if (inputs.size() > 3 && inputs.at(3))
    {
        nvinfer1::ITensor* bias = &convertToTensor(inputs.at(3), ctx);
        LOG_VERBOSE("Bias shape is: " << bias->getDimensions());
        // Reshape to [[Wb[iofc], Rb[iofc]], [WBb[iofc], RBb[iofc]]]
        nvinfer1::IShuffleLayer* reshapeBias = N_CHECK(ctx->network()->addShuffle(*bias));
        reshapeBias->setReshapeDimensions(nvinfer1::Dims3{numDirections, 2, NUM_GATES * hiddenSize});
        reshapeBias->setZeroIsPlaceholder(false);
        auto* reshapeBiasOut = N_CHECK(reshapeBias->getOutput(0));
        LOG_VERBOSE("Reshaping bias to: " << reshapeBiasOut->getDimensions());
        auto reduceLayer
            = N_CHECK(ctx->network()->addReduce(*reshapeBiasOut, nvinfer1::ReduceOperation::kSUM, /*axis=*/0b010,
                /*keepDimensions=*/true));
        combinedBias = N_CHECK(reduceLayer->getOutput(0));
        LOG_VERBOSE("After reduction, bias shape is: " << combinedBias->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor
            = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0);
        LOG_VERBOSE("numDirectionsTensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor
            = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0);
        LOG_VERBOSE("hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, nvinfer1::Dims{1, {1}});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        std::array<nvinfer1::ITensor*, 3> tensors{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}};
        nvinfer1::IConcatenationLayer* concatenatedShape = N_CHECK(ctx->network()->addConcatenation(tensors.data(), 3));
        return N_CHECK(concatenatedShape->getOutput(0));
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    auto const getInitialInputValue = [&ctx, &gateOutputShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        auto tensorType = inputs.at(0).getType();
        if (tensorType == "HALF")
        {
            return constantOfShape(ctx,
                addConstantScalar(ctx, half_float::half(0.f), ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, 1)
                    ->getOutput(0),
                gateOutputShape);
        }
        if (tensorType == "BF16")
        {
            return constantOfShape(ctx,
                addConstantScalar(ctx, BFloat16(0.f), ::ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, 1)->getOutput(0),
                gateOutputShape);
        }
        return constantOfShape(ctx,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0),
            gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::ITensor* initialCellState = getInitialInputValue(6);
    LOG_VERBOSE("Initial cell state shape: " << initialCellState->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = N_CHECK(ctx->network()->addLoop());
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);
    ONNXTRT_CHECK_NODE(iterationInput, "Failed to add RNN input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* Ht1 = N_CHECK(loop->addRecurrence(*initialHidden));
    ctx->registerLayer(Ht1, node);
    auto Ht1Output = N_CHECK(Ht1->getOutput(0));
    LOG_VERBOSE("Hidden state shape: " << Ht1Output->getDimensions());

    // C(t-1)
    nvinfer1::IRecurrenceLayer* Ct1 = loop->addRecurrence(*initialCellState);
    auto Ct1Output = N_CHECK(Ct1->getOutput(0));
    LOG_VERBOSE("Cell state shape: " << Ct1Output->getDimensions());

    // Compute intermediate(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)). intermediate(t) has shape (numDirections,
    // batchSize, 4 * hiddenSize)
    nvinfer1::ITensor* xtWT = ctx->network()
                                  ->addMatrixMultiply(*iterationInput, nvinfer1::MatrixOperation::kNONE, *weights,
                                      nvinfer1::MatrixOperation::kTRANSPOSE)
                                  ->getOutput(0);
    LOG_VERBOSE("X(t) * W^T -> " << xtWT->getDimensions());

    nvinfer1::ITensor* ht1RT = ctx->network()
                                   ->addMatrixMultiply(*Ht1Output, nvinfer1::MatrixOperation::kNONE, *recurrenceWeights,
                                       nvinfer1::MatrixOperation::kTRANSPOSE)
                                   ->getOutput(0);
    LOG_VERBOSE("H(t-1) * R^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* intermediatet = getElementWiseResult(ctx, *xtWT, *ht1RT, eOp::kSUM);
    if (combinedBias)
    {
        intermediatet = getElementWiseResult(ctx, *intermediatet, *combinedBias, eOp::kSUM);
    }
    LOG_VERBOSE("intermediate(t) -> " << intermediatet->getDimensions());

    // Gate shape is (numDirections, batchSize, hiddenSize)
    auto const isolateGate
        = [&ctx, &hiddenSize, &gateOutputShape](nvinfer1::ITensor* gates, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolate = N_CHECK(ctx->network()->addSlice(
            *gates, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1}));
        ;
        isolate->setInput(1,
            *addConstant(ctx, std::vector<int32_t>{0, 0, gateIndex * hiddenSize},
                ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, {3}})
                 ->getOutput(0));               // Start
        isolate->setInput(2, *gateOutputShape); // Size
        return N_CHECK(isolate->getOutput(0));
    };

    // Compute peephole connections
    nvinfer1::ITensor* peephole{nullptr};
    if (inputs.size() > 7 && inputs.at(7))
    {
        peephole = &convertToTensor(inputs.at(7), ctx);
    }

    auto const addPeephole = [&ctx, &hiddenSize, &numDirections, &peephole](nvinfer1::ITensor* gate,
                                 nvinfer1::ITensor* cellState, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolatePeephole
            = N_CHECK(ctx->network()->addSlice(*peephole, nvinfer1::Dims2{0, gateIndex * hiddenSize},
                nvinfer1::Dims2{numDirections, hiddenSize}, nvinfer1::Dims2{1, 1}));
        auto* peepholeWeights = unsqueezeTensor(ctx, *isolatePeephole->getOutput(0), std::vector<int32_t>{1});
        LOG_VERBOSE("Peephole weight for gate: " << gateIndex << " shape: " << peepholeWeights->getDimensions());

        return getElementWiseResult(
            ctx, *gate, *getElementWiseResult(ctx, *peepholeWeights, *cellState, eOp::kPROD), eOp::kSUM);
    };

    // NOTE: . represents a hadamard product
    nvinfer1::ITensor* itGate = isolateGate(intermediatet, 0);

    if (peephole)
    {
        // i(t) (w/ peephole) =  i(t) + Pi . C(t-1)
        itGate = addPeephole(itGate, Ct1Output, 0);
    }

    nvinfer1::IActivationLayer* itGateAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, itGate, clip), activations.at(0)));
    itGateAct->setAlpha(activationAlphas.at(0));
    itGateAct->setBeta(activationBetas.at(0));
    itGate = N_CHECK(itGateAct->getOutput(0));

    nvinfer1::ITensor* ftGate = isolateGate(intermediatet, 2);

    if (peephole)
    {
        // f(t) (w/ peephole) =  f(t) + Pf . C(t-1)
        ftGate = addPeephole(ftGate, Ct1Output, 2);
    }

    nvinfer1::IActivationLayer* ftGateAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, ftGate, clip), activations.at(0)));
    ftGateAct->setAlpha(activationAlphas.at(0));
    ftGateAct->setBeta(activationBetas.at(0));
    ftGate = N_CHECK(ftGateAct->getOutput(0));

    // c(t) = g(intermediate(t)[:, :, 3H:4H])
    nvinfer1::IActivationLayer* ctAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, isolateGate(intermediatet, 3), clip), activations.at(1)));
    ctAct->setAlpha(activationAlphas.at(1));
    ctAct->setBeta(activationBetas.at(1));

    nvinfer1::ITensor* ctGate = N_CHECK(ctAct->getOutput(0));
    LOG_VERBOSE("c(t) -> " << ctGate->getDimensions());

    // C(t) = f(t) . C(t - 1) + i(t) . c(t)
    nvinfer1::ITensor* operandIC = getElementWiseResult(ctx, *itGate, *ctGate, eOp::kPROD);
    nvinfer1::ITensor* operandFC = getElementWiseResult(ctx, *ftGate, *Ct1Output, eOp::kPROD);
    nvinfer1::ITensor* Ct = getElementWiseResult(ctx, *operandFC, *operandIC, eOp::kSUM);

    nvinfer1::ITensor* singlePassShape = getElementWiseResult(ctx, *gateOutputShape,
        *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
            nvinfer1::Dims{1, {3}})
             ->getOutput(0),
        eOp::kDIV);

    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ct = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ct1Output, Ct, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ct1Output, Ct, maxLen, direction == "reverse");
    }

    Ct1->setInput(1, *Ct);
    LOG_VERBOSE("C(t) -> " << Ct->getDimensions());

    nvinfer1::ITensor* otGate = isolateGate(intermediatet, 1);

    if (peephole)
    {
        // o(t) (w/ peephole) =  o(t) + Po . C(t)
        otGate = addPeephole(otGate, Ct, 1);
    }

    nvinfer1::IActivationLayer* otGateAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, otGate, clip), activations.at(0)));
    otGateAct->setAlpha(activationAlphas.at(0));
    otGateAct->setBeta(activationBetas.at(0));
    otGate = N_CHECK(otGateAct->getOutput(0));

    // H(t) = o(t) . h(C(t))
    nvinfer1::IActivationLayer* hAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, Ct, clip), activations.at(2)));
    hAct->setAlpha(activationAlphas.at(2));
    hAct->setBeta(activationBetas.at(2));
    auto hActTensor = N_CHECK(hAct->getOutput(0));

    nvinfer1::ITensor* Ht = getElementWiseResult(ctx, *otGate, *hActTensor, eOp::kPROD);
    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1Output, Ht, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ht1Output, Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};

    // Add outputs. In ONNX, all LSTM outputs are optional, so check for them here.
    auto shouldAddOutput = [&node](int32_t index) { return index < node.output().size() && !node.output(index).empty(); };

    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)
    if (shouldAddOutput(0))
    {
        outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
            numDirections, inputs, direction == "reverse"));
    }
    // Yh = last value of H(t)
    if (shouldAddOutput(1))
    {
        auto yhLayer = N_CHECK(loop->addLoopOutput(*Ht1Output, nvinfer1::LoopOutput::kLAST_VALUE));
        outputs.emplace_back(N_CHECK(yhLayer->getOutput(0)));
    }
    // Yc = last value of C(t)
    if (shouldAddOutput(2))
    {
        auto ycLayer = N_CHECK(loop->addLoopOutput(*Ct1Output, nvinfer1::LoopOutput::kLAST_VALUE));
        outputs.emplace_back(N_CHECK(ycLayer->getOutput(0)));
    }

    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(LpNormalization)
{
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    using rOp = nvinfer1::ReduceOperation;

    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    auto inputType = input->getType();
    int32_t axis = attrs.get<int32_t>("axis", -1);
    int32_t p = attrs.get<int32_t>("p", 2);
    int32_t nbDims = input->getDimensions().nbDims;
    DataType dt = input->getType();
    ONNXTRT_CHECK_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF),
        "Only float inputs/outputs supported in LpNormalization. The current data type = " + getTrtDtypeName(dt) + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    convertAxis(axis, nbDims, node, nodeIdx);

    nvinfer1::ITensor* norm{nullptr};
    TensorOrWeights zeros = ctx->createNamedTempWeights(trtDataTypeToONNX(inputType), {0, {}});
    nvinfer1::ITensor* zerosTensor = &convertToTensor(zeros, ctx);
    broadcastTensor(ctx, zerosTensor, nbDims);

    if (p == 1)
    {
        // abs(x)
        nvinfer1::IUnaryLayer* absLayer = N_CHECK(ctx->network()->addUnary(*input, uOp::kABS));
        ctx->registerLayer(absLayer, node);
        norm = N_CHECK(absLayer->getOutput(0));

        // norm coeff = sum(abs(x)) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = N_CHECK(ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true));
        ctx->registerLayer(reduceLayer, node);
        norm = N_CHECK(reduceLayer->getOutput(0));
    }
    else if (p == 2)
    {
        // x^2
        auto* sqrLayer = N_CHECK(ctx->network()->addElementWise(*input, *input, eOp::kPROD));
        ctx->registerLayer(sqrLayer, node);
        norm = N_CHECK(sqrLayer->getOutput(0));

        // sum(x^2) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = N_CHECK(ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true));
        ctx->registerLayer(reduceLayer, node);
        norm = N_CHECK(reduceLayer->getOutput(0));

        // norm coeff = sqrt(sum(x^2))
        nvinfer1::IUnaryLayer* sqrtLayer = N_CHECK(ctx->network()->addUnary(*norm, uOp::kSQRT));
        ctx->registerLayer(sqrtLayer, node);
        norm = N_CHECK(sqrtLayer->getOutput(0));
    }

    // norm coeff |= 1 (change 0s to 1s, leave all other values same)
    nvinfer1::IElementWiseLayer* maskLayer = N_CHECK(ctx->network()->addElementWise(*norm, *zerosTensor, eOp::kEQUAL));
    ctx->registerLayer(maskLayer, node);
    nvinfer1::ITensor* mask = N_CHECK(maskLayer->getOutput(0));
    mask = castHelper(ctx, mask, dt);
    auto* combinedLayer = N_CHECK(ctx->network()->addElementWise(*norm, *mask, eOp::kSUM));
    ctx->registerLayer(combinedLayer, node);
    norm = N_CHECK(combinedLayer->getOutput(0));

    // x/(norm coeff)
    // norm tensor is broadcast along axis dimension to match shape of input
    auto* layer = N_CHECK(ctx->network()->addElementWise(*input, *norm, eOp::kDIV));
    ctx->registerLayer(layer, node);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(LpPool)
{
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    using pType = nvinfer1::PoolingType;

    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    int32_t p = attrs.get<int32_t>("p", 2);
    int32_t nbDims = input->getDimensions().nbDims;
    int32_t nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;

    DataType dt = input->getType();
    ONNXTRT_CHECK_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF),
        "Only float inputs/outputs supported in LpPool. The current data type = " + getTrtDtypeName(dt) + ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    nvinfer1::Dims kernelShape = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool excludePadding(false);
    bool ceilMode = static_cast<bool>(attrs.get<int32_t>("ceil_mode", 0));
    getKernelParams(ctx, node, &kernelShape, &strides, &begPadding, &endPadding, paddingMode, excludePadding, nullptr,
        nullptr, ceilMode);

    float kernelSz{1.0F};
    for (int32_t i = 0; i < kernelShape.nbDims; i++)
    {
        kernelSz *= kernelShape.d[i];
    }

    nvinfer1::IConstantLayer* kernelSzLayer;
    if (dt == DataType::kHALF)
    {
        kernelSzLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(kernelSz), ::ONNX_NAMESPACE::TensorProto::FLOAT16, nbDims);
    }
    else
    {
        kernelSzLayer = addConstantScalar(ctx, kernelSz, ::ONNX_NAMESPACE::TensorProto::FLOAT, nbDims);
    }

    nvinfer1::ITensor* output{nullptr};
    if (p == 1)
    {
        // x' = abs(x)
        nvinfer1::IUnaryLayer* absLayer = N_CHECK(ctx->network()->addUnary(*input, uOp::kABS));
        ctx->registerLayer(absLayer, node);
        output = N_CHECK(absLayer->getOutput(0));
    }
    else if (p == 2)
    {
        // x' = x^2
        auto* sqrLayer = N_CHECK(ctx->network()->addElementWise(*input, *input, eOp::kPROD));
        ctx->registerLayer(sqrLayer, node);
        output = N_CHECK(sqrLayer->getOutput(0));
    }

    // pool_avg(x')
    nvinfer1::IPoolingLayer* poolLayer = N_CHECK(ctx->network()->addPoolingNd(*output, pType::kAVERAGE, kernelShape));
    poolLayer->setPaddingMode(paddingMode);
    poolLayer->setPrePadding(begPadding);
    poolLayer->setPostPadding(endPadding);
    poolLayer->setStrideNd(strides);
    poolLayer->setAverageCountExcludesPadding(excludePadding);
    ctx->registerLayer(poolLayer, node);
    output = N_CHECK(poolLayer->getOutput(0));

    // pool_sum = pool_avg(x')*kernel_size
    auto* correctedSumLayer
        = N_CHECK(ctx->network()->addElementWise(*output, *kernelSzLayer->getOutput(0), eOp::kPROD));
    ctx->registerLayer(correctedSumLayer, node);
    output = correctedSumLayer->getOutput(0);

    // if p == 1, output = pool_sum
    // if p == 2, output = sqrt(pool_sum)
    if (p == 2)
    {
        nvinfer1::IUnaryLayer* sqrtLayer = N_CHECK(ctx->network()->addUnary(*output, uOp::kSQRT));
        ctx->registerLayer(sqrtLayer, node);
        output = N_CHECK(sqrtLayer->getOutput(0));
    }
    return {{output}};
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    checkNotInvalidType(inputs.at(0), {"INT32", "INT64"}, node, nodeIdx);
    nvinfer1::ITensor* inputA = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);

    bool needSqueezeHead = false;
    bool needSqueezeTail = false;
    int32_t const t1Dims = inputA->getDimensions().nbDims;
    int32_t const t2Dims = inputB->getDimensions().nbDims;
    if (t1Dims > t2Dims && t2Dims == 1)
    {
        // The second input is 1-D vector, promote to matrix by appending 1 in shape.
        std::vector<int32_t> axes{1};
        inputB = unsqueezeTensor(ctx, *inputB, axes);
        needSqueezeTail = true;
    }
    else if (t1Dims < t2Dims && t1Dims == 1)
    {
        // The first argument is 1-D, promote to matrix by prepending a 1 in shape.
        // This is done in broadcast extra dimensions.
        needSqueezeHead = true;
    }
    broadcastTensors(ctx, inputA, inputB);

    auto const getMatrixOp = [](nvinfer1::ITensor const& input) {
        return (input.getDimensions().nbDims == 1) ? nvinfer1::MatrixOperation::kVECTOR
                                                   : nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB);

    nvinfer1::IMatrixMultiplyLayer* matmul = N_CHECK(ctx->network()->addMatrixMultiply(*inputA, opA, *inputB, opB));
    ctx->registerLayer(matmul, node);

    auto outputTensor = N_CHECK(matmul->getOutput(0));
    if (needSqueezeHead)
    {
        // After MM we need remove the prepended 1.
        std::vector<int32_t> axes{0};
        outputTensor = squeezeTensor(ctx, *outputTensor, axes);
    }
    if (needSqueezeTail)
    {
        // After MM we need remove the appended 1.
        std::vector<int32_t> axes{outputTensor->getDimensions().nbDims - 1};
        outputTensor = squeezeTensor(ctx, *outputTensor, axes);
    }

    return {{outputTensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(Max)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool)
{
    return poolingHelper(ctx, node, nodeIdx, inputs, nvinfer1::PoolingType::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(Mean)
{
    std::vector<TensorOrWeights> sumResult
        = elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kSUM);
    auto sum_input = sumResult.at(0);
    nvinfer1::ITensor& sum_tensor = convertToTensor(sum_input, ctx);

    int32_t ndim = sum_tensor.getDimensions().nbDims;
    float scale_value = 1.f / inputs.size();
    auto scale_dtype = sum_input.isBFp16()
        ? ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        : (sum_input.isFp16() ? ::ONNX_NAMESPACE::TensorProto::FLOAT16 : ::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto scale_shape = nvinfer1::Dims{ndim, {1, 1, 1, 1, 1, 1, 1, 1}};
    auto scale_weights = ctx->createNamedTempWeights(scale_dtype, scale_shape);
    static_cast<float*>(scale_weights.values)[0] = scale_value;
    auto* constant_layer = N_CHECK(ctx->network()->addConstant(scale_weights.shape, scale_weights));
    ctx->network()->setWeightsName(scale_weights, scale_weights.getName());
    nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
    auto* outputLayer
        = ctx->network()->addElementWise(sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD);
    RETURN_FIRST_OUTPUT(outputLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(MeanVarianceNormalization)
{
    // Previous: stdDev = sqrt(E(x^2) - E(x)^2)
    // Current: stdDev = sqrt(E((x-E(x))^2))
    // The current formula avoids (E(x^2) - E(x)^2) < 0 caused by float point precision errors
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    using rOp = nvinfer1::ReduceOperation;

    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    auto const dims = input->getDimensions();
    DataType const dt = input->getType();

    ONNXTRT_CHECK_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF || dt == DataType::kBF16),
        "Only float32/float16/bfloat16 inputs/outputs supported in MeanVarianceNormalization. The current data type = "
            + getTrtDtypeName(dt) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DATATYPE);

    // convert axes vector to bitmask
    std::vector<int32_t> const defaultAxes = {0, 2, 3};
    auto const axes = attrs.get<std::vector<int32_t>>("axes", defaultAxes);
    uint32_t axesMask = 0;

    for (int32_t axis : axes)
    {
        convertAxis(axis, dims.nbDims, node, nodeIdx);
        axesMask |= 1 << axis;
    }

    // mean(x) along axes direction
    auto* reduceLayer = N_CHECK(ctx->network()->addReduce(*input, rOp::kAVG, axesMask, true));
    ctx->registerLayer(reduceLayer, node);
    auto* meanX = N_CHECK(reduceLayer->getOutput(0));

    // numerator: x-mean(x)
    auto* numSubLayer = N_CHECK(ctx->network()->addElementWise(*input, *meanX, eOp::kSUB));
    ctx->registerLayer(numSubLayer, node);
    auto* numerator = N_CHECK(numSubLayer->getOutput(0));

    // (x-mean(x))^2
    auto* sqrLayer = N_CHECK(ctx->network()->addElementWise(*numerator, *numerator, eOp::kPROD));
    ctx->registerLayer(sqrLayer, node);
    auto* sqrNumerator = N_CHECK(sqrLayer->getOutput(0));

    // mean((x-mean(x))^2)
    auto* meanLayer = N_CHECK(ctx->network()->addReduce(*sqrNumerator, rOp::kAVG, axesMask, true));
    ctx->registerLayer(meanLayer, node);
    auto* variance = N_CHECK(meanLayer->getOutput(0));

    // sqrt(mean((x-mean(x))^2))
    nvinfer1::IUnaryLayer* sqrtLayer = N_CHECK(ctx->network()->addUnary(*variance, uOp::kSQRT));
    ctx->registerLayer(sqrtLayer, node);
    auto* stdDev = N_CHECK(sqrtLayer->getOutput(0));

    // denominator: avoid division by zero
    auto* epsilonTensor
        = addConstantScalar(ctx, 1e-9f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims.nbDims)->getOutput(0);
    auto* addEpsLayer = N_CHECK(ctx->network()->addElementWise(*stdDev, *epsilonTensor, eOp::kSUM));
    ctx->registerLayer(addEpsLayer, node);
    stdDev = N_CHECK(addEpsLayer->getOutput(0));

    // division numerator/standard-deviation
    auto* divLayer = N_CHECK(ctx->network()->addElementWise(*numerator, *stdDev, eOp::kDIV));
    ctx->registerLayer(divLayer, node);
    RETURN_FIRST_OUTPUT(divLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Min)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(Mod)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    using eOp = nvinfer1::ElementWiseOperation;
    OnnxAttrs attrs(node, ctx);
    int32_t const fmod = attrs.get("fmod", 0);
    nvinfer1::ITensor* input0 = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* input1 = &convertToTensor(inputs.at(1), ctx);
    broadcastTensors(ctx, input0, input1);

    if (fmod == 0)
    {
        // fmod = 0, inputs can only be integers
        ONNXTRT_CHECK_NODE((input0->getType() == DataType::kINT32 || input0->getType() == DataType::kINT64),
            "The fmod attribute is set to 0. Inputs cannot be of floating point types. The current input type is "
                + getTrtDtypeName(input0->getType()) + ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DATATYPE);
        // Result = input0 - (input1 * floorDiv(input0, input1))
        nvinfer1::IElementWiseLayer* resultLayer = modWithIntegerInputs(ctx, input0, input1, false);

        ctx->registerLayer(resultLayer, node);
        RETURN_FIRST_OUTPUT(resultLayer, node, nodeIdx);
    }
    // Fmod with integer inputs
    else if (input0->getType() == DataType::kINT32 || input0->getType() == DataType::kINT64)
    {
        // Result = input0 - (input1 * Div(input0, input1))
        nvinfer1::IElementWiseLayer* resultLayer = modWithIntegerInputs(ctx, input0, input1, true);
        ctx->registerLayer(resultLayer, node);
        RETURN_FIRST_OUTPUT(resultLayer, node, nodeIdx);
    }
    // Fmod with floating point inputs
    else
    {
        // Calculate input0 / input1
        std::vector<TensorOrWeights> divResult = elementwiseHelper(ctx, node, nodeIdx, {input0, input1}, eOp::kDIV);
        auto* divResultTensor = &convertToTensor(divResult.at(0), ctx);

        // Calculate input0 - (input1 * floor(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivFloor = modWithFPInputs(ctx, input0, input1, divResultTensor, true);

        // Calculate input0 - (input1 * ceil(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivCeil = modWithFPInputs(ctx, input0, input1, divResultTensor, false);

        auto* zero = createZeroTensor(ctx, divResultTensor);
        std::vector<TensorOrWeights> greaterOrEqualResult
            = greaterLessOrEqual(ctx, node, nodeIdx, divResultTensor, zero, true);
        auto* condition = &convertToTensor(greaterOrEqualResult.at(0), ctx);
        auto* outputWithDivFloor = layerWithDivFloor->getOutput(0);
        auto* outputWithDivCeil = layerWithDivCeil->getOutput(0);

        // If (input0 / input1) >= 0, result = input0 - (input1 * floor(input0 / input1))
        // Else result = input0 - (input1 * ceil(input0 / input1))
        auto* result = N_CHECK(ctx->network()->addSelect(*condition, *outputWithDivFloor, *outputWithDivCeil));
        ctx->registerLayer(result, node);
        RETURN_FIRST_OUTPUT(result, node, nodeIdx);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Neg)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(NonMaxSuppression)
{
    // max_output, iou_threshold and score_threshold are optional
    ONNXTRT_CHECK_NODE(inputs.size() >= 2 && inputs.size() <= 5,
        "The node requires between 2-5 inputs. The actual input size is " << inputs.size() << ".", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // Input: boxes
    nvinfer1::ITensor* boxesTensorPtr = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE(boxesTensorPtr->getDimensions().nbDims == 3,
        "The boxes tensor must be 3D. The actual rank is " << boxesTensorPtr->getDimensions().nbDims << ".", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Input: scores
    nvinfer1::ITensor* scoresTensorPtr = &convertToTensor(inputs.at(1), ctx);
    ONNXTRT_CHECK_NODE(scoresTensorPtr->getDimensions().nbDims == 3,
        "The scores tensor must be 3D. The actual rank is " << scoresTensorPtr->getDimensions().nbDims << ".", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    int32_t const maxOutputBoxesPerClassDefault = 0;
    nvinfer1::ITensor* maxOutputBoxesPerClassTensorPtr = nullptr;
    nvinfer1::ITensor* iouThresholdTensorPtr = nullptr;
    nvinfer1::ITensor* scoreThresholdTensorPtr = nullptr;

    // Input: max_output_boxes_per_class (default = 0)
    if (inputs.size() >= 3 && !inputs.at(2).isNullTensor())
    {
        maxOutputBoxesPerClassTensorPtr = convertToScalar(inputs.at(2), ctx);
        // Consider when user chooses int64 max as input, which is reasonable. We need to convert it to int32 max first.
        nvinfer1::ITensor* int32Max = addConstantScalar(
            ctx, static_cast<int64_t>(std::numeric_limits<int32_t>::max()), ::ONNX_NAMESPACE::TensorProto::INT64)
                                          ->getOutput(0);
        maxOutputBoxesPerClassTensorPtr
            = ctx->network()
                  ->addElementWise(*maxOutputBoxesPerClassTensorPtr, *int32Max, nvinfer1::ElementWiseOperation::kMIN)
                  ->getOutput(0);
        maxOutputBoxesPerClassTensorPtr = castHelper(ctx, maxOutputBoxesPerClassTensorPtr, DataType::kINT32);
        ONNXTRT_CHECK_NODE(maxOutputBoxesPerClassTensorPtr != nullptr,
            "The max_output_boxes_per_class tensor must be 0D", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        auto* constantLayer = N_CHECK(ctx->network()->addConstant(
            nvinfer1::Dims{0, {}}, nvinfer1::Weights{DataType::kINT32, &maxOutputBoxesPerClassDefault, 1}));
        ONNXTRT_CHECK_NODE(constantLayer != nullptr, "Failed to add in constant for default max_output_boxes_per_class",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        maxOutputBoxesPerClassTensorPtr = N_CHECK(constantLayer->getOutput(0));
    }

    // Input: iou_threshold (default = 0)
    if (inputs.size() >= 4 && !inputs.at(3).isNullTensor())
    {
        iouThresholdTensorPtr = convertToScalar(inputs.at(3), ctx);
        ONNXTRT_CHECK_NODE(iouThresholdTensorPtr != nullptr, "The iou_threshold tensor must be 0D", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Input: score_threshold (default = 0)
    if (inputs.size() >= 5 && !inputs.at(4).isNullTensor())
    {
        scoreThresholdTensorPtr = convertToScalar(inputs.at(4), ctx);
        ONNXTRT_CHECK_NODE(scoreThresholdTensorPtr != nullptr, "The score_threshold tensor must be 0D", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Transpose scores tensor from [batch, classes, bounding_boxes] to [batch, bounding_boxes, classes]
    nvinfer1::Permutation perm{0, 2, 1};
    nvinfer1::ITensor* transposedScoresTensorPtr = transposeTensor(ctx, node, *scoresTensorPtr, perm);
    ONNXTRT_CHECK_NODE(transposedScoresTensorPtr, "Failed to transpose the scores input.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // Create the NMS layer
    auto* layer = N_CHECK(ctx->network()->addNMS(
        *boxesTensorPtr, *transposedScoresTensorPtr, *maxOutputBoxesPerClassTensorPtr, DataType::kINT64));
    ctx->registerLayer(layer, node);

    // Handle the optional threshold inputs
    if (iouThresholdTensorPtr != nullptr)
    {
        layer->setInput(3, *iouThresholdTensorPtr);
    }
    if (scoreThresholdTensorPtr != nullptr)
    {
        layer->setInput(4, *scoreThresholdTensorPtr);
    }

    // Attribute: center_point_box (default = 0)
    int32_t const centerPointBox = OnnxAttrs{node, ctx}.get("center_point_box", 0);
    nvinfer1::BoundingBoxFormat fmt;
    switch (centerPointBox)
    {
    case 0: fmt = nvinfer1::BoundingBoxFormat::kCORNER_PAIRS; break;
    case 1: fmt = nvinfer1::BoundingBoxFormat::kCENTER_SIZES; break;
    default:
        ONNXTRT_CHECK_NODE(false, "Invalid value provided for the center_point_box attribute", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE_ATTR);
    }
    layer->setBoundingBoxFormat(fmt);
    auto* indices = N_CHECK(layer->getOutput(0));

    return {{indices}};
}

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(OneHot)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    checkNotInvalidType(inputs.at(1), {"UINT8"}, node, nodeIdx);
    checkNotInvalidType(inputs.at(2), {"UINT8"}, node, nodeIdx);
    ONNXTRT_CHECK_NODE(node.input_size(),
        "OneHot must have exactly 3 inputs. Number of inputs = " << node.input_size() << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);

    nvinfer1::ITensor* values = &convertToTensor(inputs.at(2), ctx);

    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(0), ctx);
    if (!inputs.at(0).isInt32())
    {
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    nvinfer1::ITensor* depth = &convertToTensor(inputs.at(1), ctx); // tensor #1 in ONNX
    if (!inputs.at(1).isInt32())
    {
        depth = castHelper(ctx, depth, DataType::kINT32);
    }
    depth = convertToScalar(ctx, depth);
    ONNXTRT_CHECK_NODE(depth, "Failed to convert the depth to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);
    auto axis = attrs.get<int32_t>("axis", -1);
    auto nbDims = indices->getDimensions().nbDims;
    convertAxis(axis, nbDims + 1, node, nodeIdx);

    auto* layer = N_CHECK(ctx->network()->addOneHot(*indices, *values, *depth, axis));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Or)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensorPtr->getDimensions().nbDims;
    ShapeTensor const tensorDims = shapeOf(*tensorPtr);

    OnnxAttrs attrs(node, ctx);
    auto const mode = attrs.get<std::string>("mode", "constant");
    float value{0.F};
    nvinfer1::ITensor* valuePtr = nullptr;
    std::vector<int64_t> onnxPadding;

    if (ctx->getOpsetVersion() < 11)
    {
        value = attrs.get<float>("value", 0.F);
        auto padding = attrs.get<std::vector<int32_t>>("pads");
        onnxPadding = std::vector<int64_t>(padding.begin(), padding.end());
        if (onnxPadding.empty())
        {
            LOG_VERBOSE("Found no-op pad in node: " + getNodeName(node));
            RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
        }
    }
    else
    {
        // In opset >= 11, padding indicies and values moved from attributes to inputs
        if (inputs.at(1).is_weights())
        {
            weightsToVector<int64_t>(inputs.at(1).weights(), &onnxPadding);
        }
        if (inputs.size() >= 3 && !inputs.at(2).isNullTensor())
        {
            bool isValueSet = false;
            if (inputs.at(2).is_weights())
            {
                auto const padWeight = inputs.at(2).weights();
                ONNXTRT_CHECK_NODE((padWeight.count() == 1), "The input constant_value is required to be a scalar.",
                    node, nodeIdx, ErrorCode::kINVALID_NODE);
                switch (padWeight.type)
                {
                case ::ONNX_NAMESPACE::TensorProto::FLOAT:
                    value = static_cast<float const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
                    value = static_cast<half_float::half const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                case ::ONNX_NAMESPACE::TensorProto::BFLOAT16:
                    value = static_cast<BFloat16 const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                default:
                    // we use trt constant layer to do the data type convertion
                    break;
                }
            }
            if (!isValueSet)
            {
                valuePtr = &convertToTensor(inputs.at(2), ctx);
            }
        }
    }

    auto padAxes
        = inputs.size() == 4 && !inputs.at(3).isNullTensor() ? ShapeTensor(ctx, inputs.at(3)) : iotaShapeVector(nbDims);

    ShapeTensor beginPads;
    ShapeTensor endPads;
    int32_t const padAxesSize = padAxes.size();
    if (!onnxPadding.empty() && padAxes.allValuesKnown())
    {
        // The pads is from initializer or attributes.
        // Passthrough path for no-op padding.
        if (std::all_of(onnxPadding.begin(), onnxPadding.end(), [](int64_t i) { return i == 0; }))
        {
            LOG_VERBOSE("Found no-op pad in node: " + getNodeName(node));
            RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
        }

        // Sanity check.
        ONNXTRT_CHECK_NODE(static_cast<int32_t>(onnxPadding.size()) == padAxesSize * 2,
            "Length of pads input must be twice the length of axes input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

        std::vector<int64_t> beginPadsVec(onnxPadding.begin(), onnxPadding.begin() + padAxesSize);
        std::vector<int64_t> endPadsVec(onnxPadding.begin() + padAxesSize, onnxPadding.end());
        beginPads = ShapeTensor(1, std::move(beginPadsVec));
        endPads = ShapeTensor(1, std::move(endPadsVec));
    }
    else
    {
        nvinfer1::ITensor* onnxPaddingPtr = &convertToTensor(inputs.at(1), ctx);
        ONNXTRT_CHECK_NODE((onnxPaddingPtr->getDimensions().nbDims == 1),
            "The padding input must be 1D. The rank of padding input = " << onnxPaddingPtr->getDimensions().nbDims
                                                                         << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(onnxPaddingPtr->getDimensions().d[0] == padAxesSize * 2,
            "pads should be twice the length of input axes i.e. "
                << 2 * padAxesSize << ", actual length is: " << onnxPaddingPtr->getDimensions().d[0],
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

        // onnxPaddingPtr is of the format [x1_begin, x2_begin, ..., x1_end, x2_end,...].
        ShapeTensor const paddingLen = gather(ctx, shapeOf(*onnxPaddingPtr), shapeVector(0));
        ShapeTensor const halfPaddingLen = floorDiv(ctx, paddingLen, shapeVector(2));
        // Obtain begins [x1_begin, x2_begin, ...,].
        nvinfer1::ISliceLayer* beginSliceLayer
            = addSlice(ctx, *onnxPaddingPtr, shapeVector(0), halfPaddingLen, shapeVector(1));
        ctx->registerLayer(beginSliceLayer, node);
        beginPads = ShapeTensor{*(beginSliceLayer->getOutput(0))};
        // Obtain ends [x1_end, x2_end, ...].
        nvinfer1::ISliceLayer* endSliceLayer
            = addSlice(ctx, *onnxPaddingPtr, halfPaddingLen, halfPaddingLen, shapeVector(1));
        ctx->registerLayer(endSliceLayer, node);
        endPads = ShapeTensor{*(endSliceLayer->getOutput(0))};
    }

    if (padAxes.allValuesKnown())
    {
        // gather() requires indices to be normalized if their values are known
        normalizeAxes(padAxes, nbDims);
    }
    auto axesDims = gather(ctx, tensorDims, padAxes);
    ShapeTensor const zeros = similar(ctx, beginPads, 0);
    ShapeTensor start = sub(ctx, zeros, beginPads);
    ShapeTensor size = add(ctx, axesDims, add(ctx, beginPads, endPads));
    ShapeTensor const stride = similar(ctx, start, 1);

    auto* layer = N_CHECK(addSlice(ctx, *tensorPtr, start, size, stride));
    ONNXTRT_CHECK_NODE(layer, "Could not create padding layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    if (padAxes.allValuesKnown())
    {
        layer->setAxes(shapeTensorToDims(padAxes, "slice axes", -nbDims, nbDims - 1));
    }
    else
    {
        layer->setInput(5, convertToTensor(inputs.at(3), ctx));
    }
    if (mode == "constant")
    {
        layer->setMode(nvinfer1::SampleMode::kFILL);

        if (valuePtr)
        {
            layer->setInput(4, *valuePtr);
        }
        else if (value != 0.F)
        {
            // constant_value must have the same data type as the input tensor
            nvinfer1::ITensor* fillValue = nullptr;
            switch (tensorPtr->getType())
            {
            case DataType::kHALF:
                fillValue = addConstantScalar(
                    ctx, static_cast<half_float::half>(value), ::ONNX_NAMESPACE::TensorProto::FLOAT16)
                                ->getOutput(0);
                break;
            case DataType::kBF16:
                fillValue
                    = addConstantScalar(ctx, static_cast<BFloat16>(value), ::ONNX_NAMESPACE::TensorProto::BFLOAT16)
                          ->getOutput(0);
                break;
            case DataType::kFLOAT:
            case DataType::kINT8:
                fillValue = addConstantScalar(ctx, value, ::ONNX_NAMESPACE::TensorProto::FLOAT)->getOutput(0);
                break;
            default:
                fillValue = addConstantScalar(ctx, static_cast<int32_t>(value), ::ONNX_NAMESPACE::TensorProto::INT32)
                                ->getOutput(0);
                break;
            }
            ONNXTRT_CHECK_NODE(
                fillValue, "Could not create layer for constant_value", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            layer->setInput(4, *fillValue);
        }
    }
    else if (mode == "reflect")
    {
        layer->setMode(nvinfer1::SampleMode::kREFLECT);
    }
    else if (mode == "edge")
    {
        layer->setMode(nvinfer1::SampleMode::kCLAMP);
    }
    else if (mode == "wrap")
    {
        layer->setMode(nvinfer1::SampleMode::kWRAP);
    }
    else
    {
        ONNXTRT_THROW(MAKE_ERROR("Unsupported pad mode", ErrorCode::kUNSUPPORTED_NODE));
    }

    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ParametricSoftplus)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSOFTPLUS, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Pow)
{
    // TensorRT doesn't support mixed input types for POW operations. Cast operands to float and cast output to base
    // operand type.
    auto baseType = inputs.at(0).getDataType();
    auto expType = inputs.at(1).getDataType();
    bool mixedInputTypes = baseType != expType;

    if (mixedInputTypes)
    {
        LOG_VERBOSE("Found mixed input types in POW operation (base is "
            << baseType << " and exponent is " << expType
            << "). Performing operation in float32 and converting back to " << baseType);
        inputs[0] = TensorOrWeights(castHelper(ctx, &convertToTensor(inputs.at(0), ctx), DataType::kFLOAT));
        inputs[1] = TensorOrWeights(castHelper(ctx, &convertToTensor(inputs.at(1), ctx), DataType::kFLOAT));
    }

    auto powOutput = elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kPOW);
    if (mixedInputTypes && baseType != DataType::kFLOAT)
    {
        powOutput[0] = castHelper(ctx, &convertToTensor(powOutput.at(0), ctx), baseType);
    }
    return powOutput;
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu)
{
    checkNotInvalidType(inputs.at(0), {"INT32", "INT64"}, node, nodeIdx);
    checkNotInvalidType(inputs.at(1), {"INT32", "INT64"}, node, nodeIdx);
    ONNXTRT_CHECK_NODE((inputs.size() == 2),
        "The PRelu operator requires exactly 2 inputs. Current input size = " << inputs.size() << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* slopes = &convertToTensor(inputs.at(1), ctx);
    broadcastTensors(ctx, input, slopes);
    auto* layer = N_CHECK(ctx->network()->addParametricReLU(*input, *slopes));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

NodeOutputs randomHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    ShapeTensor const& inputShape, OnnxAttrs const& attrs, DataType const& inputDType, nvinfer1::FillOperation op)
{
    auto* fillLayer = addFill(ctx, inputShape, op);
    ONNXTRT_CHECK_NODE(fillLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(fillLayer, node);

    bool const isUniform = op == nvinfer1::FillOperation::kRANDOM_UNIFORM;

    // Set datatype of output:
    //      RandomUniform / RandomNormal: dtype is required and defaults to 1
    //      RandomUniformLike / RandomNormalLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int32_t>("dtype", 1);
        switch (dtype)
        {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: fillLayer->setToType(DataType::kFLOAT); break;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: fillLayer->setToType(DataType::kHALF); break;
        default: ONNXTRT_THROW(MAKE_ERROR("Unsupported data type", ErrorCode::kINVALID_VALUE));
        }
    }
    else
    {
        fillLayer->setToType(inputDType);
    }

    std::string const alphaName = isUniform ? "low" : "mean";
    std::string const betaName = isUniform ? "high" : "scale";
    auto alpha = attrs.get<float>(alphaName, 0.F);
    auto beta = attrs.get<float>(betaName, 1.F);

    fillLayer->setAlpha(alpha);
    fillLayer->setBeta(beta);

    // TensorRT does not support "seed" field now. The support will be added in future versions.
    if (attrs.count("seed"))
    {
        LOG_WARNING(
            "TensorRT currently ignores the \"seed\" field in RandomUniform or RandomNormal op. Random seeds will be "
            "used.");
    }

    RETURN_FIRST_OUTPUT(fillLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniform)
{
    OnnxAttrs attrs(node, ctx);
    auto const shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    ShapeTensor const inputShape{1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end())};

    return randomHelper(
        ctx, node, nodeIdx, inputShape, attrs, DataType::kFLOAT, nvinfer1::FillOperation::kRANDOM_UNIFORM);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniformLike)
{
    ONNXTRT_CHECK_NODE((inputs.size() == 1),
        "The RandomUniformLike operator requires exactly 1 input. Current input size = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    // Copies shape and type information from the input.
    auto inputLike = inputs.at(0);
    auto const inputShape = shapeOf(inputLike);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = inputLike.getDataType();

    return randomHelper(ctx, node, nodeIdx, inputShape, attrs, dType, nvinfer1::FillOperation::kRANDOM_UNIFORM);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomNormal)
{
    OnnxAttrs attrs(node, ctx);
    auto const shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    ShapeTensor const inputShape{1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end())};

    return randomHelper(
        ctx, node, nodeIdx, inputShape, attrs, DataType::kFLOAT, nvinfer1::FillOperation::kRANDOM_NORMAL);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomNormalLike)
{
    ONNXTRT_CHECK_NODE((inputs.size() == 1),
        "The RandomNormalLike operator requires exactly 1 input. Current input size = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    // Copies shape and type information from the input.
    auto inputLike = inputs.at(0);
    auto const inputShape = shapeOf(inputLike);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = inputLike.getDataType();

    return randomHelper(ctx, node, nodeIdx, inputShape, attrs, dType, nvinfer1::FillOperation::kRANDOM_NORMAL);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    ONNXTRT_CHECK_NODE(
        (inputs.at(0).getType() == inputs.at(1).getType() && inputs.at(0).getType() == inputs.at(2).getType()),
        "For range operator types for start, limit, and delta must be identical. Type of start = "
            + inputs.at(0).getType() + ", type of limit = " + inputs.at(1).getType()
            + ", type of delta = " + inputs.at(2).getType() + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    bool const isInt32 = inputs.at(0).isInt32();
    bool const isInt64 = inputs.at(0).isInt64();
    bool const isFp32 = inputs.at(0).isFp32();
    ONNXTRT_CHECK_NODE((isInt32 || isInt64 || isFp32),
        "This version of TensorRT only supports int32, int64, and float input types for Range!", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // "start : T
    //     Scalar. First entry for the range of output values.
    //  limit : T
    //     Scalar. Exclusive upper limit for the range of output values.
    //  delta : T
    //     Scalar. Value to step by."
    ShapeTensor start{};
    ShapeTensor limit{};
    ShapeTensor delta{};
    if (isFp32)
    {
        start = ShapeTensor{ctx, inputs.at(0)};
        limit = ShapeTensor{ctx, inputs.at(1)};
        delta = ShapeTensor{ctx, inputs.at(2)};
    }
    else
    {
        nvinfer1::ITensor* input0 = castHelper(ctx, &convertToTensor(inputs.at(0), ctx), DataType::kINT64);
        nvinfer1::ITensor* input1 = castHelper(ctx, &convertToTensor(inputs.at(1), ctx), DataType::kINT64);
        nvinfer1::ITensor* input2 = castHelper(ctx, &convertToTensor(inputs.at(2), ctx), DataType::kINT64);
        start = ShapeTensor{*input0};
        limit = ShapeTensor{*input1};
        delta = ShapeTensor{*input2};
    }

    // "number_of_elements = max( ceil( (limit - start) / delta ) , 0 )"
    //
    // To implement this in TensorRT using only operations allowed on
    // shape tensors, rewrite as:
    //      "number_of_elements = max(0 - floor((start - limit) / delta), 0)
    //
    ShapeTensor zero{};
    ShapeTensor fQuotient{};
    ShapeTensor quotient{};
    ShapeTensor numberOfElements{};

    zero = shapeScalar(0);
    fQuotient = floorDiv(ctx, sub(ctx, start, limit), delta);
    quotient = (isFp32 || isInt32) ? castToInt64(ctx, fQuotient) : fQuotient;
    numberOfElements = max(ctx, sub(ctx, zero, quotient), zero);

    nvinfer1::IFillLayer* layer = addFill(ctx, convertTo1D(ctx, numberOfElements), nvinfer1::FillOperation::kLINSPACE);
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);

    // TensorRT requires that alpha and beta both be dynamic or both be static.
    if (start.allValuesKnown() && delta.allValuesKnown() && !isInt64)
    {
        layer->setAlpha(start[0]);
        layer->setBeta(delta[0]);
    }
    else if (inputs.at(0).is_weights() && inputs.at(2).is_weights() && isInt32)
    {
        // For constant int32 start and delta, we can set to layer params directly.
        // This might not be required if TRT-20829 is done.
        ONNXTRT_CHECK_NODE(inputs.at(0).weights().count() == 1, "Start must only be a single value!", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        ONNXTRT_CHECK_NODE(inputs.at(2).weights().count() == 1, "Delta must only be a single value!", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        layer->setAlpha(inputs.at(0).weights().at<int32_t>(0));
        layer->setBeta(inputs.at(2).weights().at<int32_t>(0));
    }
    else
    {
        layer->setInput(1, start.tensor(ctx));
        auto* delta1D = &convertTo1D(ctx, delta).tensor(ctx);
        layer->setInput(2, *delta1D);
    }

    if (isInt32)
    {
        layer->setToType(DataType::kINT32);
    }
    else if (isInt64)
    {
        layer->setToType(DataType::kINT64);
    }

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Reciprocal)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

DEFINE_BUILTIN_OP_IMPORTER(ReduceL1)
{
    if (IsReduceNoOp(ctx, node, inputs))
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }
    std::vector<TensorOrWeights> absResult
        = unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kABS);

    return reduceTensor(ctx, node, nodeIdx, absResult.at(0), nvinfer1::ReduceOperation::kSUM,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSum);
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSum)
{
    if (IsReduceNoOp(ctx, node, inputs))
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    auto sum_result = importReduceSum(ctx, node, nodeIdx, inputs);
    TensorOrWeights sum_input = sum_result.at(0);
    return unaryHelper(ctx, node, nodeIdx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    if (IsReduceNoOp(ctx, node, inputs))
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    std::vector<TensorOrWeights> expResult
        = unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);

    // Include the axes input if present to ensure reduction is performed correctly
    if (inputs.size() >= 2)
    {
        expResult.push_back(inputs.at(1));
    }

    return importReduceLogSum(ctx, node, nodeIdx, expResult);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSumSquare);
DEFINE_BUILTIN_OP_IMPORTER(ReduceL2)
{
    if (IsReduceNoOp(ctx, node, inputs))
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    auto sum_sqr_result = importReduceSumSquare(ctx, node, nodeIdx, inputs);
    TensorOrWeights sum_sqr = sum_sqr_result.at(0);
    return unaryHelper(ctx, node, nodeIdx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMax)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kMAX,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMean)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kAVG,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMin)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kMIN,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceProd)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kPROD,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSum)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kSUM,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare)
{
    if (IsReduceNoOp(ctx, node, inputs))
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    auto* sqr_layer = N_CHECK(ctx->network()->addElementWise(tensor, tensor, nvinfer1::ElementWiseOperation::kPROD));
    nvinfer1::ITensor* sqr_tensorPtr = N_CHECK(sqr_layer->getOutput(0));
    return reduceTensor(ctx, node, nodeIdx, sqr_tensorPtr, nvinfer1::ReduceOperation::kSUM,
        inputs.size() >= 2 ? inputs.at(1) : TensorOrWeights());
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kRELU);
}

DEFINE_BUILTIN_OP_IMPORTER(RMSNormalization)
{
    auto* input = &convertToTensor(inputs.at(0), ctx);
    auto* scale = &convertToTensor(inputs.at(1), ctx);

    int32_t const nbDims = getNbDims(input);

    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    nvinfer1::DataType computeType = nvinfer1::DataType::kFLOAT;
    convertDtype(attrs.get<int32_t>("stash_type", 1), &computeType);

    // Cast input to computeType if necessary
    nvinfer1::ITensor* compInput = input;
    if (compInput->getType() != computeType)
    {
        compInput = castHelper(ctx, compInput, computeType);
    }

    // XSquared = Mul(X, X)
    auto* sqLayer
        = N_CHECK(ctx->network()->addElementWise(*compInput, *compInput, nvinfer1::ElementWiseOperation::kPROD));
    ctx->registerLayer(sqLayer, node);
    nvinfer1::ITensor* xSquared = N_CHECK(sqLayer->getOutput(0));

    // XSquaredMean = ReduceMean<axes=[axis..rank-1]>(XSquared), keepdims=true
    int32_t axis = attrs.get("axis", -1);
    convertAxis(axis, nbDims, node, nodeIdx);

    uint32_t axesMask{0};
    for (int32_t i = axis; i < nbDims; ++i)
    {
        axesMask |= 1U << i;
    }

    auto* meanLayer
        = N_CHECK(ctx->network()->addReduce(*xSquared, nvinfer1::ReduceOperation::kAVG, axesMask, /*keepdims=*/true));
    ctx->registerLayer(meanLayer, node);
    nvinfer1::ITensor* xSquaredMean = N_CHECK(meanLayer->getOutput(0));

    // MeanSquareEpsilon = Add(XSquaredMean, epsilon)
    nvinfer1::ITensor* epsTensor
        = N_CHECK(addConstantScalar(ctx, static_cast<float>(epsilon), ::ONNX_NAMESPACE::TensorProto::FLOAT))
              ->getOutput(0);
    epsTensor = castHelper(ctx, epsTensor, computeType);

    broadcastTensors(ctx, xSquaredMean, epsTensor);

    auto* addLayer
        = N_CHECK(ctx->network()->addElementWise(*xSquaredMean, *epsTensor, nvinfer1::ElementWiseOperation::kSUM));
    ctx->registerLayer(addLayer, node);
    nvinfer1::ITensor* meanSquareEps = N_CHECK(addLayer->getOutput(0));

    // RMS = Sqrt(MeanSquareEpsilon)
    auto* sqrtLayer = N_CHECK(ctx->network()->addUnary(*meanSquareEps, nvinfer1::UnaryOperation::kSQRT));
    ctx->registerLayer(sqrtLayer, node);
    nvinfer1::ITensor* rmsTensor = N_CHECK(sqrtLayer->getOutput(0));

    // Normalized = Div(X, RMS)
    auto* normLayer
        = N_CHECK(ctx->network()->addElementWise(*compInput, *rmsTensor, nvinfer1::ElementWiseOperation::kDIV));
    ctx->registerLayer(normLayer, node);
    nvinfer1::ITensor* normalized = N_CHECK(normLayer->getOutput(0));

    // Cast back to original type if needed
    auto const originalType = input->getType();
    if (normalized->getType() != originalType)
    {
        normalized = castHelper(ctx, normalized, originalType);
    }

    // Ensure scale matches output dtype
    if (scale->getType() != normalized->getType())
    {
        scale = castHelper(ctx, scale, normalized->getType());
    }

    broadcastTensors(ctx, normalized, scale);
    // Y = Mul(Normalized, Scale)
    auto* yLayer
        = N_CHECK(ctx->network()->addElementWise(*normalized, *scale, nvinfer1::ElementWiseOperation::kPROD));
    ctx->registerLayer(yLayer, node);
    RETURN_FIRST_OUTPUT(yLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Sign)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(Round)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kROUND);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    checkNotInvalidType(inputs.at(0), {"BOOL", "UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const inputRank = input.getDimensions().nbDims;
    ShapeTensor const inputDims = shapeOf(input);
    ONNXTRT_CHECK_NODE(
        (inputRank > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    // Add resize layer
    nvinfer1::IResizeLayer* layer = N_CHECK(ctx->network()->addResize(input));
    ctx->registerLayer(layer, node);
    OnnxAttrs attrs(node, ctx);

    auto mode = attrs.get<std::string>("mode", "nearest");
    auto interpolationMode = nvinfer1::InterpolationMode::kNEAREST;

    if (mode == "cubic")
    {
        interpolationMode = nvinfer1::InterpolationMode::kCUBIC;
    }
    else if (mode == "linear")
    {
        interpolationMode = nvinfer1::InterpolationMode::kLINEAR;
    }

    // Obtain axes, if provided. Axes must be unique and in the range [-inputRank, inputRank-1].
    auto resizeAxes = attrs.get<std::vector<int32_t>>("axes", std::vector<int32_t>());
    bool isCompleteIota = (static_cast<int32_t>(resizeAxes.size()) == inputRank);
    int32_t counter = 0;
    for (int32_t& axis : resizeAxes)
    {
        convertAxis(axis, inputRank, node, nodeIdx);
        isCompleteIota &= (axis == counter++);
    }
    bool const axesInterlacingNeeded = !resizeAxes.empty() && !isCompleteIota;
    // Note: This check is done after the conversion of axes to be in range [0, inputRank - 1] to make sure there are no
    // duplicates.
    ONNXTRT_CHECK_NODE(std::unordered_set<int32_t>(resizeAxes.begin(), resizeAxes.end()).size() == resizeAxes.size(),
        "The input axes must have unique elements.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // set transformation
    std::string transformationMode = "half_pixel";

    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kHALF_DOWN);
    if (ctx->getOpsetVersion() >= 11)
    {
        // Check for TRT-supported resize attributes
        transformationMode = attrs.get<std::string>("coordinate_transformation_mode", "half_pixel");
        auto nearest_mode = attrs.get<std::string>("nearest_mode", "round_prefer_floor");

        // clang-format off
        // The existence of a fourth input means a shape was passed as the resize parameter
        // For ONNX resize with the "sizes", TensorRT's resize maps to ONNX's in the following ways:
        // Nearest&Linear&Cubic:
        //     align_corners        -> ResizeCoordinateTransformation::kALIGN_CORNERS ResizeSelector::kFORMULA ResizeRoundMode::kFLOOR
        //     half_pixel           -> ResizeCoordinateTransformation::kHALF_PIXEL    ResizeSelector::kFORMULA ResizeRoundMode::kFLOOR
        //     asymmetric           -> ResizeCoordinateTransformation::kASYMMETRIC    ResizeSelector::kFORMULA ResizeRoundMode::kFLOOR
        //     pytorch_half_pixel   -> ResizeCoordinateTransformation::kHALF_PIXEL    ResizeSelector::kUPPER   ResizeRoundMode::kFLOOR
        //     tf_half_pixel_for_nn -> ResizeCoordinateTransformation::kHALF_PIXEL    ResizeSelector::kFORMULA ResizeRoundMode::kFLOOR
        // clang-format on

        if (transformationMode == "align_corners")
        {
            layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS);
        }
        else if (transformationMode == "tf_half_pixel_for_nn")
        {
            // `tf_half_pixel_for_nn` has been deprecated after Resize-11. Leaving it as is for backward compatibility.
            layer->setNearestRounding(nvinfer1::ResizeRoundMode::kCEIL);
            layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
        }
        else if (transformationMode == "pytorch_half_pixel")
        {
            layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kUPPER);
            layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
        }
        else if (transformationMode == "half_pixel")
        {
            layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
        }
        else if (transformationMode == "asymmetric")
        {
            layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
        }
        else
        {
            // NOTE: Currently `half_pixel_symmetric` and `tf_crop_and_resize` are not supported. `extrapolation_value`
            // attribute and `roi` input are relevant only for `tf_crop_and_resize` and hence, aren't supported.
            ONNXTRT_CHECK_NODE(false, "Unsupported coordinate transformation mode " << transformationMode, node,
                nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        }

        if (transformationMode != "tf_half_pixel_for_nn")
        {
            if (nearest_mode == "floor")
            {
                layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
            }
            else if (nearest_mode == "ceil")
            {
                layer->setNearestRounding(nvinfer1::ResizeRoundMode::kCEIL);
            }
            else if (nearest_mode == "round_prefer_floor")
            {
                layer->setNearestRounding(nvinfer1::ResizeRoundMode::kHALF_DOWN);
            }
            else if (nearest_mode == "round_prefer_ceil")
            {
                layer->setNearestRounding(nvinfer1::ResizeRoundMode::kHALF_UP);
            }

            // set exclude_outside, only support after opset 11.
            auto excludeOutside = static_cast<bool>(attrs.get<int32_t>("exclude_outside", 0));
            layer->setExcludeOutside(excludeOutside);

            // set bicubic, only support after opset 11.
            if (interpolationMode == nvinfer1::InterpolationMode::kCUBIC)
            {
                auto cubicCoeff = attrs.get<float>("cubic_coeff_a", -0.75F);
                layer->setCubicCoeff(cubicCoeff);
            }

            if (inputs.size() == 4 && !inputs.at(3).isNullTensor())
            {
                if (inputs.at(3).is_weights())
                {
                    ONNXTRT_CHECK_NODE((inputs.at(3).weights().shape.nbDims == 1),
                        "The sizes input must be 1D. Sizes rank = " << inputs.at(3).weights().shape.nbDims << ".", node,
                        nodeIdx, ErrorCode::kINVALID_NODE);
                    std::vector<int64_t> sizesVec;
                    weightsToVector<int64_t>(inputs.at(3).weights(), &sizesVec);
                    if (axesInterlacingNeeded)
                    {
                        ONNXTRT_CHECK_NODE(sizesVec.size() == resizeAxes.size(),
                            "Length of sizes input must be same as length of axes attribute.", node, nodeIdx,
                            ErrorCode::kINVALID_NODE);
                        std::vector<int64_t> tempVec(inputDims.begin(), inputDims.end());
                        for (size_t idx = 0; idx < resizeAxes.size(); idx++)
                        {
                            int32_t const currAxis = resizeAxes[idx];
                            tempVec[currAxis] = sizesVec[idx];
                        }
                        sizesVec = std::move(tempVec);
                    }
                    ONNXTRT_CHECK_NODE((static_cast<int32_t>(sizesVec.size()) == inputRank),
                        "The shape of weights must align with input data. Length of sizes = "
                            << sizesVec.size() << ", rank of input = " << inputRank << ".",
                        node, nodeIdx, ErrorCode::kINVALID_NODE);
                    nvinfer1::Dims resizeShape{inputRank, {}};
                    for (int32_t i = 0; i < inputRank; i++)
                    {
                        resizeShape.d[i] = static_cast<int32_t>(sizesVec[i]);
                    }
                    layer->setOutputDimensions(resizeShape);
                }
                else
                {
                    auto* resizeShape = &convertToTensor(inputs.at(3), ctx);
                    if (axesInterlacingNeeded)
                    {
                        ONNXTRT_CHECK_NODE(resizeShape->getDimensions().d[0] == static_cast<int64_t>(resizeAxes.size()),
                            "sizes input tensor should be of the same length as axes attribute i.e. "
                                << resizeAxes.size() << ", actual length is: " << resizeShape->getDimensions().d[0],
                            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
                        std::vector<int64_t> axesLongInt(resizeAxes.begin(), resizeAxes.end());
                        ShapeTensor const subscripts{
                            axesToInterlaceSubscripts(ShapeTensor(1, std::move(axesLongInt)), inputRank)};
                        ShapeTensor const orderedShapeTensor
                            = interlace(ctx, inputDims, ShapeTensor(*resizeShape), subscripts);
                        resizeShape = &orderedShapeTensor.tensor(ctx);
                    }
                    layer->setInput(1, *resizeShape);
                }
                layer->setResizeMode(interpolationMode);
                RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
            }
        }
    }
    // For opset 10 resize, the only supported mode is asymmetric resize with scales. Nearest resizes use floor
    // rounding.
    else
    {
        transformationMode = "asymmetric";
        layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
        if (mode == "nearest")
        {
            layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
        }
    }

    // Resizes that use scale factors have the same import logic between opsets
    auto scales = ctx->getOpsetVersion() >= 11 ? inputs.at(2) : inputs.at(1);

    if (scales.is_weights())
    {
        // TRT-15340: Remove this and use else path when safety support nbDims == 1.
        ONNXTRT_CHECK_NODE((scales.weights().shape.nbDims == 1),
            "The scales input must be 1D. Scales rank = " << scales.weights().shape.nbDims << ".", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        int32_t const scaleSize = scales.weights().shape.d[0];

        std::vector<float> scalesVec;
        weightsToVector<float>(scales.weights(), &scalesVec);
        if (axesInterlacingNeeded)
        {
            ONNXTRT_CHECK_NODE(scalesVec.size() == resizeAxes.size(),
                "Length of scales input must be same as length of axes attribute.", node, nodeIdx,
                ErrorCode::kINVALID_NODE);
            std::vector<float> tempVec(inputRank, 1.0);
            for (size_t idx = 0; idx < resizeAxes.size(); idx++)
            {
                int32_t const currAxis = resizeAxes[idx];
                tempVec[currAxis] = scalesVec[idx];
            }
            // Update scalesVec to hold the ordered information.
            scalesVec = std::move(tempVec);
        }
        ONNXTRT_CHECK_NODE((static_cast<int32_t>(scalesVec.size()) == inputRank),
            "The shape of weights must align with input data. Length of scales = "
                << scalesVec.size() << ", rank of input = " << inputRank << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);

        // check resize dims
        if (interpolationMode == nvinfer1::InterpolationMode::kLINEAR)
        {
            ONNXTRT_CHECK_NODE(canUseNDResize(scaleSize, scalesVec.data(), 3),
                "This version of TensorRT only supports linear resizing on the outermost 3 dimensions.", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        else if (interpolationMode == nvinfer1::InterpolationMode::kCUBIC)
        {
            ONNXTRT_CHECK_NODE(canUseNDResize(scaleSize, scalesVec.data(), 2),
                "This version of TensorRT only supports cubic resizing on the outermost 2 dimensions.", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scalesVec.data(), inputRank);
    }
    else
    {
        // Currently, interlacing of `scales` tensor with `axes` is not supported because interlacing needs a
        // `ShapeTensor` of scales (float values) and a `ShapeTensor` holding float values isn't supported yet.
        ONNXTRT_CHECK_NODE(!axesInterlacingNeeded,
            "Currently, `axes` attribute is supported with `scales` tensor only when it's trivial i.e. it's an "
            "iota vector of same length as input rank.",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, input, scales);
        layer->setInput(1, *resizeShape);
    }

    layer->setResizeMode(interpolationMode);

    LOG_VERBOSE("Running resize layer with:");
    LOG_VERBOSE("    Transformation mode: " << transformationMode);
    LOG_VERBOSE("    Resize mode: " << mode);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(RotaryEmbedding)
{
    OnnxAttrs attrs(node, ctx);
    bool const interleaved = attrs.get<bool>("interleaved", false);
    int32_t const rotaryEmbeddingDim = attrs.get<int32_t>("rotary_embedding_dim", 0);

    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    bool const inputIs3D = input->getDimensions().nbDims == 3;

    if (inputIs3D) // (batchSize, sequenceLength, hiddenSize=numHeads * headSize)
    {
        ONNXTRT_CHECK_NODE(attrs.count("num_heads"), "num_heads attribute is required for 3D input tensors.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        int32_t const numHeads = attrs.get<int32_t>("num_heads");
        int32_t const hiddenSize = input->getDimensions().d[2];
        ONNXTRT_CHECK_NODE(hiddenSize % numHeads == 0, "hiddenSize must be divisible by num_heads.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        auto shuffleLayer = N_CHECK(ctx->network()->addShuffle(*input));
        shuffleLayer->setReshapeDimensions(nvinfer1::Dims{4, {0, 0, numHeads, -1}}); // (batchSize, sequenceLength, numHeads, headSize)
        shuffleLayer->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3}); // (batchSize, numHeads, sequenceLength, headSize)
        input = N_CHECK(shuffleLayer->getOutput(0));
    }

    nvinfer1::ITensor* cosCache = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* sinCache = &convertToTensor(inputs.at(2), ctx);

    auto layer = N_CHECK(ctx->network()->addRotaryEmbedding(*input, *cosCache, *sinCache, interleaved, rotaryEmbeddingDim));
    ctx->registerLayer(layer, node);

    if (inputs.size() > 3)
    {
        nvinfer1::ITensor* positionIds = &convertToTensor(inputs.at(3), ctx);
        layer->setInput(3, *positionIds);
    }

    auto* output = N_CHECK(layer->getOutput(0));

    if (inputIs3D) // Keep the output in the same shape as the input
    {
        auto shuffleLayer = N_CHECK(ctx->network()->addShuffle(*output));
        shuffleLayer->setFirstTranspose(nvinfer1::Permutation{0, 2, 1, 3}); // (batchSize, sequenceLength, numHeads, headSize)
        shuffleLayer->setReshapeDimensions(nvinfer1::Dims{3, {0, 0, -1}}); // (batchSize, sequenceLength, hiddenSize=numHeads * headSize)
        output = N_CHECK(shuffleLayer->getOutput(0));
    }

    return {{output}};
}


DEFINE_BUILTIN_OP_IMPORTER(Reshape)
{
    // "data : T
    // An input tensor"
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);

    // The attribute allowzero was introduced in opset 14, but as an extension
    // recognize it for opset >= 5.
    int32_t allowZero = 0;
    if (ctx->getOpsetVersion() >= 5)
    {
        OnnxAttrs attrs{node, ctx};
        if (attrs.count("allowzero"))
        {
            allowZero = attrs.get<int32_t>("allowzero");
            if (ctx->getOpsetVersion() < 14)
            {
                LOG_WARNING(
                    getNodeName(node) << ": Using attribute allowzero with opset < 14 is a TensorRT extension.");
            }
        }
    }

    ShapeTensor shape;
    if (ctx->getOpsetVersion() >= 5)
    {
        // "shape : tensor(int64)
        // Specified shape for output."
        shape = ShapeTensor{ctx, inputs.at(1)};
    }
    else
    {
        // "Reshape-1
        // ...
        // shape : list of ints
        // New shape"
        OnnxAttrs attrs{node, ctx};
        auto const shapeAsIntList = attrs.get<std::vector<int32_t>>("shape");
        shape = ShapeTensor(1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end()));
    }

    // "A dimension could also be 0, in which case the actual dimension
    // value is unchanged (i.e. taken from the input tensor)."
    nvinfer1::IShuffleLayer* layer = addShuffle(ctx, data, shape, /*zeroIsPlaceholder=*/!allowZero);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ReverseSequence)
{
    ONNXTRT_CHECK_NODE((inputs.size() == 2),
        "ReverseSequence expects two input tensors: input and sequence_lens. Current input size = " << inputs.size()
                                                                                                    << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* sequenceLens = &convertToTensor(inputs.at(1), ctx);
    auto const inputDims = input->getDimensions();
    auto const sequenceLensDims = sequenceLens->getDimensions();
    ONNXTRT_CHECK_NODE((inputDims.nbDims >= 2),
        "Rank of input must be at least two. Current rank of inputs = " << inputDims.nbDims << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((sequenceLensDims.nbDims == 1),
        "Rank of sequence_lens must be one. Current rank of sequence lens = " << sequenceLensDims.nbDims << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs{node, ctx};
    int32_t const batchAxis = attrs.get<int32_t>("batch_axis", 1);
    int32_t const sequenceAxis = attrs.get<int32_t>("time_axis", 0);
    ONNXTRT_CHECK_NODE((batchAxis >= 0 && batchAxis <= inputDims.nbDims), "Invalid batch_axis", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE_ATTR);
    ONNXTRT_CHECK_NODE((sequenceAxis >= 0 && sequenceAxis <= inputDims.nbDims), "Invalid time_axis", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE_ATTR);

    auto layer = N_CHECK(ctx->network()->addReverseSequence(*input, *sequenceLens));
    ctx->registerLayer(layer, node);
    ONNXTRT_CHECK_NODE(layer, "Failed to add ReverseSequence layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    layer->setBatchAxis(batchAxis);
    layer->setSequenceAxis(sequenceAxis);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(RNN)
{
    OnnxAttrs attrs{node, ctx};

    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int32_t numDirections = (direction == "bidirectional") ? 2 : 1;
    const int32_t hiddenSize = attrs.get<int32_t>("hidden_size");

    float const clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);

    std::vector<nvinfer1::ActivationType> defaultActs{nvinfer1::ActivationType::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {nvinfer1::ActivationType::kTANH});
    }
    std::vector<nvinfer1::ActivationType> activations
        = attrs.get<std::vector<nvinfer1::ActivationType>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    std::transform(activations.begin() + activationAlphas.size(), activations.end(),
        std::back_inserter(activationAlphas), &getActivationDefaultAlpha);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    std::transform(activations.begin() + activationBetas.size(), activations.end(), std::back_inserter(activationBetas),
        &getActivationDefaultBeta);

    // Roll Rb into Wb (and RBb into WBb). Bias is in the form  [Wb[iofc], Rb[iofc], WBb[iofc], RBb[iofc]].
    // So reshape such that we can perform a reduction to add Wb and Rb.
    nvinfer1::ITensor* combinedBias{nullptr};
    if (inputs.size() > 3 && inputs.at(3))
    {
        nvinfer1::ITensor* bias = &convertToTensor(inputs.at(3), ctx);
        LOG_VERBOSE("Bias shape is: " << bias->getDimensions());
        // Reshape to [[Wb[iofc], Rb[iofc]], [WBb[iofc], RBb[iofc]]]
        nvinfer1::IShuffleLayer* reshapeBias = N_CHECK(ctx->network()->addShuffle(*bias));
        reshapeBias->setReshapeDimensions(nvinfer1::Dims3{numDirections, 2, hiddenSize});
        reshapeBias->setZeroIsPlaceholder(false);
        auto reshapeBiasOutput = N_CHECK(reshapeBias->getOutput(0));
        LOG_VERBOSE("Reshaping bias to: " << reshapeBiasOutput->getDimensions());
        auto reduceLayer
            = N_CHECK(ctx->network()->addReduce(*reshapeBiasOutput, nvinfer1::ReduceOperation::kSUM, /*axis=*/0b010,
                /*keepDimensions=*/true));
        combinedBias = N_CHECK(reduceLayer->getOutput(0));
        LOG_VERBOSE("After reduction, bias shape is: " << combinedBias->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor = N_CHECK(
            addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0));
        LOG_VERBOSE("numDirectionsTensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor = N_CHECK(
            addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0));
        LOG_VERBOSE("hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, nvinfer1::Dims{1, {1}});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        std::array<nvinfer1::ITensor*, 3> tensors{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}};
        nvinfer1::IConcatenationLayer* concatenatedShape = N_CHECK(ctx->network()->addConcatenation(tensors.data(), 3));
        return N_CHECK(concatenatedShape->getOutput(0));
    };

    auto const getInitialInputValue = [&ctx, &initialStateShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx,
            N_CHECK(addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0)),
            initialStateShape());
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = N_CHECK(ctx->network()->addLoop());
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);
    ONNXTRT_CHECK_NODE(iterationInput, "Failed to add RNN input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(hiddenState, node);
    LOG_VERBOSE("Hidden state shape: " << hiddenState->getOutput(0)->getDimensions());

    // Compute intermediate(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)).
    auto xtWTLayer = N_CHECK(ctx->network()->addMatrixMultiply(
        *iterationInput, nvinfer1::MatrixOperation::kNONE, *weights, nvinfer1::MatrixOperation::kTRANSPOSE));
    nvinfer1::ITensor* xtWT = N_CHECK(xtWTLayer->getOutput(0));
    LOG_VERBOSE("X(t) * W^T -> " << xtWT->getDimensions());

    auto ht1RTLayer = N_CHECK(ctx->network()->addMatrixMultiply(*hiddenState->getOutput(0),
        nvinfer1::MatrixOperation::kNONE, *recurrenceWeights, nvinfer1::MatrixOperation::kTRANSPOSE));
    nvinfer1::ITensor* ht1RT = N_CHECK(ht1RTLayer->getOutput(0));
    LOG_VERBOSE("H(t-1) * R^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* intermediatet = getElementWiseResult(ctx, *xtWT, *ht1RT, nvinfer1::ElementWiseOperation::kSUM);
    if (combinedBias)
    {
        intermediatet = getElementWiseResult(ctx, *intermediatet, *combinedBias, nvinfer1::ElementWiseOperation::kSUM);
    }

    // H(t) = f(intermediate(t))
    nvinfer1::IActivationLayer* hAct
        = N_CHECK(ctx->network()->addActivation(*addClip(ctx, intermediatet, clip), activations.at(0)));
    hAct->setAlpha(activationAlphas.at(0));
    hAct->setBeta(activationBetas.at(0));
    nvinfer1::ITensor* Ht = N_CHECK(hAct->getOutput(0));

    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape = getElementWiseResult(ctx, *initialStateShape(),
        *N_CHECK(addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
            nvinfer1::Dims{1,
                {3}})->getOutput(0)),
        nvinfer1::ElementWiseOperation::kDIV);

    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, hiddenState->getOutput(0), Ht, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, hiddenState->getOutput(0), Ht, maxLen, direction == "reverse");
    }

    hiddenState->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};

    // Add outputs. In ONNX, all RNN outputs are optional, so check for them here.
    auto shouldAddOutput = [&node](int32_t index) { return index < node.output().size() && !node.output(index).empty(); };

    // Y = concatenation of all H(t) for each element of the sequence
    if (shouldAddOutput(0))
    {
        outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
            numDirections, inputs, direction == "reverse"));
    }
    // Yh = last value of H(t)
    if (shouldAddOutput(1))
    {
        outputs.emplace_back(
            loop->addLoopOutput(*hiddenState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    }

    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(RoiAlign)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* roisPtr = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* batchIndicesPtr = &convertToTensor(inputs.at(2), ctx);
    batchIndicesPtr = castHelper(ctx, batchIndicesPtr, DataType::kINT32);

    // Sanity checking
    auto roiDims = roisPtr->getDimensions();
    ONNXTRT_CHECK_NODE(roiDims.nbDims == 2 && roiDims.d[1] == 4,
        "Found incorrect dimensions for ROIs input! Rank of ROI input = " << roiDims.nbDims
                                                                          << ", roiDims.d[1] = " << roiDims.d[1] << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);

    int32_t coordinateTransformationMode{};

    if (ctx->getOpsetVersion() >= 16)
    {
        coordinateTransformationMode
            = attrs.get("coordinate_transformation_mode", std::string("half_pixel")) == "half_pixel" ? 1 : 0;
    }
    else
    {
        // RoiAlign-10 does not support coordinate_transformation_mode
        // Fall-back to 'output_half_pixel' in RoiAlign-16
        coordinateTransformationMode = 0;
    }

    int32_t mode = attrs.get("mode", std::string("avg")) == "avg" ? 1 : 0;
    int32_t outputHeight = attrs.get("output_height", 1);
    int32_t outputWidth = attrs.get("output_width", 1);
    int32_t samplingRatio = attrs.get("sampling_ratio", 1);
    float spatialScale = attrs.get("spatial_scale", 1.0F);

    // Populate RoiAlign plugin properties.
    std::string const pluginName = "ROIAlign_TRT";
    std::string const pluginVersion = "2";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back(
        "coordinate_transformation_mode", &coordinateTransformationMode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("mode", &mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("output_height", &outputHeight, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("output_width", &outputWidth, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("sampling_ratio", &samplingRatio, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("spatial_scale", &spatialScale, nvinfer1::PluginFieldType::kFLOAT32, 1);

    // Create plugin from registry
    auto const plugin = createPlugin(ctx, node, getNodeName(node), kTRT_STD_PLUGIN_NAMESPACE,
        static_cast<nvinfer1::IPluginCreatorV3One*>(importPluginCreator(ctx, pluginName, pluginVersion)), f);

    ONNXTRT_CHECK_NODE(plugin != nullptr, "ROIAlign plugin was not found in the plugin registry!", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* const inputTensorsPtr[3] = {tensorPtr, roisPtr, batchIndicesPtr};
    auto* layer = N_CHECK(ctx->network()->addPluginV3(inputTensorsPtr, 3, nullptr, 0, *plugin));
    ctx->registerLayer(layer, node);

    // ROIAlign requires nvinfer_vc_plugin when using VC.
#if defined(_WIN32)
    ctx->addUsedVCPluginLibrary(
        node, pluginName.c_str(), ("nvinfer_vc_plugin_" + std::to_string(NV_TENSORRT_MAJOR)).c_str());
#else
    ctx->addUsedVCPluginLibrary(node, pluginName.c_str(), "nvinfer_vc_plugin");
#endif

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ScaledTanh)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSCALED_TANH, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Scan)
{
    OnnxAttrs attrs(node, ctx);

    // In opset 8, the scan node is defined differently than in later opsets.
    //     1. It has an optional input `sequence_lens`
    //     2. The scan input/output axis are always set to 1
    const int32_t opset8Offset = ctx->getOpsetVersion() == 8 ? 1 : 0;
    if (opset8Offset == 1)
    {
        ONNXTRT_CHECK_NODE(inputs.at(0).isNullTensor(), "TensorRT doesn't support sequence_lens input for this node!",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }
    int32_t const nbInputs = node.input().size() - opset8Offset;
    int32_t const nbScanInputs = attrs.get<int>("num_scan_inputs");
    // The number of state variables on the input and output is the same.
    int32_t const nbStateVars = nbInputs - nbScanInputs;
    int32_t const nbScanOutputs = node.output().size() - nbStateVars;

    // Populate scan input axis
    std::vector<int32_t> defaultScanInputArgs(nbScanInputs);
    std::fill(defaultScanInputArgs.begin(), defaultScanInputArgs.end(), opset8Offset);
    std::vector<int32_t> scanInputAxes(attrs.get("scan_input_axes", defaultScanInputArgs));

    // Populate scan input directions
    std::vector<int32_t> defaultInputScanDirection(nbScanInputs);
    std::fill(defaultInputScanDirection.begin(), defaultInputScanDirection.end(), 0);
    std::vector<int32_t> const scanInputDirections(attrs.get("scan_input_directions", defaultInputScanDirection));

    // Populate scan output axis
    std::vector<int32_t> defaultScanOutputArgs(nbScanOutputs);
    std::fill(defaultScanOutputArgs.begin(), defaultScanOutputArgs.end(), opset8Offset);
    std::vector<int32_t> scanOutputAxes(attrs.get("scan_output_axes", defaultScanOutputArgs));

    // Populate scan ouput directions
    std::vector<int32_t> defaultOutputScanDirection(nbScanOutputs);
    std::fill(defaultOutputScanDirection.begin(), defaultOutputScanDirection.end(), 0);
    std::vector<int32_t> const scanOutputDirections(attrs.get("scan_output_directions", defaultOutputScanDirection));

    ::ONNX_NAMESPACE::GraphProto const& body = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("body");

    // Support possible negative axis for input and output axes:
    for (auto& axis : scanInputAxes)
    {
        convertAxis(axis, nvinfer1::Dims::MAX_DIMS, node, nodeIdx);
    }

    for (auto& axis : scanOutputAxes)
    {
        convertAxis(axis, nvinfer1::Dims::MAX_DIMS, node, nodeIdx);
    }

    auto loop = N_CHECK(ctx->network()->addLoop());
    // When multiple scan inputs are present, scan behaves like zip, so it is sufficient
    // to use only one scan input to determine trip limit.
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, &convertToTensor(inputs.back(), ctx), scanInputAxes.back());
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Establish scope for names local to the subgraph.
    NameScope nameScope(*ctx);

    // Add initial state inputs using recurrent layers, and scan inputs using iterators.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (int32_t i = 0; i < nbStateVars; ++i)
    {
        stateVars.emplace_back(N_CHECK(loop->addRecurrence(convertToTensor(inputs.at(i + opset8Offset), ctx))));
        ctx->registerTensor(TensorOrWeights{N_CHECK(stateVars.back()->getOutput(0))}, body.input(i).name());
    }
    ctx->registerLayer(stateVars.at(0), node);

    for (int32_t i = 0; i < nbScanInputs; ++i)
    {
        const int32_t index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput
            = N_CHECK(loop->addIterator(convertToTensor(inputs.at(index + opset8Offset), ctx)));
        scanInput->setAxis(scanInputAxes.at(i));
        scanInput->setReverse(scanInputDirections.at(i) == 1);
        ctx->registerTensor(TensorOrWeights{N_CHECK(scanInput->getOutput(0))}, body.input(index).name());
    }

    // Loop Body. This is handled by dispatching to other op converters.
    std::vector<Status> errors{};
    onnx2trt::parseGraph(
        ctx, body, errors, /*deserializingINetwork=*/false, /*currentNode=*/nullptr, /*subgraphParentIdx=*/nodeIdx);

    // Set up recurrence outputs (first N body graph outputs).
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int32_t i = 0; i < nbStateVars; ++i)
    {
        auto const& bodyOutputName = body.output(i).name();
        auto& stateOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For state variable output: " << bodyOutputName
                                                  << ", found matching tensor: " << stateOutput.getName()
                                                  << ", with shape: " << stateOutput.getDimensions());
        stateVars.at(i)->setInput(1, stateOutput);
        // Each state variable is also a loop output
        auto output = N_CHECK(stateVars.at(i)->getOutput(0));
        auto outputLayer = N_CHECK(loop->addLoopOutput(*output, nvinfer1::LoopOutput::kLAST_VALUE));
        auto outputTensor = N_CHECK(outputLayer->getOutput(0));
        nodeOutputs.emplace_back(outputTensor);
    }
    // Finally, set up scan outputs.
    for (int32_t i = 0; i < nbScanOutputs; ++i)
    {
        int32_t const index = nbStateVars + i;
        auto const& bodyOutputName = body.output(index).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        // For scanOutputDirections, 0 indicates appending, and 1, prepending.
        auto const scanDirection
            = (scanOutputDirections.at(i) == 0) ? nvinfer1::LoopOutput::kCONCATENATE : nvinfer1::LoopOutput::kREVERSE;
        auto const scanAxis = scanOutputAxes.at(i);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
                                        << ", with shape: " << scanOutput.getDimensions() << ". Using scan direction: "
                                        << static_cast<int32_t>(scanDirection) << ", and scan axis: " << scanAxis);
        nvinfer1::ILoopOutputLayer* trtScanOut = N_CHECK(loop->addLoopOutput(scanOutput, scanDirection, scanAxis));
        trtScanOut->setInput(1, *tripLimit);
        nodeOutputs.emplace_back(N_CHECK(trtScanOut->getOutput(0)));
    }

    return {nodeOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(GridSample)
{
    checkNotInvalidType(inputs.at(0), {"BOOL", "UINT8"}, node, nodeIdx);
    checkNotInvalidType(inputs.at(1), {"BOOL", "UINT8"}, node, nodeIdx);
    ONNXTRT_CHECK_NODE(
        (inputs.size() == 2), "TRT expects two input tensors: grid and input", node, nodeIdx, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const inputRank = input.getDimensions().nbDims;
    ONNXTRT_CHECK_NODE(
        (inputRank > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor& grid = convertToTensor(inputs.at(1), ctx);
    int32_t const gridRank = grid.getDimensions().nbDims;
    ONNXTRT_CHECK_NODE(
        (gridRank > 0), "The grid tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE((gridRank == inputRank),
        "The input tensor and the grid tensor must have the same rank. Rank of grid tensor = "
            << gridRank << ", rank of input = " << inputRank << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Add grid sample layer
    nvinfer1::IGridSampleLayer* layer = N_CHECK(ctx->network()->addGridSample(input, grid));
    ctx->registerLayer(layer, node);
    OnnxAttrs attrs(node, ctx);

    auto paddingMode = attrs.get<std::string>("padding_mode", "zeros");
    nvinfer1::SampleMode sampleMode{nvinfer1::SampleMode::kFILL};
    if (paddingMode == "zeros")
    {
        sampleMode = nvinfer1::SampleMode::kFILL;
    }
    else if (paddingMode == "border")
    {
        sampleMode = nvinfer1::SampleMode::kCLAMP;
    }
    else if (paddingMode == "reflection")
    {
        sampleMode = nvinfer1::SampleMode::kREFLECT;
    }

    auto mode = attrs.get<std::string>("mode", "bilinear");
    nvinfer1::InterpolationMode interpolationMode{nvinfer1::InterpolationMode::kNEAREST};
    if (mode == "nearest")
    {
        interpolationMode = nvinfer1::InterpolationMode::kNEAREST;
    }
    else if (mode == "bilinear")
    {
        interpolationMode = nvinfer1::InterpolationMode::kLINEAR;
    }
    else if (mode == "bicubic")
    {
        interpolationMode = nvinfer1::InterpolationMode::kCUBIC;
    }

    bool const alignCorners{attrs.get<int32_t>("align_corners", 0) == 1};

    ONNXTRT_CHECK_NODE(
        layer->setSampleMode(sampleMode), "Failed to set sample mode!", node, nodeIdx, ErrorCode::kINVALID_VALUE);
    layer->setAlignCorners(alignCorners);
    layer->setInterpolationMode(interpolationMode);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterND)
{
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("reduction", "none");
    ONNXTRT_CHECK_NODE(
        mode == "none", "ScatterND with reduction is not supported.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_ATTR);
    return addScatterLayer(ctx, node, nodeIdx, inputs, nvinfer1::ScatterMode::kND);
}

NodeOutputs scatterPluginHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    std::vector<TensorOrWeights>& inputs, int32_t axis, std::string const& reduction)
{
    // Populate scatter plugin properties.
    std::string const pluginName = "ScatterElements";
    std::string const pluginVersion = "2";
    std::vector<nvinfer1::PluginField> f;

    // populate fields axis, reduction type

    f.emplace_back("axis", &axis, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("reduction", reduction.c_str(), nvinfer1::PluginFieldType::kCHAR, reduction.size());

    // Create plugin from registry
    auto const plugin = createPlugin(ctx, node, getNodeName(node), kTRT_STD_PLUGIN_NAMESPACE,
        static_cast<nvinfer1::IPluginCreatorV3One*>(importPluginCreator(ctx, pluginName, pluginVersion)), f);

    ONNXTRT_CHECK_NODE(plugin != nullptr, "ScatterReduction plugin was not found in the plugin registry!", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Create vector of inputs
    std::vector<nvinfer1::ITensor*> pluginInputs{};
    for (auto& input : inputs)
    {
        pluginInputs.emplace_back(&convertToTensor(input, ctx));
    }

    auto* layer = N_CHECK(ctx->network()->addPluginV3(pluginInputs.data(), pluginInputs.size(), nullptr, 0, *plugin));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterElements)
{
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    convertAxis(axis, nbDims, node, nodeIdx);

    auto reduction = attrs.get<std::string>("reduction", "none");
    if (reduction != "none")
    {
        return scatterPluginHelper(ctx, node, nodeIdx, inputs, axis, reduction);
    }

    return addScatterLayer(ctx, node, nodeIdx, inputs, nvinfer1::ScatterMode::kELEMENT, axis);
}

DEFINE_BUILTIN_OP_IMPORTER(Scatter)
{
    // Scatter was deprecated in Opset 11 and replaced by ScatterElements
    if (ctx->getOpsetVersion() >= 11)
    {
        LOG_WARNING("Scatter was deprecated in Opset 11. Node: \"" << getNodeName(node)
                                                                   << "\" will be converted to ScatterElements.");
    }

    return importScatterElements(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Selu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.6732f);
    float beta = attrs.get("gamma", 1.0507f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSELU, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Shape)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    auto* layer = N_CHECK(ctx->network()->addShape(input));
    ctx->registerLayer(layer, node);
    auto* layerOutput = N_CHECK(layer->getOutput(0));

    nvinfer1::ICastLayer* cast = N_CHECK(ctx->network()->addCast(*layerOutput, DataType::kINT64));

    auto const rank = input.getDimensions().nbDims;
    auto start = attrs.get<int32_t>("start", 0);
    auto end = attrs.get<int32_t>("end", rank);
    auto const getInRank = [](int32_t x, int32_t rank) {
        if (x < 0)
        {
            x += rank;
        }
        x = std::min(std::max(0, x), rank);
        return x;
    };
    start = getInRank(start, rank);
    end = getInRank(end, rank);
    if (start == 0 && end == rank)
    {
        RETURN_FIRST_OUTPUT(cast, node, nodeIdx);
    }

    auto const size = std::max(0, end - start);
    auto castOutput = N_CHECK(cast->getOutput(0));
    auto* slice = ctx->network()->addSlice(*castOutput, makeDims(1, start), makeDims(1, size), makeDims(1, 1));
    RETURN_FIRST_OUTPUT(slice, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSIGMOID);
}

DEFINE_BUILTIN_OP_IMPORTER(Sin)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kSIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Sinh)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kSINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Size)
{
    // "data : T
    // An input tensor."
    auto const shape = shapeOf(inputs.at(0));

    // "outputs a int64 scalar that equals to the total number of elements of the input tensor."
    ShapeTensor const size = product(ctx, shape, 0, shape.size(), /*rank=*/0);

    return {{&size.tensor(ctx)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Slice)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    int32_t const nbInputs = node.input().size();
    // "...it uses this information to slice the input data tensor."
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    auto const dims = shapeOf(data);

    // "Slices uses starts, ends, axes and steps inputs to specify the start and
    // end dimension and step for each axis in the list of axes..."
    ShapeTensor starts;
    ShapeTensor ends;
    ShapeTensor axes;
    ShapeTensor steps;

    // If opset version >= 10 slice parameters are weights instead of attributes.
    if (ctx->getOpsetVersion() >= 10)
    {
        auto isWeightsOrEmpty
            = [&inputs, &nbInputs](int32_t index) { return nbInputs <= index || inputs.at(index).is_weights(); };

        auto isInt32 = inputs.at(1).isInt32();

        // Fast path for all INT32 constants. Required for safety engines that do not support INT64.
        if (isInt32 && isWeightsOrEmpty(1) && isWeightsOrEmpty(2) && isWeightsOrEmpty(3) && isWeightsOrEmpty(4)
            && !isDynamic(data.getDimensions()))
        {
            return staticSliceImporter(ctx, node, nodeIdx, inputs, data);
        }

        nvinfer1::ITensor* input1 = castHelper(ctx, &convertToTensor(inputs.at(1), ctx), DataType::kINT64);
        nvinfer1::ITensor* input2 = castHelper(ctx, &convertToTensor(inputs.at(2), ctx), DataType::kINT64);
        starts = ShapeTensor{*input1};
        ends = ShapeTensor{*input2};
        // "If axes are omitted, they are set to [0, ..., ndim-1]."
        axes = nbInputs > 3 && !inputs.at(3).isNullTensor() ? ShapeTensor(ctx, inputs.at(3))
                                                            : iotaShapeVector(dims.size());
        ONNXTRT_CHECK_NODE((starts.size() == axes.size()),
            "The shape of input starts misaligns with the shape of input axes. Shape of input starts = "
                << starts.size() << ", shape of input axes = " << axes.size() << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        ONNXTRT_CHECK_NODE(ends.size() == axes.size(),
            "The shape of input ends misaligns with the shape of input axes. Shape of input ends = "
                << ends.size() << ", sahpe of input axes = " << axes.size() << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        // "If steps are omitted, they are set to [1, ..., 1] of length len(starts)."
        steps = inputs.size() > 4 ? ShapeTensor(ctx, inputs.at(4)) : similar(ctx, starts, 1);
    }
    else
    {
        OnnxAttrs attrs(node, ctx);
        starts = ShapeTensor(1, attrs.get<std::vector<int64_t>>("starts"));
        ends = ShapeTensor(1, attrs.get<std::vector<int64_t>>("ends"));
        // "It's optional. If not present, will be treated as [0, 1, ..., len(starts) - 1]."
        axes = attrs.count("axes") ? ShapeTensor(1, attrs.get<std::vector<int64_t>>("axes"))
                                   : iotaShapeVector(starts.size());
        steps = similar(ctx, starts, 1);
    }

    if (axes.allValuesKnown())
    {
        // gather() requires indices to be normalized if their values are known
        normalizeAxes(axes, dims.size());
    }
    // Get dimensions of dims that correspond to axes for the computation of sizes
    auto const axesDims = gather(ctx, dims, axes);

    // ONNX has a bunch of rules for converting out of bounds starts/ends
    // indices into the actual indices to use.
    decodeOnnxStartsAndEnds(ctx, axesDims, steps, starts, ends);

    // TensorRT uses sizes of the output dimensions instead of ends.
    ShapeTensor sizes = computeSliceSizes(ctx, starts, ends, steps, axesDims);

    // Negative sizes signifies an empty slice, so clamp sizes to 0
    ShapeTensor const zeros = similar(ctx, axesDims, 0);
    sizes = max(ctx, zeros, sizes);

    nvinfer1::ISliceLayer* slice = addSlice(ctx, data, starts, sizes, steps);
    if (axes.allValuesKnown())
    {
        auto const nbDims = data.getDimensions().nbDims;
        slice->setAxes(shapeTensorToDims(axes, "slice axes", -nbDims, nbDims - 1));
    }
    else
    {
        slice->setInput(5, convertToTensor(inputs.at(3), ctx));
    }

    ctx->registerLayer(slice, node);

    RETURN_FIRST_OUTPUT(slice, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    auto& input = convertToTensor(inputs.at(0), ctx);
    auto* softmax = addSoftmax(ctx, node, nodeIdx, input);
    ONNXTRT_CHECK_NODE(softmax, "Failed to create softmax layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *softmax, shapeOf(input));
    ONNXTRT_CHECK_NODE(reshapeLayer, "Failed to create reshape layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(reshapeLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Softsign)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSOFTSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(Softplus)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSOFTPLUS);
}

DEFINE_BUILTIN_OP_IMPORTER(SpaceToDepth)
{
    checkNotInvalidType(inputs.at(0), {"BOOL", "UINT8"}, node, nodeIdx);
    // Input tensor is in NCHW format
    ONNXTRT_CHECK_NODE((inputs.at(0).shape().nbDims == 4), "The input tensor must be in the NCHW format.", node,
        nodeIdx, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    // Extract attributes
    OnnxAttrs attrs(node, ctx);
    auto blockSize = attrs.get<int32_t>("blocksize");

    nvinfer1::Permutation const perm{0, 3, 5, 1, 2, 4};

    auto inputShape = shapeOf(*tensorPtr);

    auto const N = gather(ctx, inputShape, shapeVector(0));
    auto const C = gather(ctx, inputShape, shapeVector(1));
    auto const H = gather(ctx, inputShape, shapeVector(2));
    auto const W = gather(ctx, inputShape, shapeVector(3));
    auto const blockSizeTensor = shapeVector(blockSize);

    auto const C_2 = mul(ctx, C, mul(ctx, blockSizeTensor, blockSizeTensor));
    auto const H_2 = floorDiv(ctx, H, blockSizeTensor);
    auto const W_2 = floorDiv(ctx, W, blockSizeTensor);

    // First reshape to {N, C, H / blockSize, blockSize, W / blockSize, blockSize}

    auto const firstShapeDims = concat(
        ctx, N, concat(ctx, C, concat(ctx, H_2, concat(ctx, blockSizeTensor, concat(ctx, W_2, blockSizeTensor)))));

    auto* firstShuffle = addShuffle(ctx, *tensorPtr, firstShapeDims);
    firstShuffle->setSecondTranspose(perm);
    ctx->registerLayer(firstShuffle, node);
    tensorPtr = N_CHECK(firstShuffle->getOutput(0));

    // Reshape to {N, C * blockSize * blockSize, H / blockSize, W / blockSize}
    auto secondShapeDims = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShapeDims);
    tensorPtr = N_CHECK(secondShuffle->getOutput(0));

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    size_t const numOutputs = node.output().size();

    // "input : T
    // The tensor to split"
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(inputTensor);

    // "axis : int (default is 0)
    // Which axis to split on."
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);

    // "A negative value means counting dimensions from the back.
    // Accepted range is [-rank, rank-1] where r = rank(input)."
    convertAxis(axis, inputDims.size(), node, nodeIdx);

    std::vector<int64_t> tmp(inputDims.size());
    std::iota(tmp.begin(), tmp.end(), 0);
    tmp[axis] = inputDims.size();
    ShapeTensor const subscripts = ShapeTensor(1, std::move(tmp));

    // "split : list of ints"
    // "length of each output"
    std::vector<int32_t> splitList;
    ShapeTensor sizes;
    ShapeTensor sizesLastOne;
    ShapeTensor sizeSliceAxis;
    ShapeTensor sizeSliceAxisLastOne;
    ShapeTensor splitSizesTensor;
    bool const hasSplitList = (ctx->getOpsetVersion() >= 13) ? (inputs.size() == 2) : attrs.count("split");
    if (hasSplitList)
    {
        // "Lengths of the parts can be specified using argument split."
        // In opset >= 13, split lengths are an optional input.
        if (ctx->getOpsetVersion() >= 13)
        {
            if (inputs.at(1).is_weights())
            {
                auto const splitWeights = inputs.at(1).weights();
                int64_t const* splitValues = static_cast<int64_t const*>(splitWeights.values);
                for (size_t i = 0; i < splitWeights.count(); i++)
                {
                    splitList.push_back(static_cast<int32_t>(splitValues[i]));
                }
            }
            else
            {
                splitSizesTensor = {ctx, inputs.at(1)};
            }
            // In opset >= 18, a new attribute num_outputs has been added.
            // "Either input 'split' or the attribute 'num_outputs' should be specified, but not both."
            if (ctx->getOpsetVersion() >= 18)
            {
                ONNXTRT_CHECK_NODE(!attrs.count("num_outputs"),
                    "Either 'split' should be provided as an input or 'num_outputs' should be provided as an "
                    "attribute. But not both.",
                    node, nodeIdx, ErrorCode::kINVALID_NODE);
            }
        }
        // Pre-opset 13 split lengths are provided as an attribute
        else
        {
            splitList = attrs.get<std::vector<int32_t>>("split");
        }
        ONNXTRT_CHECK_NODE((splitList.empty() || (splitList.size() == numOutputs)),
            "The number of the split attribute misaligns with the number of outputs. Number of split attributes = "
                << splitList.size() << ", number of outputs = " << numOutputs << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
    }
    else
    {
        // In opset >= 18, a new attribute 'num_outputs' has been added.
        if (ctx->getOpsetVersion() >= 18 && attrs.count("num_outputs"))
        {
            ONNXTRT_CHECK_NODE(attrs.get<int32_t>("num_outputs") == static_cast<int32_t>(numOutputs),
                "The number of node outputs is not the same as the value of 'num_outputs' attribute. num_outputs "
                "attribute value = "
                    << attrs.get<int32_t>("num_outputs") << ", number of node outputs = " << numOutputs << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
        }
        // "Otherwise, the tensor is split to equal sized parts."
        ShapeTensor const dimAxis = gather(ctx, inputDims, shapeVector(axis));
        sizeSliceAxis = floorDiv(ctx, add(ctx, dimAxis, shapeVector(numOutputs - 1)), shapeVector(numOutputs));
        sizeSliceAxisLastOne = sub(ctx, dimAxis, mul(ctx, sizeSliceAxis, shapeVector(numOutputs - 1)));
        // Check for invalid size.
        if (sizeSliceAxisLastOne.allValuesKnown())
        {
            ONNXTRT_CHECK_NODE(sizeSliceAxisLastOne[0] >= 0,
                "The last chunk size is negative, see details in https://github.com/onnx/onnx/issues/5766", node,
                nodeIdx, ErrorCode::kINVALID_NODE);
        }
        sizes = interlace(ctx, inputDims, sizeSliceAxis, subscripts);
        sizesLastOne = interlace(ctx, inputDims, sizeSliceAxisLastOne, subscripts);
    }

    std::vector<TensorOrWeights> outputs;
    outputs.reserve(numOutputs);

    ShapeTensor const zeros = similar(ctx, inputDims, 0);
    ShapeTensor const ones = similar(ctx, inputDims, 1);
    ShapeTensor starts = zeros;
    ShapeTensor startSliceAxis = shapeVector(0);
    for (int32_t i = 0; i < static_cast<int32_t>(numOutputs); ++i)
    {
        if (i)
        {
            // Advance from previous start.
            startSliceAxis = add(ctx, startSliceAxis, sizeSliceAxis);
            starts = interlace(ctx, zeros, startSliceAxis, subscripts);
        }
        if (hasSplitList)
        {
            if (splitList.empty())
            {
                sizeSliceAxis = gather(ctx, splitSizesTensor, ShapeTensor(1, std::vector<int64_t>{i}));
            }
            else
            {
                sizeSliceAxis = shapeVector(splitList[i]);
            }
            sizes = interlace(ctx, inputDims, sizeSliceAxis, subscripts);
        }
        else if (i == static_cast<int32_t>(numOutputs) - 1)
        {
            sizes = sizesLastOne;
        }

        nvinfer1::ISliceLayer* slice = addSlice(ctx, inputTensor, starts, sizes, ones);
        ctx->registerLayer(slice, node);
        outputs.emplace_back(N_CHECK(slice->getOutput(0)));
    }

    return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
}

DEFINE_BUILTIN_OP_IMPORTER(Squeeze)
{
    // "data : T
    // Tensor with at least max(dims) dimensions."
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* axesTensor{nullptr};
    // In opset >= 13, axes are an optional input
    if (ctx->getOpsetVersion() >= 13)
    {
        if (inputs.size() == 2)
        {
            axesTensor = &convertToTensor(inputs.at(1), ctx);
        }
    }
    // Pre-opset 13 axes are provided as an attribute
    else
    {
        OnnxAttrs attrs(node, ctx);
        if (attrs.count("axes"))
        {
            std::vector<int64_t> axes = attrs.get<std::vector<int64_t>>("axes");
            axesTensor = N_CHECK(
                addConstant(ctx, axes, ::ONNX_NAMESPACE::TensorProto::INT64, {1, {static_cast<int64_t>(axes.size())}})
                    ->getOutput(0));
        }
    }

    // If axes are ommitted, squeeze all dimensions with values 1
    if (!axesTensor)
    {
        auto const shape = data.getDimensions();
        ONNXTRT_CHECK_NODE(!isDynamic(shape),
            "Cannot infer squeeze dimensions from a dynamic shape! Please re-export your model with the Squeeze axes "
            "input set.",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DYNAMIC);
        std::vector<int64_t> axes{};
        for (int32_t i = 0; i < shape.nbDims; i++)
        {
            if (shape.d[i] == 1)
            {
                axes.push_back(i);
            }
        }
        axesTensor = N_CHECK(
            addConstant(ctx, axes, ::ONNX_NAMESPACE::TensorProto::INT64, {1, {static_cast<int64_t>(axes.size())}})
                ->getOutput(0));
    }

    ONNXTRT_CHECK_NODE(
        axesTensor != nullptr, "Failed to create squeeze axes!", node, nodeIdx, ErrorCode::kINTERNAL_ERROR);

    // Unsqueeze axes may be a scalar. Convert to a vector if neccessary.
    axesTensor = convertScalarToVector(ctx, axesTensor);

    // "squeezed : T
    // Reshaped tensor with same data as input."
    auto* squeezeLayer = N_CHECK(ctx->network()->addSqueeze(data, *axesTensor));
    ctx->registerLayer(squeezeLayer, node);
    RETURN_FIRST_OUTPUT(squeezeLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(STFT)
{
    /*
    STFT is implemented via decomposion into 1D convs.

    This assumes that the input signal contains non-complex values, and that all non-signal inputs are initializers.

                                STFT Input (Batch, SignalLength, 1) or (Batch, SignalLength)
                                    |
                                    |
                                Reshape to (Batch, 1, 1, SignalLength)
                                    |
                        ____________|____________
                        |                       |
                        |                       |
                        |                       |
                    Conv 1D (Real)       Conv 1D (Imaginary) with weights (dftUniqueOutputs, 1, 1, FrameLength)
                        |                       |    output shape is (Batch, dftUniqueOutputs, 1, Frames)
                        |                       |
                        |_______________________|
                                    |
                                    |
                                Concat on axis 2 - (Batch, dftUniqueOutputs, 2, Frames)
                                    |
                                    |
                                Transpose to ONNX output shape (Batch, Frames, dftUniqueOutputs, 2)
                                    |
                                    |
                                  Output
    */

    OnnxAttrs attrs(node, ctx);
    int64_t onesided = attrs.get<int64_t>("onesided", 1);
    auto* input = &convertToTensor(inputs.at(0), ctx);
    auto dims = input->getDimensions();
    // Signal must composed of real-valued inputs only - if rank == 2 or rank == 3 && dims.d[2] == 1
    ONNXTRT_CHECK_NODE(dims.nbDims == 2 || (dims.nbDims == 3 && dims.d[2] == 1),
        "TensorRT only supports STFT on real-valued signals!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Squeeze 3D shape of (Batch, SignalLength, 1) down to 2D shape of (Batch, SignalLength) to unify future unsqueeze
    // logic.
    if (dims.nbDims == 3)
    {
        std::vector<int32_t> const axes{2};
        input = squeezeTensor(ctx, *input, axes);
    }

    // Float only support.
    ONNXTRT_CHECK_NODE(input->getType() == nvinfer1::DataType::kFLOAT,
        "Input to STFT must be Float32. Received type: " << input->getType(), node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    int64_t frameStep{0};
    ShapedWeights windowWeights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    int64_t frameLength{0};

    // Frame step - must be constant as this corresponds to the strides of the convolution.
    ONNXTRT_CHECK_NODE(
        inputs.at(1).is_weights(), "FrameStep must be an initializer!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    frameStep = static_cast<int64_t*>(inputs.at(1).weights().values)[0];

    // Window - optional.
    if (inputs.size() >= 3 && !inputs.at(2).isNullTensor())
    {
        ONNXTRT_CHECK_NODE(inputs.at(2).is_weights(), "windowWeights must be an initializer!", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
        windowWeights = inputs.at(2).weights();
    }
    // Frame length - scalar value (optional)
    if (inputs.size() >= 4 && !inputs.at(3).isNullTensor())
    {
        ONNXTRT_CHECK_NODE(inputs.at(3).is_weights(), "Frame length must be an initializer!", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
        frameLength = static_cast<int64_t*>(inputs.at(3).weights().values)[0];
    }
    // If both windowWeights and frameLength are not provided, we cannot infer the size for the Window
    if (frameLength == 0 && windowWeights.values == nullptr)
    {
        ONNXTRT_CHECK_NODE(false, "Both frame_length and window inputs are missing for STFT!", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Generate missing values if necessary.
    if (frameLength == 0 && windowWeights.values != nullptr)
    {
        frameLength = windowWeights.shape.d[0];
    }
    if (frameLength != 0 && windowWeights.values == nullptr)
    {
        windowWeights = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, {frameLength}});
        for (int64_t c = 0; c < frameLength; c++)
        {
            static_cast<float*>(windowWeights.values)[c] = 1.F;
        }
    }

    // Calculate dftUniqueBins depending on the onesided attribute.
    int64_t dftUniqueBins = onesided == 1 ? ((frameLength >> 1) + 1) : frameLength;

    /*
        Generate the weights for the convolutions, of shape (dftUniqueBins, 1, 1, frameLength).

        We need to generate weights of values window[k] * e^(-2 * pi * j * w * k / n), where:
        j is the imaginary number sqrt(-1)
        w is the frequency for 0 <= w < dftUniqueBins
        k is the index in the window, for 0 <= k < frameLength
        n is equal to frameLength.

        The real and imaginary components are generated separately using Euler's formula.

        for each w in dftUniqueBins:
            for each k in frameLength:
                realWeights[w, 1, 1, k] = cos(-2 * pi * k * w / n) * window[k]
                imagWeights[w, 1, 1, k] = sin(-2 * pi * k * w / n) * window[k]
    */

    auto realWeights = ctx->createNamedTempWeights(
        ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4, {dftUniqueBins, 1, 1, frameLength}});
    auto imaginaryWeights = ctx->createNamedTempWeights(
        ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4, {dftUniqueBins, 1, 1, frameLength}});

    for (int64_t w = 0; w < static_cast<int64_t>(dftUniqueBins); w++)
    {
        for (int64_t k = 0; k < static_cast<int64_t>(frameLength); k++)
        {
            int64_t weightIndex = w * frameLength + k;
            auto angle = -2.F * M_PI * w * k / frameLength;
            static_cast<float*>(realWeights.values)[weightIndex]
                = static_cast<float>(cos(angle)) * static_cast<float*>(windowWeights.values)[k];
            static_cast<float*>(imaginaryWeights.values)[weightIndex]
                = static_cast<float>(sin(angle)) * static_cast<float*>(windowWeights.values)[k];
        }
    }

    // Unsqueeze input to [batch, 1, 1, numFrames]
    auto signalReshaped = unsqueezeTensor(ctx, *input, {1, 2});

    // 1D Convolution to calculate the real part of the signal.
    auto convReal = N_CHECK(
        ctx->network()->addConvolutionNd(*signalReshaped, dftUniqueBins, {2, {1, frameLength}}, realWeights, {}));
    convReal->setStrideNd(nvinfer1::Dims{2, {1, frameStep}});
    auto* convRealOutput = N_CHECK(convReal->getOutput(0));

    // 1D Convolution to caclulate the imaginary part of the signal.
    auto convImag
        = ctx->network()->addConvolutionNd(*signalReshaped, dftUniqueBins, {2, {1, frameLength}}, imaginaryWeights, {});
    convImag->setStrideNd(nvinfer1::Dims{2, {1, frameStep}});
    auto* convImagOutput = N_CHECK(convImag->getOutput(0));

    // Concat outputs together on axis 2, convolution outputs have shape: (Batch, dftUniqueOutputs, 1, Frames)
    std::vector<nvinfer1::ITensor*> concatInputs{convRealOutput, convImagOutput};
    auto concatLayer = N_CHECK(ctx->network()->addConcatenation(concatInputs.data(), concatInputs.size()));
    concatLayer->setAxis(2);
    auto* concatOutput = N_CHECK(concatLayer->getOutput(0));

    // Transpose to ONNX expected output shape - (Batch, dftUniqueOutputs, 2, Frames) -> (Batch, Frames,
    // dftUniqueOutputs, 2)
    auto transpose = N_CHECK(ctx->network()->addShuffle(*concatOutput));
    transpose->setFirstTranspose(nvinfer1::Permutation{{0, 3, 1, 2}});
    RETURN_FIRST_OUTPUT(transpose, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Sub)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kSUB);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Tan)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kTAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kTANH);
}

DEFINE_BUILTIN_OP_IMPORTER(TensorScatter)
{
    OnnxAttrs attrs(node, ctx);

    auto const cacheDims = inputs.at(0).shape().nbDims;
    auto const updateDims = inputs.at(1).shape().nbDims;

    // Check that past_cache and update inputs are 4D (batch_size, num_heads, sequence_length, head_size)
    ONNXTRT_CHECK_NODE(cacheDims == 4, "The past_cache tensor is required to be 4D, got " << cacheDims << "D.", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE(updateDims == 4, "The update tensor is required to be 4D, got " << updateDims << "D.", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Convert inputs to tensors
    nvinfer1::ITensor& cache = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& update = convertToTensor(inputs.at(1), ctx);

    // Get the axis attribute
    auto axis = attrs.get<int32_t>("axis", -2);
    if (axis > 0)
    {
        axis -= cacheDims;
    }
    ONNXTRT_CHECK_NODE(
        axis == -2, "Sequence dimension must be -2, got " << axis, node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Get the mode attribute
    auto const mode = attrs.get<std::string>("mode", "linear");
    ONNXTRT_CHECK_NODE(mode == "linear", "Only linear mode is supported for now, got " << mode, node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::KVCacheMode cacheMode = nvinfer1::KVCacheMode::kLINEAR;

    // Handle optional write_indices input
    nvinfer1::ITensor* writeIndices{nullptr};
    if (inputs.size() == 3 && !inputs.at(2).isNullTensor())
    {
        writeIndices = castHelper(ctx, &convertToTensor(inputs.at(2), ctx), DataType::kINT32);
    }
    else
    {
        // Create default write_indices: [batch_size] filled with zeros
        auto const cacheShape = shapeOf(cache);
        auto const batchSize = gather(ctx, cacheShape, shapeVector(0));
        writeIndices = constantOfShape(ctx,
            N_CHECK(addConstantScalar(ctx, 0, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, 1)->getOutput(0)),
            &batchSize.tensor(ctx));
    }

    // Create KVCacheUpdate layer
    auto* layer = N_CHECK(ctx->network()->addKVCacheUpdate(cache, update, *writeIndices, cacheMode));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kTHRESHOLDED_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Tile)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    // "input : T
    // Input tensor of any shape."
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(input);

    // "repeats : T1
    // 1D int64 tensor of the same length as input's dimension number,
    // includes numbers of repeated copies along input's dimensions.
    ShapeTensor const repeats{ctx, inputs.at(1)};

    ShapeTensor outputShape = mul(ctx, inputDims, repeats);
    nvinfer1::ISliceLayer* tile
        = addSlice(ctx, input, similar(ctx, inputDims, 0), outputShape, similar(ctx, inputDims, 1));
    ctx->registerLayer(tile, node);
    tile->setMode(nvinfer1::SampleMode::kWRAP);

    RETURN_FIRST_OUTPUT(tile, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get("axis", -1);
    int32_t k{1};
    if (ctx->getOpsetVersion() < 10)
    {
        k = attrs.get<int>("k");
    }
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    convertAxis(axis, nbDims, node, nodeIdx);
    uint32_t axisMask = 1 << axis;

    bool needToExpandDims = (nbDims == 1);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{1};
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK_NODE(tensorPtr, "Failed to unsqueeze input x.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    // Default is top max k.
    auto operation = nvinfer1::TopKOperation::kMAX;
    if (ctx->getOpsetVersion() >= 11)
    {
        int32_t const largest = attrs.get<int32_t>("largest", 1);
        if (largest == 0)
        {
            operation = nvinfer1::TopKOperation::kMIN;
        }
    }
    nvinfer1::ITopKLayer* layer = N_CHECK(ctx->network()->addTopK(*tensorPtr, operation, k, axisMask));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    if (ctx->getOpsetVersion() >= 10)
    {
        ONNXTRT_CHECK_NODE((inputs.size() == 2),
            "Expects two input tensors for opset >= 10: X and K. Current input size = " << inputs.size() << ".", node,
            nodeIdx, ErrorCode::kINVALID_NODE);
        nvinfer1::ITensor* kPtr = &convertToTensor(inputs.at(1), ctx);
        kPtr = convertToScalar(ctx, kPtr);
        layer->setInput(1, *kPtr);
    }
    ctx->registerLayer(layer, node);
    ONNXTRT_CHECK_NODE(layer, "Failed to add TopK layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* values = N_CHECK(layer->getOutput(0));
    nvinfer1::ITensor* indices = N_CHECK(layer->getOutput(1));

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> axes{1};
        values = squeezeTensor(ctx, *values, axes);
        ONNXTRT_CHECK_NODE(values, "Failed to squeeze the input values.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        indices = squeezeTensor(ctx, *indices, axes);
        ONNXTRT_CHECK_NODE(
            indices, "Failed to squeeze the input indices.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    // TensorRT will fuse TopK and Cast if dimension exceeds INT32_MAX
    indices = castHelper(ctx, indices, DataType::kINT64);
    return {{values, indices}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node, ctx);
    int32_t ndim = input.shape().nbDims;
    ONNXTRT_CHECK_NODE((ndim <= nvinfer1::Dims::MAX_DIMS),
        "The rank of the input tensor exceeds the maximum supported by this version of TensorRT. Current rank of "
        "inputs = "
            << ndim << ", max supported rank = " << nvinfer1::Dims::MAX_DIMS << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Permutation default_perm; // Default is to reverse dims
    for (int32_t i = 0; i < ndim; ++i)
    {
        default_perm.order[i] = ndim - 1 - i;
    }
    nvinfer1::Permutation perm = attrs.get("perm", default_perm);
    for (int32_t i = 0; i < ndim; ++i)
    {
        convertAxis(perm.order[i], ndim, node, nodeIdx);
    }
    nvinfer1::ITensor& itensor = input.is_tensor() ? input.tensor() : convertToTensor(input, ctx);
    nvinfer1::ITensor* output_tensor = transposeTensor(ctx, node, itensor, perm);
    ONNXTRT_CHECK_NODE(output_tensor, "Failed to transpose the input.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    return {{output_tensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(Trilu)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    // Data Tensor
    using eOp = nvinfer1::ElementWiseOperation;
    auto* data = &convertToTensor(inputs.at(0), ctx);
    auto const nbDims = data->getDimensions().nbDims;
    ONNXTRT_CHECK_NODE((nbDims >= 2),
        "Trilu input must have at least 2 dimensions! Current number of dimensions = " << nbDims << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    int32_t const upper = attrs.get("upper", 0);

    // Input may be in a batch so we need to get NxM dimensions
    int64_t const N = nbDims - 2;
    int64_t const M = nbDims - 1;

    // Create iota dims of NxM
    const ShapeTensor iotadims
        = concat(ctx, gather(ctx, shapeOf(*data), shapeVector(N)), gather(ctx, shapeOf(*data), shapeVector(M)));

    // Trilu can be represented via trl(A) = select(R >= C, A, 0) for keeping the lower diagonals
    // Simiarly, trl(A) = select(R <= C, A, 0) represents keeping the upper diagonals

    auto* rows = iota(ctx, iotadims, 0);
    auto* cols = iota(ctx, iotadims, 1);
    auto* zero = createZeroTensor(ctx, data);

    // k tensor shifts the number of diagonals we accept. Positive means to include k number of diagonals
    // above the main, while a negative k means to exclude the main and k number of diagonals below the main.
    // Adjust column tensor accordingly.
    if (inputs.size() == 2)
    {
        auto* k = &convertToTensor(inputs.at(1), ctx);
        std::vector<TensorOrWeights> shiftResult = elementwiseHelper(ctx, node, nodeIdx, {cols, k}, eOp::kSUB);
        cols = &convertToTensor(shiftResult.at(0), ctx);
    }

    // Unsqueeze to broadcast rows/cols if necessary during next elementwise operation.
    if (nbDims > 2)
    {
        std::vector<int32_t> batchDims(nbDims - 2);
        std::iota(batchDims.begin(), batchDims.end(), 0);
        rows = unsqueezeTensor(ctx, *rows, batchDims);
        cols = unsqueezeTensor(ctx, *cols, batchDims);
    }

    // For lower Trilus, use greaterOrEquals. For upper Trilus, use lessOrEquals
    bool const greater = upper == 0;
    std::vector<TensorOrWeights> greaterOrEqualResult = greaterLessOrEqual(ctx, node, nodeIdx, rows, cols, greater);
    auto* condition = &convertToTensor(greaterOrEqualResult.at(0), ctx);
    auto* result = N_CHECK(ctx->network()->addSelect(*condition, *data, *zero));

    RETURN_FIRST_OUTPUT(result, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Unsqueeze)
{
    // "data : T
    // Original tensor"
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* axesTensor{nullptr};

    if (ctx->getOpsetVersion() >= 13)
    {
        // Per ONNX the 2nd input is mandatory starting at opset 13, but PyTorch
        // does not comply ATM so allow for single input.
        // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#unsqueeze-13
        if (inputs.size() == 2)
        {
            axesTensor = &convertToTensor(inputs.at(1), ctx);
        }
    }

    if (!axesTensor)
    {
        OnnxAttrs attrs(node, ctx);
        // "axes : list of ints (required)
        // List of integers indicating the dimensions to be inserted."
        std::vector<int64_t> axes = attrs.get<std::vector<int64_t>>("axes");
        axesTensor = N_CHECK(
            addConstant(ctx, axes, ::ONNX_NAMESPACE::TensorProto::INT64, {1, {static_cast<int64_t>(axes.size())}})
                ->getOutput(0));
    }

    // Unsqueeze axes may be a scalar. Convert to a vector if neccessary.
    axesTensor = convertScalarToVector(ctx, axesTensor);

    ONNXTRT_CHECK_NODE(
        axesTensor != nullptr, "Failed to create unsqueeze axes!", node, nodeIdx, ErrorCode::kINTERNAL_ERROR);
    // "expanded : T
    // Reshaped tensor with same data as input."
    auto* unsqueezeLayer = N_CHECK(ctx->network()->addUnsqueeze(data, *axesTensor));
    ctx->registerLayer(unsqueezeLayer, node);
    RETURN_FIRST_OUTPUT(unsqueezeLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    checkNotInvalidType(inputs.at(0), {"BOOL", "UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensor.getDimensions().nbDims;
    ONNXTRT_CHECK_NODE((nbDims > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IResizeLayer* const layer = N_CHECK(ctx->network()->addResize(tensor));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    auto mode = attrs.get<std::string>("mode", "nearest");

    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::InterpolationMode interpolationMode = (mode == "linear" || mode == "bilinear")
        ? nvinfer1::InterpolationMode::kLINEAR
        : nvinfer1::InterpolationMode::kNEAREST;

    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ONNXTRT_CHECK_NODE((inputs.size() == 2),
            "Operator Upsample requires exactly 2 inputs. Current input size = " << inputs.size() << ".", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        if (scales_input.is_weights())
        {
            // TRT-15340: Remove this and use else path when safety support nbDims == 1.
            ShapedWeights scales_weights = scales_input.weights();
            ONNXTRT_CHECK_NODE((scales_weights.shape.nbDims == 1),
                "The scales input must be 1D. Current rank of scales input = " << scales_weights.shape.nbDims << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            // Scale factors has batch dimension.
            ONNXTRT_CHECK_NODE((scales_weights.count() == static_cast<size_t>(nbDims)),
                "The shape of the scales input must align with the dimensions of the input. Shape of scales input = "
                    << scales_weights.count() << ", dimension of input = " << nbDims << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            ONNXTRT_CHECK_NODE((scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT),
                "This version of TensorRT only supports FLOAT scales input. Current scales weight type = "
                    << scales_weights.type << ".",
                node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            float const* scales_ptr = static_cast<float const*>(scales_weights.values);
            std::vector<float> scale_factors(nbDims, 1.0F);
            for (int32_t i = 0; i < nbDims; i++)
            {
                scale_factors[i] = scales_ptr[i];
            }
            if (mode == "linear" || mode == "bilinear")
            {
                ONNXTRT_CHECK_NODE(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3),
                    "This version of TensorRT only supports linear resizing on the outermost 3 dimensions", node,
                    nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            }
            layer->setScales(scale_factors.data(), nbDims);
        }
        else
        {
            nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, tensor, scales_input);
            nvinfer1::Dims const outDims = resizeShape->getDimensions();
            ONNXTRT_CHECK_NODE((outDims.nbDims == 1),
                "The scales input must be 1D. Current rank of the scales input = " << outDims.nbDims << ".", node,
                nodeIdx, ErrorCode::kINVALID_NODE);
            // Scale factors has batch dimension.
            ONNXTRT_CHECK_NODE((outDims.d[0] == nbDims),
                "The shape of the scales input must align with the dimensions of the input. Current shape of the "
                "scales input = "
                    << outDims.nbDims << ", dimension of the input = " << nbDims << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            ONNXTRT_CHECK_NODE((resizeShape->getType() == DataType::kINT32),
                "Resize output shape type must be integral. The actual type is "
                    + getTrtDtypeName(resizeShape->getType()) + ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            layer->setInput(1, *resizeShape);
        }
    }
    else
    {
        // TRT-15340: Adapt to use resizeShapeTensor instead when safety support nbDims == 1.
        ONNXTRT_CHECK_NODE(
            attrs.count("scales"), "Attribute scales is missing.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_ATTR);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ONNXTRT_CHECK_NODE((static_cast<int32_t>(scales.size()) == nbDims),
            "The shape of the scales input must align with the dimensions of the input. Current shape of the scales "
            "input = "
                << scales.size() << ", dimension of the input = " << nbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        std::vector<float> scale_factors(nbDims, 1.0F);
        for (int32_t i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales[i];
        }
        if (mode == "linear" || mode == "bilinear")
        {
            ONNXTRT_CHECK_NODE(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3),
                "This version of TensorRT only supports linear resizing on the outermost 3 dimensions", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scale_factors.data(), nbDims);
    }
    ctx->registerLayer(layer, node);
    layer->setResizeMode(interpolationMode);
    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
    layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Where)
{
    checkNotInvalidType(inputs.at(1), {"UINT8"}, node, nodeIdx);
    checkNotInvalidType(inputs.at(2), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor* condition = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* y = &convertToTensor(inputs.at(2), ctx);

    broadcastTensors(ctx, x, y, condition);

    nvinfer1::Dims cDims = condition->getDimensions();
    nvinfer1::Dims xDims = x->getDimensions();
    nvinfer1::Dims yDims = y->getDimensions();

    ONNXTRT_CHECK_NODE((cDims.nbDims == xDims.nbDims),
        "The rank of the condition input tensor must be the same of the input x tensor. Rank of the condition input "
        "tensor = "
            << cDims.nbDims << ", rank of the input x tensor = " << xDims.nbDims << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((cDims.nbDims == yDims.nbDims),
        "The rank of the condition input tensor must be the same of the input y tensor. Rank of the condition input "
        "tensor = "
            << cDims.nbDims << ", rank of the input y tensor = " << yDims.nbDims << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto* layer = N_CHECK(ctx->network()->addSelect(*condition, *x, *y));
    ctx->registerLayer(layer, node);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

// Copies the given field into the fieldData map, returns data and number of T elements in the vector in which the data
// was copied into.
template <typename T>
std::tuple<void const*, size_t> copyField(
    T const& field, std::string const& fieldName, StringMap<std::vector<uint8_t>>& fieldData)
{
    constexpr size_t nbBytes{sizeof(T)};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), &field, nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size() / nbBytes);
}

template <typename T>
std::tuple<void const*, size_t> copyField(
    std::vector<T> const& repeatedField, std::string const& fieldName, StringMap<std::vector<uint8_t>>& fieldData)
{
    size_t const nbBytes{sizeof(T) * repeatedField.size()};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), repeatedField.data(), nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size() / sizeof(T));
}

std::tuple<void const*, size_t> copyField(
    std::string const& field, std::string const& fieldName, StringMap<std::vector<uint8_t>>& fieldData)
{
    static_assert(sizeof(std::string::value_type) == sizeof(uint8_t), "String type does not have 1 byte elements");
    std::copy(field.begin(), field.end(), std::back_inserter(fieldData[fieldName]));
    // Append \0 as end of C style string.
    fieldData[fieldName].push_back('\0');
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<void const*, size_t> copyField(std::vector<std::string> const& repeatedField, std::string const& fieldName,
    StringMap<std::vector<uint8_t>>& fieldData)
{
    static_assert(sizeof(std::string::value_type) == sizeof(uint8_t), "String type does not have 1 byte elements");
    for (auto const& field : repeatedField)
    {
        std::copy(field.begin(), field.end(), std::back_inserter(fieldData[fieldName]));
        // Append \0 as end of C style string.
        fieldData[fieldName].push_back('\0');
    }
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<void const*, size_t> copyField(
    ShapedWeights const& field, std::string const& fieldName, StringMap<std::vector<uint8_t>>& fieldData)
{
    // Weights do not require a copy
    return std::make_tuple(field.values, field.count());
}

// Load plugin fields from an ONNX node, using fieldData for temporary allocations.
std::vector<nvinfer1::PluginField> loadFields(StringMap<std::vector<uint8_t>>& fieldData, OnnxAttrs const& attrs,
    nvinfer1::PluginFieldCollection const* fieldNames, ImporterContext* ctx)
{
    std::vector<nvinfer1::PluginField> fields{};
    for (int32_t i = 0; i < fieldNames->nbFields; ++i)
    {
        // Some plugins may have default values for fields that map to optional attributes in an ONNX graph.
        if (!attrs.count(fieldNames->fields[i].name))
        {
            LOG_WARNING("Attribute " << fieldNames->fields[i].name
                                     << " not found in plugin node! Ensure that the plugin creator has a default value "
                                        "defined or the engine may fail to build.");
            continue;
        }

        // Name must be retrieved from the map so that it is alive for long enough.
        std::string const& fieldName
            = fieldData.emplace(fieldNames->fields[i].name, std::vector<uint8_t>{}).first->first;
        void const* data{nullptr};
        int32_t length{0};
        nvinfer1::PluginFieldType type{};
        switch (attrs.type(fieldName))
        {
        case ::ONNX_NAMESPACE::AttributeProto::FLOAT:
            std::tie(data, length) = copyField(attrs.get<float>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kFLOAT32;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::INT:
            std::tie(data, length) = copyField(attrs.get<int32_t>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kINT32;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::STRING:
            std::tie(data, length) = copyField(attrs.get<std::string>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kCHAR;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::FLOATS:
            std::tie(data, length) = copyField(attrs.get<std::vector<float>>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kFLOAT32;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::INTS:
            std::tie(data, length) = copyField(attrs.get<std::vector<int>>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kINT32;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::STRINGS:
            std::tie(data, length) = copyField(attrs.get<std::vector<std::string>>(fieldName), fieldName, fieldData);
            type = nvinfer1::PluginFieldType::kCHAR;
            break;
        case ::ONNX_NAMESPACE::AttributeProto::TENSOR:
        {
            ShapedWeights tensor{attrs.get<ShapedWeights>(fieldName)};
            std::tie(data, length) = copyField(tensor, fieldName, fieldData);
            switch (tensor.type)
            {
            case ::ONNX_NAMESPACE::TensorProto::FLOAT: type = nvinfer1::PluginFieldType::kFLOAT32; break;
            case ::ONNX_NAMESPACE::TensorProto::INT8: type = nvinfer1::PluginFieldType::kINT8; break;
            case ::ONNX_NAMESPACE::TensorProto::INT16: type = nvinfer1::PluginFieldType::kINT16; break;
            case ::ONNX_NAMESPACE::TensorProto::INT32: type = nvinfer1::PluginFieldType::kINT32; break;
            case ::ONNX_NAMESPACE::TensorProto::STRING: type = nvinfer1::PluginFieldType::kCHAR; break;
            case ::ONNX_NAMESPACE::TensorProto::FLOAT16: type = nvinfer1::PluginFieldType::kFLOAT16; break;
            case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: type = nvinfer1::PluginFieldType::kBF16; break;
            case ::ONNX_NAMESPACE::TensorProto::DOUBLE: type = nvinfer1::PluginFieldType::kFLOAT64; break;
            case ::ONNX_NAMESPACE::TensorProto::UNDEFINED:
            case ::ONNX_NAMESPACE::TensorProto::UINT8:
            case ::ONNX_NAMESPACE::TensorProto::UINT16:
            case ::ONNX_NAMESPACE::TensorProto::INT64:
            case ::ONNX_NAMESPACE::TensorProto::BOOL:
            case ::ONNX_NAMESPACE::TensorProto::UINT32:
            case ::ONNX_NAMESPACE::TensorProto::UINT64:
            case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:
            case ::ONNX_NAMESPACE::TensorProto::COMPLEX128:
                // ::ONNX_NAMESPACE::TensorProto::DataType_Name function not available in protobuf-lite.
                std::stringstream ss{};
                ss << "Tensor type: ";
#if USE_LITE_PROTOBUF
                ss << static_cast<::ONNX_NAMESPACE::TensorProto::DataType>(tensor.type);
#else
                ss << ::ONNX_NAMESPACE::TensorProto::DataType_Name(
                    static_cast<::ONNX_NAMESPACE::TensorProto::DataType>(tensor.type));
#endif // USE_LITE_PROTOBUF
                ss << " is unsupported in plugin fields." << std::endl;
                MAKE_ERROR(ss.str(), ErrorCode::kUNSUPPORTED_NODE);
            }
            break;
        }
        case ::ONNX_NAMESPACE::AttributeProto::UNDEFINED:
        case ::ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR:
        case ::ONNX_NAMESPACE::AttributeProto::GRAPH:
        case ::ONNX_NAMESPACE::AttributeProto::TENSORS:
        case ::ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS:
        case ::ONNX_NAMESPACE::AttributeProto::GRAPHS:
        case ::ONNX_NAMESPACE::AttributeProto::TYPE_PROTO:
        case ::ONNX_NAMESPACE::AttributeProto::TYPE_PROTOS:
            // ::ONNX_NAMESPACE::AttributeProto::AttributeType_Name function not available in protobuf-lite.
            std::stringstream ss{};
            ss << "Attributes of type: ";
#if USE_LITE_PROTOBUF
            ss << attrs.type(fieldName);
#else
            ss << ::ONNX_NAMESPACE::AttributeProto::AttributeType_Name(attrs.type(fieldName));
#endif // USE_LITE_PROTOBUF
            ss << " are unsupported as plugin fields." << std::endl;
            MAKE_ERROR(ss.str(), ErrorCode::kUNSUPPORTED_NODE);
        }
        fields.emplace_back(fieldName.c_str(), data, type, length);
    }
    return fields;
}

nvinfer1::IPluginV2Layer* addPluginLayer(ImporterContext* ctx, std::vector<nvinfer1::ITensor*> const& pluginInputs,
    std::vector<nvinfer1::ITensor*> const& /* pluginShapeInputs */, nvinfer1::IPluginV2& plugin)
{
    return N_CHECK(ctx->network()->addPluginV2(pluginInputs.data(), pluginInputs.size(), plugin));
}

nvinfer1::IPluginV3Layer* addPluginLayer(ImporterContext* ctx, std::vector<nvinfer1::ITensor*> const& pluginInputs,
    std::vector<nvinfer1::ITensor*> const& pluginShapeInputs, nvinfer1::IPluginV3& plugin)
{
    return N_CHECK(ctx->network()->addPluginV3(
        pluginInputs.data(), pluginInputs.size(), pluginShapeInputs.data(), pluginShapeInputs.size(), plugin));
}

template <typename TPluginCreator>
NodeOutputs addPluginWithCreator(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const& nodeIdx,
    std::string const& pluginNamespace, std::vector<TensorOrWeights>& inputs, OnnxAttrs const& attrs,
    nvinfer1::IPluginCreatorInterface* creator)
{
    ONNXTRT_CHECK_NODE(creator, "Invalid plugin creator.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::PluginFieldCollection const* fieldNames = static_cast<TPluginCreator*>(creator)->getFieldNames();

    StringMap<std::vector<uint8_t>> fieldData{};
    std::vector<nvinfer1::PluginField> fields = loadFields(fieldData, attrs, fieldNames, ctx);

    std::unordered_set<int32_t> shapeInputIdxsSet{};
    auto const nbInputs{static_cast<int32_t>(inputs.size())};

    if (attrs.count("tensorrt_plugin_shape_input_indices"))
    {
        if (std::strcmp(creator->getInterfaceInfo().kind, "PLUGIN CREATOR_V1") != 0)
        {
            ONNXTRT_CHECK_NODE(
                attrs.type("tensorrt_plugin_shape_input_indices") == ::ONNX_NAMESPACE::AttributeProto::INTS,
                "Shape input indices defined, but attribute tensorrt_plugin_shape_input_indices has unsupported type. "
                "Only AttributeProto::INTS is supported.",
                node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            auto shapeInputIdxsVec = attrs.get<std::vector<int32_t>>("tensorrt_plugin_shape_input_indices");
            std::vector<bool> issuedWarningIdxs(nbInputs, false);
            for (auto const& curr : shapeInputIdxsVec)
            {
                // check for out-of-range indices: index must be in [-n, n-1] where n = number of inputs.
                ONNXTRT_CHECK_NODE((curr < nbInputs) && (curr >= (-nbInputs)),
                    "Out-of-range shape input index: " << curr, node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

                auto const& e = (curr >= 0) ? curr : nbInputs + curr;
                auto res = shapeInputIdxsSet.insert(e);
                if (!res.second && !issuedWarningIdxs[e])
                {
                    LOG_WARNING("Node " << getNodeName(node) << " has duplicate shape input index: " << curr);
                    issuedWarningIdxs[e] = true;
                }
            }
        }
        else
        {
            LOG_WARNING("Node " << getNodeName(node)
                                << " has shape input indices defined, but plugin creator is for an "
                                   "IPluginV2-derivative plugin. Ignoring shape input indices.");
        }
    }

    std::string const pluginName{node.op_type()};

    auto const plugin
        = createPlugin(ctx, node, getNodeName(node), pluginNamespace, static_cast<TPluginCreator*>(creator), fields);

    ONNXTRT_CHECK_NODE(plugin, "Could not create the plugin.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    std::vector<nvinfer1::ITensor*> pluginInputs{};
    std::vector<nvinfer1::ITensor*> pluginShapeInputs{};

    for (int32_t idx{}; idx < nbInputs; ++idx)
    {
        auto input = inputs[idx];
        auto pluginInputVec
            = shapeInputIdxsSet.find(idx) != shapeInputIdxsSet.end() ? &pluginShapeInputs : &pluginInputs;
        if (input.isNullTensor())
        {
            LOG_VERBOSE("Found unset input for " << pluginName << ".");
            pluginInputVec->push_back(nullptr);
            continue;
        }
        nvinfer1::ITensor* inputTensor = &convertToTensor(input, ctx);
        if (onlySupportInt32TRTPlugin(pluginName) && inputTensor->getType() == nvinfer1::DataType::kINT64)
        {
            LOG_VERBOSE("The TRT plugin (" << pluginName << ") doesn't support INT64 for inputs, will cast to INT32.");
            pluginInputVec->emplace_back(castHelper(ctx, inputTensor, nvinfer1::DataType::kINT32));
        }
        else
        {
            pluginInputVec->emplace_back(inputTensor);
        }
    }

    LOG_INFO("Successfully created plugin: " << pluginName);

    auto* layer = addPluginLayer(ctx, pluginInputs, pluginShapeInputs, *plugin);
    ONNXTRT_CHECK_NODE(layer, "Could not add the plugin layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Xor)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kXOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Shrink)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);

    auto originalType = x->getType();
    ONNXTRT_CHECK_NODE(
        (originalType == DataType::kFLOAT || originalType == DataType::kHALF || originalType == DataType::kINT8
            || originalType == DataType::kINT32 || originalType == DataType::kINT64),
        "Only FLOAT, HALF, INT8, INT32, and INT64 are supported in Shrink. The current type = "
            + getTrtDtypeName(originalType) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    x = castHelper(ctx, x, DataType::kFLOAT);

    // get attrs
    OnnxAttrs attrs(node, ctx);
    float const lambd = attrs.get<float>("lambd", 0.5F);
    float const bias = attrs.get<float>("bias", 0.0F);

    // prepare Constant Tensors
    nvinfer1::ITensor* lambdTensor
        = addConstant(ctx, std::vector<float>{lambd}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    broadcastTensors(ctx, lambdTensor, x); // align rank

    nvinfer1::ITensor* negLambdTensor
        = addConstant(ctx, std::vector<float>{-lambd}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    broadcastTensors(ctx, negLambdTensor, x);

    nvinfer1::ITensor* biasTensor
        = addConstant(ctx, std::vector<float>{bias}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    broadcastTensors(ctx, biasTensor, x);

    nvinfer1::ITensor* zeroTensor
        = addConstant(ctx, std::vector<float>{0.}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    broadcastTensors(ctx, zeroTensor, x);

    // If x > lambd, y = x - bias; Otherwise, y = 0
    std::vector<TensorOrWeights> xGreaterThanLambd
        = elementwiseHelper(ctx, node, nodeIdx, {x, lambdTensor}, nvinfer1::ElementWiseOperation::kGREATER);

    std::vector<TensorOrWeights> xMinusBias
        = elementwiseHelper(ctx, node, nodeIdx, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUB);

    auto firstSelectLayer = N_CHECK(ctx->network()->addSelect(
        convertToTensor(xGreaterThanLambd.at(0), ctx), convertToTensor(xMinusBias.at(0), ctx), *zeroTensor));
    nvinfer1::ITensor* output = N_CHECK(firstSelectLayer->getOutput(0));

    // If x < -lambd, y = x + bias;
    std::vector<TensorOrWeights> xLessThanMinusLambd
        = elementwiseHelper(ctx, node, nodeIdx, {x, negLambdTensor}, nvinfer1::ElementWiseOperation::kLESS);

    std::vector<TensorOrWeights> xAddBias
        = elementwiseHelper(ctx, node, nodeIdx, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUM);

    auto* layer = N_CHECK(ctx->network()->addSelect(
        convertToTensor(xLessThanMinusLambd.at(0), ctx), convertToTensor(xAddBias.at(0), ctx), *output));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);

    // cast back to originalType
    return {{castHelper(ctx, N_CHECK(layer->getOutput(0)), originalType)}};
}

DEFINE_BUILTIN_OP_IMPORTER(HardSwish)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE(
        (x->getType() == DataType::kFLOAT || x->getType() == DataType::kHALF || x->getType() == DataType::kINT8),
        "Only FLOAT, HALF or INT8 input is supported for the HardSwish operator in this version of TensorRT. "
        "The current type = "
            + getTrtDtypeName(x->getType()) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // activationHelper does not support const and constexpr (compile failed)
    float kALPHA{1.F / 6};
    float kBETA{0.5F};
    std::vector<TensorOrWeights> activationResult
        = activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &kALPHA, &kBETA);

    return elementwiseHelper(ctx, node, nodeIdx, {x, activationResult.at(0)}, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(NonZero)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    auto const t = x->getType();
    ONNXTRT_CHECK_NODE((t == DataType::kFLOAT || t == DataType::kHALF || t == DataType::kBF16 || t == DataType::kINT32
                           || t == DataType::kINT64 || t == DataType::kINT8 || t == DataType::kBOOL),
        "Only FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, INT8 or BOOL input is supported for the NonZero operator in "
        "this version of TensorRT. The current type is "
            + getTrtDtypeName(t) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    auto* layer = N_CHECK(ctx->network()->addNonZero(*x, DataType::kINT64));
    ctx->registerLayer(layer, node);
    return {{N_CHECK(layer->getOutput(0))}};
}

DEFINE_BUILTIN_OP_IMPORTER(Mish)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    ONNXTRT_CHECK_NODE((x->getType() == DataType::kFLOAT || x->getType() == DataType::kHALF),
        "Only FLOAT32 or FLOAT16 input is supported for the Mish operator in this version of TensorRT. "
        "The current type = "
            + getTrtDtypeName(x->getType()) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    std::vector<TensorOrWeights> softPlusOutput
        = activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kSOFTPLUS);

    std::vector<TensorOrWeights> tanhOutput
        = activationHelper(ctx, node, nodeIdx, softPlusOutput, nvinfer1::ActivationType::kTANH);

    return elementwiseHelper(ctx, node, nodeIdx, {x, tanhOutput.at(0)}, nvinfer1::ElementWiseOperation::kPROD);
}

// Any ops that are not supported will attempt to import as plugins.
DEFINE_BUILTIN_OP_IMPORTER(FallbackPluginImporter)
{
    OnnxAttrs attrs(node, ctx);
    std::string const pluginName{node.op_type()};
    std::string const pluginVersion{attrs.get<std::string>("plugin_version", "1")};
    std::string pluginNamespace{attrs.get<std::string>("plugin_namespace", "")};

    LOG_INFO("Searching for plugin: " << pluginName << ", plugin_version: " << pluginVersion
                                      << ", plugin_namespace: " << pluginNamespace);
    nvinfer1::IPluginCreatorInterface* creator = importPluginCreator(ctx, pluginName, pluginVersion, pluginNamespace);

    // Fallback check to the node's domain
    if (!creator)
    {
        pluginNamespace = node.domain();
        LOG_INFO("Searching for plugin wth node domain namespace: " << pluginNamespace);
        creator = importPluginCreator(ctx, pluginName, pluginVersion, pluginNamespace);
    }

    ONNXTRT_CHECK_NODE(creator, "Plugin not found, are the plugin name, version, and namespace correct?", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    auto const creatorVersion = getPluginCreatorVersion(creator);

    switch (creatorVersion)
    {
    case CreatorVersion::kV1:
    {
        return addPluginWithCreator<nvinfer1::IPluginCreator>(
            ctx, node, nodeIdx, pluginNamespace, inputs, attrs, creator);
    }
    case CreatorVersion::kV3ONE:
    {
        return addPluginWithCreator<nvinfer1::IPluginCreatorV3One>(
            ctx, node, nodeIdx, pluginNamespace, inputs, attrs, creator);
    }
    case CreatorVersion::kV3QUICK:
    {
        return addPluginWithCreator<nvinfer1::IPluginCreatorV3Quick>(
            ctx, node, nodeIdx, pluginNamespace, inputs, attrs, creator);
    }
    default: ONNXTRT_CHECK(false, "Unsupported plugin creator version.", ErrorCode::kUNSUPPORTED_NODE);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(LocalFunctionImporter)
{
    auto function = ctx->localFunctions().at(node.op_type());
    ONNXTRT_CHECK_NODE(node.input().size() == function.input().size(),
        "LocalFunction has an unexpected number of inputs! Number of node inputs = "
            << node.input().size() << ", number of function inputs = " << function.input().size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Create a namescope local to the subgraph in the local function
    NameScope nameScope(*ctx);

    // We need to map local input names to the tensors from the outside scope. These are the cases
    // to consider:
    //
    // When the insideScopeName != outsideScopeName, there are two cases:
    //     1. The insideScopeName does not intersect with any names from the outer scope. We can add
    //        it directly to ctx->tensors() and remove it when we exit the local function.
    //     2. The insideScopeName is the same as one from the outer scope. In this case we need to
    //        save the original tensor before adding the new inside scope mapping. We will move the
    //        saved tensor back into ctx->tensors() once we exit the local function.
    //
    // When the insideScopeName == outsideScopeName, we just use the tensor from the outside scope directly.

    std::vector<std::string> localInputs;
    StringMap<TensorOrWeights> savedTensors;
    for (int32_t i = 0; i < function.input().size(); i++)
    {
        auto outsideScopeName = node.input(i);
        auto insideScopeName = function.input(i);
        if (outsideScopeName != insideScopeName)
        {
            // Case one: simply keep track of the name to remove after.
            localInputs.push_back(insideScopeName);

            // Case two: also save the original tensor to restore it after.
            if (auto handle = ctx->tensors().extract(insideScopeName))
            {
                savedTensors.insert(std::move(handle));
            }
            ctx->tensors().insert({insideScopeName, ctx->tensors().at(outsideScopeName)});
        }
        ONNXTRT_CHECK_NODE(ctx->tensors().count(insideScopeName), "Could not find mapping of local function input!",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
    }

    // Create attribute map for the local function instance. Attributes can have default values (from the parent
    // FunctionProto) or local values (from the NodeProto instance of the Function).

    StringMap<::ONNX_NAMESPACE::AttributeProto const*> attrMap;
    // Add local values first as they override any default values.
    for (auto const& attr : node.attribute())
    {
        attrMap.insert({attr.name(), &attr});
    }
    // Add default values
    for (auto const& attr : function.attribute_proto())
    {
        attrMap.insert({attr.name(), &attr});
    }

    // Push current function name to top of stack in order to properly set layer metadata and track attributes
    ctx->localFunctionStack().push_back({node.op_type(), getNodeName(node), attrMap});

    // Log current stack of functions for debugging nested functions.
    auto prettyPrintFunctionStack = [ctx]() {
        std::stringstream stackStream;
        stackStream << "Function stack: [";
        size_t stackSize = ctx->localFunctionStack().size();
        for (size_t i = 0; i < stackSize; i++)
        {
            auto const& func = ctx->localFunctionStack()[i];
            stackStream << func.nodeName << " (" << func.functionName << ")";
            if (i != stackSize - 1)
            {
                stackStream << ", ";
            }
        }
        stackStream << "]";
        return stackStream.str();
    };

    LOG_VERBOSE(prettyPrintFunctionStack());

    for (auto const& node : function.node())
    {
        try
        {
            onnx2trt::parseNode(ctx, node, nodeIdx);
        }
        catch (OnnxTrtException& e)
        {
            if (ctx->localFunctions().count(node.op_type()))
            {
                ctx->localFunctionStack().pop_back();
                ONNXTRT_THROW(e.getStatus());
            }
            else
            {
                // This is a leaf node. Add local function stack to error log.
                size_t stackSize = ctx->localFunctionStack().size();
                std::vector<std::string> localFunctionStackString{};
                std::vector<char const*> localFunctionStackChar{};
                for (size_t i = 0; i < stackSize; i++)
                {
                    auto const& func = ctx->localFunctionStack()[i];
                    localFunctionStackString.push_back(func.nodeName + " (" + func.functionName + ")");
                }
                ctx->localFunctionErrors().push_back(localFunctionStackString);
                for (size_t i = 0; i < stackSize; i++)
                {
                    localFunctionStackChar.push_back(ctx->localFunctionErrors().back()[i].c_str());
                }
                ctx->localFunctionStack().pop_back();

                Status status = e.getStatus();
                ONNXTRT_THROW(Status(status.code(), std::string(status.desc()), std::string(status.file()),
                    status.line(), std::string(status.func()), status.node(), std::string(status.nodeName()),
                    std::string(status.nodeOperator()), localFunctionStackChar));
            }
        }
    }

    // Create output vector
    std::vector<TensorOrWeights> outputs;
    for (auto const& output : function.output())
    {
        ONNXTRT_CHECK_NODE(
            ctx->tensors().count(output), "Could not find output tensor!", node, nodeIdx, ErrorCode::kINVALID_NODE);
        outputs.push_back(TensorOrWeights(ctx->tensors().at(output)));
    }

    // When exiting the function scope, remove all localInputs and restore the tensors in savedTensors.
    for (auto& name : localInputs)
    {
        ctx->tensors().erase(name);
        if (auto handle = savedTensors.extract(name))
        {
            ctx->tensors().insert(std::move(handle));
        }
    }

    ctx->localFunctionStack().pop_back();

    return outputs;
}

// INetwork Serialization importer functions - TODO: Move to it's own file?

DEFINE_BUILTIN_OP_IMPORTER(TRT_Scale)
{
    ONNXTRT_CHECK_NODE((inputs.size() >= 1), "Input is required.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE(
        (inputs.at(0).is_tensor()), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    if (inputs.size() >= 2)
    {
        ONNXTRT_CHECK_NODE((inputs.at(1).is_weights()), "The second input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
    }
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int32_t counter = 1;

    nvinfer1::ScaleMode mode = attrs.get<nvinfer1::ScaleMode>("mode");

    // check if there's no weights at all
    // if no weights, just choose datatype of the input tensor
    // This is based on the assumption that weights should be
    // the same datatype as inputs
    auto type = inputs.size() > 1 ? inputs.at(1).weights().type : trtDataTypeToONNX(inputs.at(0).tensor().getType());

    auto scale = ShapedWeights::empty(type);
    auto shift = ShapedWeights::empty(type);
    auto power = ShapedWeights::empty(type);

    if (attrs.get<bool>("scale"))
    {
        ONNXTRT_CHECK_NODE((inputs.at(counter).is_weights()), "The scale input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        scale = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("shift"))
    {
        ONNXTRT_CHECK_NODE((inputs.at(counter).is_weights()), "The shift input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        shift = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("power"))
    {
        ONNXTRT_CHECK_NODE((inputs.at(counter).is_weights()), "The power input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        power = inputs.at(counter++).weights();
    }

    nvinfer1::IScaleLayer* layer = N_CHECK(ctx->network()->addScale(input, mode, shift, scale, power));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");
    bool zeroIsPlaceholder = attrs.get<bool>("zero_is_placeholder");

    nvinfer1::IShuffleLayer* layer = N_CHECK(ctx->network()->addShuffle(input));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    layer->setFirstTranspose(perm1);
    layer->setSecondTranspose(perm2);
    layer->setZeroIsPlaceholder(zeroIsPlaceholder);

    if (inputs.size() == 1)
    {
        if (attrs.count("reshape_dims"))
        {
            nvinfer1::Dims reshapeDims = attrs.get<nvinfer1::Dims>("reshape_dims");
            layer->setReshapeDimensions(reshapeDims);
        }
    }
    else
    {
        ONNXTRT_CHECK_NODE(inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_TopK_Min)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "The second input must be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kWeights = inputs.at(1).weights();
    int k = *static_cast<int*>(kWeights.values);

    int32_t axes = 1 << (attrs.get<int32_t>("axis"));

    nvinfer1::ITopKLayer* layer = N_CHECK(ctx->network()->addTopK(input, nvinfer1::TopKOperation::kMIN, k, axes));
    ctx->registerLayer(layer, node);

    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MatMul)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input0 = inputs.at(0).tensor();
    auto& input1 = inputs.at(1).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::MatrixOperation op0 = attrs.get<nvinfer1::MatrixOperation>("op_0");
    nvinfer1::MatrixOperation op1 = attrs.get<nvinfer1::MatrixOperation>("op_1");

    nvinfer1::IMatrixMultiplyLayer* layer = N_CHECK(ctx->network()->addMatrixMultiply(input0, op0, input1, op1));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RaggedSoftmax)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();
    auto& bounds = inputs.at(1).tensor();

    nvinfer1::IRaggedSoftMaxLayer* layer = N_CHECK(ctx->network()->addRaggedSoftMax(input, bounds));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int32_t nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;
    nvinfer1::Dims kernelSize = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool excludePadding(true);
    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, excludePadding);
    float blend = attrs.get<float>("blend");

    nvinfer1::IPoolingLayer* layer
        = N_CHECK(ctx->network()->addPoolingNd(input, nvinfer1::PoolingType::kMAX_AVERAGE_BLEND, kernelSize));
    ctx->registerLayer(layer, node);
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(excludePadding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    layer->setBlendFactor(blend);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

#if ENABLE_STD_PLUGIN
DEFINE_BUILTIN_OP_IMPORTER(TRT_PluginV2)
{
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IPluginRegistry& registry = ctx->network()->getBuilder().getPluginRegistry();

    std::string name = attrs.get<std::string>("name");
    std::string version = attrs.get<std::string>("version");
    std::string nspace = attrs.get<std::string>("namespace");
    std::string buffer = attrs.get<std::string>("data");

    nvinfer1::IPluginCreator* creator = registry.getPluginCreator(name.c_str(), version.c_str(), nspace.c_str());
    ONNXTRT_CHECK_NODE(creator, "Plugin not found, are the plugin name, version, and namespace correct?", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);

    auto const plugin = creator->deserializePlugin("", buffer.data(), buffer.size());

    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        ONNXTRT_CHECK_NODE(input.is_tensor(), "The input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        nvinfer1::ITensor* inputTensor = &input.tensor();
        if (onlySupportInt32TRTPlugin(name) && inputTensor->getType() == DataType::kINT64)
        {
            LOG_VERBOSE("The TRT plugin (" << name << ") doesn't support INT64 for inputs, will cast to INT32.");
            tensors.emplace_back(castHelper(ctx, inputTensor, DataType::kINT32));
        }
        else
        {
            tensors.emplace_back(inputTensor);
        }
    }

    nvinfer1::IPluginV2Layer* layer = N_CHECK(ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin));
    ctx->registerLayer(layer, node);
    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}
#endif // ENABLE_STD_PLUGIN

DEFINE_BUILTIN_OP_IMPORTER(TRT_Gather)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& data = inputs.at(0).tensor();
    auto& indices = inputs.at(1).tensor();
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbElementWiseDims = attrs.get<int32_t>("nbElementWiseDims", 0);
    int32_t r = data.getDimensions().nbDims;

    ONNXTRT_CHECK_NODE((indices.getType() == DataType::kINT32),
        "This version of TensorRT only supports INT32 input indices. The current indices type = "
            + getTrtDtypeName(indices.getType()) + ".",
        node, nodeIdx, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE((r >= 1), "0D input data is not allowed.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ONNXTRT_CHECK_NODE((-r <= axis && axis < r),
        "The attribute axis should be in range [-r, r-1], where r is the rank of the input. Provided r = "
            << r << ", axis = " << axis << ".",
        node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);

    if (axis < 0)
    {
        axis += r;
    }

    nvinfer1::IGatherLayer* layer = N_CHECK(ctx->network()->addGather(data, indices, axis));
    ONNXTRT_CHECK_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    layer->setNbElementWiseDims(nbElementWiseDims);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Slice)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::ISliceLayer* layer;
    // If only one input, then, start, size, stride are all attributes.
    if (inputs.size() == 1)
    {
        OnnxAttrs attrs(node, ctx);
        auto start = attrs.get<nvinfer1::Dims>("start");
        auto size = attrs.get<nvinfer1::Dims>("size");
        auto stride = attrs.get<nvinfer1::Dims>("stride");
        layer = N_CHECK(ctx->network()->addSlice(input, start, size, stride));
    }
    else
    {
        // start, size, stride are all inputs
        ONNXTRT_CHECK_NODE((inputs.size() == 4),
            "Exactly 4 inputs are required by TRT_Slice. Current input size = " << inputs.size() << ".", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        ShapeTensor const start{ctx, inputs.at(1)};
        ShapeTensor const size{ctx, inputs.at(2)};
        ShapeTensor const stride{ctx, inputs.at(3)};
        layer = addSlice(ctx, input, start, size, stride);
    }
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Resize)
{
    ONNXTRT_CHECK_NODE(
        inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = N_CHECK(ctx->network()->addResize(input));
    ctx->registerLayer(layer, node);

    OnnxAttrs attrs(node, ctx);
    auto const mode = attrs.get<nvinfer1::InterpolationMode>("mode");
    auto const transformation = attrs.get<nvinfer1::ResizeCoordinateTransformation>("coordTransform");
    auto const selector = attrs.get<nvinfer1::ResizeSelector>("resizeSelector");
    auto const roundMode = attrs.get<nvinfer1::ResizeRoundMode>("round_mode");

    layer->setResizeMode(mode);
    layer->setSelectorForSinglePixel(selector);
    layer->setCoordinateTransformation(transformation);
    layer->setNearestRounding(roundMode);

    if (inputs.size() == 1)
    {
        auto outputDims = attrs.get<nvinfer1::Dims>("output_dims", nvinfer1::Dims{-1, {}});
        if (outputDims.nbDims > 0)
        {
            layer->setOutputDimensions(outputDims);
        }
        else
        {
            // TRT-15340: Adapt to use resizeShapeTensor instead when safety support nbDims == 1.
            auto scales = attrs.get<std::vector<float>>("scales");
            ONNXTRT_CHECK_NODE(
                (scales.size() > 0), "Attribute scales is missing.", node, nodeIdx, ErrorCode::kINVALID_NODE);
            layer->setScales(&scales[0], scales.size());
        }
    }
    else
    {
        ONNXTRT_CHECK_NODE((inputs.at(1).is_tensor()), "The output dimension input must be a tensor.", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FloorDiv)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Conv)
{
    return importConv(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Deconv)
{
    return importConvTranspose(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxPool)
{
    return importMaxPool(ctx, node, nodeIdx, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_AveragePool)
{
    return importAveragePool(ctx, node, nodeIdx, inputs);
}


} // namespace

} // namespace onnx2trt
