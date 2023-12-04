/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "builtin_op_importers.hpp"
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
#include "half.h"
#include "onnx2trt_utils.hpp"

#include <algorithm> // For std::min, std::max
#include <array>
#include <cmath>
#include <cstring> // For std::memcpy, std::memset
#include <iostream>
#include <iterator>
#include <numeric> // For std::iota
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
    NodeImportResult import##op(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,  \
        std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op)                                                                                 \
    NodeImportResult import##op(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,  \
        std::vector<TensorOrWeights>& inputs);                                                                         \
    static bool const op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeImportResult import##op(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,  \
        std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer, node, nodeIdx)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT_NODE(layer_ptr, "Input layer is null.", node, nodeIdx, ErrorCode::kINVALID_NODE);                       \
        return {{layer_ptr->getOutput(0)}};                                                                            \
    } while (0)

#define RETURN_IDENTITY(input, node, nodeIdx)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        TensorOrWeights output = identity(ctx, input);                                                                 \
        ASSERT_NODE(output, "Failed to add an identity layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);          \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_ALL_OUTPUTS(layer, node, nodeIdx)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT_NODE(layer_ptr, "The input layer is null.", node, nodeIdx, ErrorCode::kINVALID_NODE);                   \
        std::vector<TensorOrWeights> outputs;                                                                          \
        for (int i = 0; i < layer_ptr->getNbOutputs(); ++i)                                                            \
            outputs.push_back(layer_ptr->getOutput(i));                                                                \
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

NodeImportResult batchnormFallback(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    std::vector<TensorOrWeights>& inputs)
{
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;

    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const rank = input.getDimensions().nbDims;

    nvinfer1::ITensor* scale = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* bias = &convertToTensor(inputs.at(2), ctx);
    nvinfer1::ITensor* mean = &convertToTensor(inputs.at(3), ctx);
    nvinfer1::ITensor* variance = &convertToTensor(inputs.at(4), ctx);

    // Reshape batchnorm weights from [C] to [N, C, ...]
    bool const needsExpandDims = rank > 1;
    if (needsExpandDims)
    {
        std::vector<int32_t> axes(rank - 1);
        axes[0] = 0;
        std::iota(axes.begin() + 1, axes.end(), 2);

        scale = unsqueezeTensor(ctx, node, *scale, axes);
        bias = unsqueezeTensor(ctx, node, *bias, axes);
        mean = unsqueezeTensor(ctx, node, *mean, axes);
        variance = unsqueezeTensor(ctx, node, *variance, axes);
    }

    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5F);

    nvinfer1::Dims scalarShape{rank};
    std::fill(scalarShape.d, scalarShape.d + scalarShape.nbDims, 1);

    auto varType = variance->getType();
    nvinfer1::IConstantLayer* epsLayer;
    if (varType == DataType::kHALF)
    {
        epsLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(eps), ::ONNX_NAMESPACE::TensorProto::FLOAT16, scalarShape);
    }
    else if (varType == DataType::kBF16)
    {
        epsLayer
            = addConstantScalar(ctx, static_cast<BFloat16>(eps), ::ONNX_NAMESPACE::TensorProto::BFLOAT16, scalarShape);
    }
    else
    {
        epsLayer = addConstantScalar(ctx, eps, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarShape);
    }
    nvinfer1::ITensor* epsilon = epsLayer->getOutput(0);

    // batchnorm = scale * (input - mean) / sqrt(variance + epsilon) + bias
    // The WAR is split the single c++ code line into 3 to avoid the sequence swap by compiler.
    nvinfer1::ITensor* divisor
        = ctx->network()
              ->addUnary(*ctx->network()->addElementWise(*variance, *epsilon, eOp::kSUM)->getOutput(0), uOp::kSQRT)
              ->getOutput(0);
    nvinfer1::ITensor* dividend = ctx->network()->addElementWise(input, *mean, eOp::kSUB)->getOutput(0);
    nvinfer1::IElementWiseLayer* layer = ctx->network()->addElementWise(
        *ctx->network()
             ->addElementWise(
                 *scale, *ctx->network()->addElementWise(*dividend, *divisor, eOp::kDIV)->getOutput(0), eOp::kPROD)
             ->getOutput(0),
        *bias, eOp::kSUM);

    ctx->registerLayer(layer, node);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

template <typename T>
NodeImportResult batchnormWeightHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    std::vector<TensorOrWeights>& inputs)
{
    auto const scale = inputs.at(1).weights();
    auto const bias = inputs.at(2).weights();
    auto const mean = inputs.at(3).weights();
    auto const variance = inputs.at(4).weights();

    T const* scaleValues = static_cast<T*>(scale.values);
    T const* biasValues = static_cast<T*>(bias.values);
    T const* meanValues = static_cast<T*>(mean.values);
    T const* varianceValues = static_cast<T*>(variance.values);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    OnnxAttrs attrs(node, ctx);
    T eps = static_cast<T>(attrs.get<float>("epsilon", 1e-5f));

    // Fold the weights together into a single bias and scale
    int32_t const nbChannels = scale.shape.d[0];
    ShapedWeights::DataType weightType = typeid(T).hash_code() == typeid(BFloat16).hash_code()
        ? ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        : (typeid(T).hash_code() == typeid(half_float::half).hash_code() ? ::ONNX_NAMESPACE::TensorProto::FLOAT16
                                                                         : ::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto combinedScale = ctx->createTempWeights(weightType, scale.shape);
    auto combinedBias = ctx->createTempWeights(weightType, bias.shape);
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
    ASSERT_NODE((inputs.at(1).shape().nbDims == 1), "The shape of the scale input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ASSERT_NODE((inputs.at(2).shape().nbDims == 1), "The shape of the bias input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ASSERT_NODE((inputs.at(3).shape().nbDims == 1), "The shape of the mean input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ASSERT_NODE((inputs.at(4).shape().nbDims == 1), "The shape of the var input must be (C, )", node, nodeIdx,
        ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);

    bool const allInputsWeights = inputs.at(1).is_weights() && inputs.at(2).is_weights() && inputs.at(3).is_weights()
        && inputs.at(4).is_weights();

    if (!allInputsWeights)
    {
        return batchnormFallback(ctx, node, nodeIdx, inputs);
    }

    auto tensorType = inputs.at(0).getType();
    if (tensorType == inputs.at(1).getType() && tensorType == inputs.at(2).getType()
        && tensorType == inputs.at(3).getType() && tensorType == inputs.at(4).getType())
    {
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
        ASSERT_NODE(false, "Invalid data type provided for BatchNormalization", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE_DATATYPE);
    }

    auto const scale = inputs.at(1).weights();
    auto const bias = inputs.at(2).weights();
    auto const mean = inputs.at(3).weights();
    auto const variance = inputs.at(4).weights();

    // get the values of constant inputs and cast them to float32
    float const* scaleValues = getFP32Values(scale, ctx);
    float const* biasValues = getFP32Values(bias, ctx);
    float const* meanValues = getFP32Values(mean, ctx);
    float const* varianceValues = getFP32Values(variance, ctx);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    float eps = attrs.get<float>("epsilon", 1e-5f);

    // Fold the weights together into a single bias and scale
    int32_t const nbChannels = scale.shape.d[0];
    auto combinedScale = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, scale.shape);
    auto combinedBias = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, bias.shape);
    for (int32_t i = 0; i < nbChannels; ++i)
    {
        combinedScale.at<float>(i) = scaleValues[i] / sqrtf(varianceValues[i] + eps);
        combinedBias.at<float>(i) = biasValues[i] - meanValues[i] * combinedScale.at<float>(i);
    }

    return scaleHelper(ctx, node, nodeIdx, *tensorPtr, nvinfer1::ScaleMode::kCHANNEL, combinedBias, combinedScale,
        ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT), combinedBias.getName(), combinedScale.getName());
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
    convertDtype(onnxType, &newType);

    // Add the layer.
    nvinfer1::ICastLayer* layer = ctx->network()->addCast(tensor, newType);
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
    nvinfer1::ICastLayer* layer = ctx->network()->addCast(tensor, type);
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

    ASSERT_NODE((!inputs.empty()), "Inputs vector is empty.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    TensorOrWeights input = inputs.at(0);
    float alpha = attrs.get<float>("alpha", 1.0);

    TensorOrWeights weightsOfZero = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
    ShapedWeights weightsOfOnes = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
    std::vector<float> ones{1};
    std::memcpy(weightsOfOnes.values, ones.data(), weightsOfOnes.count() * sizeof(float));
    ShapedWeights weightsOfAlpha = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
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
        ASSERT_NODE(tensor_ptr->getDimensions().nbDims == maxNbDims, "Failed to broadcast tensors elementwise!", node,
            nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }

    // Calculate (x/alpha)
    std::vector<TensorOrWeights> tempInputs{newInputs[0], newInputs[3]};
    CHECK(elementwiseCheck(tempInputs, eOp::kDIV, node, nodeIdx));
    nvinfer1::ITensor* combined = inputTensors.at(0);
    auto* divLayer = ctx->network()->addElementWise(*combined, *inputTensors.at(3), eOp::kDIV);
    ctx->registerLayer(divLayer, node);
    ASSERT_NODE(divLayer, "Failed to register layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    combined = divLayer->getOutput(0);

    // Calculate exp(x/alpha) -> 4
    nvinfer1::IUnaryLayer* uLayer = ctx->network()->addUnary(*combined, uOp::kEXP);
    ASSERT_NODE(uLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(uLayer, node);
    combined = uLayer->getOutput(0);
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
        CHECK(elementwiseCheck(tempInputs, op, node, nodeIdx));
        ASSERT_NODE((firstTensor->getDimensions().nbDims == secondTensor->getDimensions().nbDims),
            "The rank of operands should be the same adding inputs. First tensor rank is "
                << firstTensor->getDimensions().nbDims << ", but second tensor rank is "
                << secondTensor->getDimensions().nbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = ctx->network()->addElementWise(*firstTensor, *secondTensor, op);
        ctx->registerLayer(layer, node);
        ASSERT_NODE(layer, "Failed to register layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(layer->getOutput(0));
    }
    return {{inputTensors.back()}};
}

// Helper function to perform clip through elementwise operations
template <typename ScalarType>
NodeImportResult elementwiseClipHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, size_t numInputs, int32_t onnxType)
{
    OnnxAttrs attrs(node, ctx);
    auto* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* alphaT{nullptr};
    nvinfer1::ITensor* betaT{nullptr};
    ScalarType alpha = std::numeric_limits<ScalarType>::lowest();
    ScalarType beta = std::numeric_limits<ScalarType>::max();
    if (numInputs == 1)
    {
        alphaT = addConstantScalar(ctx, alpha, onnxType)->getOutput(0);
        betaT = addConstantScalar(ctx, beta, onnxType)->getOutput(0);
    }
    else if (numInputs == 2)
    {
        alphaT = &convertToTensor(inputs.at(1), ctx);
        betaT = addConstantScalar(ctx, beta, onnxType)->getOutput(0);
    }
    else if (numInputs == 3)
    {
        // "min" can be optional if "max" is specified. Check for this case here
        if (!inputs.at(1).isNullTensor())
        {
            alphaT = &convertToTensor(inputs.at(1), ctx);
        }
        else
        {
            alphaT = addConstantScalar(ctx, alpha, onnxType)->getOutput(0);
        }
        if (!inputs.at(2).isNullTensor())
        {
            betaT = &convertToTensor(inputs.at(2), ctx);
        }
        else
        {
            betaT = addConstantScalar(ctx, beta, onnxType)->getOutput(0);
        }
    }

    // Now that we have alphaT and betaT, do the elementwise calculation
    using eOp = nvinfer1::ElementWiseOperation;
    CHECK(broadcastTensors(ctx, input, alphaT));
    CHECK(broadcastTensors(ctx, input, betaT));
    auto lowerClip = ctx->network()->addElementWise(*input, *alphaT, eOp::kMAX)->getOutput(0);
    auto upperClip = ctx->network()->addElementWise(*lowerClip, *betaT, eOp::kMIN)->getOutput(0);
    return {{upperClip}};
}

DEFINE_BUILTIN_OP_IMPORTER(Clip)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
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
        ASSERT_NODE((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16
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
            ASSERT_NODE(inputs.at(1).is_weights(), "Clip min value must be an initializer!", node, nodeIdx,
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
                ASSERT_NODE(inputs.at(1).is_weights(), "Clip min value must be an initializer!", node, nodeIdx,
                    ErrorCode::kINVALID_NODE);
                auto min = inputs.at(1).weights();
                alpha = getSingleValueAsFloat(min);
            }

            if (!inputs.at(2).isNullTensor())
            {
                ASSERT_NODE(inputs.at(2).is_weights(), "Clip max value must be an initializer!", node, nodeIdx,
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

    nvinfer1::ITensor* clipOut
        = &activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kCLIP, &alpha, &beta)
               .value()
               .at(0)
               .tensor();

    return {{clipOut}};
}

DEFINE_BUILTIN_OP_IMPORTER(Concat)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        auto* tensorPtr = &convertToTensor(input, ctx);
        tensors.push_back(tensorPtr);
    }
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis");
    int32_t nbDims = inputs.at(0).shape().nbDims;
    CHECK(convertAxis(axis, nbDims));
    auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    ctx->registerLayer(layer, node);
    ASSERT_NODE(layer, "Failed to register layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        auto* layer = ctx->network()->addConstant(weights.shape, weights);
        ctx->network()->setWeightsName(weights, weights.getName());
        RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
    }

    if (ctx->getOpsetVersion() >= 12)
    {
        if (attrs.count("value_float"))
        {
            ShapedWeights convertedWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {}});
            float value = attrs.get<float>("value_float");
            std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(float));
            return {{convertedWeights}};
        }

        if (attrs.count("value_floats"))
        {
            std::vector<float> values = attrs.get<std::vector<float>>("value_floats");
            int32_t valueSize = values.size();
            ShapedWeights convertedWeights
                = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {1, {valueSize}});
            std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(float));
            return {{convertedWeights}};
        }
        if (attrs.count("value_int"))
        {
            ShapedWeights convertedWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {0, {}});
            int64_t value = attrs.get<int64_t>("value_int");
            std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(int64_t));
            return {{convertedWeights}};
        }

        if (attrs.count("value_ints"))
        {
            std::vector<int64_t> values = attrs.get<std::vector<int64_t>>("value_ints");
            int32_t valueSize = values.size();
            ShapedWeights convertedWeights
                = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT64, {1, {valueSize}});
            std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(int64_t));
            return {{convertedWeights}};
        }
    }

    return {{attrs.get<ShapedWeights>("value")}};
}

DEFINE_BUILTIN_OP_IMPORTER(ConstantOfShape)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* shape = &convertToTensor(inputs.at(0), ctx);

    ShapedWeights zeroWeights
        = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, 1});
    static_cast<float*>(zeroWeights.values)[0] = 0.f;
    auto valueWeights = TensorOrWeights{attrs.get("value", zeroWeights)};
    nvinfer1::ITensor* value = &convertToTensor(valueWeights, ctx);
    return {{constantOfShape(ctx, node, value, shape)}};
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
    ASSERT_NODE(dims.nbDims >= 0, "TensorRT could not compute output dimensions of Conv", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    bool const needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT_NODE(tensorPtr, "Failed to unsqueeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }
    if (kernelWeights.shape.nbDims == 3)
    {
        kernelWeights.shape.nbDims = 4;
        kernelWeights.shape.d[3] = 1;
    }

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT_NODE((nbSpatialDims == kernelWeights.shape.nbDims - 2),
        "The number of spatial dimensions and the kernel shape doesn't match up for the Conv operator. Number of "
        "spatial dimensions = "
            << nbSpatialDims << ", number of kernel dimensions = " << kernelWeights.shape.nbDims << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        assertIsWeights(inputs.at(2), "The bias tensor is required to be an initializer for the Conv operator.");
        auto shapedBiasWeights = inputs.at(2).weights();
        // Unsqueeze scalar weights to 1D
        if (shapedBiasWeights.shape.nbDims == 0)
        {
            shapedBiasWeights.shape = {1, {1}};
        }
        ASSERT_NODE((shapedBiasWeights.shape.nbDims == 1), "The bias tensor is required to be 1D.", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        ASSERT_NODE((shapedBiasWeights.shape.d[0] == kernelWeights.shape.d[0]),
            "The shape of the bias tensor misaligns with the weight tensor. Shape of bias weights = "
                << shapedBiasWeights.shape.d[0] << ", shape of kernel weights = " << kernelWeights.shape.d[0] << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        bias_weights = shapedBiasWeights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernelWeights.type);
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
    bool exclude_padding;
    getKernelParams(
        ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, exclude_padding, &dilations);

    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT_NODE((kernelSize.d[nbSpatialDims - i] == kernelWeights.shape.d[kernelWeights.shape.nbDims - i]),
            "The size of spatial dimension and the size of kernel shape are not equal for the Conv operator. "
            "Size of spatial dimensions = "
                << kernelSize.d[nbSpatialDims - i]
                << ", size of kernel dimensions = " << kernelWeights.shape.d[kernelWeights.shape.nbDims - i] << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    int32_t nchan = dims.d[1];
    int32_t noutput = kernelWeights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*tensorPtr, noutput, kernelSize, kernelWeights, bias_weights);

    ASSERT_NODE(layer, "Failed to add a convolution layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int32_t ngroup = attrs.get("group", 1);
    ASSERT_NODE((nchan == -1 || kernelWeights.shape.d[1] * ngroup == nchan),
        "Kernel weight dimension failed to broadcast to input.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    // Register layer name as well as kernel weights and bias weights (if any)
    ctx->registerLayer(layer, node);
    ctx->network()->setWeightsName(kernelWeights, inputs.at(1).weights().getName());
    if (inputs.size() == 3)
    {
        ctx->network()->setWeightsName(bias_weights, inputs.at(2).weights().getName());
    }
    tensorPtr = layer->getOutput(0);
    dims = tensorPtr->getDimensions();

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT_NODE(tensorPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    LOG_VERBOSE("Using kernel: " << kernelSize << ", strides: " << strides << ", prepadding: " << begPadding
                                 << ", postpadding: " << endPadding << ", dilations: " << dilations
                                 << ", numOutputs: " << noutput);
    LOG_VERBOSE("Convolution output dimensions: " << dims);

    return {{tensorPtr}};
}

// TRT only supports 2D or 3D deconvolutions (Layout: [N,C,D1,D2,(D3)])
// Inputs should be of dimension 4 or 5.
// When input.nbDims = 3, we expand it to 4D
DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose)
{
    // Expand spatial dims from 1D to 2D, return true if reshaped activation
    auto const NCWtoNCHW = [&ctx, &node](nvinfer1::ITensor*& tensor, nvinfer1::Dims& tensorShape) {
        if (tensor && tensor->getDimensions().nbDims == 3)
        {
            std::vector<int32_t> const axes{3};
            tensor = unsqueezeTensor(ctx, node, *tensor, axes);
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

    ASSERT_NODE(
        inputs.size() >= 2, "deconvolution require at least 2 inputs.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    auto inputType = tensorPtr->getType();
    nvinfer1::ITensor* kernelTensorPtr = inputs.at(1).is_tensor() ? &convertToTensor(inputs.at(1), ctx) : nullptr;
    nvinfer1::ITensor* biasTensorPtr
        = inputs.size() > 2 && inputs.at(2).is_tensor() ? &convertToTensor(inputs.at(2), ctx) : nullptr;

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    // Deconvolution input must be at least 3D and at most 5D.
    ASSERT_NODE(dims.nbDims >= 3 && dims.nbDims <= 5,
        "Deconvolution input must be at least 3D and at most 5D! The current input is rank " << dims.nbDims
                                                                                                   << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernelShape = inputs.at(1).shape();

    bool needReshapeBack = NCWtoNCHW(tensorPtr, dims);
    NCWtoNCHW(kernelTensorPtr, kernelShape);

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT_NODE((nbSpatialDims == kernelShape.nbDims - 2),
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
    bool exclude_padding = false;

    int32_t ngroup = attrs.get("group", 1);
    int32_t noutput = kernelShape.d[1] * ngroup; // Note: Weights order is CKRS

    // Get static bias weights
    nvinfer1::Weights staticBiasWeights;
    if (inputs.size() > 2 && biasTensorPtr == nullptr)
    {
        auto shapedBiasWeights = inputs.at(2).weights();
        // ONNX requires shapedBiasWeights to be 1D
        ASSERT_NODE(shapedBiasWeights.shape.nbDims == 1,
            "The bias tensor is required to be 1D. Provided bias has rank "
                << shapedBiasWeights.shape.nbDims << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        ASSERT_NODE((shapedBiasWeights.shape.d[0] == noutput),
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

    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, exclude_padding,
        &dilations, &outputPadding);

    for (int32_t i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT_NODE((kernelSize.d[nbSpatialDims - i] == kernelShape.d[kernelShape.nbDims - i]),
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
            ASSERT_NODE(begPadding.d[i] >= 0,
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
    const auto biasWeights = biasTensorPtr ? nvinfer1::Weights{inputType, nullptr, 0} : staticBiasWeights;
    // Create a deconvolution layer and set known attributes - strides,ngroups, and dilations
    // If there is still output padding, remove the bias weights. Bias will be added below.
    auto* layer = ctx->network()->addDeconvolutionNd(
        *tensorPtr, noutput, kernelSize, kernelWeights, hasOutputPadding ? emptyBiasWeights : biasWeights);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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

    LOG_VERBOSE("Running deconvolution with: "
        << "\n"
        << "Padding mode: " << autoPadMode << "\n"
        << "Pre-padding: " << begPadding << "\n"
        << "Post-padding: " << endPadding);

    // Register layer, along with refittable kernel weights and bias weights (if any)
    ctx->registerLayer(layer, node);
    tensorPtr = layer->getOutput(0);
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
        ASSERT_NODE(convertOnnxPadding(ctx, dims.nbDims, combinePadding, start, totalPadding),
            "Failed to convert padding!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        auto const size
            = ctx->network()
                  ->addElementWise(shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM)
                  ->getOutput(0);
        auto const stride = makeDims(dims.nbDims, 1);
        auto const& dummy = stride;
        auto* sliceLayer = ctx->network()->addSlice(*tensorPtr, dummy, dummy, stride);
        ASSERT_NODE(sliceLayer, "Could not create slice layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        sliceLayer->setInput(1, *start);
        sliceLayer->setInput(2, *size);
        sliceLayer->setMode(nvinfer1::SliceMode::kFILL);
        tensorPtr = sliceLayer->getOutput(0);

        // This bias is not handled by deconv. Use an elementwise to handle it.
        if (biasWeights.count != 0)
        {
            // Set C dimension to weights count and set other dimensions to 1 to enable broadcast
            auto constantDims = makeDims(dims.nbDims, 1);
            constantDims.d[dims.nbDims - nbSpatialDims - 1] = biasWeights.count;
            auto biasConstant = ctx->network()->addConstant(constantDims, biasWeights);
            tensorPtr
                = ctx->network()
                      ->addElementWise(*tensorPtr, *biasConstant->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)
                      ->getOutput(0);
        }
    }

    if (inputs.size() > 2 && biasTensorPtr == nullptr)
    {
        ctx->network()->setWeightsName(biasWeights, inputs.at(2).weights().getName());
    }

    if (needReshapeBack)
    {
        std::vector<int32_t> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT_NODE(tensorPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
    auto dims = input->getDimensions();

    assertIsWeights(inputs.at(1), "Axis input for CumSum must be an initializer!");
    ShapedWeights axisWeights = inputs.at(1).weights();
    int32_t axis = static_cast<int32_t*>(axisWeights.values)[0];
    CHECK(convertAxis(axis, dims.nbDims));

    // Create "inputSliced" tensor that is sliced on dimension[axis] to length 1
    auto inputSliced = sliceAcrossAxis(ctx, node, input, axis);

    /* For exclusive CumSums, it is equivalent as a non-exclusive CumSum on a modified input tensor

        Forward summations:
            concat(0, data[0:length-1:1])

        Reverse summations:
            concat(data[1:length:1], 0)

    */
    if (exclusive)
    {
        auto zero = createZeroTensor(ctx, inputSliced);
        std::vector<nvinfer1::ITensor*> concatTensors = reverse == 1 ? std::vector<nvinfer1::ITensor*>{input, zero}
                                                                     : std::vector<nvinfer1::ITensor*>{zero, input};

        auto concat = ctx->network()->addConcatenation(concatTensors.data(), concatTensors.size());
        concat->setAxis(axis);
        input = concat->getOutput(0);

        if (reverse == 0)
        {
            ShapeTensor const subscripts{axesToInterlaceSubscripts(shapeVector(axis), dims.nbDims)};
            ShapeTensor starts = fillShapeVector(ctx, 0, shapeVector(dims.nbDims));
            ShapeTensor sizes = interlace(ctx, shapeOf(*input),
                sub(ctx, gather(ctx, shapeOf(*input), shapeVector(axis)), shapeVector(1)), subscripts);
            ShapeTensor strides = fillShapeVector(ctx, 1, shapeVector(dims.nbDims));
            input = addSlice(ctx, *input, starts, sizes, strides)->getOutput(0);
        }
        else
        {
            ShapeTensor const subscripts{axesToInterlaceSubscripts(shapeVector(axis), dims.nbDims)};
            ShapeTensor starts
                = interlace(ctx, fillShapeVector(ctx, 0, shapeVector(dims.nbDims)), shapeVector(1), subscripts);
            ShapeTensor sizes = interlace(ctx, shapeOf(*input),
                sub(ctx, gather(ctx, shapeOf(*input), shapeVector(axis)), shapeVector(1)), subscripts);
            ShapeTensor strides = fillShapeVector(ctx, 1, shapeVector(dims.nbDims));
            input = addSlice(ctx, *input, starts, sizes, strides)->getOutput(0);
        }
    }

    // Scan through each slice across summation axis and add it to the running sum
    auto loop = ctx->network()->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, axis);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
    auto iterator = loop->addIterator(*input, axis, reverse);
    auto data = iterator->getOutput(0);

    // Squeeze inputSliced down to same shape as `data`
    inputSliced = squeezeTensor(ctx, node, *inputSliced, {axis});
    auto zeroTensor = createZeroTensor(ctx, inputSliced);
    auto runningSum = loop->addRecurrence(*zeroTensor);
    auto runningSumTensor = runningSum->getOutput(0);

    auto curSum = ctx->network()->addElementWise(*data, *runningSumTensor, nvinfer1::ElementWiseOperation::kSUM);
    runningSum->setInput(1, *curSum->getOutput(0));

    auto reverseFlag = reverse == 1 ? nvinfer1::LoopOutput::kREVERSE : nvinfer1::LoopOutput::kCONCATENATE;
    nvinfer1::ILoopOutputLayer* loopOut = loop->addLoopOutput(*curSum->getOutput(0), reverseFlag, axis);
    loopOut->setInput(1, *tripLimit);

    RETURN_FIRST_OUTPUT(loopOut, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    // Input tensor is in NCHW format
    ASSERT_NODE((inputs.at(0).shape().nbDims == 4), "The input tensor must be in NCHW format.", node, nodeIdx,
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
    tensorPtr = firstShuffle->getOutput(0);

    // Finally reshape to {N, C / (blockSize * blockSize), H * blockSize, W * blockSize};
    auto secondShape = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShape);
    tensorPtr = secondShuffle->getOutput(0);

    return {{tensorPtr}};
}

// This is a helper function for QuantizeLinear/DequantizeLinear
NodeImportResult QuantDequantLinearHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    size_t nodeIdx, std::vector<TensorOrWeights>& inputs, bool isDQ, bool isCustomOpFp8)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));

    // For QuantizeLinear, the output type (and thus quantization type) is dependent on the second input (zero point).
    if (!isDQ && inputs.size() >= 3)
    {
        CHECK(notInvalidType(inputs.at(2), {"UINT8"}));
    }
    auto addConstantLayer
        = [ctx, node](nvinfer1::INetworkDefinition& network, ShapedWeights const& weights) -> nvinfer1::ITensor* {
        nvinfer1::IConstantLayer* constLayer = network.addConstant(weights.shape, weights);
        ctx->registerLayer(constLayer, weights.getName(), &node);
        network.setWeightsName(weights, weights.getName());
        return constLayer->getOutput(0);
    };

    auto newConstantInput = [&](int32_t i) {
        return inputs.at(i).is_weights() && (ctx->getConstantLayer(inputs.at(i).weights().getName()) == nullptr);
    };

    // Read the optional quantization axis attribute. Set it to the rank of the input tensor if not provided
    ASSERT_NODE((inputs.size() >= 2),
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
        ASSERT_NODE(
            scale.count() > 0, "Cannot have scale with no coefficients.", node, nodeIdx, ErrorCode::kINVALID_NODE);

        bool scaleAllPositive = false;
        if (inputs.at(1).isFp32())
        {
            auto const* scaleVal = static_cast<float const*>(scale.values);
            scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](float x) { return x > 0; });
        }
        else if (inputs.at(1).isFp16())
        {
            auto const* scaleVal = static_cast<half_float::half const*>(scale.values);
            scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](half_float::half x) { return x > 0; });
        }
        else if (inputs.at(1).isBFp16())
        {
            auto const* scaleVal = static_cast<BFloat16 const*>(scale.values);
            scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](BFloat16 x) { return x > 0; });
        }
        ASSERT_NODE(
            scaleAllPositive, "Scale coefficients must all be positive", node, nodeIdx, ErrorCode::kINVALID_NODE);

        // If the scale is concrete weights, then add a ConstantLayer that will be an input which
        // will initialize the scale weights.
        scaleInput = addConstantLayer(*ctx->network(), scale);
    }
    else
    {
        scaleInput = &convertToTensor(inputs.at(1), ctx);
    }
    const auto scaleSize = volume(scaleInput->getDimensions());
    const auto scaleType = scaleInput->getType();

    // Input 2 initializes the layer's zero-point.
    nvinfer1::ITensor* zeroPointInput = nullptr;
    // ONNX default is UINT8, TRT will default to INT8 as TRT doesn't allow UINT8 quantization
    // When importing CustomOp Q/DQ, default to FP8
    DataType chosenDataType = isCustomOpFp8 ? DataType::kFP8 : DataType::kINT8;
    if (inputs.size() > 2)
    {
        // ONNX spec definition is that when zero point is set, use its datatype for quantization
        DataType zeroPointDataType = inputs.at(2).getDataType();
        if (zeroPointDataType == DataType::kFP8 || zeroPointDataType == DataType::kINT8)
        {
            chosenDataType = zeroPointDataType;
        }
        else
        {
            // If zero point is set to UINT8, default to INT8.
            LOG_WARNING("TensorRT doesn't support QuantizeLinear/DequantizeLinear with UINT8 zero_point. TensorRT will use INT8 instead.");
            chosenDataType = DataType::kINT8;
        }

        if (chosenDataType != DataType::kFP8)
        {
            if (newConstantInput(2))
            {
                //TRT-20277: after enabling ConstantInt8Validator to support Int8+Q, this constraint can be changed to allow Int8 zeropoint
                //Right now, zero point is allowed to be only 0, hence there is no real usage of it, so no functionality change is expected
                if (chosenDataType == DataType::kINT8 && isDQ)
                {
                    // Zero-point verification.
                    auto zeroPoint = inputs.at(2).weights();
                    ASSERT_NODE(shiftIsAllZeros(zeroPoint),
                        "TensorRT only supports symmetric quantization. The zero point for the "
                        "QuantizeLinear/DequantizeLinear operator must be all zeros.",
                        node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
                    // Convert the zero-point to float because TRT uses float for zero-point.
                    auto fpZeroPoint = createZeroShifts(zeroPoint, ::ONNX_NAMESPACE::TensorProto::FLOAT, ctx);
                    fpZeroPoint.setName(zeroPoint.getName());
                    zeroPointInput = addConstantLayer(*ctx->network(), fpZeroPoint);
                }
            }
            else
            {
                zeroPointInput = &convertToTensor(inputs.at(2), ctx);
            }

            if (zeroPointInput)
            {
                auto const zeroPointSize = volume(zeroPointInput->getDimensions());
                // ONNX may represent a scalar using either 0-D or 1-D, so compare sizes instead of shapes.
                ASSERT_NODE(zeroPointSize == scaleSize,
                    "The scale and zero point must have the same volume. Size of zero point = "
                        << zeroPointSize << ", size of the scale = " << scaleSize << ".",
                    node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
            }
        }
    }

    OnnxAttrs attrs(node, ctx);
    int32_t const nbDims = dataInput->getDimensions().nbDims;
    int32_t axis = attrs.get<int32_t>("axis", nbDims);
    CHECK(convertAxis(axis, nbDims));

    if (scaleSize != 1)
    {
        // Per-Channel Quantization.
        // We assume this is weight-quantization with dimensions KCRS (K is # output channels).
        // Activations-quantization does not support per-axis quantization.
        if (axis == nbDims)
        {
            axis = 0;
        }
        // Ensure that number of scale-coefficients is equal to the number of output channels.
        int64_t const K = dataInput->getDimensions().d[axis];
        ASSERT_NODE(K == scaleSize,
            "The number of scales is not equal to the number of output channels. Number of output channels = "
                << K << ", number of scales = " << scaleSize << ".",
            node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
    }
    else
    {
        // Per-Tensor Quantization.
        // Currently axis is ignored by TRT, but it is required here by addScaleNd (for computing nbSpatialDims). Set to
        // a sane default depending on rank the input tensor.
        axis = nbDims <= 1 ? 0 : 1;
    }

    // TRT does not support scalar data input for Q/DQ layers, convert 0-D tensor to 1-D first.
    if (nbDims == 0)
    {
        dataInput = reshapeTensor(ctx, *dataInput, nvinfer1::Dims{1, {1}});
    }

    nvinfer1::ILayer* layer = nullptr;
    ASSERT_NODE((chosenDataType == DataType::kINT8 || chosenDataType == DataType::kFP8),
        "TensorRT allows only FP8 or INT8 quantization. The requested quantization type is" + getTrtDtypeName(chosenDataType)
            + ".",
        node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);

    bool stronglyTyped = ctx->network()->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    if (isDQ)
    {
        // Add and configure a DequantizeLayer.
        if (stronglyTyped)
        {
            // Input type is inferred. Layer output type is specified with scaleType.
            nvinfer1::IDequantizeLayer* dq = ctx->network()->addDequantize(*dataInput, *scaleInput, scaleType);
            ASSERT_NODE(dq, "Failed to create Dequantize layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            dq->setAxis(axis);
            layer = dq;
        }
        else
        {
            // Use legacy API for weakly typed network.
            nvinfer1::IDequantizeLayer* dq = ctx->network()->addDequantize(*dataInput, *scaleInput);
            ASSERT_NODE(dq, "Failed to create Dequantize layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            dq->setAxis(axis);
            layer = dq;
            // This implictly sets layer input type.
            layer->setPrecision(chosenDataType);
            // Type constraint for layer output type.
            layer->setOutputType(0, scaleType);
        }
    }
    else
    {
        // Add and configure a QuantizeLayer.
        if (stronglyTyped)
        {
            if (ctx->getOpsetVersion() < 19 && scaleInput->getType() != dataInput->getType())
            {
                // Ensure that Q scale type matches input type.
                scaleInput = ctx->network()->addCast(*scaleInput, dataInput->getType())->getOutput(0);
            }
            // Input type is inferred. Layer output type is specified with chosenDataType.
            nvinfer1::IQuantizeLayer* q = ctx->network()->addQuantize(*dataInput, *scaleInput, chosenDataType);
            ASSERT_NODE(q, "Failed to create Quantize layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            q->setAxis(axis);
            layer = q;
        }
        else
        {
            // Use legacy API for weakly typed network.
            nvinfer1::IQuantizeLayer* q = ctx->network()->addQuantize(*dataInput, *scaleInput);
            ASSERT_NODE(q, "Failed to create Quantize layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            q->setAxis(axis);
            layer = q;
            // This implictly sets layer input type.
            layer->setPrecision(scaleType);
            // Type constraint for layer output type.
            layer->setOutputType(0, chosenDataType);
        }
    }

    layer->setName(nodeName.c_str());
    if (zeroPointInput)
    {
        layer->setInput(2, *zeroPointInput);
    }

    // Return layer output
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(QuantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, false /*isDQ*/, false /*isCustomOpFp8*/);
}

DEFINE_BUILTIN_OP_IMPORTER(DequantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, true /*isDQ*/, false /*isCustomOpFp8*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FP8QuantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, false /*isDQ*/, true /*isCustomOpFp8*/);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FP8DequantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, nodeIdx, inputs, true /*isDQ*/, true /*isCustomOpFp8*/);
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
        ASSERT_NODE(inputs.at(2).is_weights(),
            "This Version of TensorRT only supports the training_mode input as an initializer.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
        std::vector<int64_t> trainingMode;
        weightsToVector<int64_t>(inputs.at(2).weights(), &trainingMode);
        ASSERT_NODE(!trainingMode[0], "TensorRT does not support the Dropout operator in training mode.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
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
            maskTensor = ctx->network()
                             ->addElementWise(inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kEQUAL)
                             ->getOutput(0);
        }
        else
        {
            maskTensor = ctx->network()
                             ->addElementWise(inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kLESS)
                             ->getOutput(0);
        }
        outputs.push_back(TensorOrWeights(maskTensor));
        return outputs;
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Einsum)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    OnnxAttrs attrs(node, ctx);
    std::string equation = attrs.get<std::string>("equation");

    ASSERT_NODE((!inputs.empty()), "Inputs vector is empty.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    std::vector<nvinfer1::ITensor*> inputTensors;

    for (auto input : inputs)
    {
        auto* tensor_ptr = &convertToTensor(input, ctx);
        inputTensors.push_back(tensor_ptr);
    }
    auto nbInputs = static_cast<int64_t>(inputTensors.size());

    bool withEllipsis{false};
    if (equation.find("...") != std::string::npos)
    {
        withEllipsis = true;
    }

    if (withEllipsis || nbInputs > 2)
    {
        LOG_VERBOSE("Equation before preprocessing ellipsis and output: " << equation);
        processEllipsisAndImplicitOutput(inputTensors, equation, withEllipsis);
        LOG_VERBOSE("Equation after preprocessing ellipsis and output: " << equation);
    }

    nvinfer1::IEinsumLayer* einsumLayer{nullptr};
    if (nbInputs > 2)
    {
        einsumLayer = parseGraphWithMoreInputs(ctx, node, inputTensors, nbInputs, equation);
    }
    else
    {
        einsumLayer = ctx->network()->addEinsum(inputTensors.data(), nbInputs, equation.c_str());
        ctx->registerLayer(einsumLayer, node);
    }

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
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // "Broadcast the input tensor following the given shape and the broadcast rule."
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(inputTensor);
    auto const inputRank = shapeOf(inputDims);

    // "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
    ASSERT_NODE((inputs.at(1).shape().nbDims == 1), "The shape tensor is required to be 1D.", node, nodeIdx,
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
    // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
    ShapeTensor const one = shapeVector(1);
    ShapeTensor const strides = min(ctx, one, sub(ctx, newDims, one));

    nvinfer1::ISliceLayer* sliceLayer = addSlice(ctx, newInputTensor, starts, sizes, strides);
    ctx->registerLayer(sliceLayer, node);

    RETURN_FIRST_OUTPUT(sliceLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(EyeLike)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t k = attrs.get("k", 0);

    // "Only 2D tensors are supported, i.e. input T1 must be of rank 2..."
    nvinfer1::Dims dims = tensor.getDimensions();
    ASSERT_NODE(dims.nbDims == 2, "Only 2D tensors are supported. Input must be of rank 2.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // The data type can be specified by the 'dtype' argument
    DataType dtype = tensor.getType();
    if (attrs.count("dtype"))
    {
        auto onnxType = attrs.get<int32_t>("dtype");
        ASSERT_NODE(convertDtype(onnxType, &dtype), "Unsupported cast!", node, nodeIdx, ErrorCode::kINVALID_NODE);
        LOG_VERBOSE("Casting to type: " << dtype);
    }

    // Create weights and constant layer
    ASSERT_NODE(!isDynamic(dims), "Eyelike does not work for dynamically shaped tensors.", node, nodeIdx,
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

    ShapedWeights tempWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, dims);
    std::memcpy(tempWeights.values, values.data(), values.size() * sizeof(int));
    auto* layer = ctx->network()->addConstant(dims, tempWeights);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);

    if (dtype != DataType::kINT32)
    {
        return {{castHelper(ctx, layer->getOutput(0), dtype)}};
    }
    return {{layer->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    int32_t axis = attrs.get("axis", 1);
    CHECK(convertAxis(axis, nbDims));

    // No-op Flatten: (a, b) => Flatten(axis = 1) => (a, b)
    // Add identity layer to avoid name mangling of engine bindings
    // For rest of configurations, we must flatten.
    if (nbDims == 2 && axis == 1)
    {
        RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
    }

    tensorPtr = flattenTensor(ctx, node, *tensorPtr, axis, true);
    ASSERT_NODE(tensorPtr, "Failed to flatten the tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    CHECK(convertAxis(axis, nbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);

    if (inputs.at(0).getType() != "INT64")
    {
        // Int64 indices only supported for Int64 data
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    auto* layer = ctx->network()->addGather(data, *indices, axis);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherElements)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::Dims const& dataDims = data.getDimensions();

    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t const dataNbDims = dataDims.nbDims;
    CHECK(convertAxis(axis, dataNbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);

    if (inputs.at(0).getType() != "INT64")
    {
        // Int64 indices only supported for Int64 data
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    auto* layer = ctx->network()->addGatherV2(data, *indices, nvinfer1::GatherMode::kELEMENT);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setGatherAxis(axis);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherND)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
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
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setNbElementWiseDims(nbElementWiseDims);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "INT64"}));
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& inputB = convertToTensor(inputs.at(1), ctx);
    // Validate inputs
    ASSERT_NODE(inputA.getDimensions().nbDims == 2 && inputB.getDimensions().nbDims == 2,
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

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(inputA, opA, inputB, opB);
    ctx->registerLayer(matmul, node);

    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        CHECK(broadcastTensors(ctx, alphaConstantTensor, matmulTensor));
        nvinfer1::IElementWiseLayer* scaledMatmul = ctx->network()->addElementWise(
            *alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        matmulTensor = scaledMatmul->getOutput(0);
    }

    // In opset 11, the bias tensor is an optional input
    if (inputs.size() > 2)
    {
        nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);

        // Scale C if needed
        if (beta != 1.f)
        {
            nvinfer1::IConstantLayer* betaConstant
                = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
            CHECK(broadcastTensors(ctx, betaConstantTensor, biasTensor));
            nvinfer1::IElementWiseLayer* scaledBias = ctx->network()->addElementWise(
                *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
            biasTensor = scaledBias->getOutput(0);
        }
        CHECK(broadcastTensors(ctx, matmulTensor, biasTensor));
        nvinfer1::IElementWiseLayer* biasAdd
            = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
        return {{biasAdd->getOutput(0)}};
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
    ASSERT_NODE((inputType == DataType::kFLOAT || inputType == DataType::kHALF),
        "Only FLOAT and HALF are supported in GlobalLpPool. The current type = " + getTrtDtypeName(inputType) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims dims = tensor.getDimensions();

    OnnxAttrs attrs{node, ctx};
    float p = static_cast<float>(attrs.get("p", 2));

    // Add constants for p and 1/p
    nvinfer1::Dims scalarDims{dims.nbDims};
    std::fill(scalarDims.d, scalarDims.d + scalarDims.nbDims, 1);
    nvinfer1::IConstantLayer* pLayer;
    nvinfer1::IConstantLayer* pInvLayer;
    if (inputType == DataType::kHALF)
    {
        pLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(p), ::ONNX_NAMESPACE::TensorProto::FLOAT16, scalarDims);
        pInvLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(1.F / p), ::ONNX_NAMESPACE::TensorProto::FLOAT16, scalarDims);
    }
    else
    {
        pLayer = addConstantScalar(ctx, p, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims);
        pInvLayer = addConstantScalar(ctx, 1.F / p, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims);
    }

    // firstPow = pow(x, p)
    auto* firstPow = ctx->network()
                         ->addElementWise(tensor, *pLayer->getOutput(0), nvinfer1::ElementWiseOperation::kPOW)
                         ->getOutput(0);
    // reduced = reduce_sum(firstPow)
    auto* reduced = globalPoolingHelper(ctx, node, *firstPow, nvinfer1::ReduceOperation::kSUM);
    // finalPow = pow(reduced, 1./p)
    auto* finalPow = ctx->network()
                         ->addElementWise(*reduced, *pInvLayer->getOutput(0), nvinfer1::ElementWiseOperation::kPOW)
                         ->getOutput(0);
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
    return {{greaterLessOrEqual(ctx, node, nodeIdx, &convertToTensor(inputs.at(0), ctx),
        &convertToTensor(inputs.at(1), ctx),
        /*greater*/ true)}};
}

DEFINE_BUILTIN_OP_IMPORTER(GroupNormalization)
{
    return normalizationHelper(ctx, node, nodeIdx, inputs);
}

// singlePassShape is the shape of the output from a single pass.
nvinfer1::ITensor* concatenateRNNOutputs(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ILoop* loop, nvinfer1::ITensor* singlePassShape, nvinfer1::ITensor* sequenceLength,
    nvinfer1::ITensor* concatenatedOutput, int numDirections, std::vector<TensorOrWeights>& inputs,
    bool reverse = false)
{
    nvinfer1::ITensor* yOutput{nullptr};
    if (numDirections == 2)
    {
        nvinfer1::ITensor* forwardStart = addConstant(
            ctx, std::vector<int32_t>{0, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
                                              ->getOutput(0);
        nvinfer1::ITensor* reverseStart = addConstant(
            ctx, std::vector<int32_t>{1, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
                                              ->getOutput(0);

        LOG_VERBOSE("Concatenated output shape: " << concatenatedOutput->getDimensions());
        nvinfer1::ISliceLayer* HtForwardLayer = ctx->network()->addSlice(
            *concatenatedOutput, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
        LOG_VERBOSE("Forward pass shape: " << HtForwardLayer->getOutput(0)->getDimensions());
        HtForwardLayer->setInput(1, *forwardStart);
        HtForwardLayer->setInput(2, *singlePassShape);

        nvinfer1::ISliceLayer* HtBackwardLayer = ctx->network()->addSlice(
            *concatenatedOutput, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
        LOG_VERBOSE("Reverse pass shape: " << HtBackwardLayer->getOutput(0)->getDimensions());
        HtBackwardLayer->setInput(1, *reverseStart);
        HtBackwardLayer->setInput(2, *singlePassShape);

        auto forwardHt = HtForwardLayer->getOutput(0);
        auto backwardHt = HtBackwardLayer->getOutput(0);
        if (inputs.size() > 4 && inputs.at(4))
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
            forwardHt = clearMissingSequenceElements(ctx, node, loop, seqLens, forwardHt, sequenceLength);
            backwardHt
                = clearMissingSequenceElements(ctx, node, loop, seqLens, backwardHt, sequenceLength, /*reverse=*/true);
        }

        nvinfer1::ILoopOutputLayer* forwardOutput
            = loop->addLoopOutput(*forwardHt, nvinfer1::LoopOutput::kCONCATENATE, 0);
        forwardOutput->setInput(1, *sequenceLength);
        nvinfer1::ILoopOutputLayer* reverseOutput = loop->addLoopOutput(*backwardHt, nvinfer1::LoopOutput::kREVERSE, 0);
        reverseOutput->setInput(1, *sequenceLength);

        std::array<nvinfer1::ITensor*, 2> passes{{forwardOutput->getOutput(0), reverseOutput->getOutput(0)}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(passes.data(), passes.size());
        concat->setAxis(1);
        yOutput = concat->getOutput(0);
    }
    else
    {
        if (inputs.size() > 4 && inputs.at(4))
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
            concatenatedOutput
                = clearMissingSequenceElements(ctx, node, loop, seqLens, concatenatedOutput, sequenceLength, reverse);
        }
        nvinfer1::ILoopOutputLayer* scanOut = loop->addLoopOutput(
            *concatenatedOutput, (reverse ? nvinfer1::LoopOutput::kREVERSE : nvinfer1::LoopOutput::kCONCATENATE), 0);
        scanOut->setInput(1, *sequenceLength);
        yOutput = scanOut->getOutput(0);
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
    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

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
        = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* hiddenSizeTensor
        = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* hiddenSizeDoubledTensor
        = addConstantScalar(ctx, 2 * hiddenSize, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 1})->getOutput(0);
    nvinfer1::ITensor* eDimTensor = getAxisLength(ctx, input, 2, Dims{1, 1});

    nvinfer1::ITensor* weightsZRStart
        = addConstant(ctx, std::vector<int32_t>{0, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 3})
              ->getOutput(0);
    nvinfer1::ITensor* weightsZRSize
        = net->addConcatenation(
                 std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, hiddenSizeDoubledTensor, eDimTensor}}.data(),
                 3)
              ->getOutput(0);
    nvinfer1::ISliceLayer* weightsZRLayer = net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1});
    ASSERT_NODE(weightsZRLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    weightsZRLayer->setInput(1, *weightsZRStart);
    weightsZRLayer->setInput(2, *weightsZRSize);
    nvinfer1::ITensor* weightsZR = weightsZRLayer->getOutput(0);
    LOG_VERBOSE("Weights for ZR gates shape is: " << weightsZR->getDimensions());

    nvinfer1::ITensor* weightsHStart
        = addConstant(ctx, std::vector<int32_t>{0, 2 * hiddenSize, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, Dims{1, 3})
              ->getOutput(0);
    nvinfer1::ITensor* weightsHSize
        = net->addConcatenation(
                 std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, hiddenSizeTensor, eDimTensor}}.data(), 3)
              ->getOutput(0);
    nvinfer1::ISliceLayer* weightsHLayer = net->addSlice(weights, Dims{3}, Dims{3}, Dims3{1, 1, 1});
    ASSERT_NODE(weightsHLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    weightsHLayer->setInput(1, *weightsHStart);
    weightsHLayer->setInput(2, *weightsHSize);
    nvinfer1::ITensor* weightsH = weightsHLayer->getOutput(0);
    LOG_VERBOSE("Weights for H gate shape is: " << weightsH->getDimensions());

    nvinfer1::ITensor* recurrenceWeightsZR = net->addSlice(recurrenceWeights, Dims3{0, 0, 0},
                                                    Dims3{numDirections, 2 * hiddenSize, hiddenSize}, Dims3{1, 1, 1})
                                                 ->getOutput(0);
    LOG_VERBOSE("Recurrence weights for ZR gates shape is: " << recurrenceWeightsZR->getDimensions());
    nvinfer1::ITensor* recurrenceWeightsH = net->addSlice(recurrenceWeights, Dims3{0, 2 * hiddenSize, 0},
                                                   Dims3{numDirections, hiddenSize, hiddenSize}, Dims3{1, 1, 1})
                                                ->getOutput(0);
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
        nvinfer1::IShuffleLayer* unsqueeze = net->addShuffle(*concatenatedBias);
        unsqueeze->setReshapeDimensions(Dims3{numDirections, 1, 2 * NUM_GATES * hiddenSize});
        unsqueeze->setZeroIsPlaceholder(false);
        concatenatedBias = unsqueeze->getOutput(0);

        biasZR
            = net->addSlice(*concatenatedBias, Dims3{0, 0, 0}, Dims3{numDirections, 1, 2 * hiddenSize}, Dims3{1, 1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("Bias for ZR gates shape is: " << biasZR->getDimensions());
        biasH = net->addSlice(*concatenatedBias, Dims3{0, 0, 2 * hiddenSize}, Dims3{numDirections, 1, hiddenSize},
                       Dims3{1, 1, 1})
                    ->getOutput(0);
        LOG_VERBOSE("Bias for H gate shape is: " << biasH->getDimensions());

        recurrenceBiasZR = net->addSlice(*concatenatedBias, Dims3{0, 0, NUM_GATES * hiddenSize},
                                  Dims3{numDirections, 1, 2 * hiddenSize}, Dims3{1, 1, 1})
                               ->getOutput(0);
        LOG_VERBOSE("Recurrence bias for ZR gates shape is: " << recurrenceBiasZR->getDimensions());
        recurrenceBiasH = net->addSlice(*concatenatedBias, Dims3{0, 0, (NUM_GATES + 2) * hiddenSize},
                                 Dims3{numDirections, 1, hiddenSize}, Dims3{1, 1, 1})
                              ->getOutput(0);
        LOG_VERBOSE("Recurrence bias for H gate shape is: " << recurrenceBiasH->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input, &net]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor
            = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("numDirections is: " << numDirections
                                         << ", numDirections Tensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor
            = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE(
            "hiddenSize is: " << hiddenSize << ", hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, Dims{1, 1});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        nvinfer1::IConcatenationLayer* concatenatedShape = net->addConcatenation(
            std::array<nvinfer1::ITensor*, 3>{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}}.data(), 3);
        return concatenatedShape->getOutput(0);
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = net->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);
    ASSERT_NODE(iterationInput, "Failed to add RNN input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // H(t-1)
    auto const getInitialInputValue = [&ctx, &gateOutputShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx, node,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, Dims{1, 1})->getOutput(0),
            gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(Ht1, node);
    LOG_VERBOSE("Hidden state shape: " << Ht1->getOutput(0)->getDimensions());

    // Compute stackedZR(t) = f(X(t) * W[zr]^T + H(t-1) * R[zr]^T + (Wb[zr] + Rb[zr])). stackedZR(t) has shape
    // (numDirections, batchSize, 2 * hiddenSize)
    nvinfer1::ITensor* xtWTZR
        = net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsZR, mOp::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("X(t) * W[zr]^T -> " << xtWTZR->getDimensions());

    nvinfer1::ITensor* ht1RT
        = net->addMatrixMultiply(*Ht1->getOutput(0), mOp::kNONE, *recurrenceWeightsZR, mOp::kTRANSPOSE)->getOutput(0);
    LOG_VERBOSE("H(t-1) * R[zr]^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* stackedZRt = net->addElementWise(*xtWTZR, *ht1RT, eOp::kSUM)->getOutput(0);
    if (biasZR && recurrenceBiasZR)
    {
        stackedZRt = net->addElementWise(*stackedZRt, *biasZR, eOp::kSUM)->getOutput(0);
        stackedZRt = net->addElementWise(*stackedZRt, *recurrenceBiasZR, eOp::kSUM)->getOutput(0);
    }

    nvinfer1::IActivationLayer* stackedZRtLayer
        = net->addActivation(*addClip(ctx, stackedZRt, clip), activations.at(0));
    stackedZRtLayer->setAlpha(activationAlphas.at(0));
    stackedZRtLayer->setBeta(activationBetas.at(0));
    stackedZRt = stackedZRtLayer->getOutput(0);
    LOG_VERBOSE("stackedZR(t) -> " << stackedZRt->getDimensions());

    auto const isolateGate = [&ctx, &hiddenSize, &gateOutputShape, &net](
                                 nvinfer1::ITensor* gates, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolateGate = net->addSlice(*gates, Dims3{0, 0, 0}, Dims3{0, 0, 0}, Dims3{1, 1, 1});
        isolateGate->setInput(1,
            *addConstant(ctx, std::vector<int32_t>{0, 0, gateIndex * hiddenSize},
                ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 3})
                 ->getOutput(0));                   // Start
        isolateGate->setInput(2, *gateOutputShape); // Size
        return isolateGate->getOutput(0);
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
    nvinfer1::ITensor* xtWTH
        = net->addMatrixMultiply(*iterationInput, mOp::kNONE, *weightsH, mOp::kTRANSPOSE)->getOutput(0);
    if (linearBeforeReset == 0)
    {
        // h(t) = g(xtWTH + (r(t) . H(t-1)) * (R[h]^T) + Rb[h] + Wb[h])
        // rtHt1 = (r(t) . H(t-1))
        nvinfer1::ITensor* rtHt1 = net->addElementWise(*rt, *Ht1->getOutput(0), eOp::kPROD)->getOutput(0);
        // rtHt1Rh = (r(t) . H(t-1)) * (R[h]^T)
        nvinfer1::ITensor* rtHt1Rh
            = net->addMatrixMultiply(*rtHt1, mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE)->getOutput(0);

        // (xtWTH + rtHt1Rh) + (Rb[h] + Wb[h])
        nvinfer1::ITensor* actInput = net->addElementWise(*xtWTH, *rtHt1Rh, eOp::kSUM)->getOutput(0);

        // If bias is defines, both recurrence and normal bias must be present
        if (recurrenceBiasH && biasH)
        {
            nvinfer1::ITensor* secondSum = net->addElementWise(*recurrenceBiasH, *biasH, eOp::kSUM)->getOutput(0);
            actInput = net->addElementWise(*actInput, *secondSum, eOp::kSUM)->getOutput(0);
        }

        nvinfer1::IActivationLayer* htLayer = net->addActivation(*addClip(ctx, actInput, clip), activations.at(1));
        ASSERT_NODE(htLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = htLayer->getOutput(0);
    }
    else
    {
        // h(t) = g(xtWTH + (r(t) . (H(t-1) * (R[h]^T) + Rb[h])) + Wb[h])
        // ht1Rh = H(t-1) * (R[h]^T)
        nvinfer1::ITensor* ht1Rh
            = net->addMatrixMultiply(*Ht1->getOutput(0), mOp::kNONE, *recurrenceWeightsH, mOp::kTRANSPOSE)
                  ->getOutput(0);

        // rtHtRhRbh = r(t) . (ht1Rh + Rb[h])
        if (recurrenceBiasH)
        {
            ht1Rh = net->addElementWise(*ht1Rh, *recurrenceBiasH, eOp::kSUM)->getOutput(0);
        }
        nvinfer1::ITensor* rtHtRhRbh = net->addElementWise(*rt, *ht1Rh, eOp::kPROD)->getOutput(0);

        // h(t) = g(xtWTH + rtHtRhRbh + Wb[h])
        if (biasH)
        {
            rtHtRhRbh = net->addElementWise(*rtHtRhRbh, *biasH, eOp::kSUM)->getOutput(0);
        }
        nvinfer1::IActivationLayer* htLayer = net->addActivation(
            *addClip(ctx, net->addElementWise(*xtWTH, *rtHtRhRbh, eOp::kSUM)->getOutput(0), clip), activations.at(1));
        ASSERT_NODE(htLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        htLayer->setAlpha(activationAlphas.at(1));
        htLayer->setBeta(activationBetas.at(1));
        ht = htLayer->getOutput(0);
    }
    LOG_VERBOSE("h(t) -> " << ht->getDimensions());

    // H(t) = (1 - z(t)) . h(t) + (z(t) . H(t-1))
    nvinfer1::ITensor* Ht
        = net->addElementWise(
                 *net->addElementWise(*net->addElementWise(*addConstantScalar(ctx, 1.f,
                                                               ::ONNX_NAMESPACE::TensorProto::FLOAT, Dims3{1, 1, 1})
                                                                ->getOutput(0),
                                              *zt, eOp::kSUB)
                                           ->getOutput(0),
                         *ht, eOp::kPROD)
                      ->getOutput(0),
                 *net->addElementWise(*zt, *Ht1->getOutput(0), eOp::kPROD)->getOutput(0), eOp::kSUM)
              ->getOutput(0);

    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape = ctx->network()
                                             ->addElementWise(*gateOutputShape,
                                                 *addConstant(ctx, std::vector<int32_t>{numDirections, 1, 1},
                                                     ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})
                                                      ->getOutput(0),
                                                 nvinfer1::ElementWiseOperation::kDIV)
                                             ->getOutput(0);
    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1->getOutput(0), Ht, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ht1->getOutput(0), Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
        numDirections, inputs, direction == "reverse"));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*Ht1->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(Hardmax)
{
    CHECK(notInvalidType(inputs.at(0), {"INT64", "INT32", "INT8", "UINT8", "BOOL"}));
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* values = &convertToTensor(inputs.at(0), ctx);
    auto originalDims = values->getDimensions();

    int32_t axis = attrs.get("axis", ctx->getOpsetVersion() < 13 ? 1 : -1);
    CHECK(convertAxis(axis, originalDims.nbDims));
    if (ctx->getOpsetVersion() < 13)
    {
        // Reshape into 2D tensor
        values = flattenTensor(ctx, node, *values, axis, true);
        axis = 1;
    }
    uint32_t axisMask = 1 << axis;

    auto* topKLayer = ctx->network()->addTopK(*values, nvinfer1::TopKOperation::kMAX, /* k */ 1, axisMask);
    ASSERT_NODE(topKLayer != nullptr, "Failed to add TopK layer in HardMax importer!", node, nodeIdx,
        ErrorCode::kINTERNAL_ERROR);

    auto* squeezedIndices = squeezeTensor(ctx, node, *topKLayer->getOutput(1), {axis});
    auto* zeroOneTensor = addConstant(ctx, std::vector<int32_t>{0, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
        nvinfer1::Dims{
            1, {2}})->getOutput(0);
    auto* depth = getAxisLength(ctx, values, axis, nvinfer1::Dims{0});
    auto* oneHotLayer = ctx->network()->addOneHot(*squeezedIndices, *zeroOneTensor, *depth, axis);
    ASSERT_NODE(oneHotLayer != nullptr, "Failed to add OneHot layer in HardMax importer!", node, nodeIdx,
        ErrorCode::kINTERNAL_ERROR);
    auto* oneHotOutput = oneHotLayer->getOutput(0);

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
    auto* layer = ctx->network()->addIdentity(convertToTensor(inputs.at(0), ctx));
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(If)
{
    OnnxAttrs attrs(node, ctx);
    auto cond = inputs.at(0);

    ::ONNX_NAMESPACE::GraphProto const& thenGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("then_branch");
    ::ONNX_NAMESPACE::GraphProto const& elseGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("else_branch");

    // Number of outputs are the same between the two branches.
    ASSERT_NODE(thenGraph.output_size() == elseGraph.output_size(),
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
        CHECK(onnx2trt::parseGraph(ctx, body, errors));
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
    ASSERT_NODE(condTensor, "Failed to convert the input cond to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto conditional = ctx->network()->addIfConditional();
    conditional->setName(getNodeName(node).c_str());
    conditional->setCondition(*condTensor);

    std::vector<nvinfer1::ILayer*> thenLayers, elseLayers;
    std::vector<TensorOrWeights> thenSubgraphTensors;
    std::vector<TensorOrWeights> elseSubgraphTensors;
    CHECK(importSubgraph(ctx, thenGraph, thenLayers, thenSubgraphTensors));
    CHECK(importSubgraph(ctx, elseGraph, elseLayers, elseSubgraphTensors));

    using InputsMap = std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer*>;
    InputsMap inputsMap;
    CHECK(addIfInputLayers(ctx, conditional, inputsMap, thenLayers));
    CHECK(addIfInputLayers(ctx, conditional, inputsMap, elseLayers));

    ASSERT_NODE(thenSubgraphTensors.size() == elseSubgraphTensors.size(),
        "Found different number of output tensors in If conditional subgraphs! then outputs = "
            << thenSubgraphTensors.size() << ", else outputs = " << elseSubgraphTensors.size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    for (size_t i = 0; i < thenSubgraphTensors.size(); i++)
    {
        auto* thenOut = &convertToTensor(thenSubgraphTensors[i], ctx);
        auto* elseOut = &convertToTensor(elseSubgraphTensors[i], ctx);
        auto* outputLayer = conditional->addOutput(*thenOut, *elseOut);
        ASSERT_NODE(outputLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(outputLayer, std::string(conditional->getName()) + "_OutputLayer", nullptr);
        graphOutputs.emplace_back(outputLayer->getOutput(0));
    }

    return {graphOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(ImageScaler)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs{node, ctx};
    // Shift the input by a per-channel bias value.
    std::vector<float> biases = attrs.get<std::vector<float>>("bias");
    nvinfer1::Dims dims{1, static_cast<int>(biases.size())};
    ShapedWeights shiftWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::copy(biases.begin(), biases.end(), static_cast<float*>(shiftWeights.values));
    // Scale is applied to every element of the input, but we need to duplicate it over every channel.
    float scale = attrs.get<float>("scale", 1.0f);
    ShapedWeights scaleWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, dims);
    std::fill(static_cast<float*>(scaleWeights.values), static_cast<float*>(scaleWeights.values) + scaleWeights.count(),
        scale);
    // Finally add the scale layer.
    auto layer = ctx->network()->addScale(
        tensor, nvinfer1::ScaleMode::kCHANNEL, shiftWeights, scaleWeights, nvinfer1::Weights{});
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    auto inputDataType = inputs.at(0).getDataType();
    auto scaleDataType = inputs.at(1).getDataType();
    auto biasDataType = inputs.at(2).getDataType();

    ASSERT_NODE((inputDataType == DataType::kFLOAT || inputDataType == DataType::kHALF),
        "Inputs must be either FLOAT or FLOAT16. Input type is " + getTrtDtypeName(inputDataType) + ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    ASSERT_NODE((inputDataType == scaleDataType && scaleDataType == biasDataType),
        "Inputs must be either all FLOAT or all FLOAT16. Input type = " + getTrtDtypeName(inputDataType)
            + ", scale type = " + getTrtDtypeName(scaleDataType)
            + ", bias type = " + getTrtDtypeName(biasDataType) + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Choose plugin implementation for non-VC and non-HC engines, and native implementation
    // for VC and HC engines.
    auto flags = ctx->getFlags();
    uint32_t nativeInstanceNormFlag = 1U << static_cast<uint32_t>(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);
    if (flags & nativeInstanceNormFlag)
    {
        return normalizationHelper(ctx, node, nodeIdx, inputs);
    }
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
    auto inputDims = input.getDimensions();
    nvinfer1::Dims scalarDims{inputDims.nbDims};
    std::fill(scalarDims.d, scalarDims.d + scalarDims.nbDims, 1);
    auto& zeroTensor = *addConstantScalar(ctx, 0.F, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims)->getOutput(0);

    if (detectNegative)
    {
        auto* isNeg
            = ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kLESS)->getOutput(0);
        auto* isInf = ctx->network()->addUnary(input, nvinfer1::UnaryOperation::kISINF)->getOutput(0);
        RETURN_FIRST_OUTPUT(
            ctx->network()->addElementWise(*isNeg, *isInf, nvinfer1::ElementWiseOperation::kAND), node, nodeIdx);
    }
    if (detectPositive)
    {
        auto* isPos
            = ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kGREATER)->getOutput(0);
        auto* isInf = ctx->network()->addUnary(input, nvinfer1::UnaryOperation::kISINF)->getOutput(0);
        RETURN_FIRST_OUTPUT(
            ctx->network()->addElementWise(*isPos, *isInf, nvinfer1::ElementWiseOperation::kAND), node, nodeIdx);
    }
    // In this case, always return false.
    auto* isPos
        = ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kGREATER)->getOutput(0);
    auto* isNeg
        = ctx->network()->addElementWise(input, zeroTensor, nvinfer1::ElementWiseOperation::kLESS)->getOutput(0);
    RETURN_FIRST_OUTPUT(
        ctx->network()->addElementWise(*isPos, *isNeg, nvinfer1::ElementWiseOperation::kAND), node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(IsNaN)
{
    // IEEE arithmetic guarantees that x == x is false if x is a NaN, and true otherwise.
    std::vector<TensorOrWeights> const newInputs{inputs[0], inputs[0]};
    auto equalResult = elementwiseHelper(ctx, node, nodeIdx, newInputs, nvinfer1::ElementWiseOperation::kEQUAL);
    if (equalResult.is_error())
    {
        return equalResult;
    }
    auto equalRet = equalResult.value().at(0);
    return unaryHelper(ctx, node, nodeIdx, equalRet, nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(LayerNormalization)
{
    auto* input = &convertToTensor(inputs.at(0), ctx);
    auto* scale = &convertToTensor(inputs.at(1), ctx);

    auto dt = input->getType();
    nvinfer1::IConstantLayer* biasLayer;
    if (dt == DataType::kHALF)
    {
        biasLayer = addConstantScalar(ctx, static_cast<half_float::half>(0), ::ONNX_NAMESPACE::TensorProto::FLOAT16);
    }
    else if (dt == DataType::kBF16)
    {
        biasLayer = addConstantScalar(ctx, static_cast<BFloat16>(0), ::ONNX_NAMESPACE::TensorProto::BFLOAT16);
    }
    else
    {
        biasLayer = addConstantScalar(ctx, static_cast<float>(0), ::ONNX_NAMESPACE::TensorProto::FLOAT);
    }
    auto* bias = inputs.size() == 3 ? &convertToTensor(inputs.at(2), ctx) : biasLayer->getOutput(0);

    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    int32_t axis = attrs.get("axis", -1);
    nvinfer1::DataType computeType = nvinfer1::DataType::kFLOAT;
    convertDtype(attrs.get<int32_t>("stash_type", 1), &computeType);

    int32_t const nbDims = input->getDimensions().nbDims;
    CHECK(convertAxis(axis, nbDims));
    uint32_t axesMask{0};

    // Populate axesMask with axis values
    for (int32_t i = axis; i < nbDims; i++)
    {
        axesMask |= 1 << i;
    }

    // Broadcast scale and bias to input size
    broadcastTensors(ctx, input, scale);
    broadcastTensors(ctx, input, bias);

    auto* layer = ctx->network()->addNormalization(*input, *scale, *bias, axesMask);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setEpsilon(epsilon);
    layer->setComputePrecision(computeType);
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
    return {{greaterLessOrEqual(ctx, node, nodeIdx, &convertToTensor(inputs.at(0), ctx),
        &convertToTensor(inputs.at(1), ctx),
        /*greater*/ false)}};
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
    auto* softmax = addSoftmax(ctx, node, input);
    nvinfer1::IUnaryLayer* unaryLayer = ctx->network()->addUnary(*softmax, nvinfer1::UnaryOperation::kLOG);
    ASSERT_NODE(unaryLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *unaryLayer->getOutput(0), shapeOf(input));
    RETURN_FIRST_OUTPUT(reshapeLayer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Loop)
{
    constexpr int32_t NB_NON_STATE_INPUTS = 2; // First 2 inputs are trip count and condition respectively.
    constexpr int32_t NB_DISCARDED_OUTPUTS
        = 1; // First output is the updated value of the condition, and is ignored by the outer loop node.
    constexpr int32_t DUMMY_SCAN_OUTPUT_LENGTH = 1024;
    ASSERT_NODE((inputs.size() >= 2),
        "The Loop operator requires at least 2 inputs. The current number of inputs = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    int32_t const nbInputs = node.input().size();
    // The number of state variables on the input and output is the same.
    int32_t const nbStateVars = nbInputs - NB_NON_STATE_INPUTS;

    ::ONNX_NAMESPACE::GraphProto const& body = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("body");

    auto loop = ctx->network()->addLoop();
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
            ASSERT_NODE(tripLimit, "Failed to convert the trip-count input to a scalar.", node, nodeIdx,
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
        ASSERT_NODE(cond, "Failed to convert the input cond to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        ctx->loopTensors()[body.input(1).name()] = node.input(1);
        ctx->registerTensor(cond, body.input(1).name());
    }
    // Add initial state inputs using recurrent layers.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (size_t i = 2; i < inputs.size(); ++i)
    {
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs[i], ctx)));
        ctx->loopTensors()[body.input(i).name()] = node.input(i);
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
        LOG_VERBOSE("Mapped Loop node input " << node.input(i) << " to loop body input " << body.input(i).name());
    }

    // Loop body
    std::vector<Status> errors{};
    CHECK(onnx2trt::parseGraph(ctx, body, errors));

    if (cond)
    {
        // Add recurrence for loop condition
        auto recurrence = loop->addRecurrence(*cond);
        auto const& bodyOutputName = body.output(0).name();
        auto condOutput = convertToScalar(ctx, &convertToTensor(ctx->tensors().at(bodyOutputName), ctx));
        recurrence->setInput(1, *condOutput);
        loop->addTripLimit(*recurrence->getOutput(0), nvinfer1::TripLimit::kWHILE);
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
        nodeOutputs.emplace_back(
            loop->addLoopOutput(*stateVars.at(i)->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    }
    int32_t const nbOutputs = body.output_size();
    // Finally, set up scan outputs if there are any
    for (int32_t i = nbStateVars + NB_DISCARDED_OUTPUTS; i < nbOutputs; ++i)
    {
        auto const& bodyOutputName = body.output(i).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
                                        << ", with shape: " << scanOutput.getDimensions());
        nvinfer1::ILoopOutputLayer* trtScanOut = loop->addLoopOutput(scanOutput, nvinfer1::LoopOutput::kCONCATENATE, 0);
        // If trip limit is set, we can set the loop output to the tripLimit, otherwise, set to some dummy constant
        // value.
        // In the latter case, the scan outputs must not be used in the rest of the model.
        if (tripLimit)
        {
            trtScanOut->setInput(1, *tripLimit);
        }
        else
        {
            trtScanOut->setInput(1,
                *addConstantScalar(ctx, DUMMY_SCAN_OUTPUT_LENGTH, ::ONNX_NAMESPACE::TensorProto_DataType_INT32)
                     ->getOutput(0));
        }
        nodeOutputs.emplace_back(trtScanOut->getOutput(0));
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
    auto* layer = ctx->network()->addLRN(tensor, size, alpha, beta, bias);
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
        nvinfer1::IShuffleLayer* reshapeBias = ctx->network()->addShuffle(*bias);
        reshapeBias->setReshapeDimensions(nvinfer1::Dims3{numDirections, 2, NUM_GATES * hiddenSize});
        reshapeBias->setZeroIsPlaceholder(false);
        LOG_VERBOSE("Reshaping bias to: " << reshapeBias->getOutput(0)->getDimensions());
        combinedBias = ctx->network()
                           ->addReduce(*reshapeBias->getOutput(0), nvinfer1::ReduceOperation::kSUM, /*axis=*/0b010,
                               /*keepDimensions=*/true)
                           ->getOutput(0);
        LOG_VERBOSE("After reduction, bias shape is: " << combinedBias->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor
            = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("numDirectionsTensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor
            = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, nvinfer1::Dims{1, 1});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        std::array<nvinfer1::ITensor*, 3> tensors{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}};
        nvinfer1::IConcatenationLayer* concatenatedShape = ctx->network()->addConcatenation(tensors.data(), 3);
        return concatenatedShape->getOutput(0);
    };
    nvinfer1::ITensor* gateOutputShape = initialStateShape();
    LOG_VERBOSE("Gate output rank (equal to initial hidden/cell state rank): " << gateOutputShape->getDimensions());

    auto const getInitialInputValue = [&ctx, &gateOutputShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx, node,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, 1})
                ->getOutput(0),
            gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::ITensor* initialCellState = getInitialInputValue(6);
    LOG_VERBOSE("Initial cell state shape: " << initialCellState->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = ctx->network()->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);
    ASSERT_NODE(iterationInput, "Failed to add RNN input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(Ht1, node);
    LOG_VERBOSE("Hidden state shape: " << Ht1->getOutput(0)->getDimensions());

    // C(t-1)
    nvinfer1::IRecurrenceLayer* Ct1 = loop->addRecurrence(*initialCellState);
    LOG_VERBOSE("Cell state shape: " << Ct1->getOutput(0)->getDimensions());

    // Compute intermediate(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)). intermediate(t) has shape (numDirections,
    // batchSize, 4 * hiddenSize)
    nvinfer1::ITensor* xtWT = ctx->network()
                                  ->addMatrixMultiply(*iterationInput, nvinfer1::MatrixOperation::kNONE, *weights,
                                      nvinfer1::MatrixOperation::kTRANSPOSE)
                                  ->getOutput(0);
    LOG_VERBOSE("X(t) * W^T -> " << xtWT->getDimensions());

    nvinfer1::ITensor* ht1RT = ctx->network()
                                   ->addMatrixMultiply(*Ht1->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                       *recurrenceWeights, nvinfer1::MatrixOperation::kTRANSPOSE)
                                   ->getOutput(0);
    LOG_VERBOSE("H(t-1) * R^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* intermediatet = ctx->network()->addElementWise(*xtWT, *ht1RT, eOp::kSUM)->getOutput(0);
    if (combinedBias)
    {
        intermediatet = ctx->network()->addElementWise(*intermediatet, *combinedBias, eOp::kSUM)->getOutput(0);
    }
    LOG_VERBOSE("intermediate(t) -> " << intermediatet->getDimensions());

    // Gate shape is (numDirections, batchSize, hiddenSize)
    auto const isolateGate
        = [&ctx, &hiddenSize, &gateOutputShape](nvinfer1::ITensor* gates, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolate = ctx->network()->addSlice(
            *gates, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
        isolate->setInput(1,
            *addConstant(ctx, std::vector<int32_t>{0, 0, gateIndex * hiddenSize},
                ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})
                 ->getOutput(0));               // Start
        isolate->setInput(2, *gateOutputShape); // Size
        return isolate->getOutput(0);
    };

    // Compute peephole connections
    nvinfer1::ITensor* peephole{nullptr};
    if (inputs.size() > 7 && inputs.at(7))
    {
        peephole = &convertToTensor(inputs.at(7), ctx);
    }

    auto const addPeephole = [&ctx, &node, &hiddenSize, &numDirections, &peephole](nvinfer1::ITensor* gate,
                                 nvinfer1::ITensor* cellState, int32_t gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolatePeephole
            = ctx->network()->addSlice(*peephole, nvinfer1::Dims2{0, gateIndex * hiddenSize},
                nvinfer1::Dims2{numDirections, hiddenSize}, nvinfer1::Dims2{1, 1});
        auto* peepholeWeights = unsqueezeTensor(ctx, node, *isolatePeephole->getOutput(0), std::vector<int32_t>{1});
        LOG_VERBOSE("Peephole weight for gate: " << gateIndex << " shape: " << peepholeWeights->getDimensions());

        return ctx->network()
            ->addElementWise(*gate,
                *ctx->network()->addElementWise(*peepholeWeights, *cellState, eOp::kPROD)->getOutput(0), eOp::kSUM)
            ->getOutput(0);
    };

    // NOTE: . represents a hadamard product
    nvinfer1::ITensor* itGate = isolateGate(intermediatet, 0);

    if (peephole)
    {
        // i(t) (w/ peephole) =  i(t) + Pi . C(t-1)
        itGate = addPeephole(itGate, Ct1->getOutput(0), 0);
    }

    nvinfer1::IActivationLayer* itGateAct
        = ctx->network()->addActivation(*addClip(ctx, itGate, clip), activations.at(0));
    itGateAct->setAlpha(activationAlphas.at(0));
    itGateAct->setBeta(activationBetas.at(0));
    itGate = itGateAct->getOutput(0);

    nvinfer1::ITensor* ftGate = isolateGate(intermediatet, 2);

    if (peephole)
    {
        // f(t) (w/ peephole) =  f(t) + Pf . C(t-1)
        ftGate = addPeephole(ftGate, Ct1->getOutput(0), 2);
    }

    nvinfer1::IActivationLayer* ftGateAct
        = ctx->network()->addActivation(*addClip(ctx, ftGate, clip), activations.at(0));
    ftGateAct->setAlpha(activationAlphas.at(0));
    ftGateAct->setBeta(activationBetas.at(0));
    ftGate = ftGateAct->getOutput(0);

    // c(t) = g(intermediate(t)[:, :, 3H:4H])
    nvinfer1::IActivationLayer* ctAct
        = ctx->network()->addActivation(*addClip(ctx, isolateGate(intermediatet, 3), clip), activations.at(1));
    ctAct->setAlpha(activationAlphas.at(1));
    ctAct->setBeta(activationBetas.at(1));

    nvinfer1::ITensor* ctGate = ctAct->getOutput(0);
    LOG_VERBOSE("c(t) -> " << ctGate->getDimensions());

    // C(t) = f(t) . C(t - 1) + i(t) . c(t)
    nvinfer1::ITensor* operandIC = ctx->network()->addElementWise(*itGate, *ctGate, eOp::kPROD)->getOutput(0);
    nvinfer1::ITensor* operandFC
        = ctx->network()->addElementWise(*ftGate, *Ct1->getOutput(0), eOp::kPROD)->getOutput(0);
    nvinfer1::ITensor* Ct = ctx->network()->addElementWise(*operandFC, *operandIC, eOp::kSUM)->getOutput(0);

    nvinfer1::ITensor* singlePassShape = ctx->network()
                                             ->addElementWise(*gateOutputShape,
                                                 *addConstant(ctx, std::vector<int>{numDirections, 1, 1},
                                                     ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})
                                                      ->getOutput(0),
                                                 eOp::kDIV)
                                             ->getOutput(0);

    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ct = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ct1->getOutput(0), Ct, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ct1->getOutput(0), Ct, maxLen, direction == "reverse");
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
        = ctx->network()->addActivation(*addClip(ctx, otGate, clip), activations.at(0));
    otGateAct->setAlpha(activationAlphas.at(0));
    otGateAct->setBeta(activationBetas.at(0));
    otGate = otGateAct->getOutput(0);

    // H(t) = o(t) . h(C(t))
    nvinfer1::IActivationLayer* hAct = ctx->network()->addActivation(*addClip(ctx, Ct, clip), activations.at(2));
    hAct->setAlpha(activationAlphas.at(2));
    hAct->setBeta(activationBetas.at(2));

    nvinfer1::ITensor* Ht = ctx->network()->addElementWise(*otGate, *hAct->getOutput(0), eOp::kPROD)->getOutput(0);
    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2
            ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1->getOutput(0), Ht, singlePassShape)
            : maskRNNHidden(ctx, node, loop, seqLens, Ht1->getOutput(0), Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)

    outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
        numDirections, inputs, direction == "reverse"));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*Ht1->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    // Yc = last value of C(t)
    outputs.emplace_back(loop->addLoopOutput(*Ct1->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));

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
    ASSERT_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF),
        "Only float inputs/outputs supported in LpNormalization. The current data type = " + getTrtDtypeName(dt) + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    CHECK(convertAxis(axis, nbDims));

    nvinfer1::ITensor* norm{nullptr};
    TensorOrWeights zeros = ctx->createTempWeights(trtDataTypeToONNX(inputType), {0, {}});
    nvinfer1::ITensor* zerosTensor = &convertToTensor(zeros, ctx);
    broadcastTensor(ctx, zerosTensor, nbDims);

    if (p == 1)
    {
        // abs(x)
        nvinfer1::IUnaryLayer* absLayer = ctx->network()->addUnary(*input, uOp::kABS);
        ASSERT_NODE(absLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(absLayer, node);
        norm = absLayer->getOutput(0);

        // norm coeff = sum(abs(x)) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true);
        ASSERT_NODE(reduceLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(reduceLayer, node);
        norm = reduceLayer->getOutput(0);
    }
    else if (p == 2)
    {
        // x^2
        auto* sqrLayer = ctx->network()->addElementWise(*input, *input, eOp::kPROD);
        ASSERT_NODE(sqrLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(sqrLayer, node);
        norm = sqrLayer->getOutput(0);

        // sum(x^2) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true);
        ASSERT_NODE(reduceLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(reduceLayer, node);
        norm = reduceLayer->getOutput(0);

        // norm coeff = sqrt(sum(x^2))
        nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*norm, uOp::kSQRT);
        ASSERT_NODE(sqrtLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(sqrtLayer, node);
        norm = sqrtLayer->getOutput(0);
    }

    // norm coeff |= 1 (change 0s to 1s, leave all other values same)
    nvinfer1::IElementWiseLayer* maskLayer = ctx->network()->addElementWise(*norm, *zerosTensor, eOp::kEQUAL);
    ASSERT_NODE(maskLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(maskLayer, node);
    nvinfer1::ITensor* mask = maskLayer->getOutput(0);
    mask = castHelper(ctx, mask, dt);
    auto* combinedLayer = ctx->network()->addElementWise(*norm, *mask, eOp::kSUM);
    ASSERT_NODE(combinedLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(combinedLayer, node);
    norm = combinedLayer->getOutput(0);

    // x/(norm coeff)
    // norm tensor is broadcast along axis dimension to match shape of input
    auto* layer = ctx->network()->addElementWise(*input, *norm, eOp::kDIV);
    ctx->registerLayer(layer, node);
    ASSERT_NODE(layer, "Failed to register layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

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
    ASSERT_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF),
        "Only float inputs/outputs supported in LpPool. The current data type = " + getTrtDtypeName(dt) + ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    nvinfer1::Dims kernelShape = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding(false);
    getKernelParams(ctx, node, &kernelShape, &strides, &begPadding, &endPadding, paddingMode, exclude_padding);

    nvinfer1::Dims scalarDims = makeDims(nbDims, 1);
    float kernelSz{1.0F};
    for (int32_t i = 0; i < kernelShape.nbDims; i++)
    {
        kernelSz *= kernelShape.d[i];
    }

    nvinfer1::IConstantLayer* kernelSzLayer;
    if (dt == DataType::kHALF)
    {
        kernelSzLayer = addConstantScalar(
            ctx, static_cast<half_float::half>(kernelSz), ::ONNX_NAMESPACE::TensorProto::FLOAT16, scalarDims);
    }
    else
    {
        kernelSzLayer
            = addConstantScalar(ctx, kernelSz, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims);
    }

    nvinfer1::ITensor* output{nullptr};
    if (p == 1)
    {
        // x' = abs(x)
        nvinfer1::IUnaryLayer* absLayer = ctx->network()->addUnary(*input, uOp::kABS);
        ASSERT_NODE(absLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(absLayer, node);
        output = absLayer->getOutput(0);
    }
    else if (p == 2)
    {
        // x' = x^2
        auto* sqrLayer = ctx->network()->addElementWise(*input, *input, eOp::kPROD);
        ASSERT_NODE(sqrLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(sqrLayer, node);
        output = sqrLayer->getOutput(0);
    }

    // pool_avg(x')
    nvinfer1::IPoolingLayer* poolLayer = ctx->network()->addPoolingNd(*output, pType::kAVERAGE, kernelShape);
    ASSERT_NODE(poolLayer, "Failed to create a Pooling layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    poolLayer->setPaddingMode(paddingMode);
    poolLayer->setPrePadding(begPadding);
    poolLayer->setPostPadding(endPadding);
    poolLayer->setStrideNd(strides);
    poolLayer->setAverageCountExcludesPadding(exclude_padding);
    ctx->registerLayer(poolLayer, node);
    output = poolLayer->getOutput(0);

    // pool_sum = pool_avg(x')*kernel_size
    auto* correctedSumLayer = ctx->network()->addElementWise(*output, *kernelSzLayer->getOutput(0), eOp::kPROD);
    ASSERT_NODE(correctedSumLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(correctedSumLayer, node);
    output = correctedSumLayer->getOutput(0);

    // if p == 1, output = pool_sum
    // if p == 2, output = sqrt(pool_sum)
    if (p == 2)
    {
        nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*output, uOp::kSQRT);
        ASSERT_NODE(sqrtLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerLayer(sqrtLayer, node);
        output = sqrtLayer->getOutput(0);
    }
    return {{output}};
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "INT64"}));
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
        inputB = unsqueezeTensor(ctx, node, *inputB, axes);
        needSqueezeTail = true;
    }
    else if (t1Dims < t2Dims && t1Dims == 1)
    {
        // The first argument is 1-D, promote to matrix by prepending a 1 in shape.
        // This is done in broadcast extra dimensions.
        needSqueezeHead = true;
    }
    CHECK(broadcastTensors(ctx, inputA, inputB));

    auto const getMatrixOp = [](nvinfer1::ITensor const& input) {
        return (input.getDimensions().nbDims == 1) ? nvinfer1::MatrixOperation::kVECTOR
                                                   : nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB);

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputA, opA, *inputB, opB);
    ctx->registerLayer(matmul, node);

    auto outputTensor = matmul->getOutput(0);
    if (needSqueezeHead)
    {
        // After MM we need remove the prepended 1.
        std::vector<int32_t> axes{0};
        outputTensor = squeezeTensor(ctx, node, *outputTensor, axes);
    }
    if (needSqueezeTail)
    {
        // After MM we need remove the appended 1.
        std::vector<int32_t> axes{outputTensor->getDimensions().nbDims - 1};
        outputTensor = squeezeTensor(ctx, node, *outputTensor, axes);
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
    auto sum_result = elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kSUM);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    auto& sum_input = sum_result.value().at(0);
    nvinfer1::ITensor& sum_tensor = sum_input.tensor();

    int32_t ndim = sum_tensor.getDimensions().nbDims;
    float scale_value = 1.f / inputs.size();
    auto scale_dtype = sum_input.isBFp16()
        ? ::ONNX_NAMESPACE::TensorProto::BFLOAT16
        : (sum_input.isFp16() ? ::ONNX_NAMESPACE::TensorProto::FLOAT16 : ::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto scale_shape = nvinfer1::Dims{ndim, {1, 1, 1, 1, 1, 1, 1, 1}};
    auto scale_weights = ctx->createTempWeights(scale_dtype, scale_shape);
    static_cast<float*>(scale_weights.values)[0] = scale_value;
    auto* constant_layer = ctx->network()->addConstant(scale_weights.shape, scale_weights);
    ASSERT_NODE(constant_layer, "Failed to create the scalar tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->network()->setWeightsName(scale_weights, scale_weights.getName());
    nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
    RETURN_FIRST_OUTPUT(
        ctx->network()->addElementWise(sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD), node,
        nodeIdx);
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

    ASSERT_NODE((dt == DataType::kFLOAT || dt == DataType::kHALF || dt == DataType::kBF16),
        "Only float32/float16/bfloat16 inputs/outputs supported in MeanVarianceNormalization. The current data type = "
            + getTrtDtypeName(dt) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DATATYPE);

    // convert axes vector to bitmask
    std::vector<int32_t> const defaultAxes = {0, 2, 3};
    auto const axes = attrs.get<std::vector<int32_t>>("axes", defaultAxes);
    uint32_t axesMask = 0;

    for (int32_t axis : axes)
    {
        CHECK(convertAxis(axis, dims.nbDims));
        axesMask |= 1 << axis;
    }

    // mean(x) along axes direction
    auto* reduceLayer = ctx->network()->addReduce(*input, rOp::kAVG, axesMask, true);
    ASSERT_NODE(reduceLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(reduceLayer, node);
    auto* meanX = reduceLayer->getOutput(0);

    // numerator: x-mean(x)
    auto* numSubLayer = ctx->network()->addElementWise(*input, *meanX, eOp::kSUB);
    ASSERT_NODE(numSubLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(numSubLayer, node);
    auto* numerator = numSubLayer->getOutput(0);

    // (x-mean(x))^2
    auto* sqrLayer = ctx->network()->addElementWise(*numerator, *numerator, eOp::kPROD);
    ASSERT_NODE(sqrLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(sqrLayer, node);
    auto* sqrNumerator = sqrLayer->getOutput(0);

    // mean((x-mean(x))^2)
    auto* meanLayer = ctx->network()->addReduce(*sqrNumerator, rOp::kAVG, axesMask, true);
    ASSERT_NODE(meanLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(meanLayer, node);
    auto* variance = meanLayer->getOutput(0);

    // sqrt(mean((x-mean(x))^2))
    nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*variance, uOp::kSQRT);
    ASSERT_NODE(sqrtLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(sqrtLayer, node);
    auto* stdDev = sqrtLayer->getOutput(0);

    // denominator: avoid division by zero
    nvinfer1::Dims scalarShape{dims.nbDims};
    std::fill(scalarShape.d, scalarShape.d + scalarShape.nbDims, 1);
    auto* epsilonTensor
        = addConstantScalar(ctx, 1e-9f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, scalarShape)->getOutput(0);
    auto* addEpsLayer = ctx->network()->addElementWise(*stdDev, *epsilonTensor, eOp::kSUM);
    ASSERT_NODE(addEpsLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(addEpsLayer, node);
    stdDev = addEpsLayer->getOutput(0);

    // division numerator/standard-deviation
    auto* divLayer = ctx->network()->addElementWise(*numerator, *stdDev, eOp::kDIV);
    ctx->registerLayer(divLayer, node);

    ASSERT_NODE(divLayer, "Failed to register layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

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
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    using eOp = nvinfer1::ElementWiseOperation;
    OnnxAttrs attrs(node, ctx);
    int32_t const fmod = attrs.get("fmod", 0);
    nvinfer1::ITensor* input0 = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* input1 = &convertToTensor(inputs.at(1), ctx);
    CHECK(broadcastTensors(ctx, input0, input1));

    if (fmod == 0)
    {
        // fmod = 0, inputs can only be integers
        ASSERT_NODE((input0->getType() == DataType::kINT32 || input0->getType() == DataType::kINT64),
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
        TensorOrWeights divResult = elementwiseHelper(ctx, node, nodeIdx, {input0, input1}, eOp::kDIV).value().at(0);

        // Calculate input0 - (input1 * floor(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivFloor
            = modWithFPInputs(ctx, input0, input1, &divResult.tensor(), true);

        // Calculate input0 - (input1 * ceil(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivCeil
            = modWithFPInputs(ctx, input0, input1, &divResult.tensor(), false);

        auto* zero = createZeroTensor(ctx, &divResult.tensor());
        auto* condition = greaterLessOrEqual(ctx, node, nodeIdx, &divResult.tensor(), zero, true);
        auto* outputWithDivFloor = layerWithDivFloor->getOutput(0);
        auto* outputWithDivCeil = layerWithDivCeil->getOutput(0);

        // If (input0 / input1) >= 0, result = input0 - (input1 * floor(input0 / input1))
        // Else result = input0 - (input1 * ceil(input0 / input1))
        auto* result = ctx->network()->addSelect(*condition, *outputWithDivFloor, *outputWithDivCeil);
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
    ASSERT_NODE(inputs.size() >= 2 && inputs.size() <= 5,
        "The node requires between 2-5 inputs. The actual input size is " << inputs.size() << ".", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // Input: boxes
    nvinfer1::ITensor* boxesTensorPtr = &convertToTensor(inputs.at(0), ctx);
    ASSERT_NODE(boxesTensorPtr->getDimensions().nbDims == 3,
        "The boxes tensor must be 3D. The actual rank is " << boxesTensorPtr->getDimensions().nbDims << ".", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // Input: scores
    nvinfer1::ITensor* scoresTensorPtr = &convertToTensor(inputs.at(1), ctx);
    ASSERT_NODE(scoresTensorPtr->getDimensions().nbDims == 3,
        "The scores tensor must be 3D. The actual rank is " << scoresTensorPtr->getDimensions().nbDims << ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

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
        ASSERT_NODE(maxOutputBoxesPerClassTensorPtr != nullptr, "The max_output_boxes_per_class tensor must be 0D",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        auto* constantLayer = ctx->network()->addConstant(
            nvinfer1::Dims{0, {}}, nvinfer1::Weights{DataType::kINT32, &maxOutputBoxesPerClassDefault, 1});
        ASSERT_NODE(constantLayer != nullptr, "Failed to add in constant for default max_output_boxes_per_class", node,
            nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        maxOutputBoxesPerClassTensorPtr = constantLayer->getOutput(0);
    }

    // Input: iou_threshold (default = 0)
    if (inputs.size() >= 4 && !inputs.at(3).isNullTensor())
    {
        iouThresholdTensorPtr = convertToScalar(inputs.at(3), ctx);
        ASSERT_NODE(iouThresholdTensorPtr != nullptr, "The iou_threshold tensor must be 0D", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Input: score_threshold (default = 0)
    if (inputs.size() >= 5 && !inputs.at(4).isNullTensor())
    {
        scoreThresholdTensorPtr = convertToScalar(inputs.at(4), ctx);
        ASSERT_NODE(scoreThresholdTensorPtr != nullptr, "The score_threshold tensor must be 0D", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Transpose scores tensor from [batch, classes, bounding_boxes] to [batch, bounding_boxes, classes]
    nvinfer1::Permutation perm{0, 2, 1};
    nvinfer1::ITensor* transposedScoresTensorPtr = transposeTensor(ctx, node, *scoresTensorPtr, perm);
    ASSERT_NODE(transposedScoresTensorPtr, "Failed to transpose the scores input.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    // Create the NMS layer
    auto* layer = ctx->network()->addNMS(*boxesTensorPtr, *transposedScoresTensorPtr, *maxOutputBoxesPerClassTensorPtr);
    ASSERT_NODE(layer != nullptr, "Failed to create NMS layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT_NODE(false, "Invalid value provided for the center_point_box attribute", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE_ATTR);
    }
    layer->setBoundingBoxFormat(fmt);
    auto* indices = layer->getOutput(0);
    indices = castHelper(ctx, indices, DataType::kINT64);

    return {{indices}};
};

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(OneHot)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    CHECK(notInvalidType(inputs.at(1), {"UINT8"}));
    CHECK(notInvalidType(inputs.at(2), {"UINT8"}));
    ASSERT_NODE(node.input_size(), "OneHot must have exactly 3 inputs. Number of inputs = " << node.input_size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

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
    ASSERT_NODE(depth, "Failed to convert the depth to a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);
    auto axis = attrs.get<int32_t>("axis", -1);
    auto nbDims = indices->getDimensions().nbDims;
    CHECK(convertAxis(axis, nbDims + 1));

    auto* layer = ctx->network()->addOneHot(*indices, *values, *depth, axis);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Or)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensorPtr->getDimensions().nbDims;

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
                ASSERT_NODE((padWeight.count() == 1), "The input constant_value is required to be a scalar.", node,
                    nodeIdx, ErrorCode::kINVALID_NODE);
                switch (padWeight.type)
                {
                case ::ONNX_NAMESPACE::TensorProto::FLOAT:
                    value = static_cast<float const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
                    value = reinterpret_cast<half_float::half const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                case ::ONNX_NAMESPACE::TensorProto::BFLOAT16:
                    value = reinterpret_cast<BFloat16 const*>(padWeight.values)[0];
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
        // Opset 16 optional `axes` input
        if (inputs.size() == 4 && !inputs.at(3).isNullTensor())
        {
            ASSERT_NODE(false, "TensorRT does not support dynamic axes for pad!", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE_INPUT);
        }
    }

    nvinfer1::ITensor* start{};
    nvinfer1::ITensor* size{};
    if (onnxPadding.empty())
    {
        // the pads is from activation instead of initializer or attributes
        nvinfer1::ITensor* onnxPaddingPtr = &convertToTensor(inputs.at(1), ctx);
        ASSERT_NODE((onnxPaddingPtr->getDimensions().nbDims == 1),
            "The padding input must be 1D. The rank of padding input = " << onnxPaddingPtr->getDimensions().nbDims
                                                                         << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT_NODE(onnxPaddingPtr->getDimensions().d[0] == nbDims * 2,
            "pads should be a 1D tensor of shape " << 2 * nbDims
                                                   << ", actual shape is: " << onnxPaddingPtr->getDimensions().d[0],
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

        auto pre = ctx->network()
                       ->addSlice(
                           *onnxPaddingPtr, nvinfer1::Dims{1, {0}}, nvinfer1::Dims{1, {nbDims}}, nvinfer1::Dims{1, {1}})
                       ->getOutput(0);
        auto post = ctx->network()
                        ->addSlice(*onnxPaddingPtr, nvinfer1::Dims{1, {nbDims}}, nvinfer1::Dims{1, {nbDims}},
                            nvinfer1::Dims{1, {1}})
                        ->getOutput(0);

        std::vector<int64_t> const zerosVal(nbDims, 0);
        auto const zeros = addConstant(ctx, zerosVal, ::ONNX_NAMESPACE::TensorProto::INT64,
            nvinfer1::Dims{
                1, {nbDims}})->getOutput(0);
        start = ctx->network()->addElementWise(*zeros, *pre, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
        auto const totalPadding
            = ctx->network()->addElementWise(*pre, *post, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
        size
            = ctx->network()
                  ->addElementWise(shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM)
                  ->getOutput(0);
    }
    else
    {
        // passthrough path for no-op padding
        if (std::all_of(onnxPadding.begin(), onnxPadding.end(), [](int64_t i) { return i == 0; }))
        {
            LOG_VERBOSE("Found no-op pad in node: " + getNodeName(node));
            RETURN_IDENTITY(inputs.at(0), node, nodeIdx);
        }

        // the pads is from initializer or attributes
        nvinfer1::ITensor* totalPadding = nullptr;
        ASSERT_NODE(convertOnnxPadding(ctx, nbDims, onnxPadding, start, totalPadding), "Failed to convert padding!",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        size
            = ctx->network()
                  ->addElementWise(shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM)
                  ->getOutput(0);
    }

    // add slice node
    auto const stride = makeDims(nbDims, 1);
    auto const& dummy = stride;
    auto* layer = ctx->network()->addSlice(*tensorPtr, dummy, dummy, stride);
    ASSERT_NODE(layer, "Could not create padding layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setInput(1, *start);
    layer->setInput(2, *size);
    if (mode == "constant")
    {
        layer->setMode(nvinfer1::SliceMode::kFILL);

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
                fillValue = addConstant(ctx, std::vector<half_float::half>{static_cast<half_float::half>(value)},
                    ::ONNX_NAMESPACE::TensorProto::FLOAT16,
                    nvinfer1::Dims{
                        0, {0}})->getOutput(0);
                break;
            case DataType::kBF16:
                fillValue = addConstant(ctx, std::vector<BFloat16>{static_cast<BFloat16>(value)},
                    ::ONNX_NAMESPACE::TensorProto::BFLOAT16,
                    nvinfer1::Dims{
                        0, {0}})->getOutput(0);
                break;
            case DataType::kFLOAT:
            case DataType::kINT8:
                fillValue = addConstant(ctx, std::vector<float>{value}, ::ONNX_NAMESPACE::TensorProto::FLOAT,
                    nvinfer1::Dims{
                        0, {0}})->getOutput(0);
                break;
            default:
                fillValue = addConstant(ctx, std::vector<int32_t>{static_cast<int32_t>(value)},
                    ::ONNX_NAMESPACE::TensorProto::INT32,
                    nvinfer1::Dims{
                        0, {0}})->getOutput(0);
                break;
            }
            ASSERT_NODE(
                fillValue, "Could not create layer for constant_value", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            layer->setInput(4, *fillValue);
        }
    }
    else if (mode == "reflect")
    {
        layer->setMode(nvinfer1::SliceMode::kREFLECT);
    }
    else if (mode == "edge")
    {
        layer->setMode(nvinfer1::SliceMode::kCLAMP);
    }
    else
    {
        return MAKE_ERROR("Unsupported pad mode", ErrorCode::kUNSUPPORTED_NODE);
    }

    ctx->registerLayer(layer, node);
    return {{layer->getOutput(0)}};
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
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kPOW);
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "INT64"}));
    CHECK(notInvalidType(inputs.at(1), {"INT32", "INT64"}));
    ASSERT_NODE((inputs.size() == 2),
        "The PRelu operator requires exactly 2 inputs. Current input size = " << inputs.size() << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* slopes = &convertToTensor(inputs.at(1), ctx);
    CHECK(broadcastTensors(ctx, input, slopes));
    auto* layer = ctx->network()->addParametricReLU(*input, *slopes);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

NodeImportResult randomHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t nodeIdx,
    ShapeTensor const& inputShape, OnnxAttrs const& attrs, DataType const& inputDType, nvinfer1::FillOperation op)
{
    auto* fillLayer = addFill(ctx, inputShape, op);
    ASSERT_NODE(fillLayer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        default: return MAKE_ERROR("Unsupported data type", ErrorCode::kINVALID_VALUE);
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
    ASSERT_NODE((inputs.size() == 1),
        "The RandomUniformLike operator requires exactly 1 input. Current input size = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE((inputs.at(0).is_tensor()), "The input tensor cannot be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    auto const inputShape = shapeOf(input);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = input.getType();

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
    ASSERT_NODE((inputs.size() == 1),
        "The RandomNormalLike operator requires exactly 1 input. Current input size = " << inputs.size() << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE((inputs.at(0).is_tensor()), "The input tensor cannot be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    auto const inputShape = shapeOf(input);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = input.getType();

    return randomHelper(ctx, node, nodeIdx, inputShape, attrs, dType, nvinfer1::FillOperation::kRANDOM_NORMAL);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    ASSERT_NODE((inputs.at(0).getType() == inputs.at(1).getType(), inputs.at(0).getType() == inputs.at(2).getType()),
        "For range operator types for start, limit, and delta must be identical. Type of start = "
            + inputs.at(0).getType() + ", type of limit = " + inputs.at(1).getType()
            + ", type of delta = " + inputs.at(2).getType() + ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    bool const isInt32 = inputs.at(0).isInt32();
    bool const isInt64 = inputs.at(0).isInt64();
    bool const isFp32 = inputs.at(0).isFp32();
    ASSERT_NODE((isInt32 || isInt64 || isFp32),
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
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
    NodeImportResult abs_result = unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
    if (abs_result.is_error())
    {
        return abs_result;
    }
    TensorOrWeights abs_input = abs_result.value().at(0);
    return reduceTensor(ctx, node, nodeIdx, abs_input, nvinfer1::ReduceOperation::kSUM);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSum);
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSum)
{
    auto sum_result = importReduceSum(ctx, node, nodeIdx, inputs);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    TensorOrWeights sum_input = sum_result.value().at(0);
    return unaryHelper(ctx, node, nodeIdx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    // TODO: Abstract this sequence with a function or macro
    auto exp_result = unaryHelper(ctx, node, nodeIdx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
    if (exp_result.is_error())
    {
        return exp_result;
    }
    auto exp_inputs = exp_result.value();
    return importReduceLogSum(ctx, node, nodeIdx, exp_inputs);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSumSquare);
DEFINE_BUILTIN_OP_IMPORTER(ReduceL2)
{
    auto sum_sqr_result = importReduceSumSquare(ctx, node, nodeIdx, inputs);
    if (sum_sqr_result.is_error())
    {
        return sum_sqr_result;
    }
    TensorOrWeights sum_sqr = sum_sqr_result.value().at(0);
    return unaryHelper(ctx, node, nodeIdx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMax)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kMAX);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMean)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kAVG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMin)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kMIN);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceProd)
{
    return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kPROD);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSum)
{
    if (ctx->getOpsetVersion() >= 13 && inputs.size() >= 2)
    {
        return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kSUM, inputs.at(1));
    }
    else
    {
        return reduceTensor(ctx, node, nodeIdx, inputs.at(0), nvinfer1::ReduceOperation::kSUM);
    }
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    auto* sqr_layer = ctx->network()->addElementWise(tensor, tensor, nvinfer1::ElementWiseOperation::kPROD);
    ASSERT_NODE(sqr_layer, "Failed to add an ElementWise layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* sqr_tensorPtr = sqr_layer->getOutput(0);
    return reduceTensor(ctx, node, nodeIdx, sqr_tensorPtr, nvinfer1::ReduceOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kRELU);
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
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t inputRank = input.getDimensions().nbDims;
    ASSERT_NODE((inputRank > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    OnnxAttrs attrs(node, ctx);

    auto mode = attrs.get<std::string>("mode", "nearest");
    auto resizeMode = nvinfer1::ResizeMode::kNEAREST;

    if (mode == "cubic")
    {
        resizeMode = nvinfer1::ResizeMode::kCUBIC;
    }
    else if (mode == "linear")
    {
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }

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
            if (resizeMode == nvinfer1::ResizeMode::kCUBIC)
            {
                auto cubicCoeff = attrs.get<float>("cubic_coeff_a", -0.75F);
                layer->setCubicCoeff(cubicCoeff);
            }

            if (inputs.size() == 4 && !inputs.at(3).isNullTensor())
            {
                if (inputs.at(3).is_weights())
                {
                    ShapedWeights sizeWeights = inputs.at(3).weights();
                    ASSERT_NODE((sizeWeights.shape.nbDims == 1 && sizeWeights.shape.d[0] == inputRank),
                        "The shape of weights must align with input data. Weights rank = "
                            << sizeWeights.shape.nbDims << ", shape of weights = " << sizeWeights.shape.d[0]
                            << ", rank of input = " << inputRank << ".",
                        node, nodeIdx, ErrorCode::kINVALID_NODE);
                    int32_t* sizeValues
                        = convertINT64(static_cast<int64_t*>(sizeWeights.values), {1, {inputRank}}, ctx);
                    nvinfer1::Dims resizeShape{inputRank, {}};
                    for (int32_t i = 0; i < inputRank; i++)
                    {
                        resizeShape.d[i] = sizeValues[i];
                    }
                    layer->setOutputDimensions(resizeShape);
                }
                else
                {
                    auto* resizeShape = &convertToTensor(inputs.at(3), ctx);
                    layer->setInput(1, *resizeShape);
                }
                layer->setResizeMode(resizeMode);
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
        ShapedWeights scales_weights = scales.weights();
        ASSERT_NODE((scales_weights.shape.nbDims == 1),
            "The scales input must be 1D. Scales rank = "
                << scales_weights.shape.nbDims << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        int32_t scaleSize = scales_weights.shape.d[0];
        ASSERT_NODE((scaleSize == inputRank),
            "The shape of input scales must align with the input rank. Rank of input scales = "
                << scaleSize << ", rank of inputs = " << inputRank << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        float const* scaleValues = static_cast<float const*>(scales_weights.values);

        // check resize dims
        if (resizeMode == nvinfer1::ResizeMode::kLINEAR)
        {
            ASSERT_NODE(canUseNDResize(scaleSize, scaleValues, 3),
                "This version of TensorRT only supports linear resizing on the outermost 3 dimensions.", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        else if (resizeMode == nvinfer1::ResizeMode::kCUBIC)
        {
            ASSERT_NODE(canUseNDResize(scaleSize, scaleValues, 2),
                "This version of TensorRT only supports cubic resizing on the outermost 2 dimensions.", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scaleValues, inputRank);
    }
    else
    {
        nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, input, scales);
        layer->setInput(1, *resizeShape);
    }

    layer->setResizeMode(resizeMode);

    LOG_VERBOSE("Running resize layer with: \n"
        << "Transformation mode: " << transformationMode << "\n"
        << "Resize mode: " << mode << "\n");

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
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
    ASSERT_NODE((inputs.size() == 2),
        "ReverseSequence expects two input tensors: input and sequence_lens. Current input size = " << inputs.size()
                                                                                                    << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* sequenceLens = &convertToTensor(inputs.at(1), ctx);
    auto const inputDims = input->getDimensions();
    auto const sequenceLensDims = sequenceLens->getDimensions();
    ASSERT_NODE((inputDims.nbDims >= 2),
        "Rank of input must be at least two. Current rank of inputs = " << inputDims.nbDims << ".", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
    ASSERT_NODE((sequenceLensDims.nbDims == 1),
        "Rank of sequence_lens must be one. Current rank of sequence lens = " << sequenceLensDims.nbDims << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs{node, ctx};
    int32_t const batchAxis = attrs.get<int32_t>("batch_axis", 1);
    int32_t const sequenceAxis = attrs.get<int32_t>("time_axis", 0);
    ASSERT_NODE((batchAxis >= 0 && batchAxis <= inputDims.nbDims), "Invalid batch_axis", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE_ATTR);
    ASSERT_NODE((sequenceAxis >= 0 && sequenceAxis <= inputDims.nbDims), "Invalid time_axis", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE_ATTR);

    auto layer = ctx->network()->addReverseSequence(*input, *sequenceLens);
    ctx->registerLayer(layer, node);
    ASSERT_NODE(layer, "Failed to add ReverseSequence layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

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

    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

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
        nvinfer1::IShuffleLayer* reshapeBias = ctx->network()->addShuffle(*bias);
        reshapeBias->setReshapeDimensions(nvinfer1::Dims3{numDirections, 2, hiddenSize});
        reshapeBias->setZeroIsPlaceholder(false);
        LOG_VERBOSE("Reshaping bias to: " << reshapeBias->getOutput(0)->getDimensions());
        combinedBias = ctx->network()
                           ->addReduce(*reshapeBias->getOutput(0), nvinfer1::ReduceOperation::kSUM, /*axis=*/0b010,
                               /*keepDimensions=*/true)
                           ->getOutput(0);
        LOG_VERBOSE("After reduction, bias shape is: " << combinedBias->getDimensions());
    }

    // Get a shape tensor containing: (numDirections, batchSize, hiddenSize)
    auto const initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
        // Get batchSize from input shape
        nvinfer1::ITensor* numDirectionsTensor
            = addConstantScalar(ctx, numDirections, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("numDirectionsTensor shape: " << numDirectionsTensor->getDimensions());
        nvinfer1::ITensor* hiddenSizeTensor
            = addConstantScalar(ctx, hiddenSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("hiddenSizeTensor shape: " << hiddenSizeTensor->getDimensions());
        nvinfer1::ITensor* batchSizeTensor = getAxisLength(ctx, input, 1, nvinfer1::Dims{1, 1});
        LOG_VERBOSE("batchSizeTensor shape: " << batchSizeTensor->getDimensions());

        std::array<nvinfer1::ITensor*, 3> tensors{{numDirectionsTensor, batchSizeTensor, hiddenSizeTensor}};
        nvinfer1::IConcatenationLayer* concatenatedShape = ctx->network()->addConcatenation(tensors.data(), 3);
        return concatenatedShape->getOutput(0);
    };

    auto const getInitialInputValue
        = [&ctx, &initialStateShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx, node,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, 1})
                ->getOutput(0),
            initialStateShape());
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    LOG_VERBOSE("Entering Loop");
    // Scan over the S dimension of the input
    auto loop = ctx->network()->addLoop();
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, input, 0);
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add X(t)
    nvinfer1::ITensor* iterationInput = addRNNInput(ctx, node, loop, inputs, direction);
    ASSERT_NODE(iterationInput, "Failed to add RNN input.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(hiddenState, node);
    LOG_VERBOSE("Hidden state shape: " << hiddenState->getOutput(0)->getDimensions());

    // Compute intermediate(t) = (X(t) * W^T + H(t-1) * R^T + (Wb + Rb)).
    nvinfer1::ITensor* xtWT = ctx->network()
                                  ->addMatrixMultiply(*iterationInput, nvinfer1::MatrixOperation::kNONE, *weights,
                                      nvinfer1::MatrixOperation::kTRANSPOSE)
                                  ->getOutput(0);
    LOG_VERBOSE("X(t) * W^T -> " << xtWT->getDimensions());

    nvinfer1::ITensor* ht1RT = ctx->network()
                                   ->addMatrixMultiply(*hiddenState->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                       *recurrenceWeights, nvinfer1::MatrixOperation::kTRANSPOSE)
                                   ->getOutput(0);
    LOG_VERBOSE("H(t-1) * R^T -> " << ht1RT->getDimensions());

    nvinfer1::ITensor* intermediatet
        = ctx->network()->addElementWise(*xtWT, *ht1RT, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    if (combinedBias)
    {
        intermediatet = ctx->network()
                            ->addElementWise(*intermediatet, *combinedBias, nvinfer1::ElementWiseOperation::kSUM)
                            ->getOutput(0);
    }

    // H(t) = f(intermediate(t))
    nvinfer1::IActivationLayer* hAct
        = ctx->network()->addActivation(*addClip(ctx, intermediatet, clip), activations.at(0));
    hAct->setAlpha(activationAlphas.at(0));
    hAct->setBeta(activationBetas.at(0));
    nvinfer1::ITensor* Ht = hAct->getOutput(0);

    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape = ctx->network()
                                             ->addElementWise(*initialStateShape(),
                                                 *addConstant(ctx, std::vector<int>{numDirections, 1, 1},
                                                     ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})
                                                      ->getOutput(0),
                                                 nvinfer1::ElementWiseOperation::kDIV)
                                             ->getOutput(0);

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
    // Y = concatenation of all H(t) for each element of the sequence
    outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht,
        numDirections, inputs, direction == "reverse"));
    // Yh = last value of H(t)
    outputs.emplace_back(
        loop->addLoopOutput(*hiddenState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));

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
    ASSERT_NODE(roiDims.nbDims == 2 && roiDims.d[1] == 4,
        "Found incorrect dimensions for ROIs input! Rank of ROI input = " << roiDims.nbDims << ", roiDims.d[1] = "
                                                                             << roiDims.d[1] << ".",
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
    std::string const pluginVersion = "1";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back(
        "coordinate_transformation_mode", &coordinateTransformationMode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("mode", &mode, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("output_height", &outputHeight, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("output_width", &outputWidth, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("sampling_ratio", &samplingRatio, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("spatial_scale", &spatialScale, nvinfer1::PluginFieldType::kFLOAT32, 1);

    // Create plugin from registry
    auto const plugin = createPlugin(getNodeName(node), importPluginCreator(ctx, pluginName, pluginVersion), f);

    ASSERT_NODE(plugin != nullptr, "ROIAlign plugin was not found in the plugin registry!", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* const inputTensorsPtr[3] = {tensorPtr, roisPtr, batchIndicesPtr};
    auto* layer = ctx->network()->addPluginV2(inputTensorsPtr, 3, *plugin);
    ctx->registerLayer(layer, node);

    // ROIAlign requires nvinfer_vc_plugin when using VC.
    ctx->addUsedVCPluginLibrary(node, pluginName.c_str(), "nvinfer_vc_plugin");

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
    //     1. It has an optonal input `sequence_lens`
    //     2. The scan input/output axis are always set to 1
    const int32_t opset8Offset = ctx->getOpsetVersion() == 8 ? 1 : 0;
    if (opset8Offset == 1)
    {
        ASSERT_NODE(inputs.at(0).isNullTensor(), "TensorRT doesn't support sequence_lens input for this node!", node,
            nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        CHECK(convertAxis(axis, nvinfer1::Dims::MAX_DIMS));
    }

    for (auto& axis : scanOutputAxes)
    {
        CHECK(convertAxis(axis, nvinfer1::Dims::MAX_DIMS));
    }

    auto loop = ctx->network()->addLoop();
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
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs.at(i + opset8Offset), ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }
    ctx->registerLayer(stateVars.at(0), node);

    for (int32_t i = 0; i < nbScanInputs; ++i)
    {
        const int32_t index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput = loop->addIterator(convertToTensor(inputs.at(index + opset8Offset), ctx));
        scanInput->setAxis(scanInputAxes.at(i));
        scanInput->setReverse(scanInputDirections.at(i) == 1);
        ctx->registerTensor(TensorOrWeights{scanInput->getOutput(0)}, body.input(index).name());
    }

    // Loop Body. This is handled by dispatching to other op converters.
    std::vector<Status> errors{};
    CHECK(onnx2trt::parseGraph(ctx, body, errors));

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
        nodeOutputs.emplace_back(
            loop->addLoopOutput(*stateVars.at(i)->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
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
        nvinfer1::ILoopOutputLayer* trtScanOut = loop->addLoopOutput(scanOutput, scanDirection, scanAxis);
        trtScanOut->setInput(1, *tripLimit);
        nodeOutputs.emplace_back(trtScanOut->getOutput(0));
    }

    return {nodeOutputs};
}

DEFINE_BUILTIN_OP_IMPORTER(GridSample)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    CHECK(notInvalidType(inputs.at(1), {"BOOL", "UINT8"}));
    ASSERT_NODE(
        (inputs.size() == 2), "TRT expects two input tensors: grid and input", node, nodeIdx, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const inputRank = input.getDimensions().nbDims;
    ASSERT_NODE((inputRank > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor& grid = convertToTensor(inputs.at(1), ctx);
    int32_t const gridRank = grid.getDimensions().nbDims;
    ASSERT_NODE((gridRank > 0), "The grid tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT_NODE((gridRank == inputRank),
        "The input tensor and the grid tensor must have the same rank. Rank of grid tensor = "
            << gridRank << ", rank of input = " << inputRank << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Add grid sample layer
    nvinfer1::IGridSampleLayer* layer = ctx->network()->addGridSample(input, grid);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        interpolationMode = nvinfer1::ResizeMode::kNEAREST;
    }
    else if (mode == "bilinear")
    {
        interpolationMode = nvinfer1::ResizeMode::kLINEAR;
    }
    else if (mode == "bicubic")
    {
        interpolationMode = nvinfer1::ResizeMode::kCUBIC;
    }

    bool const alignCorners{attrs.get<int32_t>("align_corners", 0) == 1};

    ASSERT_NODE(
        layer->setSampleMode(sampleMode), "Failed to set sample mode!", node, nodeIdx, ErrorCode::kINVALID_VALUE);
    layer->setAlignCorners(alignCorners);
    layer->setInterpolationMode(interpolationMode);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterND)
{
    return addScatterLayer(ctx, node, nodeIdx, inputs, nvinfer1::ScatterMode::kND);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterElements)
{
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    CHECK(convertAxis(axis, nbDims));
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
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    auto* layer = ctx->network()->addShape(input);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);

    nvinfer1::ICastLayer* cast = ctx->network()->addCast(*(layer->getOutput(0)), DataType::kINT64);
    RETURN_FIRST_OUTPUT(cast, node, nodeIdx);
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
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
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
        auto isWeightsOrEmpty = [&inputs, &nbInputs](int32_t index)
        {
            return nbInputs <= index || inputs.at(index).is_weights();
        };

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
        axes = nbInputs > 3 ? ShapeTensor(ctx, inputs.at(3)) : iotaShapeVector(dims.size());
        // Doesn't support dynamic axes currently.
        ASSERT_NODE((axes.allValuesKnown()), "This version of TensorRT does not support dynamic axes.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
        ASSERT_NODE((starts.size() == axes.size()),
            "The shape of input starts misaligns with the shape of input axes. Shape of input starts = "
                << starts.size() << ", shape of input axes = " << axes.size() << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
        ASSERT_NODE(ends.size() == axes.size(),
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

    // Decode axes.
    // Also inspect whether axes form an "iota" sequence 0, 1, 2, ....
    bool isIota = true;
    int32_t j = 0;
    std::vector<int64_t> newAxes;
    newAxes.reserve(axes.size());

    for (int64_t axis : axes)
    {
        // "Accepted range is [-r, r-1] where r = rank(data)."
        int32_t const r = dims.size();
        ASSERT_NODE((-r <= axis && axis < r),
            "The range of axis must be in [-r, r-1], where r is the rank of input data. Provided axis = " << axis << ", r = "
                                                                                                 << r << ".",
            node, nodeIdx, ErrorCode::kINVALID_VALUE);
        // "Negative value means counting dimensions from the back."
        if (axis < 0)
        {
            axis += r;
        }
        newAxes.push_back(axis);
        isIota &= axis == j;
        ++j;
    }
    axes = ShapeTensor(1, std::move(newAxes));

    // Check for duplicate axes.
    ASSERT_NODE((std::unordered_set<int64_t>(axes.begin(), axes.end()).size() == static_cast<size_t>(axes.size())),
        "No duplicated axes are allowed.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    if (axes.size() < dims.size() || !isIota)
    {
        // Axes specify a subset of the dimensions, or out of order.
        // Convert starts/ends/steps to complete in-order form.
        ShapeTensor const subscripts{axesToInterlaceSubscripts(axes, dims.size())};
        starts = interlace(ctx, similar(ctx, dims, 0), starts, subscripts);
        ends = interlace(ctx, dims, ends, subscripts);
        steps = interlace(ctx, similar(ctx, dims, 1), steps, subscripts);
    }

    // ONNX has a bunch of rules for converting out of bounds starts/ends
    // indices into the actual indices to use.
    decodeOnnxStartsAndEnds(ctx, dims, steps, starts, ends);

    // TensorRT uses sizes of the output dimensions instead of ends.
    ShapeTensor sizes = computeSliceSizes(ctx, starts, ends, steps, dims);

    // Negative sizes signifies an empty slice, so clamp sizes to 0
    ShapeTensor const zeros = similar(ctx, dims, 0);
    sizes = max(ctx, zeros, sizes);

    nvinfer1::ISliceLayer* slice = addSlice(ctx, data, starts, sizes, steps);

    ctx->registerLayer(slice, node);

    RETURN_FIRST_OUTPUT(slice, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    auto& input = convertToTensor(inputs.at(0), ctx);
    auto* softmax = addSoftmax(ctx, node, input);
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *softmax, shapeOf(input));
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
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    // Input tensor is in NCHW format
    ASSERT_NODE((inputs.at(0).shape().nbDims == 4), "The input tensor must be in the NCHW format.", node, nodeIdx,
        ErrorCode::kINVALID_NODE);
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
    tensorPtr = firstShuffle->getOutput(0);

    // Reshape to {N, C * blockSize * blockSize, H / blockSize, W / blockSize}
    auto secondShapeDims = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShapeDims);
    tensorPtr = secondShuffle->getOutput(0);

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
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
    CHECK(convertAxis(axis, inputDims.size()));

    std::vector<int64_t> tmp(inputDims.size());
    std::iota(tmp.begin(), tmp.end(), 0);
    tmp[axis] = inputDims.size();
    ShapeTensor const subscripts = ShapeTensor(1, std::move(tmp));

    // "split : list of ints"
    // "length of each output"
    std::vector<int32_t> splitList;
    ShapeTensor sizes;
    ShapeTensor sizeSliceAxis;
    ShapeTensor splitSizesTensor;
    bool const hasSplitList = (ctx->getOpsetVersion() >= 13) ? (inputs.size() == 2) : attrs.count("split");
    if (hasSplitList)
    {
        // "Lengths of the parts can be specified using argument split."
        // In opset >= 13, split lengths are an optional input
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
        }
        // Pre-opset 13 split lengths are provided as an attribute
        else
        {
            splitList = attrs.get<std::vector<int32_t>>("split");
        }
        ASSERT_NODE((splitList.empty() || (splitList.size() == numOutputs)),
            "The number of the split attribute misaligns with the number of outputs. Number of split attributes = "
                << splitList.size() << ", number of outputs = " << numOutputs << ".",
            node, nodeIdx, ErrorCode::kINVALID_NODE);
    }
    else
    {
        // "Otherwise, the tensor is split to equal sized parts."
        sizeSliceAxis = floorDiv(ctx, gather(ctx, inputDims, shapeVector(axis)), shapeVector(numOutputs));
        sizes = interlace(ctx, inputDims, sizeSliceAxis, subscripts);
    }

    std::vector<TensorOrWeights> outputs;
    outputs.reserve(numOutputs);

    ShapeTensor const zeros = similar(ctx, inputDims, 0);
    ShapeTensor const ones = similar(ctx, inputDims, 1);
    ShapeTensor starts = zeros;
    ShapeTensor startSliceAxis = shapeVector(0);
    for (int32_t i = 0; i < (int32_t) numOutputs; ++i)
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

        nvinfer1::ISliceLayer* slice = addSlice(ctx, inputTensor, starts, sizes, ones);
        ctx->registerLayer(slice, node);
        outputs.emplace_back(slice->getOutput(0));
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
    std::vector<int32_t> axes;
    // In opset >= 13, axes are an optional input
    if (ctx->getOpsetVersion() >= 13)
    {
        if (inputs.size() == 2)
        {
            assertIsWeights(inputs.at(1), "Squeeze axes input must be an initializer!");
            // Map weights value to axes
            auto axesWeights = inputs.at(1).weights();
            int64_t* axesValues = static_cast<int64_t*>(axesWeights.values);
            for (size_t i = 0; i < axesWeights.count(); i++)
            {
                int64_t axesValue = axesValues[i];
                ASSERT_NODE(axesValue >= std::numeric_limits<int32_t>::min()
                        && axesValue <= std::numeric_limits<int32_t>::max(),
                    "Axes value truncated.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DATATYPE);
                axes.push_back(axesValues[i]);
            }
        }
    }
    // Pre-opset 13 axes are provided as an attribute
    else
    {
        OnnxAttrs attrs(node, ctx);
        if (attrs.count("axes"))
        {
            axes = attrs.get<std::vector<int>>("axes");
        }
    }

    // If axes are ommitted, squeeze all dimensions with values 1
    if (axes.size() == 0)
    {
        auto const shape = data.getDimensions();
        ASSERT_NODE(!isDynamic(shape),
            "Cannot infer squeeze dimensions from a dynamic shape! Please re-export your model with the Squeeze axes "
            "input set.",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_DYNAMIC);
        for (int32_t i = 0; i < shape.nbDims; i++)
        {
            if (shape.d[i] == 1)
            {
                axes.push_back(i);
            }
        }
    }

    int32_t rank = data.getDimensions().nbDims;
    for (auto& axis : axes)
    {
        CHECK(convertAxis(axis, rank));
    }

    // "squeezed : T
    // Reshaped tensor with same data as input."
    auto* squeezed = squeezeTensor(ctx, node, data, axes, true);

    ASSERT_NODE(squeezed, "Failed to squeeze tensor!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    return {{squeezed}};
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

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kTHRESHOLDED_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Tile)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
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
    tile->setMode(nvinfer1::SliceMode::kWRAP);

    RETURN_FIRST_OUTPUT(tile, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get("axis", -1);
    int32_t k{1};
    if (ctx->getOpsetVersion() < 10)
    {
        k = attrs.get<int>("k");
    }
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    CHECK(convertAxis(axis, nbDims));
    uint32_t axisMask = 1 << axis;

    bool needToExpandDims = (nbDims == 1);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{1};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT_NODE(tensorPtr, "Failed to unsqueeze input x.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(*tensorPtr, operation, k, axisMask);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    if (ctx->getOpsetVersion() >= 10)
    {
        ASSERT_NODE((inputs.size() == 2),
            "Expects two input tensors for opset >= 10: X and K. Current input size = " << inputs.size() << ".", node,
            nodeIdx, ErrorCode::kINVALID_NODE);
        nvinfer1::ITensor* kPtr = &convertToTensor(inputs.at(1), ctx);
        kPtr = convertToScalar(ctx, kPtr);
        layer->setInput(1, *kPtr);
    }
    ctx->registerLayer(layer, node);
    ASSERT_NODE(layer, "Failed to add TopK layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* values = layer->getOutput(0);
    nvinfer1::ITensor* indices = layer->getOutput(1);

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> axes{1};
        values = squeezeTensor(ctx, node, *values, axes);
        ASSERT_NODE(values, "Failed to squeeze the input values.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        indices = squeezeTensor(ctx, node, *indices, axes);
        ASSERT_NODE(indices, "Failed to squeeze the input indices.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    // TensorRT doesn't support int64 for TopK indices
    indices = castHelper(ctx, indices, DataType::kINT64);
    return {{values, indices}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node, ctx);
    int32_t ndim = input.shape().nbDims;
    ASSERT_NODE((ndim <= nvinfer1::Dims::MAX_DIMS),
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
    nvinfer1::ITensor& itensor = input.is_tensor() ? input.tensor() : convertToTensor(input, ctx);
    nvinfer1::ITensor* output_tensor = transposeTensor(ctx, node, itensor, perm);
    ASSERT_NODE(output_tensor, "Failed to transpose the input.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    return {{output_tensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(Trilu)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // Data Tensor
    using eOp = nvinfer1::ElementWiseOperation;
    auto* data = &convertToTensor(inputs.at(0), ctx);
    auto const nbDims = data->getDimensions().nbDims;
    ASSERT_NODE((nbDims >= 2),
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
        cols = &elementwiseHelper(ctx, node, nodeIdx, {cols, k}, eOp::kSUB).value().at(0).tensor();
    }

    // Unsqueeze to broadcast rows/cols if necessary during next elementwise operation.
    if (nbDims > 2)
    {
        std::vector<int32_t> batchDims(nbDims - 2);
        std::iota(batchDims.begin(), batchDims.end(), 0);
        rows = unsqueezeTensor(ctx, node, *rows, batchDims);
        cols = unsqueezeTensor(ctx, node, *cols, batchDims);
    }

    // For lower Trilus, use greaterOrEquals. For upper Trilus, use lessOrEquals
    bool const greater = upper == 0 ? true : false;
    auto* condition = greaterLessOrEqual(ctx, node, nodeIdx, rows, cols, greater);
    auto* result = ctx->network()->addSelect(*condition, *data, *zero);

    RETURN_FIRST_OUTPUT(result, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Unsqueeze)
{
    // "data : T
    // Original tensor"
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    std::vector<int32_t> axes;

    if (ctx->getOpsetVersion() >= 13)
    {
        // Per ONNX the 2nd input is mandatory starting at opset 13, but PyTorch
        // does not comply ATM so allow for single input.
        // https://github.com/onnx/onnx/blob/master/docs/Changelog.md#unsqueeze-13
        if (inputs.size() == 2)
        {
            ShapeTensor const axesInput{ctx, inputs.at(1)};
            ASSERT_NODE(axesInput.allValuesKnown(), "Axes input for unsqueeze operation should be a constant tensor.",
                node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            for (auto& a : axesInput)
            {
                axes.push_back(a);
            }
        }
    }

    if (axes.empty())
    {
        OnnxAttrs attrs(node, ctx);
        // "axes : list of ints (required)
        // List of integers indicating the dimensions to be inserted."
        axes = attrs.get<std::vector<int32_t>>("axes");
    }
    // "Negative value means counting dimensions from the back."
    int32_t const newSize = data.getDimensions().nbDims + axes.size();
    for (auto& axis : axes)
    {
        CHECK(convertAxis(axis, newSize));
    }
    // "expanded : T
    // Reshaped tensor with same data as input."
    auto* expanded = unsqueezeTensor(ctx, node, data, axes, true);

    ASSERT_NODE(expanded, "Failed to unsqueeze tensor!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    return {{expanded}};
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensor.getDimensions().nbDims;
    ASSERT_NODE((nbDims > 0), "The input tensor cannot be a scalar.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    auto mode = attrs.get<std::string>("mode", "nearest");

    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode
        = (mode == "linear" || mode == "bilinear") ? nvinfer1::ResizeMode::kLINEAR : nvinfer1::ResizeMode::kNEAREST;

    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ASSERT_NODE((inputs.size() == 2),
            "Operator Upsample requires exactly 2 inputs. Current input size = " << inputs.size() << ".", node, nodeIdx,
            ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        if (scales_input.is_weights())
        {
            // TRT-15340: Remove this and use else path when safety support nbDims == 1.
            ShapedWeights scales_weights = scales_input.weights();
            ASSERT_NODE((scales_weights.shape.nbDims == 1),
                "The scales input must be 1D. Current rank of scales input = " << scales_weights.shape.nbDims
                                                                                    << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            // Scale factors has batch dimension.
            ASSERT_NODE((scales_weights.count() == static_cast<size_t>(nbDims)),
                "The shape of the scales input must align with the dimensions of the input. Shape of scales input = "
                    << scales_weights.count() << ", dimension of input = " << nbDims << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            ASSERT_NODE((scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT),
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
                ASSERT_NODE(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3),
                    "This version of TensorRT only supports linear resizing on the outermost 3 dimensions", node,
                    nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            }
            layer->setScales(scale_factors.data(), nbDims);
        }
        else
        {
            nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, tensor, scales_input);
            nvinfer1::Dims const outDims = resizeShape->getDimensions();
            ASSERT_NODE((outDims.nbDims == 1),
                "The scales input must be 1D. Current rank of the scales input = " << outDims.nbDims << ".", node,
                nodeIdx, ErrorCode::kINVALID_NODE);
            // Scale factors has batch dimension.
            ASSERT_NODE((outDims.d[0] == nbDims),
                "The shape of the scales input must align with the dimensions of the input. Current shape of the "
                "scales input = "
                    << outDims.nbDims << ", dimension of the input = " << nbDims << ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            ASSERT_NODE((resizeShape->getType() == DataType::kINT32),
                "Resize output shape type must be integral. The actual type is "
                    + getTrtDtypeName(resizeShape->getType()) + ".",
                node, nodeIdx, ErrorCode::kINVALID_NODE);
            layer->setInput(1, *resizeShape);
        }
    }
    else
    {
        // TRT-15340: Adapt to use resizeShapeTensor instead when safety support nbDims == 1.
        ASSERT_NODE(
            attrs.count("scales"), "Attribute scales is missing.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE_ATTR);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ASSERT_NODE((static_cast<int32_t>(scales.size()) == nbDims),
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
            ASSERT_NODE(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3),
                "This version of TensorRT only supports linear resizing on the outermost 3 dimensions", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scale_factors.data(), nbDims);
    }
    ctx->registerLayer(layer, node);
    layer->setResizeMode(resizeMode);
    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
    layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(Where)
{
    CHECK(notInvalidType(inputs.at(1), {"UINT8"}));
    CHECK(notInvalidType(inputs.at(2), {"UINT8"}));
    nvinfer1::ITensor* condition = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* y = &convertToTensor(inputs.at(2), ctx);

    CHECK(broadcastTensors(ctx, x, y, condition));

    nvinfer1::Dims cDims = condition->getDimensions();
    nvinfer1::Dims xDims = x->getDimensions();
    nvinfer1::Dims yDims = y->getDimensions();

    ASSERT_NODE((cDims.nbDims == xDims.nbDims),
        "The rank of the condition input tensor must be the same of the input x tensor. Rank of the condition input "
        "tensor = "
            << cDims.nbDims << ", rank of the input x tensor = " << xDims.nbDims << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE((cDims.nbDims == yDims.nbDims),
        "The rank of the condition input tensor must be the same of the input y tensor. Rank of the condition input "
        "tensor = "
            << cDims.nbDims << ", rank of the input y tensor = " << yDims.nbDims << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    auto* layer = ctx->network()->addSelect(*condition, *x, *y);
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
    nvinfer1::PluginFieldCollection const* fieldNames, IImporterContext* ctx)
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
                MAKE_ERROR("Tensor type: "
                        + ::ONNX_NAMESPACE::TensorProto::DataType_Name(
                            static_cast<::ONNX_NAMESPACE::TensorProto::DataType>(tensor.type))
                        + " is unsupported",
                    ErrorCode::kUNSUPPORTED_NODE);
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
            MAKE_ERROR("Attributes of type: "
                    + ::ONNX_NAMESPACE::AttributeProto::AttributeType_Name(attrs.type(fieldName)) + " are unsupported",
                ErrorCode::kUNSUPPORTED_NODE);
        }
        fields.emplace_back(fieldName.c_str(), data, type, length);
    }
    return fields;
}

DEFINE_BUILTIN_OP_IMPORTER(Xor)
{
    return elementwiseHelper(ctx, node, nodeIdx, inputs, nvinfer1::ElementWiseOperation::kXOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Shrink)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);

    auto originalType = x->getType();
    ASSERT_NODE((originalType == DataType::kFLOAT || originalType == DataType::kHALF || originalType == DataType::kINT8
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
    CHECK(broadcastTensors(ctx, lambdTensor, x)); // align rank

    nvinfer1::ITensor* negLambdTensor
        = addConstant(ctx, std::vector<float>{-lambd}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    CHECK(broadcastTensors(ctx, negLambdTensor, x));

    nvinfer1::ITensor* biasTensor
        = addConstant(ctx, std::vector<float>{bias}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    CHECK(broadcastTensors(ctx, biasTensor, x));

    nvinfer1::ITensor* zeroTensor
        = addConstant(ctx, std::vector<float>{0.}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    CHECK(broadcastTensors(ctx, zeroTensor, x));

    // If x > lambd, y = x - bias; Otherwise, y = 0
    nvinfer1::ITensor* xGreaterThanLambd
        = &elementwiseHelper(ctx, node, nodeIdx, {x, lambdTensor}, nvinfer1::ElementWiseOperation::kGREATER)
               .value()
               .at(0)
               .tensor();
    nvinfer1::ITensor* xMinusBias
        = &elementwiseHelper(ctx, node, nodeIdx, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUB)
               .value()
               .at(0)
               .tensor();
    nvinfer1::ITensor* output = ctx->network()->addSelect(*xGreaterThanLambd, *xMinusBias, *zeroTensor)->getOutput(0);

    // If x < -lambd, y = x + bias;
    nvinfer1::ITensor* xLessThanMinusLambd
        = &elementwiseHelper(ctx, node, nodeIdx, {x, negLambdTensor}, nvinfer1::ElementWiseOperation::kLESS)
               .value()
               .at(0)
               .tensor();
    nvinfer1::ITensor* xAddBias
        = &elementwiseHelper(ctx, node, nodeIdx, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUM)
               .value()
               .at(0)
               .tensor();

    auto* layer = ctx->network()->addSelect(*xLessThanMinusLambd, *xAddBias, *output);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);

    // cast back to originalType
    return {{castHelper(ctx, layer->getOutput(0), originalType)}};
}

DEFINE_BUILTIN_OP_IMPORTER(HardSwish)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    ASSERT_NODE(
        (x->getType() == DataType::kFLOAT || x->getType() == DataType::kHALF || x->getType() == DataType::kINT8),
        "Only FLOAT, HALF or INT8 input is supported for the HardSwish operator in this version of TensorRT. "
        "The current type = "
            + getTrtDtypeName(x->getType()) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    // activationHelper does not s nodeIdx,upport const and constexpr (compile failed)
    float kALPHA{1.F / 6};
    float kBETA{0.5F};
    nvinfer1::ITensor* hardSigmoid
        = &activationHelper(ctx, node, nodeIdx, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &kALPHA, &kBETA)
               .value()
               .at(0)
               .tensor();

    return elementwiseHelper(ctx, node, nodeIdx, {x, hardSigmoid}, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(NonZero)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    auto const t = x->getType();
    ASSERT_NODE((t == DataType::kFLOAT || t == DataType::kHALF || t == DataType::kBF16 || t == DataType::kINT32
                    || t == DataType::kINT64 || t == DataType::kINT8 || t == DataType::kBOOL),
        "Only FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, INT8 or BOOL input is supported for the NonZero operator in "
        "this version of TensorRT. The current type is "
            + getTrtDtypeName(t) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    auto* layer = ctx->network()->addNonZero(*x);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    return {{castHelper(ctx, layer->getOutput(0), DataType::kINT64)}};
}

// Any ops that are not supported will attempt to import as plugins.
DEFINE_BUILTIN_OP_IMPORTER(FallbackPluginImporter)
{
    OnnxAttrs attrs(node, ctx);
    std::string const pluginName{node.op_type()};
    std::string const pluginVersion{attrs.get<std::string>("plugin_version", "1")};
    std::string const pluginNamespace{attrs.get<std::string>("plugin_namespace", "")};

    LOG_INFO("Searching for plugin: " << pluginName << ", plugin_version: " << pluginVersion
                                      << ", plugin_namespace: " << pluginNamespace);
    nvinfer1::IPluginCreator* creator = importPluginCreator(ctx, pluginName, pluginVersion, pluginNamespace);
    ASSERT_NODE(creator, "Plugin not found, are the plugin name, version, and namespace correct?", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::PluginFieldCollection const* fieldNames = creator->getFieldNames();
    // Field data needs to be type erased, we use fieldData for temporary allocations.
    StringMap<std::vector<uint8_t>> fieldData{};
    std::vector<nvinfer1::PluginField> fields = loadFields(fieldData, attrs, fieldNames, ctx);

    auto const plugin = createPlugin(getNodeName(node), creator, fields);
    ASSERT_NODE(plugin, "Could not create the plugin.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    std::vector<nvinfer1::ITensor*> pluginInputs{};
    for (auto& input : inputs)
    {
        pluginInputs.emplace_back(&convertToTensor(input, ctx));
    }
    LOG_INFO("Successfully created plugin: " << pluginName);
    auto* layer = ctx->network()->addPluginV2(pluginInputs.data(), pluginInputs.size(), *plugin);
    ctx->registerLayer(layer, node);
    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(LocalFunctionImporter)
{
    auto function = ctx->localFunctions().at(node.op_type());
    ASSERT_NODE(node.input().size() == function.input().size(),
        "LocalFunction has an unexpected number of inputs! Number of node inputs = "
            << node.input().size() << ", number of function inputs = " << function.input().size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_NODE);

    // Create a namescope local to the subgraph in the local function
    NameScope nameScope(*ctx);

    // We need to map local input names to the tensors from the outside scope. Keep track of
    // local input names in order to remove them later
    std::vector<std::string> localInputs;
    for (int32_t i = 0; i < function.input().size(); i++)
    {
        auto outsideScopeName = node.input(i);
        auto insideScopeName = function.input(i);
        if (outsideScopeName != insideScopeName)
        {
            if (ctx->tensors().count(insideScopeName))
            {
                LOG_WARNING("Found input: " << insideScopeName << " that does not correspond to an outside scope name. Behavior may be incorrect.");
                continue;
            }
            ctx->tensors().insert({insideScopeName, ctx->tensors().at(outsideScopeName)});
            localInputs.push_back(insideScopeName);
        }
        ASSERT_NODE(ctx->tensors().count(insideScopeName), "Could not find mapping of local function input!", node,
            nodeIdx, ErrorCode::kINVALID_NODE);
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
    ctx->localFunctionStack().push_back({node.op_type(), attrMap});

    // Log current stack of functions for debugging nested functions.
    auto prettyPrintFunctionStack = [ctx]()
    {
        std::stringstream stackStream;
        stackStream << "Function stack: [";
        size_t stackSize = ctx->localFunctionStack().size();
        for (size_t i = 0; i < stackSize; i++)
        {
            stackStream << ctx->localFunctionStack()[i].first;
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
        CHECK(onnx2trt::parseNode(ctx, node, nodeIdx));
    }

    // Create output vector
    std::vector<TensorOrWeights> outputs;
    for (auto const& output : function.output())
    {
        ASSERT_NODE(
            ctx->tensors().count(output), "Could not find output tensor!", node, nodeIdx, ErrorCode::kINVALID_NODE);
        outputs.push_back(TensorOrWeights(ctx->tensors().at(output)));
    }

    // Remove all localInputs as we exit the local function scope, and pop the current function name from the stack
    for (auto& name : localInputs)
    {
        ctx->tensors().erase(name);
    }
    ctx->localFunctionStack().pop_back();

    return outputs;
}

// INetwork Serialization importer functions - TODO: Move to it's own file?

DEFINE_BUILTIN_OP_IMPORTER(TRT_Scale)
{
    ASSERT_NODE((inputs.size() >= 1), "Input is required.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE(
        (inputs.at(0).is_tensor()), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    if (inputs.size() >= 2)
    {
        ASSERT_NODE((inputs.at(1).is_weights()), "The second input must be an initializer.", node, nodeIdx,
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
        ASSERT_NODE((inputs.at(counter).is_weights()), "The scale input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        scale = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("shift"))
    {
        ASSERT_NODE((inputs.at(counter).is_weights()), "The shift input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        shift = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("power"))
    {
        ASSERT_NODE((inputs.at(counter).is_weights()), "The power input must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        power = inputs.at(counter++).weights();
    }

    nvinfer1::IScaleLayer* layer = ctx->network()->addScale(input, mode, shift, scale, power);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");
    bool zeroIsPlaceholder = attrs.get<bool>("zero_is_placeholder");

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT_NODE(inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_TopK_Min)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    ASSERT_NODE(inputs.at(1).is_weights(), "The second input must be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kWeights = inputs.at(1).weights();
    int k = *static_cast<int*>(kWeights.values);

    int32_t axes = 1 << (attrs.get<int32_t>("axis"));

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(input, nvinfer1::TopKOperation::kMIN, k, axes);
    ctx->registerLayer(layer, node);

    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MatMul)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input0 = inputs.at(0).tensor();
    auto& input1 = inputs.at(1).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::MatrixOperation op0 = attrs.get<nvinfer1::MatrixOperation>("op_0");
    nvinfer1::MatrixOperation op1 = attrs.get<nvinfer1::MatrixOperation>("op_1");

    nvinfer1::IMatrixMultiplyLayer* layer = ctx->network()->addMatrixMultiply(input0, op0, input1, op1);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

typedef std::function<void(int32_t, nvinfer1::RNNGateType, nvinfer1::Weights)> RNNWeightsAdder;

bool addRNNv2Weights(RNNWeightsAdder adder, int32_t layerNb, std::vector<nvinfer1::RNNGateType> const& gates,
    std::vector<TensorOrWeights>& inputs, int32_t& counter)
{
    for (nvinfer1::RNNGateType gate : gates)
    {
        if (!inputs.at(counter).is_weights())
            return false;
        auto& weights = inputs.at(counter++).weights();
        adder(layerNb, gate, weights);
    }
    return true;
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RNNv2)
{
    OnnxAttrs attrs(node, ctx);

    int32_t layerCount = attrs.get<int32_t>("layer_count");
    int32_t hiddenSize = attrs.get<int32_t>("hidden_size");
    int32_t maxSeqLen = attrs.get<int32_t>("max_seq_length");
    nvinfer1::RNNOperation op = attrs.get<nvinfer1::RNNOperation>("rnn_op");
    nvinfer1::RNNInputMode inputMode = attrs.get<nvinfer1::RNNInputMode>("input_mode");
    nvinfer1::RNNDirection direction = attrs.get<nvinfer1::RNNDirection>("direction");

    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    int32_t counter = 1;
    nvinfer1::IRNNv2Layer* layer = ctx->network()->addRNNv2(input, layerCount, hiddenSize, maxSeqLen, op);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    layer->setInputMode(inputMode);
    layer->setDirection(direction);

    if (attrs.get<bool>("has_hidden_state"))
    {
        ASSERT_NODE(inputs.at(counter).is_tensor(), "The input hidden_state must be a tensor.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setHiddenState(inputs.at(counter++).tensor());
    }
    if (op == nvinfer1::RNNOperation::kLSTM && attrs.get<bool>("has_cell_state", false))
    {
        ASSERT_NODE(inputs.at(counter).is_tensor(), "The input cell_state must be a tensor.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setCellState(inputs.at(counter++).tensor());
    }
    if (attrs.get<bool>("has_seq_lengths"))
    {
        ASSERT_NODE(inputs.at(counter).is_tensor(), "The input seq_lengths must be a tensor.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setSequenceLengths(inputs.at(counter++).tensor());
    }

    int32_t nbWeights = (direction == nvinfer1::RNNDirection::kBIDIRECTION ? 2 : 1) * layerCount;
    const int32_t K = direction == nvinfer1::RNNDirection::kUNIDIRECTION ? 1 : 2;
    std::vector<nvinfer1::RNNGateType> gates;
    switch (op)
    {
    case nvinfer1::RNNOperation::kRELU:
    case nvinfer1::RNNOperation::kTANH:
        gates = std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT});
        break;
    case nvinfer1::RNNOperation::kLSTM:
        gates = std::vector<nvinfer1::RNNGateType>({nvinfer1::RNNGateType::kINPUT, nvinfer1::RNNGateType::kOUTPUT,
            nvinfer1::RNNGateType::kFORGET, nvinfer1::RNNGateType::kCELL});
        break;
    case nvinfer1::RNNOperation::kGRU:
        gates = std::vector<nvinfer1::RNNGateType>(
            {nvinfer1::RNNGateType::kUPDATE, nvinfer1::RNNGateType::kRESET, nvinfer1::RNNGateType::kHIDDEN});
        break;
    }

    RNNWeightsAdder weightsAdder = [&layer](int32_t n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentWeightsAdder = [&layer](int32_t n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, false, weights);
    };
    RNNWeightsAdder biasAdder = [&layer](int32_t n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentBiasAdder = [&layer](int32_t n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, false, weights);
    };

    for (int32_t n = 0; n < nbWeights; ++n)
    {
        if (n >= K || inputMode == nvinfer1::RNNInputMode::kLINEAR)
        {
            ASSERT_NODE(addRNNv2Weights(weightsAdder, n, gates, inputs, counter),
                "Failed to add weights to the RNN layer.", node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        ASSERT_NODE(addRNNv2Weights(recurrentWeightsAdder, n, gates, inputs, counter),
            "Failed to add recurrent weights to the RNN layer.", node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT_NODE(addRNNv2Weights(biasAdder, n, gates, inputs, counter), "Failed to add bias to the RNN layer.", node,
            nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT_NODE(addRNNv2Weights(recurrentBiasAdder, n, gates, inputs, counter),
            "Failed to add recurrent bias to the RNN layer.", node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);
    }

    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RaggedSoftmax)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();
    auto& bounds = inputs.at(1).tensor();

    nvinfer1::IRaggedSoftMaxLayer* layer = ctx->network()->addRaggedSoftMax(input, bounds);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

// This op is to be removed when deprecated IFullyConnectedLayer is removed
DEFINE_BUILTIN_OP_IMPORTER(TRT_FullyConnected)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int32_t nbChannels = attrs.get<int32_t>("channels");

    ASSERT_NODE(inputs.at(1).is_weights(), "The input kernel must be an initializer.", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kernelWeights = inputs.at(1).weights();

    ShapedWeights biasWeights = ShapedWeights::empty(kernelWeights.type);
    if (inputs.size() == 3)
    {
        ASSERT_NODE(inputs.at(2).is_weights(), "The input bias must be an initializer.", node, nodeIdx,
            nvonnxparser::ErrorCode::kINVALID_NODE);
        biasWeights = inputs.at(2).weights();
    }

    nvinfer1::IFullyConnectedLayer* layer
        = ctx->network()->addFullyConnected(input, nbChannels, kernelWeights, biasWeights);
    ctx->registerLayer(layer, node);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int32_t nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;
    nvinfer1::Dims kernelSize = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding(true);
    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, exclude_padding);
    float blend = attrs.get<float>("blend");

    nvinfer1::IPoolingLayer* layer
        = ctx->network()->addPoolingNd(input, nvinfer1::PoolingType::kMAX_AVERAGE_BLEND, kernelSize);
    ASSERT_NODE(layer, "Failed to create a Pooling layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(exclude_padding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    layer->setBlendFactor(blend);

    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

#if ENABLE_STD_PLUGIN
DEFINE_BUILTIN_OP_IMPORTER(TRT_PluginV2)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        ASSERT_NODE(input.is_tensor(), "The input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
        tensors.push_back(&input.tensor());
    }
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IPluginRegistry* registry = getPluginRegistry();

    std::string name = attrs.get<std::string>("name");
    std::string version = attrs.get<std::string>("version");
    std::string nspace = attrs.get<std::string>("namespace");
    std::string buffer = attrs.get<std::string>("data");

    nvinfer1::IPluginCreator* creator = registry->getPluginCreator(name.c_str(), version.c_str(), nspace.c_str());
    ASSERT_NODE(creator, "Plugin not found, are the plugin name, version, and namespace correct?", node, nodeIdx,
        nvonnxparser::ErrorCode::kINVALID_NODE);

    auto const plugin = creator->deserializePlugin("", buffer.data(), buffer.size());

    nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
    ctx->registerLayer(layer, node);
    RETURN_ALL_OUTPUTS(layer, node, nodeIdx);
}
#endif // ENABLE_STD_PLUGIN

DEFINE_BUILTIN_OP_IMPORTER(TRT_Gather)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE(
        inputs.at(1).is_tensor(), "The second input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& data = inputs.at(0).tensor();
    auto& indices = inputs.at(1).tensor();
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbElementWiseDims = attrs.get<int32_t>("nbElementWiseDims", 0);
    int32_t r = data.getDimensions().nbDims;

    ASSERT_NODE((indices.getType() == DataType::kINT32),
        "This version of TensorRT only supports INT32 input indices. The current indices type = "
            + getTrtDtypeName(indices.getType()) + ".",
        node, nodeIdx, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT_NODE((r >= 1), "0D input data is not allowed.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    ASSERT_NODE((-r <= axis && axis < r),
        "The attribute axis should be in range [-r, r-1], where r is the rank of the input. Provided r = " << r << ", axis = "
                                                                                                  << axis << ".",
        node, nodeIdx, nvonnxparser::ErrorCode::kINVALID_NODE);

    if (axis < 0)
    {
        axis += r;
    }

    nvinfer1::IGatherLayer* layer = ctx->network()->addGather(data, indices, axis);
    ASSERT_NODE(layer, "Failed to create layer", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node);
    layer->setNbElementWiseDims(nbElementWiseDims);
    RETURN_FIRST_OUTPUT(layer, node, nodeIdx);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Slice)
{
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::ISliceLayer* layer;
    // If only one input, then, start, size, stride are all attributes.
    if (inputs.size() == 1)
    {
        OnnxAttrs attrs(node, ctx);
        auto start = attrs.get<nvinfer1::Dims>("start");
        auto size = attrs.get<nvinfer1::Dims>("size");
        auto stride = attrs.get<nvinfer1::Dims>("stride");
        layer = ctx->network()->addSlice(input, start, size, stride);
    }
    else
    {
        // start, size, stride are all inputs
        ASSERT_NODE((inputs.size() == 4),
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
    ASSERT_NODE(inputs.at(0).is_tensor(), "The first input must be a tensor.", node, nodeIdx, ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = ctx->network()->addResize(input);
    ctx->registerLayer(layer, node);

    OnnxAttrs attrs(node, ctx);
    auto const mode = attrs.get<nvinfer1::ResizeMode>("mode");
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
            ASSERT_NODE((scales.size() > 0), "Attribute scales is missing.", node, nodeIdx, ErrorCode::kINVALID_NODE);
            layer->setScales(&scales[0], scales.size());
        }
    }
    else
    {
        ASSERT_NODE((inputs.at(1).is_tensor()), "The output dimension input must be a tensor.", node, nodeIdx,
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
