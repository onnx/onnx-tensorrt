/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Include assert.h prior to the below WAR to ensure that assert()
// definition obeys the NDEBUG define
#include <assert.h>

// Workaround for NDEBUG causing functional
// differences in ONNX / protobuf code on aarch64 platforms which leads
// to corruption.
#if defined(__aarch64__) && defined(__linux__) && defined(NDEBUG)
#undef NDEBUG
#define REDEFINE_NDEBUG
#endif

#include <onnx/onnx_pb.h>

#if defined(REDEFINE_NDEBUG)
#define NDEBUG
#undef REDEFINE_NDEBUG
#endif

#include "ConditionalHelpers.hpp"
#include "LoopHelpers.hpp"
#include "ModelImporter.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "OnnxAttrs.hpp"
#include "RNNHelpers.hpp"
#include "ShapeTensor.hpp"
#include "builtin_op_importers.hpp"
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

string_map<NodeImporter>& getBuiltinOpImporterMap()
{
    static string_map<NodeImporter> builtin_op_importers;
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
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op)                                                                                 \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);         \
    static bool const op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr && "Input layer is null.", ErrorCode::kUNSUPPORTED_NODE);                                                               \
        return {{layer_ptr->getOutput(0)}};                                                                            \
    } while (0)

#define RETURN_IDENTITY(input)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        TensorOrWeights output = identity(ctx, input);                                                                 \
        ASSERT(output && "Failed to add an identity layer.", ErrorCode::kUNSUPPORTED_NODE);                                                                  \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_ALL_OUTPUTS(layer)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr && "The input layer is null.", ErrorCode::kUNSUPPORTED_NODE);                                 \
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
        msg << " Try applying constant folding on the model using Polygraphy: https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy/examples/cli/surgeon/02_folding_constants";
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
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acos)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kACOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acosh)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kACOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(And)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kAND);
}

DEFINE_BUILTIN_OP_IMPORTER(Asin)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kASIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asinh)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kASINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Atan)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kATAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Atanh)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kATANH);
}

DEFINE_BUILTIN_OP_IMPORTER(Add)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMax)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMin)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool)
{
    return poolingHelper(ctx, node, inputs, nvinfer1::PoolingType::kAVERAGE);
}

NodeImportResult batchnormFallback(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)
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
    float eps = attrs.get<float>("epsilon", 1e-5f);

    nvinfer1::Dims scalarShape{rank};
    std::fill(scalarShape.d, scalarShape.d + scalarShape.nbDims, 1);
    nvinfer1::ITensor* epsilon
        = addConstantScalar(ctx, eps, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarShape)->getOutput(0);

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

    ctx->registerLayer(layer, getNodeName(node));

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
    ASSERT(
        (inputs.at(1).shape().nbDims == 1) && "The shape of the scale input must be (C, )", ErrorCode::kINVALID_NODE);
    ASSERT((inputs.at(2).shape().nbDims == 1) && "The shape of the bias input must be (C, )", ErrorCode::kINVALID_NODE);
    ASSERT((inputs.at(3).shape().nbDims == 1) && "The shape of the mean input must be (C, )", ErrorCode::kINVALID_NODE);
    ASSERT((inputs.at(4).shape().nbDims == 1) && "The shape of the var input must be (C, )", ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);
    auto const isTraining = attrs.get<int32_t>("training_mode", 0);
    ASSERT(!isTraining
            && "This version of TensorRT does not support training_mode == 1 in BatchNormalization.",
        ErrorCode::kUNSUPPORTED_NODE);

    bool const allInputsWeights = inputs.at(1).is_weights() && inputs.at(2).is_weights() && inputs.at(3).is_weights()
        && inputs.at(4).is_weights();

    if (!allInputsWeights)
    {
        return batchnormFallback(ctx, node, inputs);
    }

    auto const scale = inputs.at(1).weights();
    auto const bias = inputs.at(2).weights();
    auto const mean = inputs.at(3).weights();
    auto const variance = inputs.at(4).weights();

    // get the values of constant inputs and cast them to float32
    auto const getValuesFP32 = [&](ShapedWeights const& w) -> float const* {
        return (w.type == ::ONNX_NAMESPACE::TensorProto::FLOAT) ? static_cast<float*>(w.values)
                                                                : convertFP16Data(w.values, w.shape, ctx);
    };

    float const* scaleValues = getValuesFP32(scale);
    float const* biasValues = getValuesFP32(bias);
    float const* meanValues = getValuesFP32(mean);
    float const* varianceValues = getValuesFP32(variance);

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

    return scaleHelper(ctx, node, *tensorPtr, nvinfer1::ScaleMode::kCHANNEL, combinedBias, combinedScale,
        ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT), combinedBias.getName(), combinedScale.getName());
}

DEFINE_BUILTIN_OP_IMPORTER(Cast)
{
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    // Get data type to cast to.
    DataType dtype = tensor.getType();
    auto onnxType = attrs.get<int32_t>("to");
    DataType newType{DataType::kFLOAT};
    ASSERT(
        convertDtype(onnxType, &newType) && "Unsupported data type for the Cast operator!", ErrorCode::kINVALID_NODE);
    LOG_VERBOSE("Casting to type: " << newType);
    // UINT8 type casting is only supported from FP types.
    if (dtype == DataType::kUINT8)
    {
        ASSERT(newType == DataType::kFLOAT || newType == DataType::kHALF, ErrorCode::kUNSUPPORTED_NODE);
    }
    if (newType == DataType::kUINT8)
    {
        ASSERT(dtype == DataType::kFLOAT || dtype == DataType::kHALF, ErrorCode::kUNSUPPORTED_NODE);
    }

    // Add the layer.
    nvinfer1::IIdentityLayer* layer = ctx->network()->addIdentity(tensor);
    layer->setOutputType(0, newType);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kCEIL);
}

DEFINE_BUILTIN_OP_IMPORTER(Celu)
{

    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    using eOpInstuctor = std::tuple<int, int, const nvinfer1::ElementWiseOperation>;

    ASSERT( (!inputs.empty()) && "Inputs vector is empty.", ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    TensorOrWeights input = inputs.at(0);
    float alpha = attrs.get<float>("alpha", 1.0);

    TensorOrWeights weightsOfZero = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0,{}});
    ShapedWeights weightsOfOnes = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0,{}});
    std::vector<float> ones{1};
    std::memcpy(weightsOfOnes.values, ones.data(), weightsOfOnes.count() * sizeof(float));
    ShapedWeights weightsOfAlpha = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0,{}});
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
        ASSERT(tensor_ptr->getDimensions().nbDims == maxNbDims && "Failed to broadcast tensors elementwise!",
            ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }

    // Calculate (x/alpha)
    std::vector<TensorOrWeights> tempInputs{newInputs[0], newInputs[3]};
    ASSERT(elementwiseCheck(tempInputs, eOp::kDIV) && "Elementwise layer does not support the given inputs and operator.", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* combined = inputTensors.at(0);
    auto* divLayer = ctx->network()->addElementWise(*combined, *inputTensors.at(3), eOp::kDIV);
    ctx->registerLayer(divLayer, getNodeName(node));
    ASSERT(divLayer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);
    combined = divLayer->getOutput(0);

    // Calculate exp(x/alpha) -> 4
    nvinfer1::IUnaryLayer* uLayer = ctx->network()->addUnary(*combined, uOp::kEXP);
    ctx->registerLayer(uLayer, getNodeName(node));
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
        ASSERT(
            (elementwiseCheck(tempInputs, op)) && "Elementwise layer does not support the given inputs and operator.",
            ErrorCode::kUNSUPPORTED_NODE);
        ASSERT((firstTensor->getDimensions().nbDims == secondTensor->getDimensions().nbDims)
                && "The number of dimensions should remain the same adding inputs.",
            ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = ctx->network()->addElementWise(*firstTensor, *secondTensor, op);
        ctx->registerLayer(layer, getNodeName(node));
        ASSERT(layer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);
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
    if (ctx->getOpsetVersion() >= 11)
    {
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
    }
    else
    {
        alpha = static_cast<ScalarType>(attrs.get("min", alpha));
        alphaT = addConstantScalar(ctx, alpha, onnxType)->getOutput(0);
        beta = static_cast<ScalarType>(attrs.get("max", beta));
        betaT = addConstantScalar(ctx, beta, onnxType)->getOutput(0);
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
    bool elementwiseClip = inputs.at(0).isInt32();
    for (size_t i = 1; i < numInputs; i++)
    {
        elementwiseClip |= inputs.at(i).is_tensor();
    }
    if (elementwiseClip)
    {
        auto type = convertToTensor(inputs.at(0), ctx).getType();
        ASSERT((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT32)
                && "This version of TensorRT only supports FLOAT or INT32 inputs for Clip!",
            ErrorCode::kUNSUPPORTED_NODE);
        if (type == DataType::kFLOAT || type == DataType::kHALF)
        {
            return elementwiseClipHelper<float>(ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::FLOAT);
        }
        else
        {
            return elementwiseClipHelper<int32_t>(ctx, node, inputs, numInputs, ::ONNX_NAMESPACE::TensorProto::INT32);
        }
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
            ASSERT(inputs.at(1).is_weights() && "Clip min value must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
            auto min = inputs.at(1).weights();
            bool fp16 = inputs.at(1).isFp16();
            alpha = getSingleValueAsFloat(min.values, /*isFp16*/ fp16);
        }
        // Handle both "min" and "max" node inputs
        else if (numInputs == 3)
        {
            // "min" can be optional if "max" is specified. Check for this case here
            if (!inputs.at(1).isNullTensor())
            {
                ASSERT(inputs.at(1).is_weights() && "Clip min value must be an initializer!",
                    ErrorCode::kUNSUPPORTED_NODE);
                auto min = inputs.at(1).weights();
                bool fp16 = inputs.at(1).isFp16();
                alpha = getSingleValueAsFloat(min.values, /*isFp16*/ fp16);
            }

            if (!inputs.at(2).isNullTensor())
            {
                ASSERT(inputs.at(2).is_weights() && "Clip max value must be an initializer!",
                    ErrorCode::kUNSUPPORTED_NODE);
                auto max = inputs.at(2).weights();
                bool fp16 = inputs.at(2).isFp16();
                beta = getSingleValueAsFloat(max.values, /*isFp16*/ fp16);
            }
        }
    }
    else
    {
        alpha = attrs.get("min", std::numeric_limits<float>::lowest());
        beta = attrs.get("max", std::numeric_limits<float>::max());
    }

    nvinfer1::ITensor* clipOut
        = &activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kCLIP, &alpha, &beta).value().at(0).tensor();

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
    ctx->registerLayer(layer, getNodeName(node));
    ASSERT(layer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);
    layer->setAxis(axis);
    RETURN_FIRST_OUTPUT(layer);
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
        RETURN_FIRST_OUTPUT(layer);
    }

    ASSERT((!attrs.count("sparse_value")) && (!attrs.count("value_string")) && (!attrs.count("value_strings"))
        && "This version of TensorRT does not support the sparse_value, value_string and value_strings attributes.", ErrorCode::kUNSUPPORTED_NODE);

    if (ctx->getOpsetVersion() >=12)
    {
        if (attrs.count("value_float"))
        {
            ShapedWeights convertedWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0,{}});
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
            ShapedWeights convertedWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, {0,{}});
            float value = attrs.get<float>("value_int");
            std::memcpy(convertedWeights.values, &value, convertedWeights.count() * sizeof(int));
            return {{convertedWeights}};
        }

        if (attrs.count("value_ints"))
        {
            std::vector<float> values = attrs.get<std::vector<float>>("value_ints");
            int32_t valueSize = values.size();
            ShapedWeights convertedWeights
                = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, {1, {valueSize}});
            std::memcpy(convertedWeights.values, values.data(), convertedWeights.count() * sizeof(float));
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
    CHECK(notInvalidType(valueWeights, {"UINT8"}));
    nvinfer1::ITensor* value = &convertToTensor(valueWeights, ctx);
    return {{constantOfShape(ctx, node, value, shape)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv)
{
    if (inputs.at(1).is_tensor() || (inputs.size() > 2 && inputs.at(2).is_tensor()))
    {
        // Handle dynamic weights convolution
        return convMultiInput(ctx, node, inputs);
    }

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    auto kernelWeights = inputs.at(1).weights();

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    LOG_VERBOSE("Convolution input dimensions: " << dims);
    ASSERT(dims.nbDims >= 0 && "TensorRT could not compute output dimensions of Conv", ErrorCode::kUNSUPPORTED_NODE);

    bool const needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }
    if (kernelWeights.shape.nbDims == 3)
    {
        kernelWeights.shape.nbDims = 4;
        kernelWeights.shape.d[3] = 1;
    }

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT((nbSpatialDims == kernelWeights.shape.nbDims - 2)
            && "The number of spatial dimensions and the kernel shape doesn't match up for the Conv operator.",
        ErrorCode::kUNSUPPORTED_NODE);

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
        ASSERT( (shapedBiasWeights.shape.nbDims == 1) && "The bias tensor is required to be 1D.", ErrorCode::kINVALID_NODE);
        ASSERT( (shapedBiasWeights.shape.d[0] == kernelWeights.shape.d[0]) && "The shape of the bias tensor misaligns with the weight tensor.", ErrorCode::kINVALID_NODE);
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
        ASSERT((kernelSize.d[nbSpatialDims - i] == kernelWeights.shape.d[kernelWeights.shape.nbDims - i])
                && "The size of spatial dimension and the size of kernel shape are not equal for the Conv operator.",
            ErrorCode::kUNSUPPORTED_NODE);
    }

    int32_t nchan = dims.d[1];
    int32_t noutput = kernelWeights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*tensorPtr, noutput, kernelSize, kernelWeights, bias_weights);

    ASSERT(layer && "Failed to add a convolution layer.", ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int32_t ngroup = attrs.get("group", 1);
    ASSERT((nchan == -1 || kernelWeights.shape.d[1] * ngroup == nchan)
            && "Kernel weight dimension failed to broadcast to input.",
        ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    // Register layer name as well as kernel weights and bias weights (if any)
    ctx->registerLayer(layer, getNodeName(node));
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
        ASSERT(tensorPtr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }

    LOG_VERBOSE("Using kernel: " << kernelSize << ", strides: " << strides << ", prepadding: " << begPadding
        << ", postpadding: " << endPadding << ", dilations: " << dilations << ", numOutputs: " << noutput);
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

    ASSERT(inputs.size() >= 2 && "deconvolution require at least 2 inputs.", ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* kernelTensorPtr = inputs.at(1).is_tensor() ? &convertToTensor(inputs.at(1), ctx) : nullptr;
    nvinfer1::ITensor* biasTensorPtr
        = inputs.size() > 2 && inputs.at(2).is_tensor() ? &convertToTensor(inputs.at(2), ctx) : nullptr;

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    // Deconvolution input must be at least 3D and at most 5D.
    ASSERT(dims.nbDims >= 3 && dims.nbDims <= 5 && "TensorRT only supports 1D, 2D or 3D deconvolutions!",
        ErrorCode::kUNSUPPORTED_NODE);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernelShape = inputs.at(1).shape();

    bool needReshapeBack = NCWtoNCHW(tensorPtr, dims);
    NCWtoNCHW(kernelTensorPtr, kernelShape);

    int32_t const nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT((nbSpatialDims == kernelShape.nbDims - 2)
            && "The number of spatial dimensions and the kernel shape doesn't match up",
        ErrorCode::kUNSUPPORTED_NODE);

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
        ASSERT(
            shapedBiasWeights.shape.nbDims == 1 && "The bias tensor is required to be 1D.", ErrorCode::kINVALID_NODE);
        ASSERT((shapedBiasWeights.shape.d[0] == noutput)
                && "The shape of the bias tensor does not align with the shape of the output.",
            ErrorCode::kINVALID_NODE);
        staticBiasWeights = shapedBiasWeights;
    }
    else
    {
        staticBiasWeights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);
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
        ASSERT((kernelSize.d[nbSpatialDims - i] == kernelShape.d[kernelShape.nbDims - i])
                && "Attribute kernel_shape misaligns with the dimensions of the weight tensor.",
            ErrorCode::kUNSUPPORTED_NODE);
    }

    // Set padding. ONNX ConvTranspose supports many different padding modes. Order of priority for padding:
    // 1. Output shape is specified - calculate expected pre and post padding.
    // 2. AUTO_PAD != NOTSET: ignore all other padding values and set padding mode with layer->setPaddingMode.
    //    Pad the resulting output vector with values from output_padding
    // 3. Use specified "pads" values from the node. Pad the resulting output vector with values from output_padding

    auto autoPadMode = attrs.get("auto_pad", std::string("NOTSET"));
    if (attrs.count("output_shape") && autoPadMode == std::string("NOTSET"))
    {
        outputShape = attrs.get<nvinfer1::Dims>("output_shape");

        // This function takes references to begPadding, endPadding and outputPadding and will update them with correct values
        generatePadding(dims, outputShape, kernelSize, strides, dilations, nbSpatialDims, begPadding, endPadding,
            outputPadding, paddingMode);

        // NOTE: it is possible for generatePadding to produce negative values for pre and post padding, which usually
        // happens when output_shape is provided but output_padding is not. Any negative values generated for
        // post-padding can be translated into outputPadding to pad the output tensor post deconvolution. Any negative
        // values for pre-padding are unsupported.

        for (int32_t i = 0; i < nbSpatialDims; i++)
        {
            ASSERT(
                begPadding.d[i] >= 0 && "TensorRT does not support negative pre-padding in the ConvTranspose operator!",
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

    auto const emptyBiasWeights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto const kernelWeights
        = kernelTensorPtr ? nvinfer1::Weights{DataType::kFLOAT, nullptr, 0} : inputs.at(1).weights();
    const auto biasWeights
        = biasTensorPtr ? nvinfer1::Weights{DataType::kFLOAT, nullptr, 0} : staticBiasWeights;
    // Create a deconvolution layer and set known attributes - strides,ngroups, and dilations
    // If there is still output padding, remove the bias weights. Bias will be added below.
    auto* layer = ctx->network()->addDeconvolutionNd(
        *tensorPtr, noutput, kernelSize, kernelWeights, hasOutputPadding ? emptyBiasWeights : biasWeights);
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
    ctx->registerLayer(layer, getNodeName(node));
    tensorPtr = layer->getOutput(0);
    dims = tensorPtr->getDimensions();

    // There is still output padding. Add a padding layer to handle it.
    if (hasOutputPadding)
    {
        LOG_VERBOSE("Padding output deconvolution tensor with: " << outputPadding);

        // Add padding layer
        nvinfer1::ITensor* start{};
        nvinfer1::ITensor* totalPadding{};
        std::vector<int32_t> combinePadding{};
        for (int32_t i = 0; i < outputPadding.nbDims; ++i)
        {
            combinePadding.insert(combinePadding.begin(), 0);
            combinePadding.push_back(outputPadding.d[i]);
        }
        ASSERT(
            convertOnnxPadding(ctx, dims.nbDims, combinePadding, start, totalPadding) && "Failed to convert padding!",
            ErrorCode::kUNSUPPORTED_NODE);
        auto const size
            = ctx->network()
                  ->addElementWise(shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM)
                  ->getOutput(0);
        auto const stride = makeDims(dims.nbDims, 1);
        auto const& dummy = stride;
        auto* sliceLayer = ctx->network()->addSlice(*tensorPtr, dummy, dummy, stride);
        ASSERT(sliceLayer && "Could not create padding layer", ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT(tensorPtr && "Failed to squeeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Cos)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kCOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Cosh)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kCOSH);
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
        std::vector<nvinfer1::ITensor*> concatTensors = reverse == 1 ? std::vector<nvinfer1::ITensor*>{input, zero} : std::vector<nvinfer1::ITensor*>{zero, input};

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

    RETURN_FIRST_OUTPUT(loopOut);
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    // Input tensor is in NCHW format
    ASSERT(
        (inputs.at(0).shape().nbDims == 4) && "The input tensor must be in NCHW format.", ErrorCode::kUNSUPPORTED_NODE);
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
        firstShape = concat(ctx, N,
            concat(ctx, blockSizeTensor,
                concat(ctx, blockSizeTensor,
                    concat(ctx, C_2,
                        concat(ctx, H, W)))));
        std::copy(std::begin(DCRPerm), std::end(DCRPerm), std::begin(perm.order));
    }
    else
    {
        // First reshape to {N, C / (blockSize * blockSize), blockSize, blockSize, H, W}
        firstShape = concat(ctx, N,
            concat(ctx, C_2,
                concat(ctx, blockSizeTensor,
                    concat(ctx, blockSizeTensor,
                        concat(ctx, H, W)))));
        std::copy(std::begin(CRDPerm), std::end(CRDPerm), std::begin(perm.order));
    }

    auto* firstShuffle = addShuffle(ctx, *tensorPtr, firstShape);
    firstShuffle->setSecondTranspose(perm);
    ctx->registerLayer(firstShuffle, getNodeName(node));
    tensorPtr = firstShuffle->getOutput(0);

    // Finally reshape to {N, C / (blockSize * blockSize), H * blockSize, W * blockSize};
    auto secondShape = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShape);
    tensorPtr = secondShuffle->getOutput(0);

    return {{tensorPtr}};
}

// This is a helper function for QuantizeLinear/DequantizeLinear
NodeImportResult QuantDequantLinearHelper(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs, bool isDQ)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));

    // For QuantizeLinear, the output type (and thus quantization type) is dependent on the second input (zero point).
    if (!isDQ && inputs.size() >= 3)
    {
        CHECK(notInvalidType(inputs.at(2), {"UINT8"}));
    }
    auto addConstantLayer
        = [ctx](nvinfer1::INetworkDefinition& network, ShapedWeights const& weights) -> nvinfer1::ITensor* {
        nvinfer1::IConstantLayer* constLayer = network.addConstant(weights.shape, weights);
        ctx->registerLayer(constLayer, weights.getName());
        network.setWeightsName(weights, weights.getName());
        return constLayer->getOutput(0);
    };

    auto newConstantInput = [&](int32_t i) {
        return inputs.at(i).is_weights() && (ctx->getConstantLayer(inputs.at(i).weights().getName()) == nullptr);
    };

    ASSERT((inputs.size() == 3) && "This version of TensorRT requires 3 inputs for the DequantizeLinear operator.",
        nvonnxparser::ErrorCode::kINVALID_NODE);

    std::string nodeName = getNodeName(node);
    // Input 0 is the data to quantize or dequantize.
    nvinfer1::ITensor& dataInput = convertToTensor(inputs.at(0), ctx);

    // Input 1 initializes the layer's scale weights.
    nvinfer1::ITensor* scaleInput = nullptr;
    if (newConstantInput(1))
    {
        // Scale is concrete so verify it now.
        auto scale = inputs.at(1).weights();
        ASSERT(scale.count() > 0 && "Cannot have scale with no coefficients.", nvonnxparser::ErrorCode::kINVALID_NODE);
        auto const* scaleVal = static_cast<float const*>(scale.values);
        auto scaleAllPositive = std::all_of(scaleVal, scaleVal + scale.count(), [](float x) { return x > 0; });
        ASSERT(scaleAllPositive && "Scale coefficients must all be positive", nvonnxparser::ErrorCode::kINVALID_NODE);

        // If the scale is concrete weights, then add a ConstantLayer that will be an input which
        // will initialize the scale weights.
        scaleInput = addConstantLayer(*ctx->network(), scale);
    }
    else
    {
        scaleInput = &convertToTensor(inputs.at(1), ctx);
    }
    const auto scaleSize = volume(scaleInput->getDimensions());

    // Input 2 initializes the layer's zero-point.
    nvinfer1::ITensor* zeroPointInput = nullptr;
    if (newConstantInput(2))
    {
        // Zero-point verification.
        auto zeroPoint = inputs.at(2).weights();
        ASSERT(shiftIsAllZeros(zeroPoint) && "TRT only supports symmetric quantization - zeroPt must be all zeros",
            nvonnxparser::ErrorCode::kINVALID_NODE);
        // Convert the zero-point to Float because that's TRT uses float for zero-point.
        auto fpZeroPoint = createZeroShifts(zeroPoint, ::ONNX_NAMESPACE::TensorProto::FLOAT, ctx);
        fpZeroPoint.setName(zeroPoint.getName());
        zeroPointInput = addConstantLayer(*ctx->network(), fpZeroPoint);
    }
    else
    {
        zeroPointInput = &convertToTensor(inputs.at(2), ctx);
    }
    auto const zeroPointSize = volume(zeroPointInput->getDimensions());
    // ONNX may represent a scalar using either 0-D or 1-D, so compare sizes instead of shapes.
    ASSERT(zeroPointSize == scaleSize && "The scale and zero-point must have the same size",
        nvonnxparser::ErrorCode::kINVALID_NODE);

    // Read the optional quantization axis attribute. Set it to the rank of the input tensor if not provided
    OnnxAttrs attrs(node, ctx);
    int32_t const nbDims = dataInput.getDimensions().nbDims;
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
        int64_t K = dataInput.getDimensions().d[axis];
        ASSERT(K == scaleSize && "The number of scales is not equal to the number of output channels.",
            nvonnxparser::ErrorCode::kINVALID_NODE);
    }
    else
    {
        // Per-Tensor Quantization.
        // Currently axis is ignored by TRT, but it is required here by addScaleNd (for computing nbSpatialDims). Set to a sane default depending on rank the input tensor.
        axis = nbDims <= 1 ? 0 : 1;
    }

    nvinfer1::ILayer* layer = nullptr;
    if (isDQ)
    {
        // Add and configure a DequantizeLayer.
        nvinfer1::IDequantizeLayer* dq = ctx->network()->addDequantize(dataInput, *scaleInput);
        ASSERT(dq && "Failed to create Dequantize layer.", ErrorCode::kUNSUPPORTED_NODE);
        dq->setAxis(axis);
        layer = dq;
    }
    else
    {
        // Add and configure a QuantizeLayer.
        nvinfer1::IQuantizeLayer* q = ctx->network()->addQuantize(dataInput, *scaleInput);
        ASSERT(q && "Failed to create Quantize layer.", ErrorCode::kUNSUPPORTED_NODE);
        q->setAxis(axis);
        layer = q;
    }

    layer->setName(nodeName.c_str());
    layer->setInput(2, *zeroPointInput);

    // Return layer output
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(DequantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, inputs, true /*isDQ*/);
}

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kDIV);
}

DEFINE_BUILTIN_OP_IMPORTER(Dropout)
{
    // TensorRT does not support the Dropout operator with training mode.
    // The source of training mode information comes from :
    // 1. Pre-opset 6: attribute is_test = 0
    // 2. Post-opset 12: input[2] training_mode = true.
    //      We can deal with the cases where training_mode is an initializer.
    if (ctx->getOpsetVersion() <= 6)
    {
        OnnxAttrs attrs(node, ctx);
        int32_t isTestingMode = attrs.get<int32_t>("is_test", 1);
        ASSERT(isTestingMode && "TensorRT does not support the Droupout operator with training mode.",
            ErrorCode::kUNSUPPORTED_NODE);
    }
    else if (ctx->getOpsetVersion() >= 12 && node.input().size() == 3)
    {
        ASSERT(inputs.at(2).is_weights()
                && "This Version of TensorRT only supports the training_mode input as an initializer.",
            ErrorCode::kUNSUPPORTED_NODE);
        std::vector<int64_t> trainingMode;
        weightsToVector<int64_t>(inputs.at(2).weights(), &trainingMode);
        ASSERT(!trainingMode[0] && "TensorRT does not support the Dropout operator in training mode.",
            ErrorCode::kUNSUPPORTED_NODE);
    }

    size_t noutputs = node.output().size();
    if (noutputs == 1)
    {
        RETURN_IDENTITY(inputs.at(0));
    }
    else
    {
        // Add identity layer twice for both Dropout outputs: (output + mask)
        std::vector<TensorOrWeights> outputs;
        outputs.push_back(inputs.at(0));

        // Add mask tensor, which is the same shape as the input tensor
        auto& inputTensor = inputs.at(0).tensor();
        nvinfer1::ITensor* maskTensor{nullptr};
        // Post opset 12 the mask tensor contains all 1s. Prior to opset 12 the mask tensor contains all 0s.
        if (ctx->getOpsetVersion() >= 12)
        {
            maskTensor = ctx->network()->addElementWise(inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kEQUAL)->getOutput(0);
        }
        else
        {
            maskTensor = ctx->network()->addElementWise(inputTensor, inputTensor, nvinfer1::ElementWiseOperation::kLESS)->getOutput(0);
        }
        outputs.push_back(TensorOrWeights(maskTensor));
        return outputs;
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Einsum)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    OnnxAttrs attrs(node, ctx);
    std::string const equation = attrs.get<std::string>("equation");

    std::string invalidCharacters;
    for (char c : equation)
    {
        if ((c < 'a' || c > 'z') && c != '-' && c != '>' && c != '.' && c != ',' && c != ' ')
        {
            invalidCharacters.push_back(c);
            invalidCharacters.push_back(',');
        }
    }

    if (!invalidCharacters.empty())
    {
        invalidCharacters.pop_back();
        return MAKE_ERROR("Invalid character(s) in Einsum equation: " + invalidCharacters, ErrorCode::kINVALID_NODE);
    }

    ASSERT((!inputs.empty()) && "Inputs vector is empty.", ErrorCode::kINVALID_NODE);

    std::vector<nvinfer1::ITensor*> inputTensors;

    for (auto input : inputs)
    {
        auto* tensor_ptr = &convertToTensor(input, ctx);
        inputTensors.push_back(tensor_ptr);
    }
    auto nbInputs = static_cast<int32_t>(inputTensors.size());
    nvinfer1::IEinsumLayer* einsumLayer = ctx->network()->addEinsum(inputTensors.data(), nbInputs, equation.c_str());
    ctx->registerLayer(einsumLayer, getNodeName(node));

    RETURN_FIRST_OUTPUT(einsumLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Elu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Equal)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kEQUAL);
}

DEFINE_BUILTIN_OP_IMPORTER(Erf)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kERF);
}

DEFINE_BUILTIN_OP_IMPORTER(Exp)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Expand)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // "Broadcast the input tensor following the given shape and the broadcast rule."
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(inputTensor);
    auto const inputRank = shapeOf(inputDims);

    // "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
    ASSERT((inputs.at(1).shape().nbDims == 1) && "The shape tensor is required to be 1D.", ErrorCode::kINVALID_VALUE);
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
    ctx->registerLayer(sliceLayer, getNodeName(node));

    RETURN_FIRST_OUTPUT(sliceLayer);
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
    ASSERT(dims.nbDims == 2 && "Only 2D tensors are supported. Input must be of rank 2.", ErrorCode::kUNSUPPORTED_NODE);

    // The data type can be specified by the 'dtype' argument
    DataType dtype = tensor.getType();
    if (attrs.count("dtype"))
    {
        auto onnxType = attrs.get<int32_t>("dtype");
        ASSERT(convertDtype(onnxType, &dtype) && "Unsupported cast!", ErrorCode::kINVALID_NODE);
        LOG_VERBOSE("Casting to type: " << dtype);
    }

    // Create weights and constant layer
    ASSERT(!isDynamic(dims) && "Eyelike does not work for dynamically shaped tensors.", ErrorCode::kUNSUPPORTED_NODE);
    int totalWeights = dims.d[0] * dims.d[1];
    std::vector<int> values(totalWeights);
    for (int32_t r = 0; r < dims.d[0]; ++r)
    {
        for (int32_t c = 0; c < dims.d[1]; ++c)
        {
            values[r * dims.d[1] + c] = 0;
            if (c - r == k)
            {
                values[r*dims.d[1] + c] = 1;
            }
        }
    }

    ShapedWeights tempWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, dims);
    std::memcpy(tempWeights.values, values.data(), values.size() * sizeof(int));
    auto* layer = ctx->network()->addConstant(dims, tempWeights);
    layer->setOutputType(0, DataType::kINT32);
    ctx->registerLayer(layer, node.name());

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
        RETURN_IDENTITY(inputs.at(0));
    }

    tensorPtr = flattenTensor(ctx, node, *tensorPtr, axis, true);
    ASSERT(tensorPtr && "Failed to flatten the tensor.", ErrorCode::kUNSUPPORTED_NODE);
    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    CHECK(convertAxis(axis, nbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);
    auto* layer = ctx->network()->addGather(data, indices, axis);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherElements)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    nvinfer1::Dims const& dataDims = data.getDimensions();

    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t const dataNbDims = dataDims.nbDims;
    CHECK(convertAxis(axis, dataNbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);

    auto* layer = ctx->network()->addGatherV2(data, indices, nvinfer1::GatherMode::kELEMENT);
    layer->setGatherAxis(axis);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(GatherND)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);

    OnnxAttrs attrs(node, ctx);
    auto const nbElementWiseDims = attrs.get<int32_t>("batch_dims", 0);

    auto* layer = ctx->network()->addGatherV2(data, indices, nvinfer1::GatherMode::kND);
    layer->setNbElementWiseDims(nbElementWiseDims);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32"}));
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& inputB = convertToTensor(inputs.at(1), ctx);
    // Validate inputs
    ASSERT(inputA.getDimensions().nbDims == 2 && inputB.getDimensions().nbDims == 2 && "GEMM must have 2D inputs!",
        ErrorCode::kINVALID_NODE);

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
    ctx->registerLayer(matmul, getNodeName(node));
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
    nvinfer1::Dims dims = tensor.getDimensions();

    OnnxAttrs attrs{node, ctx};
    float p = static_cast<float>(attrs.get("p", 2));

    // Add constants for p and 1/p
    nvinfer1::Dims scalarDims{dims.nbDims};
    std::fill(scalarDims.d, scalarDims.d + scalarDims.nbDims, 1);
    auto& pTensor = *addConstantScalar(ctx, p, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims)->getOutput(0);
    auto& pInvTensor = *addConstantScalar(ctx, 1.F / p, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims)->getOutput(0);

    // firstPow = pow(x, p)
    auto* firstPow = ctx->network()->addElementWise(tensor, pTensor, nvinfer1::ElementWiseOperation::kPOW)->getOutput(0);
    // reduced = reduce_sum(firstPow)
    auto* reduced = globalPoolingHelper(ctx, node, *firstPow, nvinfer1::ReduceOperation::kSUM);
    // finalPow = pow(reduced, 1./p)
    auto* finalPow = ctx->network()->addElementWise(*reduced, pInvTensor, nvinfer1::ElementWiseOperation::kPOW)->getOutput(0);
    return {{finalPow}};
}


DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool)
{
    LOG_VERBOSE("GlobalMaxPool operators are implemented via Reduce layers rather than Pooling layers");
    return {{globalPoolingHelper(ctx, node, convertToTensor(inputs.at(0), ctx), nvinfer1::ReduceOperation::kMAX)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Greater)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kGREATER);
}

DEFINE_BUILTIN_OP_IMPORTER(GreaterOrEqual)
{
    return {{greaterLessOrEqual(ctx, node, &convertToTensor(inputs.at(0), ctx), &convertToTensor(inputs.at(1), ctx),
        /*greater*/ true)}};
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
        nvinfer1::ITensor* forwardStart = addConstant(ctx, std::vector<int32_t>{0, 0, 0},
                                              ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
                                              ->getOutput(0);
        nvinfer1::ITensor* reverseStart = addConstant(ctx, std::vector<int32_t>{1, 0, 0},
                                              ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
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
            backwardHt = clearMissingSequenceElements(ctx, node, loop, seqLens, backwardHt, sequenceLength, /*reverse=*/ true);
        }

        nvinfer1::ILoopOutputLayer* forwardOutput
            = loop->addLoopOutput(*forwardHt, nvinfer1::LoopOutput::kCONCATENATE, 0);
        forwardOutput->setInput(1, *sequenceLength);
        nvinfer1::ILoopOutputLayer* reverseOutput
            = loop->addLoopOutput(*backwardHt, nvinfer1::LoopOutput::kREVERSE, 0);
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
            concatenatedOutput = clearMissingSequenceElements(ctx, node, loop, seqLens, concatenatedOutput, sequenceLength, reverse);
        }
        nvinfer1::ILoopOutputLayer* scanOut
            = loop->addLoopOutput(*concatenatedOutput, (reverse ? nvinfer1::LoopOutput::kREVERSE : nvinfer1::LoopOutput::kCONCATENATE), 0);
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

    constexpr int32_t NUM_ACTIVATIONS = 2;
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

    // TODO: Support cases where in bidirectional GRUs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
    }

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
    ASSERT(iterationInput && "Failed to add RNN input.", ErrorCode::kINVALID_NODE);

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
    ctx->registerLayer(Ht1, getNodeName(node));
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
        Ht = numDirections == 2 ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1->getOutput(0), Ht, singlePassShape) : maskRNNHidden(ctx, node, loop, seqLens, Ht1->getOutput(0), Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections, inputs, direction == "reverse"));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*Ht1->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(HardSigmoid)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.2f);
    float beta = attrs.get<float>("beta", 0.5f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Identity)
{
    auto* layer = ctx->network()->addIdentity(convertToTensor(inputs.at(0), ctx));
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(If)
{
    OnnxAttrs attrs(node, ctx);
    auto cond = inputs.at(0);

    ::ONNX_NAMESPACE::GraphProto const& thenGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("then_branch");
    ::ONNX_NAMESPACE::GraphProto const& elseGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("else_branch");

    // Number of outputs are the same between the two branches.
    ASSERT(thenGraph.output_size() == elseGraph.output_size()
            && "then/else subgraphs should have the same number of outputs.",
        ErrorCode::kINVALID_NODE);
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

        CHECK(onnx2trt::parseGraph(ctx, body));
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
    ASSERT(condTensor && "Failed to convert the input cond to a scalar.", ErrorCode::kINVALID_NODE);

    auto conditional = ctx->network()->addIfConditional();
    conditional->setName(getNodeName(node).c_str());
    conditional->setCondition(*condTensor);

    std::vector<nvinfer1::ILayer*> thenLayers, elseLayers;
    StringMap<TensorOrWeights> thenSubgraphTensors;
    StringMap<TensorOrWeights> elseSubgraphTensors;
    CHECK(importSubgraph(ctx, thenGraph, thenLayers, thenSubgraphTensors));
    CHECK(importSubgraph(ctx, elseGraph, elseLayers, elseSubgraphTensors));

    using InputsMap = std::unordered_map<std::string, nvinfer1::IIfConditionalInputLayer*>;
    InputsMap inputsMap;
    CHECK(addIfInputLayers(ctx, conditional, inputsMap, thenLayers));
    CHECK(addIfInputLayers(ctx, conditional, inputsMap, elseLayers));
    CHECK(addIfOutputLayers(ctx, conditional, thenGraph, thenLayers, thenSubgraphTensors, elseGraph, elseLayers,
        elseSubgraphTensors, graphOutputs));

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
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    // Scales and biases must be initializers
    ASSERT(inputs.at(1).is_weights() && "The scale tensor is required to be an initializer.",
        ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(
        inputs.at(2).is_weights() && "The bias tensor is required to be an initializer.", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    ASSERT(nbDims >= 3 && nbDims <= 5 && "TensorRT only supports InstanceNormalization on 3D, 4D, or 5D tensors!",
        ErrorCode::kUNSUPPORTED_NODE);

    const bool needToExpandDims = (nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> const axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    int32_t const relu{0};  // the ONNX instance norm op does not use the relu parameter
    float const alpha{0.f}; // the ONNX instance norm op does not use the alpha parameter

    // Populate instanceNormalization plugin properties.
    std::string const pluginName = "InstanceNormalization_TRT";
    std::string const pluginVersion = "1";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", scale_weights.values, nvinfer1::PluginFieldType::kFLOAT32, scale_weights.count());
    f.emplace_back("bias", bias_weights.values, nvinfer1::PluginFieldType::kFLOAT32, bias_weights.count());
    f.emplace_back("relu", &relu, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1);

    // Create plugin from registry
    auto const plugin = createPlugin(getNodeName(node), importPluginCreator(pluginName, pluginVersion), f);

    ASSERT(plugin != nullptr && "InstanceNormalization plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addPluginV2(&tensorPtr, 1, *plugin);
    ctx->registerLayer(layer, getNodeName(node));
    tensorPtr = layer->getOutput(0);

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> const axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(IsNaN)
{
    // IEEE arithmetic guarantees that x == x is false if x is a NaN, and true otherwise.
    std::vector<TensorOrWeights> const newInputs{inputs[0], inputs[0]};
    auto equalResult = elementwiseHelper(ctx, node, newInputs, nvinfer1::ElementWiseOperation::kEQUAL);
    if (equalResult.is_error())
    {
        return equalResult;
    }
    auto equalRet = equalResult.value().at(0);
    return unaryHelper(ctx, node, equalRet, nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(LeakyRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 0.01f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kLEAKY_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Less)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kLESS);
}

DEFINE_BUILTIN_OP_IMPORTER(LessOrEqual)
{
    return {{greaterLessOrEqual(ctx, node, &convertToTensor(inputs.at(0), ctx), &convertToTensor(inputs.at(1), ctx),
        /*greater*/ false)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Log)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    auto& input = convertToTensor(inputs.at(0), ctx);
    // Don't use softmax converter since it adds a shuffle layer
    // which prevents the builder to fuse softmax and log operations.
    auto* softmax = addSoftmax(ctx, node, input);
    nvinfer1::IUnaryLayer* unaryLayer = ctx->network()->addUnary(*softmax, nvinfer1::UnaryOperation::kLOG);
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *unaryLayer->getOutput(0), shapeOf(input));
    RETURN_FIRST_OUTPUT(reshapeLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Loop)
{
    constexpr int32_t NB_NON_STATE_INPUTS = 2; // First 2 inputs are trip count and condition respectively.
    constexpr int32_t NB_DISCARDED_OUTPUTS
        = 1; // First output is the updated value of the condition, and is ignored by the outer loop node.
    constexpr int32_t MAX_SCAN_OUTPUT_LENGTH = 1024; // Maximum length for scan outputs if trip count is not set.
    ASSERT((inputs.size() >= 2) && "The Loop operator requires at least 2 inputs.", ErrorCode::kINVALID_NODE);
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
        bool const isMaxTripCount = inputs[0].is_weights()
            && static_cast<int32_t*>(inputs[0].weights().values)[0] == std::numeric_limits<int32_t>::max();
        if (!isMaxTripCount)
        {
            tripLimit = convertToScalar(ctx, &convertToTensor(inputs[0], ctx));
            ASSERT(tripLimit && "Failed to convert the trip-count input to a scalar.", ErrorCode::kINVALID_NODE);
            ctx->loopTensors()[body.input(0).name()] = node.input(0);
            loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
            // First graph input is iteration_num, so create a loop counter
            auto counter = addLoopCounter(ctx, loop, 0);
            ctx->registerTensor(counter, body.input(0).name());
        }
    }
    nvinfer1::ITensor* cond{nullptr};
    if (inputs[1])
    {
        cond = convertToScalar(ctx, &convertToTensor(inputs[1], ctx));
        ASSERT(cond && "Failed to convert the input cond to a scalar.", ErrorCode::kINVALID_NODE);
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
    }

    // Loop body
    CHECK(onnx2trt::parseGraph(ctx, body));

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
            trtScanOut->setInput(
                1, *addConstantScalar(ctx, MAX_SCAN_OUTPUT_LENGTH, ::ONNX_NAMESPACE::TensorProto_DataType_INT32)
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
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
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
    int32_t const inputForget = attrs.get("input_forget", 0);
    float const clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    ASSERT(
        inputForget == 0 && "Coupled input/forget is unsupported in the LSTM converter", ErrorCode::kUNSUPPORTED_NODE);

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);

    constexpr int32_t NUM_ACTIVATIONS = 3;
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

    // TODO: Support cases where in bidirectional LSTMs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activation alphas for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activation betas for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
    }

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
    ASSERT(iterationInput && "Failed to add RNN input.", ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(Ht1, getNodeName(node));
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
    nvinfer1::ITensor* operandFC = ctx->network()->addElementWise(*ftGate, *Ct1->getOutput(0), eOp::kPROD)->getOutput(0);
    nvinfer1::ITensor* Ct
        = ctx->network()
              ->addElementWise(*operandFC, *operandIC, eOp::kSUM)
              ->getOutput(0);

    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*gateOutputShape,
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
                       ->getOutput(0),
                  eOp::kDIV)
              ->getOutput(0);

    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ct = numDirections == 2 ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ct1->getOutput(0), Ct, singlePassShape) : maskRNNHidden(ctx, node, loop, seqLens, Ct1->getOutput(0), Ct, maxLen, direction == "reverse");
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
        Ht = numDirections == 2 ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, Ht1->getOutput(0), Ht, singlePassShape) : maskRNNHidden(ctx, node, loop, seqLens, Ht1->getOutput(0), Ht, maxLen, direction == "reverse");
    }
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)

    outputs.emplace_back(
        concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections, inputs, direction == "reverse"));
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
    int32_t axis = attrs.get<int32_t>("axis", -1);
    int32_t p = attrs.get<int32_t>("p", 2);
    int32_t nbDims = input->getDimensions().nbDims;
    DataType dt = input->getType();
    ASSERT((dt == DataType::kFLOAT || dt == DataType::kHALF) && "Only float inputs/outputs supported in LpNormalization.", ErrorCode::kINVALID_NODE);

    CHECK(convertAxis(axis, nbDims));

    ASSERT((p == 1 || p == 2) && "Only L1 and L2 normalization are supported.", ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* norm{nullptr};
    TensorOrWeights zeros = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, {0,{}});
    nvinfer1::ITensor* zerosTensor = &convertToTensor(zeros, ctx);
    broadcastTensor(ctx, zerosTensor, nbDims);

    if (p == 1) {
        // abs(x)
        nvinfer1::IUnaryLayer* absLayer = ctx->network()->addUnary(*input, uOp::kABS);
        ctx->registerLayer(absLayer, getNodeName(node));
        norm = absLayer->getOutput(0);

        // norm coeff = sum(abs(x)) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true);
        ctx->registerLayer(reduceLayer, getNodeName(node));
        norm = reduceLayer->getOutput(0);
    }
    else if (p == 2)
    {
        // x^2
        auto* sqrLayer = ctx->network()->addElementWise(*input, *input, eOp::kPROD);
        ctx->registerLayer(sqrLayer, getNodeName(node));
        norm = sqrLayer->getOutput(0);

        // sum(x^2) along axis dimension
        nvinfer1::IReduceLayer* reduceLayer = ctx->network()->addReduce(*norm, rOp::kSUM, 1 << axis, true);
        ctx->registerLayer(reduceLayer, getNodeName(node));
        norm = reduceLayer->getOutput(0);

        // norm coeff = sqrt(sum(x^2))
        nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*norm, uOp::kSQRT);
        ctx->registerLayer(sqrtLayer, getNodeName(node));
        norm = sqrtLayer->getOutput(0);
    }

    // norm coeff |= 1 (change 0s to 1s, leave all other values same)
    nvinfer1::IElementWiseLayer* maskLayer = ctx->network()->addElementWise(*norm, *zerosTensor, eOp::kEQUAL);
    ctx->registerLayer(maskLayer, getNodeName(node));
    nvinfer1::ITensor* mask = maskLayer->getOutput(0);
    mask = castHelper(ctx, mask, dt);
    auto* combinedLayer = ctx->network()->addElementWise(*norm, *mask, eOp::kSUM);
    ctx->registerLayer(combinedLayer, getNodeName(node));
    norm = combinedLayer->getOutput(0);

    // x/(norm coeff)
    // norm tensor is broadcast along axis dimension to match shape of input
    auto *layer = ctx->network()->addElementWise(
        *input, *norm, eOp::kDIV);
    ctx->registerLayer(layer, getNodeName(node));
    ASSERT(layer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);

    RETURN_FIRST_OUTPUT(layer);
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
    ASSERT((dt == DataType::kFLOAT || dt == DataType::kHALF) && "Only float inputs/outputs supported in LpPool.", ErrorCode::kINVALID_NODE);
    ASSERT((p == 1 || p == 2) && "Only L1 and L2 normalization are supported.", ErrorCode::kINVALID_NODE);

    nvinfer1::Dims kernelShape = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding(false);
    getKernelParams(ctx, node, &kernelShape, &strides, &begPadding, &endPadding, paddingMode, exclude_padding);

    nvinfer1::Dims scalarDims = makeDims(nbDims, 1);
    float kernelSz{1.0f};
    for (int32_t i = 0; i < kernelShape.nbDims; i++)
    {
        kernelSz *= kernelShape.d[i];
    }
    nvinfer1::ITensor* kernelSzTensor
        = addConstantScalar(ctx, kernelSz, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims)->getOutput(0);

    nvinfer1::ITensor* output{nullptr};
    if (p == 1) {
        // x' = abs(x)
        nvinfer1::IUnaryLayer* absLayer = ctx->network()->addUnary(*input, uOp::kABS);
        ctx->registerLayer(absLayer, getNodeName(node));
        output = absLayer->getOutput(0);
    } else if (p == 2) {
        // x' = x^2
        auto* sqrLayer = ctx->network()->addElementWise(*input, *input, eOp::kPROD);
        ctx->registerLayer(sqrLayer, getNodeName(node));
        output = sqrLayer->getOutput(0);
    }

    // pool_avg(x')
    nvinfer1::IPoolingLayer* poolLayer = ctx->network()->addPoolingNd(*output, pType::kAVERAGE, kernelShape);
    poolLayer->setPaddingMode(paddingMode);
    poolLayer->setPrePadding(begPadding);
    poolLayer->setPostPadding(endPadding);
    poolLayer->setStrideNd(strides);
    poolLayer->setAverageCountExcludesPadding(exclude_padding);
    ctx->registerLayer(poolLayer, getNodeName(node));
    output = poolLayer->getOutput(0);

    // pool_sum = pool_avg(x')*kernel_size
    auto* correctedSumLayer = ctx->network()->addElementWise(*output, *kernelSzTensor, eOp::kPROD);
    ctx->registerLayer(correctedSumLayer, getNodeName(node));
    output = correctedSumLayer->getOutput(0);

    // if p == 1, output = pool_sum
    // if p == 2, output = sqrt(pool_sum)
    if (p == 2) {
        nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*output, uOp::kSQRT);
        ctx->registerLayer(sqrtLayer, getNodeName(node));
        output = sqrtLayer->getOutput(0);
    }
    return {{output}};
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32"}));
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
    ctx->registerLayer(matmul, getNodeName(node));

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
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool)
{
    ASSERT(node.output().size() == 1 && "TensorRT does not support the indices output in MaxPool!",
        ErrorCode::kUNSUPPORTED_NODE);
    return poolingHelper(ctx, node, inputs, nvinfer1::PoolingType::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(Mean)
{
    auto sum_result = elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    auto& sum_input = sum_result.value().at(0);
    nvinfer1::ITensor& sum_tensor = sum_input.tensor();

    int32_t ndim = sum_tensor.getDimensions().nbDims;
    float scale_value = 1.f / inputs.size();
    auto scale_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    auto scale_shape = nvinfer1::Dims{ndim, {1, 1, 1, 1, 1, 1, 1, 1}};
    auto scale_weights = ctx->createTempWeights(scale_dtype, scale_shape);
    static_cast<float*>(scale_weights.values)[0] = scale_value;
    auto* constant_layer = ctx->network()->addConstant(scale_weights.shape, scale_weights);
    ASSERT(constant_layer && "Failed to create the scalar tensor.", ErrorCode::kUNSUPPORTED_NODE);
    ctx->network()->setWeightsName(scale_weights, scale_weights.getName());
    nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
    RETURN_FIRST_OUTPUT(
        ctx->network()->addElementWise(sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD));
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

    ASSERT((dt == DataType::kFLOAT || dt == DataType::kHALF) && "Only float/half inputs/outputs supported in MeanVarianceNormalization.", ErrorCode::kINVALID_NODE);

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
    ctx->registerLayer(reduceLayer, getNodeName(node));
    auto* meanX = reduceLayer->getOutput(0);

    // numerator: x-mean(x)
    auto* numSubLayer = ctx->network()->addElementWise(*input, *meanX, eOp::kSUB);
    ctx->registerLayer(numSubLayer, getNodeName(node));
    auto* numerator = numSubLayer->getOutput(0);

    // (x-mean(x))^2
    auto* sqrLayer = ctx->network()->addElementWise(*numerator, *numerator, eOp::kPROD);
    ctx->registerLayer(sqrLayer, getNodeName(node));
    auto* sqrNumerator = sqrLayer->getOutput(0);

    // mean((x-mean(x))^2)
    auto* meanLayer = ctx->network()->addReduce(*sqrNumerator, rOp::kAVG, axesMask, true);
    ctx->registerLayer(meanLayer, getNodeName(node));
    auto* variance = meanLayer->getOutput(0);

    // sqrt(mean((x-mean(x))^2))
    nvinfer1::IUnaryLayer* sqrtLayer = ctx->network()->addUnary(*variance, uOp::kSQRT);
    ctx->registerLayer(sqrtLayer, getNodeName(node));
    auto* stdDev = sqrtLayer->getOutput(0);

    // denominator: avoid division by zero
    nvinfer1::Dims scalarShape{dims.nbDims};
    std::fill(scalarShape.d, scalarShape.d + scalarShape.nbDims, 1);
    auto* epsilonTensor = addConstantScalar(ctx, 1e-9f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, scalarShape)->getOutput(0);
    auto* addEpsLayer = ctx->network()->addElementWise(*stdDev, *epsilonTensor, eOp::kSUM);
    ctx->registerLayer(addEpsLayer, getNodeName(node));
    stdDev = addEpsLayer->getOutput(0);

    // division numerator/standard-deviation
    auto* divLayer = ctx->network()->addElementWise(*numerator, *stdDev, eOp::kDIV);
    ctx->registerLayer(divLayer, getNodeName(node));

    ASSERT(divLayer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);

    RETURN_FIRST_OUTPUT(divLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Min)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPROD);
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

    if (fmod == 0){
        // fmod = 0, inputs can only be integers
        ASSERT((input0->getType() == DataType::kINT32)
        && "The fmod attribute is set to 0. Inputs cannot be of floating point types.",
        ErrorCode::kINVALID_NODE);
        // Result = input0 - (input1 * floorDiv(input0, input1))
        nvinfer1::IElementWiseLayer* resultLayer = modWithIntegerInputs(ctx, input0, input1, false);

        ctx->registerLayer(resultLayer, getNodeName(node));
        RETURN_FIRST_OUTPUT(resultLayer);
    }
    // Fmod with integer inputs
    else if (input0->getType() == DataType::kINT32)
    {
        // Result = input0 - (input1 * Div(input0, input1))
        nvinfer1::IElementWiseLayer* resultLayer = modWithIntegerInputs(ctx, input0, input1, true);
        ctx->registerLayer(resultLayer, getNodeName(node));
        RETURN_FIRST_OUTPUT(resultLayer);
    }
    // Fmod with floating point inputs
    else
    {
        // Calculate input0 / input1
        TensorOrWeights divResult
            = elementwiseHelper(ctx, node, {input0, input1}, eOp::kDIV).value().at(0);

        // Calculate input0 - (input1 * floor(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivFloor = modWithFPInputs(ctx, input0, input1, &divResult.tensor(), true);

        // Calculate input0 - (input1 * ceil(input0 / input1))
        nvinfer1::IElementWiseLayer* layerWithDivCeil = modWithFPInputs(ctx, input0, input1, &divResult.tensor(), false);

        auto* zero = createZeroTensor(ctx, &divResult.tensor());
        auto* condition = greaterLessOrEqual(ctx, node, &divResult.tensor(), zero, true);
        auto* outputWithDivFloor = layerWithDivFloor->getOutput(0);
        auto* outputWithDivCeil = layerWithDivCeil->getOutput(0);

        // If (input0 / input1) >= 0, result = input0 - (input1 * floor(input0 / input1))
        // Else result = input0 - (input1 * ceil(input0 / input1))
        auto* result = ctx->network()->addSelect(*condition, *outputWithDivFloor, *outputWithDivCeil);
        ctx->registerLayer(result, getNodeName(node));
        RETURN_FIRST_OUTPUT(result);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Neg)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(NonMaxSuppression)
{
    // max_output, iou_threshold and score_threshold are optional
    ASSERT(inputs.size() >= 2 && inputs.size() <= 5 && "The node requires between 2-5 inputs",
           ErrorCode::kUNSUPPORTED_NODE);

    // Input: boxes
    nvinfer1::ITensor* boxesTensorPtr = &convertToTensor(inputs.at(0), ctx);
    ASSERT(boxesTensorPtr->getDimensions().nbDims == 3 && "The boxes tensor must be 3D",
           ErrorCode::kUNSUPPORTED_NODE);

    // Input: scores
    nvinfer1::ITensor* scoresTensorPtr = &convertToTensor(inputs.at(1), ctx);
    ASSERT(
        scoresTensorPtr->getDimensions().nbDims == 3 && "The scores tensor must be 3D", ErrorCode::kUNSUPPORTED_NODE);

    int32_t const maxOutputBoxesPerClassDefault = 0;
    nvinfer1::ITensor* maxOutputBoxesPerClassTensorPtr = nullptr;
    nvinfer1::ITensor* iouThresholdTensorPtr = nullptr;
    nvinfer1::ITensor* scoreThresholdTensorPtr = nullptr;

    // Input: max_output_boxes_per_class (default = 0)
    if (inputs.size() >= 3 && !inputs.at(2).isNullTensor())
    {
        maxOutputBoxesPerClassTensorPtr = convertToScalar(inputs.at(2), ctx);
        ASSERT(maxOutputBoxesPerClassTensorPtr != nullptr && "The max_output_boxes_per_class tensor must be 0D",
           ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        auto* constantLayer = ctx->network()->addConstant(nvinfer1::Dims{0, {}}, nvinfer1::Weights{DataType::kINT32, &maxOutputBoxesPerClassDefault, 1});
        ASSERT(constantLayer != nullptr && "Failed to add in constant for default max_output_boxes_per_class", ErrorCode::kUNSUPPORTED_NODE);
        maxOutputBoxesPerClassTensorPtr = constantLayer->getOutput(0);
    }

    // Input: iou_threshold (default = 0)
    if (inputs.size() >= 4 && !inputs.at(3).isNullTensor())
    {
        iouThresholdTensorPtr = convertToScalar(inputs.at(3), ctx);
        ASSERT(iouThresholdTensorPtr != nullptr && "The iou_threshold tensor must be 0D",
           ErrorCode::kUNSUPPORTED_NODE);
    }

    // Input: score_threshold (default = 0)
    if (inputs.size() >= 5 && !inputs.at(4).isNullTensor())
    {
        scoreThresholdTensorPtr = convertToScalar(inputs.at(4), ctx);
        ASSERT(scoreThresholdTensorPtr != nullptr && "The score_threshold tensor must be 0D",
           ErrorCode::kUNSUPPORTED_NODE);
    }

    // Transpose scores tensor from [batch, classes, bounding_boxes] to [batch, bounding_boxes, classes]
    nvinfer1::Permutation perm{0, 2, 1};
    nvinfer1::ITensor* transposedScoresTensorPtr = transposeTensor(ctx, node, *scoresTensorPtr, perm);
    ASSERT(transposedScoresTensorPtr && "Failed to transpose the scores input.", ErrorCode::kUNSUPPORTED_NODE);

    // Create the NMS layer
    auto* layer = ctx->network()->addNMS(*boxesTensorPtr, *transposedScoresTensorPtr, *maxOutputBoxesPerClassTensorPtr);
    ASSERT(layer != nullptr && "Failed to create NMS layer.", ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, getNodeName(node));

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
    default: ASSERT(0 && "Invalid value provided for the center_point_box attribute", ErrorCode::kUNSUPPORTED_NODE);
    }
    layer->setBoundingBoxFormat(fmt);

    RETURN_FIRST_OUTPUT(layer);
};

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(OneHot)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    CHECK(notInvalidType(inputs.at(1), {"UINT8"}));
    CHECK(notInvalidType(inputs.at(2), {"UINT8"}));
    ASSERT(node.input_size() && "OneHot must have exactly 3 inputs", ErrorCode::kINVALID_NODE);

    nvinfer1::ITensor* values = &convertToTensor(inputs.at(2), ctx);

    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(0), ctx);
    if (!inputs.at(0).isInt32())
    {
        indices = castHelper(ctx, indices, DataType::kINT32);
    }
    nvinfer1::ITensor* depth = &convertToTensor(inputs.at(1), ctx); //tensor #1 in ONNX
    if (!inputs.at(1).isInt32())
    {
        depth = castHelper(ctx, depth, DataType::kINT32);
    }
    depth = convertToScalar(ctx, depth);
    ASSERT(depth && "Failed to convert the depth to a scalar.", ErrorCode::kINVALID_NODE);

    OnnxAttrs attrs(node, ctx);
    auto axis = attrs.get<int32_t>("axis", -1);
    auto nbDims = indices->getDimensions().nbDims;
    CHECK(convertAxis(axis, nbDims+1));

    auto* layer = ctx->network()->addOneHot(*indices, *values, *depth, axis);

    auto const outDims = layer->getOutput(0)->getDimensions();

    ASSERT((outDims.d[axis] != -1) && "OneHot does not support dynamic depth input", ErrorCode::kINVALID_NODE);

    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Or)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensorPtr->getDimensions().nbDims;

    ASSERT(tensorPtr->getType() != DataType::kUINT8, ErrorCode::kUNSUPPORTED_NODE);

    OnnxAttrs attrs(node, ctx);
    auto const mode = attrs.get<std::string>("mode", "constant");
    float value{0.F};
    nvinfer1::ITensor* valuePtr = nullptr;
    std::vector<int32_t> onnxPadding;

    if (ctx->getOpsetVersion() < 11)
    {
        value = attrs.get<float>("value", 0.F);
        auto padding = attrs.get<std::vector<int32_t>>("pads");
        onnxPadding = std::vector<int32_t>(padding.begin(), padding.end());
        if (onnxPadding.empty())
        {
            LOG_VERBOSE("Found no-op pad in node: " + getNodeName(node));
            RETURN_IDENTITY(inputs.at(0));
        }
    }
    else
    {
        // In opset >= 11, padding indicies and values moved from attributes to inputs
        if (inputs.at(1).is_weights())
        {
            weightsToVector<int32_t>(inputs.at(1).weights(), &onnxPadding);
        }
        if (inputs.size() >= 3 && !inputs.at(2).isNullTensor())
        {
            bool isValueSet = false;
            if (inputs.at(2).is_weights())
            {
                auto const padWeight = inputs.at(2).weights();
                ASSERT((padWeight.count() == 1) && "The input constant_value is required to be a scalar.",
                    ErrorCode::kINVALID_NODE);
                switch (padWeight.type)
                {
                case ::ONNX_NAMESPACE::TensorProto::FLOAT:
                    value = static_cast<float const*>(padWeight.values)[0];
                    isValueSet = true;
                    break;
                case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
                    value = float(reinterpret_cast<half_float::half const*>(padWeight.values)[0]);
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
            ASSERT(false && "TensorRT does not support dynamic axes for pad!", ErrorCode::kUNSUPPORTED_NODE);
        }
    }

    nvinfer1::ITensor* start{};
    nvinfer1::ITensor* size{};
    if (onnxPadding.empty())
    {
        // the pads is from activation instead of initializer or attributes
        nvinfer1::ITensor* onnxPaddingPtr = &convertToTensor(inputs.at(1), ctx);
        ASSERT((onnxPaddingPtr->getDimensions().nbDims == 1) && "The pads input must be 1D.",
            ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(onnxPaddingPtr->getDimensions().d[0] == nbDims * 2
                && "pads should be a 1D tensor of shape [2 * input_rank]",
            ErrorCode::kUNSUPPORTED_NODE);

        auto pre = ctx->network()
                       ->addSlice(
                           *onnxPaddingPtr, nvinfer1::Dims{1, {0}}, nvinfer1::Dims{1, {nbDims}}, nvinfer1::Dims{1, {1}})
                       ->getOutput(0);
        auto post = ctx->network()
                        ->addSlice(*onnxPaddingPtr, nvinfer1::Dims{1, {nbDims}}, nvinfer1::Dims{1, {nbDims}},
                            nvinfer1::Dims{1, {1}})
                        ->getOutput(0);

        std::vector<int32_t> const zerosVal(nbDims, 0);
        auto const zeros = addConstant(ctx, zerosVal, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, {nbDims}})
                               ->getOutput(0);
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
        if (std::all_of(onnxPadding.begin(), onnxPadding.end(), [](int32_t i) { return i == 0; }))
        {
            LOG_VERBOSE("Found no-op pad in node: " + getNodeName(node));
            RETURN_IDENTITY(inputs.at(0));
        }

        // the pads is from initializer or attributes
        nvinfer1::ITensor* totalPadding = nullptr;
        ASSERT(convertOnnxPadding(ctx, nbDims, onnxPadding, start, totalPadding) && "Failed to convert padding!",
            ErrorCode::kUNSUPPORTED_NODE);
        size
            = ctx->network()
                  ->addElementWise(shapeOf(*tensorPtr).tensor(ctx), *totalPadding, nvinfer1::ElementWiseOperation::kSUM)
                  ->getOutput(0);
    }

    // add slice node
    auto const stride = makeDims(nbDims, 1);
    auto const& dummy = stride;
    auto* layer = ctx->network()->addSlice(*tensorPtr, dummy, dummy, stride);
    ASSERT(layer && "Could not create padding layer", ErrorCode::kUNSUPPORTED_NODE);
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
            case DataType::kFLOAT:
            case DataType::kHALF:
            case DataType::kINT8:
                fillValue = addConstant(
                    ctx, std::vector<float>{value}, ::ONNX_NAMESPACE::TensorProto::FLOAT, nvinfer1::Dims{0, {0}})
                                ->getOutput(0);
                break;
            default:
                fillValue = addConstant(ctx, std::vector<int32_t>{static_cast<int32_t>(value)},
                    ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{0, {0}})
                                ->getOutput(0);
                break;
            }
            ASSERT(fillValue && "Could not create layer for constant_value", ErrorCode::kUNSUPPORTED_NODE);
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

    ctx->registerLayer(layer, getNodeName(node));
    return {{layer->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(ParametricSoftplus)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Pow)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPOW);
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32"}));
    CHECK(notInvalidType(inputs.at(1), {"INT32"}));
    ASSERT((inputs.size() == 2) && "The PRelu operator requires exactly 2 inputs.", ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* slopes = &convertToTensor(inputs.at(1), ctx);
    CHECK(broadcastTensors(ctx, input, slopes));
    auto* layer = ctx->network()->addParametricReLU(*input, *slopes);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(QuantizeLinear)
{
    return QuantDequantLinearHelper(ctx, node, inputs, false /*isDQ*/);
}

NodeImportResult randomUniformHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    ShapeTensor const& inputShape, OnnxAttrs const& attrs, DataType const& inputDType)
{
    auto* fillLayer = addFill(ctx, inputShape, nvinfer1::FillOperation::kRANDOM_UNIFORM);
    ctx->registerLayer(fillLayer, getNodeName(node));

    // Set datatype of output:
    //      RandomUniform: dype is required and defaults to 1
    //      RandomUniformLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int32_t>("dtype", 1);
        switch (dtype)
        {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: fillLayer->setOutputType(0, DataType::kFLOAT); break;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: fillLayer->setOutputType(0, DataType::kHALF); break;
        default: return MAKE_ERROR("Unsupported data type", ErrorCode::kINVALID_VALUE);
        }
    }
    else
    {
        fillLayer->setOutputType(0, inputDType);
    }

    auto high = attrs.get<float>("high", 1.f);
    auto low = attrs.get<float>("low", 0.f);

    // Set "low" and "high" values of the fillLayer.
    fillLayer->setAlpha(low);
    fillLayer->setBeta(high);

    // TensorRT does not support "seed" field now. The support will be added in future versions.
    if (attrs.count("seed"))
    {
        LOG_WARNING("TensorRT currently ignores the \"seed\" field in RandomUniform op. Random seeds will be used.");
    }

    RETURN_FIRST_OUTPUT(fillLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniform)
{
    OnnxAttrs attrs(node, ctx);
    auto const shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    ShapeTensor const inputShape{1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end())};

    return randomUniformHelper(ctx, node, inputShape, attrs, DataType::kFLOAT);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniformLike)
{
    ASSERT(
        (inputs.size() == 1) && "The RandomUniformLike operator requires exactly 1 input.", ErrorCode::kINTERNAL_ERROR);
    ASSERT((inputs.at(0).is_tensor()) && "The input tensor cannot be an initializer.",
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    auto const inputShape = shapeOf(input);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = input.getType();

    return randomUniformHelper(ctx, node, inputShape, attrs, dType);
}

NodeImportResult randomNormalHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
     ShapeTensor const& inputShape, OnnxAttrs const& attrs, DataType const& inputDType)
{
    auto* fillLayer = addFill(ctx, inputShape, nvinfer1::FillOperation::kRANDOM_NORMAL);
    ctx->registerLayer(fillLayer, getNodeName(node));

    // Set datatype of output:
    //      RandomNormal: dype is required and defaults to 1
    //      RandomNormalLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int32_t>("dtype", 1);
        switch (dtype)
        {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: fillLayer->setOutputType(0, DataType::kFLOAT); break;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: fillLayer->setOutputType(0, DataType::kHALF); break;
        default: return MAKE_ERROR("Unsupported data type", ErrorCode::kINVALID_VALUE);
        }
    }
    else
    {
        fillLayer->setOutputType(0, inputDType);
    }

    auto const mean = attrs.get<float>("mean", 0.F);
    auto const scale = attrs.get<float>("scale", 1.F); // std dev

    // Set "mean" and "scale" values of the fillLayer.
    fillLayer->setAlpha(mean);
    fillLayer->setBeta(scale);

    // TensorRT does not support "seed" field now. The support will be added in future versions.
    if (attrs.count("seed"))
    {
        LOG_WARNING("TensorRT currently ignores the \"seed\" field in RandomNormal op. Random seeds will be used.");
    }

    RETURN_FIRST_OUTPUT(fillLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomNormal)
{
    OnnxAttrs attrs(node, ctx);
    auto const shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    ShapeTensor const inputShape{1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end())};

    return randomNormalHelper(ctx, node, inputShape, attrs, DataType::kFLOAT);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomNormalLike)
{
    ASSERT(
        (inputs.size() == 1) && "The RandomNormalLike operator requires exactly 1 input.", ErrorCode::kINTERNAL_ERROR);
    ASSERT((inputs.at(0).is_tensor()) && "The input tensor cannot be an initializer.",
        nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    auto const inputShape = shapeOf(input);
    OnnxAttrs const attrs(node, ctx);
    auto const dType = input.getType();

    return randomNormalHelper(ctx, node, inputShape, attrs, dType);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    ASSERT((inputs.at(0).isInt32() || inputs.at(0).isFp32())
            && "This version of TensorRT only supports int32 and float input types for Range!",
        ErrorCode::kUNSUPPORTED_NODE);

    // "start : T
    //     Scalar. First entry for the range of output values.
    //  limit : T
    //     Scalar. Exclusive upper limit for the range of output values.
    //  delta : T
    //     Scalar. Value to step by."
    ShapeTensor const start{ctx, inputs.at(0)};
    ShapeTensor const limit{ctx, inputs.at(1)};
    ShapeTensor const delta{ctx, inputs.at(2)};

    ASSERT((start.isFloat() == limit.isFloat() && start.isFloat() == delta.isFloat())
            && "For range operator types for start, limit, and delta must be identical.",
        ErrorCode::kUNSUPPORTED_NODE);

    // "number_of_elements = max( ceil( (limit - start) / delta ) , 0 )"
    //
    // To implement this in TensorRT using only operations allowed on
    // shape tensors, rewrite as:
    //      "number_of_elements = max(0 - floor((start - limit) / delta), 0)
    //
    ShapeTensor const zero = shapeScalar(0);
    ShapeTensor const fQuotient =  floorDiv(ctx, sub(ctx, start, limit), delta);
    ShapeTensor const quotient = start.isFloat() ? castToInt32(ctx, fQuotient) : fQuotient;
    ShapeTensor const numberOfElements = max(ctx, sub(ctx, zero, quotient), zero);

    nvinfer1::IFillLayer* layer = addFill(ctx, convertTo1D(ctx, numberOfElements), nvinfer1::FillOperation::kLINSPACE);
    ctx->registerLayer(layer, getNodeName(node));

    // TensorRT requires that alpha and beta both be dynamic or both be static.
    if (start.allValuesKnown() && delta.allValuesKnown())
    {
        layer->setAlpha(start[0]);
        layer->setBeta(delta[0]);
        if (!start.isFloat())
        {
            // Set output type to INT32 for ranges that should be INT32, since TRT only accepts
            // double type for setAlpha and setBeta
            layer->setOutputType(0, DataType::kINT32);
        }
    }
    else
    {
        layer->setInput(1, start.tensor(ctx));
        layer->setInput(2, convertTo1D(ctx, delta).tensor(ctx));
        if (inputs.at(0).isInt32())
        {
            layer->setOutputType(0, DataType::kINT32);
        }
    }

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Reciprocal)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

DEFINE_BUILTIN_OP_IMPORTER(ReduceL1)
{
    NodeImportResult abs_result = unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kABS);
    if (abs_result.is_error())
    {
        return abs_result;
    }
    TensorOrWeights abs_input = abs_result.value().at(0);
    return reduceTensor(ctx, node, abs_input, nvinfer1::ReduceOperation::kSUM);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSum);
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSum)
{
    auto sum_result = importReduceSum(ctx, node, inputs);
    if (sum_result.is_error())
    {
        return sum_result;
    }
    TensorOrWeights sum_input = sum_result.value().at(0);
    return unaryHelper(ctx, node, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    // TODO: Abstract this sequence with a function or macro
    auto exp_result = unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
    if (exp_result.is_error())
    {
        return exp_result;
    }
    auto exp_inputs = exp_result.value();
    return importReduceLogSum(ctx, node, exp_inputs);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSumSquare);
DEFINE_BUILTIN_OP_IMPORTER(ReduceL2)
{
    auto sum_sqr_result = importReduceSumSquare(ctx, node, inputs);
    if (sum_sqr_result.is_error())
    {
        return sum_sqr_result;
    }
    TensorOrWeights sum_sqr = sum_sqr_result.value().at(0);
    return unaryHelper(ctx, node, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMax)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMAX);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMean)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kAVG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMin)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMIN);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceProd)
{
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kPROD);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSum)
{
    if (ctx->getOpsetVersion() >= 13 && inputs.size() >= 2)
    {
        return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kSUM, inputs.at(1));
    }
    else
    {
        return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kSUM);
    }
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    auto* sqr_layer = ctx->network()->addElementWise(tensor, tensor, nvinfer1::ElementWiseOperation::kPROD);
    ASSERT(sqr_layer && "Failed to add an ElementWise layer.", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* sqr_tensorPtr = sqr_layer->getOutput(0);
    return reduceTensor(ctx, node, sqr_tensorPtr, nvinfer1::ReduceOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kRELU);
}

DEFINE_BUILTIN_OP_IMPORTER(Sign)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(Round)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kROUND);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t inputRank = input.getDimensions().nbDims;
    ASSERT((inputRank > 0) && "The input tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);
    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);
    ctx->registerLayer(layer, getNodeName(node));
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
    else
    {
        ASSERT((mode == "nearest") && "Invalid Resize mode", ErrorCode::kUNSUPPORTED_NODE);
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

        ASSERT((transformationMode != "tf_half_pixel_for_nn" || nearest_mode == "round_prefer_floor")
                && "This version of TensorRT only support round_prefer_floor nearest mode in tf_half_pixel_for_nn!",
            ErrorCode::kUNSUPPORTED_NODE);

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
        else
        {
            ASSERT(
                !"TensorRT only supports half_pixel, pytorch_half_pixel, tf_half_pixel_for_nn, asymmetric and "
                "align_corners transformation modes!",
                ErrorCode::kUNSUPPORTED_NODE);
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
                auto* resizeShape = &convertToTensor(inputs.at(3), ctx);
                layer->setInput(1, *resizeShape);
                layer->setResizeMode(resizeMode);
                RETURN_FIRST_OUTPUT(layer);
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
        ASSERT((scales_weights.shape.nbDims == 1) && "The scales input must be 1D.", ErrorCode::kUNSUPPORTED_NODE);
        int32_t scaleSize = scales_weights.shape.d[0];
        ASSERT((scaleSize == inputRank) && "The shape of input scales must align with the input rank.",
            ErrorCode::kINVALID_NODE);
        float const* scaleValues = static_cast<float const*>(scales_weights.values);

        // check resize dims
        if (resizeMode == nvinfer1::ResizeMode::kLINEAR)
        {
            ASSERT(canUseNDResize(scaleSize, scaleValues, 3)
                    && "This version of TensorRT only supports linear resizing on the outermost 3 dimensions.",
                ErrorCode::kUNSUPPORTED_NODE);
        }
        else if (resizeMode == nvinfer1::ResizeMode::kCUBIC)
        {
            ASSERT(canUseNDResize(scaleSize, scaleValues, 2)
                    && "This version of TensorRT only supports cubic resizing on the outermost 2 dimensions.",
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

    RETURN_FIRST_OUTPUT(layer);
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
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(ReverseSequence)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    OnnxAttrs attrs{node, ctx};
    int32_t const batch_axis = attrs.get<int32_t>("batch_axis", 1);

    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    auto const dims = input->getDimensions();
    int32_t const rank = dims.nbDims;
    // Sequence tensor: indices tensor of rank = 1 and shape = [batchsize]
    nvinfer1::ITensor* sequences = &convertToTensor(inputs.at(1), ctx);
    std::vector<nvinfer1::ITensor*> tensors;
    // Determine length of batch axis
    int32_t const size = isDynamic(sequences->getDimensions()) ? dims.d[batch_axis] : sequences->getDimensions().d[0];
    ASSERT(size != -1 && "This version of TensorRT does not support dynamic ReverseSequence lengths!",
        ErrorCode::kUNSUPPORTED_NODE);

    for (int32_t i = 0; i < size; i++)
    {

        /*  Slice across each element in batch_axis

        For batch_axis = 1
            Starts =  {0, i, 0, 0...}
            Sizes =   {D0, 1, D2, D3...}
            Strides = {1, 1, 1, ...}

        For batch_axis = 0
            Starts =  {i, 0, 0, 0...}
            Sizes =   {1, D1, D2, D3...}
            Strides = {1, 1, 1, ...}
        */

        ShapeTensor starts = batch_axis == 0 ? concat(ctx, shapeVector(i), shapeVector(0))
                                             : concat(ctx, shapeVector(0), shapeVector(i));
        ShapeTensor sizes = batch_axis == 0
            ? concat(ctx, shapeVector(1), ShapeTensor(*getAxisLength(ctx, input, 1, {1, {1}})))
            : concat(ctx, ShapeTensor(*getAxisLength(ctx, input, 0, {1, {1}})), shapeVector(1));
        ShapeTensor strides = fillShapeVector(ctx, 1, shapeVector(rank));

        for (int32_t j = 2; j < rank; j++)
        {
            starts = concat(ctx, starts, shapeVector(0));
            sizes = concat(ctx, sizes, ShapeTensor(*getAxisLength(ctx, input, j, {1, {1}})));
        }

        auto s1 = addSlice(ctx, *input, starts, sizes, strides);
        nvinfer1::ITensor* data = s1->getOutput(0);
        data = squeezeTensor(ctx, node, *data, {batch_axis});
        // Get sequence length for the current slice
        auto seqIndex = ctx->network()->addSlice(*sequences, {1, {i}}, {1, {1}}, {1, {1}})->getOutput(0);

        // First slice = slices data[seqIndex - 1 : 0 : -1] on axis 0
        /*
            Starts =  {seqIndex - 1, 0, 0 ...}
            Sizes =   {seqIndex, D1, D2, ...}
            Strides = {-1, 1, 1, ...}
        */

        int32_t sliceRank = data->getDimensions().nbDims;
        starts = sub(ctx, ShapeTensor(*seqIndex), shapeVector(1));
        ShapeTensor startsFill = fillShapeVector(ctx, 0, shapeVector(sliceRank - 1));
        starts = concat(ctx, starts, startsFill);

        sizes = ShapeTensor(*seqIndex);
        for (int32_t j = 1; j < sliceRank; j++)
        {
            sizes = concat(ctx, sizes, ShapeTensor(*getAxisLength(ctx, data, j, {1, {1}})));
        }

        strides = shapeVector(-1);
        ShapeTensor stridesFill = fillShapeVector(ctx, 1, shapeVector(sliceRank - 1));
        strides = concat(ctx, strides, stridesFill);

        auto firstSlice = addSlice(ctx, *data, starts, sizes, strides);
        auto slice1 = firstSlice->getOutput(0);

        // Second slice = slices data[seqIndex:end:1] on axis 0

        /*
            Starts =  {seqIndex, 0, 0 ... 0}
            Sizes =   {D0 - seqIndex, D1, D2 ...}
            Strides = {1, 1, 1, 1 ...}
        */

        starts = ShapeTensor(*seqIndex);
        startsFill = fillShapeVector(ctx, 0, shapeVector(sliceRank - 1));
        starts = concat(ctx, starts, startsFill);

        sizes = sub(ctx, ShapeTensor(*getAxisLength(ctx, data, 0, {1, {1}})), ShapeTensor(*seqIndex));
        for (int32_t j = 1; j < sliceRank; j++)
        {
            sizes = concat(ctx, sizes, ShapeTensor(*getAxisLength(ctx, data, j, {1, {1}})));
        }

        strides = fillShapeVector(ctx, 1, shapeVector(sliceRank));

        auto secondSlice = addSlice(ctx, *data, starts, sizes, strides);
        auto slice2 = secondSlice->getOutput(0);

        // Concat the two slices together
        std::vector<nvinfer1::ITensor*> slices{slice1, slice2};
        auto fullSliceLayer = ctx->network()->addConcatenation(slices.data(), slices.size());
        tensors.emplace_back(unsqueezeTensor(ctx, node, *fullSliceLayer->getOutput(0), {batch_axis}));
    }

    auto concatLayer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    concatLayer->setAxis(batch_axis);
    RETURN_FIRST_OUTPUT(concatLayer);
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

    constexpr int32_t NUM_ACTIVATIONS = 1;
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

    // TODO: Support cases where in bidirectional RNNs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
    }

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
    ASSERT(iterationInput && "Failed to add RNN input.", ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(hiddenState, getNodeName(node));
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
    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*initialStateShape(),
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
                       ->getOutput(0),
                  nvinfer1::ElementWiseOperation::kDIV)
              ->getOutput(0);

    if (inputs.size() > 4 && inputs.at(4))
    {
        nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(4), ctx);
        auto maxLen = getAxisLength(ctx, input, 0);
        Ht = numDirections == 2 ? maskBidirRNNHidden(ctx, node, loop, seqLens, maxLen, hiddenState->getOutput(0), Ht, singlePassShape) : maskRNNHidden(ctx, node, loop, seqLens, hiddenState->getOutput(0), Ht, maxLen, direction == "reverse");
    }

    hiddenState->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    outputs.emplace_back(concatenateRNNOutputs(ctx, node, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections, inputs, direction == "reverse"));
    // Yh = last value of H(t)
    outputs.emplace_back(loop->addLoopOutput(*hiddenState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));

    return {{outputs}};
}

DEFINE_BUILTIN_OP_IMPORTER(RoiAlign)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* roisPtr = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* batchIndicesPtr = &convertToTensor(inputs.at(2), ctx);

    // Sanity checking
    auto roiDims = roisPtr->getDimensions();
    ASSERT(roiDims.nbDims == 2 && roiDims.d[1] == 4 && "Found incorrect dimensions for ROIs input!",
        ErrorCode::kUNSUPPORTED_NODE);

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
    auto const plugin = createPlugin(getNodeName(node), importPluginCreator(pluginName, pluginVersion), f);

    ASSERT(plugin != nullptr && "ROIAlign plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* const inputTensorsPtr[3] = {tensorPtr, roisPtr, batchIndicesPtr};
    auto* layer = ctx->network()->addPluginV2(inputTensorsPtr, 3, *plugin);
    ctx->registerLayer(layer, getNodeName(node));

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(ScaledTanh)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSCALED_TANH, &alpha, &beta);
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
        ASSERT(inputs.at(0).isNullTensor() && "TensorRT doesn't support sequence_lens input for this node!",
            ErrorCode::kUNSUPPORTED_NODE);
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
    ctx->registerLayer(stateVars.at(0), getNodeName(node));
    for (int32_t i = 0; i < nbScanInputs; ++i)
    {
        const int32_t index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput = loop->addIterator(convertToTensor(inputs.at(index + opset8Offset), ctx));
        scanInput->setAxis(scanInputAxes.at(i));
        scanInput->setReverse(scanInputDirections.at(i) == 1);
        ctx->registerTensor(TensorOrWeights{scanInput->getOutput(0)}, body.input(index).name());
    }

    // Loop Body. This is handled by dispatching to other op converters.
    CHECK(onnx2trt::parseGraph(ctx, body));

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
    ASSERT((inputs.size() == 2) && "TRT expects two input tensors: grid and input", ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int32_t const inputRank = input.getDimensions().nbDims;
    ASSERT((inputRank > 0) && "The input tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor& grid = convertToTensor(inputs.at(1), ctx);
    int32_t const gridRank = grid.getDimensions().nbDims;
    ASSERT((gridRank > 0) && "The grid tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT((gridRank == inputRank) && "The input tensor and the grid tensor must have the same rank.",
        ErrorCode::kUNSUPPORTED_NODE);

    // Add grid sample layer
    nvinfer1::IGridSampleLayer* layer = ctx->network()->addGridSample(input, grid);
    ctx->registerLayer(layer, getNodeName(node));
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

    ASSERT(layer->setSampleMode(sampleMode), ErrorCode::kINVALID_VALUE);
    layer->setAlignCorners(alignCorners);
    layer->setInterpolationMode(interpolationMode);

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterND)
{
    return addScatterLayer(ctx, node, inputs, nvinfer1::ScatterMode::kND);
}

DEFINE_BUILTIN_OP_IMPORTER(ScatterElements)
{
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int>("axis", 0);
    int32_t nbDims = inputs.at(0).shape().nbDims;
    CHECK(convertAxis(axis, nbDims));
    return addScatterLayer(ctx, node, inputs, nvinfer1::ScatterMode::kELEMENT, axis);
}

DEFINE_BUILTIN_OP_IMPORTER(Scatter)
{
    // Scatter was deprecated in Opset 11 and replaced by ScatterElements
    if (ctx->getOpsetVersion() >= 11)
    {
        LOG_WARNING("Scatter was deprecated in Opset 11. Node: \"" << getNodeName(node) << "\" will be converted to ScatterElements.");
    }

    return importScatterElements(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Selu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.6732f);
    float beta = attrs.get("gamma", 1.0507f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSELU, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Shape)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    auto* layer = ctx->network()->addShape(input);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSIGMOID);
}

DEFINE_BUILTIN_OP_IMPORTER(Sin)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kSIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Sinh)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kSINH);
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
        ASSERT( (nbInputs >= 3 && nbInputs <= 5) && "Post-opset 10 Slice operator requires 3 - 5 inputs.", ErrorCode::kUNSUPPORTED_NODE);
        starts = ShapeTensor{ctx, inputs.at(1)};
        ends = ShapeTensor{ctx, inputs.at(2)};
        // "If axes are omitted, they are set to [0, ..., ndim-1]."
        axes = nbInputs > 3 ? ShapeTensor(ctx, inputs.at(3)) : iotaShapeVector(dims.size());
        // Doesn't support dynamic axes currently.
        ASSERT( (axes.allValuesKnown()) && "This version of TensorRT does not support dynamic axes.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT( (starts.size() == axes.size()) && "The shape of input starts misaligns with the shape of input axes.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(ends.size() == axes.size() && "The shape of input ends misaligns with the shape of input axes.", ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT((-r <= axis && axis < r) && "The range of axis must be in [-r, r-1], where r is the rank of input data.",
            ErrorCode::kINVALID_VALUE);
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
    ASSERT( (std::unordered_set<int64_t>(axes.begin(), axes.end()).size() == static_cast<size_t>(axes.size()))
            && "No duplicated axes are allowed.",
        ErrorCode::kINVALID_NODE);

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

    ctx->registerLayer(slice, getNodeName(node));

    RETURN_FIRST_OUTPUT(slice);
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    auto& input = convertToTensor(inputs.at(0), ctx);
    auto* softmax = addSoftmax(ctx, node, input);
    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *softmax, shapeOf(input));
    RETURN_FIRST_OUTPUT(reshapeLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Softsign)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(Softplus)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS);
}

DEFINE_BUILTIN_OP_IMPORTER(SpaceToDepth)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    // Input tensor is in NCHW format
    ASSERT((inputs.at(0).shape().nbDims == 4) && "The input tensor must be in the NCHW format.",
        ErrorCode::kUNSUPPORTED_NODE);
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
    ctx->registerLayer(firstShuffle, getNodeName(node));
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
                int32_t const* splitValues = static_cast<int32_t const*>(splitWeights.values);
                for (size_t i = 0; i < splitWeights.count(); i++)
                {
                    splitList.push_back(splitValues[i]);
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
        ASSERT((splitList.empty() || (splitList.size() == numOutputs))
                && "The shape of the split attribute misaligns with the number of outputs.",
            ErrorCode::kINVALID_NODE);
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
        ctx->registerLayer(slice, getNodeName(node));
        outputs.emplace_back(slice->getOutput(0));
    }

    return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
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
            int32_t* axesValues = static_cast<int32_t*>(axesWeights.values);
            for (size_t i = 0; i < axesWeights.count(); i++)
            {
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
        ASSERT(!isDynamic(shape) && "Cannot infer squeeze dimensions from a dynamic shape! Please re-export your model with the Squeeze axes input set.", ErrorCode::kUNSUPPORTED_NODE);
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

    ASSERT(squeezed && "Failed to squeeze tensor!", ErrorCode::kUNSUPPORTED_NODE);

    return {{squeezed}};
}

DEFINE_BUILTIN_OP_IMPORTER(Sub)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUB);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Tan)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kTAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kTANH);
}

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kTHRESHOLDED_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Tile)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // "input : T
    // Input tensor of any shape."
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    auto const inputDims = shapeOf(input);
    ASSERT(input.getType() != DataType::kUINT8, ErrorCode::kUNSUPPORTED_NODE);

    // "repeats : T1
    // 1D int64 tensor of the same length as input's dimension number,
    // includes numbers of repeated copies along input's dimensions.
    ShapeTensor const repeats{ctx, inputs.at(1)};

    ShapeTensor outputShape = mul(ctx, inputDims, repeats);
    nvinfer1::ISliceLayer* tile
        = addSlice(ctx, input, similar(ctx, inputDims, 0), outputShape, similar(ctx, inputDims, 1));
    ctx->registerLayer(tile, getNodeName(node));
    tile->setMode(nvinfer1::SliceMode::kWRAP);

    RETURN_FIRST_OUTPUT(tile);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "UINT8"}));
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get("axis", -1);
    int32_t k;
    if (ctx->getOpsetVersion() >= 10)
    {
        assertIsWeights(inputs.at(1), "This version of TensorRT only supports input K as an initializer.");
        ASSERT((inputs.at(1).weights().count() == 1) && "The input K must contain exactly 1 value.",
            ErrorCode::kUNSUPPORTED_NODE);
        k = *static_cast<int*>(inputs.at(1).weights().values);
    }
    else
    {
        ASSERT( (attrs.count("k")) && "Attribute k is missing.", ErrorCode::kINVALID_NODE);
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
        ASSERT(tensorPtr && "Failed to unsqueeze input x.", ErrorCode::kUNSUPPORTED_NODE);
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
    ctx->registerLayer(layer, getNodeName(node));
    ASSERT(layer && "Failed to add TopK layer.", ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* values = layer->getOutput(0);
    nvinfer1::ITensor* indices = layer->getOutput(1);

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> axes{1};
        values = squeezeTensor(ctx, node, *values, axes);
        ASSERT(values && "Failed to squeeze the input values.", ErrorCode::kUNSUPPORTED_NODE);
        indices = squeezeTensor(ctx, node, *indices, axes);
        ASSERT(indices && "Failed to squeeze the input indices.", ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{values, indices}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node, ctx);
    int32_t ndim = input.shape().nbDims;
    ASSERT((ndim <= nvinfer1::Dims::MAX_DIMS)
            && "The rank of the input tensor exceeds the maximum supported by this version of TensorRT.",
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Permutation default_perm; // Default is to reverse dims
    for (int32_t i = 0; i < ndim; ++i)
    {
        default_perm.order[i] = ndim - 1 - i;
    }
    nvinfer1::Permutation perm = attrs.get("perm", default_perm);
    nvinfer1::ITensor& itensor = input.is_tensor() ? input.tensor() : convertToTensor(input, ctx);
    nvinfer1::ITensor* output_tensor = transposeTensor(ctx, node, itensor, perm);
    ASSERT(output_tensor && "Failed to transpose the input.", ErrorCode::kUNSUPPORTED_NODE);
    return {{output_tensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(Trilu)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    // Data Tensor
    using eOp = nvinfer1::ElementWiseOperation;
    auto* data = &convertToTensor(inputs.at(0), ctx);
    auto const nbDims = data->getDimensions().nbDims;
    ASSERT((nbDims == 2 || nbDims == 3) && "Trilu input must have 2 or 3 input dimensions!", ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    int32_t const upper = attrs.get("upper", 0);

    // Input may be in a batch so we need to get NxM dimensions
    int64_t const N = nbDims == 2 ? 0 : 1;
    int64_t const M = nbDims == 2 ? 1 : 2;

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
        cols = &elementwiseHelper(ctx, node, {cols, k}, eOp::kSUB).value().at(0).tensor();
    }

    // Unsqueeze to broadcast rows/cols to 3D if necessary during next elementwise operation
    if (nbDims == 3)
    {
        rows = unsqueezeTensor(ctx, node, *rows, {0});
        cols = unsqueezeTensor(ctx, node, *cols, {0});
    }

    // For lower Trilus, use greaterOrEquals. For upper Trilus, use lessOrEquals
    bool const greater = upper == 0 ? true : false;
    auto* condition = greaterLessOrEqual(ctx, node, rows, cols, greater);
    auto* result = ctx->network()->addSelect(*condition, *data, *zero);

    RETURN_FIRST_OUTPUT(result);
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
            ASSERT(axesInput.allValuesKnown() && "Axes input for unsqueeze operation should be a constant tensor.",
                ErrorCode::kUNSUPPORTED_NODE);
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

    ASSERT(expanded && "Failed to unsqueeze tensor!", ErrorCode::kUNSUPPORTED_NODE);

    return {{expanded}};
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    CHECK(notInvalidType(inputs.at(0), {"BOOL", "UINT8"}));
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    int32_t const nbDims = tensor.getDimensions().nbDims;
    ASSERT((nbDims > 0) && "The input tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT((mode == "nearest" || mode == "linear" || mode == "bilinear")
            && "The attribute mode can only be nearest, linear, or bilinear.",
        ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode
        = (mode == "linear" || mode == "bilinear") ? nvinfer1::ResizeMode::kLINEAR : nvinfer1::ResizeMode::kNEAREST;

    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ASSERT((inputs.size() == 2) && "Operator Upsample requires exactly 2 inputs.", ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        if (scales_input.is_weights())
        {
            // TRT-15340: Remove this and use else path when safety support nbDims == 1.
            ShapedWeights scales_weights = scales_input.weights();
            ASSERT((scales_weights.shape.nbDims == 1) && "The scales input must be 1D.", ErrorCode::kUNSUPPORTED_NODE);
            // Scale factors has batch dimension.
            ASSERT((scales_weights.count() == static_cast<size_t>(nbDims))
                    && "The shape of the scales input must align with the dimensions of the input.",
                ErrorCode::kUNSUPPORTED_NODE);
            ASSERT((scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT)
                    && "This version of TensorRT only supports FLOAT scales input.",
                ErrorCode::kINVALID_NODE);
            float const* scales_ptr = static_cast<float const*>(scales_weights.values);
            std::vector<float> scale_factors(nbDims, 1.0F);
            for (int32_t i = 0; i < nbDims; i++)
            {
                scale_factors[i] = scales_ptr[i];
            }
            if (mode == "linear" || mode == "bilinear")
            {
                ASSERT(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3)
                        && "This version of TensorRT only supports linear resizing on the outermost 3 dimensions",
                    ErrorCode::kUNSUPPORTED_NODE);
            }
            layer->setScales(scale_factors.data(), nbDims);
        }
        else
        {
            nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, tensor, scales_input);
            nvinfer1::Dims const outDims = resizeShape->getDimensions();
            ASSERT((outDims.nbDims == 1) && "The scales input must be 1D.", ErrorCode::kUNSUPPORTED_NODE);
            // Scale factors has batch dimension.
            ASSERT((outDims.d[0] == nbDims)
                    && "The shape of the scales input must align with the dimensions of the input.",
                ErrorCode::kUNSUPPORTED_NODE);
            ASSERT((resizeShape->getType() == DataType::kINT32) && "Resize output shape type must be integral.",
                ErrorCode::kINVALID_NODE);
            layer->setInput(1, *resizeShape);
        }
    }
    else
    {
        // TRT-15340: Adapt to use resizeShapeTensor instead when safety support nbDims == 1.
        ASSERT(attrs.count("scales") && "Attribute scales is missing.", ErrorCode::kUNSUPPORTED_NODE);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ASSERT((static_cast<int32_t>(scales.size()) == nbDims)
                && "The shape of the scales input must align with the dimensions of the input.",
            ErrorCode::kUNSUPPORTED_NODE);
        std::vector<float> scale_factors(nbDims, 1.0F);
        for (int32_t i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales[i];
        }
        if (mode == "linear" || mode == "bilinear")
        {
            ASSERT(canUseNDResize(scale_factors.size(), &scale_factors.front(), 3)
                    && "This version of TensorRT only supports linear resizing on the outermost 3 dimensions",
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scale_factors.data(), nbDims);
    }
    ctx->registerLayer(layer, getNodeName(node));
    layer->setResizeMode(resizeMode);
    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
    layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    RETURN_FIRST_OUTPUT(layer);
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

    ASSERT( (cDims.nbDims == xDims.nbDims) && "The shape of the condition input tensor must be the same of the input x tensor." , ErrorCode::kUNSUPPORTED_NODE);
    ASSERT( (cDims.nbDims == yDims.nbDims) && "The shape of the condition input tensor must be the same of the input y tensor.", ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addSelect(*condition, *x, *y);
    ctx->registerLayer(layer, getNodeName(node));

    RETURN_FIRST_OUTPUT(layer);
}

// Copies the given field into the fieldData map, returns data and number of T elements in the vector in which the data
// was copied into.
template <typename T>
std::tuple<void const*, size_t> copyField(
    T const& field, std::string const& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    constexpr size_t nbBytes{sizeof(T)};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), &field, nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size() / nbBytes);
}

template <typename T>
std::tuple<void const*, size_t> copyField(
    std::vector<T> const& repeatedField, std::string const& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    size_t const nbBytes{sizeof(T) * repeatedField.size()};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), repeatedField.data(), nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size() / sizeof(T));
}

std::tuple<void const*, size_t> copyField(
    std::string const& field, std::string const& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    static_assert(sizeof(std::string::value_type) == sizeof(uint8_t), "String type does not have 1 byte elements");
    std::copy(field.begin(), field.end(), std::back_inserter(fieldData[fieldName]));
    // Append \0 as end of C style string.
    fieldData[fieldName].push_back('\0');
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<void const*, size_t> copyField(std::vector<std::string> const& repeatedField, std::string const& fieldName,
    string_map<std::vector<uint8_t>>& fieldData)
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
    ShapedWeights const& field, std::string const& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    // Weights do not require a copy
    return std::make_tuple(field.values, field.count());
}

// Load plugin fields from an ONNX node, using fieldData for temporary allocations.
std::vector<nvinfer1::PluginField> loadFields(string_map<std::vector<uint8_t>>& fieldData, OnnxAttrs const& attrs,
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
                MAKE_ERROR(
                    "Attributes of type: " + ::ONNX_NAMESPACE::AttributeProto::AttributeType_Name(attrs.type(fieldName))
                        + " are unsupported",
                    ErrorCode::kUNSUPPORTED_NODE);
            }
            fields.emplace_back(fieldName.c_str(), data, type, length);
    }
    return fields;
}

DEFINE_BUILTIN_OP_IMPORTER(Xor)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kXOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Shrink)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);

    auto originalType = x->getType();
    ASSERT( (originalType == DataType::kFLOAT || originalType == DataType::kHALF || originalType == DataType::kINT8 || originalType == DataType::kINT32)
        && "Only FLOAT, HALF, INT8 and INT32 are supported in Shrink.", ErrorCode::kUNSUPPORTED_NODE);
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
        = &elementwiseHelper(ctx, node, {x, lambdTensor}, nvinfer1::ElementWiseOperation::kGREATER)
               .value()
               .at(0)
               .tensor();
    nvinfer1::ITensor* xMinusBias
        = &elementwiseHelper(ctx, node, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUB).value().at(0).tensor();
    nvinfer1::ITensor* output = ctx->network()->addSelect(*xGreaterThanLambd, *xMinusBias, *zeroTensor)->getOutput(0);

    // If x < -lambd, y = x + bias;
    nvinfer1::ITensor* xLessThanMinusLambd
        = &elementwiseHelper(ctx, node, {x, negLambdTensor}, nvinfer1::ElementWiseOperation::kLESS)
               .value()
               .at(0)
               .tensor();
    nvinfer1::ITensor* xAddBias
        = &elementwiseHelper(ctx, node, {x, biasTensor}, nvinfer1::ElementWiseOperation::kSUM).value().at(0).tensor();

    auto* layer = ctx->network()->addSelect(*xLessThanMinusLambd, *xAddBias, *output);
    ctx->registerLayer(layer, getNodeName(node));

    // cast back to originalType
    return {{castHelper(ctx, layer->getOutput(0), originalType)}};
}


DEFINE_BUILTIN_OP_IMPORTER(HardSwish)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    ASSERT((x->getType() == DataType::kFLOAT || x->getType() == DataType::kHALF || x->getType() == DataType::kINT8)
        && "Only FLOAT, HALF or INT8 input is supported for the HardSwish operator in this version of TensorRT",
        ErrorCode::kUNSUPPORTED_NODE);

    // activationHelper does not support const and constexpr (compile failed)
    float kALPHA{1.F / 6};
    float kBETA{0.5F};
    nvinfer1::ITensor* hardSigmoid =
        &activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &kALPHA, &kBETA).value().at(0).tensor();

    return elementwiseHelper(ctx, node, {x, hardSigmoid}, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(NonZero)
{
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(0), ctx);
    ASSERT((x->getType() == DataType::kFLOAT || x->getType() == DataType::kHALF || x->getType() == DataType::kINT32 || x->getType() == DataType::kINT8 || x->getType() == DataType::kBOOL)
        && "Only FLOAT, HALF, INT32, INT8 or BOOL input is supported for the NonZero operator in this version of TensorRT", ErrorCode::kUNSUPPORTED_NODE);
    auto* layer = ctx->network()->addNonZero(*x);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
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
    nvinfer1::IPluginCreator* creator = importPluginCreator(pluginName, pluginVersion, pluginNamespace);
    ASSERT(creator && "Plugin not found, are the plugin name, version, and namespace correct?",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::PluginFieldCollection const* fieldNames = creator->getFieldNames();
    // Field data needs to be type erased, we use fieldData for temporary allocations.
    string_map<std::vector<uint8_t>> fieldData{};
    std::vector<nvinfer1::PluginField> fields = loadFields(fieldData, attrs, fieldNames, ctx);

    auto const plugin = createPlugin(getNodeName(node), creator, fields);
    ASSERT(plugin && "Could not create plugin", ErrorCode::kUNSUPPORTED_NODE);

    std::vector<nvinfer1::ITensor*> pluginInputs{};
    for (auto& input : inputs)
    {
        pluginInputs.emplace_back(&convertToTensor(input, ctx));
    }
    LOG_INFO("Successfully created plugin: " << pluginName);
    auto* layer = ctx->network()->addPluginV2(pluginInputs.data(), pluginInputs.size(), *plugin);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_ALL_OUTPUTS(layer);
}

// INetwork Serialization importer functions - TODO: Move to it's own file?

DEFINE_BUILTIN_OP_IMPORTER(TRT_Scale)
{
    ASSERT( (inputs.size() >= 1) && "Input is required.", nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT( (inputs.at(0).is_tensor()) && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    if (inputs.size() >= 2)
    {
        ASSERT( (inputs.at(1).is_weights()) && "The second input must be an initializer.", nvonnxparser::ErrorCode::kINVALID_NODE);
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
        ASSERT( (inputs.at(counter).is_weights()) && "The scale input must be an initializer.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        scale = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("shift"))
    {
        ASSERT( (inputs.at(counter).is_weights()) && "The shift input must be an initializer.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        shift = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("power"))
    {
        ASSERT( (inputs.at(counter).is_weights())  && "The power input must be an initializer.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        power = inputs.at(counter++).weights();
    }

    nvinfer1::IScaleLayer* layer = ctx->network()->addScale(input, mode, shift, scale, power);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");
    bool zeroIsPlaceholder = attrs.get<bool>("zero_is_placeholder");

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    ctx->registerLayer(layer, getNodeName(node));
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
        ASSERT(inputs.at(1).is_tensor() && "The second input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_TopK_Min)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    ASSERT(inputs.at(1).is_weights() && "The second input must be an initializer.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kWeights = inputs.at(1).weights();
    int k = *static_cast<int*>(kWeights.values);

    int32_t axes = 1 << (attrs.get<int32_t>("axis"));

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(input, nvinfer1::TopKOperation::kMIN, k, axes);
    ctx->registerLayer(layer, getNodeName(node));

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MatMul)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor() && "The second input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input0 = inputs.at(0).tensor();
    auto& input1 = inputs.at(1).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::MatrixOperation op0 = attrs.get<nvinfer1::MatrixOperation>("op_0");
    nvinfer1::MatrixOperation op1 = attrs.get<nvinfer1::MatrixOperation>("op_1");

    nvinfer1::IMatrixMultiplyLayer* layer = ctx->network()->addMatrixMultiply(input0, op0, input1, op1);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
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

    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    int32_t counter = 1;
    nvinfer1::IRNNv2Layer* layer = ctx->network()->addRNNv2(input, layerCount, hiddenSize, maxSeqLen, op);
    ctx->registerLayer(layer, getNodeName(node));
    layer->setInputMode(inputMode);
    layer->setDirection(direction);

    if (attrs.get<bool>("has_hidden_state"))
    {
        ASSERT(inputs.at(counter).is_tensor() && "The input hidden_state must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setHiddenState(inputs.at(counter++).tensor());
    }
    if (op == nvinfer1::RNNOperation::kLSTM && attrs.get<bool>("has_cell_state", false))
    {
        ASSERT(inputs.at(counter).is_tensor() && "The input cell_state must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setCellState(inputs.at(counter++).tensor());
    }
    if (attrs.get<bool>("has_seq_lengths"))
    {
        ASSERT(inputs.at(counter).is_tensor() && "The input seq_lengths must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
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
            ASSERT(addRNNv2Weights(weightsAdder, n, gates, inputs, counter) && "Failed to add weights to the RNN layer.", nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        ASSERT(addRNNv2Weights(recurrentWeightsAdder, n, gates, inputs, counter) && "Failed to add recurrent weights to the RNN layer.", nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(biasAdder, n, gates, inputs, counter) && "Failed to add bias to the RNN layer.", nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(recurrentBiasAdder, n, gates, inputs, counter) && "Failed to add recurrent bias to the RNN layer.", nvonnxparser::ErrorCode::kINVALID_NODE);
    }

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RaggedSoftmax)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor() && "The second input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();
    auto& bounds = inputs.at(1).tensor();

    nvinfer1::IRaggedSoftMaxLayer* layer = ctx->network()->addRaggedSoftMax(input, bounds);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FullyConnected)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int32_t nbChannels = attrs.get<int32_t>("channels");

    ASSERT(inputs.at(1).is_weights() && "The input kernel must be an initializer.",
        nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kernelWeights = inputs.at(1).weights();

    ShapedWeights biasWeights = ShapedWeights::empty(kernelWeights.type);
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights() && "The input bias must be an initializer.", nvonnxparser::ErrorCode::kINVALID_NODE);
        biasWeights = inputs.at(2).weights();
    }

    nvinfer1::IFullyConnectedLayer* layer
        = ctx->network()->addFullyConnected(input, nbChannels, kernelWeights, biasWeights);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kINVALID_NODE);
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
    ASSERT(layer && "Failed to create a Pooling layer.", ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, getNodeName(node));
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(exclude_padding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    layer->setBlendFactor(blend);

    RETURN_FIRST_OUTPUT(layer);
}

#if ENABLE_STD_PLUGIN
DEFINE_BUILTIN_OP_IMPORTER(TRT_PluginV2)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        ASSERT(input.is_tensor() && "The input must be a tensor.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        tensors.push_back(&input.tensor());
    }
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IPluginRegistry* registry = getPluginRegistry();

    std::string name = attrs.get<std::string>("name");
    std::string version = attrs.get<std::string>("version");
    std::string nspace = attrs.get<std::string>("namespace");
    std::string buffer = attrs.get<std::string>("data");

    nvinfer1::IPluginCreator* creator = registry->getPluginCreator(name.c_str(), version.c_str(), nspace.c_str());
    ASSERT(creator && "Plugin not found, are the plugin name, version, and namespace correct?",
        nvonnxparser::ErrorCode::kINVALID_NODE);

    auto const plugin = creator->deserializePlugin("", buffer.data(), buffer.size());

    nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_ALL_OUTPUTS(layer);
}
#endif // ENABLE_STD_PLUGIN

DEFINE_BUILTIN_OP_IMPORTER(TRT_Gather)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(1).is_tensor() && "The second input must be a tensor.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& data = inputs.at(0).tensor();
    auto& indices = inputs.at(1).tensor();
    OnnxAttrs attrs(node, ctx);
    int32_t axis = attrs.get<int32_t>("axis", 0);
    int32_t nbElementWiseDims = attrs.get<int32_t>("nbElementWiseDims", 0);
    int32_t r = data.getDimensions().nbDims;

    ASSERT((indices.getType() == DataType::kINT32)
            && "This version of TensorRT only supports INT32 input indices.",
        nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT((r >= 1) && "0D input data is not allowed.", nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT( (-r <= axis && axis < r) && "The attribute axis should be in range [-r, r-1], where r is the rank of the input." , nvonnxparser::ErrorCode::kINVALID_NODE);

    if (axis < 0)
    {
        axis += r;
    }

    nvinfer1::IGatherLayer* layer = ctx->network()->addGather(data, indices, axis);
    ctx->registerLayer(layer, getNodeName(node));
    layer->setNbElementWiseDims(nbElementWiseDims);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Slice)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT((inputs.size() == 4) && "Exactly 4 inputs are required by TRT_Slice.",
            nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        ShapeTensor const start{ctx, inputs.at(1)};
        ShapeTensor const size{ctx, inputs.at(2)};
        ShapeTensor const stride{ctx, inputs.at(3)};
        layer = addSlice(ctx, input, start, size, stride);
    }
    ctx->registerLayer(layer, getNodeName(node));
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Resize)
{
    ASSERT(inputs.at(0).is_tensor() && "The first input must be a tensor.", nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = ctx->network()->addResize(input);
    ctx->registerLayer(layer, getNodeName(node));

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
            ASSERT((scales.size() > 0) && "Attribute scales is missing.", nvonnxparser::ErrorCode::kINVALID_NODE);
            layer->setScales(&scales[0], scales.size());
        }
    }
    else
    {
        ASSERT( (inputs.at(1).is_tensor()) && "The output dimension input must be a tensor." , nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FloorDiv)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Conv)
{
    return importConv(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Deconv)
{
    return importConvTranspose(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxPool)
{
    return importMaxPool(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_AveragePool)
{
    return importAveragePool(ctx, node, inputs);
}

} // namespace

} // namespace onnx2trt
