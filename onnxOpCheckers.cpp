/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxOpCheckers.hpp"
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
#include "importerUtils.hpp"

#include <array>
#include <iostream>
#include <iterator>
#include <tuple>

namespace onnx2trt
{

StringMap<OpStaticErrorChecker>& getOpStaticErrorCheckerMap()
{
    static StringMap<OpStaticErrorChecker> error_checkers;
    return error_checkers;
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

#define DECLARE_OP_CHECKER(op)                                                                                         \
    void check##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors,         \
        size_t const nodeIndex)

#define DEFINE_OP_CHECKER(op)                                                                                          \
    void check##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors,         \
        size_t const nodeIndex);                                                                                       \
    static bool const op##_registered_op_checker = registerOpStaticErrorChecker(#op, check##op);                       \
    IGNORE_UNUSED_GLOBAL(op##_registered_op_checker);                                                                  \
    void check##op(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors,         \
        size_t const nodeIndex)

#define DEFINE_OP_EMPTY_CHECKER(op)                                                                                    \
    DEFINE_OP_CHECKER(op) {}

bool registerOpStaticErrorChecker(std::string op, OpStaticErrorChecker const& checker)
{
    bool inserted = getOpStaticErrorCheckerMap().insert({op, checker}).second;
    assert(inserted);
    return inserted;
}

void argMinMaxCheckHelper(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, std::vector<Status>& errors, size_t const nodeIndex)
{
    OnnxAttrs attrs(node, ctx);
    int32_t selectLastIndex = attrs.get<int32_t>("select_last_index", 0);
    STATIC_CHECK((!selectLastIndex || (selectLastIndex && ctx->getOpsetVersion() >= 12))
            && "Per-opset 12 ONNX does not support the select_last_index attribute.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

void poolingCheckHelper(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, std::vector<Status>& errors, size_t const nodeIndex)
{
    OnnxAttrs attrs(node, ctx);
    if (ctx->getOpsetVersion() >= 10)
    {
        auto const dilations = attrs.get<std::vector<int32_t>>("dilations", std::vector<int32_t>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            STATIC_CHECK((dilations[i] == 1) && "This version of TensorRT does not support dilations other than 1.",
                ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

void randomUniformCheckHelper(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors, size_t const nodeIndex)
{
    OnnxAttrs attrs(node, ctx);

    // Set datatype of output:
    //      RandomUniform: dype is required and defaults to 1
    //      RandomUniformLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int32_t>("dtype", 1);
        if (dtype != ::ONNX_NAMESPACE::TensorProto::FLOAT && dtype != ::ONNX_NAMESPACE::TensorProto::FLOAT16)
        {
            ADD_STATIC_ERROR(
                "Unsupported data type in randomUniform", ErrorCode::kINVALID_VALUE, node, nodeIndex, errors);
        }
    }
}

void randomNormalCheckHelper(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors, size_t const nodeIndex)
{
    OnnxAttrs attrs(node, ctx);

    // Set datatype of output:
    //      RandomNormal: dype is required and defaults to 1
    //      RandomNormalLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int32_t>("dtype", 1);
        if (dtype != ::ONNX_NAMESPACE::TensorProto::FLOAT && dtype != ::ONNX_NAMESPACE::TensorProto::FLOAT16)
        {
            ADD_STATIC_ERROR(
                "Unsupported data type in randomNormal", ErrorCode::kINVALID_VALUE, node, nodeIndex, errors);
        }
    }
}

void emptyOutputChecker(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors,
    size_t const nodeIndex, int32_t numSupportedOutputs)
{
    int32_t numOutputs = node.output().size();
    for (int32_t i = 0; i < numOutputs; i++)
    {
        if (i < numSupportedOutputs)
        {
            continue;
        }
        if (!node.output(i).empty())
        {
            std::ostringstream ssMsg{};
            ssMsg << "This version of TensorRT doesn't support mode than " << numSupportedOutputs << " outputs for "
                  << node.op_type() << " nodes!";
            ADD_STATIC_ERROR(ssMsg.str(), ErrorCode::kUNSUPPORTED_NODE, node, nodeIndex, errors);
        }
    }
}

DEFINE_OP_EMPTY_CHECKER(Abs)

DEFINE_OP_EMPTY_CHECKER(Acos)

DEFINE_OP_EMPTY_CHECKER(Acosh)

DEFINE_OP_EMPTY_CHECKER(And)

DEFINE_OP_EMPTY_CHECKER(Asin)

DEFINE_OP_EMPTY_CHECKER(Asinh)

DEFINE_OP_EMPTY_CHECKER(Atan)

DEFINE_OP_EMPTY_CHECKER(Atanh)

DEFINE_OP_CHECKER(Attention)
{
    OnnxAttrs attrs(node, ctx);

    // Check for unsupported attributes.
    float const softcap = attrs.get<float>("softcap", 0.0);
    STATIC_CHECK(softcap == 0.0 && "Setting softcap is not supported in TensorRT Attention.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);

    bool const hasSoftmaxPrecision = static_cast<bool>(attrs.count("softmax_precision"));
    STATIC_CHECK(!hasSoftmaxPrecision && "Setting softmax precision is not supported in TensorRT Attention.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(TRT_QuantizedAttention)

DEFINE_OP_EMPTY_CHECKER(Add)

DEFINE_OP_CHECKER(ArgMax)
{
    argMinMaxCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(ArgMin)
{
    argMinMaxCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(AveragePool)
{
    poolingCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(BatchNormalization)
{
    OnnxAttrs attrs(node, ctx);
    auto const isTraining = attrs.get<int32_t>("training_mode", 0);
    STATIC_CHECK(!isTraining && "This version of TensorRT does not support training_mode == 1 in BatchNormalization.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    // Also check the number of outputs. TRT only supports the first output.
    emptyOutputChecker(ctx, node, errors, nodeIndex, 1);
}

DEFINE_OP_EMPTY_CHECKER(BlackmanWindow)

DEFINE_OP_CHECKER(Cast)
{
    OnnxAttrs attrs(node, ctx);
    // Get data type to cast to.
    auto onnxType = attrs.get<int32_t>("to");
    DataType newType{DataType::kFLOAT};
    STATIC_CHECK(convertDtype(onnxType, &newType) && "Unsupported data type for the Cast operator!",
        ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(CastLike)

DEFINE_OP_EMPTY_CHECKER(Ceil)

DEFINE_OP_EMPTY_CHECKER(Celu)

DEFINE_OP_EMPTY_CHECKER(Clip)

DEFINE_OP_EMPTY_CHECKER(Concat)

DEFINE_OP_CHECKER(Constant)
{
    OnnxAttrs attrs(node, ctx);
    // Having the trt_outputs_range_min attributes means it's from
    // serialized iNetworkDefinition which does not have this check.
    if (attrs.get<std::vector<float>>("trt_outputs_range_min", {}).empty())
    {
        STATIC_CHECK((!attrs.count("sparse_value")) && (!attrs.count("value_string")) && (!attrs.count("value_strings"))
            && "This version of TensorRT does not support the sparse_value, value_string and value_strings attributes.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_CHECKER(ConstantOfShape)
{
    OnnxAttrs attrs(node, ctx);
    ShapedWeights zeroWeights
        = ctx->createNamedTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, nvinfer1::Dims{1, {1}});
    static_cast<float*>(zeroWeights.values)[0] = 0.f;
    auto valueWeights = TensorOrWeights{attrs.get("value", zeroWeights)};
    STATIC_CHECK(notInvalidType(valueWeights, {"UINT8"}, node, nodeIndex).is_success()
            && "Invalid input type for ConstantOfShape",
        ErrorCode::kUNSUPPORTED_NODE_DATATYPE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(Conv)

DEFINE_OP_EMPTY_CHECKER(ConvTranspose)

DEFINE_OP_EMPTY_CHECKER(Cos)

DEFINE_OP_EMPTY_CHECKER(Cosh)

DEFINE_OP_EMPTY_CHECKER(CumSum)

DEFINE_OP_EMPTY_CHECKER(DeformConv)

DEFINE_OP_EMPTY_CHECKER(DepthToSpace)

DEFINE_OP_EMPTY_CHECKER(QuantizeLinear)

DEFINE_OP_EMPTY_CHECKER(DequantizeLinear)



DEFINE_OP_EMPTY_CHECKER(TRT_FP8QuantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_FP8DequantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_INT4QuantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_INT4DequantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_FP4DynamicQuantize)

DEFINE_OP_EMPTY_CHECKER(TRT_DynamicQuantize)

DEFINE_OP_EMPTY_CHECKER(TRT_BlockQuantize)

DEFINE_OP_EMPTY_CHECKER(TRT_BlockDequantize)

DECLARE_OP_CHECKER(Mul);

DEFINE_OP_EMPTY_CHECKER(Div)

DEFINE_OP_CHECKER(Dropout)
{
    // TensorRT does not support the Dropout operator with training mode.
    if (ctx->getOpsetVersion() <= 6)
    {
        OnnxAttrs attrs(node, ctx);
        int32_t isTestingMode = attrs.get<int32_t>("is_test", 1);
        STATIC_CHECK(isTestingMode && "TensorRT does not support the Droupout operator with training mode.",
            ErrorCode::kUNSUPPORTED_NODE_ATTR, node, errors, nodeIndex);
    }
}

DEFINE_OP_CHECKER(Einsum)
{
    OnnxAttrs attrs(node, ctx);
    std::string equation = attrs.get<std::string>("equation");
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
        ADD_STATIC_ERROR("Invalid character(s) in Einsum equation: " + invalidCharacters, ErrorCode::kINVALID_NODE,
            node, nodeIndex, errors);
    }
}

DEFINE_OP_EMPTY_CHECKER(Elu)

DEFINE_OP_EMPTY_CHECKER(Equal)

DEFINE_OP_EMPTY_CHECKER(Erf)

DEFINE_OP_EMPTY_CHECKER(Exp)

DEFINE_OP_EMPTY_CHECKER(Expand)

DEFINE_OP_EMPTY_CHECKER(EyeLike)

DEFINE_OP_EMPTY_CHECKER(Flatten)

DEFINE_OP_EMPTY_CHECKER(Floor)

DEFINE_OP_EMPTY_CHECKER(Gather)

DEFINE_OP_EMPTY_CHECKER(GatherElements)

DEFINE_OP_EMPTY_CHECKER(GatherND)

DEFINE_OP_EMPTY_CHECKER(Gelu)

DEFINE_OP_EMPTY_CHECKER(Gemm)

DEFINE_OP_EMPTY_CHECKER(GlobalAveragePool)

DEFINE_OP_EMPTY_CHECKER(GlobalLpPool)

DEFINE_OP_EMPTY_CHECKER(GlobalMaxPool)

DEFINE_OP_EMPTY_CHECKER(Greater)

DEFINE_OP_EMPTY_CHECKER(GreaterOrEqual)

DEFINE_OP_EMPTY_CHECKER(GroupNormalization)

DEFINE_OP_CHECKER(GRU)
{
    using trtAct = nvinfer1::ActivationType;

    OnnxAttrs attrs{node, ctx};

    std::string const direction = attrs.get<std::string>("direction", "forward");
    int32_t const numDirections = (direction == "bidirectional") ? 2 : 1;

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
        STATIC_CHECK(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_EMPTY_CHECKER(HammingWindow)

DEFINE_OP_EMPTY_CHECKER(HannWindow)

DEFINE_OP_EMPTY_CHECKER(Hardmax)

DEFINE_OP_EMPTY_CHECKER(HardSigmoid)

DEFINE_OP_EMPTY_CHECKER(Identity)

DEFINE_OP_CHECKER(If)
{
    OnnxAttrs attrs(node, ctx);
    ::ONNX_NAMESPACE::GraphProto const& thenGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("then_branch");
    ::ONNX_NAMESPACE::GraphProto const& elseGraph = attrs.get<::ONNX_NAMESPACE::GraphProto const&>("else_branch");

    // Number of outputs are the same between the two branches.
    STATIC_CHECK(thenGraph.output_size() == elseGraph.output_size()
            && "then/else subgraphs should have the same number of outputs.",
        ErrorCode::kUNSUPPORTED_NODE_ATTR, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(ImageScaler)

DEFINE_OP_EMPTY_CHECKER(InstanceNormalization)

DEFINE_OP_EMPTY_CHECKER(IsInf)

DEFINE_OP_EMPTY_CHECKER(IsNaN)

DEFINE_OP_CHECKER(LayerNormalization)
{
    // TRT only expects one valid output. Other outputs are training artifacts that should've been removed for inference graphs.
    emptyOutputChecker(ctx, node, errors, nodeIndex, 1);
}

DEFINE_OP_EMPTY_CHECKER(LeakyRelu)

DEFINE_OP_EMPTY_CHECKER(Less)

DEFINE_OP_EMPTY_CHECKER(LessOrEqual)

DEFINE_OP_EMPTY_CHECKER(Log)

DEFINE_OP_EMPTY_CHECKER(LogSoftmax)

DEFINE_OP_EMPTY_CHECKER(Loop)

DEFINE_OP_EMPTY_CHECKER(LRN)

DEFINE_OP_CHECKER(LSTM)
{
    using trtAct = nvinfer1::ActivationType;

    OnnxAttrs attrs{node, ctx};
    std::string const direction = attrs.get<std::string>("direction", "forward");
    int32_t const numDirections = (direction == "bidirectional") ? 2 : 1;
    int32_t const inputForget = attrs.get("input_forget", 0);

    STATIC_CHECK(inputForget == 0 && "Coupled input/forget is unsupported in the LSTM converter",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);

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
        STATIC_CHECK(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activation alphas for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activation betas for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_CHECKER(LpNormalization)
{
    OnnxAttrs attrs(node, ctx);
    int32_t p = attrs.get<int32_t>("p", 2);

    STATIC_CHECK((p == 1 || p == 2) && "Only L1 and L2 normalization are supported.", ErrorCode::kUNSUPPORTED_NODE,
        node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(LpPool)
{
    OnnxAttrs attrs(node, ctx);
    int32_t p = attrs.get<int32_t>("p", 2);

    STATIC_CHECK((p == 1 || p == 2) && "Only L1 and L2 normalization are supported.", ErrorCode::kUNSUPPORTED_NODE,
        node, errors, nodeIndex);
    poolingCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(MatMul)

DEFINE_OP_EMPTY_CHECKER(Max)

DEFINE_OP_CHECKER(MaxPool)
{
    // TRT only expects one valid output. `Indices` output is unsupported.
    emptyOutputChecker(ctx, node, errors, nodeIndex, 1);
    poolingCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(Mean)

DEFINE_OP_EMPTY_CHECKER(MeanVarianceNormalization)

DEFINE_OP_EMPTY_CHECKER(Min)

DEFINE_OP_EMPTY_CHECKER(Mul)

DEFINE_OP_EMPTY_CHECKER(Mod)

DEFINE_OP_EMPTY_CHECKER(Neg)

DEFINE_OP_EMPTY_CHECKER(NonMaxSuppression)

DEFINE_OP_EMPTY_CHECKER(Not)

DEFINE_OP_EMPTY_CHECKER(OneHot)

DEFINE_OP_EMPTY_CHECKER(Or)

DEFINE_OP_EMPTY_CHECKER(Pad)

DEFINE_OP_EMPTY_CHECKER(ParametricSoftplus)

DEFINE_OP_CHECKER(Pow)
{
    int32_t const nbInputs = node.input().size();
    STATIC_CHECK(
        nbInputs == 2 && "POW operator expects two inputs!", ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(PRelu)

DEFINE_OP_CHECKER(RandomUniform)
{
    randomUniformCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(RandomUniformLike)
{
    randomUniformCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(RandomNormal)
{
    randomNormalCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(RandomNormalLike)
{
    randomNormalCheckHelper(ctx, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(Range)

DEFINE_OP_EMPTY_CHECKER(Reciprocal)

DEFINE_OP_EMPTY_CHECKER(ReduceL1)

DECLARE_OP_CHECKER(ReduceSum);

DEFINE_OP_EMPTY_CHECKER(ReduceLogSum)

DEFINE_OP_EMPTY_CHECKER(ReduceLogSumExp)

DECLARE_OP_CHECKER(ReduceSumSquare);

DEFINE_OP_EMPTY_CHECKER(ReduceL2)

DEFINE_OP_EMPTY_CHECKER(ReduceMax)

DEFINE_OP_EMPTY_CHECKER(ReduceMean)

DEFINE_OP_EMPTY_CHECKER(ReduceMin)

DEFINE_OP_EMPTY_CHECKER(ReduceProd)

DEFINE_OP_EMPTY_CHECKER(ReduceSum)

DEFINE_OP_EMPTY_CHECKER(ReduceSumSquare)

DEFINE_OP_EMPTY_CHECKER(Relu)

DEFINE_OP_EMPTY_CHECKER(RMSNormalization)

DEFINE_OP_EMPTY_CHECKER(Sign)


DEFINE_OP_EMPTY_CHECKER(Round)

DEFINE_OP_CHECKER(Resize)
{
    OnnxAttrs attrs(node, ctx);

    auto mode = attrs.get<std::string>("mode", "nearest");
    if (mode != "cubic" || mode != "linear")
    {
        STATIC_CHECK((mode == "cubic" || mode == "linear" || mode == "nearest") && "Invalid Resize mode",
            ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }

    // set transformation
    std::string transformationMode = "half_pixel";

    if (ctx->getOpsetVersion() >= 11)
    {
        // Check for TRT-supported resize attributes
        transformationMode = attrs.get<std::string>("coordinate_transformation_mode", "half_pixel");
        auto const nearest_mode = attrs.get<std::string>("nearest_mode", "round_prefer_floor");

        STATIC_CHECK((transformationMode != "tf_half_pixel_for_nn" || nearest_mode == "round_prefer_floor")
                && "This version of TensorRT only support round_prefer_floor nearest mode in tf_half_pixel_for_nn!",
            ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);

        STATIC_CHECK((transformationMode == "align_corners" || transformationMode == "tf_half_pixel_for_nn" || transformationMode == "pytorch_half_pixel"
                || transformationMode == "half_pixel" || transformationMode == "asymmetric")
                && "TensorRT only supports half_pixel, pytorch_half_pixel, tf_half_pixel_for_nn, asymmetric and "
                "align_corners transformation modes!", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }

    // Antialiasing is not supported currently.
    auto const antialias = attrs.get<int32_t>("antialias", 0);
    STATIC_CHECK((antialias == 0) && "Antialiasing is not supported currently.", ErrorCode::kUNSUPPORTED_NODE_ATTR,
        node, errors, nodeIndex);

    // Only stretch keep_aspect_ratio_policy is supported currently.
    auto const keep_aspect_ratio_policy = attrs.get<std::string>("keep_aspect_ratio_policy", "stretch");
    STATIC_CHECK((keep_aspect_ratio_policy == "stretch")
            && "Only `stretch` is supported currently as `keep_aspect_ratio_policy`.",
        ErrorCode::kUNSUPPORTED_NODE_ATTR, node, errors, nodeIndex);

    // Axes provided must be unique.
    auto const resizeAxes = attrs.get<std::vector<int32_t>>("axes", std::vector<int32_t>());
    STATIC_CHECK((std::unordered_set<int32_t>(resizeAxes.begin(), resizeAxes.end()).size() == resizeAxes.size())
            && "The input axes must have unique elements.",
        ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(Reshape)

DEFINE_OP_CHECKER(ReverseSequence)
{
    OnnxAttrs attrs{node, ctx};
    int32_t const batchAxis = attrs.get<int32_t>("batch_axis", 1);
    int32_t const sequenceAxis = attrs.get<int32_t>("time_axis", 0);
    STATIC_CHECK((batchAxis != sequenceAxis) && "batch_axis and time_axis cannot be the same",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(RNN)
{
    OnnxAttrs attrs{node, ctx};

    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int32_t numDirections = (direction == "bidirectional") ? 2 : 1;

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
        STATIC_CHECK(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
        STATIC_CHECK(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS)
            && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_CHECKER(RoiAlign)
{
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get("mode", std::string("avg"));
    STATIC_CHECK((mode == "avg" || mode == "max") && "Mode must be avg or max!", ErrorCode::kINVALID_NODE, node, errors,
        nodeIndex);

    auto samplingRatio = attrs.get<int32_t>("sampling_ratio", 0);
    STATIC_CHECK(
        samplingRatio >= 0 && "Sampling ratio cannot be negative!", ErrorCode::kINVALID_NODE, node, errors, nodeIndex);

    // Opset 16 attributes
    if (ctx->getOpsetVersion() >= 16)
    {
        auto ctm = attrs.get("coordinate_transformation_mode", std::string("half_pixel"));
        STATIC_CHECK((ctm == "half_pixel" || ctm == "output_half_pixel")
                && "Coordinate transformation mode must be half_pixel or output_half_pixel!",
            ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_EMPTY_CHECKER(RotaryEmbedding)

DEFINE_OP_EMPTY_CHECKER(ScaledTanh)

DEFINE_OP_EMPTY_CHECKER(Scan)

DEFINE_OP_EMPTY_CHECKER(GridSample)

DEFINE_OP_EMPTY_CHECKER(ScatterND)

DEFINE_OP_EMPTY_CHECKER(ScatterElements)

DEFINE_OP_EMPTY_CHECKER(Scatter)

DEFINE_OP_EMPTY_CHECKER(Selu)

DEFINE_OP_EMPTY_CHECKER(Shape)

DEFINE_OP_EMPTY_CHECKER(Sigmoid)

DEFINE_OP_EMPTY_CHECKER(Sin)

DEFINE_OP_EMPTY_CHECKER(Sinh)

DEFINE_OP_EMPTY_CHECKER(Size)

DEFINE_OP_CHECKER(Slice)
{
    int32_t const nbInputs = node.input().size();

    if (ctx->getOpsetVersion() >= 10)
    {
        STATIC_CHECK((nbInputs >= 3 && nbInputs <= 5) && "Post-opset 10 Slice operator requires 3 - 5 inputs.",
            ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_EMPTY_CHECKER(Softmax)

DEFINE_OP_EMPTY_CHECKER(Softsign)

DEFINE_OP_EMPTY_CHECKER(Softplus)

DEFINE_OP_EMPTY_CHECKER(SpaceToDepth)

DEFINE_OP_EMPTY_CHECKER(Split)

DEFINE_OP_EMPTY_CHECKER(Sqrt)

DEFINE_OP_EMPTY_CHECKER(Squeeze)

DEFINE_OP_EMPTY_CHECKER(Sub)

DEFINE_OP_EMPTY_CHECKER(Sum)

DEFINE_OP_EMPTY_CHECKER(Tan)

DEFINE_OP_EMPTY_CHECKER(Tanh)

DEFINE_OP_EMPTY_CHECKER(TensorScatter)

DEFINE_OP_EMPTY_CHECKER(ThresholdedRelu)

DEFINE_OP_EMPTY_CHECKER(Tile)

DEFINE_OP_CHECKER(TopK)
{
    OnnxAttrs attrs(node, ctx);

    if (ctx->getOpsetVersion() < 10)
    {
        STATIC_CHECK(
            (attrs.count("k")) && "Attribute k is missing.", ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
    }
}

DEFINE_OP_EMPTY_CHECKER(Transpose)

DEFINE_OP_EMPTY_CHECKER(Trilu)

DEFINE_OP_EMPTY_CHECKER(Unsqueeze)

DEFINE_OP_CHECKER(Upsample)
{
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("mode", "nearest");
    STATIC_CHECK((mode == "nearest" || mode == "linear" || mode == "bilinear")
            && "The attribute mode can only be nearest, linear, or bilinear.",
        ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(Where)

DEFINE_OP_EMPTY_CHECKER(Xor)

DEFINE_OP_EMPTY_CHECKER(Shrink)

DEFINE_OP_EMPTY_CHECKER(HardSwish)

DEFINE_OP_EMPTY_CHECKER(NonZero)

DEFINE_OP_EMPTY_CHECKER(Mish)

// Any ops that are not supported will attempt to import as plugins.
DEFINE_OP_CHECKER(FallbackPluginImporter)
{
    OnnxAttrs attrs(node, ctx);
    std::string const pluginName{node.op_type()};
    std::string const pluginVersion{attrs.get<std::string>("plugin_version", "1")};
    std::string pluginNamespace{attrs.get<std::string>("plugin_namespace", "")};

    nvinfer1::IPluginCreatorInterface* creator = importPluginCreator(ctx, pluginName, pluginVersion, pluginNamespace);

    // Fallback check to the node's domain
    if (!creator)
    {
        LOG_INFO("Plugin not found with node attribute namespace: using pluginName: "
            << pluginName << ", pluginVersion: " << pluginVersion << ", pluginNamespace: " << pluginNamespace);
        LOG_INFO("Searching for plugin with node domain namespace: using pluginName: "
            << pluginName << ", pluginVersion: " << pluginVersion << ", pluginNamespace: " << node.domain());
        pluginNamespace = node.domain();
        creator = importPluginCreator(ctx, pluginName, pluginVersion, pluginNamespace);
    }

    STATIC_CHECK(creator && "Plugin not found, are the plugin name, version, and namespace correct?",
        nvonnxparser::ErrorCode::kINVALID_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(LocalFunctionImporter)
{
    auto function = ctx->localFunctions().at(node.op_type());
    STATIC_CHECK(node.input().size() == function.input().size(), ErrorCode::kINVALID_NODE, node, errors, nodeIndex);

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

    for (auto const& node : function.node())
    {
        onnx2trt::parseNodeStaticCheck(ctx, node, errors, nodeIndex);
    }

    // Pop the current function name from stack
    ctx->localFunctionStack().pop_back();
}

DEFINE_OP_EMPTY_CHECKER(TRT_Scale)

DEFINE_OP_EMPTY_CHECKER(TRT_Shuffle)

DEFINE_OP_EMPTY_CHECKER(TRT_TopK_Min)

DEFINE_OP_EMPTY_CHECKER(TRT_MatMul)

DEFINE_OP_EMPTY_CHECKER(TRT_RNNv2)

DEFINE_OP_EMPTY_CHECKER(TRT_RaggedSoftmax)

DEFINE_OP_EMPTY_CHECKER(TRT_FullyConnected)

DEFINE_OP_EMPTY_CHECKER(TRT_MaxAverageBlendPool)

#if ENABLE_STD_PLUGIN
DEFINE_OP_EMPTY_CHECKER(TRT_PluginV2)
#endif // ENABLE_STD_PLUGIN

DEFINE_OP_EMPTY_CHECKER(TRT_Gather)

DEFINE_OP_EMPTY_CHECKER(TRT_Slice)

DEFINE_OP_EMPTY_CHECKER(TRT_Resize)

DEFINE_OP_EMPTY_CHECKER(TRT_FloorDiv)

DEFINE_OP_EMPTY_CHECKER(TRT_Conv)

DEFINE_OP_EMPTY_CHECKER(TRT_Deconv)

DEFINE_OP_CHECKER(TRT_MaxPool)
{
    checkMaxPool(ctx, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(TRT_AveragePool)
{
    checkAveragePool(ctx, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(TRT_MXFP8QuantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_MXFP8DequantizeLinear)

DEFINE_OP_EMPTY_CHECKER(TRT_MXFP8DynamicQuantize)

// Define unsupported node checkers
DEFINE_OP_CHECKER(BitShift)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(BitwiseAnd)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(BitwiseNot)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(BitwiseOr)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(BitwiseXor)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Col2Im)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Compress)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(ConcatFromSequence)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(ConvInteger)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(DFT)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Det)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(ImageDecoder)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(MatMulInteger)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(MaxRoiPool)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(MaxUnpool)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(MelWeightMatrix)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Multinomial)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Optional)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(OptionalGetElement)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(OptionalHasElement)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(QLinearConv)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(QLinearMatMul)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(RegexFullMatch)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_EMPTY_CHECKER(STFT)

DEFINE_OP_CHECKER(SequenceAt)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceConstruct)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceEmpty)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceErase)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceInsert)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceLength)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SplitToSequence)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(StringConcat)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(StringNormalizer)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(StringSplit)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(TfIdfVectorizer)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Unique)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(AffineGrid)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(Bernoulli)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(CenterCropPad)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(DynamicQuantizeLinear)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(NegativeLogLikelihoodLoss)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SequenceMap)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

DEFINE_OP_CHECKER(SoftmaxCrossEntropyLoss)
{
    STATIC_CHECK(false, ErrorCode::kUNSUPPORTED_NODE, node, errors, nodeIndex);
}

} // namespace

} // namespace onnx2trt
