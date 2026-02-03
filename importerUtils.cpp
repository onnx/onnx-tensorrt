/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "importerUtils.hpp"
#include "OnnxAttrs.hpp"
#include "Status.hpp"
#include "bfloat16.hpp"
#include "errorHelpers.hpp"
#include <ctype.h>
#include <ostream>
#include <regex>
#include <set>
namespace onnx2trt
{

void PluginDeleter::operator()(nvinfer1::IPluginV2* t)
{
    t->destroy();
}

Status notInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx)
{
    bool invalid = std::any_of(invalidTypes.begin(), invalidTypes.end(),
        [&](std::string invalidType) { return input.getType() == invalidType; });
    if (invalid)
    {
        ASSERT_NODE(
            false, "Found unsupported input type of " << input.getType(), node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }
    return Status::success();
}

void checkNotInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx)
{
    Status status = notInvalidType(input, invalidTypes, node, nodeIdx);
    ONNXTRT_CHECK_NODE(status.is_success(), "Found unsupported input type of " << input.getType(), node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
}

NodeOutputs activationHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha, float* beta)
{
    checkNotInvalidType(inputs.at(0), {"INT32", "BOOL", "UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::IActivationLayer* layer = N_CHECK(ctx->network()->addActivation(input, op));
    ONNXTRT_CHECK_NODE(layer, "Failed to add activation layer!", node, nodeIdx, ErrorCode::kINTERNAL_ERROR);
    if (alpha)
    {
        layer->setAlpha(*alpha);
    }
    if (beta)
    {
        layer->setBeta(*beta);
    }
    ctx->registerLayer(layer, node);
    auto* output = N_CHECK(layer->getOutput(0));
    return {{output}};
}

nvinfer1::ITensor* addClip(ImporterContext* ctx, nvinfer1::ITensor* input, float clip)
{
    if (clip >= 0.f)
    {
        nvinfer1::IActivationLayer* layer
            = N_CHECK(ctx->network()->addActivation(*input, nvinfer1::ActivationType::kCLIP));
        layer->setAlpha(-clip);
        layer->setBeta(clip);
        return N_CHECK(layer->getOutput(0));
    }
    return input;
}

NodeOutputs argMinMaxHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor* tensor = &convertToTensor(inputs.at(0), ctx);

    // Get attributes.
    OnnxAttrs attrs(node, ctx);
    int32_t keepdims = attrs.get("keepdims", 1);
    int32_t axis = attrs.get("axis", 0);
    int32_t selectLastIndex = attrs.get<int32_t>("select_last_index", 0);

    // Insert a TopK layer with k set to 1.
    int32_t nbDims = tensor->getDimensions().nbDims;
    convertAxis(axis, nbDims, node, nodeIdx);
    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer;

    // New attribute added to Opset-12
    // Whether to select the last index or the first index if the {name} appears in multiple indices, default is False
    // (first index).
    if (selectLastIndex)
    {
        // Need to flip the data input along the given axis using the Slice operator
        auto const dims = shapeOf(*tensor);
        ShapeTensor starts = shapeVector(-1);
        ShapeTensor ends = shapeVector(static_cast<int64_t>(INT_MIN));
        ShapeTensor axes = shapeVector(axis);
        ShapeTensor steps = shapeVector(-1);

        if (axes.size() < dims.size())
        {
            // axes specify a subset of the dimensions, or out of order.
            // Convert starts/ends/steps to complete in-order form.
            ShapeTensor const subscripts{axesToInterlaceSubscripts(axes, dims.size())};
            starts = interlace(ctx, similar(ctx, dims, 0), starts, subscripts);
            ends = interlace(ctx, dims, ends, subscripts);
            steps = interlace(ctx, similar(ctx, dims, 1), steps, subscripts);
        }
        decodeOnnxStartsAndEnds(ctx, dims, steps, starts, ends);
        // TensorRT uses sizes of the output dimensions instead of ends.
        ShapeTensor const sizes = computeSliceSizes(ctx, starts, ends, steps, dims);

        nvinfer1::ISliceLayer* slice = addSlice(ctx, *tensor, starts, sizes, steps);
        auto flippedTensor = N_CHECK(slice->getOutput(0));
        layer = N_CHECK(ctx->network()->addTopK(*flippedTensor, op, 1, axisMask));
    }
    else
    {
        layer = N_CHECK(ctx->network()->addTopK(*tensor, op, 1, axisMask));
    }

    ctx->registerLayer(layer, node);

    // We don't care about the TopK values, just the indices.
    nvinfer1::ITensor* indices = N_CHECK(layer->getOutput(1));
    indices = castHelper(ctx, indices, nvinfer1::DataType::kINT64);

    // If selectLastIndex is true, the TopK operation was performed on reversed data on the provided axis.
    // Convert reversed indices back to forward indices by calculating the following:
    // indices = shape(tensor)[axis] - indices - 1
    if (selectLastIndex)
    {
        // Use shapeTensor semantics to support dynamic shapes
        auto const dims = shapeOf(*tensor);
        auto const indicesDims = shapeOf(*indices);
        auto const axisTensor = shapeVector(axis);
        auto const dimOnAxis = gather(ctx, dims, axisTensor);

        // Create constant of shape indicesDims with values tensor.shape[axis]
        auto const tensorDimOnAxis = constantOfShape(ctx, &dimOnAxis.tensor(ctx), &indicesDims.tensor(ctx));

        // Create constant of shape indicesDims with values of 1
        auto const ones = constantOfShape(ctx, &shapeVector(1).tensor(ctx), &indicesDims.tensor(ctx));

        std::vector<TensorOrWeights> newInputs{tensorDimOnAxis, indices, ones};
        std::vector<TensorOrWeights> indicesUpdate
            = elementwiseHelper(ctx, node, nodeIdx, newInputs, nvinfer1::ElementWiseOperation::kSUB);
        indices = &convertToTensor(indicesUpdate.at(0), ctx);
    }
    // The default behavior of the TopK layer is to keepdims.
    // Otherwise, we need to squeeze the axis dimension.
    if (!keepdims)
    {
        std::vector<int32_t> axes{axis};
        indices = squeezeTensor(ctx, *indices, axes);
    }
    // TensorRT doesn't support int64 for TopK indices
    indices = castHelper(ctx, indices, nvinfer1::DataType::kINT64);
    return {{indices}};
}

void broadcastTensor(ImporterContext* ctx, nvinfer1::ITensor*& t, int const nbDims)
{
    ONNXTRT_CHECK(ctx->getOpsetVersion() >= 7,
        "Pre-opset 7 broadcasting is unsupported in this version of the ONNX parser", ErrorCode::kUNSUPPORTED_NODE);
    auto const inputDims = shapeOf(*t);
    int const nbInputDims = inputDims.size();
    ONNXTRT_CHECK((nbInputDims <= nbDims), "Cannot broadcast a higher rank tensor to a lower rank tensor.",
        ErrorCode::kUNSUPPORTED_NODE);
    if (nbInputDims < nbDims)
    {
        nvinfer1::IShuffleLayer* reshape
            = addShuffle(ctx, *t, concat(ctx, fillShapeVector(ctx, 1, shapeVector(nbDims - nbInputDims)), shapeOf(*t)));
        ctx->registerLayer(reshape, "ONNXTRT_Broadcast", nullptr);
        t = N_CHECK(reshape->getOutput(0));
    }
}

void broadcastTensors(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    int const t1Dims = t1->getDimensions().nbDims;
    int const t2Dims = t2->getDimensions().nbDims;

    if (t1Dims == t2Dims)
    {
        return;
    }

    if (t1Dims > t2Dims)
    {
        return broadcastTensor(ctx, t2, t1Dims);
    }
    return broadcastTensor(ctx, t1, t2Dims);
}

void broadcastTensors(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3)
{
    int const maxDims = std::max({t1->getDimensions().nbDims, t2->getDimensions().nbDims, t3->getDimensions().nbDims});
    broadcastTensor(ctx, t1, maxDims);
    broadcastTensor(ctx, t2, maxDims);
    broadcastTensor(ctx, t3, maxDims);
}

// Helper functions for calculateBias:
int32_t getBias(std::vector<int32_t> const& dimension_count, std::vector<int32_t> const& pitches, int32_t axis)
{
    int32_t result{0};
    for (int32_t i = 0; i < static_cast<int32_t>(dimension_count.size()); i++)
    {
        if (i != axis)
        {
            result += dimension_count[i] * pitches[i];
        }
    }
    return result;
}

void incrementOuterDimension(std::vector<int32_t>& dimensionCount, nvinfer1::Dims idxDims)
{
    // Start at [x,x,0]. Increment starting from the outer dimension.
    int32_t rank = dimensionCount.size();

    for (int32_t i = rank - 1; i >= 0; i--)
    {
        int dimLimit = idxDims.d[i];
        // If we're not at the limit, increment current axis and return
        if (++dimensionCount[i] != dimLimit)
        {
            break;
        }
        // Else, we increment on the next dimension and reset current one
        dimensionCount[i] = 0;
    }
}

std::vector<int32_t> calculateBias(
    nvinfer1::Dims const& daDims, nvinfer1::Dims const& idxDims, std::vector<int32_t> const& pitches, int32_t axis)
{
    std::vector<int32_t> biasVector;
    std::vector<int32_t> dimensionCount(daDims.nbDims, 0);
    int64_t total = volume(idxDims);

    for (int64_t i = 0; i < total; i++)
    {
        int32_t bias = getBias(dimensionCount, pitches, axis);
        biasVector.push_back(bias);
        incrementOuterDimension(dimensionCount, idxDims);
    }
    return biasVector;
}

std::vector<int32_t> calculatePitches(nvinfer1::Dims const& inputDims)
{
    int32_t pitch = 1;
    int32_t nbDims = inputDims.nbDims;
    std::vector<int32_t> pitches(nbDims);
    pitches[nbDims - 1] = pitch;
    for (int32_t i = nbDims - 2; i >= 0; i--)
    {
        pitch *= inputDims.d[i + 1];
        pitches[i] = pitch;
    }
    return pitches;
}

bool canUseNDResize(size_t const scaleSize, float const* scaleFactors, size_t const n)
{
    // Linear resize supports up to 3D resize on the outermost dimensions (n = 3).
    if (scaleSize > n)
    {
        for (size_t i = 0; i < scaleSize - n; i++)
        {
            if (scaleFactors[i] != 1)
            {
                return false;
            }
        }
    }
    return true;
}

nvinfer1::ITensor* castHelper(ImporterContext* ctx, nvinfer1::ITensor* input, nvinfer1::DataType dtype)
{
    nvinfer1::ICastLayer* cast = N_CHECK(ctx->network()->addCast(*input, dtype));
    ctx->registerLayer(cast, "ONNXTRT_castHelper", nullptr);
    return N_CHECK(cast->getOutput(0));
}

nvinfer1::ITensor* constantOfShape(ImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape)
{
    ShapeTensor shapeT{*shape};
    ShapeTensor zeros = similar(ctx, shapeT, 0);
    // `constant` must be broadcasted to the same rank as `shape`.
    ShapeTensor broadcastedShape = similar(ctx, shapeT, 1);
    constant = &reshape(ctx, *constant, broadcastedShape);
    auto* l = N_CHECK(addSlice(ctx, *constant, zeros, shapeT, zeros));
    return N_CHECK(l->getOutput(0));
}

void convertAxis(int32_t& axis, int32_t const nbDims, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx)
{
    // Support negative indexing
    if (axis < 0)
    {
        axis += nbDims;
    }
    // Support nbDims as a valid axis for QuantDequantLinearHelper
    ONNXTRT_CHECK_NODE((axis >= 0 && axis <= nbDims),
        "Axis must be in the range [0, nbDims (" << nbDims << ")]. Provided axis is: " << axis, node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
}

bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype)
{
    switch (onnx_dtype)
    {
    case ::ONNX_NAMESPACE::TensorProto::DOUBLE: *trt_dtype = nvinfer1::DataType::kFLOAT; break;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: *trt_dtype = nvinfer1::DataType::kFLOAT; break;
    case ::ONNX_NAMESPACE::TensorProto::INT8: *trt_dtype = nvinfer1::DataType::kINT8; break;
    case ::ONNX_NAMESPACE::TensorProto::UINT8: *trt_dtype = nvinfer1::DataType::kUINT8; break;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16: *trt_dtype = nvinfer1::DataType::kHALF; break;
    case ::ONNX_NAMESPACE::TensorProto::BFLOAT16: *trt_dtype = nvinfer1::DataType::kBF16; break;
    case ::ONNX_NAMESPACE::TensorProto::BOOL: *trt_dtype = nvinfer1::DataType::kBOOL; break;
    case ::ONNX_NAMESPACE::TensorProto::INT32: *trt_dtype = nvinfer1::DataType::kINT32; break;
    case ::ONNX_NAMESPACE::TensorProto::INT64: *trt_dtype = nvinfer1::DataType::kINT64; break;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN: *trt_dtype = nvinfer1::DataType::kFP8; break;
    case ::ONNX_NAMESPACE::TensorProto::INT4: *trt_dtype = nvinfer1::DataType::kINT4; break;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1: *trt_dtype = nvinfer1::DataType::kFP4; break;
    default:
        std::cerr << "Unsupported ONNX data type: " << getDtypeName(onnx_dtype) << " (" << std::to_string(onnx_dtype)
                  << ")" << std::endl;
        return false;
    }
    return true;
}

bool convertOnnxPadding(ImporterContext* ctx, int32_t nbInputDims, std::vector<int64_t> const& onnxPadding,
    nvinfer1::ITensor*& startTensor, nvinfer1::ITensor*& totalPaddingTensor)
{
    std::vector<int64_t> start;
    std::vector<int64_t> totalPadding;
    if (onnxPadding.size() % 2U != 0)
    {
        return false;
    }
    auto const diff = nbInputDims - static_cast<int32_t>(onnxPadding.size() / 2U);
    if (diff < 0)
    {
        return false;
    }
    start.resize(nbInputDims, 0);
    totalPadding.resize(nbInputDims, 0);

    for (int32_t i = diff; i < nbInputDims; i++)
    {
        auto const idx = i - diff;
        auto const pre = onnxPadding[idx];
        auto const post = onnxPadding[onnxPadding.size() / 2U + idx];
        if (pre < 0 || post < 0)
        {
            return false;
        }

        start[i] = -pre;
        totalPadding[i] = pre + post;
    }
    auto* startLayer
        = N_CHECK(addConstant(ctx, start, ::ONNX_NAMESPACE::TensorProto::INT64, nvinfer1::Dims{1, {nbInputDims}}));
    startTensor = N_CHECK(startLayer->getOutput(0));

    auto* totalPaddingLayer = N_CHECK(
        addConstant(ctx, totalPadding, ::ONNX_NAMESPACE::TensorProto::INT64, nvinfer1::Dims{1, {nbInputDims}}));
    totalPaddingTensor = N_CHECK(totalPaddingLayer->getOutput(0));
    return startTensor && totalPaddingTensor;
}

bool shiftIsAllZeros(ShapedWeights const& shift)
{
    // Check if all of the values in the shift tensor are zeros. Shift dtype is one of [INT8, UINT8, INT4, UINT4]
    auto const* v = static_cast<int8_t const*>(shift.values);
    size_t const count = shift.size_bytes();
    auto allZeros = std::all_of(v, v + count, [](int8_t x) { return x == 0; });
    return allZeros;
}

onnx2trt::ShapedWeights createZeroShifts(onnx2trt::ShapedWeights const& shiftInt, int32_t type, ImporterContext* ctx)
{
    auto shift = ctx->createNamedTempWeights(type, shiftInt.shape);
    float* sh = static_cast<float*>(shift.values);
    for (int i = 0, n = shift.count(); i < n; i++)
    {
        sh[i] = 0.0f;
    }
    return shift;
}

nvinfer1::ITensor* createZeroTensor(ImporterContext* ctx, nvinfer1::ITensor* data)
{
    auto shape = shapeOf(*data);
    auto* zeros = N_CHECK(addConstantScalar(ctx, 0.0F, ::ONNX_NAMESPACE::TensorProto::FLOAT)->getOutput(0));
    zeros = castHelper(ctx, zeros, data->getType());
    return constantOfShape(ctx, zeros, &shape.tensor(ctx));
}

nvinfer1::ITensor* convertToScalar(ImporterContext* ctx, nvinfer1::ITensor* inpTensor)
{
    if (inpTensor->getDimensions().nbDims == 0)
    {
        return inpTensor;
    }
    auto const tensorVolume = volume(inpTensor->getDimensions());
    if (tensorVolume != 1)
    {
        LOG_VERBOSE("Cannot convert tensor to scalar. Note: Tensor dimensions were: "
            << inpTensor->getDimensions() << ", with volume: " << tensorVolume);
        return nullptr;
    }
    nvinfer1::IShuffleLayer* reshape = N_CHECK(ctx->network()->addShuffle(*inpTensor));
    reshape->setReshapeDimensions(nvinfer1::Dims{0});
    ctx->registerLayer(reshape, "ONNXTRT_convertToScalar", nullptr);
    // Do not need to call setZeroIsPlaceholder, since reshape dimensions are empty.
    return N_CHECK(reshape->getOutput(0));
}

nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, ImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return input.tensor();
    }
    // Handle non-tensor indices input by adding a new constant layer to the network.
    ShapedWeights& weights = input.weights();

    auto const existingConstantLayer = ctx->getConstantLayer(weights.getName());
    if (existingConstantLayer != nullptr)
    {
        return *N_CHECK(existingConstantLayer->getOutput(0));
    }
    auto* constantLayer = N_CHECK(ctx->network()->addConstant(weights.shape, weights));
    // Register layer and constant name (if set) into RefitMap:
    if (weights.getName())
    {
        ctx->registerLayer(constantLayer, weights.getName(), nullptr);
        ctx->network()->setWeightsName(weights, weights.getName());
    }

    auto* output = N_CHECK(constantLayer->getOutput(0));

    return *output;
}

nvinfer1::ITensor* convertToScalar(TensorOrWeights& input, ImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return convertToScalar(ctx, &input.tensor());
    }
    ShapedWeights& weights = input.weights();
    if (volume(weights.shape) != 1)
    {
        LOG_VERBOSE("Cannot convert weights to scalar. Note: Tensor dimensions were: "
            << weights.shape << ", with volume: " << volume(weights.shape));
        return nullptr;
    }
    auto* scalarLayer = N_CHECK(ctx->network()->addConstant(nvinfer1::Dims{0, {0}}, weights));
    return N_CHECK(scalarLayer->getOutput(0));
}

nvinfer1::ITensor* convertScalarToVector(ImporterContext* ctx, nvinfer1::ITensor* input)
{
    if (input->getDimensions().nbDims != 0)
    {
        return input;
    }
    std::vector<int32_t> axes{0};
    return unsqueezeTensor(ctx, *input, axes);
}

int divCeil(int n, int d)
{
    return (n - 1) / d + 1;
}

std::string getTrtDtypeName(nvinfer1::DataType TrtDtype)
{
    switch (TrtDtype)
    {
    case nvinfer1::DataType::kFLOAT: return "FLOAT";
    case nvinfer1::DataType::kHALF: return "HALF";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kBOOL: return "BOOL";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    case nvinfer1::DataType::kFP8: return "FP8";
    case nvinfer1::DataType::kBF16: return "BF16";
    case nvinfer1::DataType::kINT64: return "INT64";
    case nvinfer1::DataType::kINT4: return "INT4";
    case nvinfer1::DataType::kFP4: return "FP4";
    case nvinfer1::DataType::kE8M0: return "E8M0";
    default: return "<UNKNOWN>";
    }
}

std::string getElementWiseOpName(nvinfer1::ElementWiseOperation op)
{
    switch (op)
    {
    case nvinfer1::ElementWiseOperation::kSUM: return "SUM";
    case nvinfer1::ElementWiseOperation::kPROD: return "PROD";
    case nvinfer1::ElementWiseOperation::kMAX: return "MAX";
    case nvinfer1::ElementWiseOperation::kMIN: return "MIN";
    case nvinfer1::ElementWiseOperation::kSUB: return "SUB";
    case nvinfer1::ElementWiseOperation::kDIV: return "DIV";
    case nvinfer1::ElementWiseOperation::kPOW: return "POW";
    case nvinfer1::ElementWiseOperation::kFLOOR_DIV: return "FLOOR_DIV";
    case nvinfer1::ElementWiseOperation::kAND: return "AND";
    case nvinfer1::ElementWiseOperation::kOR: return "OR";
    case nvinfer1::ElementWiseOperation::kXOR: return "XOR";
    case nvinfer1::ElementWiseOperation::kEQUAL: return "EQUAL";
    case nvinfer1::ElementWiseOperation::kGREATER: return "GREATER";
    case nvinfer1::ElementWiseOperation::kLESS: return "LESS";
    default: return "<UNKNOWN>";
    }
}

void elementwiseCheck(std::vector<TensorOrWeights> const& inputs, const nvinfer1::ElementWiseOperation op,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx)
{
    switch (op)
    {
    // These operations only support boolean inputs
    case nvinfer1::ElementWiseOperation::kAND:
    case nvinfer1::ElementWiseOperation::kOR:
    case nvinfer1::ElementWiseOperation::kXOR:
        ONNXTRT_CHECK_NODE(
            std::all_of(inputs.begin(), inputs.end(), [](TensorOrWeights const& input) { return input.isBool(); }),
            "Elementwise layer only supports operator " + getElementWiseOpName(op)
                + " and the given inputs with type BOOL.",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        break;
    // These operations do not support boolean types
    case nvinfer1::ElementWiseOperation::kDIV:
    case nvinfer1::ElementWiseOperation::kFLOOR_DIV:
    case nvinfer1::ElementWiseOperation::kGREATER:
    case nvinfer1::ElementWiseOperation::kLESS:
    case nvinfer1::ElementWiseOperation::kMAX:
    case nvinfer1::ElementWiseOperation::kMIN:
    case nvinfer1::ElementWiseOperation::kPOW:
    case nvinfer1::ElementWiseOperation::kPROD:
    case nvinfer1::ElementWiseOperation::kSUB:
    case nvinfer1::ElementWiseOperation::kSUM:
        ONNXTRT_CHECK_NODE(
            !std::any_of(inputs.begin(), inputs.end(), [](TensorOrWeights const& input) { return input.isBool(); }),
            "Elementwise layer does not support operator " + getElementWiseOpName(op)
                + " and the given inputs with type BOOL.",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        break;
    // Equal supports all types.
    case nvinfer1::ElementWiseOperation::kEQUAL: break;
    }
}

NodeOutputs elementwiseHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights> const& inputs, nvinfer1::ElementWiseOperation binary_op)
{
    ONNXTRT_CHECK_NODE((!inputs.empty()), "Inputs vector is empty.", node, nodeIdx, ErrorCode::kINVALID_NODE);

    std::vector<nvinfer1::ITensor*> inputTensors;
    int maxNbDims = -1;
    for (auto input : inputs)
    {
        maxNbDims = std::max(maxNbDims, input.shape().nbDims);
    }

    for (auto input : inputs)
    {
        auto* tensor_ptr = &convertToTensor(input, ctx);

        // Broadcast all input tensors to size of maxNbDims
        broadcastTensor(ctx, tensor_ptr, maxNbDims);
        ONNXTRT_CHECK_NODE(tensor_ptr->getDimensions().nbDims == maxNbDims,
            "The number of dimensions should remain the same adding inputs: " << tensor_ptr->getDimensions().nbDims
                                                                              << " != " << maxNbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }
    elementwiseCheck(inputs, binary_op, node, nodeIdx);

    // Use the first tensor input as the base for the elementwise operation
    nvinfer1::ITensor* combined = inputTensors.at(0);
    if (inputTensors.size() == 1)
    {
        // Note: Single input must be wrapped in identity to avoid messing up network outputs
        return {{identity(ctx, combined)}};
    }
    for (size_t i = 1; i < inputTensors.size(); ++i)
    {
        nvinfer1::ITensor* tensor = inputTensors.at(i);
        ONNXTRT_CHECK_NODE((tensor->getDimensions().nbDims == combined->getDimensions().nbDims),
            "The number of dimensions should remain the same adding inputs: "
                << tensor->getDimensions().nbDims << " != " << combined->getDimensions().nbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = N_CHECK(ctx->network()->addElementWise(*combined, *tensor, binary_op));
        ctx->registerLayer(layer, node);
        combined = N_CHECK(layer->getOutput(0));
    }
    return {{combined}};
}

nvinfer1::ITensor* flattenTensor(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ITensor& tensor, int axis, bool regLayer)
{
    auto const dims = shapeOf(tensor);
    auto const d0 = product(ctx, dims, 0, axis, 1);
    auto const d1 = product(ctx, dims, axis, dims.size(), 1);

    // ShuffleLayer here interprets dim extent 0 as empty dim to support empty tensor
    nvinfer1::IShuffleLayer* flattenLayer
        = N_CHECK(addShuffle(ctx, tensor, concat(ctx, d0, d1), /*zeroIsPlaceholder=*/false));
    if (regLayer)
    {
        ctx->registerLayer(flattenLayer, node);
    }
    else
    {
        ctx->registerLayer(flattenLayer, "ONNXTRT_flattenTensor", nullptr);
    }
    return N_CHECK(flattenLayer->getOutput(0));
}

nvinfer1::ITensor* extractDimension(ImporterContext* ctx, nvinfer1::ITensor* shapeTensor, int dim, nvinfer1::Dims shape)
{
    // Comparing with gather, slice is more flexible. It does not need to convert dim into a constant.
    // It is important for refit as when add an additional constant, this gather may not be optimized out.
    auto* slice = N_CHECK(ctx->network()->addSlice(
        *shapeTensor, nvinfer1::Dims{1, {dim}}, nvinfer1::Dims{1, {1}}, nvinfer1::Dims{1, {1}}));
    ctx->registerLayer(slice, "ONNXTRT_extractDimension", nullptr);
    if (shape != nvinfer1::Dims{1, {1}})
    {
        auto* reshape = N_CHECK(ctx->network()->addShuffle(*slice->getOutput(0)));
        reshape->setReshapeDimensions(shape);
        ctx->registerLayer(reshape, "ONNXTRT_extractDimensionReshape", nullptr);
        return N_CHECK(reshape->getOutput(0));
    }
    return N_CHECK(slice->getOutput(0));
}

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims inputShape, nvinfer1::Dims outputShape, nvinfer1::Dims kernelSize,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, int const nbSpatialDims, nvinfer1::Dims& begPadding,
    nvinfer1::Dims& endPadding, nvinfer1::Dims& outputPadding, nvinfer1::PaddingMode paddingMode)
{
    nvinfer1::Dims totalPadding{nbSpatialDims, {}};
    // Pre and post padding calculated as per https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
    // Note that output shape is inconsistent in the spec - can either be in full dimensions form (i.e. NCHW) or just
    // spatial dimensions form (i.e. HW). Calculate potential offset here.
    auto const outputOffset = outputShape.nbDims - nbSpatialDims;
    for (int32_t i = 0; i < nbSpatialDims; i++)
    {
        totalPadding.d[i] = strides.d[i] * (inputShape.d[2 + i] - 1) + outputPadding.d[i]
            + ((kernelSize.d[i] - 1) * dilations.d[i] + 1) - outputShape.d[outputOffset + i];
        // Same upper is calculated differently
        if (paddingMode != nvinfer1::PaddingMode::kSAME_UPPER)
        {
            begPadding.d[i] = totalPadding.d[i] / 2;
            endPadding.d[i] = totalPadding.d[i] - (totalPadding.d[i] / 2);
        }
        else
        {
            begPadding.d[i] = totalPadding.d[i] - (totalPadding.d[i] / 2);
            endPadding.d[i] = (totalPadding.d[i] / 2);
        }
    }
}

float getActivationDefaultAlpha(nvinfer1::ActivationType type)
{
    switch (type)
    {
    case nvinfer1::ActivationType::kCLIP: return 0.f;
    case nvinfer1::ActivationType::kELU: return 1.0f;
    case nvinfer1::ActivationType::kGELU_ERF: return 0.f;
    case nvinfer1::ActivationType::kGELU_TANH: return 0.f;
    case nvinfer1::ActivationType::kHARD_SIGMOID: return 0.2f;
    case nvinfer1::ActivationType::kLEAKY_RELU: return 0.01f;
    case nvinfer1::ActivationType::kRELU: return 0.f;
    case nvinfer1::ActivationType::kSCALED_TANH: return 1.0f;
    case nvinfer1::ActivationType::kSELU: return 1.67326319217681884765625f;
    case nvinfer1::ActivationType::kSIGMOID: return 0.f;
    case nvinfer1::ActivationType::kSOFTPLUS: return 0.f;
    case nvinfer1::ActivationType::kSOFTSIGN: return 0.f;
    case nvinfer1::ActivationType::kTANH: return 0.f;
    case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 1.0f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

float getActivationDefaultBeta(nvinfer1::ActivationType type)
{
    switch (type)
    {
    case nvinfer1::ActivationType::kCLIP: return 0.f;
    case nvinfer1::ActivationType::kELU: return 0.f;
    case nvinfer1::ActivationType::kGELU_ERF: return 0.f;
    case nvinfer1::ActivationType::kGELU_TANH: return 0.f;
    case nvinfer1::ActivationType::kHARD_SIGMOID: return 0.5f;
    case nvinfer1::ActivationType::kLEAKY_RELU: return 0.f;
    case nvinfer1::ActivationType::kRELU: return 0.f;
    case nvinfer1::ActivationType::kSCALED_TANH: return 1.0f;
    case nvinfer1::ActivationType::kSELU: return 1.05070102214813232421875f;
    case nvinfer1::ActivationType::kSIGMOID: return 0.f;
    case nvinfer1::ActivationType::kSOFTPLUS: return 0.f;
    case nvinfer1::ActivationType::kSOFTSIGN: return 0.f;
    case nvinfer1::ActivationType::kTANH: return 0.f;
    case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 0.f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

nvinfer1::ITensor* getAxisLength(ImporterContext* ctx, nvinfer1::ITensor* inpTensor, int32_t axis, nvinfer1::Dims shape)
{
    // Let TRT handle the shape tensor optimization.
    auto* shapeLayer = N_CHECK(ctx->network()->addShape(*inpTensor));
    nvinfer1::ITensor* inpShape = N_CHECK(shapeLayer->getOutput(0));
    // TRT-22536 - remove the cast and fix clients of getAxisLength to use 64-bit lengths.
    auto* castLayer = N_CHECK(ctx->network()->addCast(*inpShape, nvinfer1::DataType::kINT32));
    inpShape = N_CHECK(castLayer->getOutput(0));
    return extractDimension(ctx, inpShape, axis, shape);
}

nvinfer1::ITensor* getAxisLengthInt64(
    ImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape)
{
    auto* shapeLayer = N_CHECK(ctx->network()->addShape(*inpTensor));
    nvinfer1::ITensor* inpShape = N_CHECK(shapeLayer->getOutput(0));
    return extractDimension(ctx, inpShape, axis, shape);
}

nvinfer1::ITensor* getElementWiseResult(
    ImporterContext* ctx, nvinfer1::ITensor& lhs, nvinfer1::ITensor& rhs, nvinfer1::ElementWiseOperation op)
{
    auto* elemLayer = N_CHECK(ctx->network()->addElementWise(lhs, rhs, op));
    return N_CHECK(elemLayer->getOutput(0));
}

nvinfer1::ITensor* getUnaryResult(ImporterContext* ctx, nvinfer1::ITensor& input, nvinfer1::UnaryOperation op)
{
    auto* unaryLayer = N_CHECK(ctx->network()->addUnary(input, op));
    return N_CHECK(unaryLayer->getOutput(0));
}

void getKernelParams(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::Dims* kernelSize,
    nvinfer1::Dims* strides, nvinfer1::Dims* begPadding, nvinfer1::Dims* endPadding, nvinfer1::PaddingMode& paddingMode,
    bool& countExcludePadding, nvinfer1::Dims* dilations, nvinfer1::Dims* outputPadding, bool const poolingCeilMode)
{
    int32_t const nbSpatialDims = kernelSize->nbDims;
    OnnxAttrs attrs(node, ctx);
    if (attrs.count("kernel_shape"))
    {
        auto const* onnxKernelSize = attrs.at("kernel_shape");
        setAttr(kernelSize, onnxKernelSize, nbSpatialDims, 1);
    }
    if (attrs.count("strides"))
    {
        auto const* onnxStrides = attrs.at("strides");
        setAttr(strides, onnxStrides, nbSpatialDims, 1);
    }
    if (dilations && attrs.count("dilations"))
    {
        auto const* onnxDilations = attrs.at("dilations");
        setAttr(dilations, onnxDilations, nbSpatialDims, 1);
    }
    if (attrs.count("count_include_pad"))
    {
        auto const* includePad = attrs.at("count_include_pad");
        int32_t val = includePad->i();
        val == 1 ? countExcludePadding = false : countExcludePadding = true;
    }
    // For ConvTranspose Layer
    if (attrs.count("output_padding"))
    {
        auto const* onnxOutputPadding = attrs.at("output_padding");
        setAttr(outputPadding, onnxOutputPadding, nbSpatialDims, 0);
    }

    paddingMode
        = poolingCeilMode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    auto onnxAutoPad = attrs.get("auto_pad", std::string("NOTSET"));
    if (onnxAutoPad != "SAME_LOWER" && onnxAutoPad != "SAME_UPPER")
    {
        if (attrs.count("pads"))
        {
            auto onnxPadding = attrs.get<std::vector<int32_t>>("pads");
            int32_t ndim = onnxPadding.size() / 2;
            for (int32_t i = 0; i < nbSpatialDims; ++i)
            {
                if (i < ndim)
                {
                    begPadding->d[i] = onnxPadding.at(i);
                    endPadding->d[i] = onnxPadding.at(i + ndim);
                }
                else
                {
                    begPadding->d[i] = 0;
                    endPadding->d[i] = 0;
                }
            }
        }
        if (onnxAutoPad == "EXPLICIT_ROUND_UP")
        {
            paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP;
        }
    }
    else
    {
        // If auto_pad is SAME_LOWER or SAME_UPPER, input padding should be calculated
        // "pads" attribute should not be specified
        ONNXTRT_CHECK(!attrs.count("pads"),
            "Pads attribute should not be specified with SAME_LOWER or SAME_UPPER auto padding!",
            ErrorCode::kINVALID_NODE);
        // Note: ONNX is always NCHW ordering
        if (onnxAutoPad == "SAME_LOWER")
        {
            paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
        }
        else if (onnxAutoPad == "SAME_UPPER")
        {
            paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
        }
        else
        {
            ONNXTRT_THROW(MAKE_ERROR("invalid autopad attribute was set!", ErrorCode::kINVALID_NODE));
        }
    }
}

float getSingleValueAsFloat(ShapedWeights const& weights)
{
    assert(weights.count() == 1 && "Expected weights to contain only a single value");
    switch (weights.type)
    {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: return static_cast<float const*>(weights.values)[0];
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
        return static_cast<float>(static_cast<half_float::half const*>(weights.values)[0]);
    case ::ONNX_NAMESPACE::TensorProto::BFLOAT16:
        return static_cast<float>(static_cast<BFloat16 const*>(weights.values)[0]);
    default: assert(false && "Unsupported type!"); return 0.F;
    }
    return 0.F;
}

nvinfer1::ITensor* globalPoolingHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ITensor& tensor, nvinfer1::ReduceOperation op)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
    uint32_t reduceAxes = ((1 << dims.nbDims) - 1) & ~0b11;
    auto* layer = N_CHECK(ctx->network()->addReduce(tensor, op, reduceAxes, /*keepDimensions=*/true));
    ctx->registerLayer(layer, node);
    return N_CHECK(layer->getOutput(0));
}

NodeOutputs greaterLessOrEqual(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    nvinfer1::ITensor* inputA, nvinfer1::ITensor* inputB, bool greater)
{
    nvinfer1::ElementWiseOperation op
        = greater ? nvinfer1::ElementWiseOperation::kGREATER : nvinfer1::ElementWiseOperation::kLESS;
    std::vector<TensorOrWeights> firstOpResults = elementwiseHelper(ctx, node, nodeIdx, {inputA, inputB}, op);
    std::vector<TensorOrWeights> equalsResult
        = elementwiseHelper(ctx, node, nodeIdx, {inputA, inputB}, nvinfer1::ElementWiseOperation::kEQUAL);
    return elementwiseHelper(
        ctx, node, nodeIdx, {firstOpResults.at(0), equalsResult.at(0)}, nvinfer1::ElementWiseOperation::kOR);
}

nvinfer1::IPluginCreatorInterface* importPluginCreator(ImporterContext* ctx, std::string const& pluginName,
    std::string const& pluginVersion, std::string const& pluginNamespace)
{
    nvinfer1::IPluginCreatorInterface* creator = nullptr;

    int32_t numCreators = 0;
    std::vector<nvinfer1::IPluginCreatorInterface*> creators;

    int32_t numStdCreators = 0;
    auto& stdPluginRegistry = ctx->network()->getBuilder().getPluginRegistry();
    auto stdCreators = stdPluginRegistry.getAllCreatorsRecursive(&numStdCreators);
    if (numStdCreators > 0)
    {
        creators.insert(creators.end(), stdCreators, stdCreators + numStdCreators);
    }

    numCreators = creators.size();
    // Helper function to check if a plugin creator matches the requested plugin parameters
    auto matchesPlugin = [&](char const* name, char const* version, char const* ns) -> bool {
        return std::string(name) == pluginName && std::string(version) == pluginVersion
            && std::string(ns) == pluginNamespace;
    };

    // Search for a creator that matches the requested plugin
    // the creators are guaranteed to be unique
    for (int32_t i = 0; i < numCreators; i++)
    {
        auto currentCreator = creators[i];
        if (!currentCreator)
        {
            continue;
        }

        // Get the creator version to determine the appropriate type
        auto const creatorVersion = getPluginCreatorVersion(currentCreator);
        bool matches = false;

        switch (creatorVersion)
        {
        case CreatorVersion::kV1:
        {
            auto const v1Creator = static_cast<nvinfer1::IPluginCreator const*>(currentCreator);
            matches = matchesPlugin(
                v1Creator->getPluginName(), v1Creator->getPluginVersion(), v1Creator->getPluginNamespace());
            break;
        }
        case CreatorVersion::kV3ONE:
        {
            auto const v3Creator = static_cast<nvinfer1::IPluginCreatorV3One const*>(currentCreator);
            matches = matchesPlugin(
                v3Creator->getPluginName(), v3Creator->getPluginVersion(), v3Creator->getPluginNamespace());
            break;
        }
        case CreatorVersion::kV3QUICK:
        {
            auto const v3QuickCreator = static_cast<nvinfer1::IPluginCreatorV3Quick const*>(currentCreator);
            matches = matchesPlugin(v3QuickCreator->getPluginName(), v3QuickCreator->getPluginVersion(),
                v3QuickCreator->getPluginNamespace());
            break;
        }
       // No default case as the creatorVersion is guaranteed to be one of the above as per
       // `getPluginCreatorVersion()`. For any future plugin creator versions added, this switch statement will
       // need to be updated along with `getPluginCreatorVersion()`.
        }

        if (matches)
        {
            creator = currentCreator;
            break;
        }
    }


    return creator;
}

std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(ImporterContext* ctx,
    ::ONNX_NAMESPACE::NodeProto const& node, std::string const& name, std::string const& /* pluginNamespace */,
    nvinfer1::IPluginCreator* pluginCreator, std::vector<nvinfer1::PluginField> const& pluginFields)
{
    if (!pluginCreator)
    {
        return nullptr;
    }

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter>{pluginCreator->createPlugin(name.c_str(), &fc)};
}

namespace
{
constexpr char const* kV1_CREATOR_IFACE_KIND = "PLUGIN CREATOR_V1";
constexpr char const* kV3_CREATOR_ONE_IFACE_KIND = "PLUGIN CREATOR_V3ONE";
constexpr char const* kV3_CREATOR_QUICK_IFACE_KIND = "PLUGIN CREATOR_V3QUICK";

bool isKind(nvinfer1::InterfaceInfo const& info, std::string_view kind)
{
    ONNXTRT_CHECK(
        info.kind != nullptr, "Invalid plugin creator interface with NULL kind.", ErrorCode::kUNSUPPORTED_NODE);
    return info.kind == kind;
}

} // namespace

CreatorVersion getPluginCreatorVersion(nvinfer1::IPluginCreatorInterface const* pluginCreator)
{
    ONNXTRT_CHECK(pluginCreator != nullptr, "Null plugin creator.", ErrorCode::kINTERNAL_ERROR);
    auto const ifaceInfo = pluginCreator->getInterfaceInfo();
    if (isKind(ifaceInfo, kV1_CREATOR_IFACE_KIND))
    {
        return CreatorVersion::kV1;
    }
    if (isKind(ifaceInfo, kV3_CREATOR_ONE_IFACE_KIND))
    {
        return CreatorVersion::kV3ONE;
    }
    if (isKind(ifaceInfo, kV3_CREATOR_QUICK_IFACE_KIND))
    {
        return CreatorVersion::kV3QUICK;
    }
    ONNXTRT_CHECK(false, "Unknown plugin creator version.", ErrorCode::kINTERNAL_ERROR);
}

std::unique_ptr<nvinfer1::IPluginV3> createPlugin(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::string const& name, std::string const& pluginNamespace, nvinfer1::IPluginCreatorInterface* pluginCreator,
    std::vector<nvinfer1::PluginField> const& pluginFields)
{
    if (!pluginCreator)
    {
        return nullptr;
    }

    auto const creatorVersion = getPluginCreatorVersion(pluginCreator);

    ONNXTRT_CHECK((creatorVersion == CreatorVersion::kV3ONE || creatorVersion == CreatorVersion::kV3QUICK),
        "Only IPluginCreatorV3One and IPluginCreatorV3Quick are supported for V3 plugin imports.",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    if (creatorVersion == CreatorVersion::kV3ONE)
    {
        return std::unique_ptr<nvinfer1::IPluginV3>{
            static_cast<nvinfer1::IPluginCreatorV3One*>(pluginCreator)
                ->createPlugin(name.c_str(), &fc, nvinfer1::TensorRTPhase::kBUILD)};
    }
    else if (creatorVersion == CreatorVersion::kV3QUICK)
    {
        // QDP framework relies on the name passed to createPlugin() being exactly equal to the registered plugin name
        // So the name should be set to the op type
        std::string const pluginOpName{node.op_type()};

        OnnxAttrs attrs(node, ctx);
        nvinfer1::QuickPluginCreationRequest request;

        // Node-level specifications override network-level preferences
        if (attrs.count("aot"))
        {
            auto const aotOrJit = static_cast<bool>(attrs.get<int>("aot", 0));
            if (aotOrJit)
            {
                request = nvinfer1::QuickPluginCreationRequest::kSTRICT_AOT;
            }
            else
            {
                request = nvinfer1::QuickPluginCreationRequest::kSTRICT_JIT;
            }
        }
        else
        {
            auto const preferAOT
                = ctx->network()->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kPREFER_AOT_PYTHON_PLUGINS);
            auto const preferJIT
                = ctx->network()->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kPREFER_JIT_PYTHON_PLUGINS);
            ONNXTRT_CHECK(!(preferAOT && preferJIT),
                "Both NetworkDefinitionCreationFlag::kPREFER_AOT_PYTHON_PLUGINS and "
                "NetworkDefinitionCreationFlag::kPREFER_JIT_PYTHON_PLUGINS cannot be specified at the same time.",
                ErrorCode::kUNSUPPORTED_GRAPH);

            // If neither flag is specified, defer to the plugin creator to pick whichever implementation has actually
            // been defined.
            //  - If both are defined, the plugin creator will raise an error.
            request = preferJIT ? nvinfer1::QuickPluginCreationRequest::kPREFER_JIT
                                : (preferAOT ? nvinfer1::QuickPluginCreationRequest::kPREFER_AOT
                                             : nvinfer1::QuickPluginCreationRequest::kUNKNOWN);
        }

        return std::unique_ptr<nvinfer1::IPluginV3>{static_cast<nvinfer1::IPluginCreatorV3Quick*>(pluginCreator)
                                                        ->createPlugin(pluginOpName.c_str(), pluginNamespace.c_str(),
                                                            &fc, nvinfer1::TensorRTPhase::kBUILD, request)};
    }
    ONNXTRT_CHECK(false, "Found invalid creator version when creating a V3 plugin.", ErrorCode::kINTERNAL_ERROR);
}

NodeOutputs staticSliceImporter(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ITensor& data)
{

    auto const nbInputs = inputs.size();
    nvinfer1::Dims inputDims = data.getDimensions();
    auto const nbDims = inputDims.nbDims;

    // Create start, sizes, and steps with default values
    nvinfer1::Dims starts{nbDims, {}};
    nvinfer1::Dims sizes{nbDims, {}};
    nvinfer1::Dims steps{nbDims, {}};

    for (int32_t i = 0; i < nbDims; i++)
    {
        starts.d[i] = 0;
        sizes.d[i] = inputDims.d[i];
        steps.d[i] = 1;
    }

    // Default axes / steps values
    std::vector<int32_t> defaultAxes(nbDims);
    std::iota(defaultAxes.begin(), defaultAxes.end(), 0);
    std::vector<int32_t> defaultSteps(nbDims, 1);

    // Get int32 pointer representation of ONNX provided values
    int32_t* startVals = static_cast<int32_t*>(inputs.at(1).weights().values);
    int32_t* endVals = static_cast<int32_t*>(inputs.at(2).weights().values);
    int32_t* axesVals = nbInputs > 3 ? static_cast<int32_t*>(inputs.at(3).weights().values) : defaultAxes.data();
    int32_t* stepVals = nbInputs > 4 ? static_cast<int32_t*>(inputs.at(4).weights().values) : defaultSteps.data();

    // Handle non-standard values for slice
    // Start values must in range of [0, dims.d[i]] for + steps, [0, dims.d[i] - 1] for - steps
    auto convertStarts = [](int32_t start, int32_t upper, int32_t stepSign) {
        int32_t newStarts = start < 0 ? start + upper : start;
        newStarts = std::min(std::max(newStarts, 0), upper + stepSign);
        return newStarts;
    };

    // End values must in range of [0, dims.d[i]] for + steps, [-1, dims.d[i]] for - steps
    auto convertEnds = [](int32_t end, int32_t upper, int32_t stepSign) {
        int32_t newEnds = end < 0 ? end + upper : end;
        newEnds = std::min(std::max(newEnds, stepSign), upper);
        return newEnds;
    };
    // Axes values must in range of [0, nbDims]
    auto convertAxes = [&nbDims](int32_t axis) { return axis < 0 ? axis + nbDims : axis; };

    // Since axes can be sparse, get the expected number of provided values
    auto const nbValues = inputs.at(1).shape().d[0];

    for (int32_t i = 0; i < nbValues; i++)
    {
        auto axesIndex = convertAxes(axesVals[i]);
        // Modify starts
        int32_t stepSign = stepVals[i] < 0 ? -1 : 0;
        starts.d[axesIndex] = convertStarts(startVals[i], inputDims.d[axesIndex], stepSign);
        // Modify ends
        int32_t modifiedEnds = convertEnds(endVals[i], inputDims.d[axesIndex], stepSign);
        steps.d[axesIndex] = stepVals[i];
        // Perform ceil integer division of (ends - starts) / steps to compute sizes.
        // Note ceil(x/y) = (x+y-1) / y for postive x & y, and ceil(x/y) = (x+y+1)/y for negative x&y
        // Negative sizes indicates an empty slice, so clamp to 0
        sizes.d[axesIndex] = std::max<int64_t>(
            (modifiedEnds - starts.d[axesIndex] + steps.d[axesIndex] - (steps.d[axesIndex] > 0 ? 1 : -1))
                / steps.d[axesIndex],
            0);
    }

    auto* slice = N_CHECK(ctx->network()->addSlice(data, starts, sizes, steps));
    ctx->registerLayer(slice, node);
    auto output = N_CHECK(slice->getOutput(0));
    return {{output}};
}

bool isDynamic(nvinfer1::Dims const& shape)
{
    return std::any_of(shape.d, shape.d + shape.nbDims, [](int dim) { return dim < 0; });
}

NodeOutputs modulatedDeformableConvPluginHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    size_t const nodeIdx, std::vector<TensorOrWeights>& inputs)
{
    nvinfer1::ITensor* inputXPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weightPtr = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* offsetPtr = &convertToTensor(inputs.at(2), ctx);
    int32_t nbDims = inputXPtr->getDimensions().nbDims;
    ONNXTRT_CHECK_NODE(nbDims >= 3 && nbDims <= 4, "TensorRT only supports DeformConv on 3D, or 4D tensors!", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    bool const needToExpandDims = (nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> const axes{3};
        inputXPtr = unsqueezeTensor(ctx, *inputXPtr, axes);
        weightPtr = unsqueezeTensor(ctx, *weightPtr, axes);
        offsetPtr = unsqueezeTensor(ctx, *offsetPtr, axes);
        ONNXTRT_CHECK(inputXPtr, "Failed to unsqueeze the input tensor.", ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK(weightPtr, "Failed to unsqueeze the weight tensor.", ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK(offsetPtr, "Failed to unsqueeze the offset tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }

    // Parse attributes
    OnnxAttrs attrs(node, ctx);
    int32_t nbSpatialDims = nbDims - 2;
    if (attrs.count("kernel_shape"))
    {
        ONNXTRT_CHECK(nbSpatialDims == attrs.at("kernel_shape")->ints().size(),
            "The attribute kernel_shape misaligns with the shape of the weight tensor.", ErrorCode::kUNSUPPORTED_NODE);
        ONNXTRT_CHECK_NODE(((nbSpatialDims == 1 && needToExpandDims) || nbSpatialDims == 2),
            "The attribute kernel_shape misaligns with the shape of the input tensor.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::Dims dilations = makeDims(nbSpatialDims, /*Default value of dilations*/ 1);
    if (attrs.count("dilations"))
    {
        auto const* onnxDilations = attrs.at("dilations");
        setAttr(&dilations, onnxDilations, nbSpatialDims, 1);
    }

    nvinfer1::Dims kernelShape = makeDims(nbSpatialDims, 0);
    if (attrs.count("kernel_shape"))
    {
        auto const* onnxKernelShape = attrs.at("kernel_shape");
        setAttr(&kernelShape, onnxKernelShape, nbSpatialDims, 0);
    }
    else
    {
        auto weightTensorShape = inputs.at(1).shape();
        for (int32_t i = 0; i < nbSpatialDims; i++)
        {
            kernelShape.d[i] = weightTensorShape.d[2 + i];
        }
    }

    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, /*Default value of pads*/ 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, /*Default value of pads*/ 0);

    if (attrs.count("pads"))
    {
        auto onnxPadding = attrs.get<std::vector<int32_t>>("pads");
        int32_t ndim = onnxPadding.size() / 2;
        ONNXTRT_CHECK(ndim == nbSpatialDims,
            "The given pads attribute mismatch with the spatial dimensions of the weight tensor.",
            ErrorCode::kUNSUPPORTED_NODE);
        for (int32_t i = 0; i < nbSpatialDims; ++i)
        {
            begPadding.d[i] = onnxPadding.at(i);
            endPadding.d[i] = onnxPadding.at(i + ndim);
        }
    }

    ONNXTRT_CHECK(begPadding == endPadding,
        "TensorRT only support the pads attribute of the DeformConv operator where the same number of pixels are added "
        "to the beginning and the end of the corresponding axis.",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Dims strides = makeDims(nbSpatialDims, /*Default value of strides*/ 1);
    if (attrs.count("strides"))
    {
        auto const* onnxStrides = attrs.at("strides");
        setAttr(&strides, onnxStrides, nbSpatialDims, 1);
    }

    int32_t group = attrs.get("group", 1);
    int32_t offset_group = attrs.get("offset_group", 1);

    // Populate instanceNormalization plugin properties.
    std::string const pluginName = "ModulatedDeformConv2d";
    std::string const pluginVersion = "2";
    std::vector<nvinfer1::PluginField> f;

    // Unsqueeze the list attributes if necessary
    int32_t listAttrSize = nbSpatialDims == 1 ? 2 : nbSpatialDims;
    std::vector<int32_t> dilationValues(listAttrSize, /*Default value of dilations*/ 1);
    std::vector<int32_t> strideValues(listAttrSize, /*Default value of strides*/ 1);
    std::vector<int32_t> paddingValues(listAttrSize, /*Default value of pads*/ 0);

    for (int32_t i = 0; i < nbSpatialDims; i++)
    {
        dilationValues[i] = static_cast<int32_t>(dilations.d[i]);
        strideValues[i] = static_cast<int32_t>(strides.d[i]);
        paddingValues[i] = static_cast<int32_t>(begPadding.d[i]);
    }

    f.emplace_back("group", &group, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("deformable_group", &offset_group, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("stride", strideValues.data(), nvinfer1::PluginFieldType::kINT32, listAttrSize);
    f.emplace_back("padding", paddingValues.data(), nvinfer1::PluginFieldType::kINT32, listAttrSize);
    f.emplace_back("dilation", dilationValues.data(), nvinfer1::PluginFieldType::kINT32, listAttrSize);

    // Create plugin from registry
    auto const plugin = createPlugin(ctx, node, pluginName, kTRT_STD_PLUGIN_NAMESPACE,
        static_cast<nvinfer1::IPluginCreatorInterface*>(importPluginCreator(ctx, pluginName, pluginVersion)), f);

    ONNXTRT_CHECK_NODE(plugin != nullptr, "ModulatedDeformConv2d plugin was not found in the plugin registry!", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* biasPtr = nullptr;
    nvinfer1::ITensor* maskPtr = nullptr;

    // Create the default mask input if not provided.
    // The mask input is optional in ONNX but is required by the ModulatedDeformConv plugin.
    if (inputs.size() > 4)
    {
        // Add the optional Mask tensor input.
        maskPtr = &convertToTensor(inputs.at(4), ctx);
        if (needToExpandDims)
        {
            // Expand spatial dims from 1D to 2D
            std::vector<int32_t> const axes{3};
            maskPtr = unsqueezeTensor(ctx, *maskPtr, axes);
            ONNXTRT_CHECK(maskPtr, "Failed to unsqueeze the mask tensor.", ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    else
    {
        // Create the default mask input as a tensor of ones.
        // The offset and mask inputs have the same shape.
        nvinfer1::ITensor& maskShape = shapeOf(*offsetPtr).tensor(ctx);
        ShapedWeights defaultMaskWeights
            = ctx->createNamedTempWeights(inputs.at(0).getONNXDataType(), nvinfer1::Dims{1, {1}});

        if (inputs.at(0).getDataType() == nvinfer1::DataType::kHALF)
        {
            static_cast<half_float::half*>(defaultMaskWeights.values)[0] = 1.0;
            auto maskTensor = TensorOrWeights{defaultMaskWeights};
            maskPtr = constantOfShape(ctx, &convertToTensor(maskTensor, ctx), &maskShape);
        }
        else
        {
            static_cast<float*>(defaultMaskWeights.values)[0] = 1.F;
            auto maskTensor = TensorOrWeights{defaultMaskWeights};
            maskPtr = constantOfShape(ctx, &convertToTensor(maskTensor, ctx), &maskShape);
        }
    }

    if (inputs.size() > 3)
    {
        // Add the optional Bias tensor input.
        biasPtr = &convertToTensor(inputs.at(3), ctx);
    }

    std::vector<nvinfer1::ITensor*> inputTensorsPtrs = {inputXPtr, offsetPtr, maskPtr, weightPtr};
    if (biasPtr != nullptr)
    {
        inputTensorsPtrs.push_back(biasPtr);
    }

    auto* layer
        = N_CHECK(ctx->network()->addPluginV3(inputTensorsPtrs.data(), inputTensorsPtrs.size(), nullptr, 0, *plugin));
    ctx->registerLayer(layer, node);
    nvinfer1::ITensor* outputPtr = N_CHECK(layer->getOutput(0));

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> const axes{3};
        outputPtr = squeezeTensor(ctx, *outputPtr, axes);
        ONNXTRT_CHECK_NODE(outputPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{outputPtr}};
}

NodeOutputs instanceNormPluginHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    size_t const nodeIdx, std::vector<TensorOrWeights>& inputs)
{
    // Scales and biases must be initializers
    ONNXTRT_CHECK_NODE(inputs.at(1).is_weights(), "The scale tensor is required to be an initializer.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
    ONNXTRT_CHECK_NODE(inputs.at(2).is_weights(), "The bias tensor is required to be an initializer.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int32_t nbDims = tensorPtr->getDimensions().nbDims;
    ONNXTRT_CHECK_NODE(nbDims >= 3 && nbDims <= 5,
        "TensorRT only supports InstanceNormalization on 3D, 4D, or 5D tensors!", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    bool const needToExpandDims = (nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> const axes{3};
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK(tensorPtr, "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }
    auto scaleWeights = inputs.at(1).weights();
    auto biasWeights = inputs.at(2).weights();
    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5F);
    int32_t const relu{0};  // the ONNX instance norm op does not use the relu parameter
    float const alpha{0.F}; // the ONNX instance norm op does not use the alpha parameter

    // Populate instanceNormalization plugin properties.
    std::string const pluginName = "InstanceNormalization_TRT";
    std::string const pluginVersion = "3";
    std::vector<nvinfer1::PluginField> f;

    // get the values of constant inputs and cast them to float32
    float const* scaleValues = ctx->getWeightsContext().getFP32Values(scaleWeights);
    float const* biasValues = ctx->getWeightsContext().getFP32Values(biasWeights);

    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", scaleValues, nvinfer1::PluginFieldType::kFLOAT32, scaleWeights.count());
    f.emplace_back("bias", biasValues, nvinfer1::PluginFieldType::kFLOAT32, biasWeights.count());
    f.emplace_back("relu", &relu, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1);

    // Create plugin from registry
    auto const plugin = createPlugin(ctx, node, getNodeName(node), kTRT_STD_PLUGIN_NAMESPACE,
        static_cast<nvinfer1::IPluginCreatorV3One*>(importPluginCreator(ctx, pluginName, pluginVersion)), f);

    ONNXTRT_CHECK_NODE(plugin != nullptr, "InstanceNormalization plugin was not found in the plugin registry!", node,
        nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = N_CHECK(ctx->network()->addPluginV3(&tensorPtr, 1, nullptr, 0, *plugin));
    ctx->registerLayer(layer, node);
    tensorPtr = N_CHECK(layer->getOutput(0));

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> const axes{3};
        tensorPtr = squeezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK_NODE(tensorPtr, "Failed to squeeze tensor.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensorPtr}};
}

nvinfer1::ITensor* iota(ImporterContext* ctx, ShapeTensor iotaDims, int32_t axis)
{
    std::vector<int32_t> deltaVals(iotaDims.size(), 0);
    deltaVals[axis] = 1;
    auto* iota
        = N_CHECK(ctx->network()->addFill({0, {0}}, nvinfer1::FillOperation::kLINSPACE, nvinfer1::DataType::kINT32));
    auto* alphaLayer = N_CHECK(addConstantScalar(ctx, static_cast<int32_t>(0), ::ONNX_NAMESPACE::TensorProto::INT32));
    auto* alpha = N_CHECK(alphaLayer->getOutput(0));
    auto* deltaLayer
        = N_CHECK(addConstant(ctx, deltaVals, ::ONNX_NAMESPACE::TensorProto::INT32, {1, {iotaDims.size()}}));
    auto* delta = N_CHECK(deltaLayer->getOutput(0));
    iota->setInput(0, iotaDims.tensor(ctx));
    iota->setInput(1, *alpha);
    iota->setInput(2, *delta);
    ctx->registerLayer(iota, "ONNXTRT_iota", nullptr);
    return castHelper(ctx, N_CHECK(iota->getOutput(0)), nvinfer1::DataType::kINT64);
}

TensorOrWeights identity(ImporterContext* ctx, TensorOrWeights input)
{
    if (input.is_weights())
    {
        return input;
    }
    else
    {
        auto* layer = N_CHECK(ctx->network()->addIdentity(input.tensor()));
        ctx->registerLayer(layer, "ONNXTRT_identity", nullptr);
        return N_CHECK(layer->getOutput(0));
    }
}

nvinfer1::Dims makeDims(int32_t nbDims, int64_t val)
{
    // Zero all the dimensions, so that unused dimensions are deterministic even if accidentally used.
    nvinfer1::Dims dims{nbDims, {}};
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

NodeOutputs normalizationHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, bool const useV2)
{
    auto* input = &convertToTensor(inputs.at(0), ctx);
    auto* scale = &convertToTensor(inputs.at(1), ctx);
    auto* bias = &convertToTensor(inputs.at(2), ctx);

    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    int32_t nbGroups = attrs.get("num_groups", 1);

    auto nbDims = input->getDimensions().nbDims;
    ONNXTRT_CHECK_NODE(nbDims >= 3,
        "Input to normalization should be at least 3D, the actual number of dimensions is " << nbDims << ".", node,
        nodeIdx, ErrorCode::kINVALID_NODE);

    // Need to broadcast scale and bias to the input shape. Note that normal broadcasting rules cannot be applied
    // as scale and bias are 1D and need to be broadcasted to shape [1, S, 1, 1, ...].
    uint32_t axesMask{0};
    std::vector<int32_t> unsqueezeAxes;

    for (int32_t i = 0; i < nbDims; i++)
    {
        if (i == 1)
        {
            continue;
        }
        // Axes should correspond to the spatial dimensions
        if (i >= 2)
        {
            axesMask |= 1 << i;
        }
        unsqueezeAxes.push_back(i);
    }

    scale = unsqueezeTensor(ctx, *scale, unsqueezeAxes);
    bias = unsqueezeTensor(ctx, *bias, unsqueezeAxes);

    auto* layer = useV2 ? N_CHECK(ctx->network()->addNormalizationV2(*input, *scale, *bias, axesMask)) : N_CHECK(ctx->network()->addNormalization(*input, *scale, *bias, axesMask));
    layer->setEpsilon(epsilon);
    layer->setNbGroups(nbGroups);
    ctx->registerLayer(layer, node);
    auto* output = N_CHECK(layer->getOutput(0));
    return {{output}};
}

void normalizeAxes(ShapeTensor& axes, int32_t const rank)
{
    ONNXTRT_CHECK(axes.allValuesKnown(), "Axes should not contain unknown values.", ErrorCode::kINTERNAL_ERROR);
    std::vector<int64_t> newAxes;
    newAxes.reserve(axes.size());
    for (int64_t axis : axes)
    {
        ONNXTRT_CHECK(
            (-rank <= axis && axis < rank), "Axis must be in the range of [-rank, rank-1].", ErrorCode::kINVALID_VALUE);
        // "Negative value means counting dimensions from the back."
        if (axis < 0)
        {
            axis += rank;
        }
        newAxes.push_back(axis);
    }
    axes = ShapeTensor(1, std::move(newAxes));
}

nvinfer1::Dims insertDimension(nvinfer1::Dims const& dims, int const axis, int const value)
{
    if (axis >= nvinfer1::Dims::MAX_DIMS || dims.nbDims >= nvinfer1::Dims::MAX_DIMS)
    {
        throw std::invalid_argument("Cannot insert a dimension past Dims::MAX_DIMS!");
    }
    nvinfer1::Dims newDims{};
    newDims.nbDims = dims.nbDims + 1;
    std::copy(dims.d, dims.d + axis, newDims.d);
    newDims.d[axis] = value;
    std::copy(dims.d + axis, dims.d + dims.nbDims, newDims.d + axis + 1);
    return newDims;
}

std::vector<float> parseLSTMActivationValues(std::vector<nvinfer1::ActivationType> const& activationTypes,
    std::vector<float> const& activationValues, bool isAlpha)
{
    size_t actIndex{0};
    std::vector<float> tmpActs{};
    for (size_t i = 0; i < activationTypes.size(); ++i)
    {
        float defaultVal
            = isAlpha ? getActivationDefaultAlpha(activationTypes[i]) : getActivationDefaultBeta(activationTypes[i]);
        if (defaultVal == 0.f || actIndex == activationValues.size())
        {
            tmpActs.push_back(defaultVal);
        }
        else
        {
            tmpActs.push_back(activationValues[actIndex]);
            actIndex++;
        }
    }
    return tmpActs;
}

NodeOutputs poolingHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensorPtr->getDimensions();
    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> axes{3};
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ONNXTRT_CHECK(tensorPtr, "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }

    OnnxAttrs attrs(node, ctx);
    int nbSpatialDims = attrs.at("kernel_shape")->ints().size();
    ONNXTRT_CHECK_NODE(((nbSpatialDims == 1 && needToExpandDims) || nbSpatialDims == 2 || nbSpatialDims == 3),
        "The attribute kernel_shape misaligns with the shape of the input tensor.", node, nodeIdx,
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Dims kernelSize = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;

    bool exclude_padding(true);

    // Ceiling-mode output padding and dilations added in opset 10
    bool ceilMode(false);
    if (ctx->getOpsetVersion() >= 10)
    {
        ceilMode = static_cast<bool>(attrs.get<int>("ceil_mode", 0));
    }

    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, exclude_padding, nullptr,
        nullptr, ceilMode);

    if (needToExpandDims)
    {
        kernelSize = insertDimension(kernelSize, nbSpatialDims, 1);
        strides = insertDimension(strides, nbSpatialDims, 1);
        begPadding = insertDimension(begPadding, nbSpatialDims, 0);
        endPadding = insertDimension(endPadding, nbSpatialDims, 0);
    }

    nvinfer1::IPoolingLayer* poolingLayer = N_CHECK(ctx->network()->addPoolingNd(*tensorPtr, type, kernelSize));
    poolingLayer->setStrideNd(strides);
    // This member is ignored in maxpooling
    poolingLayer->setAverageCountExcludesPadding(exclude_padding);
    poolingLayer->setPaddingMode(paddingMode);
    poolingLayer->setPrePadding(begPadding);
    poolingLayer->setPostPadding(endPadding);

    ctx->registerLayer(poolingLayer, node);
    tensorPtr = N_CHECK(poolingLayer->getOutput(0));
    dims = tensorPtr->getDimensions();
    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> axes{3};
        tensorPtr = squeezeTensor(ctx, *tensorPtr, axes);
    }
    return {{tensorPtr}};
}

bool IsReduceNoOp(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights> const& inputs)
{
    OnnxAttrs attrs(node, ctx);
    return (attrs.get("noop_with_empty_axes", 0) == 1) && (!attrs.count("axes")) && (inputs.size() == 1);
}

NodeOutputs reduceTensor(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    TensorOrWeights input, nvinfer1::ReduceOperation operation, TensorOrWeights inputAxes)
{
    // TensorRT does not support reduction on Bool or UINT8 tensors.
    checkNotInvalidType(input, {"BOOL", "UINT8"}, node, nodeIdx);

    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor& tensor = convertToTensor(input, ctx);
    bool keepdims = attrs.get("keepdims", 1);
    int32_t ndim = tensor.getDimensions().nbDims;
    std::vector<int32_t> axes;
    if (attrs.count("axes"))
    {
        axes = attrs.get<std::vector<int32_t>>("axes");
    }
    else if (!inputAxes.isNullTensor())
    {
        ONNXTRT_CHECK_NODE(
            inputAxes.is_weights(), "Axis input must be an initializer!", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        weightsToVector<int32_t>(inputAxes.weights(), &axes);
    }
    // It's possible that the axes tensor, axes initializer, or axes attribute was empty. Handle such cases here.
    if (axes.empty())
    {
        // Fast return path for no-op case.
        if (attrs.get("noop_with_empty_axes", 0) == 1)
        {
            TensorOrWeights output = identity(ctx, input);
            return {{output}};
        }
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }

    uint32_t axisMask = 0;
    for (int32_t axis : axes)
    {
        convertAxis(axis, ndim, node, nodeIdx);
        axisMask |= 1 << axis;
    }

    auto* layer = N_CHECK(ctx->network()->addReduce(tensor, operation, axisMask, keepdims));
    ctx->registerLayer(layer, node);
    auto output = N_CHECK(layer->getOutput(0));
    return {{output}};
}

nvinfer1::ITensor* reshapeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape)
{
    if (shape == tensor.getDimensions())
    {
        return &tensor;
    }
    nvinfer1::IShuffleLayer* layer = N_CHECK(ctx->network()->addShuffle(tensor));
    layer->setReshapeDimensions(shape);
    layer->setZeroIsPlaceholder(false);
    ctx->registerLayer(layer, "ONNXTRT_reshapeTensor", nullptr);
    return N_CHECK(layer->getOutput(0));
}

NodeOutputs scaleHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode, nvinfer1::Weights const& shift,
    nvinfer1::Weights const& scale, nvinfer1::Weights const& power, char const* shiftName, char const* scaleName)
{
    nvinfer1::ITensor* tensorPtr = &tensor_;
    const ShapeTensor origShape = shapeOf(*tensorPtr);

    // TensorRT scale layers support 4D(NCHW) or 5D(NCDHW) input.
    // For input other than 4D or 5D will be expanded or squeezed to 4D.
    bool needToReshape = (origShape.size() != 4 && origShape.size() != 5);
    if (needToReshape)
    {
        if (origShape.size() < 4)
        {
            std::vector<int> expandAxes(4 - origShape.size());
            std::iota(expandAxes.begin(), expandAxes.end(), origShape.size());
            tensorPtr = unsqueezeTensor(ctx, *tensorPtr, expandAxes);
        }
        else
        {
            // Collapse trailing dimensions if origShape.size() > 5
            const ShapeTensor collapsedDim = product(ctx, origShape, 3, origShape.size(), 1);
            const ShapeTensor collapsedShape = concat(ctx, gather(ctx, origShape, iotaShapeVector(3)), collapsedDim);
            tensorPtr = &reshape(ctx, *tensorPtr, collapsedShape);
        }
    }

    auto* layer = N_CHECK(ctx->network()->addScaleNd(*tensorPtr, mode, shift, scale, power, 1));
    // Register layer name, and shift and scale weight names for the refit map.
    ctx->registerLayer(layer, node);
    ctx->network()->setWeightsName(shift, shiftName);
    ctx->network()->setWeightsName(scale, scaleName);

    tensorPtr = N_CHECK(layer->getOutput(0));

    if (needToReshape)
    {
        tensorPtr = &reshape(ctx, *tensorPtr, origShape);
    }
    return {{tensorPtr}};
}

void setAttr(nvinfer1::Dims* trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int32_t nbSpatialDims,
    int32_t defaultVal)
{
    assert(trtAttr->nbDims == nbSpatialDims);
    int32_t ndim = onnxAttr->ints().size();
    for (int32_t i = 0; i < nbSpatialDims; ++i)
    {
        if (i < ndim)
        {
            trtAttr->d[i] = onnxAttr->ints(i);
        }
        else
        {
            trtAttr->d[i] = defaultVal;
        }
    }
}

nvinfer1::ITensor* sliceAcrossAxis(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* data, int32_t const axis)
{
    ShapeTensor starts, sizes, strides;
    ShapeTensor axisLength = ShapeTensor(*getAxisLengthInt64(ctx, data, axis, {1, {1}}));
    int32_t const nbDims = data->getDimensions().nbDims;

    std::vector<int64_t> values(nbDims, 0);
    starts = ShapeTensor(1, std::move(values));
    sizes = axis == 0 ? shapeVector(1) : ShapeTensor(*getAxisLengthInt64(ctx, data, 0, {1, {1}}));
    strides = axis == 0 ? axisLength : shapeVector(1);

    // On axis dimension, set strides = lengthOfDim and sizes = 1
    for (int32_t i = 1; i < nbDims; i++)
    {
        if (i == axis)
        {
            strides = concat(ctx, strides, axisLength);
            sizes = concat(ctx, sizes, shapeVector(1));
        }
        else
        {
            ShapeTensor currLength = ShapeTensor(*getAxisLengthInt64(ctx, data, i, {1, {1}}));
            strides = concat(ctx, strides, shapeVector(1));
            sizes = concat(ctx, sizes, currLength);
        }
    }
    auto* sliceLayer = N_CHECK(addSlice(ctx, *data, starts, sizes, strides));
    return N_CHECK(sliceLayer->getOutput(0));
}

nvinfer1::ITensor* squeezeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, std::vector<int32_t> const& axes)
{
    auto* axesTensor
        = N_CHECK(addConstant(ctx, axes, ::ONNX_NAMESPACE::TensorProto::INT32, {1, {static_cast<int64_t>(axes.size())}})
                      ->getOutput(0));
    auto* squeezeLayer = N_CHECK(ctx->network()->addSqueeze(tensor, *axesTensor));
    auto* squeezedTensor = N_CHECK(squeezeLayer->getOutput(0));
    LOG_VERBOSE("Original shape: " << shapeOf(tensor) << ", squeezing to: " << shapeOf(*squeezedTensor));
    ctx->registerLayer(squeezeLayer, "ONNXTRT_squeezeTensor", nullptr);
    return squeezedTensor;
}

nvinfer1::ITensor* transposeTensor(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm)
{
    nvinfer1::IShuffleLayer* layer = N_CHECK(ctx->network()->addShuffle(tensor));
    ctx->registerLayer(layer, node);
    layer->setFirstTranspose(perm);
    return N_CHECK(layer->getOutput(0));
}

::ONNX_NAMESPACE::TensorProto_DataType trtDataTypeToONNX(nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kFLOAT: return ::ONNX_NAMESPACE::TensorProto::FLOAT;
    case nvinfer1::DataType::kHALF: return ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    case nvinfer1::DataType::kBF16: return ::ONNX_NAMESPACE::TensorProto::BFLOAT16;
    case nvinfer1::DataType::kINT32: return ::ONNX_NAMESPACE::TensorProto::INT32;
    case nvinfer1::DataType::kINT64: return ::ONNX_NAMESPACE::TensorProto::INT64;
    case nvinfer1::DataType::kINT8: return ::ONNX_NAMESPACE::TensorProto::INT8;
    case nvinfer1::DataType::kBOOL: return ::ONNX_NAMESPACE::TensorProto::BOOL;
    case nvinfer1::DataType::kUINT8: return ::ONNX_NAMESPACE::TensorProto::UINT8;
    case nvinfer1::DataType::kFP8: return ::ONNX_NAMESPACE::TensorProto::FLOAT8E4M3FN;
    case nvinfer1::DataType::kINT4: return ::ONNX_NAMESPACE::TensorProto::INT4;
    case nvinfer1::DataType::kFP4: return ::ONNX_NAMESPACE::TensorProto::FLOAT4E2M1;
    case nvinfer1::DataType::kE8M0: break;
    }
    return ::ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

std::string getUnaryOpName(nvinfer1::UnaryOperation op)
{
    switch (op)
    {
    case nvinfer1::UnaryOperation::kEXP: return "EXP";
    case nvinfer1::UnaryOperation::kLOG: return "LOG";
    case nvinfer1::UnaryOperation::kSQRT: return "SQRT";
    case nvinfer1::UnaryOperation::kRECIP: return "RECIP";
    case nvinfer1::UnaryOperation::kABS: return "ABS";
    case nvinfer1::UnaryOperation::kNEG: return "NEG";
    case nvinfer1::UnaryOperation::kSIN: return "SIN";
    case nvinfer1::UnaryOperation::kCOS: return "COS";
    case nvinfer1::UnaryOperation::kTAN: return "TAN";
    case nvinfer1::UnaryOperation::kSINH: return "SINH";
    case nvinfer1::UnaryOperation::kCOSH: return "COSH";
    case nvinfer1::UnaryOperation::kASIN: return "ASIN";
    case nvinfer1::UnaryOperation::kACOS: return "ACOS";
    case nvinfer1::UnaryOperation::kATAN: return "ATAN";
    case nvinfer1::UnaryOperation::kASINH: return "ASINH";
    case nvinfer1::UnaryOperation::kACOSH: return "ACOSH";
    case nvinfer1::UnaryOperation::kATANH: return "ATANH";
    case nvinfer1::UnaryOperation::kCEIL: return "CEIL";
    case nvinfer1::UnaryOperation::kFLOOR: return "FLOOR";
    case nvinfer1::UnaryOperation::kERF: return "ERF";
    case nvinfer1::UnaryOperation::kNOT: return "NOT";
    case nvinfer1::UnaryOperation::kSIGN: return "SIGN";
    case nvinfer1::UnaryOperation::kROUND: return "ROUND";
    case nvinfer1::UnaryOperation::kISINF: return "ISINF";
    default: return "<UNKNOWN>";
    }
}

NodeOutputs unaryHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    TensorOrWeights& input, nvinfer1::UnaryOperation op)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(input, ctx);
    auto const inputType = tensorPtr->getType();

    bool validUnaryType = true;
    switch (op)
    {
    // TRT only supports BOOL types for the NOT operation
    case nvinfer1::UnaryOperation::kNOT:
    {
        validUnaryType = inputType == nvinfer1::DataType::kBOOL;
        break;
    }
    // ABS and SIGN supports everything except BOOL and UINT8.
    case nvinfer1::UnaryOperation::kABS:
    case nvinfer1::UnaryOperation::kSIGN:
    {
        validUnaryType = (inputType != nvinfer1::DataType::kBOOL && inputType != nvinfer1::DataType::kUINT8);
        break;
    }
    case nvinfer1::UnaryOperation::kNEG:
    {
        // WAR: NEG can work with INT32 types via ElementWise Layer: (0 - x)
        if (inputType == nvinfer1::DataType::kINT32)
        {
            // Calculate the rank of the input, and set all size to one and rely on broadcasting
            auto* zeroLayer
                = N_CHECK(addConstant(ctx, std::vector<int32_t>{0}, ::ONNX_NAMESPACE::TensorProto::INT32, {0, {1}}));
            nvinfer1::ITensor* zeroTensor = N_CHECK(zeroLayer->getOutput(0));
            broadcastTensors(ctx, zeroTensor, tensorPtr);
            std::vector<TensorOrWeights> layerInputs = {zeroTensor, tensorPtr};
            return elementwiseHelper(ctx, node, nodeIdx, layerInputs, nvinfer1::ElementWiseOperation::kSUB);
        }
        validUnaryType = (inputType != nvinfer1::DataType::kBOOL && inputType != nvinfer1::DataType::kUINT8);
        break;
    }
    default:
    {
        // By default TRT does not support BOOL, INT32, INT64, and UINT8 types for Unary operations.
        validUnaryType = (inputType != nvinfer1::DataType::kBOOL && inputType != nvinfer1::DataType::kINT32
            && inputType != nvinfer1::DataType::kINT64 && inputType != nvinfer1::DataType::kUINT8);
    }
    }

    ONNXTRT_CHECK_NODE(validUnaryType,
        "This version of TensorRT does not support the given operator " + getUnaryOpName(op)
            + " with the given input data type " + getTrtDtypeName(inputType) + ".",
        node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::IUnaryLayer* layer = N_CHECK(ctx->network()->addUnary(*tensorPtr, op));
    ctx->registerLayer(layer, node);
    tensorPtr = N_CHECK(layer->getOutput(0));

    return {{tensorPtr}};
}

NodeOutputs convMultiInput(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs)
{
    ONNXTRT_CHECK(inputs.size() >= 2, "Convolution require at least 2 inputs.", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = input->getDimensions();
    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int32_t> const axes{3};
        input = unsqueezeTensor(ctx, *input, axes);
        dims = input->getDimensions();
    }
    auto const nbSpatialDims = dims.nbDims - 2;

    nvinfer1::Dims kernelDims;
    kernelDims.nbDims = nbSpatialDims;

    // Populate spatial dims from the shape of the convolution weights.
    if (needToExpandDims)
    {
        kernelDims.d[0] = inputs.at(1).shape().d[2];
        kernelDims.d[1] = 1;
    }
    else
    {
        for (int32_t i = 1; i <= nbSpatialDims; ++i)
        {
            kernelDims.d[nbSpatialDims - i] = inputs.at(1).shape().d[inputs.at(1).shape().nbDims - i];
        }
    }

    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims begPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims endPadding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool excludePadding{false};
    getKernelParams(
        ctx, node, &kernelDims, &strides, &begPadding, &endPadding, paddingMode, excludePadding, &dilations);
    auto const nChannel = dims.d[1];
    auto const K = inputs.at(1).shape().d[0];
    auto const C = inputs.at(1).shape().d[1];

    auto kernelWeights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto biasWeights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);

    auto const checkSpatialDims = [&nbSpatialDims, &kernelDims](nvinfer1::Dims const& dims) {
        // Check that the number of spatial dimensions and the kernel shape matches up.
        if (nbSpatialDims != dims.nbDims - 2)
        {
            return false;
        }

        return std::equal(kernelDims.d, kernelDims.d + nbSpatialDims, dims.d + dims.nbDims - nbSpatialDims);
    };

    nvinfer1::ITensor* kernelTensor{nullptr};
    nvinfer1::ITensor* biasTensor{nullptr};
    if (inputs.at(1).is_tensor())
    {
        kernelTensor = &convertToTensor(inputs.at(1), ctx);
        if (needToExpandDims)
        {
            // Expand spatial dims from 1D to 2D
            std::vector<int32_t> const axes{3};
            kernelTensor = unsqueezeTensor(ctx, *kernelTensor, axes);
            ONNXTRT_CHECK(kernelTensor, "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        }
        ONNXTRT_CHECK(checkSpatialDims(kernelTensor->getDimensions()),
            "The input tensor shape misaligns with the input kernel shape.", ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        kernelWeights = inputs.at(1).weights();
        if (needToExpandDims)
        {
            kernelWeights.shape.nbDims = 4;
            kernelWeights.shape.d[3] = 1;
        }
        ONNXTRT_CHECK_NODE(checkSpatialDims(kernelWeights.shape),
            "The input tensor shape misaligns with the input kernel shape.", node, nodeIdx,
            ErrorCode::kUNSUPPORTED_NODE);
    }

    if (inputs.size() == 3)
    {
        if (inputs.at(2).is_weights())
        {
            biasWeights = inputs.at(2).weights();
        }
        else
        {
            biasTensor = &convertToTensor(inputs.at(2), ctx);
        }
    }

    OnnxAttrs attrs(node, ctx);
    int32_t ngroup = attrs.get("group", 1);
    ONNXTRT_CHECK_NODE((nChannel == -1 || C * ngroup == nChannel),
        "The attribute group and the kernel shape misalign with the channel size of the input tensor. ", node, nodeIdx,
        ErrorCode::kINVALID_NODE);

    nvinfer1::IConvolutionLayer* layer
        = N_CHECK(ctx->network()->addConvolutionNd(*input, K, kernelDims, kernelWeights, biasWeights));
    ONNXTRT_CHECK_NODE(layer, "Failed to add the Convolution layer.", node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);
    layer->setDilationNd(dilations);
    layer->setNbGroups(ngroup);

    // Set dynamic weights
    if (kernelTensor)
    {
        layer->setInput(1, *kernelTensor);
    }
    if (biasTensor)
    {
        layer->setInput(2, *biasTensor);
    }
    ctx->registerLayer(layer, node);
    if (kernelWeights)
    {
        ctx->network()->setWeightsName(kernelWeights, inputs.at(1).getName().c_str());
    }
    if (biasWeights && inputs.size() == 3)
    {
        ctx->network()->setWeightsName(biasWeights, inputs.at(2).getName().c_str());
    }

    nvinfer1::ITensor* outputTensor = N_CHECK(layer->getOutput(0));
    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int32_t> const axes{3};
        outputTensor = squeezeTensor(ctx, *outputTensor, axes);
    }

    return {{outputTensor}};
}

nvinfer1::ITensor* unsqueezeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, std::vector<int32_t> const& axes)
{
    auto* axesTensor
        = N_CHECK(addConstant(ctx, axes, ::ONNX_NAMESPACE::TensorProto::INT32, {1, {static_cast<int64_t>(axes.size())}})
                      ->getOutput(0));
    auto* unsqueezeLayer = N_CHECK(ctx->network()->addUnsqueeze(tensor, *axesTensor));
    auto* unsqueezedTensor = N_CHECK(unsqueezeLayer->getOutput(0));
    LOG_VERBOSE("Original shape: " << shapeOf(tensor) << ", unsqueezing to: " << shapeOf(*unsqueezedTensor));
    ctx->registerLayer(unsqueezeLayer, "ONNXTRT_unsqueezeTensor", nullptr);
    return N_CHECK(unsqueezeLayer->getOutput(0));
}

nvinfer1::ITensor* resizeShapeTensor(ImporterContext* ctx, nvinfer1::ITensor& input, TensorOrWeights& scales)
{
    // Create below subnetwork for processing resize scale tensor or weights.
    // clang-format off
    // scale weights (convert to tensor) or scale tensor -> elementwise mul -> transformation(floor, ceil, round) -> identity (cast to int) -> resize shape tensor
    //      input -> shapeof -> identity (cast to float) ->
    // clang-format on
    auto* floatCast = N_CHECK(ctx->network()->addCast(shapeOf(input).tensor(ctx), nvinfer1::DataType::kFLOAT));
    ctx->registerLayer(floatCast, "ONNXTRT_resizeShapeTensor_floatCast", nullptr);
    auto* inputShapeTensor = N_CHECK(floatCast->getOutput(0));

    auto& scaleTensor = convertToTensor(scales, ctx);
    auto* prodLayer = N_CHECK(
        ctx->network()->addElementWise(scaleTensor, *inputShapeTensor, nvinfer1::ElementWiseOperation::kPROD));
    ctx->registerLayer(prodLayer, "ONNXTRT_resizeShapeTensor_prod", nullptr);

    auto* prod = N_CHECK(prodLayer->getOutput(0));
    auto* floorLayer = N_CHECK(ctx->network()->addUnary(*prod, nvinfer1::UnaryOperation::kFLOOR));
    ctx->registerLayer(floorLayer, "ONNXTRT_resizeShapeTensor_floor", nullptr);

    auto* floor = N_CHECK(floorLayer->getOutput(0));
    auto* intCast = N_CHECK(ctx->network()->addCast(*floor, nvinfer1::DataType::kINT32));
    ctx->registerLayer(intCast, "ONNXTRT_resizeShapeTensor_intCast", nullptr);
    return N_CHECK(intCast->getOutput(0));
}

std::string const getNodeName(::ONNX_NAMESPACE::NodeProto const& node)
{
    if (node.name().empty() && (node.output_size() != 0))
    {
        return "node_of_" + node.output(0);
    }
    else
    {
        return node.name();
    }
}

//! Return ShapeTensor representing x clamped to closed interval [lowerBound,upperBound].
static ShapeTensor clamp(
    ImporterContext* ctx, ShapeTensor const& x, ShapeTensor const& lowerBound, ShapeTensor const& upperBound)
{
    return min(ctx, max(ctx, x, lowerBound), upperBound);
}

//! Return ShapeTensor representing indices < 0 ? inputDims + indices : indices
static ShapeTensor bumpIfNegative(ImporterContext* ctx, ShapeTensor const& inputDims, ShapeTensor const& indices)
{
    auto const signs = clamp(ctx, indices, shapeVector(-1), shapeVector(0));
    return sub(ctx, indices, mul(ctx, signs, inputDims));
}

void decodeOnnxStartsAndEnds(ImporterContext* ctx, ShapeTensor const& inputDims, ShapeTensor const& steps,
    ShapeTensor& starts, ShapeTensor& ends)
{
    //! The ONNX specification is unreliable (https://github.com/onnx/onnx/issues/3063)
    //! thus the logic here is designed to match that in
    //! https://github.com/onnx/onnx/blob/master/onnx/defs/tensor/defs.cc .

    // Set stepSign to step < 0 ? -1 : 0.
    auto const stepSign = clamp(ctx, steps, shapeVector(-1), shapeVector(0));

    // Update starts.
    starts = bumpIfNegative(ctx, inputDims, starts);
    starts = clamp(ctx, starts, shapeVector(0), add(ctx, inputDims, stepSign));

    // Update ends
    ends = bumpIfNegative(ctx, inputDims, ends);
    ends = clamp(ctx, ends, stepSign, inputDims);
}

ShapeTensor axesToInterlaceSubscripts(ShapeTensor const& axes, int nbDims)
{
    std::vector<int64_t> subscripts(nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (int32_t i = 0; i < axes.size(); ++i)
    {
        subscripts[axes[i]] = nbDims + i;
    }
    return ShapeTensor(1, std::move(subscripts));
}

ShapeTensor computeSliceSizes(ImporterContext* ctx, ShapeTensor const& starts, ShapeTensor const& ends,
    ShapeTensor const& steps, ShapeTensor const& dims)
{
    if (steps.isAll(1))
    {
        // The general formula in the else is correct,
        // but creates much debris for this common case.
        return sub(ctx, ends, starts);
    }
    // "If a negative value is passed for step, it represents slicing backward."
    // Compute ceil((end-start)/step) using only operations available on ShapeTensor,
    // using the identity ceil(x) = -floor(-x).
    return sub(ctx, similar(ctx, dims, 0), floorDiv(ctx, sub(ctx, starts, ends), steps));
}

nvinfer1::ITensor* addSoftmax(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx, nvinfer1::ITensor& input)
{
    OnnxAttrs attrs(node, ctx);
    // "axis : int (default is opset specific)"
    int32_t const defaultAxis = (ctx->getOpsetVersion() >= 13) ? -1 : 1;
    int32_t axis = attrs.get("axis", defaultAxis);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    auto const rank = shapeOf(input).size();
    convertAxis(axis, rank, node, nodeIdx);

    nvinfer1::ISoftMaxLayer* softMax{nullptr};
    if (ctx->getOpsetVersion() >= 13)
    {
        softMax = N_CHECK(ctx->network()->addSoftMax(input));
        softMax->setAxes(1 << axis);
    }
    else
    {
        // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
        auto* flattened = flattenTensor(ctx, node, input, axis);
        softMax = N_CHECK(ctx->network()->addSoftMax(*flattened));
        // ONNX softmax is always on second dimension.
        softMax->setAxes(1 << 1);
    }
    ctx->registerLayer(softMax, node);
    return N_CHECK(softMax->getOutput(0));
}

NodeOutputs addScatterLayer(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ScatterMode mode, int32_t axis)
{
    checkNotInvalidType(inputs.at(0), {"UINT8"}, node, nodeIdx);
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& updates = convertToTensor(inputs.at(2), ctx);

    // Validate input dimensions
    if (mode == nvinfer1::ScatterMode::kELEMENT)
    {
        auto const dataDims = data.getDimensions();
        auto const indicesDims = indices.getDimensions();
        auto const updatesDims = updates.getDimensions();

        // Ranks must all be the same
        ONNXTRT_CHECK_NODE(dataDims.nbDims == indicesDims.nbDims && dataDims.nbDims == updatesDims.nbDims,
            "Input dimensions to ScatterElements must have the same rank! data rank =  "
                << dataDims.nbDims << ", indices rank = " << indicesDims.nbDims
                << ", updates rank = " << updatesDims.nbDims << ".",
            node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);

        // Corresponding dimensions of indices and updates must be <= data
        for (int32_t i = 0; i < dataDims.nbDims; ++i)
        {
            if (indicesDims.d[i] != -1 && dataDims.d[i] != -1)
            {
                ONNXTRT_CHECK_NODE(indicesDims.d[i] <= dataDims.d[i],
                    "Indices dimensions must be less than data dimensions! indices dimension = "
                        << indicesDims.d[i] << ", data dimension = " << dataDims.d[i] << " on index " << i << ".",
                    node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            }
            if (updatesDims.d[i] != -1 && dataDims.d[i] != -1)
            {
                ONNXTRT_CHECK_NODE(updatesDims.d[i] <= dataDims.d[i],
                    "Updates dimensions must be less than data dimensions! updates dimension = "
                        << updatesDims.d[i] << ", data dimension = " << dataDims.d[i] << " on index " << i << ".",
                    node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
            }
        }
    }

    // Only cast to INT32 for weakly-typed networks (strongly-typed supports INT64 indices)
    nvinfer1::ITensor* indicesForScatter = &indices;
    if (!ctx->isStronglyTyped() && indices.getType() == nvinfer1::DataType::kINT64)
    {
        auto* cast = N_CHECK(ctx->network()->addCast(indices, nvinfer1::DataType::kINT32));
        indicesForScatter = N_CHECK(cast->getOutput(0));
    }

    auto* layer = N_CHECK(ctx->network()->addScatter(data, *indicesForScatter, updates, mode));

    layer->setAxis(axis);
    ctx->registerLayer(layer, node);
    auto output = N_CHECK(layer->getOutput(0));
    return {{output}};
}

//! Helper function to calculate mod(A, B)
nvinfer1::IElementWiseLayer* modWithIntegerInputs(
    ImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, bool fmod)
{
    using eOp = nvinfer1::ElementWiseOperation;
    auto divOp = fmod ? eOp::kDIV : eOp::kFLOOR_DIV;

    // input0 - (input1 * divOp(input0, input1))
    auto rhs = getElementWiseResult(ctx, *input1, *getElementWiseResult(ctx, *input0, *input1, divOp), eOp::kPROD);
    return N_CHECK(ctx->network()->addElementWise(*input0, *rhs, eOp::kSUB));
}

nvinfer1::IElementWiseLayer* modWithFPInputs(ImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1,
    nvinfer1::ITensor* divResult, bool sameSign)
{
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    // divResult need to be round towards 0
    // When inputs have the same sign, round down (input0 / input1), else round up.
    auto roundOp = sameSign ? uOp::kFLOOR : uOp::kCEIL;

    // input0 - (input1 * round_towards_0(input0/ input1))
    auto rhs = getElementWiseResult(ctx, *input1, *getUnaryResult(ctx, *divResult, roundOp), eOp::kPROD);
    return N_CHECK(ctx->network()->addElementWise(*input0, *rhs, eOp::kSUB));
}

int32_t getNbDims(nvinfer1::ITensor const* tensor)
{
    return tensor->getDimensions().nbDims;
}

std::string truncateString(std::string const& s, int64_t limit)
{
    if (static_cast<int64_t>(s.size()) <= limit)
    {
        return s;
    }
    return s.substr(0, limit / 2U) + " ... " + s.substr(s.size() - limit / 2U);
}

void processMetadata(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ILayer* layer)
{
    // Create a docstring that that holds node metadata and assign it to the corresponding TRT layer.
    // The format of the string is as follows:
    // [ONNX Layer: <name> | property1 | property2 | property3 ...]

    std::ostringstream metadata;
    metadata << "[ONNX Layer: " << getNodeName(node);

    // Generate local function stack string.
    for (auto it = ctx->localFunctionStack().crbegin(); it < ctx->localFunctionStack().crend(); ++it)
    {
        metadata << " | " << it->nodeName << " (" << it->functionName << ")";
    }

    metadata << "]";

    // Truncate very long metadata since TRT API has a limit.
    constexpr int64_t kMETADATA_LIMIT{4000};
    layer->setMetadata(truncateString(metadata.str(), kMETADATA_LIMIT).c_str());
}

void processMetadata(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::IAttention* attention)
{
    // Create a docstring that holds node metadata and assign it to the corresponding IAttention.
    // The format of the string is as follows:
    // [ONNX Layer: <name> | property1 | property2 | property3 ...]

    std::ostringstream metadata;
    metadata << "[ONNX Layer: " << getNodeName(node);

    // Generate local function stack string.
    for (auto it = ctx->localFunctionStack().crbegin(); it < ctx->localFunctionStack().crend(); ++it)
    {
        metadata << " | " << it->nodeName << " (" << it->functionName << ")";
    }

    metadata << "]";

    // Truncate very long metadata since TRT API has a limit.
    constexpr int64_t kMETADATA_LIMIT{4000};
    attention->setMetadata(truncateString(metadata.str(), kMETADATA_LIMIT).c_str());
}

nvinfer1::ITensor* generateWindow(ImporterContext* ctx, nvinfer1::ITensor* N)
{
    auto shapeOfN = ShapeTensor(*N, 0);
    nvinfer1::IFillLayer* layer = N_CHECK(addFill(ctx, convertTo1D(ctx, shapeOfN), nvinfer1::FillOperation::kLINSPACE));
    layer->setAlpha(0.0F);
    layer->setBeta(1.0F);
    auto* fillOutput = N_CHECK(layer->getOutput(0));
    return fillOutput;
}

nvinfer1::ITensor* windowHelper(ImporterContext* ctx, float numerator, nvinfer1::ITensor* n, nvinfer1::ITensor* N,
    nvinfer1::UnaryOperation op, int32_t periodic)
{
    auto* numeratorTensor
        = N_CHECK(addConstantScalar(ctx, numerator, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1)->getOutput(0));
    auto numeratorLayer
        = N_CHECK(ctx->network()->addElementWise(*numeratorTensor, *n, nvinfer1::ElementWiseOperation::kPROD));
    auto numeratorOutput = N_CHECK(numeratorLayer->getOutput(0));

    // If periodic is 0, subtract 1 from the denominator (N)
    if (periodic == 0)
    {
        auto* one = N_CHECK(addConstantScalar(ctx, 1, ::ONNX_NAMESPACE::TensorProto_DataType_INT32)->getOutput(0));
        one = castHelper(ctx, one, N->getType());
        auto minusOne = N_CHECK(ctx->network()->addElementWise(*N, *one, nvinfer1::ElementWiseOperation::kSUB));
        N = N_CHECK(minusOne->getOutput(0));
    }

    auto NFloat = N_CHECK(castHelper(ctx, N, nvinfer1::DataType::kFLOAT));
    broadcastTensors(ctx, n, NFloat);
    auto divLayer
        = N_CHECK(ctx->network()->addElementWise(*numeratorOutput, *NFloat, nvinfer1::ElementWiseOperation::kDIV));
    auto divOutput = N_CHECK(divLayer->getOutput(0));

    auto trigLayer = N_CHECK(ctx->network()->addUnary(*divOutput, op));
    auto trigOutput = N_CHECK(trigLayer->getOutput(0));

    return N_CHECK(trigOutput);
}

} // namespace onnx2trt
