/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx2trt_utils.hpp"
#include "OnnxAttrs.hpp"
#include <set>

namespace onnx2trt
{

void PluginDeleter::operator()(nvinfer1::IPluginV2* t)
{
    t->destroy();
}

Status notInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes)
{
    bool invalid = std::any_of(invalidTypes.begin(), invalidTypes.end(),
        [&](std::string invalidType) { return input.getType() == invalidType; });
    if (invalid)
    {
        return MAKE_ERROR("Found invalid input type of " + input.getType(), ErrorCode::kUNSUPPORTED_NODE);
    }
    return Status::success();
}

NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha, float* beta)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "BOOL", "UINT8"}));
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::IActivationLayer* layer = ctx->network()->addActivation(input, op);
    if (alpha)
    {
        layer->setAlpha(*alpha);
    }
    if (beta)
    {
        layer->setBeta(*beta);
    }
    ctx->registerLayer(layer, getNodeName(node));
    return {{layer->getOutput(0)}};
}

nvinfer1::ITensor* addClip(IImporterContext* ctx, nvinfer1::ITensor* input, float clip)
{
    if (clip >= 0.f)
    {
        nvinfer1::IActivationLayer* layer = ctx->network()->addActivation(*input, nvinfer1::ActivationType::kCLIP);
        layer->setAlpha(-clip);
        layer->setBeta(clip);
        return layer->getOutput(0);
    }
    return input;
};

NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op)
{
    CHECK(notInvalidType(inputs.at(0), {"INT32", "UINT8"}));
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    // Get attributes.
    OnnxAttrs attrs(node, ctx);
    int keepdims = attrs.get("keepdims", 1);
    int axis = attrs.get("axis", 0);
    int selectLastIndex = attrs.get<int>("select_last_index", 0);
    ASSERT((!selectLastIndex || (selectLastIndex && ctx->getOpsetVersion() >= 12))
            && "Per-opset 12 ONNX does not support the select_last_index attribute.",
        ErrorCode::kUNSUPPORTED_NODE);

    // Insert a TopK layer with k set to 1.
    int nbDims = tensor.getDimensions().nbDims;
    CHECK(convertAxis(axis, nbDims));
    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer;

    // New attribute added to Opset-12
    // Whether to select the last index or the first index if the {name} appears in multiple indices, default is False
    // (first index).
    if (selectLastIndex)
    {
        // Need to flip the data input along the given axis using the Slice operator
        const auto dims = shapeOf(tensor);
        ShapeTensor starts = shapeVector(-1);
        ShapeTensor ends = shapeVector(static_cast<int64_t>(INT_MIN));
        ShapeTensor axes = shapeVector(axis);
        ShapeTensor steps = shapeVector(-1);

        if (axes.size() < dims.size())
        {
            // axes specify a subset of the dimensions, or out of order.
            // Convert starts/ends/steps to complete in-order form.
            const ShapeTensor subscripts{axesToInterlaceSubscripts(axes, dims.size())};
            starts = interlace(ctx, similar(ctx, dims, 0), starts, subscripts);
            ends = interlace(ctx, dims, ends, subscripts);
            steps = interlace(ctx, similar(ctx, dims, 1), steps, subscripts);
        }
        decodeOnnxStartsAndEnds(ctx, dims, steps, starts, ends);
        // TensorRT uses sizes of the output dimensions instead of ends.
        const ShapeTensor sizes = computeSliceSizes(ctx, starts, ends, steps, dims);

        nvinfer1::ISliceLayer* slice = addSlice(ctx, tensor, starts, sizes, steps);
        nvinfer1::ITensor& flippedTensor = *slice->getOutput(0);
        layer = ctx->network()->addTopK(flippedTensor, op, 1, axisMask);
    }
    else
    {
        layer = ctx->network()->addTopK(tensor, op, 1, axisMask);
    }

    ctx->registerLayer(layer, getNodeName(node));
    ASSERT(layer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);
    // We don't care about the TopK values, just the indices.
    nvinfer1::ITensor* indices = layer->getOutput(1);
    indices->setType(nvinfer1::DataType::kINT32);

    // If selectLastIndex is true, the TopK operation was performed on reversed data on the provided axis.
    // Convert reversed indices back to forward indices by calculating the following:
    // indices = shape(tensor)[axis] - indices - 1
    if (selectLastIndex)
    {
        // Use shapeTensor semantics to support dynamic shapes
        auto const dims = shapeOf(tensor);
        auto const indicesDims = shapeOf(*indices);
        auto const axisTensor = shapeVector(axis);
        auto const dimOnAxis = gather(ctx, dims, axisTensor);

        // Create constant of shape indicesDims with values tensor.shape[axis]
        auto const tensorDimOnAxis = constantOfShape(ctx, node, &dimOnAxis.tensor(ctx), &indicesDims.tensor(ctx));

        // Create constant of shape indicesDims with values of 1
        auto const ones = constantOfShape(ctx, node, &shapeVector(1).tensor(ctx), &indicesDims.tensor(ctx));

        std::vector<TensorOrWeights> newInputs{tensorDimOnAxis, indices, ones};
        indices = &elementwiseHelper(ctx, node, newInputs, nvinfer1::ElementWiseOperation::kSUB).value().at(0).tensor();
    }
    if (keepdims)
    {
        // The default behavior of the TopK layer is to keepdims.
        return {{indices}};
    }
    else
    {
        // Otherwise, we need to squeeze the axis dimension
        std::vector<int> axes{axis};
        indices = squeezeTensor(ctx, node, *indices, axes);
        return {{indices}};
    }
}

Status broadcastTensor(IImporterContext* ctx, nvinfer1::ITensor*& t, const int nbDims)
{
    ASSERT(ctx->getOpsetVersion() >= 7 && "Pre-opset 7 broadcasting is unsupported in this version of the ONNX parser",
        ErrorCode::kUNSUPPORTED_NODE);
    const auto inputDims = shapeOf(*t);
    const int nbInputDims = inputDims.size();
    ASSERT((nbInputDims <= nbDims) && "Cannot broadcast a higher rank tensor to a lower rank tensor.",
        ErrorCode::kUNSUPPORTED_NODE);
    if (nbInputDims < nbDims)
    {
        nvinfer1::IShuffleLayer* reshape
            = addShuffle(ctx, *t, concat(ctx, fillShapeVector(ctx, 1, shapeVector(nbDims - nbInputDims)), shapeOf(*t)));
        t = reshape->getOutput(0);
    }
    return Status::success();
}

Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    const int t1Dims = t1->getDimensions().nbDims;
    const int t2Dims = t2->getDimensions().nbDims;

    if (t1Dims == t2Dims)
    {
        return Status::success();
    }

    if (t1Dims > t2Dims)
    {
        return broadcastTensor(ctx, t2, t1Dims);
    }
    return broadcastTensor(ctx, t1, t2Dims);
}

Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3)
{
    const int maxDims = std::max({t1->getDimensions().nbDims, t2->getDimensions().nbDims, t3->getDimensions().nbDims});
    CHECK(broadcastTensor(ctx, t1, maxDims));
    CHECK(broadcastTensor(ctx, t2, maxDims));
    CHECK(broadcastTensor(ctx, t3, maxDims));
    return Status::success();
}

Status isBroadcastValid(IImporterContext* ctx, const nvinfer1::Dims& firstShape, const nvinfer1::Dims& secondShape)
{
    const auto firstRank = firstShape.nbDims;
    const auto secondRank = secondShape.nbDims;
    if (firstRank != secondRank)
    {
        return MAKE_ERROR("Cannot broadcast shapes that have different ranks!", ErrorCode::kUNSUPPORTED_NODE);
    }
    for (int32_t i = 0; i < firstRank; i++)
    {
        const auto firstDim = firstShape.d[i];
        const auto secondDim = secondShape.d[i];
        if (firstDim != secondDim && firstDim != 1 && secondDim != 1)
        {
            if (firstDim == -1 || secondDim == -1)
            {
                LOG_WARNING(
                    "Found dynamic dimensions when checking for broadcast compatibility! TensorRT may fail at "
                    "build-time if the final shapes do not conform to broadcasting rules.");
            }
            else
            {
                MAKE_ERROR("Found incompatible shapes for tensors that need to be broadcastable!",
                    ErrorCode::kUNSUPPORTED_NODE);
            }
        }
    }
    return Status::success();
}

// Helper functions for calculateBias:
int32_t getBias(const std::vector<int32_t>& dimension_count, const std::vector<int32_t>& pitches, int32_t axis)
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
    const nvinfer1::Dims& daDims, const nvinfer1::Dims& idxDims, const std::vector<int32_t>& pitches, int32_t axis)
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

std::vector<int32_t> calculatePitches(const nvinfer1::Dims& inputDims)
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

nvinfer1::ITensor* castHelper(IImporterContext* ctx, nvinfer1::ITensor* input, nvinfer1::DataType dtype)
{
    nvinfer1::IIdentityLayer* cast = ctx->network()->addIdentity(*input);
    cast->setOutputType(0, dtype);
    return cast->getOutput(0);
}

nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor* constant, nvinfer1::ITensor* shape)
{
    int rank = shape->getDimensions().d[0];

    std::vector<int> starts(rank);
    std::fill(starts.begin(), starts.end(), 0);

    nvinfer1::Dims strides{rank};
    std::fill(strides.d, strides.d + strides.nbDims, 0);

    // Slice will not work if constant does not have the same rank as start/size/strides.
    nvinfer1::Dims unsqueezeDims{rank};
    std::fill(unsqueezeDims.d, unsqueezeDims.d + unsqueezeDims.nbDims, 1);
    nvinfer1::IShuffleLayer* unsqueeze = ctx->network()->addShuffle(*constant);
    unsqueeze->setReshapeDimensions(unsqueezeDims);
    unsqueeze->setZeroIsPlaceholder(false);
    constant = unsqueeze->getOutput(0);

    nvinfer1::ISliceLayer* broadcast = ctx->network()->addSlice(*constant, nvinfer1::Dims{}, nvinfer1::Dims{}, strides);
    broadcast->setInput(1,
        *addConstant(ctx, starts, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, rank})->getOutput(0));
    broadcast->setInput(2, *shape);
    ctx->registerLayer(broadcast, getNodeName(node));
    return broadcast->getOutput(0);
}

Status convertAxis(int& axis, int nbDims)
{
    // Support negative indexing
    if (axis < 0)
    {
        axis += nbDims;
    }
    // Support nbDims as a valid axis for QuantDequantLinearHelper
    ASSERT((axis >= 0 && axis <= nbDims) && "Axis must be in the range [0, nbDims].", ErrorCode::kUNSUPPORTED_NODE);
    return Status::success();
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
    case ::ONNX_NAMESPACE::TensorProto::BOOL: *trt_dtype = nvinfer1::DataType::kBOOL; break;
    case ::ONNX_NAMESPACE::TensorProto::INT32: *trt_dtype = nvinfer1::DataType::kINT32; break;
    // See convertOnnxWeights for sanity check if all values can be safetly downcasted to INT32
    case ::ONNX_NAMESPACE::TensorProto::INT64: *trt_dtype = nvinfer1::DataType::kINT32; break;
    default:
        std::cerr << "Unsupported ONNX data type: " << getDtypeName(onnx_dtype) << " (" << std::to_string(onnx_dtype)
                  << ")" << std::endl;
        return false;
    }
    return true;
}

int32_t* convertINT64(const int64_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx)
{
    static bool logged = false;
    if (!logged)
    {
        LOG_WARNING(
            "Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. "
            "Attempting to cast down to INT32.");
        logged = true;
    }

    const size_t nbWeights = volume(shape);
    int32_t* int32Weights{
        reinterpret_cast<int32_t*>(ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, shape).values)};

    bool outOfBounds{false};
    for (size_t i = 0; i < nbWeights; i++)
    {
        if (weightValues[i] > static_cast<int64_t>(INT32_MAX) || weightValues[i] < static_cast<int64_t>(INT32_MIN))
        {
            int32Weights[i] = static_cast<int32_t>(
                std::max(std::min(weightValues[i], static_cast<int64_t>(INT32_MAX)), static_cast<int64_t>(INT32_MIN)));
            LOG_VERBOSE("Weight at index " << i << ": " << weightValues[i]
                                           << " is out of range. Clamping to: " << int32Weights[i]);
            outOfBounds = true;
        }
        else
        {
            int32Weights[i] = static_cast<int32_t>(weightValues[i]);
        }
    }
    if (outOfBounds)
    {
        LOG_WARNING("One or more weights outside the range of INT32 was clamped");
    }

    return int32Weights;
}

bool convertOnnxPadding(IImporterContext* ctx, int32_t nbInputDims, const std::vector<int32_t>& onnxPadding,
    nvinfer1::ITensor*& startTensor, nvinfer1::ITensor*& totalPaddingTensor)
{
    std::vector<int32_t> start;
    std::vector<int32_t> totalPadding;
    if (onnxPadding.size() % 2U != 0)
    {
        return false;
    }
    const auto diff = nbInputDims - static_cast<int32_t>(onnxPadding.size() / 2U);
    if (diff < 0)
    {
        return false;
    }
    start.resize(nbInputDims, 0);
    totalPadding.resize(nbInputDims, 0);

    for (int32_t i = diff; i < nbInputDims; i++)
    {
        const auto idx = i - diff;
        const auto pre = onnxPadding[idx];
        const auto post = onnxPadding[onnxPadding.size() / 2U + idx];
        if (pre < 0 || post < 0)
        {
            return false;
        }

        start[i] = -pre;
        totalPadding[i] = pre + post;
    }

    startTensor
        = addConstant(ctx, start, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, {nbInputDims}})->getOutput(0);
    totalPaddingTensor
        = addConstant(ctx, totalPadding, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, {nbInputDims}})
              ->getOutput(0);
    return startTensor && totalPaddingTensor;
}

bool shiftIsAllZeros(const ShapedWeights& shiftInt8)
{
    // Check if all of the values in the shift tensor are zeros
    const auto* v = static_cast<const int8_t*>(shiftInt8.values);
    auto allZeros = std::all_of(v, v + shiftInt8.count(), [](int8_t x) { return x == 0; });
    return allZeros;
}

onnx2trt::ShapedWeights createZeroShifts(const onnx2trt::ShapedWeights& shiftInt8, int32_t type, IImporterContext* ctx)
{
    const auto* v = static_cast<const int8_t*>(shiftInt8.values);
    if (!std::all_of(v, v + shiftInt8.count(), [](int8_t x) { return x == 0; }))
    {
        LOG_WARNING("TensorRT currenly supports only zero shifts values for QuatizeLinear/DequantizeLinear ops");
    }
    auto shift = ctx->createTempWeights(type, shiftInt8.shape);
    float* sh = static_cast<float*>(shift.values);
    for (int i = 0, n = shift.count(); i < n; i++)
    {
        sh[i] = 0.0f;
    }
    return shift;
}

nvinfer1::ITensor* createZeroTensor(IImporterContext* ctx, nvinfer1::ITensor* data)
{
    nvinfer1::ITensor* zero
        = addConstant(ctx, std::vector<float>{0.f}, ::ONNX_NAMESPACE::TensorProto::FLOAT, {0, {1}})->getOutput(0);
    zero = castHelper(ctx, zero, data->getType());
    broadcastTensors(ctx, zero, data);
    zero = ctx->network()->addElementWise(*data, *zero, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    return zero;
}

template <typename DataType>
DataType* convertINT32Data(const int32_t* weightValues, nvinfer1::Dims shape, int32_t onnxdtype, IImporterContext* ctx)
{
    const size_t nbWeights = volume(shape);
    DataType* newWeights{reinterpret_cast<DataType*>(ctx->createTempWeights(onnxdtype, shape).values)};

    for (size_t i = 0; i < nbWeights; i++)
    {
        newWeights[i] = static_cast<DataType>(weightValues[i]);
    }
    return newWeights;
}

int32_t* convertUINT8(const uint8_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx)
{
    const size_t nbWeights = volume(shape);
    int32_t* int32Weights{
        reinterpret_cast<int32_t*>(ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, shape).values)};

    for (size_t i = 0; i < nbWeights; i++)
    {
        int32Weights[i] = static_cast<int32_t>(weightValues[i]);
    }
    return int32Weights;
}

float* convertDouble(const double* weightValues, nvinfer1::Dims shape, IImporterContext* ctx)
{
    static bool logged = false;
    if (!logged)
    {
        LOG_WARNING(
            "Your ONNX model has been generated with double-typed weights, while TensorRT does not natively support "
            "double. "
            "Attempting to cast down to float.");
        logged = true;
    }
    const size_t nbWeights = volume(shape);
    float* floatWeights{
        reinterpret_cast<float*>(ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, shape).values)};

    bool outOfBounds{false};
    const double floatMax = static_cast<double>(std::numeric_limits<float>::max());
    const double floatMin = static_cast<double>(std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < nbWeights; i++)
    {
        if (weightValues[i] > floatMax || weightValues[i] < floatMin)
        {
            floatWeights[i] = static_cast<float>(std::max(std::min(weightValues[i], floatMax), floatMin));
            LOG_WARNING("Weight at index " << i << ": " << weightValues[i]
                                           << " is out of range. Clamping to: " << floatWeights[i]);
            outOfBounds = true;
        }
        else
        {
            floatWeights[i] = static_cast<float>(weightValues[i]);
        }
    }
    if (outOfBounds)
    {
        LOG_WARNING("One or more weights outside the range of FLOAT was clamped");
    }

    return floatWeights;
}

bool convertOnnxWeights(
    const ::ONNX_NAMESPACE::TensorProto& onnxTensor, onnx2trt::ShapedWeights* weights, IImporterContext* ctx)
{
    void* dataPtr{nullptr};
    size_t nbytes{0};
    auto onnxDtype = onnxTensor.data_type();

    nvinfer1::Dims shape{};
    shape.nbDims = onnxTensor.dims().size();
    std::copy_n(onnxTensor.dims().begin(), shape.nbDims, shape.d);

    // ONNX weight values can be stored in either the TensorProto itself, or in an external file in the case
    // of large models. Check for this here.
    auto dataLocation = onnxTensor.data_location();
    // External Data
    if (dataLocation == 1)
    {
        std::string location{""};
        int64_t offset{0};
        int64_t length{0};

        // onnxTensor.external_data() is a String : String map that holds metadata about how to read from an external
        // file
        for (auto onnxMapEntry : onnxTensor.external_data())
        {
            auto keyName = onnxMapEntry.key();
            if (keyName == "location")
            {
                location = onnxMapEntry.value();
            }
            else if (keyName == "offset")
            {
                offset = std::atoll(onnxMapEntry.value().c_str());
            }
            else if (keyName == "length")
            {
                length = std::atoll(onnxMapEntry.value().c_str());
            }
            // Not used at the moment
            else if (keyName == "checksum")
            {
                continue;
            }
            else
            {
                LOG_ERROR("Key value of: " << keyName << " was not expected!");
                return false;
            }
        }

        // Buffer to hold the data read from the file
        std::vector<char> dataBuf;
        // Will update dataBuf and nbytes by reference.
        if (!parseExternalWeights(ctx, location, ctx->getOnnxFileLocation(), offset, length, dataBuf, nbytes))
        {
            return false;
        }

        // For weights parsed from external files, createTempWeights is necessary to keep them in scope
        ShapedWeights externalWeights;

        // Cast non-native TRT types to their corresponding proxy types
        if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT64)
        {
            dataPtr = dataBuf.data();
            dataPtr = convertINT64(reinterpret_cast<const int64_t*>(dataPtr), shape, ctx);
            nbytes = nbytes / (sizeof(int64_t) / sizeof(int32_t));
            onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
            externalWeights = ctx->createTempWeights(onnxDtype, shape);
            std::memcpy(externalWeights.values, dataPtr, nbytes);
        }
        else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::UINT8)
        {
            dataPtr = dataBuf.data();
            dataPtr = convertUINT8(reinterpret_cast<const uint8_t*>(dataPtr), shape, ctx);
            nbytes = nbytes * (sizeof(int32_t) / sizeof(uint8_t));
            onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
            externalWeights = ctx->createTempWeights(onnxDtype, shape);
            std::memcpy(externalWeights.values, dataPtr, nbytes);
        }
        else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::DOUBLE)
        {
            dataPtr = dataBuf.data();
            dataPtr = convertDouble(reinterpret_cast<const double*>(dataPtr), shape, ctx);
            nbytes = nbytes / (sizeof(double) / sizeof(float));
            onnxDtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
            externalWeights = ctx->createTempWeights(onnxDtype, shape);
            std::memcpy(externalWeights.values, dataPtr, nbytes);
        }
        // Copy weight values directly to externalWeights
        else
        {
            externalWeights = ctx->createTempWeights(onnxDtype, shape);
            std::memcpy(externalWeights.values, dataBuf.data(), nbytes);
        }

        *weights = externalWeights;
        return true;
    }

    // Weights information is within the TensorProto itself

    // Cast non-native TRT types to their corresponding proxy types
    if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertINT64(reinterpret_cast<const int64_t*>(onnxTensor.raw_data().data()), shape, ctx);
            nbytes = onnxTensor.raw_data().size() / (sizeof(int64_t) / sizeof(int32_t));
        }
        else if (onnxTensor.int64_data().size() > 0)
        {
            dataPtr = convertINT64(onnxTensor.int64_data().data(), shape, ctx);
            nbytes = onnxTensor.int64_data().size() * sizeof(int32_t);
        }
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::UINT8)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertUINT8(reinterpret_cast<const uint8_t*>(onnxTensor.raw_data().data()), shape, ctx);
            nbytes = onnxTensor.raw_data().size() * (sizeof(int32_t) / sizeof(uint8_t));
        }
        else if (onnxTensor.int32_data().size() > 0)
        {
            dataPtr = (void*) onnxTensor.int32_data().data();
            nbytes = onnxTensor.int32_data().size() * sizeof(int32_t);
        }
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::DOUBLE)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = convertDouble(reinterpret_cast<const double*>(onnxTensor.raw_data().data()), shape, ctx);
            nbytes = onnxTensor.raw_data().size() / (sizeof(double) / sizeof(float));
        }
        else if (onnxTensor.double_data().size() > 0)
        {
            dataPtr = convertDouble(onnxTensor.double_data().data(), shape, ctx);
            nbytes = onnxTensor.double_data().size() * sizeof(float);
        }
        onnxDtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    }

    // Check for supported types that can be found in the int32_data field in the TensorProto
    // https://github.com/onnx/onnx/blob/master/onnx/onnx.proto#L528
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT32 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT16
        || onnxDtype == ::ONNX_NAMESPACE::TensorProto::INT8 || onnxDtype == ::ONNX_NAMESPACE::TensorProto::BOOL)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = (void*) (onnxTensor.raw_data().data());
            nbytes = onnxTensor.raw_data().size();
        }
        else
        {
            switch (onnxDtype)
            {
            case ::ONNX_NAMESPACE::TensorProto::INT32: dataPtr = (void*) (onnxTensor.int32_data().data()); break;
            case ::ONNX_NAMESPACE::TensorProto::FLOAT16:
                dataPtr = convertINT32Data<uint16_t>(onnxTensor.int32_data().data(), shape, onnxDtype, ctx);
                break;
            case ::ONNX_NAMESPACE::TensorProto::INT8:
                dataPtr = convertINT32Data<int8_t>(onnxTensor.int32_data().data(), shape, onnxDtype, ctx);
                break;
            case ::ONNX_NAMESPACE::TensorProto::BOOL:
                dataPtr = convertINT32Data<uint8_t>(onnxTensor.int32_data().data(), shape, onnxDtype, ctx);
                break;
            default:
                LOG_ERROR("Found unsupported datatype (" << onnxDtype
                                                         << ") when importing initializer: " << onnxTensor.name());
                break;
            }
            nbytes = onnxTensor.int32_data().size() * getDtypeSize(onnxDtype);
        }
    }
    else if (onnxDtype == ::ONNX_NAMESPACE::TensorProto::FLOAT)
    {
        if (onnxTensor.raw_data().size() > 0)
        {
            dataPtr = (void*) (onnxTensor.raw_data().data());
            nbytes = onnxTensor.raw_data().size();
        }
        else
        {
            dataPtr = (void*) (onnxTensor.float_data().data());
            nbytes = onnxTensor.float_data().size() * sizeof(float);
        }
    }
    else
    {
        LOG_ERROR("Found unsupported datatype (" << onnxDtype << ") when importing initializer: " << onnxTensor.name());
        return false;
    }
    onnx2trt::ShapedWeights trt_weights(onnxDtype, dataPtr, shape);
    // Sanity check that weights were converted properly
    if (trt_weights.size_bytes() != nbytes)
    {
        LOG_ERROR("Size mismatch when importing initializer: " << onnxTensor.name() << ". Expected size: " << nbytes
                                                               << " , actual size: " << trt_weights.size_bytes());
        return false;
    }
    *weights = trt_weights;
    return true;
}

nvinfer1::ITensor* convertToScalar(IImporterContext* ctx, nvinfer1::ITensor* inpTensor)
{
    if (inpTensor->getDimensions().nbDims == 0)
    {
        return inpTensor;
    }
    const auto tensorVolume = volume(inpTensor->getDimensions());
    if (tensorVolume != 1)
    {
        LOG_VERBOSE("Cannot convert tensor to scalar. Note: Tensor dimensions were: "
            << inpTensor->getDimensions() << ", with volume: " << tensorVolume);
        return nullptr;
    }
    nvinfer1::IShuffleLayer* reshape = ctx->network()->addShuffle(*inpTensor);
    reshape->setReshapeDimensions(nvinfer1::Dims{0});
    // Do not need to call setZeroIsPlaceholder, since reshape dimensions are empty.
    return reshape->getOutput(0);
}

nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx)
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
        return *(existingConstantLayer->getOutput(0));
    }
    auto* constantLayer = ctx->network()->addConstant(weights.shape, weights);
    // Register layer and constant name (if set) into RefitMap:
    if (weights.getName())
    {
        ctx->registerLayer(constantLayer, weights.getName());
        ctx->network()->setWeightsName(weights, weights.getName());
    }
    return *(constantLayer->getOutput(0));
}

nvinfer1::ITensor* convertToScalar(TensorOrWeights& input, IImporterContext* ctx)
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

    return ctx->network()->addConstant(nvinfer1::Dims{0, {0}}, weights)->getOutput(0);
}

int divCeil(int n, int d)
{
    return (n - 1) / d + 1;
}

bool elementwiseCheck(const std::vector<TensorOrWeights>& inputs, const nvinfer1::ElementWiseOperation op)
{
    switch (op)
    {
    // These operations only support boolean inputs
    case nvinfer1::ElementWiseOperation::kAND:
    case nvinfer1::ElementWiseOperation::kOR:
    case nvinfer1::ElementWiseOperation::kXOR:
        if (!std::all_of(inputs.begin(), inputs.end(), [](const TensorOrWeights& input) { return input.isBool(); }))
        {
            return false;
        }
        break;
    // These operations do not support boolean types
    case nvinfer1::ElementWiseOperation::kDIV:
    case nvinfer1::ElementWiseOperation::kFLOOR_DIV:
    case nvinfer1::ElementWiseOperation::kGREATER:
    case nvinfer1::ElementWiseOperation::kLESS:
    case nvinfer1::ElementWiseOperation::kMAX:
    case nvinfer1::ElementWiseOperation::kMIN:
    case nvinfer1::ElementWiseOperation::kPROD:
    case nvinfer1::ElementWiseOperation::kSUB:
    case nvinfer1::ElementWiseOperation::kSUM:
        if (std::any_of(inputs.begin(), inputs.end(), [](const TensorOrWeights& input) { return input.isBool(); }))
        {
            return false;
        }
        break;
    // Pow does not support bool or INT32 types
    case nvinfer1::ElementWiseOperation::kPOW:
        if (std::any_of(inputs.begin(), inputs.end(),
                [](const TensorOrWeights& input) { return input.isBool() || input.isInt32(); }))
        {
            return false;
        }
        break;
    // Equal supports all types.
    case nvinfer1::ElementWiseOperation::kEQUAL:
        break;
    }
    return true;
}

NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    const std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op)
{
    ASSERT((!inputs.empty()) && "Inputs vector is empty.", ErrorCode::kINVALID_NODE);

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
        ASSERT(tensor_ptr->getDimensions().nbDims == maxNbDims && "Failed to broadcast tensors elementwise!",
            ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }
    ASSERT(elementwiseCheck(inputs, binary_op) && "Elementwise layer does not support the given inputs and operator.",
        ErrorCode::kUNSUPPORTED_NODE);

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
        ASSERT((tensor->getDimensions().nbDims == combined->getDimensions().nbDims)
                && "The number of dimensions should remain the same adding inputs.",
            ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = ctx->network()->addElementWise(*combined, *tensor, binary_op);
        ctx->registerLayer(layer, getNodeName(node));
        ASSERT(layer && "Failed to register layer.", ErrorCode::kUNSUPPORTED_NODE);
        combined = layer->getOutput(0);
    }
    return {{combined}};
}

nvinfer1::ITensor* flattenTensor(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ITensor& tensor, int axis, bool regLayer)
{
    const auto dims = shapeOf(tensor);
    const auto d0 = product(ctx, dims, 0, axis, 1);
    const auto d1 = product(ctx, dims, axis, dims.size(), 1);

    // ShuffleLayer here interprets dim extent 0 as empty dim to support empty tensor
    nvinfer1::IShuffleLayer* flattenLayer = addShuffle(ctx, tensor, concat(ctx, d0, d1), /*zeroIsPlaceholder=*/false);
    if (regLayer)
    {
        ctx->registerLayer(flattenLayer, getNodeName(node));
    }
    return flattenLayer->getOutput(0);
}

nvinfer1::ITensor* gatherDimension(IImporterContext* ctx, nvinfer1::ITensor* shapeTensor, int dim, nvinfer1::Dims shape)
{
    auto& axisValue = *addConstantScalar(ctx, dim, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, shape)->getOutput(0);
    return ctx->network()->addGather(*shapeTensor, axisValue, 0)->getOutput(0);
}

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims inputShape, nvinfer1::Dims outputShape, nvinfer1::Dims kernelSize,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& begPadding,
    nvinfer1::Dims& endPadding, nvinfer1::Dims& outputPadding, nvinfer1::PaddingMode paddingMode)
{
    nvinfer1::Dims totalPadding{nbSpatialDims, {}};
    // Pre and post padding calculated as per https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
    // Note that output shape is inconsistent in the spec - can either be in full dimensions form (i.e. NCHW) or just spatial
    // dimensions form (i.e. HW). Calculate potential offset here.
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
    case nvinfer1::ActivationType::kRELU: return 0.f;
    case nvinfer1::ActivationType::kSIGMOID: return 0.f;
    case nvinfer1::ActivationType::kTANH: return 0.f;
    case nvinfer1::ActivationType::kLEAKY_RELU: return 0.01f;
    case nvinfer1::ActivationType::kELU: return 1.0f;
    case nvinfer1::ActivationType::kSELU: return 1.67326319217681884765625f;
    case nvinfer1::ActivationType::kSOFTSIGN: return 0.f;
    case nvinfer1::ActivationType::kSOFTPLUS: return 0.f;
    case nvinfer1::ActivationType::kCLIP: return 0.f;
    case nvinfer1::ActivationType::kHARD_SIGMOID: return 0.2f;
    case nvinfer1::ActivationType::kSCALED_TANH: return 1.0f;
    case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 1.0f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

float getActivationDefaultBeta(nvinfer1::ActivationType type)
{
    switch (type)
    {
    case nvinfer1::ActivationType::kRELU: return 0.f;
    case nvinfer1::ActivationType::kSIGMOID: return 0.f;
    case nvinfer1::ActivationType::kTANH: return 0.f;
    case nvinfer1::ActivationType::kLEAKY_RELU: return 0.f;
    case nvinfer1::ActivationType::kELU: return 0.f;
    case nvinfer1::ActivationType::kSELU: return 1.05070102214813232421875f;
    case nvinfer1::ActivationType::kSOFTSIGN: return 0.f;
    case nvinfer1::ActivationType::kSOFTPLUS: return 0.f;
    case nvinfer1::ActivationType::kCLIP: return 0.f;
    case nvinfer1::ActivationType::kHARD_SIGMOID: return 0.5f;
    case nvinfer1::ActivationType::kSCALED_TANH: return 1.0f;
    case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return 0.f;
    }
    throw std::runtime_error{"Unrecognized activation type"};
}

nvinfer1::ITensor* getAxisLength(IImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape)
{
    // fast path for static dims
    auto dims = inpTensor->getDimensions();
    int d = dims.d[axis];
    if (d >= 0)
    {
        return addConstantScalar(ctx, d, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, shape)->getOutput(0);
    }
    else
    {
        nvinfer1::ITensor* inpShape = ctx->network()->addShape(*inpTensor)->getOutput(0);
        return gatherDimension(ctx, inpShape, axis, shape);
    }
}

int getConvOutputSize(int input_size, int filter_size, int stride, int dilation_rate, int total_padding);

const char* getDtypeName(int32_t onnxDtype)
{
    switch (onnxDtype)
    {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: return "FLOAT";
    case ::ONNX_NAMESPACE::TensorProto::UINT8: return "UINT8";
    case ::ONNX_NAMESPACE::TensorProto::INT8: return "INT8";
    case ::ONNX_NAMESPACE::TensorProto::UINT16: return "UINT16";
    case ::ONNX_NAMESPACE::TensorProto::INT16: return "INT16";
    case ::ONNX_NAMESPACE::TensorProto::INT32: return "INT32";
    case ::ONNX_NAMESPACE::TensorProto::INT64: return "INT64";
    case ::ONNX_NAMESPACE::TensorProto::STRING: return "STRING";
    case ::ONNX_NAMESPACE::TensorProto::BOOL: return "BOOL";
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return "FLOAT16";
    case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return "DOUBLE";
    case ::ONNX_NAMESPACE::TensorProto::UINT32: return "UINT32";
    case ::ONNX_NAMESPACE::TensorProto::UINT64: return "UINT64";
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX64: return "COMPLEX64";
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return "COMPLEX128";
    default: return "<UNKNOWN>";
    }
}

int getDtypeSize(int32_t onnxDtype)
{
    switch (onnxDtype)
    {
    case ::ONNX_NAMESPACE::TensorProto::FLOAT16: return 2;
    case ::ONNX_NAMESPACE::TensorProto::FLOAT: return 4;
    case ::ONNX_NAMESPACE::TensorProto::DOUBLE: return 8;
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX64: return 8;
    case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return 16;
    case ::ONNX_NAMESPACE::TensorProto::UINT8: return 1;
    case ::ONNX_NAMESPACE::TensorProto::INT8: return 1;
    case ::ONNX_NAMESPACE::TensorProto::UINT16: return 2;
    case ::ONNX_NAMESPACE::TensorProto::INT16: return 2;
    case ::ONNX_NAMESPACE::TensorProto::UINT32: return 4;
    // Booleans are stored in int32 tensors in ONNX
    case ::ONNX_NAMESPACE::TensorProto::BOOL: return 1;
    case ::ONNX_NAMESPACE::TensorProto::INT32: return 4;
    case ::ONNX_NAMESPACE::TensorProto::UINT64: return 8;
    case ::ONNX_NAMESPACE::TensorProto::INT64: return 8;
    default: return -1;
    }
}

void getKernelParams(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& onnx_node, nvinfer1::Dims* kernel_size,
    nvinfer1::Dims* strides, nvinfer1::Dims* beg_padding, nvinfer1::Dims* end_padding,
    nvinfer1::PaddingMode& paddingMode, bool& count_exclude_padding, nvinfer1::Dims* dilations,
    nvinfer1::Dims* output_padding, const bool poolingCeilMode)
{
    const int nbSpatialDims = kernel_size->nbDims;
    OnnxAttrs attrs(onnx_node, ctx);
    if (attrs.count("kernel_shape"))
    {
        auto const* onnx_kernel_size = attrs.at("kernel_shape");
        setAttr(kernel_size, onnx_kernel_size, nbSpatialDims, 1);
    }
    if (attrs.count("strides"))
    {
        auto const* onnx_strides = attrs.at("strides");
        setAttr(strides, onnx_strides, nbSpatialDims, 1);
    }
    if (dilations && attrs.count("dilations"))
    {
        auto const* onnx_dilations = attrs.at("dilations");
        setAttr(dilations, onnx_dilations, nbSpatialDims, 1);
    }
    if (attrs.count("count_include_pad"))
    {
        auto const* include_pad = attrs.at("count_include_pad");
        int val = include_pad->i();
        val == 1 ? count_exclude_padding = false : count_exclude_padding = true;
    }
    // For ConvTranspose Layer
    if (attrs.count("output_padding"))
    {
        auto const* onnxOutputPadding = attrs.at("output_padding");
        setAttr(output_padding, onnxOutputPadding, nbSpatialDims, 0);
    }

    paddingMode
        = poolingCeilMode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
    auto onnx_auto_pad = attrs.get("auto_pad", std::string("NOTSET"));
    if (onnx_auto_pad != "SAME_LOWER" && onnx_auto_pad != "SAME_UPPER")
    {
        if (attrs.count("pads"))
        {
            auto onnx_padding = attrs.get<std::vector<int>>("pads");
            int ndim = onnx_padding.size() / 2;
            for (int i = 0; i < nbSpatialDims; ++i)
            {
                if (i < ndim)
                {
                    beg_padding->d[i] = onnx_padding.at(i);
                    end_padding->d[i] = onnx_padding.at(i + ndim);
                }
                else
                {
                    beg_padding->d[i] = 0;
                    end_padding->d[i] = 0;
                }
            }
        }
        if (onnx_auto_pad != "VALID" && onnx_auto_pad != "NOTSET")
        {
            if (onnx_auto_pad == "EXPLICIT_ROUND_UP")
            {
                paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP;
            }
            else if (onnx_auto_pad == "CAFFE_ROUND_DOWN")
            {
                paddingMode = nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN;
            }
            else if (onnx_auto_pad == "CAFFE_ROUND_UP")
            {
                paddingMode = nvinfer1::PaddingMode::kCAFFE_ROUND_UP;
            }
        }
    }
    else
    {
        // If auto_pad is SAME_LOWER or SAME_UPPER, input padding should be calculated
        // "pads" attribute should not be specified
        assert((!attrs.count("pads")) && "The attribute pads should not be specified.");
        // Note: ONNX is always NCHW ordering
        if (onnx_auto_pad == "SAME_LOWER")
        {
            paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
        }
        else if (onnx_auto_pad == "SAME_UPPER")
        {
            paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
        }
        else
        {
            throw std::invalid_argument("Unexpected auto_pad value: " + onnx_auto_pad);
        }
    }
}

float getSingleValueAsFloat(void* data, bool fp16)
{
    if (fp16)
    {
        return static_cast<float>(reinterpret_cast<half_float::half const*>(data)[0]);
    }
    return static_cast<float*>(data)[0];
}

nvinfer1::ITensor* globalPoolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ITensor& tensor, nvinfer1::ReduceOperation op)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
    uint32_t reduceAxes = ((1 << dims.nbDims) - 1) & ~0b11;
    auto* layer = ctx->network()->addReduce(tensor, op, reduceAxes, /*keepDimensions=*/true);
    ctx->registerLayer(layer, getNodeName(node));
    return layer->getOutput(0);
}

nvinfer1::ITensor* greaterLessOrEqual(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* inputA, nvinfer1::ITensor* inputB, bool greater)
{
    nvinfer1::ElementWiseOperation op = greater ? nvinfer1::ElementWiseOperation::kGREATER : nvinfer1::ElementWiseOperation::kLESS;
    nvinfer1::ITensor* firstOp = &elementwiseHelper(ctx, node, {inputA, inputB}, op).value().at(0).tensor();
    nvinfer1::ITensor* equal = &elementwiseHelper(ctx, node, {inputA, inputB}, nvinfer1::ElementWiseOperation::kEQUAL).value().at(0).tensor();
    nvinfer1::ITensor* result = &elementwiseHelper(ctx, node, {firstOp, equal}, nvinfer1::ElementWiseOperation::kOR).value().at(0).tensor();
    return result;
}

nvinfer1::IPluginCreator* importPluginCreator(
    const std::string& pluginName, const std::string& pluginVersion, const std::string& pluginNamespace)
{
    nvinfer1::IPluginCreator* creator = nullptr;

#if ENABLE_STD_PLUGIN
    auto pluginRegistry = nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    if (pluginRegistry != nullptr)
    {
        creator = pluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), pluginNamespace.c_str());
    }
#endif // ENABLE_STD_PLUGIN

#if ENABLE_SAFE_PLUGIN
    auto safetyPluginRegistry = nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSAFETY);
    if (creator == nullptr && safetyPluginRegistry != nullptr)
    {
        creator = safetyPluginRegistry->getPluginCreator(
            pluginName.c_str(), pluginVersion.c_str(), pluginNamespace.c_str());
    }
#endif // ENABLE_SAFE_PLUGIN

    return creator;
}

std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(const std::string& name,
    nvinfer1::IPluginCreator* pluginCreator, const std::vector<nvinfer1::PluginField>& pluginFields)
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

bool isDynamic(const nvinfer1::Dims& shape)
{
    return std::any_of(shape.d, shape.d + shape.nbDims, [](int dim) { return dim < 0; });
}

nvinfer1::ITensor* iota(IImporterContext* ctx, ShapeTensor iotaDims, int32_t axis)
{
    std::vector<int32_t> deltaVals(iotaDims.size(), 0);
    deltaVals[axis] = 1;
    auto* iota = ctx->network()->addFill({0,{0}}, nvinfer1::FillOperation::kLINSPACE);
    auto* alpha = addConstantScalar(ctx, 0, ::ONNX_NAMESPACE::TensorProto::INT32)->getOutput(0);
    auto* delta = addConstant(ctx, deltaVals, ::ONNX_NAMESPACE::TensorProto::INT32, {1, {iotaDims.size()}})->getOutput(0);
    iota->setInput(0, iotaDims.tensor(ctx));
    iota->setInput(1, *alpha);
    iota->setInput(2, *delta);
    iota->setOutputType(0, nvinfer1::DataType::kINT32);
    return iota->getOutput(0);
}

bool isTransposeRequired(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm)
{
    int ndim = shape.nbDims;
    int prev_significant_dim = 0;
    for (int dst_i = 0; dst_i < ndim; ++dst_i)
    {
        int src_i = perm.order[dst_i];
        int dim_i = shape.d[src_i];
        if (dim_i != 1)
        {
            // We must do a transpose for dynamically shaped tensors
            if (dim_i == -1)
            {
                return true;
            }
            if (src_i < prev_significant_dim)
            {
                return true;
            }
            prev_significant_dim = src_i;
        }
    }
    return false;
}

NodeImportResult lstmLegacyImporter(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)
{
    // Input
    nvinfer1::ITensor& raw_input = convertToTensor(inputs.at(0), ctx);
    ASSERT(3 == raw_input.getDimensions().nbDims && "Input tensor must be 3 dimensional", ErrorCode::kINVALID_NODE);
    ASSERT((raw_input.getType() == nvinfer1::DataType::kFLOAT || raw_input.getType() == nvinfer1::DataType::kHALF)
            && "Only fp16 and fp32 inputs are supported",
        ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::DataType input_type = raw_input.getType();
    const int32_t max_seq_len = raw_input.getDimensions().d[0];
    const int32_t batch_size = raw_input.getDimensions().d[1];

    // Attributes
    OnnxAttrs attrs(node, ctx);
    const std::string direction_str = attrs.get<std::string>("direction", "forward");
    ASSERT((direction_str == "forward" || direction_str == "bidirectional") && "Reverse LSTM unsupported",
        ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::RNNDirection direction
        = (direction_str == "forward") ? nvinfer1::RNNDirection::kUNIDIRECTION : nvinfer1::RNNDirection::kBIDIRECTION;
    const int num_directions = (direction_str == "forward") ? 1 : 2;
    // There are three distinct uses of an activation function within the LSTM equations
    // One for the input/forget/output gates, one for the cell state, and one for the output
    // RNNv2 only supports the default choice for each, listed here (duplicated for bidirectional)
    std::vector<std::string> default_activations = {"Sigmoid", "Tanh", "Tanh"};
    if (num_directions == 2)
    {
        default_activations.insert(default_activations.end(), {"Sigmoid", "Tanh", "Tanh"});
    }
    const std::vector<std::string> activations
        = attrs.get<std::vector<std::string>>("activations", default_activations);
    ASSERT(activations == default_activations && "Nonstandard activations within LSTM unsupported",
        ErrorCode::kUNSUPPORTED_NODE);
    const float clip = attrs.get<float>("clip", 0.0f);
    ASSERT(clip == 0.0f && "Clipping unsupported", ErrorCode::kUNSUPPORTED_NODE);
    const int32_t hidden_size = attrs.get<int>("hidden_size");
    ASSERT(hidden_size > 0, ErrorCode::kINVALID_NODE);
    const int32_t input_forget = attrs.get<int>("input_forget", 0);
    ASSERT(0 == input_forget && "Coupled input/forget unsupported", ErrorCode::kUNSUPPORTED_NODE);

    // Optional Inputs
    bool has_bias = false;
    nvinfer1::ITensor* sequence_lens = nullptr;
    nvinfer1::ITensor* initial_h = nullptr;
    nvinfer1::ITensor* initial_c = nullptr;
    for (int i = 3; i < node.input_size(); i++)
    {
        const std::string& input_name = node.input(i);
        if (input_name == "B")
        {
            has_bias = true;
        }
        else if (input_name == "sequence_lens")
        {
            sequence_lens = &(convertToTensor(inputs.at(i), ctx));
            ASSERT(sequence_lens && sequence_lens->getType() == nvinfer1::DataType::kINT32
                    && "Failed to process sequence_lens (sequence_lens must be int32)",
                ErrorCode::kINVALID_NODE);
        }
        else if (input_name == "initial_h" || input_name == "initial_c")
        {
            nvinfer1::ITensor* output = nullptr;
            if (inputs.at(i).is_weights())
            {
                /* constant->shuffle bug (NVBug 2650549), so we do the transpose manually */
                ShapedWeights weights = inputs.at(i).weights();
                const int dtype_size = getDtypeSize(weights.type);
                const size_t len = num_directions * batch_size * hidden_size * dtype_size;
                auto* source = reinterpret_cast<unsigned char*>(weights.values);
                std::vector<unsigned char> buffer;
                buffer.resize(len);
                for (int d = 0; d < num_directions; d++)
                {
                    for (int j = 0; j < batch_size; j++)
                    {
                        for (int k = 0; k < hidden_size; k++)
                        {
                            for (int b = 0; b < dtype_size; b++)
                            {
                                int src_idx = d * batch_size * hidden_size * dtype_size + j * hidden_size * dtype_size
                                    + k * dtype_size + b;
                                int buf_idx = j * num_directions * hidden_size * dtype_size
                                    + d * hidden_size * dtype_size + k * dtype_size + b;
                                buffer.at(buf_idx) = source[src_idx];
                            }
                        }
                    }
                }
                std::memcpy(weights.values, static_cast<void*>(buffer.data()), len);
                const nvinfer1::Dims new_dims = {3, {batch_size, num_directions, hidden_size}};
                output = ctx->network()->addConstant(new_dims, weights)->getOutput(0);
                ASSERT(output && "Failed to convert initial_h or initial_c weights to constant layer",
                    ErrorCode::kINTERNAL_ERROR);
            }
            else
            {
                /* TODO: Once NVBug 2650549 is fixed, we can use just this path instead */
                /* nvinfer1::ITensor& source = convertToTensor(inputs.at(i), ctx); */
                nvinfer1::ITensor& source = inputs.at(i).tensor();
                auto* shuffle_layer = ctx->network()->addShuffle(source);
                ASSERT(shuffle_layer && "Failed to create initial_h shuffle layer", ErrorCode::kINTERNAL_ERROR);
                shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
                output = shuffle_layer->getOutput(0);
            }
            ASSERT(output->getType() == input_type && "initial_h and initial_c datatype must match input",
                ErrorCode::kINVALID_NODE);
            if (input_name == "initial_h")
            {
                initial_h = output;
            }
            else
            {
                initial_c = output;
            }
        }
        else if (input_name == "P")
        {
            ASSERT(false && "Peephole connections not supported", ErrorCode::kUNSUPPORTED_NODE);
        }
    }

    // Input Shuffle Layer
    auto* input_shuffle_layer = ctx->network()->addShuffle(raw_input);
    ASSERT(input_shuffle_layer && "Failed to create input shuffle layer", ErrorCode::kINTERNAL_ERROR);
    input_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});

    // RNNv2 Layer
    nvinfer1::ITensor& input_seqs = *(input_shuffle_layer->getOutput(0));
    const nvinfer1::RNNOperation op = nvinfer1::RNNOperation::kLSTM;
    const int32_t layer_count = 1;
    auto* layer = ctx->network()->addRNNv2(input_seqs, layer_count, hidden_size, max_seq_len, op);
    ASSERT(layer && "Failed to create RNNv2 layer", ErrorCode::kINTERNAL_ERROR);
    layer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    layer->setDirection(direction);
    if (sequence_lens)
    {
        layer->setSequenceLengths(*sequence_lens);
    }
    if (initial_h)
    {
        layer->setHiddenState(*initial_h);
    }
    if (initial_c)
    {
        layer->setCellState(*initial_c);
    }

    // Weights
    ASSERT(inputs.at(1).is_weights() && "W must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights() && "R must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights gate_weights = inputs.at(1).weights();
    ShapedWeights rcur_weights = inputs.at(2).weights();

    nvinfer1::DataType gate_weights_type, rcur_weights_type;
    ASSERT(convertDtype(gate_weights.type, &gate_weights_type) && "Bad datatype in W", ErrorCode::kINTERNAL_ERROR);
    ASSERT(convertDtype(rcur_weights.type, &rcur_weights_type) && "Bad datatype in R", ErrorCode::kINTERNAL_ERROR);
    ASSERT(input_type == gate_weights_type && "W datatype must match X", ErrorCode::kINVALID_NODE);
    ASSERT(input_type == rcur_weights_type && "R datatype must match X", ErrorCode::kINVALID_NODE);

    ShapedWeights bias_weights;
    if (has_bias)
    {
        ASSERT(inputs.at(3).is_weights() && "B must be constant", ErrorCode::kUNSUPPORTED_NODE);
        bias_weights = inputs.at(3).weights();
        nvinfer1::DataType bias_weights_type;
        ASSERT(convertDtype(bias_weights.type, &bias_weights_type) && "Bad datatype in B", ErrorCode::kINTERNAL_ERROR);
        ASSERT(input_type == bias_weights_type && "B datatype must match X", ErrorCode::kINVALID_NODE);
    }

    const int data_size = (input_type == nvinfer1::DataType::kFLOAT) ? 4 : 2;
    const int input_size = gate_weights.shape.d[2];

    auto weightBuilder
        = [input_type, data_size, hidden_size, ctx](int layer_index, ShapedWeights& src, int stride, int idx) {
              nvinfer1::Weights w;
              int direction_offset = data_size * layer_index * 4 * hidden_size * stride;
              int gate_offset = data_size * hidden_size * stride * idx;
              w.type = input_type;
              w.values = reinterpret_cast<void*>(
                  reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
              w.count = hidden_size * stride;
              return w;
          };

    // RNNv2 requires that a bias be set, even if none is provided
    auto zeroes = ctx->createTempWeights(gate_weights.type, nvinfer1::Dims{1, {hidden_size}});
    std::memset(zeroes.values, 0, data_size * hidden_size);

    auto biasBuilder
        = [input_type, data_size, hidden_size, has_bias, zeroes](int layer_index, ShapedWeights& src, int idx) {
              nvinfer1::Weights b;
              int direction_offset = data_size * layer_index * 8 * hidden_size;
              int gate_offset = data_size * hidden_size * idx;
              b.type = input_type;
              if (has_bias)
              {
                  b.values = reinterpret_cast<void*>(
                      reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
              }
              else
              {
                  b.values = zeroes.values;
              }
              b.count = hidden_size;
              return b;
          };

    for (int layer_index = 0; layer_index < num_directions; layer_index++)
    {
        nvinfer1::Weights W_i = weightBuilder(layer_index, gate_weights, input_size, 0);
        nvinfer1::Weights W_o = weightBuilder(layer_index, gate_weights, input_size, 1);
        nvinfer1::Weights W_f = weightBuilder(layer_index, gate_weights, input_size, 2);
        nvinfer1::Weights W_c = weightBuilder(layer_index, gate_weights, input_size, 3);
        nvinfer1::Weights R_i = weightBuilder(layer_index, rcur_weights, hidden_size, 0);
        nvinfer1::Weights R_o = weightBuilder(layer_index, rcur_weights, hidden_size, 1);
        nvinfer1::Weights R_f = weightBuilder(layer_index, rcur_weights, hidden_size, 2);
        nvinfer1::Weights R_c = weightBuilder(layer_index, rcur_weights, hidden_size, 3);

        bool isW = true;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT, isW, W_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, W_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, W_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL, isW, W_c);
        isW = false;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT, isW, R_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, R_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, R_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL, isW, R_c);

        nvinfer1::Weights B_wi = biasBuilder(layer_index, bias_weights, 0);
        nvinfer1::Weights B_wo = biasBuilder(layer_index, bias_weights, 1);
        nvinfer1::Weights B_wf = biasBuilder(layer_index, bias_weights, 2);
        nvinfer1::Weights B_wc = biasBuilder(layer_index, bias_weights, 3);
        nvinfer1::Weights B_ri = biasBuilder(layer_index, bias_weights, 4);
        nvinfer1::Weights B_ro = biasBuilder(layer_index, bias_weights, 5);
        nvinfer1::Weights B_rf = biasBuilder(layer_index, bias_weights, 6);
        nvinfer1::Weights B_rc = biasBuilder(layer_index, bias_weights, 7);

        isW = true;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT, isW, B_wi);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_wo);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_wf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL, isW, B_wc);
        isW = false;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT, isW, B_ri);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_ro);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_rf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL, isW, B_rc);
    }

    // Outputs
    ASSERT((layer->getNbOutputs() == 3) && "3 outputs are required.", ErrorCode::kINTERNAL_ERROR);
    ASSERT(
        (node.output_size() <= 3) && "At most 3 outputs are allowed for the LSTM operator.", ErrorCode::kINVALID_NODE);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < node.output_size(); i++)
    {
        auto* shuffle_layer = ctx->network()->addShuffle(*(layer->getOutput(i)));
        ASSERT(shuffle_layer && "Failed to create output shuffle layer", ErrorCode::kINTERNAL_ERROR);
        shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1, 0, 2});
        if (i == 0)
        {
            nvinfer1::Dims Y_dims{4, {max_seq_len, batch_size, num_directions, hidden_size}};
            shuffle_layer->setReshapeDimensions(Y_dims);
            shuffle_layer->setZeroIsPlaceholder(false);
            shuffle_layer->setSecondTranspose(nvinfer1::Permutation{0, 2, 1, 3});
        }
        outputs.emplace_back(shuffle_layer->getOutput(0));
    }
    return {outputs};
}

nvinfer1::Dims makeDims(int nbDims, int val)
{
    // Zero all the dimensions, so that unused dimensions are deterministic even if accidentally used.
    nvinfer1::Dims dims{nbDims, {}};
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

nvinfer1::Dims insertDimension(const nvinfer1::Dims& dims, const int axis, const int value)
{
    assert(dims.nbDims < nvinfer1::Dims::MAX_DIMS && axis < nvinfer1::Dims::MAX_DIMS);
    nvinfer1::Dims newDims{};
    newDims.nbDims = dims.nbDims + 1;
    std::copy(dims.d, dims.d + axis, newDims.d);
    newDims.d[axis] = value;
    std::copy(dims.d + axis, dims.d + dims.nbDims, newDims.d + axis + 1);
    return newDims;
}

std::vector<float> parseLSTMActivationValues(const std::vector<nvinfer1::ActivationType>& activationTypes, const std::vector<float>& activationValues, bool isAlpha)
{
    size_t actIndex{0};
    std::vector<float> tmpActs{};
    for (size_t i = 0; i < activationTypes.size(); ++i)
    {
        float defaultVal = isAlpha ? getActivationDefaultAlpha(activationTypes[i]) : getActivationDefaultBeta(activationTypes[i]);
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

bool parseExternalWeights(IImporterContext* ctx, std::string file, std::string path, int64_t offset, int64_t length,
    std::vector<char>& weightsBuf, size_t& size)
{
    // The weight paths in the ONNX model are relative paths to the main ONNX file.
#ifdef _MSC_VER
    size_t slash = path.rfind("\\");
    // When using WSL path can have "\" or "/". Need to check both options here.
    // See bug https://nvbugs/3635640 for an example
    if (slash == std::string::npos)
    {
        slash = path.rfind("/");
    }
#else
    size_t slash = path.rfind("/");
#endif
    if (slash != std::string::npos)
    {
        path.replace(slash + 1, path.size() - (slash + 1), file);
    }
    else
    {
        path = file;
    }
    std::ifstream relPathFile(path, std::ios::binary | std::ios::ate);
    if (!relPathFile)
    {
        LOG_ERROR("Failed to open file: " << path);
        return false;
    }
    std::streamsize fileSize = relPathFile.tellg();
    relPathFile.seekg(offset, std::ios::beg);
    int64_t weightsBufSize = length == 0 ? fileSize : length;
    weightsBuf.resize(weightsBufSize);
    LOG_VERBOSE("Reading weights from external file: " << path);
    if (!relPathFile.read(weightsBuf.data(), weightsBuf.size()))
    {
        LOG_ERROR("Failed to read weights from external file: " << path);
        return false;
    }
    size = weightsBuf.size();
    return true;
}

NodeImportResult poolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensorPtr->getDimensions();
    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }

    OnnxAttrs attrs(node, ctx);
    int nbSpatialDims = attrs.at("kernel_shape")->ints().size();
    ASSERT(((nbSpatialDims == 1 && needToExpandDims) || nbSpatialDims == 2 || nbSpatialDims == 3)
            && "The attribute kernel_shape misaligns with the shape of the input tensor.",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;

    bool exclude_padding(true);

    // Ceiling-mode output padding and dilations added in opset 10
    bool ceilMode(false);
    if (ctx->getOpsetVersion() >= 10)
    {
        ceilMode = static_cast<bool>(attrs.get<int>("ceil_mode", 0));
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT((dilations[i] == 1) && "This version of TensorRT does not support dilations other than 1.",
                ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
    }

    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding,
        nullptr, nullptr, ceilMode);

    if (needToExpandDims)
    {
        kernel_size = insertDimension(kernel_size, nbSpatialDims, 1);
        strides = insertDimension(strides, nbSpatialDims, 1);
        beg_padding = insertDimension(beg_padding, nbSpatialDims, 0);
        end_padding = insertDimension(end_padding, nbSpatialDims, 0);
    }

    nvinfer1::IPoolingLayer* poolingLayer = ctx->network()->addPoolingNd(*tensorPtr, type, kernel_size);
    poolingLayer->setStrideNd(strides);
    // This member is ignored in maxpooling
    poolingLayer->setAverageCountExcludesPadding(exclude_padding);
    poolingLayer->setPaddingMode(paddingMode);
    poolingLayer->setPrePadding(beg_padding);
    poolingLayer->setPostPadding(end_padding);

    ctx->registerLayer(poolingLayer, getNodeName(node));
    tensorPtr = poolingLayer->getOutput(0);
    dims = tensorPtr->getDimensions();
    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr && "Failed to squeeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensorPtr}};
}

NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation, TensorOrWeights inputAxes)
{
    nvinfer1::ITensor& tensor = convertToTensor(input, ctx);
    OnnxAttrs attrs(node, ctx);
    bool keepdims = attrs.get("keepdims", 1);
    int ndim = tensor.getDimensions().nbDims;
    std::vector<int32_t> axes;
    if (attrs.count("axes"))
    {
        axes = attrs.get<std::vector<int>>("axes");
    }
    else if (!inputAxes.isNullTensor())
    {
        ASSERT(inputAxes.is_weights() && "Axis input must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
        weightsToVector<int32_t>(inputAxes.weights(), &axes);
    }
    else
    {
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }

    uint32_t axisMask = 0;
    for (int axis : axes)
    {
        CHECK(convertAxis(axis, ndim));
        axisMask |= 1 << axis;
    }

    auto* layer = ctx->network()->addReduce(tensor, operation, axisMask, keepdims);
    ctx->registerLayer(layer, getNodeName(node));
    return {{layer->getOutput(0)}};
}

nvinfer1::ITensor* reshapeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape)
{
    if (shape == tensor.getDimensions())
    {
        return &tensor;
    }
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    if (!layer)
    {
        return nullptr;
    }
    layer->setReshapeDimensions(shape);
    layer->setZeroIsPlaceholder(false);
    return layer->getOutput(0);
}

NodeImportResult scaleHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor& tensor_,
    nvinfer1::ScaleMode mode, const nvinfer1::Weights& shift, const nvinfer1::Weights& scale,
    const nvinfer1::Weights& power, const char* shiftName, const char* scaleName)
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
            tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, expandAxes);
        }
        else
        {
            // Collapse trailing dimensions if origShape.size() > 5
            const ShapeTensor collapsedDim = product(ctx, origShape, 3, origShape.size(), 1);
            const ShapeTensor collapsedShape = concat(ctx, gather(ctx, origShape, iotaShapeVector(3)), collapsedDim);
            tensorPtr = &reshape(ctx, *tensorPtr, collapsedShape);
        }
    }

    auto* layer = ctx->network()->addScaleNd(*tensorPtr, mode, shift, scale, power, 1);
    ASSERT(layer && "Failed to add a Scale layer.", ErrorCode::kUNSUPPORTED_NODE);
    // Register layer name, and shift and scale weight names for the refit map.
    ctx->registerLayer(layer, getNodeName(node));
    ctx->network()->setWeightsName(shift, shiftName);
    ctx->network()->setWeightsName(scale, scaleName);

    tensorPtr = layer->getOutput(0);

    if (needToReshape)
    {
        tensorPtr = &reshape(ctx, *tensorPtr, origShape);
        ASSERT(tensorPtr && "Failed to reshape tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensorPtr}};
}

void setAttr(
    nvinfer1::Dims* trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal)
{
    assert(trtAttr->nbDims == nbSpatialDims);
    int ndim = onnxAttr->ints().size();
    for (int i = 0; i < nbSpatialDims; ++i)
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
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* data, const int axis)
{
    ShapeTensor starts, sizes, strides;
    ShapeTensor axisLength = ShapeTensor(*getAxisLength(ctx, data, axis, {1, {1}}));
    const int nbDims = data->getDimensions().nbDims;

    std::vector<int64_t> values(nbDims, 0);
    starts = ShapeTensor(1, std::move(values));
    sizes = axis == 0 ? shapeVector(1) : ShapeTensor(*getAxisLength(ctx, data, 0, {1, {1}}));
    strides = axis == 0 ? axisLength : shapeVector(1);

    // On axis dimension, set strides = lengthOfDim and sizes = 1
    for (int i = 1; i < nbDims; i++)
    {
        if (i == axis)
        {
            strides = concat(ctx, strides, axisLength);
            sizes = concat(ctx, sizes, shapeVector(1));
        }
        else
        {
            ShapeTensor currLength = ShapeTensor(*getAxisLength(ctx, data, i, {1, {1}}));
            strides = concat(ctx, strides, shapeVector(1));
            sizes = concat(ctx, sizes, currLength);
        }
    }
    auto output = addSlice(ctx, *data, starts, sizes, strides)->getOutput(0);
    return output;
}

nvinfer1::ITensor* squeezeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, const std::vector<int>& axes, bool regLayer)
{
    const auto dims = shapeOf(tensor);

    // Set subscripts to ShapeTensor containing positions of axes to be kept.
    // For example, if there are 6 dimensions and axes = {1,5}, set subscripts to {0,2,3,4}.
    std::vector<int64_t> subscripts(dims.size());
    std::iota(subscripts.begin(), subscripts.end(), 0);
    auto p = std::remove_if(subscripts.begin(), subscripts.end(),
        [axes](int x) { return std::find(axes.begin(), axes.end(), x) != axes.end(); });
    subscripts.resize(p - subscripts.begin());

    auto newDims = gather(ctx, dims, ShapeTensor(1, std::move(subscripts)));
    LOG_VERBOSE("Original shape: " << dims << ", squeezing to: " << newDims);
    nvinfer1::IShuffleLayer* squeezeLayer = addShuffle(ctx, tensor, newDims);
    if (regLayer)
    {
        ctx->registerLayer(squeezeLayer, getNodeName(node));
    }
    return squeezeLayer->getOutput(0);
}

nvinfer1::ITensor* transposeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm)
{
    const nvinfer1::Dims shape = tensor.getDimensions();

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    ctx->registerLayer(layer, getNodeName(node));
    if (!layer)
    {
        return nullptr;
    }

    // If a transpose is required, add transpose property to the shuffle layer.
    if (isTransposeRequired(shape, perm))
    {
        layer->setFirstTranspose(perm);
    }
    // Else, the transpose can be simplified to a reshape.
    else
    {
        nvinfer1::Dims new_shape;
        new_shape.nbDims = shape.nbDims;
        for (int i = 0; i < new_shape.nbDims; ++i)
        {
            new_shape.d[i] = shape.d[perm.order[i]];
        }
        layer->setReshapeDimensions(new_shape);
        layer->setZeroIsPlaceholder(false);
    }
    return layer->getOutput(0);
}

NodeImportResult unaryHelper(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, TensorOrWeights& input, nvinfer1::UnaryOperation op)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(input, ctx);
    const auto rank = tensorPtr->getDimensions().nbDims;
    const auto inputType = tensorPtr->getType();

    bool validUnaryType = true;
    switch (op)
    {
    case nvinfer1::UnaryOperation::kNOT:
    {
        // TRT only supports BOOL types for the NOT operation
        validUnaryType = inputType == nvinfer1::DataType::kBOOL;
        break;
    }
    case nvinfer1::UnaryOperation::kABS:
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
            nvinfer1::ITensor* zeroTensor = addConstant(ctx, std::vector<int32_t>{0}, ::ONNX_NAMESPACE::TensorProto::INT32, {0, {1}})->getOutput(0);
            CHECK(broadcastTensors(ctx, zeroTensor, tensorPtr));
            std::vector<TensorOrWeights> layerInputs = {zeroTensor, tensorPtr};
            return elementwiseHelper(ctx, node, layerInputs, nvinfer1::ElementWiseOperation::kSUB);
        }
        validUnaryType = (inputType != nvinfer1::DataType::kBOOL && inputType != nvinfer1::DataType::kUINT8);
        break;
    }
    default:
    {
        // By default TRT does not support BOOL, INT32, UINT8 types for Unary operations.
        validUnaryType = (inputType != nvinfer1::DataType::kBOOL && inputType != nvinfer1::DataType::kINT32 && inputType != nvinfer1::DataType::kUINT8);
    }
    }

    ASSERT(validUnaryType
            && "This version of TensorRT does not support the given operator with the given input data type.",
        ErrorCode::kUNSUPPORTED_NODE);

    // Support scalar inputs by unsqueezing to 1D
    if (rank == 0)
    {
        std::vector<int> axes{0};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
    }

    nvinfer1::IUnaryLayer* layer = ctx->network()->addUnary(*tensorPtr, op);
    ctx->registerLayer(layer, getNodeName(node));
    tensorPtr = layer->getOutput(0);

    // Squeeze scalar inputs back into a scalar
    if (rank == 0)
    {
        std::vector<int> axes{0};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
    }

    return {{tensorPtr}};
}

NodeImportResult convMultiInput(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, std::vector<TensorOrWeights>& inputs)
{
    ASSERT(inputs.size() >= 2 && "Convolution require at least 2 inputs.", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* input_tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims const input_dims = input_tensor_ptr->getDimensions();

    nvinfer1::Dims dims = input_dims;
    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        const std::vector<int> axes{3};
        input_tensor_ptr = unsqueezeTensor(ctx, node, *input_tensor_ptr, axes);
        ASSERT(input_tensor_ptr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        dims = input_tensor_ptr->getDimensions();
    }
    auto const nbSpatialDims = dims.nbDims - 2;

    nvinfer1::Dims filter_dim;
    filter_dim.nbDims = nbSpatialDims;
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding{false};
    getKernelParams(
        ctx, node, &filter_dim, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding, &dilations);
    auto const nChannel = dims.d[1];
    auto const K = inputs.at(1).shape().d[0];
    auto const C = inputs.at(1).shape().d[1];

    auto kernel_weights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);
    auto bias_weights = ShapedWeights::empty(::ONNX_NAMESPACE::TensorProto::FLOAT);

    auto const checkSpatialDims = [&nbSpatialDims, &filter_dim](nvinfer1::Dims const& dims) {
        // Check that the number of spatial dimensions and the kernel shape matches up.
        if (nbSpatialDims != dims.nbDims - 2)
        {
            return false;
        }

        return std::equal(filter_dim.d, filter_dim.d + nbSpatialDims, dims.d + dims.nbDims - nbSpatialDims);
    };

    nvinfer1::ITensor* kernel_tensor_ptr{nullptr};
    nvinfer1::ITensor* bias_tensor_ptr{nullptr};
    if (inputs.at(1).is_tensor())
    {
        kernel_tensor_ptr = &convertToTensor(inputs.at(1), ctx);
        if (needToExpandDims)
        {
            // Expand spatial dims from 1D to 2D
            std::vector<int32_t> const axes{3};
            kernel_tensor_ptr = unsqueezeTensor(ctx, node, *kernel_tensor_ptr, axes);
            ASSERT(kernel_tensor_ptr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
        }
        ASSERT(checkSpatialDims(kernel_tensor_ptr->getDimensions())
                && "The input tensor shape misaligns with the input kernel shape.",
            ErrorCode::kUNSUPPORTED_NODE);
    }
    else
    {
        kernel_weights = inputs.at(1).weights();
        if (needToExpandDims)
        {
            kernel_weights.shape.nbDims = 4;
            kernel_weights.shape.d[3] = 1;
        }
        ASSERT(
            checkSpatialDims(kernel_weights.shape) && "The input tensor shape misaligns with the input kernel shape.",
            ErrorCode::kUNSUPPORTED_NODE);
    }

    if (inputs.size() == 3)
    {
        if (inputs.at(2).is_weights())
        {
            bias_weights = inputs.at(2).weights();
        }
        else
        {
            bias_tensor_ptr = &convertToTensor(inputs.at(2), ctx);
        }
    }

    OnnxAttrs attrs(node, ctx);
    int ngroup = attrs.get("group", 1);
    ASSERT((nChannel == -1 || C * ngroup == nChannel)
            && "The attribute group and the kernel shape misalign with the channel size of the input tensor. ",
        ErrorCode::kINVALID_NODE);

    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*input_tensor_ptr, K, filter_dim, kernel_weights, bias_weights);
    ASSERT(layer && "Failed to add the Convolution layer.", ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
    layer->setDilationNd(dilations);
    layer->setNbGroups(ngroup);

    // Set dynamic weights
    if (kernel_tensor_ptr)
    {
        layer->setInput(1, *kernel_tensor_ptr);
    }
    if (bias_tensor_ptr)
    {
        layer->setInput(2, *bias_tensor_ptr);
    }
    ctx->registerLayer(layer, getNodeName(node));

    nvinfer1::ITensor* output_tensor_ptr = layer->getOutput(0);
    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        const std::vector<int> axes{3};
        output_tensor_ptr = squeezeTensor(ctx, node, *output_tensor_ptr, axes);
        ASSERT(output_tensor_ptr && "Failed to unsqueeze tensor.", ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{output_tensor_ptr}};
}

nvinfer1::ITensor* unsqueezeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, const std::vector<int>& axes, bool regLayer)
{
    const auto dims = shapeOf(tensor);
    const std::set<int> axesSet(axes.begin(), axes.end());

    // Ensure that result fits maximum allowed dimensions.
    if (dims.size() + axesSet.size() > nvinfer1::Dims::MAX_DIMS)
    {
        return nullptr;
    }

    // Compute interlacing subscripts.
    std::vector<int64_t> subscripts(dims.size());
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (const auto& axis : axesSet)
    {
        subscripts.insert(subscripts.begin() + axis, dims.size());
    }

    const auto newDims = interlace(ctx, dims, shapeVector(1), ShapeTensor(1, std::move(subscripts)));
    LOG_VERBOSE("Original shape: " << dims << ", unsqueezing to: " << newDims);
    nvinfer1::IShuffleLayer* unsqueezeLayer = addShuffle(ctx, tensor, newDims);
    if (regLayer)
    {
        ctx->registerLayer(unsqueezeLayer, getNodeName(node));
    }
    return unsqueezeLayer->getOutput(0);
}

nvinfer1::ITensor* resizeShapeTensor(IImporterContext* ctx, nvinfer1::ITensor& input, TensorOrWeights& scales)
{
    // Create below subnetwork for processing resize scale tensor or weights.
    // clang-format off
    // scale weights (convert to tensor) or scale tensor -> elementwise mul -> transformation(floor, ceil, round) -> identity (cast to int) -> resize shape tensor
    //      input -> shapeof -> identity (cast to float) ->
    // clang-format on
    auto* floatCast = ctx->network()->addIdentity(shapeOf(input).tensor(ctx));
    floatCast->setOutputType(0, nvinfer1::DataType::kFLOAT);
    auto* inputShapeTensor = floatCast->getOutput(0);

    auto& scaleTensor = convertToTensor(scales, ctx);
    auto* prod = ctx->network()
                     ->addElementWise(scaleTensor, *inputShapeTensor, nvinfer1::ElementWiseOperation::kPROD)
                     ->getOutput(0);
    auto* floor = ctx->network()->addUnary(*prod, nvinfer1::UnaryOperation::kFLOOR)->getOutput(0);
    auto* intCast = ctx->network()->addIdentity(*floor);
    intCast->setOutputType(0, nvinfer1::DataType::kINT32);
    return intCast->getOutput(0);
}

int64_t volume(const nvinfer1::Dims& dims)
{
    std::for_each(
        dims.d, dims.d + dims.nbDims, [](int32_t d) { assert(d >= 0 && "volume makes no sense for dynamic shapes"); });
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

const std::string getNodeName(const ::ONNX_NAMESPACE::NodeProto& node)
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
    IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& lowerBound, const ShapeTensor& upperBound)
{
    return min(ctx, max(ctx, x, lowerBound), upperBound);
}

//! Return ShapeTensor representing indices < 0 ? inputDims + indices : indices
static ShapeTensor bumpIfNegative(IImporterContext* ctx, const ShapeTensor& inputDims, const ShapeTensor& indices)
{
    const auto signs = clamp(ctx, indices, shapeVector(-1), shapeVector(0));
    return sub(ctx, indices, mul(ctx, signs, inputDims));
}

void decodeOnnxStartsAndEnds(IImporterContext* ctx, const ShapeTensor& inputDims, const ShapeTensor& steps,
    ShapeTensor& starts, ShapeTensor& ends)
{
    //! The ONNX specification is unreliable (https://github.com/onnx/onnx/issues/3063)
    //! thus the logic here is designed to match that in
    //! https://github.com/onnx/onnx/blob/master/onnx/defs/tensor/defs.cc .

    // Set stepSign to step < 0 ? -1 : 0.
    const auto stepSign = clamp(ctx, steps, shapeVector(-1), shapeVector(0));

    // Update starts.
    starts = bumpIfNegative(ctx, inputDims, starts);
    starts = clamp(ctx, starts, shapeVector(0), add(ctx, inputDims, stepSign));

    // Update ends
    ends = bumpIfNegative(ctx, inputDims, ends);
    ends = clamp(ctx, ends, stepSign, inputDims);
}

ShapeTensor axesToInterlaceSubscripts(const ShapeTensor& axes, int nbDims)
{
    std::vector<int64_t> subscripts(nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (int32_t i = 0; i < axes.size(); ++i)
    {
        subscripts[axes[i]] = nbDims + i;
    }
    return ShapeTensor(1, std::move(subscripts));
}

ShapeTensor computeSliceSizes(IImporterContext* ctx, const ShapeTensor& starts, const ShapeTensor& ends,
    const ShapeTensor& steps, const ShapeTensor& dims)
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

nvinfer1::ITensor* addSoftmax(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor& input)
{
    OnnxAttrs attrs(node, ctx);
    // "axis : int (default is opset specific)"
    const int defaultAxis = (ctx->getOpsetVersion() >= 13) ? -1 : 1;
    int axis = attrs.get("axis", defaultAxis);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    const auto rank = shapeOf(input).size();
    if (convertAxis(axis, rank).is_error())
    {
        return nullptr;
    }

    nvinfer1::ISoftMaxLayer* softMax{nullptr};
    if (ctx->getOpsetVersion() >= 13)
    {
        softMax = ctx->network()->addSoftMax(input);
        softMax->setAxes(1 << axis);
    }
    else
    {
        // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
        auto* flattened = flattenTensor(ctx, node, input, axis);
        softMax = ctx->network()->addSoftMax(*flattened);
        // ONNX softmax is always on second dimension.
        softMax->setAxes(1 << 1);
    }
    ctx->registerLayer(softMax, node.name());
    return softMax->getOutput(0);
}

NodeImportResult addScatterLayer(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ScatterMode mode, int32_t axis)
{
    CHECK(notInvalidType(inputs.at(0), {"UINT8"}));
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& updates = convertToTensor(inputs.at(2), ctx);

    // Validate input dimensions
    if (mode == nvinfer1::ScatterMode::kELEMENT)
    {
        const auto dataDims = data.getDimensions();
        const auto indicesDims = indices.getDimensions();
        const auto updatesDims = updates.getDimensions();

        // Ranks must all be the same
        ASSERT(dataDims.nbDims == indicesDims.nbDims && dataDims.nbDims == updatesDims.nbDims && "Input dimensions to ScatterElements must have the same rank!",
                ErrorCode::kUNSUPPORTED_NODE);

        // Corresponding dimensions of indices and updates must be <= data
        for (int32_t i = 0; i < dataDims.nbDims; ++i)
        {
            if (indicesDims.d[i] != -1 && dataDims.d[i] != -1)
            {
                ASSERT(indicesDims.d[i] <= dataDims.d[i] && "Indices dimensions must be less than data dimensions!", ErrorCode::kUNSUPPORTED_NODE);
            }
            if (updatesDims.d[i] != -1 && dataDims.d[i] != -1)
            {
                ASSERT(updatesDims.d[i] <= dataDims.d[i] && "Updates dimensions must be less than data dimensions!", ErrorCode::kUNSUPPORTED_NODE);
            }
        }
    }

    auto* layer = ctx->network()->addScatter(data, indices, updates, mode);
    layer->setAxis(axis);
    ctx->registerLayer(layer, getNodeName(node));
    return {{layer->getOutput(0)}};
}

//! Helper function to calculate mod(A, B)  
nvinfer1::IElementWiseLayer* modWithIntegerInputs(IImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, bool fmod){
    using eOp = nvinfer1::ElementWiseOperation;
    auto divOp = eOp::kFLOOR_DIV;
    if (fmod) divOp = eOp::kDIV;
    
    // input0 - (input1 * divOp(input0, input1))
    return ctx->network()->addElementWise(*input0,
                *ctx->network()->addElementWise(*input1,
                    *ctx->network()->addElementWise(*input0, *input1, divOp)->getOutput(0),
                eOp::kPROD)->getOutput(0),
            eOp::kSUB);
}

nvinfer1::IElementWiseLayer* modWithFPInputs(IImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, nvinfer1::ITensor* divResult, bool sameSign){
    using eOp = nvinfer1::ElementWiseOperation;
    using uOp = nvinfer1::UnaryOperation;
    // divResult need to be round towards 0
    // When inputs have the same sign, round down (input0 / input1)
    auto roundOp = uOp::kFLOOR;
    // When inputs have opposite signs, round up (input0 / input1)
    if (! sameSign) roundOp = uOp::kCEIL;

    // input0 - (input1 * round_towards_0(input0/ input1))
    return ctx->network()->addElementWise(*input0,
                *ctx->network()->addElementWise(*input1,
                    *ctx->network()->addUnary(*divResult, roundOp)->getOutput(0),
                eOp::kPROD)->getOutput(0),
            eOp::kSUB);
}

float* convertFP16Data(void* weightValues, nvinfer1::Dims shape, IImporterContext* ctx)
{
    size_t const nbWeights = volume(shape);
    float* newWeights{static_cast<float*>(ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::FLOAT, shape).values)};

    half_float::half* tempValues = static_cast<half_float::half*>(weightValues);

    for (size_t i = 0; i < nbWeights; i++)
    {
        newWeights[i] = tempValues[i];
    }
    return newWeights;
}

} // namespace onnx2trt
