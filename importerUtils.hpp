/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"
#include "OnnxAttrs.hpp"
#include "ShapeTensor.hpp"
#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "errorHelpers.hpp"
#include "weightUtils.hpp"

#include "impl/NvInferPythonPlugin.h"
#include <NvInfer.h>

#include "bfloat16.hpp"
#include "half.h"
#include <cstring> // For std::memcpy
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <typeindex>
#include <unordered_map>

namespace onnx2trt
{

constexpr char const* kTRT_STD_PLUGIN_NAMESPACE = "";

struct PluginDeleter
{
    void operator()(nvinfer1::IPluginV2* t);
};

// Helper function to create and fill a Dims object with defined values
nvinfer1::Dims makeDims(int32_t nbDims, int64_t val);

// Helper function to add a single constant value into TensorRT
template <typename ScalarType>
nvinfer1::IConstantLayer* addConstantScalar(
    ImporterContext* ctx, ScalarType scalar, ShapedWeights::DataType type, int32_t nbDims = 0)
{
    ONNXTRT_CHECK(getShapedWeightsDataType<ScalarType>() == type, "Found type mismatch when creating scalar value",
        ErrorCode::kINTERNAL_ERROR);
    ONNXTRT_CHECK(nbDims >= 0 && nbDims <= nvinfer1::Dims::MAX_DIMS,
        "nbDims must be between 0 and " << nvinfer1::Dims::MAX_DIMS, ErrorCode::kINTERNAL_ERROR);
    nvinfer1::Dims const dims = makeDims(nbDims, 1);
    ShapedWeights scalarWeights = ctx->createNamedTempWeights(type, dims);
    static_cast<ScalarType*>(scalarWeights.values)[0] = static_cast<ScalarType>(scalar);
    nvinfer1::IConstantLayer* l = N_CHECK(ctx->network()->addConstant(scalarWeights.shape, scalarWeights));
    ctx->network()->setWeightsName(scalarWeights, scalarWeights.getName());
    return l;
}

// Helper function to create a tensor given a vector of values and a shape.
template <typename ScalarType>
nvinfer1::IConstantLayer* addConstant(
    ImporterContext* ctx, std::vector<ScalarType> const& values, ShapedWeights::DataType type, nvinfer1::Dims shape)
{
    ONNXTRT_CHECK(getShapedWeightsDataType<ScalarType>() == type, "Found type mismatch when creating scalar value", ErrorCode::kINTERNAL_ERROR);
    ONNXTRT_CHECK(volume(shape) == static_cast<int64_t>(values.size()), "Shape does not match number of values provided", ErrorCode::kINTERNAL_ERROR);
    auto const sizeInBits = getDtypeSizeBits(type);
    ONNXTRT_CHECK(sizeInBits % 8 == 0, "Size of type must be a multiple of 8 bits", ErrorCode::kINTERNAL_ERROR); // TRT-22989: handle sub-byte size and shape checks
    ONNXTRT_CHECK(sizeof(ScalarType) == sizeInBits / 8, "ONNX dtype does not have the same size as the value type", ErrorCode::kINTERNAL_ERROR);
    ShapedWeights weights = ctx->createNamedTempWeights(type, shape);
    std::memcpy(weights.values, values.data(), values.size() * sizeof(ScalarType));
    nvinfer1::IConstantLayer* l = N_CHECK(ctx->network()->addConstant(weights.shape, weights));
    // Only set names for non-empty weights
    if (values.size() != 0)
    {
        ctx->network()->setWeightsName(weights, weights.getName());
    }
    return l;
}

// Helper overloads for comparisons between dimensions.
inline bool operator==(nvinfer1::Dims const& a, nvinfer1::Dims const& b)
{
    if (a.nbDims != b.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < a.nbDims; ++i)
    {
        if (a.d[i] != b.d[i])
        {
            return false;
        }
    }
    return true;
}

inline bool operator!=(nvinfer1::Dims const& a, nvinfer1::Dims const& b)
{
    return !(a == b);
}

enum ScaleOp
{
    kSHIFT,
    kSCALE,
    kPOWER,
};

// Helper function to import ONNX activation nodes into TRT
NodeOutputs activationHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha = nullptr, float* beta = nullptr);

// Add clipping to a tensor if clip is a valid value.
nvinfer1::ITensor* addClip(ImporterContext* ctx, nvinfer1::ITensor* input, float clip);

// Helper function to import ArgMax and ArgMin nodes into TRT
NodeOutputs argMinMaxHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op);

//! If t has rank less than nbDims, reshape it to have nbDims by prepending ones to its dimensions.
//! Assert failure if t has rank greater than nbDims.
void broadcastTensor(ImporterContext* ctx, nvinfer1::ITensor*& t, int const nbDims);

// Helper function to broadcast two tensors to the larger one's shape
void broadcastTensors(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

// Helper function to broadcast three tensors to the largest one's shape
void broadcastTensors(ImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3);

// Helper function to calculate the bias tensor for GatherElements.
std::vector<int32_t> calculateBias(
    nvinfer1::Dims const& daDims, nvinfer1::Dims const& idxDims, std::vector<int32_t> const& pitches, int32_t axis);

// Helper function to check that linear/cubic resize can be used
bool canUseNDResize(size_t const scaleSize, float const* scaleFactors, size_t const n);

// Helper function to calculate and return a vector representation of the pitches of a given shape
std::vector<int32_t> calculatePitches(nvinfer1::Dims const& inputDims);

// Helper function to add a Cast layer in the network
nvinfer1::ITensor* castHelper(ImporterContext* ctx, nvinfer1::ITensor* input, nvinfer1::DataType dtype);

// Helper function for constantOfShape operator. Input shape must be a shape tensor
nvinfer1::ITensor* constantOfShape(ImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape);

// Helper function to convert an ONNX axis into a TRT axis
void convertAxis(int32_t& axis, int32_t const nbDims, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx);

// Helper function to convert an ONNX datatype into a TRT datatype
bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert ONNX padding into TRT padding. Will update startTensor and totalPaddingTensor by reference
bool convertOnnxPadding(ImporterContext* ctx, int32_t nbInputDims, std::vector<int64_t> const& onnxPadding,
    nvinfer1::ITensor*& startTensor, nvinfer1::ITensor*& totalPaddingTensor);

// Helper function to check if all of the values in the shift tensor are zeros
bool shiftIsAllZeros(ShapedWeights const& shiftInt8);

// Helper function to create zero shifts for QuantizeLinear/DequantizeLinear ops
onnx2trt::ShapedWeights createZeroShifts(onnx2trt::ShapedWeights const& shiftInt8, int32_t type, ImporterContext* ctx);

// Helper function to create a tensor of all zeros with the same shape as a data tensor
nvinfer1::ITensor* createZeroTensor(ImporterContext* ctx, nvinfer1::ITensor* data);

// Helper function to convert multi input convolution
NodeOutputs convMultiInput(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs);

// Helper function to convert a 1D tensor into a scalar
nvinfer1::ITensor* convertToScalar(ImporterContext* ctx, nvinfer1::ITensor* inpTensor);

// Helper function to convert a ShapedWeights object into a tensor
nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, ImporterContext* ctx);

// Helper function to convert a ShapedWeights object into a scalar
nvinfer1::ITensor* convertToScalar(TensorOrWeights& input, ImporterContext* ctx);

// Helper function to convert a scalar into a vector.
nvinfer1::ITensor* convertScalarToVector(ImporterContext* ctx, nvinfer1::ITensor* input);

// Helper function to provide a ceiling-rounding division between two integers
int divCeil(int n, int d);

// Helper function to check that the input data types for an elementwise operation are supported
void elementwiseCheck(std::vector<TensorOrWeights> const& inputs, const nvinfer1::ElementWiseOperation op,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx);

// Helper function to import an ONNX elementwise op into TRT
NodeOutputs elementwiseHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights> const& inputs, nvinfer1::ElementWiseOperation binary_op);

// Helper function to flatten a tensor on a given axis
nvinfer1::ITensor* flattenTensor(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ITensor& tensor, int axis = 0, bool regLayer = false);

// Slice out the specified dimension from a shape tensor. e.g. extractDimension(shape=(7, 6, 5), dim=2) would return 5.
// shape specifies the shape of the returned Tensor. Must have a volume of 1.
nvinfer1::ITensor* extractDimension(
    ImporterContext* ctx, nvinfer1::ITensor* shapeTensor, int32_t dim, nvinfer1::Dims shape);

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims inputShape, nvinfer1::Dims outputShape, nvinfer1::Dims kernelSize,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, int const nbSpatialDims, nvinfer1::Dims& begPadding,
    nvinfer1::Dims& endPadding, nvinfer1::Dims& outputPadding, nvinfer1::PaddingMode paddingMode);

// Helper function to get default ONNX activation alpha values
float getActivationDefaultAlpha(nvinfer1::ActivationType type);

// Helper function to get default ONNX activation beta values
float getActivationDefaultBeta(nvinfer1::ActivationType type);

// Helper function to get the length of the specified axis
nvinfer1::ITensor* getAxisLength(
    ImporterContext* ctx, nvinfer1::ITensor* inpTensor, int32_t axis, nvinfer1::Dims shape = nvinfer1::Dims{0});

// Helper function to return the result tensor from an elementwise layer with nullptr checking.
nvinfer1::ITensor* getElementWiseResult(
    ImporterContext* ctx, nvinfer1::ITensor& lhs, nvinfer1::ITensor& rhs, nvinfer1::ElementWiseOperation op);

// Helper function to return the result tensor from an unary layer with nullptr checking.
nvinfer1::ITensor* getUnaryResult(ImporterContext* ctx, nvinfer1::ITensor& input, nvinfer1::UnaryOperation op);

// Helper function to get kernel attributes for various ONNX nodes
void getKernelParams(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::Dims* kernelSize,
    nvinfer1::Dims* strides, nvinfer1::Dims* begPadding, nvinfer1::Dims* endPadding, nvinfer1::PaddingMode& paddingMode,
    bool& countExcludePadding, nvinfer1::Dims* dilations = nullptr, nvinfer1::Dims* outputPadding = nullptr,
    bool const poolingCeilMode = false);

// Helper function to get the scaling mode for TRT's scale layer
nvinfer1::ScaleMode getScaleMode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape);

// Helper function to get a float representation of weights containing a single value.
float getSingleValueAsFloat(ShapedWeights const& weights);

// Helper function to map ONNX Global Pooling ops into TensorRT.
nvinfer1::ITensor* globalPoolingHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    nvinfer1::ITensor& tensor, nvinfer1::ReduceOperation op);

// Helper function to create a greaterOrEqual or lessOrEqual operation. Provide `greater=true` for greaterOrEqual,
// `greater=false` for lessOrEqual
NodeOutputs greaterLessOrEqual(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    nvinfer1::ITensor* inputA, nvinfer1::ITensor* inputB, bool greater);

// Helper function to determine if a shape contains dynamic dimensions
bool isDynamic(nvinfer1::Dims const& shape);

// Helper function to create an iota fill given a set of dimensions and an axis
nvinfer1::ITensor* iota(ImporterContext* ctx, ShapeTensor iotaDims, int32_t axis);

// Helper function to load a creator from the registry
nvinfer1::IPluginCreatorInterface* importPluginCreator(ImporterContext* ctx, std::string const& pluginName,
    std::string const& pluginVersion, std::string const& pluginNamespace = kTRT_STD_PLUGIN_NAMESPACE);

// Helper function to use modulatedDeformableConv2D plugin
NodeOutputs modulatedDeformableConvPluginHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    size_t const nodeIdx, std::vector<TensorOrWeights>& inputs);

// Helper function to use optimized 3D instanceNorm plugin
NodeOutputs instanceNormPluginHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    size_t const nodeIdx, std::vector<TensorOrWeights>& inputs);

// Helper function to get a plugin from the PluginRegistry
std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(ImporterContext* ctx,
    ::ONNX_NAMESPACE::NodeProto const& node, std::string const& name, std::string const& pluginNamespace,
    nvinfer1::IPluginCreator* pluginCreator, std::vector<nvinfer1::PluginField> const& pluginFields);

// Helper function to get a V3 plugin from the PluginRegistry
std::unique_ptr<nvinfer1::IPluginV3> createPlugin(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::string const& name, std::string const& pluginNamespace, nvinfer1::IPluginCreatorInterface* pluginCreator,
    std::vector<nvinfer1::PluginField> const& pluginFields);

// Helper function to return the identity of a TensorOrWeights
TensorOrWeights identity(ImporterContext* ctx, TensorOrWeights input);

// Helper function to create normalization layers for GroupNorm and InstanceNorm
NodeOutputs normalizationHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, bool const useV2);

// Given a list of axes in the range of [-rank, rank-1], where rank is the rank
// of the corresponding data tensor, normalize to [0, rank-1].
void normalizeAxes(ShapeTensor& axes, int32_t const rank);

// Helper function to parse activation values for LSTM nodes
std::vector<float> parseLSTMActivationValues(std::vector<nvinfer1::ActivationType> const& activationTypes,
    std::vector<float> const& activationValues, bool isAlpha);

// Helper function to map various ONNX pooling ops into TensorRT.
NodeOutputs poolingHelper(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type);

// Helper function to check if reduce op equals No-op
bool IsReduceNoOp(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights> const& inputs);

// Helper function to import reduce ops into TRT
NodeOutputs reduceTensor(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    TensorOrWeights input, nvinfer1::ReduceOperation operation, TensorOrWeights inputAxes = TensorOrWeights());

// Helper function to shape a Tensor given a new shape
nvinfer1::ITensor* reshapeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape);

// Helper function to map attributes to a TRT scale layer
NodeOutputs scaleHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode, nvinfer1::Weights const& shift,
    nvinfer1::Weights const& scale, nvinfer1::Weights const& power, char const* shiftName, char const* scaleName);

// Helper function to set an ONNX attribute
void setAttr(nvinfer1::Dims* trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int32_t nbSpatialDims,
    int32_t defaultVal);

// Helper function to slice away elements on a given axis dimension
nvinfer1::ITensor* sliceAcrossAxis(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* data, int32_t const axis);

// Helper function to squeeze a tensor on a given set of axes
nvinfer1::ITensor* squeezeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, std::vector<int32_t> const& axes);

// Helper function to transpose a tensor given a permutation
nvinfer1::ITensor* transposeTensor(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm);

::ONNX_NAMESPACE::TensorProto_DataType trtDataTypeToONNX(nvinfer1::DataType dt);

// Helper function to import ONNX unary ops into TRT
NodeOutputs unaryHelper(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    TensorOrWeights& input, nvinfer1::UnaryOperation op);

// Helper function to unsqueeze tensors on a given set of axes
nvinfer1::ITensor* unsqueezeTensor(ImporterContext* ctx, nvinfer1::ITensor& tensor, std::vector<int32_t> const& axes);

// Helper function to calculate and return the expected output shape of a resize given the resize scale weights or scale
// tensor.
nvinfer1::ITensor* resizeShapeTensor(ImporterContext* ctx, nvinfer1::ITensor& input, TensorOrWeights& scales);

// Helper function to convert a ShapedWeights object into a vector
template <typename WeightType>
void weightsToVector(TensorOrWeights weights, std::vector<WeightType>* weightVector)
{
    ONNXTRT_CHECK(weights.is_weights(), "Cannot convert a Tensor to a vector!", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights const sWeights = weights.weights();
    ONNXTRT_CHECK((sWeights.type == ::ONNX_NAMESPACE::TensorProto::INT32)
            || (sWeights.type == ::ONNX_NAMESPACE::TensorProto::INT64)
            || (sWeights.type == ::ONNX_NAMESPACE::TensorProto::BOOL)
            || (sWeights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT)
            || (sWeights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT16),
        "Found invalid weights type of: " << sWeights.type,
        ErrorCode::kINVALID_NODE);
    weightVector->resize(sWeights.count());
    if (sWeights.type == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        auto array_start = static_cast<int64_t*>(sWeights.values);
        std::copy(array_start, array_start + sWeights.count(), weightVector->begin());
    }
    else if (sWeights.type == ::ONNX_NAMESPACE::TensorProto::INT32)
    {
        auto array_start = static_cast<int32_t*>(sWeights.values);
        std::copy(array_start, array_start + sWeights.count(), weightVector->begin());
    }
    else if (sWeights.type == ::ONNX_NAMESPACE::TensorProto::BOOL)
    {
        auto array_start = static_cast<bool*>(sWeights.values);
        std::copy(array_start, array_start + sWeights.count(), weightVector->begin());
    }
    else if (sWeights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT)
    {
        auto array_start = static_cast<float*>(sWeights.values);
        std::copy(array_start, array_start + sWeights.count(), weightVector->begin());
    }
    else if (sWeights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT16)
    {
        auto array_start = static_cast<half_float::half*>(sWeights.values);
        std::copy(array_start, array_start + sWeights.count(), weightVector->begin());
    }
}

NodeOutputs staticSliceImporter(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ITensor& data);

// Helper function to convert ONNX node name. If no node name, using name of first output.
std::string const getNodeName(::ONNX_NAMESPACE::NodeProto const& node);

//! Decode in place the starts and ends indices according to ONNX Slice rules.
void decodeOnnxStartsAndEnds(ImporterContext* ctx, ShapeTensor const& inputDims, ShapeTensor const& steps,
    ShapeTensor& starts, ShapeTensor& ends);

//! Return ShapeTensor representing size of result of Slice.
//! starts and ends should first be decoded by decodeOnnxStartsAndEnds.
ShapeTensor computeSliceSizes(ImporterContext* ctx, ShapeTensor const& starts, ShapeTensor const& ends,
    ShapeTensor const& steps, ShapeTensor const& dims);

//! Return subscripts such that gather(concat(x,y),subscripts)
//! will return x with x[subcripts[i]] replaced by y[i].
ShapeTensor axesToInterlaceSubscripts(ShapeTensor const& axes, int nbDims);

//! Helper function to add SoftMax layer.
nvinfer1::ITensor* addSoftmax(
    ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx, nvinfer1::ITensor& input);

//! Helper function to import ONNX scatter nodes into TRT
NodeOutputs addScatterLayer(ImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ScatterMode mode, int32_t axis = 0);

//! Helper function to calculate mod(A, B), A & B are integers
nvinfer1::IElementWiseLayer* modWithIntegerInputs(
    ImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, bool fmod);

//! Helper function to calculate mod(A, B), A & B are floating point numbers
nvinfer1::IElementWiseLayer* modWithFPInputs(ImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1,
    nvinfer1::ITensor* divResult, bool sameSign);

//! Helper function to get the number of dimensions of a tensor.
int32_t getNbDims(nvinfer1::ITensor const* tensor);

//! RAII wrapper for ImporterContext::pushBaseNameScope() and popBaseNameScope().
class NameScope
{
public:
    NameScope(ImporterContext& context)
        : mContext(context)
    {
        mContext.pushBaseNameScope();
    }
    ~NameScope()
    {
        mContext.popBaseNameScope();
    }

private:
    ImporterContext& mContext;
};

// Helper function to validate input types for an ONNX node
Status notInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx);

void checkNotInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes,
    ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx);

// Helper function to truncate string to length limit
std::string truncateString(std::string const& s, int64_t limit);

void processMetadata(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ILayer* layer);

void processMetadata(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::IAttention* attention);

// Helper function to convert TensorRT datatype enum into a human-readable string.
std::string getTrtDtypeName(nvinfer1::DataType TrtDtype);

// Helper function to generate a Window tensor for Window operations (HannWindow, HammingWindow, BlackmanWindow).
nvinfer1::ITensor* generateWindow(ImporterContext* ctx, nvinfer1::ITensor* N);

// Helper function to handle Window generation ops. Calculates TrigOp(numerator*n / N) and returns the output tensor.
nvinfer1::ITensor* windowHelper(ImporterContext* ctx, float numerator, nvinfer1::ITensor* n, nvinfer1::ITensor* N,
    nvinfer1::UnaryOperation op, int32_t periodic);

//! Describes occurrence of a named dimension.
class NamedDimension
{
public:
    //! TensorRT tensor.
    nvinfer1::ITensor* tensor;

    //! Index of tensor dimension to be named.
    int32_t index;

    //! ONNX "dim param" that is the name of the dimension.
    std::string dimParam;

    //! Construct a NamedDimension where the tensor will be filled in later.
    NamedDimension(int32_t index_, std::string const& dimParam_)
        : tensor(nullptr)
        , index(index_)
        , dimParam(dimParam_)
    {
    }
};

template <typename OnnxDims>
bool convertOnnxDims(OnnxDims const& onnxDims, nvinfer1::Dims& trtDims, std::vector<NamedDimension>& namedDims)
{
    if (onnxDims.size() > nvinfer1::Dims::MAX_DIMS)
    {
        return false;
    }
    std::vector<int32_t> onnxDimsVec;
    for (auto const& onnxDim : onnxDims)
    {
        // For empty dimensions, the ONNX specification says it's a dynamic dimension
        if (!onnxDim.has_dim_value() && !onnxDim.has_dim_param())
        {
            onnxDimsVec.emplace_back(-1);
        }
        else
        {
            if (!onnxDim.dim_param().empty())
            {
                namedDims.emplace_back(static_cast<int32_t>(onnxDimsVec.size()), onnxDim.dim_param());
            }
            const int32_t dim = onnxDim.dim_param() == "" ? (onnxDim.dim_value() >= 0 ? onnxDim.dim_value() : -1) : -1;
            onnxDimsVec.emplace_back(dim);
        }
    }
    trtDims.nbDims = onnxDimsVec.size();
    std::copy(onnxDimsVec.begin(), onnxDimsVec.end(), trtDims.d);
    return true;
}

//! Helper enum for TRT plugin creator versions
enum class CreatorVersion : int32_t
{
    kV1,
    kV3ONE,
    kV3QUICK,
};

//! Evaluate CreatorVersion given a pointer to a TensorRT plugin creator
CreatorVersion getPluginCreatorVersion(nvinfer1::IPluginCreatorInterface const* pluginCreator);

} // namespace onnx2trt
