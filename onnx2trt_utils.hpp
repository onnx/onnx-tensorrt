/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ShapedWeights.hpp"
#include "ShapeTensor.hpp"
#include "Status.hpp"
#include "trt_utils.hpp"

#include <NvInfer.h>
#include <onnx/onnx_pb.h>

#include "half.h"
#include <cstring> // For std::memcpy
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <typeindex>
#include <unordered_map>

#define LOG(msg, severity)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)                                                         \
            ss << __FILENAME__ << ":" << __LINE__ << ": ";                                                             \
        ss << msg;                                                                                                     \
        ctx->logger().log(severity, ss.str().c_str());                                                                 \
    } while (0)

#define LOG_VERBOSE(msg) LOG(msg, nvinfer1::ILogger::Severity::kVERBOSE)
#define LOG_INFO(msg) LOG(msg, nvinfer1::ILogger::Severity::kINFO)
#define LOG_WARNING(msg) LOG(msg, nvinfer1::ILogger::Severity::kWARNING)
#define LOG_ERROR(msg) LOG(msg, nvinfer1::ILogger::Severity::kERROR)

// Overloads of operator<< on TensorRT types must be defined inside nvinfer1
// so that argument-dependent lookup works as expected. Declared static to
// avoid symbol clashing when statically linking with other TensorRT libraries
namespace nvinfer1
{

template <typename T>
static std::ostream& printSequence(std::ostream& stream, const T* begin, int count)
{
    stream << "(";
    if (count > 0)
    {
        std::copy_n(begin, count - 1, std::ostream_iterator<T>(stream, ", "));
        stream << begin[count - 1];
    }
    stream << ")";
    return stream;
}

static std::ostream& operator<<(std::ostream& stream, const nvinfer1::Dims& shape)
{
    return printSequence(stream, shape.d, shape.nbDims);
}

static std::ostream& operator<<(std::ostream& stream, const nvinfer1::Permutation& perm)
{
    return printSequence(stream, perm.order, nvinfer1::Dims::MAX_DIMS);
}

static std::ostream& operator<<(std::ostream& stream, const nvinfer1::DataType& dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return stream << "float32";
    case nvinfer1::DataType::kHALF: return stream << "float16";
    case nvinfer1::DataType::kINT8: return stream << "int8";
    case nvinfer1::DataType::kUINT8: return stream << "uint8";
    case nvinfer1::DataType::kINT32: return stream << "int32";
    case nvinfer1::DataType::kBOOL: return stream << "bool";
    default: throw std::runtime_error("Unknown dtype");
    }
}

} // namespace nvinfer1

namespace onnx2trt
{

struct PluginDeleter
{
    void operator()(nvinfer1::IPluginV2* t);
};

// Helper function to calculate the volume of a Dims object
int64_t volume(const nvinfer1::Dims& dims);

// Helper function to get the size in bytes of an ONNX datatype
int getDtypeSize(int32_t onnxDtype);

// Helper function to add a scalar into TRT through a constant layer.
template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstantScalar(
    IImporterContext* ctx, ScalarType scalar, ShapedWeights::DataType type, nvinfer1::Dims shape = nvinfer1::Dims{0})
{
    assert(volume(shape) == 1 && "Cannot add constant scalar with a shape that has volume > 1");
    ShapedWeights scalarWeights = ctx->createTempWeights(type, shape);
    static_cast<ScalarType*>(scalarWeights.values)[0] = static_cast<ScalarType>(scalar);
    nvinfer1::IConstantLayer* l = ctx->network()->addConstant(scalarWeights.shape, scalarWeights);
    ctx->network()->setWeightsName(scalarWeights, scalarWeights.getName());
    return l;
}

// Helper function to create a tensor given a vector of values and a shape.
template <typename ScalarType>
inline nvinfer1::IConstantLayer* addConstant(
    IImporterContext* ctx, const std::vector<ScalarType>& values, ShapedWeights::DataType type, nvinfer1::Dims shape)
{
    assert(volume(shape) == static_cast<int64_t>(values.size()) && "Shape does not match number of values provided");
    assert(sizeof(ScalarType) == getDtypeSize(type) && "ONNX dtype does not have the same size as the value type");
    ShapedWeights weights = ctx->createTempWeights(type, shape);
    std::memcpy(weights.values, values.data(), values.size() * sizeof(ScalarType));
    nvinfer1::IConstantLayer* l = ctx->network()->addConstant(weights.shape, weights);
    ctx->network()->setWeightsName(weights, weights.getName());
    return l;
}

enum ScaleOp
{
    kSHIFT,
    kSCALE,
    kPOWER,
};

// Helper function to import ONNX activation nodes into TRT
NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha = nullptr, float* beta = nullptr);

// Add clipping to a tensor if clip is a valid value.
nvinfer1::ITensor* addClip(IImporterContext* ctx, nvinfer1::ITensor* input, float clip);

// Helper function to import ArgMax and ArgMin nodes into TRT
NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op);

//! If t has rank less than nbDims, reshape it to have nbDims by prepending ones to its dimensions.
//! Assert failure if t has rank greater than nbDims.
Status broadcastTensor(IImporterContext* ctx, nvinfer1::ITensor*& t, const int nbDims);

// Helper function to broadcast two tensors to the larger one's shape
Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

// Helper function to broadcast three tensors to the largest one's shape
Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3);

// Helper funtion to check that two shapes conform to the broadcasting rules
Status isBroadcastValid(IImporterContext* ctx, const nvinfer1::Dims& firstShape, const nvinfer1::Dims& secondShape);

// Helper function to calculate the bias tensor for GatherElements.
std::vector<int32_t> calculateBias(
    const nvinfer1::Dims& daDims, const nvinfer1::Dims& idxDims, const std::vector<int32_t>& pitches, int32_t axis);

// Helper function to check that linear/cubic resize can be used
bool canUseNDResize(size_t const scaleSize, float const* scaleFactors, size_t const n);

// Helper function to calculate and return a vector representation of the pitches of a given shape
std::vector<int32_t> calculatePitches(const nvinfer1::Dims& inputDims);

// Helper function to add a Cast layer in the network
nvinfer1::ITensor* castHelper(IImporterContext* ctx, nvinfer1::ITensor* input, nvinfer1::DataType dtype);

// Helper function for constantOfShape operator. Input shape must be a shape tensor
nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor* constant, nvinfer1::ITensor* shape);

// Helper function to convert an ONNX axis into a TRT axis
Status convertAxis(int& axis, int nbDims);

// Helper function to convert an ONNX datatype into a TRT datatype
bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert INT64 weight values into INT32
int32_t* convertINT64(const int64_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx);

// Helper function to convert ONNX padding into TRT padding. Will update startTensor and totalPaddingTensor by reference
bool convertOnnxPadding(IImporterContext* ctx, int32_t nbInputDims, const std::vector<int32_t>& onnxPadding,
    nvinfer1::ITensor*& startTensor, nvinfer1::ITensor*& totalPaddingTensor);

// Helper function to check if all of the values in the shift tensor are zeros
bool shiftIsAllZeros(const ShapedWeights& shiftInt8);

// Helper function to create zero shifts for QuantizeLinear/DequantizeLinear ops
onnx2trt::ShapedWeights createZeroShifts(const onnx2trt::ShapedWeights& shiftInt8, int32_t type, IImporterContext* ctx);

// Helper function to create a tensor of all zeros with the same shape as a data tensor
nvinfer1::ITensor* createZeroTensor(IImporterContext* ctx, nvinfer1::ITensor* data);

// Helper function to convert an ONNX weight into a ShapedWeights object
bool convertOnnxWeights(
    const ::ONNX_NAMESPACE::TensorProto& onnxTensor, onnx2trt::ShapedWeights* weights, IImporterContext* ctx);

// Helper function to convert multi input convolution
NodeImportResult convMultiInput(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, std::vector<TensorOrWeights>& inputs);

// Helper function to convert a 1D tensor into a scalar
nvinfer1::ITensor* convertToScalar(IImporterContext* ctx, nvinfer1::ITensor* inpTensor);

// Helper function to convert a ShapedWeights object into a tensor
nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx);

// Helper function to convert a ShapedWeights object into a scalar
nvinfer1::ITensor* convertToScalar(TensorOrWeights& input, IImporterContext* ctx);

// Helper function to provide a ceiling-rounding division between two integers
int divCeil(int n, int d);

// Helper function to check that the input data types for an elementwise operation are supported
bool elementwiseCheck(const std::vector<TensorOrWeights>& inputs, const nvinfer1::ElementWiseOperation op);

// Helper function to import an ONNX elementwise op into TRT
NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    const std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op);

// Helper function to flatten a tensor on a given axis
nvinfer1::ITensor* flattenTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ITensor& tensor, int axis = 0, bool regLayer = false);

// Gathers the specified dimension from a shape tensor. e.g. gatherDimension(shape=(7, 6, 5), dim=2) would return 5.
// shape specifies the shape of the returned Tensor. Must have a volume of 1.
nvinfer1::ITensor* gatherDimension(
    IImporterContext* ctx, nvinfer1::ITensor* shapeTensor, int dim, nvinfer1::Dims shape);

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims inputShape, nvinfer1::Dims outputShape, nvinfer1::Dims kernelSize,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& begPadding,
    nvinfer1::Dims& endPadding, nvinfer1::Dims& outputPadding, nvinfer1::PaddingMode paddingMode);

// Helper function to get default ONNX activation alpha values
float getActivationDefaultAlpha(nvinfer1::ActivationType type);

// Helper function to get default ONNX activation beta values
float getActivationDefaultBeta(nvinfer1::ActivationType type);

// Helper function to get the length of the specified axis
nvinfer1::ITensor* getAxisLength(
    IImporterContext* ctx, nvinfer1::ITensor* inpTensor, int axis, nvinfer1::Dims shape = nvinfer1::Dims{0});

// Helper function to calculate the output size of a convolution node given its attributes
int getConvOutputSize(int input_size, int filter_size, int stride, int dilation_rate, int total_padding);

// Helper function to get the TRT datatype given an ONNX datatype
const char* getDtypeName(int32_t onnxDtype);

// Helper function to get kernel attributes for various ONNX nodes
void getKernelParams(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& onnx_node, nvinfer1::Dims* kernel_size,
    nvinfer1::Dims* strides, nvinfer1::Dims* beg_padding, nvinfer1::Dims* end_padding,
    nvinfer1::PaddingMode& paddingMode, bool& count_exclude_padding, nvinfer1::Dims* dilations = nullptr,
    nvinfer1::Dims* output_padding = nullptr, const bool poolingCeilMode = false);

// Helper function to get the scaling mode for TRT's scale layer
nvinfer1::ScaleMode getScaleMode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape);

// Helper function to get a float representation of a FLOAT or FLOAT16 ONNX type.
float getSingleValueAsFloat(void* data, bool fp16);

// Helper function to map ONNX Global Pooling ops into TensorRT.
nvinfer1::ITensor* globalPoolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, nvinfer1::ITensor& tensor, nvinfer1::ReduceOperation op);

// Helper function to create a greaterOrEqual or lessOrEqual operation. Provide `greater=true` for greaterOrEqual, `greater=false` for lessOrEqual
nvinfer1::ITensor* greaterLessOrEqual(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* inputA, nvinfer1::ITensor* inputB,
    bool greater);

// Helper function to determine if a shape contains dynamic dimensions
bool isDynamic(const nvinfer1::Dims& shape);

// Helper fucntion to create an iota fill given a set of dimensions and an axis
nvinfer1::ITensor* iota(IImporterContext* ctx, ShapeTensor iotaDims, int32_t axis);

// Helper function to load a creator from the registry
nvinfer1::IPluginCreator* importPluginCreator(
    const std::string& pluginName, const std::string& pluginVersion, const std::string& pluginNamespace = "");

// Helper function to get a plugin from the PluginRegistry
std::unique_ptr<nvinfer1::IPluginV2, PluginDeleter> createPlugin(const std::string& name,
    nvinfer1::IPluginCreator* pluginCreator, const std::vector<nvinfer1::PluginField>& pluginFields);

// Helper function to determine if a transpose is required
bool isTransposeRequired(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm);

// Helper function to import LSTM ops through the legacy CUDNN path
NodeImportResult lstmLegacyImporter(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);

// Helper function to create and fill a Dims object with defined values
nvinfer1::Dims makeDims(int nbDims, int val);

// Helper function to parse activation values for LSTM nodes
std::vector<float> parseLSTMActivationValues(const std::vector<nvinfer1::ActivationType>& activationTypes,
    const std::vector<float>& activationValues, bool isAlpha);

// Helper function to read weights from an external file
bool parseExternalWeights(IImporterContext* ctx, std::string file, std::string path, int64_t offset, int64_t length,
    std::vector<char>& weightsBuf, size_t& size);

// Helper function to map various ONNX pooling ops into TensorRT.
NodeImportResult poolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type);

// Helper function to import reduce ops into TRT
NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation, TensorOrWeights inputAxes = TensorOrWeights());

// Helper function to shape a Tensor given a new shape
nvinfer1::ITensor* reshapeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape);

// Helper function to map attributes to a TRT scale layer
NodeImportResult scaleHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor& tensor_,
    nvinfer1::ScaleMode mode, const nvinfer1::Weights& shift, const nvinfer1::Weights& scale,
    const nvinfer1::Weights& power, const char* shiftName, const char* scaleName);

// Helper function to set an ONNX attribute
void setAttr(
    nvinfer1::Dims* trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal);

// Helper function to slice away elements on a given axis dimension
nvinfer1::ITensor* sliceAcrossAxis(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor* data, const int axis);

// Helper function to squeeze a tensor on a given set of axes
nvinfer1::ITensor* squeezeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor& tensor, const std::vector<int>& axes, bool regLayer = false);

// Helper function to transpose a tensor given a permutation
nvinfer1::ITensor* transposeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm);

// Helper function to import ONNX unary ops into TRT
NodeImportResult unaryHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, TensorOrWeights& input,
    nvinfer1::UnaryOperation op);

// Helper function to unsqueeze tensors on a given set of axes
nvinfer1::ITensor* unsqueezeTensor(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    nvinfer1::ITensor& tensor, const std::vector<int>& axes, bool regLayer = false);

// Helper function to calculate and return the expected output shape of a resize given the resize scale weights or scale
// tensor.
nvinfer1::ITensor* resizeShapeTensor(IImporterContext* ctx, nvinfer1::ITensor& input, TensorOrWeights& scales);

// Helper function to convert a ShapedWeights object into a vector
template <typename WeightType>
Status weightsToVector(TensorOrWeights weights, std::vector<WeightType>* weightVector)
{
    ASSERT(weights.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT((weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32)
            || (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
            || (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::BOOL),
        ErrorCode::kINVALID_NODE);
    weightVector->resize(weights.weights().count());
    if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        auto array_start = static_cast<int64_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    else if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32)
    {
        auto array_start = static_cast<int32_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    else if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::BOOL)
    {
        auto array_start = static_cast<bool*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    return Status(ErrorCode::kSUCCESS);
}

template <typename T>
ShapedWeights::DataType getShapeWeightsDataType()
{
    static const std::unordered_map<std::type_index, ::ONNX_NAMESPACE::TensorProto::DataType> tMap(
        {{std::type_index(typeid(bool)), ::ONNX_NAMESPACE::TensorProto::BOOL},
            {std::type_index(typeid(int8_t)), ::ONNX_NAMESPACE::TensorProto::INT8},
            {std::type_index(typeid(uint8_t)), ::ONNX_NAMESPACE::TensorProto::UINT8},
            {std::type_index(typeid(int16_t)), ::ONNX_NAMESPACE::TensorProto::INT16},
            {std::type_index(typeid(uint16_t)), ::ONNX_NAMESPACE::TensorProto::UINT16},
            {std::type_index(typeid(int32_t)), ::ONNX_NAMESPACE::TensorProto::INT32},
            {std::type_index(typeid(uint32_t)), ::ONNX_NAMESPACE::TensorProto::UINT32},
            {std::type_index(typeid(int64_t)), ::ONNX_NAMESPACE::TensorProto::INT64},
            {std::type_index(typeid(uint64_t)), ::ONNX_NAMESPACE::TensorProto::UINT64},
            {std::type_index(typeid(float)), ::ONNX_NAMESPACE::TensorProto::FLOAT},
            {std::type_index(typeid(double)), ::ONNX_NAMESPACE::TensorProto::DOUBLE}});

    if (tMap.find(std::type_index(typeid(T))) != tMap.end())
    {
        return tMap.at(std::type_index(typeid(T)));
    }
    return ::ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

// Helper function to convert a vector object into a ShapedWeights object
template <typename T>
Status vectorToWeights(std::vector<T>& weightVector, TensorOrWeights* weights)
{
    nvinfer1::Dims shape{1, {static_cast<int32_t>(weightVector.size())}};
    ShapedWeights::DataType dtype = getShapeWeightsDataType<T>();
    ASSERT(dtype != ::ONNX_NAMESPACE::TensorProto::UNDEFINED && "Unsupported datatype", ErrorCode::kINVALID_VALUE);
    *weights = ShapedWeights(dtype, weightVector.data(), shape);
    return Status(ErrorCode::kSUCCESS);
}

// Helper function to convert ONNX node name. If no node name, using name of first output.
const std::string getNodeName(const ::ONNX_NAMESPACE::NodeProto& node);

//! Decode in place the starts and ends indices according to ONNX Slice rules.
void decodeOnnxStartsAndEnds(IImporterContext* ctx, const ShapeTensor& inputDims, const ShapeTensor& steps, ShapeTensor& starts, ShapeTensor& ends);

//! Return ShapeTensor representing size of result of Slice.
//! starts and ends should first be decoded by decodeOnnxStartsAndEnds.
ShapeTensor computeSliceSizes(IImporterContext* ctx, const ShapeTensor& starts, const ShapeTensor& ends,
    const ShapeTensor& steps, const ShapeTensor& dims);

//! Return subscripts such that gather(concat(x,y),subscripts)
//! will return x with x[subcripts[i]] replaced by y[i].
ShapeTensor axesToInterlaceSubscripts(const ShapeTensor& axes, int nbDims);

//! Helper function to add SoftMax layer.
nvinfer1::ITensor* addSoftmax(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ITensor& input);

//! Helper function to import ONNX scatter nodes into TRT
NodeImportResult addScatterLayer(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, std::vector<TensorOrWeights>& inputs, nvinfer1::ScatterMode mode, int32_t axis = 0);

//! Helper function to calculate mod(A, B), A & B are integers
nvinfer1::IElementWiseLayer* modWithIntegerInputs(IImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, bool fmod);

//! Helper function to calculate mod(A, B), A & B are floating point numbers
nvinfer1::IElementWiseLayer* modWithFPInputs(IImporterContext* ctx, nvinfer1::ITensor* input0, nvinfer1::ITensor* input1, nvinfer1::ITensor* divResult, bool sameSign);

//! RAII wrapper for IImporterContext::pushBaseNameScope() and popBaseNameScope().
class NameScope
{
public:
    NameScope(IImporterContext& context)
        : mContext(context)
    {
        mContext.pushBaseNameScope();
    }
    ~NameScope()
    {
        mContext.popBaseNameScope();
    }
private:
    IImporterContext& mContext;
};

// Helper function to convert weightValues' type from fp16 to fp32
float* convertFP16Data(void* weightValues, nvinfer1::Dims shape, IImporterContext* ctx);

// Helper function to validate input types for an ONNX node
Status notInvalidType(TensorOrWeights const& input, std::vector<std::string> const& invalidTypes);

} // namespace onnx2trt
