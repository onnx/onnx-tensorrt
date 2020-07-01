/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "trt_utils.hpp"

#include <NvInfer.h>
#include <onnx/onnx_pb.h>
#include <onnx/onnxifi.h>

#include <cstring> // For std::memcpy
#include <iostream>
#include <numeric>
#include <sstream>

#define LOG(msg, severity)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        ss << __FILENAME__ << ":" << __LINE__ << ": " << msg;                                                          \
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
    case nvinfer1::DataType::kINT32: return stream << "int32";
    case nvinfer1::DataType::kBOOL: return stream << "bool";
    default: throw std::runtime_error("Unknown dtype");
    }
}

} // namespace nvinfer1

namespace onnx2trt
{

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
    return ctx->network()->addConstant(scalarWeights.shape, scalarWeights);
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
    return ctx->network()->addConstant(weights.shape, weights);
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

// Helper function to broadcast two tensors to the larger one's shape
Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

// Helper function to broadcast three tensors to the largest one's shape
Status broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2, nvinfer1::ITensor*& t3);

// Helper function to check that linear resize can be used
bool canUseLinearResize(const size_t scaleSize, const float* scaleFactors);

// Helper function for constantOfShape operator. Input shape must be a shape tensor
nvinfer1::ITensor* constantOfShape(IImporterContext* ctx, nvinfer1::ITensor* constant, nvinfer1::ITensor* shape);

// Helper function to convert an ONNX axis into a TRT axis
Status convertAxis(int& axis, int nbDims);

// Helper function to convert an ONNX datatype into a TRT datatype
bool convertDtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert INT64 weight values into INT32
int32_t* convertINT64(const int64_t* weightValues, nvinfer1::Dims shape, IImporterContext* ctx);

// Helper function to convert ONNX padding into TRT padding
bool convertOnnxPadding(
    const std::vector<int64_t>& onnxPadding, nvinfer1::Dims2* begPadding, nvinfer1::Dims2* endPadding);

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

// Helper function to convert an ONNX weight descriptor into a ShapedWeights object
bool convertWeightDescriptor(
    onnxTensorDescriptorV1 const& desc, onnx2trt::ShapedWeights* weights, IImporterContext* ctx);

// Helper function to provide a ceiling-rounding division between two integers
int divCeil(int n, int d);

// Helper function to check that the input data types for an elementwise operation are supported
bool elementwiseCheck(const std::vector<TensorOrWeights>& inputs, const nvinfer1::ElementWiseOperation op);

// Helper function to import an ONNX elementwise op into TRT
NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op);

// Helper function to flatten a tensor on a given axis
nvinfer1::ITensor* flattenTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis = 0);

// Gathers the specified dimension from a shape tensor. e.g. gatherDimension(shape=(7, 6, 5), dim=2) would return 5.
// shape specifies the shape of the returned Tensor. Must have a volume of 1.
nvinfer1::ITensor* gatherDimension(
    IImporterContext* ctx, nvinfer1::ITensor* shapeTensor, int dim, nvinfer1::Dims shape);

// Helper function to generate padding values for convTranspose
void generatePadding(nvinfer1::Dims input_dims, nvinfer1::Dims output_shape, nvinfer1::Dims kernel_size,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& beg_padding,
    nvinfer1::Dims& end_padding, nvinfer1::Dims& output_padding, nvinfer1::PaddingMode paddingMode);

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

// Helper function to map ONNX Global Pooling ops into TensorRT.
nvinfer1::ITensor* globalPoolingHelper(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::ReduceOperation op);

// Helper function to get a plugin from the PluginRegistry
nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName,
    const std::vector<nvinfer1::PluginField>& pluginFields);

// Helper function to determine if a shape contains dynamic dimensions
bool isDynamic(const nvinfer1::Dims& shape);

// Helper function to determine if a ONNX tensor is empty
bool isOnnxTensorEmpty(const ::ONNX_NAMESPACE::TensorProto& onnxTensor);

// Helper function to determine if a transpose is required
bool isTransposeRequired(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm);

// Helper function to import LSTM ops through the legacy CUDNN path
NodeImportResult lstmLegacyImporter(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);

// Helper function to create and fill a Dims object with defined values
nvinfer1::Dims makeDims(int nbDims, int val);

// Helper function to map various ONNX pooling ops into TensorRT.
NodeImportResult poolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type);

// Helper function to import reduce ops into TRT
NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation);

// Helper function to shape a Tensor given a new shape
nvinfer1::ITensor* reshapeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape);

// Helper function to map attributes to a TRT scale layer
NodeImportResult scaleHelper(IImporterContext* ctx, nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode,
    nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power);

// Helper function to set an ONNX attribute
void setAttr(
    nvinfer1::Dims* trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal);

// Helper function to squeeze a tensor on a given set of axes
nvinfer1::ITensor* squeezeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes);

// Helper function to transpose a tensor given a permutation
nvinfer1::ITensor* transposeTensor(
    IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm, bool permute_dim_types = true);

// Helper function to filter out shape tensor outputs for layers that do not support it
bool supportsShapeTensor(nvinfer1::LayerType type, nvinfer1::ElementWiseOperation eleOp, nvinfer1::ReduceOperation redOp);

// Helper function to import ONNX unary ops into TRT
NodeImportResult unaryHelper(IImporterContext* ctx, TensorOrWeights& input, nvinfer1::UnaryOperation op);

// Helper function to unsqueeze tensors on a given set of axes
nvinfer1::ITensor* unsqueezeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes);

// Helper function to convert a ShapedWeights object into a vector
Status weightsToVector(TensorOrWeights weights, std::vector<int64_t>* weightVector);

} // namespace onnx2trt
