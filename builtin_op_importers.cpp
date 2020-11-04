/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "builtin_op_importers.hpp"
#include "ModelImporter.hpp"
#include "NvInferPlugin.h"
#include "OnnxAttrs.hpp"
#include "ShapeTensor.hpp"
#include "onnx2trt_utils.hpp"
#include "LoopHelpers.hpp"
#include "RNNHelpers.hpp"

#include <algorithm> // For std::min, std::max
#include <array>
#include <cmath>
#include <cstring> // For std::memcpy, std::memset
#include <iterator>
#include <numeric> // For std::iota
#include <tuple>
#include <unordered_set>
#include <iostream>

namespace onnx2trt
{

string_map<NodeImporter>& getBuiltinOpImporterMap()
{
    static string_map<NodeImporter> builtin_op_importers;
    return builtin_op_importers;
}

namespace
{

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
    static const bool op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);                                                               \
        return {{layer_ptr->getOutput(0)}};                                                                            \
    } while (0)

#define RETURN_IDENTITY(input)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        TensorOrWeights output = identity(ctx, input);                                                                 \
        ASSERT(output, ErrorCode::kUNSUPPORTED_NODE);                                                                  \
        return {{output}};                                                                                             \
    } while (0)

#define RETURN_ALL_OUTPUTS(layer)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        nvinfer1::ILayer* layer_ptr = layer;                                                                           \
        ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);                                                               \
        std::vector<TensorOrWeights> outputs;                                                                          \
        for (int i = 0; i < layer_ptr->getNbOutputs(); ++i)                                                            \
            outputs.push_back(layer_ptr->getOutput(i));                                                                \
        return {outputs};                                                                                              \
    } while (0)

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
    const int rank = input.getDimensions().nbDims;

    nvinfer1::ITensor* scale = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* bias = &convertToTensor(inputs.at(2), ctx);
    nvinfer1::ITensor* mean = &convertToTensor(inputs.at(3), ctx);
    nvinfer1::ITensor* variance = &convertToTensor(inputs.at(4), ctx);

    const bool hasCDimension = rank > 1;
    if (hasCDimension)
    {
        std::vector<int> axes(rank - 1);
        axes[0] = 0;
        std::iota(axes.begin() + 1, axes.end(), 2);

        scale = unsqueezeTensor(ctx, node, *scale, axes);
        bias = unsqueezeTensor(ctx, node, *bias, axes);
        mean = unsqueezeTensor(ctx, node, *mean, axes);
        variance = unsqueezeTensor(ctx, node,*variance, axes);
    }

    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5f);

    nvinfer1::Dims scalarShape{rank};
    std::fill(scalarShape.d, scalarShape.d + scalarShape.nbDims, 1);
    nvinfer1::ITensor* epsilon
        = addConstantScalar(ctx, eps, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarShape)->getOutput(0);

    // batchnorm = scale * (input - mean) / sqrt(variance + epsilon) + bias
    nvinfer1::IElementWiseLayer* layer = ctx->network()->addElementWise(
        *ctx->network()
             ->addElementWise(*scale,
                 *ctx->network()
                      ->addElementWise(*ctx->network()->addElementWise(input, *mean, eOp::kSUB)->getOutput(0),
                          *ctx->network()
                               ->addUnary(*ctx->network()->addElementWise(*variance, *epsilon, eOp::kSUM)->getOutput(0),
                                   uOp::kSQRT)
                               ->getOutput(0),
                          eOp::kDIV)
                      ->getOutput(0),
                 eOp::kPROD)
             ->getOutput(0),
        *bias, eOp::kSUM);

    ctx->registerLayer(layer, node.name());

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
    // Scale, bias, mean, and variance must be initializers
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    auto mean_weights = inputs.at(3).weights();
    auto variance_weights = inputs.at(4).weights();

    const bool allInputsWeights = inputs.at(1).is_weights() && inputs.at(2).is_weights() && inputs.at(3).is_weights()
        && inputs.at(4).is_weights();
    const bool allWeightsFloat = scale_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
        && bias_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
        && mean_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
        && variance_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT;
    const bool canFoldWeights = allInputsWeights && allWeightsFloat;

    if (!canFoldWeights)
    {
        return batchnormFallback(ctx, node, inputs);
    }

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5f);

    nvinfer1::Dims dims = tensorPtr->getDimensions();

    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }

    // Number of channels is equal to the length of scale_weights.
    int nchan = scale_weights.shape.d[0];
    nvinfer1::Dims weights_shape{1, {nchan}};
    ASSERT(scale_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(bias_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(mean_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    ASSERT(variance_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
    auto combined_scale_weights = ctx->createTempWeights(scale_weights.type, scale_weights.shape);
    auto combined_bias_weights = ctx->createTempWeights(bias_weights.type, bias_weights.shape);
    size_t nweight = nchan;
    // Fold the weights together into a single bias and scale
    for (size_t i = 0; i < nweight; ++i)
    {
        float scale = (static_cast<float const*>(scale_weights.values))[i];
        float bias = (static_cast<float const*>(bias_weights.values))[i];
        float mean = (static_cast<float const*>(mean_weights.values))[i];
        float variance = (static_cast<float const*>(variance_weights.values))[i];
        float& combined_scale_ref = const_cast<float*>(static_cast<float const*>(combined_scale_weights.values))[i];
        float& combined_bias_ref = const_cast<float*>(static_cast<float const*>(combined_bias_weights.values))[i];
        combined_scale_ref = scale / sqrtf(variance + eps);
        combined_bias_ref = bias - mean * combined_scale_ref;
    }

    // If dimensions were not expanded return the output of the scale operation
    if (!needToExpandDims)
    {
        return scaleHelper(
            ctx, node, *tensorPtr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {}, bias_weights.getName(), scale_weights.getName());
    }
    else
    {
        auto scaledResult = scaleHelper(
            ctx, node, *tensorPtr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {}, bias_weights.getName(), scale_weights.getName());
        // Squeeze spatial dims back to 1D
        tensorPtr = &convertToTensor(scaledResult.value().at(0), ctx);
        std::vector<int> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
        return {{tensorPtr}};
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Cast)
{
    // Get input node.
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);
    // Get data type to cast to.
    nvinfer1::DataType dtype = tensor.getType();
    auto onnxType = attrs.get<int32_t>("to");
    ASSERT(convertDtype(onnxType, &dtype) && "Unsupported cast!", ErrorCode::kINVALID_NODE);
    LOG_VERBOSE("Casting to type: " << dtype);
    // Add the layer.
    nvinfer1::IIdentityLayer* layer = ctx->network()->addIdentity(tensor);
    layer->setOutputType(0, dtype);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kCEIL);
}

DEFINE_BUILTIN_OP_IMPORTER(Clip)
{
    OnnxAttrs attrs(node, ctx);
    // beta is the upper bound
    float alpha = std::numeric_limits<float>::lowest();
    float beta = std::numeric_limits<float>::max();

    if (ctx->getOpsetVersion() >= 11)
    {
        int numInputs = inputs.size();
        // Handle "min" node input.
        if (numInputs == 2)
        {
            ASSERT(inputs.at(1).is_weights() && "Clip min value must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
            auto min = inputs.at(1).weights();
            alpha = static_cast<float*>(min.values)[0];
        }
        // Handle both "min" and "max" node inputs
        else if (numInputs == 3)
        {
            // "min" can be optional if "max" is specified. Check for this case here
            if (inputs.at(1))
            {
                ASSERT(inputs.at(1).is_weights() && "Clip min value must be an initializer!",
                    ErrorCode::kUNSUPPORTED_NODE);
                auto min = inputs.at(1).weights();
                alpha = static_cast<float*>(min.values)[0];
            }

            ASSERT(inputs.at(2).is_weights() && "Clip max value must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
            auto max = inputs.at(2).weights();
            beta = static_cast<float*>(max.values)[0];
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
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        // TRT does not support BOOL input types for this node
        ASSERT(!input.isBool(), ErrorCode::kUNSUPPORTED_NODE);
        tensors.push_back(&convertToTensor(input, ctx));
    }
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis");
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    ctx->registerLayer(layer, node.name());
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
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
        RETURN_FIRST_OUTPUT(layer);
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
    ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    if (inputs.at(1).is_tensor())
    {
        ASSERT(ctx->network()->hasExplicitPrecision() && "TensorRT only supports multi-input conv for explicit precision QAT networks!", ErrorCode::kUNSUPPORTED_NODE);
        if (inputs.size() == 3)
        {
            ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        }
        // Handle Multiinput convolution
        return convMultiInput(ctx, node, inputs);
    }

    // Convolution Weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

    auto kernelWeights = inputs.at(1).weights();

    nvinfer1::Dims dims = tensorPtr->getDimensions();
    LOG_VERBOSE("Convolution input dimensions: " << dims);

    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }
    if (kernelWeights.shape.nbDims == 3)
    {
        kernelWeights.shape.nbDims = 4;
        kernelWeights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernelWeights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shapedBiasWeights = inputs.at(2).weights();
        ASSERT(shapedBiasWeights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shapedBiasWeights.shape.d[0] == kernelWeights.shape.d[0], ErrorCode::kINVALID_NODE);
        bias_weights = shapedBiasWeights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernelWeights.type);
    }
    nvinfer1::Dims kernelSize;
    kernelSize.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
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

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernelSize.d[nbSpatialDims - i] == kernelWeights.shape.d[kernelWeights.shape.nbDims - i],
            ErrorCode::kUNSUPPORTED_NODE);
    }

    int nchan = dims.d[1];
    int noutput = kernelWeights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*tensorPtr, noutput, kernelSize, kernelWeights, bias_weights);

    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int ngroup = attrs.get("group", 1);
    ASSERT(nchan == -1 || kernelWeights.shape.d[1] * ngroup == nchan, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    // Register layer name as well as kernel weights and bias weights (if any)
    ctx->registerLayer(layer, getNodeName(node));
    ctx->insertRefitMap(inputs.at(1).weights().getName(), getNodeName(node), nvinfer1::WeightsRole::kKERNEL);
    if (inputs.size() == 3)
    {
        ctx->insertRefitMap(inputs.at(2).weights().getName(), getNodeName(node), nvinfer1::WeightsRole::kBIAS);
    }
    tensorPtr = layer->getOutput(0);
    dims = tensorPtr->getDimensions();

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
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
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensorPtr->getDimensions();
    // Deconvolution input must be at least 3D and at most 5D.
    ASSERT(dims.nbDims >= 3 && dims.nbDims <= 5 && "TensorRT only supports 1D, 2D or 3D deconvolutions!",
        ErrorCode::kUNSUPPORTED_NODE);
    // Deconvolution weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernelWeights = inputs.at(1).weights();

    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        std::vector<int> axes{3};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }
    if (kernelWeights.shape.nbDims == 3)
    {
        kernelWeights.shape.nbDims = 4;
        kernelWeights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernelWeights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

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

    int ngroup = attrs.get("group", 1);
    int noutput = kernelWeights.shape.d[1] * ngroup; // Note: Weights order is CKRS

    // Check for bias_weights
    nvinfer1::Weights biasWeights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shapedBiasWeights = inputs.at(2).weights();
        // ONNX requires shapedBiasWeights to be 1D
        ASSERT(shapedBiasWeights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shapedBiasWeights.shape.d[0] == noutput, ErrorCode::kINVALID_NODE);
        biasWeights = shapedBiasWeights;
    }
    else
    {
        biasWeights = ShapedWeights::empty(kernelWeights.type);
    }

    // Kernel shape either comes from the attributes or extracted from the kernel weights shape
    kernelSize.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        kernelSize.d[nbSpatialDims - i] = kernelWeights.shape.d[kernelWeights.shape.nbDims - i];
    }

    getKernelParams(ctx, node, &kernelSize, &strides, &begPadding, &endPadding, paddingMode, exclude_padding,
        &dilations, &outputPadding);

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernelSize.d[nbSpatialDims - i] == kernelWeights.shape.d[kernelWeights.shape.nbDims - i],
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

        // NOTE: it is possible for generatePadding to produce negative values for pre and post padding, which usually happens when
        // output_shape is provided but output_padding is not. Any negative values generated for post-padding can be translated
        // into outputPadding to pad the output tensor post deconvolution. Any negative values for pre-padding are unsupported.

        for (int i = 0; i < nbSpatialDims; i++)
        {
            ASSERT(begPadding.d[i] >= 0 && "TensorRT does not support negative pre-padding in the ConvTranspose operator!",
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
        for (int i = 0; i < nbSpatialDims; ++i)
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

    nvinfer1::Weights emptyBiasWeights = ShapedWeights::empty(kernelWeights.type);

    // Create a deconvolution layer and set known attributes - strides,ngroups, and dilations
    // If there is still output padding, remove the bias weights. Bias will be added below.
    auto* layer = ctx->network()->addDeconvolutionNd(
        *tensorPtr, noutput, kernelSize, kernelWeights, hasOutputPadding ? emptyBiasWeights : biasWeights);
    layer->setStrideNd(strides);
    layer->setNbGroups(ngroup);
    layer->setDilationNd(dilations);

    // Check that 3D deconvolution paddings is valid
    if (nbSpatialDims == 3)
    {
        ASSERT(begPadding == endPadding && "TensorRT does not support asymmetrical padding for 3D deconvolutions!",
            ErrorCode::kUNSUPPORTED_NODE);
    }

    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    LOG_VERBOSE("Running deconvolution with: " << "\n"
                << "Padding mode: " << autoPadMode << "\n"
                << "Pre-padding: " << begPadding << "\n"
                << "Post-padding: " << endPadding);

    // Register layer, along with refittable kernel weights and bias weights (if any)
    ctx->registerLayer(layer, getNodeName(node));
    ctx->insertRefitMap(inputs.at(1).weights().getName(), getNodeName(node), nvinfer1::WeightsRole::kKERNEL);
    if (inputs.size() == 3)
    {
        ctx->insertRefitMap(inputs.at(2).weights().getName(), getNodeName(node), nvinfer1::WeightsRole::kBIAS);
    }
    tensorPtr = layer->getOutput(0);
    dims = tensorPtr->getDimensions();

    // There is still output padding. Add a padding layer to handle it.
    if (hasOutputPadding)
    {
        // TRT only support 2D padding on the outermost dimensions
        ASSERT(outputPadding.nbDims == 2 || (outputPadding.nbDims == 3 && outputPadding.d[0] == 0),
            ErrorCode::kUNSUPPORTED_NODE);
        // Convert 3D padding to 2d padding
        if (nbSpatialDims == 3)
        {
            outputPadding = {2, {outputPadding.d[1], outputPadding.d[2]}};
        }
        LOG_VERBOSE("Padding output deconvolution tensor with: " << outputPadding);
        tensorPtr = ctx->network()->addPaddingNd(*tensorPtr, makeDims(2, 0), outputPadding)->getOutput(0);

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

    if (needToExpandDims)
    {
        std::vector<int> axes{3};
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
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

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    // Input tensor is in NCHW format
    ASSERT(inputs.at(0).shape().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(tensorPtr->getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);

    // Extract attributes
    OnnxAttrs attrs(node, ctx);
    auto blockSize = attrs.get<int>("blocksize");
    auto mode = attrs.get<std::string>("mode", "DCR");

    // Useful constants
    const auto inputShape = shapeOf(*tensorPtr);
    const auto N = gather(ctx, inputShape, shapeVector(0));
    const auto C = gather(ctx, inputShape, shapeVector(1));
    const auto H = gather(ctx, inputShape, shapeVector(2));
    const auto W = gather(ctx, inputShape, shapeVector(3));
    const auto blockSizeTensor = shapeVector(blockSize);
    const auto C_2 = floorDiv(ctx, C, mul(ctx, blockSizeTensor, blockSizeTensor));
    const auto H_2 = mul(ctx, H, blockSizeTensor);
    const auto W_2 = mul(ctx, W, blockSizeTensor);
    const int DCRPerm[6] = {0, 3, 4, 1, 5, 2};
    const int CRDPerm[6] = {0, 1, 4, 2, 5, 3};

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
    ctx->registerLayer(firstShuffle, node.name());
    tensorPtr = firstShuffle->getOutput(0);

    // Finally reshape to {N, C / (blockSize * blockSize), H * blockSize, W * blockSize};
    auto secondShape = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShape);
    tensorPtr = secondShuffle->getOutput(0);

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(DequantizeLinear)
{
    ASSERT(inputs.size() == 3, nvonnxparser::ErrorCode::kINVALID_NODE);

    std::string name = node.name();
    // Input 0 can be a weights or a tensor
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    std::string input_tensor_name = name + std::string("_input_weight_tensor");
    input.setName(input_tensor_name.c_str());

    // Second and third input should be a constant
    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(2).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);

    auto type = inputs.at(1).weights().type;
    auto scale = inputs.at(1).weights();
    auto power = ShapedWeights::empty(type);
    auto shift = createZeroShifts(inputs.at(2).weights(), type, ctx);

    ASSERT(scale.count() == shift.count(), nvonnxparser::ErrorCode::kINVALID_NODE);

    // Set Uniform scale mode by default.
    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kUNIFORM;
    if (scale.count() != 1)
    {
        // Ensure that number of scales are equalt to output channel.
        size_t K = input.getDimensions().d[0];
        ASSERT(K == scale.count(), nvonnxparser::ErrorCode::kINVALID_NODE);
        mode = nvinfer1::ScaleMode::kCHANNEL;
    }

    auto invScale = ctx->createTempWeights(scale.type, scale.shape);
    auto invShift = ctx->createTempWeights(shift.type, shift.shape);

    float* s = static_cast<float*>(scale.values);
    float* ns = static_cast<float*>(invScale.values);
    float* b = static_cast<float*>(shift.values);
    float* nb = static_cast<float*>(invShift.values);

    for (int i = 0, n = scale.count(); i < n; i++)
    {
        ns[i] = 1.0f / s[i];
        nb[i] = -b[i] * ns[i];
    }

    // Map Quantization node to a scale node
    auto layer = ctx->network()->addScale(input, mode, invShift, invScale, power);

    // Set output precision type of the scale node to INT8 - indicates its a quantizing scale node.
    layer->setOutputType(0, nvinfer1::DataType::kFLOAT);

    std::string dequantize_node_name = name + std::string("_dequantize_scale_node");
    std::string dequantize_node_output = dequantize_node_name + "_output_tensor";
    layer->setName(dequantize_node_name.c_str());
    layer->getOutput(0)->setName(dequantize_node_output.c_str());

    // Return layer output
    RETURN_FIRST_OUTPUT(layer);
}

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kDIV);
}

DEFINE_BUILTIN_OP_IMPORTER(Dropout)
{
    int noutputs = node.output().size();
    if (noutputs == 1)
    {
        RETURN_IDENTITY(inputs.at(0));
    }
    else
    {
        // Error if opset version >= 10 as boolean not supported right now
        ASSERT(ctx->getOpsetVersion() < 10, ErrorCode::kUNSUPPORTED_NODE);
        // Add identity layer twice for both Dropout outputs: (output + mask)
        std::vector<TensorOrWeights> outputs;
        outputs.push_back(identity(ctx, inputs.at(0)));
        outputs.push_back(identity(ctx, inputs.at(0)));
        return outputs;
    }
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
    // "Broadcast the input tensor following the given shape and the broadcast rule."
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT (inputTensor.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    const auto inputDims = shapeOf(inputTensor);
    const auto inputRank = shapeOf(inputDims);

    // "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
    ASSERT(inputs.at(1).shape().nbDims == 1, ErrorCode::kINVALID_VALUE);
    ShapeTensor shape{inputs.at(1)};
    const auto shapeLength = shapeOf(shape);

    const ShapeTensor newRank = max(ctx, shapeLength, inputRank);
    // "Dimensions are right alignment;..."
    const ShapeTensor newDims = concat(ctx, fillShapeVector(ctx, 1, sub(ctx, newRank, inputRank)), inputDims);
    nvinfer1::ITensor& newInputTensor = reshape(ctx, inputTensor, newDims);

    // ", or the shape.ndim < input.shape.ndim"
    ShapeTensor newShape = concat(ctx, fillShapeVector(ctx, 1, sub(ctx, newRank, shapeLength)), shape);

    const ShapeTensor starts = similar(ctx, newDims, 0);
    // Do the broadcast rule.
    const ShapeTensor sizes = broadcast(ctx, newDims, newShape);
    // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
    const ShapeTensor one = shapeVector(1);
    const ShapeTensor strides = min(ctx, one, sub(ctx, newDims, one));

    nvinfer1::ISliceLayer* sliceLayer = addSlice(ctx, newInputTensor, starts, sizes, strides);
    ctx->registerLayer(sliceLayer, node.name());

    RETURN_FIRST_OUTPUT(sliceLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int nbDims = tensorPtr->getDimensions().nbDims;
    int axis = attrs.get("axis", 1);
    TRT_CHECK(convertAxis(axis, nbDims));

    if (nbDims > 2)
    {
        tensorPtr = flattenTensor(ctx, node, *tensorPtr, axis, true);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(data.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);
    auto* layer = ctx->network()->addGather(data, indices, axis);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);

    // TRT does not support INT32 input types for this node
    ASSERT(!inputs.at(0).isInt32() && !inputs.at(1).isInt32()
        && "TensorRT doesn't support INT32 inputs for GEMM!", ErrorCode::kUNSUPPORTED_NODE);

    // Use FC if it is likely to be faster - which is usually when no Shuffles are required.
    bool canUseFC = inputs.at(0).is_tensor() && inputs.at(1).is_weights() && inputs.at(2).is_weights() && alpha == 1.f
        && beta == 1.f && inputs.at(0).tensor().getDimensions().nbDims == 2 && inputs.at(1).weights().shape.nbDims == 2
        && inputs.at(2).weights().shape.nbDims == 1;
    if (canUseFC)
    {
        LOG_VERBOSE("GEMM: using FC layer instead of MM because all criteria were met.");
        const std::vector<int> axesInput{2, 3};
        nvinfer1::ITensor* inputAExtendDim = unsqueezeTensor(ctx, node, inputA, axesInput);

        ShapedWeights weights = inputs.at(1).weights();
        if (!transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            transposedWeights.setName(weights.getName());
            LOG_WARNING("Weight " << transposedWeights.getName() << " has been transposed! If you plan on overwriting this weight with the Refitter API, the new weights must be pre-transposed");
            weights = transposedWeights;
        }
        ShapedWeights biases{};
        if (inputs.size() > 2)
        {
            biases = inputs.at(2).weights();
        }
        nvinfer1::IFullyConnectedLayer* fc = ctx->network()->addFullyConnected(*inputAExtendDim, biases.shape.d[0], weights, biases);
        // Register layer, kernel weights and bias weights (if any)
        ctx->registerLayer(fc, node.name());
        ctx->insertRefitMap(weights.getName(), node.name(), nvinfer1::WeightsRole::kKERNEL);
        if (inputs.size() == 3)
        {
            ctx->insertRefitMap(biases.getName(), node.name(), nvinfer1::WeightsRole::kBIAS);
        }
        const std::vector<int> axesOutput{2, 3};
        return {{squeezeTensor(ctx, node, *fc->getOutput(0), axesOutput)}};

    }

    nvinfer1::ITensor* inputB {nullptr};

    // If input B is a constant, we transpose at parse time if necessary,
    // because In some cases, A * Bt is much slower than A * B.
    if (inputs.at(1).is_weights())
    {
        ShapedWeights weights = inputs.at(1).weights();
        if (transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            transposedWeights.setName(weights.getName());
            LOG_WARNING("Weight " << transposedWeights.getName() << " has been transposed! If you plan on overwriting this weight with the Refitter API, the new weights must be pre-transposed");
            weights = transposedWeights;
            // Since we've already transposed now, we can set transpose to false.
            transB = false;
        }
        nvinfer1::IConstantLayer* weightsLayer
            = ctx->network()->addConstant(weights.shape, static_cast<nvinfer1::Weights>(weights));
        // Map the constant layer to the weights name.
        ctx->registerLayer(weightsLayer, node.input(1));
        ctx->insertRefitMap(weights.getName(), weightsLayer->getName(), nvinfer1::WeightsRole::kCONSTANT);
        inputB = weightsLayer->getOutput(0);
    }
    else
    {
        inputB = &inputs.at(1).tensor();
    }

    nvinfer1::ITensor* inputASqueezed = &inputA;
    nvinfer1::Dims newDims = squeeze_trailing_dims(inputA.getDimensions());
    // When A has more than 2 dimensions, it needs to be flattened.
    if (newDims.nbDims > 2)
    {
        newDims = nvinfer1::Dims{1, {-1}};
    }
    // Due to other TRT layers, inputA may sometimes have trailing 1s that need to be removed.
    if (newDims.nbDims < inputA.getDimensions().nbDims)
    {
        nvinfer1::IShuffleLayer* squeeze = ctx->network()->addShuffle(inputA);
        squeeze->setReshapeDimensions(newDims);
        squeeze->setZeroIsPlaceholder(false);
        inputASqueezed = squeeze->getOutput(0);
    }

    const auto getMatrixOp = [](const nvinfer1::ITensor& input, bool transpose) {
        if (input.getDimensions().nbDims == 1)
        {
            return nvinfer1::MatrixOperation::kVECTOR;
        }
        else if (transpose)
        {
            return nvinfer1::MatrixOperation::kTRANSPOSE;
        }
        return nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputASqueezed, transA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB, transB);

    LOG_VERBOSE("Using opA: " << static_cast<int>(opA) << " opB: " << static_cast<int>(opB));
    LOG_VERBOSE("GEMM: A, after squeezing: " << inputASqueezed->getDimensions());

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputASqueezed, opA, *inputB, opB);
    ctx->registerLayer(matmul, node.name());
    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        TRT_CHECK(broadcastTensors(ctx, alphaConstantTensor, matmulTensor));
        nvinfer1::IElementWiseLayer* scaledMatmul = ctx->network()->addElementWise(
            *alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        matmulTensor = scaledMatmul->getOutput(0);
    }

    // In opset 11, the bias tensor is an optional input
    if (inputs.size() == 3)
    {
        nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);

        // Scale C if needed
        if (beta != 1.f)
        {
            nvinfer1::IConstantLayer* betaConstant
                = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
            TRT_CHECK(broadcastTensors(ctx, betaConstantTensor, biasTensor));
            nvinfer1::IElementWiseLayer* scaledBias = ctx->network()->addElementWise(
                *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
            biasTensor = scaledBias->getOutput(0);
        }
        // A*B may be lower rank than C in TRT, so need to squeeze C.
        if (ctx->getOpsetVersion() < 7 && !attrs.get("broadcast", false))
        {
            nvinfer1::Dims squeezeDims = squeeze_leading_dims(biasTensor->getDimensions());
            biasTensor = reshapeTensor(ctx, *biasTensor, squeezeDims);
        }
        TRT_CHECK(broadcastTensors(ctx, matmulTensor, biasTensor));
        nvinfer1::IElementWiseLayer* biasAdd
            = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
        return {{biasAdd->getOutput(0)}};
    }

    return {{matmulTensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool)
{
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
    auto& pInvTensor = *addConstantScalar(ctx, 1.f / p, ::ONNX_NAMESPACE::TensorProto::FLOAT, scalarDims)->getOutput(0);

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
    return {{globalPoolingHelper(ctx, node, convertToTensor(inputs.at(0), ctx), nvinfer1::ReduceOperation::kMAX)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Greater)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kGREATER);
}

// singlePassShape is the shape of the output from a single pass.
nvinfer1::ITensor* concatenateRNNOutputs(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop,
    nvinfer1::ITensor* singlePassShape, nvinfer1::ITensor* sequenceLength, nvinfer1::ITensor* concatenatedOutput,
    int numDirections, std::vector<TensorOrWeights>& inputs, bool reverse = false)
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
    constexpr int NUM_GATES = 3;
    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int numDirections = (direction == "bidirectional") ? 2 : 1;
    const int hiddenSize = attrs.get<int>("hidden_size");
    const int linearBeforeReset = attrs.get<int>("linear_before_reset", 0);
    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& weights = convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& recurrenceWeights = convertToTensor(inputs.at(2), ctx);

    constexpr int NUM_ACTIVATIONS = 2;
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
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the GRU do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
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
    const auto initialStateShape = [&ctx, &numDirections, &hiddenSize, &input, &net]() -> nvinfer1::ITensor* {
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
    ASSERT(iterationInput, ErrorCode::kINVALID_NODE);

    // H(t-1)
    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
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
    ctx->registerLayer(Ht1, node.name());
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

    const auto isolateGate
        = [&ctx, &hiddenSize, &gateOutputShape, &net](nvinfer1::ITensor* gates, int gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolateGate = net->addSlice(*gates, Dims3{0, 0, 0}, Dims3{0, 0, 0}, Dims3{1, 1, 1});
        isolateGate->setInput(1, *addConstant(ctx, std::vector<int>{0, 0, gateIndex * hiddenSize},
                                      ::ONNX_NAMESPACE::TensorProto_DataType_INT32, Dims{1, 3})
                                      ->getOutput(0)); // Start
        isolateGate->setInput(2, *gateOutputShape);    // Size
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
    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*gateOutputShape,
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
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
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(If)
{
    OnnxAttrs attrs(node, ctx);
    auto cond = inputs.at(0);
    ASSERT(cond.is_weights() && cond.weights().count() == 1 && "If condition must be a initializer!",
        ErrorCode::kUNSUPPORTED_NODE);
    auto value = *(static_cast<int*>(cond.weights().values));
    std::vector<TensorOrWeights> graphOutputs;
    const ::ONNX_NAMESPACE::GraphProto& body = value == 1 ? attrs.get<const ::ONNX_NAMESPACE::GraphProto&>("then_branch") : attrs.get<const ::ONNX_NAMESPACE::GraphProto&>("else_branch");
    TRT_CHECK(onnx2trt::parseGraph(ctx, body));
    const int nbOutputs = body.output_size();
    for (int i = 0; i < nbOutputs; i++)
    {
        graphOutputs.emplace_back(ctx->tensors().at(body.output(i).name()));
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
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    // Scales and biases must be initializers
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int nbDims = tensorPtr->getDimensions().nbDims;
    ASSERT(nbDims >= 3 && nbDims <= 4 && "TensorRT only supports InstanceNormalization on 3D or 4D tensors!",
        ErrorCode::kUNSUPPORTED_NODE);
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);

    // Populate instanceNormalization plugin properties.
    const std::string pluginName = "InstanceNormalization_TRT";
    const std::string pluginVersion = "1";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", scale_weights.values, nvinfer1::PluginFieldType::kFLOAT32, scale_weights.count());
    f.emplace_back("bias", bias_weights.values, nvinfer1::PluginFieldType::kFLOAT32, bias_weights.count());

    // Create plugin from registry
    nvinfer1::IPluginV2* plugin = createPlugin(node.name(), importPluginCreator(pluginName, pluginVersion), f);

    ASSERT(plugin != nullptr && "InstanceNormalization plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addPluginV2(&tensorPtr, 1, *plugin);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
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

DEFINE_BUILTIN_OP_IMPORTER(Log)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    // Don't use softmax converter since it adds a shuffle layer
    // which prevents the builder to fuse softmax and log operations.

    OnnxAttrs attrs(node, ctx);
    // "input : T"
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const auto dims = shapeOf(input);
    // "axis : int (default is 1)"
    int axis = attrs.get("axis", 1);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, dims.size()));

    // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
    auto* flattened = flattenTensor(ctx, node, input, axis);
    auto* softMax = ctx->network()->addSoftMax(*flattened);
    ctx->registerLayer(softMax, node.name());
    // ONNX softmax is always on second dimension.
    softMax->setAxes(1 << 1);

    // Reshape back to original shape
    nvinfer1::IUnaryLayer* unaryLayer = ctx->network()->addUnary(*softMax->getOutput(0), nvinfer1::UnaryOperation::kLOG);
    auto *reshapeLayer = addShuffle(ctx, *unaryLayer->getOutput(0), dims);
    RETURN_FIRST_OUTPUT(reshapeLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Loop)
{
    constexpr int NB_NON_STATE_INPUTS = 2; // First 2 inputs are trip count and condition respectively.
    constexpr int NB_DISCARDED_OUTPUTS
        = 1; // First output is the updated value of the condition, and is ignored by the outer loop node.
    constexpr int MAX_SCAN_OUTPUT_LENGTH = 1024; // Maximum length for scan outputs if trip count is not set.
    ASSERT(inputs.size() >= 2, ErrorCode::kINVALID_NODE);
    OnnxAttrs attrs(node, ctx);
    const int nbInputs = node.input().size();
    // The number of state variables on the input and output is the same.
    const int nbStateVars = nbInputs - NB_NON_STATE_INPUTS;

    const ::ONNX_NAMESPACE::GraphProto& body = attrs.get<const ::ONNX_NAMESPACE::GraphProto&>("body");

    auto loop = ctx->network()->addLoop();
    // Trip count and condition are optional inputs.
    nvinfer1::ITensor* tripLimit{nullptr};
    if (inputs[0])
    {
        tripLimit = convertToScalar(ctx, &convertToTensor(inputs[0], ctx));
        ASSERT(tripLimit, ErrorCode::kINVALID_NODE);
        ctx->loopTensors()[body.input(0).name() + " tripLimit"] = node.input(0);
        loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
        // First graph input is iteration_num, so create a loop counter
        auto counter = addLoopCounter(ctx, loop, 0);
        ctx->registerTensor(counter, body.input(0).name());
        ctx->registerTensor(tripLimit, body.input(0).name() + " tripLimit");
    }
    if (inputs[1])
    {
        nvinfer1::ITensor* cond = convertToScalar(ctx, &convertToTensor(inputs[1], ctx));
        ASSERT(cond, ErrorCode::kINVALID_NODE);
        ctx->loopTensors()[body.input(1).name()] = node.input(1);
        loop->addTripLimit(*cond, nvinfer1::TripLimit::kWHILE);
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
    ctx->registerLayer(stateVars.at(0), node.name());

    // Loop body
    TRT_CHECK(onnx2trt::parseGraph(ctx, body));

    // Set final values of state variables.
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        // The first output of the body graph is the updated condition, which is ignored by the Loop node.
        const int index = i + NB_DISCARDED_OUTPUTS;
        const auto& bodyOutputName = body.output(index).name();
        auto& stateOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        LOG_VERBOSE("For state variable output: " << bodyOutputName
                                                  << ", found matching tensor: " << stateOutput.getName()
                                                  << ", with shape: " << stateOutput.getDimensions());
        stateVars.at(i)->setInput(1, stateOutput);
        // Each state variable is also a loop output
        nodeOutputs.emplace_back(
            loop->addLoopOutput(*stateVars.at(i)->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));
    }
    const int nbOutputs = body.output_size();
    // Finally, set up scan outputs if there are any
    for (int i = nbStateVars + NB_DISCARDED_OUTPUTS; i < nbOutputs; ++i)
    {
        const auto& bodyOutputName = body.output(i).name();
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
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(LSTM)
{
    using trtAct = nvinfer1::ActivationType;
    using eOp = nvinfer1::ElementWiseOperation;

    OnnxAttrs attrs{node, ctx};
    constexpr int NUM_GATES = 4;
    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int numDirections = (direction == "bidirectional") ? 2 : 1;
    const int hiddenSize = attrs.get<int>("hidden_size");
    const int inputForget = attrs.get("input_forget", 0);
    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    ASSERT(
        inputForget == 0 && "Coupled input/forget is unsupported in the LSTM converter", ErrorCode::kUNSUPPORTED_NODE);

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);

    constexpr int NUM_ACTIVATIONS = 3;
    std::vector<trtAct> defaultActs{trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH};
    if (numDirections == 2)
    {
        defaultActs.insert(defaultActs.end(), {trtAct::kSIGMOID, trtAct::kTANH, trtAct::kTANH});
    }
    std::vector<trtAct> activations = attrs.get<std::vector<trtAct>>("activations", defaultActs);

    std::vector<float> activationAlphas = attrs.get<std::vector<float>>("activation_alpha", std::vector<float>{});
    std::transform(activations.begin() + activationAlphas.size(), activations.end(),
        std::back_inserter(activationAlphas), &getActivationDefaultAlpha);

    std::vector<float> activationBetas = attrs.get<std::vector<float>>("activation_beta", std::vector<float>{});
    std::transform(activations.begin() + activationBetas.size(), activations.end(), std::back_inserter(activationBetas),
        &getActivationDefaultBeta);

    // TODO: Support cases where in bidirectional LSTMs, activations of reverse iteration do not match forward pass.
    // TODO: This will require splitting the input tensor in the loop when applying activations.
    if (numDirections == 2)
    {
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the LSTM do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
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
    const auto initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
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

    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
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
    ASSERT(iterationInput, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(Ht1, node.name());
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
    const auto isolateGate
        = [&ctx, &hiddenSize, &gateOutputShape](nvinfer1::ITensor* gates, int gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolate = ctx->network()->addSlice(
            *gates, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
        isolate->setInput(1, *addConstant(ctx, std::vector<int>{0, 0, gateIndex * hiddenSize},
                                  ::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, 3})
                                  ->getOutput(0)); // Start
        isolate->setInput(2, *gateOutputShape);    // Size
        return isolate->getOutput(0);
    };

    // Compute peephole connections
    nvinfer1::ITensor* peephole{nullptr};
    if (inputs.size() > 7 && inputs.at(7))
    {
        peephole = &convertToTensor(inputs.at(7), ctx);
    }

    const auto addPeephole = [&ctx, &node, &hiddenSize, &numDirections, &peephole](
        nvinfer1::ITensor* gate, nvinfer1::ITensor* cellState, int gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolatePeephole
            = ctx->network()->addSlice(*peephole, nvinfer1::Dims2{0, gateIndex * hiddenSize},
                nvinfer1::Dims2{numDirections, hiddenSize}, nvinfer1::Dims2{1, 1});
        auto* peepholeWeights = unsqueezeTensor(ctx, node, *isolatePeephole->getOutput(0), std::vector<int>{1});
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
    nvinfer1::ITensor* Ct
        = ctx->network()
              ->addElementWise(*ctx->network()->addElementWise(*ftGate, *Ct1->getOutput(0), eOp::kPROD)->getOutput(0),
                  *ctx->network()->addElementWise(*itGate, *ctGate, eOp::kPROD)->getOutput(0), eOp::kSUM)
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

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    nvinfer1::ITensor* inputA = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);
    // TRT does not support INT32 input types for this node
    ASSERT(inputA->getType() != nvinfer1::DataType::kINT32 && inputB->getType() != nvinfer1::DataType::kINT32
        && "TensorRT doesn't support INT32 inputs for MatMul!", ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims inputADims = inputA->getDimensions();
    nvinfer1::Dims inputBDims = inputB->getDimensions();

    // Use FC if possible as FC can be fused with later activation and bias layers.
    // Input unsqueezed from (N, C) to (N, 1, 1, C).
    // Output squeezed from (N, K, 1, 1) to (N, K).
    bool canUseFC = inputs.at(0).is_tensor() && inputs.at(1).is_weights() &&
                    inputADims.nbDims == 2 && inputBDims.nbDims == 2;
    if (canUseFC)
    {
        LOG_VERBOSE("GEMM: using FC layer instead of MM because all criteria were met.");
        const std::vector<int> axesInput{2, 3};
        nvinfer1::ITensor* inputAExtendDim = unsqueezeTensor(ctx, node, *inputA, axesInput);

        ShapedWeights weights = inputs.at(1).weights();
        auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
        ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);

        auto biasDtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
        auto biasShape = nvinfer1::Dims{1, {inputBDims.d[1]}};
        auto biasWeights = ctx->createTempWeights(biasDtype, biasShape);
        std::fill(static_cast<float*>(biasWeights.values), static_cast<float*>(biasWeights.values) + biasWeights.count(), 0.0);
        nvinfer1::IFullyConnectedLayer* fc = ctx->network()->addFullyConnected(*inputAExtendDim, inputBDims.d[1], transposedWeights, biasWeights);
        // Register layer name and kernel weights for FC.
        ctx->registerLayer(fc, getNodeName(node));
        ctx->insertRefitMap(weights.getName(), getNodeName(node), nvinfer1::WeightsRole::kKERNEL);
        const std::vector<int> axesOutput{2, 3};
        return {{squeezeTensor(ctx, node, *fc->getOutput(0), axesOutput)}};
    }

    TRT_CHECK(broadcastTensors(ctx, inputA, inputB));

    const auto getMatrixOp = [](const nvinfer1::ITensor& input) {
        return (input.getDimensions().nbDims == 1) ? nvinfer1::MatrixOperation::kVECTOR
                                                   : nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB);

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputA, opA, *inputB, opB);
    ctx->registerLayer(matmul, node.name());
    return {{matmul->getOutput(0)}};
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
    ASSERT(sum_input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor& sum_tensor = sum_input.tensor();

    int ndim = sum_tensor.getDimensions().nbDims;
    float scale_value = 1.f / inputs.size();
    auto scale_dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    auto scale_shape = nvinfer1::Dims{ndim, {1, 1, 1, 1, 1, 1, 1, 1}};
    auto scale_weights = ctx->createTempWeights(scale_dtype, scale_shape);
    static_cast<float*>(scale_weights.values)[0] = scale_value;
    auto* constant_layer = ctx->network()->addConstant(scale_weights.shape, scale_weights);
    ASSERT(constant_layer, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
    RETURN_FIRST_OUTPUT(
        ctx->network()->addElementWise(sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD));
}

DEFINE_BUILTIN_OP_IMPORTER(Min)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPROD);
}

DEFINE_BUILTIN_OP_IMPORTER(Neg)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return unaryHelper(ctx, node, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(Or)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    int nbDims = tensorPtr->getDimensions().nbDims;
    std::vector<int> axes;
    // TensorRT only supports 2D padding on the outermost dimensions of an input tensor that is
    // at least 4D. Unsqueeze leading dimensions to convert input tensor to 4D if necessary
    bool needToExpandDims = (nbDims < 4);
    if (needToExpandDims)
    {
        int diff = 4 - nbDims;
        axes.resize(diff);
        std::iota(axes.begin(), axes.end(), 0);
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::Dims2 begPadding, endPadding;
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("mode", "constant");
    float value{0.f};
    std::vector<int64_t> onnxPadding;

    if (ctx->getOpsetVersion() < 11)
    {
        value = attrs.get<float>("value", 0.f);
        auto padding = attrs.get<std::vector<int>>("pads");
        onnxPadding = std::vector<int64_t>(padding.begin(), padding.end());
    }
    // In opset >= 11, padding indicies and values moved from attributes to inputs
    else
    {
        ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        weightsToVector(inputs.at(1).weights(), &onnxPadding);
        if (inputs.size() == 3)
        {
            ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
            auto padWeight = inputs.at(2).weights();
            ASSERT(padWeight.count() == 1, ErrorCode::kINVALID_NODE);
            value = static_cast<float*>(padWeight.values)[0];
        }
    }

    ASSERT(mode == "constant" && value == 0.f && "This version of TensorRT only supports constant 0 padding!",
        ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(convertOnnxPadding(onnxPadding, &begPadding, &endPadding)
        && "This version of TensorRT only supports padding on the outer two dimensions!",
        ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addPaddingNd(*tensorPtr, begPadding, endPadding);
    ctx->registerLayer(layer, node.name());
    tensorPtr = layer->getOutput(0);

    // Squeeze back to original rank if necessary
    if (needToExpandDims)
    {
        tensorPtr = squeezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensorPtr}};
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
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* slopes = &convertToTensor(inputs.at(1), ctx);
    ASSERT(input->getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(slopes->getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    TRT_CHECK(broadcastTensors(ctx, input, slopes));
    auto* layer = ctx->network()->addParametricReLU(*input, *slopes);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(QuantizeLinear)
{
    ASSERT(inputs.size() == 3, nvonnxparser::ErrorCode::kINVALID_NODE);
    std::string name = node.name();
    // Input 0 can be a weights or a tensor
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    std::string input_tensor_name = name + std::string("_input_weight_tensor");
    input.setName(input_tensor_name.c_str());

    // Second and third input should be a constant
    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(2).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);

    auto type = inputs.at(1).weights().type;
    auto scale = inputs.at(1).weights();
    auto power = ShapedWeights::empty(type);
    auto shift = createZeroShifts(inputs.at(2).weights(), type, ctx);

    ASSERT(scale.count() == shift.count(), nvonnxparser::ErrorCode::kINVALID_NODE);

    // Set Uniform scale mode by default.
    nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kUNIFORM;
    if (scale.count() != 1)
    {
        // Ensure that number of scales are equalt to output channel.
        size_t K = input.getDimensions().d[0];
        ASSERT(K == scale.count(), nvonnxparser::ErrorCode::kINVALID_NODE);
        mode = nvinfer1::ScaleMode::kCHANNEL;
    }

    // Map Quantization node to a scale node
    auto layer = ctx->network()->addScale(input, mode, shift, scale, power);

    // Set output precision type of the scale node to INT8 - indicates its a quantizing scale node.
    layer->setOutputType(0, nvinfer1::DataType::kINT8);

    std::string quantize_node_name = name + std::string("_quantize_scale_node");
    std::string quantize_node_output = quantize_node_name + "_output_tensor";
    layer->setName(quantize_node_name.c_str());
    layer->getOutput(0)->setName(quantize_node_output.c_str());

    // Return layer output
    RETURN_FIRST_OUTPUT(layer);
}

NodeImportResult randomUniformHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, const ShapeTensor& inputShape, const OnnxAttrs& attrs,
    const nvinfer1::DataType& inputDType)
{
    auto* fillLayer = addFill(ctx, inputShape, nvinfer1::FillOperation::kRANDOM_UNIFORM);
    ctx->registerLayer(fillLayer, node.name());

    // Set datatype of output:
    //      RandomUniform: dype is required and defaults to 1
    //      RandomUniformLike: dtype is optional and defaults to the same type as the input
    if (attrs.count("dtype"))
    {
        auto dtype = attrs.get<int>("dtype", 1);
        switch (dtype)
        {
        case ::ONNX_NAMESPACE::TensorProto::FLOAT: fillLayer->setOutputType(0, nvinfer1::DataType::kFLOAT); break;
        case ::ONNX_NAMESPACE::TensorProto::FLOAT16: fillLayer->setOutputType(0, nvinfer1::DataType::kHALF); break;
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
    const auto shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    const ShapeTensor inputShape{1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end())};

    return randomUniformHelper(ctx, node, inputShape, attrs, nvinfer1::DataType::kFLOAT);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniformLike)
{
    ASSERT(inputs.size() == 1, ErrorCode::kINTERNAL_ERROR);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    const auto inputShape = shapeOf(input);
    const OnnxAttrs attrs(node, ctx);
    const auto dType = input.getType();

    return randomUniformHelper(ctx, node, inputShape, attrs, dType);
}

NodeImportResult staticFloatRangeImporter(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, const std::vector<TensorOrWeights>& inputs)
{
    const float start = static_cast<float*>(inputs.at(0).weights().values)[0];
    const float limit = static_cast<float*>(inputs.at(1).weights().values)[0];
    const float delta = static_cast<float*>(inputs.at(2).weights().values)[0];
    const float size = std::max(std::ceil((limit - start) / delta), 0.0f);
    ASSERT(size != 0 && "Zero-sized range operators are not supported!", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(size <= std::numeric_limits<int32_t>::max() && "range operator size must fit in int32!",
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::IFillLayer* layer
        = addFill(ctx, shapeVector(static_cast<int32_t>(size)), nvinfer1::FillOperation::kLINSPACE);
    ctx->registerLayer(layer, node.name());
    layer->setAlpha(start);
    layer->setBeta(delta);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    if (inputs.at(0).is_weights() && inputs.at(0).weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
    {
        // Floating-point case supported by TensorRT only if all inputs are static.
        if (inputs.at(0).is_weights() && inputs.at(1).is_weights() && inputs.at(2).is_weights())
        {
            return staticFloatRangeImporter(ctx, node, inputs);
        }
    }

    ASSERT(inputs.at(0).isInt32()
            && "For range operator with dynamic inputs, this version of TensorRT only supports INT32!",
        ErrorCode::kUNSUPPORTED_NODE);

    // "start : T
    //     Scalar. First entry for the range of output values.
    //  limit : T
    //     Scalar. Exclusive upper limit for the range of output values.
    //  delta : T
    //     Scalar. Value to step by."
    const ShapeTensor start{inputs.at(0)};
    const ShapeTensor limit{inputs.at(1)};
    const ShapeTensor delta{inputs.at(2)};

    // "number_of_elements = max( ceil( (limit - start) / delta ) , 0 )"
    //
    // To implement this in TensorRT using only operations allowed on
    // shape tensors, rewrite as:
    //      "number_of_elements = max(0 - floor((start - limit) / delta), 0)
    //
    const ShapeTensor zero = shapeScalar(0);
    const ShapeTensor numberOfElements = max(ctx, sub(ctx, zero, floorDiv(ctx, sub(ctx, start, limit), delta)), zero);

    nvinfer1::IFillLayer* layer = addFill(ctx, convertTo1D(ctx, numberOfElements), nvinfer1::FillOperation::kLINSPACE);
    ctx->registerLayer(layer, node.name());

    // TensorRT requires that alpha and beta both be dynamic or both be static.
    if (start.allValuesKnown() && delta.allValuesKnown())
    {
        layer->setAlpha(start[0]);
        layer->setBeta(delta[0]);
        // Set layer output type to INT32 for statically-known ranges.
        layer->setOutputType(0, nvinfer1::DataType::kINT32);
    }
    else
    {
        layer->setInput(1, start.tensor(ctx));
        layer->setInput(2, convertTo1D(ctx, delta).tensor(ctx));
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
    return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kSUM);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare)
{
    nvinfer1::ITensor& tensor = inputs.at(0).tensor();
    auto* sqr_layer = ctx->network()->addElementWise(tensor, tensor, nvinfer1::ElementWiseOperation::kPROD);
    ASSERT(sqr_layer, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* sqr_tensorPtr = sqr_layer->getOutput(0);
    return reduceTensor(ctx, node, sqr_tensorPtr, nvinfer1::ReduceOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kRELU);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    // TRT does not support INT32 nor BOOL input types for this node
    ASSERT( (input.getType() != nvinfer1::DataType::kINT32
                && input.getType() != nvinfer1::DataType::kBOOL)
                && "This version of TensorRT does not support INT32 or BOOL input for the Resize operator.", ErrorCode::kUNSUPPORTED_NODE);
    int inputRank = input.getDimensions().nbDims;
    ASSERT( (inputRank > 0) && "The input tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);
    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);
    ctx->registerLayer(layer, getNodeName(node));
    OnnxAttrs attrs(node, ctx);

    auto mode = attrs.get<std::string>("mode", "nearest");
    auto resizeMode = mode == "nearest" ? nvinfer1::ResizeMode::kNEAREST : nvinfer1::ResizeMode::kLINEAR;

    std::string transformationMode = "half_pixel";

    if (ctx->getOpsetVersion() >= 11)
    {
        // Check for TRT-supported resize attributes
        transformationMode = attrs.get<std::string>("coordinate_transformation_mode", "half_pixel");
        ASSERT((transformationMode == "asymmetric" || transformationMode == "align_corners" || transformationMode == "half_pixel" || transformationMode == "pytorch_half_pixel")
                && "This version of TensorRT only supports asymmetric, align_corners, half_pixel, and pytorch_half_pixel resize!",
                ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(mode != "cubic" && "This version of TensorRT does not support cubic interpolation!",
            ErrorCode::kUNSUPPORTED_NODE);
        auto nearest_mode = attrs.get<std::string>("nearest_mode", "round_prefer_floor");
        ASSERT((mode != "nearest" || nearest_mode == "floor")
                && "This version of TensorRT only supports floor nearest_mode!",
            ErrorCode::kUNSUPPORTED_NODE);

        // The existence of a fourth input means a shape was passed as the resize parameter
        // For ONNX resize with the "sizes", TensorRT's resize maps to ONNX's in the following ways:
        // Nearest:
        //     alignCorners = 0: ASYMMETRIC
        //     alignCorners = 1: ALIGN_CORNERS
        // Linear:
        //     alignCorners = 0: HALF_PIXEL
        //     alignCorners = 1: ALIGN_CORNERS
        if (inputs.size() == 4)
        {
            if (transformationMode == "align_corners")
            {
                layer->setAlignCorners(true);
            }
            if (mode == "nearest")
            {
                ASSERT((transformationMode == "asymmetric" || transformationMode == "align_corners") && "TensorRT only supports asymmetric and align_corners transformation modes for nearest neighbor resizes when sizes are provided!", ErrorCode::kUNSUPPORTED_NODE);
            }
            else if (mode == "linear")
            {
                ASSERT((transformationMode == "half_pixel" || transformationMode == "pytorch_half_pixel" || transformationMode == "align_corners") && "TensorRT only supports half_pixel, pytorch_half_pixel, and align_corners transofmration modes for linear resizes when sizes are provided!", ErrorCode::kUNSUPPORTED_NODE);
            }
            auto* resizeShape = &convertToTensor(inputs.at(3), ctx);
            layer->setInput(1, *resizeShape);
            layer->setResizeMode(resizeMode);
            RETURN_FIRST_OUTPUT(layer);
        }
        // For ONNX resize with "scales", TensorRT's resize maps to ONNX's in the following ways:
        // Nearest:
        //    alignCorners = 0: ASYMMETRIC
        //    alignCorners = 1: ASYMMETRIC
        // Linear:
        //    alignCorners = 0: HALF_PIXEL
        //    alignCorners = 1: ASYMMETRIC
        else
        {
            if (mode == "nearest")
            {
                ASSERT(transformationMode == "asymmetric" && "TensorRT only supports asymmetric tranformation mode for nearest neighbor resizes when scales are provided!",ErrorCode::kUNSUPPORTED_NODE);
            }
            else if (mode == "linear")
            {
                ASSERT((transformationMode == "asymmetric" || transformationMode == "pytorch_half_pixel" || transformationMode == "half_pixel") && "TensorRT only supports half pixel, pytorch half_pixel, and asymmetric tranformation mode for linear resizes when scales are provided!", ErrorCode::kUNSUPPORTED_NODE);
                if (transformationMode == "asymmetric")
                {
                    layer->setAlignCorners(true);
                }
            }
        }
    }
    // For opset 10 resize, the only supported mode is asymmetric resize with scales.
    else
    {
        transformationMode = "asymmetric";
        layer->setAlignCorners(true);
    }

    // Resizes that use scale factors have the same import logic between opsets
    auto scales = ctx->getOpsetVersion() >= 11 ? inputs.at(2) : inputs.at(1);
    ASSERT(scales.is_weights() && "Resize scales must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights scales_weights = scales.weights();
    ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
    int scaleSize = scales_weights.shape.d[0];
    ASSERT(scaleSize == inputRank, ErrorCode::kINVALID_NODE);
    float const* scaleValues = static_cast<float const*>(scales_weights.values);
    if (resizeMode == nvinfer1::ResizeMode::kLINEAR)
    {
        ASSERT(canUseLinearResize(scaleSize, scaleValues),
            ErrorCode::kUNSUPPORTED_NODE);
    }

    layer->setResizeMode(resizeMode);
    layer->setScales(scaleValues, inputRank);

    LOG_VERBOSE("Running resize layer with: \n"
                << "Transformation mode: " << transformationMode << "\n"
                << "Resize mode: " << mode << "\n");

    auto* output = layer->getOutput(0);

    // TRT maps pytorch_half_pixel resizes to half_pixel resizes, which are functionally equivalent EXCEPT for interpolations down to 1D.
    if (transformationMode == "pytorch_half_pixel")
    {
        auto outputDims = output->getDimensions();
        // Validate resized spatial dimensions are > 1 for static dimensions, log a warning otherwise
        if (!isDynamic(outputDims))
        {
            for (int i = 2; i < outputDims.nbDims; i++)
            {
                ASSERT(outputDims.d[i] != 1 && "TensorRT doesn't support pytorch_half_pixel resizing for 1D interpolation!", ErrorCode::kUNSUPPORTED_NODE);
            }
        }
        else
        {
            LOG_WARNING(
                "TensorRT currently uses half_pixel calculation for the pytorch_half_pixel transformation mode. These "
                "are equivalent except for interpolations down to 1D.");
        }
    }

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Reshape)
{
    // "data : T
    // An input tensor"
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);

    ShapeTensor shape;
    if (ctx->getOpsetVersion() >= 5)
    {
        // "shape : tensor(int64)
        // Specified shape for output."
        shape = inputs.at(1);
    }
    else
    {
        // "Reshape-1
        // ...
        // shape : list of ints
        // New shape"
        OnnxAttrs attrs{node, ctx};
        const auto shapeAsIntList = attrs.get<std::vector<int>>("shape");
        shape = ShapeTensor(1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end()));
    }

    // "A dimension could also be 0, in which case the actual dimension
    // value is unchanged (i.e. taken from the input tensor)."
    nvinfer1::IShuffleLayer* layer = addShuffle(ctx, data, shape, /*zeroIsPlaceholder=*/true);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(RNN)
{
    OnnxAttrs attrs{node, ctx};

    const std::string direction = attrs.get<std::string>("direction", "forward");
    const int numDirections = (direction == "bidirectional") ? 2 : 1;
    const int hiddenSize = attrs.get<int>("hidden_size");

    const float clip = attrs.get("clip", -1.f); // Clipping cannot be negative, so -1.0 is a good sentinel value.

    // The input is in SBE format
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* weights = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* recurrenceWeights = &convertToTensor(inputs.at(2), ctx);

    constexpr int NUM_ACTIVATIONS = 1;
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
        ASSERT(std::equal(activations.begin(), activations.begin() + NUM_ACTIVATIONS, activations.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationAlphas.begin(), activationAlphas.begin() + NUM_ACTIVATIONS, activationAlphas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(std::equal(activationBetas.begin(), activationBetas.begin() + NUM_ACTIVATIONS, activationBetas.begin() + NUM_ACTIVATIONS) && "The parser does not currently support cases where activations for the reverse pass of the RNN do not match the forward pass.", ErrorCode::kUNSUPPORTED_NODE);
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
    const auto initialStateShape = [&ctx, &numDirections, &hiddenSize, &input]() -> nvinfer1::ITensor* {
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

    const auto getInitialInputValue = [&ctx, &initialStateShape, &inputs, &node](size_t inputIdx) -> nvinfer1::ITensor* {
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
    ASSERT(iterationInput, ErrorCode::kINVALID_NODE);

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
    ctx->registerLayer(hiddenState, node.name());
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
    const int opset8Offset = ctx->getOpsetVersion() == 8 ? 1 : 0;
    if (opset8Offset == 1)
    {
        ASSERT(inputs.at(0).isNullTensor() && "TensorRT doesn't support sequence_lens input for this node!",
            ErrorCode::kUNSUPPORTED_NODE);
    }
    const int nbInputs = node.input().size() - opset8Offset;
    const int nbScanInputs = attrs.get<int>("num_scan_inputs");
    // The number of state variables on the input and output is the same.
    const int nbStateVars = nbInputs - nbScanInputs;
    const int nbScanOutputs = node.output().size() - nbStateVars;

    // Populate scan input axis
    std::vector<int> defaultScanInputArgs(nbScanInputs);
    std::fill(defaultScanInputArgs.begin(), defaultScanInputArgs.end(), opset8Offset);
    std::vector<int> scanInputAxes(attrs.get("scan_input_axes", defaultScanInputArgs));

    // Populate scan input directions
    std::vector<int> defaultInputScanDirection(nbScanInputs);
    std::fill(defaultInputScanDirection.begin(), defaultInputScanDirection.end(), 0);
    const std::vector<int> scanInputDirections(attrs.get("scan_input_directions", defaultInputScanDirection));

    // Populate scan output axis
    std::vector<int> defaultScanOutputArgs(nbScanOutputs);
    std::fill(defaultScanOutputArgs.begin(), defaultScanOutputArgs.end(), opset8Offset);
    std::vector<int> scanOutputAxes(attrs.get("scan_output_axes", defaultScanOutputArgs));

    // Populate scan ouput directions
    std::vector<int> defaultOutputScanDirection(nbScanOutputs);
    std::fill(defaultOutputScanDirection.begin(), defaultOutputScanDirection.end(), 0);
    const std::vector<int> scanOutputDirections(attrs.get("scan_output_directions", defaultOutputScanDirection));

    const ::ONNX_NAMESPACE::GraphProto& body = attrs.get<const ::ONNX_NAMESPACE::GraphProto&>("body");

    // Support possible negative axis for input and output axes:
    for (auto& axis : scanInputAxes)
    {
        TRT_CHECK(convertAxis(axis, nvinfer1::Dims::MAX_DIMS));
    }

    for (auto& axis : scanOutputAxes)
    {
        TRT_CHECK(convertAxis(axis, nvinfer1::Dims::MAX_DIMS));
    }

    auto loop = ctx->network()->addLoop();
    // When multiple scan inputs are present, scan behaves like zip, so it is sufficient
    // to use only one scan input to determine trip limit.
    nvinfer1::ITensor* tripLimit = getAxisLength(ctx, &convertToTensor(inputs.back(), ctx), scanInputAxes.back());
    loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

    // Add initial state inputs using recurrent layers, and scan inputs using iterators.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs.at(i+opset8Offset), ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }
    ctx->registerLayer(stateVars.at(0), node.name());
    for (int i = 0; i < nbScanInputs; ++i)
    {
        const int index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput = loop->addIterator(convertToTensor(inputs.at(index+opset8Offset), ctx));
        scanInput->setAxis(scanInputAxes.at(i));
        scanInput->setReverse(scanInputDirections.at(i) == 1);
        ctx->registerTensor(TensorOrWeights{scanInput->getOutput(0)}, body.input(index).name());
    }

    // Loop Body. This is handled by dispatching to other op converters.
    TRT_CHECK(onnx2trt::parseGraph(ctx, body));

    // Set up recurrence outputs (first N body graph outputs).
    std::vector<TensorOrWeights> nodeOutputs{};
    for (int i = 0; i < nbStateVars; ++i)
    {
        const auto& bodyOutputName = body.output(i).name();
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
    for (int i = 0; i < nbScanOutputs; ++i)
    {
        const int index = nbStateVars + i;
        const auto& bodyOutputName = body.output(index).name();
        auto& scanOutput = convertToTensor(ctx->tensors().at(bodyOutputName), ctx);
        // For scanOutputDirections, 0 indicates appending, and 1, prepending.
        const auto scanDirection
            = (scanOutputDirections.at(i) == 0) ? nvinfer1::LoopOutput::kCONCATENATE : nvinfer1::LoopOutput::kREVERSE;
        const auto scanAxis = scanOutputAxes.at(i);
        LOG_VERBOSE("For scan output: " << bodyOutputName << ", found matching tensor: " << scanOutput.getName()
                                        << ", with shape: " << scanOutput.getDimensions() << ". Using scan direction: "
                                        << static_cast<int>(scanDirection) << ", and scan axis: " << scanAxis);
        nvinfer1::ILoopOutputLayer* trtScanOut = loop->addLoopOutput(scanOutput, scanDirection, scanAxis);
        trtScanOut->setInput(1, *tripLimit);
        nodeOutputs.emplace_back(trtScanOut->getOutput(0));
    }

    return {nodeOutputs};
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
    ctx->registerLayer(layer, node.name());
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
    const auto shape = shapeOf(inputs.at(0));

    // "outputs a int64 scalar that equals to the total number of elements of the input tensor."
    const ShapeTensor size = product(ctx, shape, 0, shape.size(), /*rank=*/0);

    return {{&size.tensor(ctx)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Slice)
{
    const int nbInputs = node.input().size();
    // "...it uses this information to slice the input data tensor."
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(data.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    const auto dims = shapeOf(data);

    // "Slices uses starts, ends, axes and steps inputs to specify the start and
    // end dimension and step for each axis in the list of axes..."
    ShapeTensor starts;
    ShapeTensor ends;
    ShapeTensor axes;
    ShapeTensor steps;

    // If opset version >= 10 slice parameters are weights instead of attributes.
    if (ctx->getOpsetVersion() >= 10)
    {
        ASSERT(nbInputs >= 3 && nbInputs <= 5, ErrorCode::kUNSUPPORTED_NODE);
        starts = inputs.at(1);
        ends = inputs.at(2);
        // "If axes are omitted, they are set to [0, ..., ndim-1]."
        axes = nbInputs > 3 ? ShapeTensor(inputs.at(3)) : iotaShapeVector(dims.size());
        // Doesn't support dynamic axes currently.
        ASSERT(axes.allValuesKnown(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(starts.size() == axes.size(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(ends.size() == axes.size(), ErrorCode::kUNSUPPORTED_NODE);
        // "If steps are omitted, they are set to [1, ..., 1] of length len(starts)."
        steps = inputs.size() > 4 ? ShapeTensor(inputs.at(4)) : similar(ctx, starts, 1);
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
    int j = 0;
    std::vector<int64_t> newAxes;
    newAxes.reserve(axes.size());

    for (int64_t axis : axes)
    {
        // "Accepted range is [-r, r-1] where r = rank(data)."
        const int r = dims.size();
        ASSERT(-r <= axis && axis < r, ErrorCode::kINVALID_VALUE);
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
    ASSERT(std::unordered_set<int64_t>(axes.begin(), axes.end()).size() == static_cast<size_t>(axes.size()),
        ErrorCode::kINVALID_NODE);

    if (axes.size() < dims.size() || !isIota)
    {
        // Axes specify a subset of the dimensions, or out of order.
        // Convert starts/ends/steps to complete in-order form.
        const ShapeTensor subscripts{axesToInterlaceSubscripts(axes, dims.size())};
        starts = interlace(ctx, similar(ctx, dims, 0), starts, subscripts);
        ends = interlace(ctx, dims, ends, subscripts);
        steps = interlace(ctx, similar(ctx, dims, 1), steps, subscripts);
    }

    // ONNX has a bunch of rules for converting out of bounds starts/ends
    // indices into the actual indices to use.
    decodeOnnxStartsAndEnds(ctx, dims, steps, starts, ends);

    // TensorRT uses sizes of the output dimensions instead of ends.
    const ShapeTensor sizes = computeSliceSizes(ctx, starts, ends, steps, dims);

    nvinfer1::ISliceLayer* slice = addSlice(ctx, data, starts, sizes, steps);

    ctx->registerLayer(slice, node.name());

    RETURN_FIRST_OUTPUT(slice);
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    OnnxAttrs attrs(node, ctx);
    // "input : T"
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const auto dims = shapeOf(input);

    // "axis : int (default is 1)"
    int axis = attrs.get("axis", 1);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, dims.size()));

    // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
    auto* flattened = flattenTensor(ctx, node, input, axis);
    auto* softMax = ctx->network()->addSoftMax(*flattened);
    ctx->registerLayer(softMax, node.name());
    // ONNX softmax is always on second dimension.
    softMax->setAxes(1 << 1);

    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *softMax->getOutput(0), dims);
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
    // Input tensor is in NCHW format
    ASSERT(inputs.at(0).shape().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(tensorPtr->getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);

    // Extract attributes
    OnnxAttrs attrs(node, ctx);
    auto blockSize = attrs.get<int>("blocksize");

    const nvinfer1::Permutation perm{0, 3, 5, 1, 2, 4};

    auto inputShape = shapeOf(*tensorPtr);

    const auto N = gather(ctx, inputShape, shapeVector(0));
    const auto C = gather(ctx, inputShape, shapeVector(1));
    const auto H = gather(ctx, inputShape, shapeVector(2));
    const auto W = gather(ctx, inputShape, shapeVector(3));
    const auto blockSizeTensor = shapeVector(blockSize);

    const auto C_2 = mul(ctx, C, mul(ctx, blockSizeTensor, blockSizeTensor));
    const auto H_2 = floorDiv(ctx, H, blockSizeTensor);
    const auto W_2 = floorDiv(ctx, W, blockSizeTensor);

    // First reshape to {N, C, H / blockSize, blockSize, W / blockSize, blockSize}

    const auto firstShapeDims = concat(ctx, N,
        concat(ctx, C,
            concat(ctx, H_2,
                concat(ctx, blockSizeTensor,
                    concat(ctx, W_2, blockSizeTensor)))));

    auto* firstShuffle = addShuffle(ctx, *tensorPtr, firstShapeDims);
    firstShuffle->setSecondTranspose(perm);
    ctx->registerLayer(firstShuffle, node.name());
    tensorPtr = firstShuffle->getOutput(0);

    // Reshape to {N, C * blockSize * blockSize, H / blockSize, W / blockSize}
    auto secondShapeDims = concat(ctx, N, concat(ctx, C_2, concat(ctx, H_2, W_2)));
    auto* secondShuffle = addShuffle(ctx, *tensorPtr, secondShapeDims);
    tensorPtr = secondShuffle->getOutput(0);

    return {{tensorPtr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    const int numOutputs = node.output().size();

    // "input : T
    // The tensor to split"
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(inputTensor.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    const auto inputDims = shapeOf(inputTensor);

    // "axis : int (default is 0)
    // Which axis to split on."
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);

    // "A negative value means counting dimensions from the back.
    // Accepted range is [-rank, rank-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, inputDims.size()));

    std::vector<int64_t> tmp(inputDims.size());
    std::iota(tmp.begin(), tmp.end(), 0);
    tmp[axis] = inputDims.size();
    const ShapeTensor subscripts = ShapeTensor(1, std::move(tmp));

    // "split : list of ints"
    // "length of each output"
    std::vector<int> splitList;
    ShapeTensor sizes;
    ShapeTensor sizeSliceAxis;
    const bool hasSplitList = attrs.count("split");
    if (hasSplitList)
    {
        // "Lengths of the parts can be specified using argument split."
        splitList = attrs.get<std::vector<int>>("split");
        ASSERT(static_cast<int>(splitList.size()) == numOutputs, ErrorCode::kINVALID_NODE);
    }
    else
    {
        // "Otherwise, the tensor is split to equal sized parts."
        sizeSliceAxis = floorDiv(ctx, gather(ctx, inputDims, shapeVector(axis)), shapeVector(numOutputs));
        sizes = interlace(ctx, inputDims, sizeSliceAxis, subscripts);
    }

    std::vector<TensorOrWeights> outputs;
    outputs.reserve(numOutputs);

    const ShapeTensor zeros = similar(ctx, inputDims, 0);
    const ShapeTensor ones = similar(ctx, inputDims, 1);
    ShapeTensor starts = zeros;
    ShapeTensor startSliceAxis = shapeVector(0);
    for (int i = 0; i < numOutputs; ++i)
    {
        if (i)
        {
            // Advance from previous start.
            startSliceAxis = add(ctx, startSliceAxis, sizeSliceAxis);
            starts = interlace(ctx, zeros, startSliceAxis, subscripts);
        }
        if (hasSplitList)
        {
            sizeSliceAxis = shapeVector(splitList[i]);
            sizes = interlace(ctx, inputDims, sizeSliceAxis, subscripts);
        }

        nvinfer1::ISliceLayer* slice = addSlice(ctx, inputTensor, starts, sizes, ones);
        ctx->registerLayer(slice, node.name());
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

    OnnxAttrs attrs(node, ctx);
    auto axes = attrs.get<std::vector<int>>("axes");

    int rank = data.getDimensions().nbDims;
    for (auto& axis : axes)
    {
        TRT_CHECK(convertAxis(axis, rank));
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
    // "input : T
    // Input tensor of any shape."
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(input.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    const auto inputDims = shapeOf(input);

    // "repeats : T1
    // 1D int64 tensor of the same length as input's dimension number,
    // includes numbers of repeated copies along input's dimensions.
    const ShapeTensor repeats{inputs.at(1)};

    ShapeTensor outputShape = mul(ctx, inputDims, repeats);
    nvinfer1::ISliceLayer* tile = addSlice(ctx, input, similar(ctx, inputDims, 0), outputShape, similar(ctx, inputDims, 1));
    ctx->registerLayer(tile, node.name());
    tile->setMode(nvinfer1::SliceMode::kWRAP);

    RETURN_FIRST_OUTPUT(tile);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    ASSERT(tensorPtr->getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get("axis", -1);
    int k;
    // Don't support TopK with k as a tensor
    if (ctx->getOpsetVersion() >= 10)
    {
        ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(1).weights().count() == 1, ErrorCode::kUNSUPPORTED_NODE);
        k = *static_cast<int*>(inputs.at(1).weights().values);
    }
    else
    {
        ASSERT(attrs.count("k"), ErrorCode::kINVALID_NODE);
        k = attrs.get<int>("k");
    }

    int nbDims = tensorPtr->getDimensions().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    uint32_t axisMask = 1 << axis;

    bool needToExpandDims = (nbDims == 1);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{1};
        tensorPtr = unsqueezeTensor(ctx, node, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(*tensorPtr, nvinfer1::TopKOperation::kMAX, k, axisMask);
    ctx->registerLayer(layer, node.name());
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* values = layer->getOutput(0);
    nvinfer1::ITensor* indices = layer->getOutput(1);

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{1};
        values = squeezeTensor(ctx, node, *values, axes);
        ASSERT(values, ErrorCode::kUNSUPPORTED_NODE);
        indices = squeezeTensor(ctx, node, *indices, axes);
        ASSERT(indices, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{values, indices}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node, ctx);
    int ndim = input.shape().nbDims;
    ASSERT(ndim <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Permutation default_perm; // Default is to reverse dims
    for (int i = 0; i < ndim; ++i)
    {
        default_perm.order[i] = ndim - 1 - i;
    }
    nvinfer1::Permutation perm = attrs.get("perm", default_perm);
    if (input.is_tensor())
    {
        // Note: Dimension types kept unchanged in order to avoid TRT complaining about CHW order
        nvinfer1::ITensor* output_tensor = transposeTensor(ctx, node, input.tensor(), perm, false);
        ASSERT(output_tensor, ErrorCode::kUNSUPPORTED_NODE);
        return {{output_tensor}};
    }
    else
    {
        auto weights = input.weights();
        auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
        ASSERT(transposeWeights(weights, perm, &new_weights), ErrorCode::kUNSUPPORTED_NODE);
        weights = new_weights;

        return {{weights}};
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Unsqueeze)
{
    // "data : T
    // Original tensor"
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node, ctx);

    // "axes : list of ints (required)
    // List of integers indicating the dimensions to be inserted."
    auto axes = attrs.get<std::vector<int>>("axes");

    // "Negative value means counting dimensions from the back."
    const int newSize = data.getDimensions().nbDims + axes.size();
    for (auto& axis : axes)
    {
        TRT_CHECK(convertAxis(axis, newSize));
    }

    // "expanded : T
    // Reshaped tensor with same data as input."
    auto* expanded = unsqueezeTensor(ctx, node, data, axes, true);

    ASSERT(expanded && "Failed to unsqueeze tensor!", ErrorCode::kUNSUPPORTED_NODE);

    return {{expanded}};
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(tensor.getType() != nvinfer1::DataType::kINT32 && tensor.getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);
    const int nbDims = tensor.getDimensions().nbDims;
    ASSERT(nbDims > 0, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    std::vector<float> scale_factors(nbDims, 1.0f);
    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        // Retrieve and validate scale factors.
        ASSERT(scales_input.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ShapedWeights scales_weights = scales_input.weights();
        ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
        // Scale factors has batch dimension.
        ASSERT(scales_weights.count() == static_cast<size_t>(nbDims), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
        float const* scales_ptr = static_cast<float const*>(scales_weights.values);
        for (int i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales_ptr[i];
        }
    }
    else
    {
        ASSERT(attrs.count("scales"), ErrorCode::kUNSUPPORTED_NODE);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ASSERT(static_cast<int>(scales.size()) == nbDims, ErrorCode::kUNSUPPORTED_NODE);
        for (int i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales[i];
        }
    }
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
    if (mode == "linear")
    {
        ASSERT(canUseLinearResize(scale_factors.size(), &scale_factors.front()),
            ErrorCode::kUNSUPPORTED_NODE);
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }
    // Add resize layer
    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    ctx->registerLayer(layer, node.name());
    layer->setScales(scale_factors.data(), nbDims);
    layer->setResizeMode(resizeMode);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Where)
{
    nvinfer1::ITensor* condition = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* y = &convertToTensor(inputs.at(2), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT(x->getType() == y->getType() && x->getType() != nvinfer1::DataType::kBOOL, ErrorCode::kUNSUPPORTED_NODE);

    TRT_CHECK(broadcastTensors(ctx, x, y, condition));

    nvinfer1::Dims cDims = condition->getDimensions();
    nvinfer1::Dims xDims = x->getDimensions();
    nvinfer1::Dims yDims = y->getDimensions();

    ASSERT(cDims.nbDims == xDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(cDims.nbDims == yDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addSelect(*condition, *x, *y);
    ctx->registerLayer(layer, node.name());

    RETURN_FIRST_OUTPUT(layer);
}

// Copies the given field into the fieldData map, returns data and size of the vector into which the data were copied.
template <typename T>
std::tuple<const void*, size_t> copyField(const T& field, const std::string& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    constexpr size_t nbBytes{sizeof(T)};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), &field, nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

template <typename T>
std::tuple<const void*, size_t> copyField(const std::vector<T>& repeatedField, const std::string& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    const size_t nbBytes{sizeof(T) * repeatedField.size()};
    fieldData[fieldName].resize(nbBytes);
    std::memcpy(fieldData[fieldName].data(), repeatedField.data(), nbBytes);
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<const void*, size_t> copyField(const std::string& field, const std::string& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    static_assert(sizeof(std::string::value_type) == sizeof(uint8_t), "String type does not have 1 byte elements");
    std::copy(field.begin(), field.end(), std::back_inserter(fieldData[fieldName]));
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<const void*, size_t> copyField(const std::vector<std::string>& repeatedField, const std::string& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    static_assert(sizeof(std::string::value_type) == sizeof(uint8_t), "String type does not have 1 byte elements");
    for (const auto& field : repeatedField)
    {
        std::copy(field.begin(), field.end(), std::back_inserter(fieldData[fieldName]));
    }
    return std::make_tuple(fieldData[fieldName].data(), fieldData[fieldName].size());
}

std::tuple<const void*, size_t> copyField(
    const ShapedWeights& field, const std::string& fieldName, string_map<std::vector<uint8_t>>& fieldData)
{
    // Weights do not require a copy
    return std::make_tuple(field.values, field.size_bytes());
}

// Load plugin fields from an ONNX node, using fieldData for temporary allocations.
std::vector<nvinfer1::PluginField> loadFields(string_map<std::vector<uint8_t>>& fieldData, const OnnxAttrs& attrs,
    const nvinfer1::PluginFieldCollection* fieldNames)
{
    std::vector<nvinfer1::PluginField> fields{};
    for (int i = 0; i < fieldNames->nbFields; ++i)
    {
        // Name must be retrieved from the map so that it is alive for long enough.
        const std::string& fieldName = fieldData.emplace(fieldNames->fields[i].name, std::vector<uint8_t>{}).first->first;
        const void* data{nullptr};
        int32_t size{0};
        nvinfer1::PluginFieldType type{};
        switch (attrs.type(fieldName))
        {
            case ::ONNX_NAMESPACE::AttributeProto::FLOAT:
                std::tie(data, size) = copyField(attrs.get<float>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kFLOAT32;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::INT:
                std::tie(data, size) = copyField(attrs.get<int>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kINT32;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::STRING:
                std::tie(data, size) = copyField(attrs.get<std::string>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kCHAR;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::FLOATS:
                std::tie(data, size) = copyField(attrs.get<std::vector<float>>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kFLOAT32;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::INTS:
                std::tie(data, size) = copyField(attrs.get<std::vector<int>>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kINT32;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::STRINGS:
                std::tie(data, size) = copyField(attrs.get<std::vector<std::string>>(fieldName), fieldName, fieldData);
                type = nvinfer1::PluginFieldType::kCHAR;
                break;
            case ::ONNX_NAMESPACE::AttributeProto::TENSOR:
            {
                ShapedWeights tensor{attrs.get<ShapedWeights>(fieldName)};
                std::tie(data, size) = copyField(tensor, fieldName, fieldData);
                size /= getDtypeSize(tensor.type); // size is in bytes
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
                MAKE_ERROR(
                    "Attributes of type: " + ::ONNX_NAMESPACE::AttributeProto::AttributeType_Name(attrs.type(fieldName))
                        + " are unsupported",
                    ErrorCode::kUNSUPPORTED_NODE);
            }
            fields.emplace_back(fieldName.c_str(), data, type, size);
    }
    return std::move(fields);
}

// Any ops that are not supported will attempt to import as plugins.
DEFINE_BUILTIN_OP_IMPORTER(FallbackPluginImporter)
{
    OnnxAttrs attrs(node, ctx);
    const std::string pluginName{node.op_type()};
    const std::string pluginVersion{attrs.get<std::string>("plugin_version", "1")};
    const std::string pluginNamespace{attrs.get<std::string>("plugin_namespace", "")};

    LOG_INFO("Searching for plugin: " << pluginName << ", plugin_version: " << pluginVersion << ", plugin_namespace: " << pluginNamespace);
    nvinfer1::IPluginCreator* creator = importPluginCreator(pluginName, pluginVersion, pluginNamespace);
    ASSERT(creator && "Plugin not found, are the plugin name, version, and namespace correct?", ErrorCode::kUNSUPPORTED_NODE);

    const nvinfer1::PluginFieldCollection* fieldNames = creator->getFieldNames();
    // Field data needs to be type erased, we use fieldData for temporary allocations.
    string_map<std::vector<uint8_t>> fieldData{};
    std::vector<nvinfer1::PluginField> fields = loadFields(fieldData, attrs, fieldNames);

    nvinfer1::IPluginV2* plugin = createPlugin(node.name(), creator, fields);
    ASSERT(plugin && "Could not create plugin", ErrorCode::kUNSUPPORTED_NODE);

    std::vector<nvinfer1::ITensor*> pluginInputs{};
    for (auto& input : inputs)
    {
        pluginInputs.emplace_back(&convertToTensor(input, ctx));
    }
    LOG_INFO("Successfully created plugin: " << pluginName);
    auto* layer = ctx->network()->addPluginV2(pluginInputs.data(), pluginInputs.size(), *plugin);
    ctx->registerLayer(layer, node.name());
    RETURN_ALL_OUTPUTS(layer);
}


// INetwork Serialization importer functions - TODO: Move to it's own file?

DEFINE_BUILTIN_OP_IMPORTER(TRT_Scale)
{
    ASSERT(inputs.size() >= 1, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    if (inputs.size() >= 2)
    {
        ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    }
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int counter = 1;

    nvinfer1::ScaleMode mode = attrs.get<nvinfer1::ScaleMode>("mode");

    // check if there's no weigths at all
    // if no weights, just choose datatype of the input tensor
    // This is based on the assumption that weights should be
    // the same datatype as inputs
    auto type = inputs.size() > 1 ? inputs.at(1).weights().type : trtDataTypeToONNX(inputs.at(0).tensor().getType());

    auto scale = ShapedWeights::empty(type);
    auto shift = ShapedWeights::empty(type);
    auto power = ShapedWeights::empty(type);

    if (attrs.get<bool>("scale"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        scale = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("shift"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        shift = inputs.at(counter++).weights();
    }
    if (attrs.get<bool>("power"))
    {
        ASSERT(inputs.at(counter).is_weights(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        power = inputs.at(counter++).weights();
    }

    nvinfer1::IScaleLayer* layer = ctx->network()->addScale(input, mode, shift, scale, power);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");
    bool zeroIsPlaceholder = attrs.get<bool>("zero_is_placeholder");

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    ctx->registerLayer(layer, node.name());
    layer->setFirstTranspose(perm1);
    layer->setSecondTranspose(perm2);
    layer->setZeroIsPlaceholder(zeroIsPlaceholder);

    if (inputs.size() == 1)
    {
        if (attrs.count("reshape_dims") > 0)
        {
            nvinfer1::Dims reshapeDims = attrs.get<nvinfer1::Dims>("reshape_dims");
            layer->setReshapeDimensions(reshapeDims);
        }
    }
    else
    {
        ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setInput(1, inputs.at(1).tensor());
    }

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_TopK_Min)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kWeights = inputs.at(1).weights();
    int k = *static_cast<int*>(kWeights.values);

    int32_t axes = 1 << (attrs.get<int>("axis"));

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(input, nvinfer1::TopKOperation::kMIN, k, axes);
    ctx->registerLayer(layer, node.name());

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MatMul)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input0 = inputs.at(0).tensor();
    auto& input1 = inputs.at(1).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::MatrixOperation op0 = attrs.get<nvinfer1::MatrixOperation>("op_0");
    nvinfer1::MatrixOperation op1 = attrs.get<nvinfer1::MatrixOperation>("op_1");

    nvinfer1::IMatrixMultiplyLayer* layer = ctx->network()->addMatrixMultiply(input0, op0, input1, op1);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

typedef std::function<void(int, nvinfer1::RNNGateType, nvinfer1::Weights)> RNNWeightsAdder;

bool addRNNv2Weights(RNNWeightsAdder adder, int layerNb, std::vector<nvinfer1::RNNGateType> const& gates,
    std::vector<TensorOrWeights>& inputs, int& counter)
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

    int layerCount = attrs.get<int>("layer_count");
    int hiddenSize = attrs.get<int>("hidden_size");
    int maxSeqLen = attrs.get<int>("max_seq_length");
    nvinfer1::RNNOperation op = attrs.get<nvinfer1::RNNOperation>("rnn_op");
    nvinfer1::RNNInputMode inputMode = attrs.get<nvinfer1::RNNInputMode>("input_mode");
    nvinfer1::RNNDirection direction = attrs.get<nvinfer1::RNNDirection>("direction");

    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    int counter = 1;
    nvinfer1::IRNNv2Layer* layer = ctx->network()->addRNNv2(input, layerCount, hiddenSize, maxSeqLen, op);
    ctx->registerLayer(layer, node.name());
    layer->setInputMode(inputMode);
    layer->setDirection(direction);

    if (attrs.get<bool>("has_hidden_state"))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setHiddenState(inputs.at(counter++).tensor());
    }
    if (op == nvinfer1::RNNOperation::kLSTM && attrs.get<bool>("has_cell_state", false))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setCellState(inputs.at(counter++).tensor());
    }
    if (attrs.get<bool>("has_seq_lengths"))
    {
        ASSERT(inputs.at(counter).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
        layer->setSequenceLengths(inputs.at(counter++).tensor());
    }

    int nbWeights = (direction == nvinfer1::RNNDirection::kBIDIRECTION ? 2 : 1) * layerCount;
    const int K = direction == nvinfer1::RNNDirection::kUNIDIRECTION ? 1 : 2;
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

    RNNWeightsAdder weightsAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentWeightsAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setWeightsForGate(n, gate, false, weights);
    };
    RNNWeightsAdder biasAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, true, weights);
    };
    RNNWeightsAdder recurrentBiasAdder = [&layer](int n, nvinfer1::RNNGateType gate, nvinfer1::Weights weights) {
        layer->setBiasForGate(n, gate, false, weights);
    };

    for (int n = 0; n < nbWeights; ++n)
    {
        if (n >= K || inputMode == nvinfer1::RNNInputMode::kLINEAR)
        {
            ASSERT(addRNNv2Weights(weightsAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        }
        ASSERT(
            addRNNv2Weights(recurrentWeightsAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(biasAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
        ASSERT(addRNNv2Weights(recurrentBiasAdder, n, gates, inputs, counter), nvonnxparser::ErrorCode::kINVALID_NODE);
    }

    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_RaggedSoftmax)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();
    auto& bounds = inputs.at(1).tensor();

    nvinfer1::IRaggedSoftMaxLayer* layer = ctx->network()->addRaggedSoftMax(input, bounds);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_FullyConnected)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int nbChannels = attrs.get<int>("channels");

    ASSERT(inputs.at(1).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& kernelWeights = inputs.at(1).weights();

    ShapedWeights biasWeights = ShapedWeights::empty(kernelWeights.type);
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), nvonnxparser::ErrorCode::kINVALID_NODE);
        biasWeights = inputs.at(2).weights();
    }

    nvinfer1::IFullyConnectedLayer* layer
        = ctx->network()->addFullyConnected(input, nbChannels, kernelWeights, biasWeights);
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;
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
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    ctx->registerLayer(layer, node.name());
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(exclude_padding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(begPadding);
    layer->setPostPadding(endPadding);

    layer->setBlendFactor(blend);

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_PluginV2)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        ASSERT(input.is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        tensors.push_back(&input.tensor());
    }
    OnnxAttrs attrs(node, ctx);

    nvinfer1::IPluginRegistry* registry = getPluginRegistry();

    std::string name = attrs.get<std::string>("name");
    std::string version = attrs.get<std::string>("version");
    std::string nspace = attrs.get<std::string>("namespace");
    std::string buffer = attrs.get<std::string>("data");

    nvinfer1::IPluginCreator* creator = registry->getPluginCreator(name.c_str(), version.c_str(), nspace.c_str());
    ASSERT(creator != nullptr, nvonnxparser::ErrorCode::kINVALID_NODE);

    nvinfer1::IPluginV2* plugin = creator->deserializePlugin("", buffer.data(), buffer.size());

    nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
    ctx->registerLayer(layer, node.name());
    RETURN_ALL_OUTPUTS(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Gather)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& data = inputs.at(0).tensor();
    auto& indices = inputs.at(1).tensor();
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    int nbElementWiseDims = attrs.get<int>("nbElementWiseDims", 0);
    int r = data.getDimensions().nbDims;

    ASSERT(indices.getType() == nvinfer1::DataType::kINT32, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(axis != -r, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(r >= 1, nvonnxparser::ErrorCode::kINVALID_NODE);
    ASSERT(-r <= axis && axis <= r, nvonnxparser::ErrorCode::kINVALID_NODE);

    if (axis < 0)
    {
        axis += r;
    }

    nvinfer1::IGatherLayer* layer = ctx->network()->addGather(data, indices, axis);
    ctx->registerLayer(layer, node.name());
    layer->setNbElementWiseDims(nbElementWiseDims);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Slice)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
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
        ASSERT(inputs.size() == 4, nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
        const ShapeTensor start{inputs.at(1)};
        const ShapeTensor size{inputs.at(2)};
        const ShapeTensor stride{inputs.at(3)};
        layer = addSlice(ctx, input, start, size, stride);
    }
    ctx->registerLayer(layer, node.name());
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Resize)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = ctx->network()->addResize(input);
    ctx->registerLayer(layer, node.name());

    OnnxAttrs attrs(node, ctx);
    auto alignCorners = attrs.get<bool>("align_corners", false);
    auto mode = attrs.get<nvinfer1::ResizeMode>("mode");
    layer->setAlignCorners(alignCorners);
    layer->setResizeMode(mode);

    if (inputs.size() == 1)
    {
        auto outputDims = attrs.get<nvinfer1::Dims>("output_dims", nvinfer1::Dims{-1, {}});
        if (outputDims.nbDims > 0)
        {
            layer->setOutputDimensions(outputDims);
        }
        else
        {
            auto scales = attrs.get<std::vector<float>>("scales");
            ASSERT(scales.size() > 0, nvonnxparser::ErrorCode::kINVALID_NODE);
            layer->setScales(&scales[0], scales.size());
        }
    }
    else
    {
        ASSERT(inputs.at(1).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
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
