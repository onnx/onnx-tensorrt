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

#include "builtin_op_importers.hpp"
#include "ModelImporter.hpp"
#include "NvInferPlugin.h"
#include "OnnxAttrs.hpp"
#include "ShapeTensor.hpp"
#include "onnx2trt_utils.hpp"

#include <algorithm> // For std::min, std::max
#include <array>
#include <cmath>
#include <cstring> // For std::memcpy, std::memset
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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acos)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kACOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acosh)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kACOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(Add)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(And)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kAND);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMax)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(ArgMin)
{
    return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asin)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kASIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asinh)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kASINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Atan)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kATAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Atanh)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kATANH);
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

        scale = unsqueezeTensor(ctx, *scale, axes);
        bias = unsqueezeTensor(ctx, *bias, axes);
        mean = unsqueezeTensor(ctx, *mean, axes);
        variance = unsqueezeTensor(ctx, *variance, axes);
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

    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
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

    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);

    OnnxAttrs attrs(node, ctx);
    float eps = attrs.get<float>("epsilon", 1e-5f);

    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensor_ptr = unsqueezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
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
    if (!need_to_expand_dims)
    {
        return scaleHelper(
            ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
    }
    else
    {
        auto scaledResult = scaleHelper(
            ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
        // Squeeze spatial dims back to 1D
        tensor_ptr = &convertToTensor(scaledResult.value().at(0), ctx);
        std::vector<int> axes{3};
        tensor_ptr = squeezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        return {{tensor_ptr}};
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
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kCEIL);
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
        tensors.push_back(&convertToTensor(input, ctx));
    }
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis");
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setAxis(axis);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Constant)
{
    // TODO: This silently fails if the dtype is not supported
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
    return {{constantOfShape(ctx, value, shape)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv)
{
    ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    if (inputs.at(1).is_tensor())
    {
        ASSERT(inputs.at(1).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
        if (inputs.size() == 3)
        {
            ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        }
        // Handle Multiinput convolution
        return convMultiInput(ctx, node, inputs);
    }

    // Convolution Weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);

    auto kernel_weights = inputs.at(1).weights();

    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    LOG_VERBOSE("Convolution input dimensions: " << dims);

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int> axes{3};
        tensor_ptr = unsqueezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }
    if (kernel_weights.shape.nbDims == 3)
    {
        kernel_weights.shape.nbDims = 4;
        kernel_weights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernel_weights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shaped_bias_weights = inputs.at(2).weights();
        ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[0], ErrorCode::kINVALID_NODE);
        bias_weights = shaped_bias_weights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernel_weights.type);
    }
    nvinfer1::Dims kernel_size;
    kernel_size.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        kernel_size.d[nbSpatialDims - i] = kernel_weights.shape.d[kernel_weights.shape.nbDims - i];
    }
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding;
    getKernelParams(
        ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding, &dilations);

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernel_size.d[nbSpatialDims - i] == kernel_weights.shape.d[kernel_weights.shape.nbDims - i],
            ErrorCode::kUNSUPPORTED_NODE);
    }

    int nchan = dims.d[1];
    int noutput = kernel_weights.shape.d[0];
    nvinfer1::IConvolutionLayer* layer
        = ctx->network()->addConvolutionNd(*tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);

    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
    layer->setDilationNd(dilations);
    OnnxAttrs attrs(node, ctx);
    int ngroup = attrs.get("group", 1);
    ASSERT(nchan == -1 || kernel_weights.shape.d[1] * ngroup == nchan, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{3};
        tensor_ptr = squeezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }

    LOG_VERBOSE("Using kernel: " << kernel_size << ", strides: " << strides << ", padding: " << beg_padding
                                 << ", dilations: " << dilations << ", numOutputs: " << noutput);
    LOG_VERBOSE("Convolution output dimensions: " << dims);
    return {{tensor_ptr}};
}

// TRT only supports 2D or 3D deconvolutions (Layout: [N,C,D1,D2,(D3)])
// Inputs should be of dimension 4 or 5.
// When input.nbDims = 3, we expand it to 4D
DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose)
{
    // Deconvolution input must be at least 3D.
    ASSERT(inputs.at(0).shape().nbDims >= 3, ErrorCode::kUNSUPPORTED_NODE);
    // Deconvolution weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);

    // Kernel weights have layout [C, M/group, k1, k2, (k3)]
    auto kernel_weights = inputs.at(1).weights();
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        std::vector<int> axes{3};
        tensor_ptr = unsqueezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }
    if (kernel_weights.shape.nbDims == 3)
    {
        kernel_weights.shape.nbDims = 4;
        kernel_weights.shape.d[3] = 1;
    }

    const int nbSpatialDims = dims.nbDims - 2;
    // Check that the number of spatial dimensions and the kernel shape matches up.
    ASSERT(nbSpatialDims == kernel_weights.shape.nbDims - 2, ErrorCode::kUNSUPPORTED_NODE);

    // Check for bias_weights
    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3)
    {
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        auto shaped_bias_weights = inputs.at(2).weights();
        // ONNX requires shaped_bias_weights to be 1D
        ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
        ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[1], ErrorCode::kINVALID_NODE);
        bias_weights = shaped_bias_weights;
    }
    else
    {
        bias_weights = ShapedWeights::empty(kernel_weights.type);
    }

    // Get all attributes
    OnnxAttrs attrs(node, ctx);
    nvinfer1::Dims output_shape;
    nvinfer1::Dims output_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims kernel_size;
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims dilations = makeDims(nbSpatialDims, 1);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding = false;
    bool explicit_output_shape = false;

    int ngroup = attrs.get("group", 1);
    int noutput = kernel_weights.shape.d[1] * ngroup; // Note: Weights order is CKRS

    if (attrs.count("output_shape"))
    {
        output_shape = attrs.get<nvinfer1::Dims>("output_shape");
        explicit_output_shape = true;
    }

    kernel_size.nbDims = nbSpatialDims;
    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        kernel_size.d[nbSpatialDims - i] = kernel_weights.shape.d[kernel_weights.shape.nbDims - i];
    }

    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding,
        &dilations, &output_padding);
    // TRT only support 2D padding
    ASSERT(output_padding.nbDims == 2 || (output_padding.nbDims == 3 && output_padding.d[0] == 0),
        ErrorCode::kUNSUPPORTED_NODE);

    for (int i = 1; i <= nbSpatialDims; ++i)
    {
        ASSERT(kernel_size.d[nbSpatialDims - i] == kernel_weights.shape.d[kernel_weights.shape.nbDims - i],
            ErrorCode::kUNSUPPORTED_NODE);
        // TRT does not support dilated deconvolutions
        ASSERT(dilations.d[nbSpatialDims - i] == 1, ErrorCode::kUNSUPPORTED_GRAPH);
    }
    // If output shape is given, calculate the input or output padding values
    if (explicit_output_shape)
    {
        generatePadding(dims, output_shape, kernel_size, strides, dilations, nbSpatialDims, beg_padding, end_padding,
            output_padding, paddingMode);
        // TRT only support 2D padding
        ASSERT(output_padding.nbDims == 2 || (output_padding.nbDims == 3 && output_padding.d[0] == 0),
            ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::IDeconvolutionLayer* deconv_layer
        = ctx->network()->addDeconvolutionNd(*tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);
    ASSERT(deconv_layer, ErrorCode::kUNSUPPORTED_NODE);

    deconv_layer->setStrideNd(strides);
    deconv_layer->setPaddingMode(paddingMode);
    deconv_layer->setPrePadding(beg_padding);
    deconv_layer->setPostPadding(end_padding);
    deconv_layer->setNbGroups(ngroup);
    tensor_ptr = deconv_layer->getOutput(0);

    nvinfer1::Dims output_padding_HW;
    if (output_padding.nbDims == 2)
    {
        output_padding_HW = nvinfer1::Dims{2, {output_padding.d[0], output_padding.d[1]}, {}};
    }
    else
    {
        output_padding_HW = nvinfer1::Dims{2, {output_padding.d[1], output_padding.d[2]}, {}};
    }

    if (output_padding_HW != nvinfer1::Dims{2, {0, 0}, {}})
    {
        tensor_ptr
            = ctx->network()->addPaddingNd(*tensor_ptr, nvinfer1::Dims{2, {}, {}}, output_padding_HW)->getOutput(0);
    }

    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        std::vector<int> axes{3};
        tensor_ptr = squeezeTensor(ctx, *tensor_ptr, axes);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Cos)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kCOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Cosh)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kCOSH);
}

NodeImportResult depthToSpaceDynamicImporter(
    IImporterContext* ctx, const int blockSize, const std::string mode, nvinfer1::ITensor*& input)
{
    using eOp = nvinfer1::ElementWiseOperation;
    const auto N = getAxisLength(ctx, input, 0, {1, {1}});
    const auto C = getAxisLength(ctx, input, 1, {1, {1}});
    const auto H = getAxisLength(ctx, input, 2, {1, {1}});
    const auto W = getAxisLength(ctx, input, 3, {1, {1}});
    const int DCRPerm[6] = {0, 3, 4, 1, 5, 2};
    const int CRDPerm[6] = {0, 1, 4, 2, 5, 3};
    auto* blockSizeTensor
        = addConstantScalar(ctx, blockSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, {1, {1}})->getOutput(0);
    auto* blockSizeTensorSquared
        = addConstantScalar(ctx, blockSize * blockSize, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, {1, {1}})
              ->getOutput(0);
    nvinfer1::Permutation perm{};
    auto* firstShuffle = ctx->network()->addShuffle(*input);
    std::vector<nvinfer1::ITensor*> firstShapeDims;

    if (mode == "DCR")
    {
        // Concat shape tensors {N, blockSize, blockSize, C / (blockSize * blockSize), H, W}
        firstShapeDims = {N, blockSizeTensor, blockSizeTensor,
            ctx->network()->addElementWise(*C, *blockSizeTensorSquared, eOp::kDIV)->getOutput(0), H, W};
        std::copy(std::begin(DCRPerm), std::end(DCRPerm), std::begin(perm.order));
    }
    else
    {
        // Concat shape tensors: {N, C / (blockSize * blockSize), blockSize, blockSize, H, W};
        firstShapeDims = {N, ctx->network()->addElementWise(*C, *blockSizeTensorSquared, eOp::kDIV)->getOutput(0),
            blockSizeTensor, blockSizeTensor, H, W};
        std::copy(std::begin(CRDPerm), std::end(CRDPerm), std::begin(perm.order));
    }
    auto* firstShape = ctx->network()->addConcatenation(firstShapeDims.data(), firstShapeDims.size())->getOutput(0);
    firstShuffle->setInput(1, *firstShape);
    firstShuffle->setSecondTranspose(perm);
    input = firstShuffle->getOutput(0);
    auto* secondShuffle = ctx->network()->addShuffle(*input);

    // Finally reshape to {N, C / (blockSize * blockSize), H * blockSize, W * blockSize}
    std::vector<nvinfer1::ITensor*> finalDims{
        N, ctx->network()->addElementWise(*C, *blockSizeTensorSquared, eOp::kDIV)->getOutput(0),
        ctx->network()->addElementWise(*H, *blockSizeTensor, eOp::kPROD)->getOutput(0),
        ctx->network()->addElementWise(*W, *blockSizeTensor, eOp::kPROD)->getOutput(0),
    };

    auto* finalShape = ctx->network()->addConcatenation(finalDims.data(), finalDims.size())->getOutput(0);
    secondShuffle->setInput(1, *finalShape);

    RETURN_FIRST_OUTPUT(secondShuffle);
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    // Input tensor is in NCHW format
    ASSERT(inputs.at(0).shape().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    auto dims = tensorPtr->getDimensions();

    // Extract attributes
    OnnxAttrs attrs(node, ctx);
    auto blockSize = attrs.get<int>("blocksize");
    auto mode = attrs.get<std::string>("mode", "DCR");

    // Useful constants
    const auto N = dims.d[0];
    const auto C = dims.d[1];
    const auto H = dims.d[2];
    const auto W = dims.d[3];
    const int DCRPerm[6] = {0, 3, 4, 1, 5, 2};
    const int CRDPerm[6] = {0, 1, 4, 2, 5, 3};

    // Use the dynamic importer if any non-batch dimensions are dynamic.
    if (C == -1 || H == -1 || W == -1)
    {
        return depthToSpaceDynamicImporter(ctx, blockSize, mode, tensorPtr);
    }

    nvinfer1::Dims newDims{6};
    nvinfer1::Permutation perm{};

    if (mode == "DCR")
    {
        const int DCRDims[6] = {N, blockSize, blockSize, C / (blockSize * blockSize), H, W};
        std::copy(std::begin(DCRDims), std::end(DCRDims), std::begin(newDims.d));
        std::copy(std::begin(DCRPerm), std::end(DCRPerm), std::begin(perm.order));
    }
    else
    {
        const int CRDDims[6] = {N, C / (blockSize * blockSize), blockSize, blockSize, H, W};
        std::copy(std::begin(CRDDims), std::end(CRDDims), std::begin(newDims.d));
        std::copy(std::begin(CRDPerm), std::end(CRDPerm), std::begin(perm.order));
    }

    auto* firstShuffle = ctx->network()->addShuffle(*tensorPtr);
    firstShuffle->setReshapeDimensions(newDims);
    firstShuffle->setSecondTranspose(perm);
    tensorPtr = firstShuffle->getOutput(0);

    auto secondShuffle = ctx->network()->addShuffle(*tensorPtr);
    nvinfer1::Dims finalDims{4, {N, C / (blockSize * blockSize), H * blockSize, W * blockSize}};
    secondShuffle->setReshapeDimensions(finalDims);
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
    auto shift = inputs.at(2).weights();
    // Override zero points type to float
    shift.type = type;
    auto power = ShapedWeights::empty(type);

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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kERF);
}

DEFINE_BUILTIN_OP_IMPORTER(Exp)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Expand)
{
    // "Broadcast the input tensor following the given shape and the broadcast rule."
    nvinfer1::ITensor* inputTensor = &convertToTensor(inputs.at(0), ctx);
    ShapeTensor inputDims = shapeOf(ctx, *inputTensor);

    // "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
    ASSERT(inputs.at(1).shape().nbDims == 1, ErrorCode::kINVALID_VALUE);
    ShapeTensor shape{inputs.at(1)};

    // "Dimensions are right alignment;..."
    if (shape.size > inputDims.size)
    {
        ShapeTensor newDims = concat(ctx, fillShapeVector(shape.size - inputDims.size, 1), inputDims);
        inputTensor = addShuffle(ctx, *inputTensor, newDims)->getOutput(0);
        inputDims = newDims;
    }
    else if (shape.size < inputDims.size)
    {
        // ", or the shape.ndim < input.shape.ndim"
        shape = concat(ctx, fillShapeVector(inputDims.size - shape.size, 1), shape);
    }

    const ShapeTensor starts = similar(inputDims, 0);
    const ShapeTensor sizes = max(ctx, inputDims, shape);
    // Compute (x > 1 ? 1 : 0) for x in inputDims, assuming positive x, using only TensorRT operations.
    const ShapeTensor one = shapeVector(1);
    const ShapeTensor strides = min(ctx, one, sub(ctx, inputDims, one));

    nvinfer1::ISliceLayer* sliceLayer = addSlice(ctx, *inputTensor, starts, sizes, strides);

    RETURN_FIRST_OUTPUT(sliceLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node, ctx);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    int nbDims = tensor_ptr->getDimensions().nbDims;
    int axis = attrs.get("axis", 1);
    TRT_CHECK(convertAxis(axis, nbDims));

    if (nbDims > 2)
    {
        tensor_ptr = flattenTensor(ctx, *tensor_ptr, axis);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor& indices = convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convertAxis(axis, nbDims));
    LOG_VERBOSE("Using Gather axis: " << axis);
    RETURN_FIRST_OUTPUT(ctx->network()->addGather(data, indices, axis));
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);

    // Use FC if it is likely to be faster - which is usually when no Shuffles are required.
    bool canUseFC = inputs.at(0).is_tensor() && inputs.at(1).is_weights() && inputs.at(2).is_weights() && alpha == 1.f
        && beta == 1.f && inputs.at(0).tensor().getDimensions().nbDims == 3 && inputs.at(1).weights().shape.nbDims == 2
        && inputs.at(2).weights().shape.nbDims == 1;
    if (canUseFC)
    {
        LOG_VERBOSE("GEMM: using FC layer instead of MM because all criteria were met.");
        nvinfer1::ITensor& tensor = inputs.at(0).tensor();
        ShapedWeights weights = inputs.at(1).weights();
        if (!transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            weights = transposedWeights;
        }
        ShapedWeights biases = inputs.at(2).weights();
        RETURN_FIRST_OUTPUT(ctx->network()->addFullyConnected(tensor, biases.shape.d[0], weights, biases));
    }

    // If input B is a constant, we transpose at parse time if necessary,
    // because In some cases, A * Bt is much slower than A * B.
    if (inputs.at(1).is_weights())
    {
        ShapedWeights weights = inputs.at(1).weights();
        if (transB)
        {
            auto transposedWeights = ctx->createTempWeights(weights.type, weights.shape);
            ASSERT(transposeWeights(weights, {1, 0}, &transposedWeights), ErrorCode::kUNSUPPORTED_NODE);
            weights = transposedWeights;
            // Since we've already transposed now, we can set transpose to false.
            transB = false;
        }
        nvinfer1::IConstantLayer* weightsLayer
            = ctx->network()->addConstant(weights.shape, static_cast<nvinfer1::Weights>(weights));
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
        inputASqueezed = squeeze->getOutput(0);
    }

    constexpr auto getMatrixOp = [](const nvinfer1::ITensor& input, bool transpose) {
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
    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        broadcastTensors(ctx, alphaConstantTensor, matmulTensor);
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
            broadcastTensors(ctx, betaConstantTensor, biasTensor);
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
        broadcastTensors(ctx, matmulTensor, biasTensor);
        nvinfer1::IElementWiseLayer* biasAdd
            = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
        return {{biasAdd->getOutput(0)}};
    }

    return {{matmulTensor}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    int nbSpatialDims = dims.nbDims - 2;
    ASSERT((nbSpatialDims == 2 || nbSpatialDims == 3) && "TensorRT only supports 2D and 3D pooling!",
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernelSize = nbSpatialDims == 2
        ? static_cast<nvinfer1::Dims>(nvinfer1::Dims2{dims.d[2], dims.d[3]})
        : static_cast<nvinfer1::Dims>(nvinfer1::Dims3{dims.d[2], dims.d[3], dims.d[4]});
    ASSERT(!isDynamic(kernelSize) && "Cannot run global average pool on an input with dynamic spatial dimensions!",
        ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(ctx->network()->addPoolingNd(tensor, nvinfer1::PoolingType::kAVERAGE, kernelSize));
}

// TODO: GlobalLpPool: pow(reduce_mean(pow(abs(x), p)), 1./p)

DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    int nbSpatialDims = dims.nbDims - 2;
    ASSERT((nbSpatialDims == 2 || nbSpatialDims == 3) && "TensorRT only supports 2D and 3D pooling!",
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernelSize = nbSpatialDims == 2
        ? static_cast<nvinfer1::Dims>(nvinfer1::Dims2{dims.d[2], dims.d[3]})
        : static_cast<nvinfer1::Dims>(nvinfer1::Dims3{dims.d[2], dims.d[3], dims.d[4]});
    ASSERT(!isDynamic(kernelSize) && "Cannot run global average pool on an input with dynamic spatial dimensions!",
        ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(ctx->network()->addPoolingNd(tensor, nvinfer1::PoolingType::kMAX, kernelSize));
}

DEFINE_BUILTIN_OP_IMPORTER(Greater)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kGREATER);
}

// singlePassShape is the shape of the output from a single pass.
nvinfer1::ITensor* concatenateRNNOutputs(IImporterContext* ctx, nvinfer1::ILoop* loop,
    nvinfer1::ITensor* singlePassShape, nvinfer1::ITensor* sequenceLength, nvinfer1::ITensor* concatenatedOutput,
    int numDirections)
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

        nvinfer1::ILoopOutputLayer* forwardOutput
            = loop->addLoopOutput(*HtForwardLayer->getOutput(0), nvinfer1::LoopOutput::kCONCATENATE, 0);
        forwardOutput->setInput(1, *sequenceLength);
        nvinfer1::ILoopOutputLayer* reverseOutput
            = loop->addLoopOutput(*HtBackwardLayer->getOutput(0), nvinfer1::LoopOutput::kREVERSE, 0);
        reverseOutput->setInput(1, *sequenceLength);

        std::array<nvinfer1::ITensor*, 2> passes{{forwardOutput->getOutput(0), reverseOutput->getOutput(0)}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(passes.data(), passes.size());
        concat->setAxis(1);
        yOutput = concat->getOutput(0);
    }
    else
    {
        nvinfer1::ILoopOutputLayer* scanOut
            = loop->addLoopOutput(*concatenatedOutput, nvinfer1::LoopOutput::kCONCATENATE, 0);
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
        // Unsqueeze so we can broadcast later
        nvinfer1::ITensor* concatenatedBias = &convertToTensor(inputs.at(3), ctx);
        nvinfer1::IShuffleLayer* unsqueeze = net->addShuffle(*concatenatedBias);
        unsqueeze->setReshapeDimensions(Dims3{1, numDirections, 2 * NUM_GATES * hiddenSize});
        concatenatedBias = unsqueeze->getOutput(0);

        biasZR
            = net->addSlice(*concatenatedBias, Dims3{0, 0, 0}, Dims3{1, numDirections, 2 * hiddenSize}, Dims3{1, 1, 1})
                  ->getOutput(0);
        LOG_VERBOSE("Bias for ZR gates shape is: " << biasZR->getDimensions());
        biasH = net->addSlice(*concatenatedBias, Dims3{0, 0, 2 * hiddenSize}, Dims3{1, numDirections, hiddenSize},
                       Dims3{1, 1, 1})
                    ->getOutput(0);
        LOG_VERBOSE("Bias for H gate shape is: " << biasH->getDimensions());

        recurrenceBiasZR = net->addSlice(*concatenatedBias, Dims3{0, 0, NUM_GATES * hiddenSize},
                                  Dims3{1, numDirections, 2 * hiddenSize}, Dims3{1, 1, 1})
                               ->getOutput(0);
        LOG_VERBOSE("Recurrence bias for ZR gates shape is: " << recurrenceBiasZR->getDimensions());
        recurrenceBiasH = net->addSlice(*concatenatedBias, Dims3{0, 0, (NUM_GATES + 2) * hiddenSize},
                                 Dims3{1, numDirections, hiddenSize}, Dims3{1, 1, 1})
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
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse
    // iterator must be concatenated.
    nvinfer1::ITensor* iterationInput{nullptr};
    if (direction == "forward")
    {
        iterationInput = unsqueezeTensor(ctx, *loop->addIterator(*input)->getOutput(0), std::vector<int>{0});
    }
    else if (direction == "reverse")
    {
        nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
        reverseIterator->setReverse(true);
        iterationInput = unsqueezeTensor(ctx, *reverseIterator->getOutput(0), std::vector<int>{0});
    }
    else
    {
        ASSERT(direction == "bidirectional", ErrorCode::kINVALID_NODE);
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        nvinfer1::IConcatenationLayer* concat = net->addConcatenation(
            std::array<nvinfer1::ITensor*, 2>{{unsqueezeTensor(ctx, *forward->getOutput(0), std::vector<int>{0}),
                                                  unsqueezeTensor(ctx, *reverse->getOutput(0), std::vector<int>{0})}}
                .data(),
            2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());

    // H(t-1)
    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx,
            addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT, Dims{1, 1})->getOutput(0),
            gateOutputShape);
    };

    nvinfer1::ITensor* initialHidden = getInitialInputValue(5);
    LOG_VERBOSE("Initial hidden state shape: " << initialHidden->getDimensions());

    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
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

    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*gateOutputShape,
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
                       ->getOutput(0),
                  nvinfer1::ElementWiseOperation::kDIV)
              ->getOutput(0);
    outputs.emplace_back(
        concatenateRNNOutputs(ctx, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections));
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
    RETURN_FIRST_OUTPUT(ctx->network()->addIdentity(convertToTensor(inputs.at(0), ctx)));
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
    RETURN_FIRST_OUTPUT(ctx->network()->addScale(
        tensor, nvinfer1::ScaleMode::kCHANNEL, shiftWeights, scaleWeights, nvinfer1::Weights{}));
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization)
{
    // Scales and biases must be initializers
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    ASSERT(!isDynamic(tensor_ptr->getDimensions()) && "InstanceNormalization does not support dynamic inputs!",
        ErrorCode::kUNSUPPORTED_NODE);
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    OnnxAttrs attrs(node, ctx);
    float epsilon = attrs.get("epsilon", 1e-5f);
    // TensorRT only supports epsilon values >= 1e-4.
    epsilon = std::max(epsilon, 1e-4f);

    // Populate instanceNormalization plugin properties.
    const std::string pluginName = "InstanceNormalization_TRT";
    const std::string pluginVersion = "001";
    std::vector<nvinfer1::PluginField> f;
    f.emplace_back("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    f.emplace_back("scales", scale_weights.values, nvinfer1::PluginFieldType::kFLOAT32, scale_weights.count());
    f.emplace_back("bias", bias_weights.values, nvinfer1::PluginFieldType::kFLOAT32, bias_weights.count());

    // Create plugin from registry
    nvinfer1::IPluginV2* plugin = importPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);

    ASSERT(plugin != nullptr && "InstanceNormalization plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    RETURN_FIRST_OUTPUT(ctx->network()->addPluginV2(&tensor_ptr, 1, *plugin));
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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    // Don't use softmax converter since it adds a shuffle layer
    // which prevents the builder to fuse softmax and log operations.

    OnnxAttrs attrs(node, ctx);
    // "input : T"
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const ShapeTensor dims = shapeOf(ctx, input);
    // "axis : int (default is 1)"
    int axis = attrs.get("axis", 1);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, dims.size));

    // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
    auto* flattened = flattenTensor(ctx, input, axis);
    auto* softMax = ctx->network()->addSoftMax(*flattened);
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
        loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);
        ctx->registerTensor(tripLimit, body.input(0).name());
    }
    if (inputs[1])
    {
        nvinfer1::ITensor* cond = convertToScalar(ctx, &convertToTensor(inputs[1], ctx));
        ASSERT(cond, ErrorCode::kINVALID_NODE);
        loop->addTripLimit(*cond, nvinfer1::TripLimit::kWHILE);
        ctx->registerTensor(cond, body.input(1).name());
    }
    // Add initial state inputs using recurrent layers.
    std::vector<nvinfer1::IRecurrenceLayer*> stateVars{};
    for (size_t i = 2; i < inputs.size(); ++i)
    {
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs[i], ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }

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
    // Finally, set up scan outputs.
    for (int i = nbStateVars + NB_DISCARDED_OUTPUTS; i < nbInputs; ++i)
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
    float alpha = attrs.get<float>("alpha", 0.0001);
    float beta = attrs.get<float>("beta", 0.75);
    float bias = attrs.get<float>("bias", 1.0);
    RETURN_FIRST_OUTPUT(ctx->network()->addLRN(tensor, size, alpha, beta, bias));
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

    const auto getInitialInputValue = [&ctx, &gateOutputShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx,
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
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse
    // iterator must be concatenated.
    nvinfer1::ITensor* iterationInput{nullptr};
    if (direction == "forward")
    {
        iterationInput = unsqueezeTensor(ctx, *loop->addIterator(*input)->getOutput(0), std::vector<int>{0});
    }
    else if (direction == "reverse")
    {
        nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
        reverseIterator->setReverse(true);
        iterationInput = unsqueezeTensor(ctx, *reverseIterator->getOutput(0), std::vector<int>{0});
    }
    else
    {
        ASSERT(direction == "bidirectional", ErrorCode::kINVALID_NODE);
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        std::array<nvinfer1::ITensor*, 2> tensors{{unsqueezeTensor(ctx, *forward->getOutput(0), std::vector<int>{0}),
            unsqueezeTensor(ctx, *reverse->getOutput(0), std::vector<int>{0})}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());

    // H(t-1)
    nvinfer1::IRecurrenceLayer* Ht1 = loop->addRecurrence(*initialHidden);
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

    const auto addPeephole = [&ctx, &hiddenSize, &numDirections, &peephole](
        nvinfer1::ITensor* gate, nvinfer1::ITensor* cellState, int gateIndex) -> nvinfer1::ITensor* {
        nvinfer1::ISliceLayer* isolatePeephole
            = ctx->network()->addSlice(*peephole, nvinfer1::Dims2{0, gateIndex * hiddenSize},
                nvinfer1::Dims2{numDirections, hiddenSize}, nvinfer1::Dims2{1, 1});
        auto* peepholeWeights = unsqueezeTensor(ctx, *isolatePeephole->getOutput(0), std::vector<int>{1});
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
    Ht1->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*gateOutputShape,
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
                       ->getOutput(0),
                  eOp::kDIV)
              ->getOutput(0);
    outputs.emplace_back(
        concatenateRNNOutputs(ctx, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections));
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

    broadcastTensors(ctx, inputA, inputB);

    constexpr auto getMatrixOp = [](const nvinfer1::ITensor& input) {
        return (input.getDimensions().nbDims == 1) ? nvinfer1::MatrixOperation::kVECTOR
                                                   : nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB);

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputA, opA, *inputB, opB);
    return {{matmul->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Max)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool)
{
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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(Not)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNOT);
}

DEFINE_BUILTIN_OP_IMPORTER(Or)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims2 beg_padding, end_padding;
    OnnxAttrs attrs(node, ctx);
    auto mode = attrs.get<std::string>("mode", "constant");
    float value = attrs.get<float>("value", 0.f);
    ASSERT(mode == "constant" && value == 0 && "This version of TensorRT only supports constant 0 padding!",
        ErrorCode::kUNSUPPORTED_NODE);
    auto onnx_padding = attrs.get<std::vector<int>>("pads");
    ASSERT(onnx_padding.size() == 8 && onnx_padding[0] == 0 && onnx_padding[1] == 0 && onnx_padding[4] == 0
            && onnx_padding[5] == 0
            && "This version of TensorRT only supports padding on the outer two dimensions on 4D tensors!",
        ErrorCode::kUNSUPPORTED_NODE);
    beg_padding.d[0] = onnx_padding[2];
    beg_padding.d[1] = onnx_padding[3];
    end_padding.d[0] = onnx_padding[6];
    end_padding.d[1] = onnx_padding[7];
    RETURN_FIRST_OUTPUT(ctx->network()->addPaddingNd(tensor, beg_padding, end_padding));
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
    broadcastTensors(ctx, input, slopes);
    RETURN_FIRST_OUTPUT(ctx->network()->addParametricReLU(*input, *slopes));
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
    auto shift = inputs.at(2).weights();
    // Override zero points type to float
    shift.type = type;
    auto power = ShapedWeights::empty(type);

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

NodeImportResult randomUniformHelper(IImporterContext* ctx, const ShapeTensor& inputShape, const OnnxAttrs& attrs,
    const nvinfer1::DataType& inputDType)
{
    auto* fillLayer = addFill(ctx, inputShape, nvinfer1::FillOperation::kRANDOM_UNIFORM);

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
    // Set seed and dynamic "low" and "high" values if applicable
    if (attrs.count("seed"))
    {
        // TensorRT's seed input must be an integer
        int seed = static_cast<int>(attrs.get<float>("seed", 0.f));
        // TensorRT requires dynamic "low" and "high" values if a seed is provided.
        auto* alphaConstant
            = addConstantScalar(ctx, low, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        auto* betaConstant
            = addConstantScalar(ctx, high, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        auto* seedConstant
            = addConstantScalar(ctx, seed, ::ONNX_NAMESPACE::TensorProto_DataType_INT32);
        nvinfer1::ITensor* seedConstantTensor = seedConstant->getOutput(0);
        fillLayer->setInput(1, *alphaConstant->getOutput(0));
        fillLayer->setInput(2, *betaConstant->getOutput(0));
        fillLayer->setInput(3, *seedConstantTensor);
    }
    else
    {
        fillLayer->setAlpha(low);
        fillLayer->setBeta(high);
    }

    RETURN_FIRST_OUTPUT(fillLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniform)
{
    OnnxAttrs attrs(node, ctx);
    const auto shapeAsIntList = attrs.get<std::vector<int64_t>>("shape");
    ShapeTensor inputShape = ShapeTensor(1, std::vector<int64_t>(shapeAsIntList.begin(), shapeAsIntList.end()));

    return randomUniformHelper(ctx, inputShape, attrs, nvinfer1::DataType::kFLOAT);
}

DEFINE_BUILTIN_OP_IMPORTER(RandomUniformLike)
{
    ASSERT(inputs.size() == 1, ErrorCode::kINTERNAL_ERROR);
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();
    ShapeTensor inputShape = shapeOf(ctx, input);
    OnnxAttrs attrs(node, ctx);
    auto dType = input.getType();

    return randomUniformHelper(ctx, inputShape, attrs, dType);
}

NodeImportResult staticFloatRangeImporter(IImporterContext* ctx, const std::vector<TensorOrWeights>& inputs)
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
    layer->setAlpha(start);
    layer->setBeta(delta);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Range)
{
    if (inputs.at(0).weights().type == ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
    {
        // Floating-point case supported by TensorRT only if all inputs are static.
        if (inputs.at(0).is_weights() && inputs.at(1).is_weights() && inputs.at(2).is_weights())
        {
            return staticFloatRangeImporter(ctx, inputs);
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
    const ShapeTensor zero{0, std::vector<int64_t>({0})};
    const ShapeTensor numberOfElements = max(ctx, sub(ctx, zero, floorDiv(ctx, sub(ctx, start, limit), delta)), zero);

    nvinfer1::IFillLayer* layer = addFill(ctx, convertTo1D(ctx, numberOfElements), nvinfer1::FillOperation::kLINSPACE);

    // TensorRT requires that alpha and beta both be dynamic or both be static.
    if (start.valuesKnown() && delta.valuesKnown())
    {
        layer->setAlpha(start.values[0]);
        layer->setBeta(delta.values[0]);
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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

DEFINE_BUILTIN_OP_IMPORTER(ReduceL1)
{
    NodeImportResult abs_result = unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
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
    return unaryHelper(ctx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    // TODO: Abstract this sequence with a function or macro
    auto exp_result = unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
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
    return unaryHelper(ctx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
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
    nvinfer1::ITensor* sqr_tensor_ptr = sqr_layer->getOutput(0);
    return reduceTensor(ctx, node, sqr_tensor_ptr, nvinfer1::ReduceOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kRELU);
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

    nvinfer1::IShuffleLayer* layer = addShuffle(ctx, data, shape);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int inputRank = input.getDimensions().nbDims;
    ASSERT(inputRank > 0, ErrorCode::kUNSUPPORTED_NODE);
    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);
    OnnxAttrs attrs(node, ctx);

    auto mode = attrs.get<std::string>("mode", "nearest");
    auto resizeMode = mode == "nearest" ? nvinfer1::ResizeMode::kNEAREST : nvinfer1::ResizeMode::kLINEAR;

    if (ctx->getOpsetVersion() >= 11)
    {
        auto transformationMode = attrs.get<std::string>("coordinate_transformation_mode", "half_pixel");
        ASSERT((transformationMode == "asymmetric") && "This version of TensorRT only supports asymmetric resize!",
            ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(mode != "cubic" && "This version of TensorRT does not support cubic interpolation!",
            ErrorCode::kUNSUPPORTED_NODE);
        auto nearest_mode = attrs.get<std::string>("nearest_mode", "round_prefer_floor");
        ASSERT((mode != "nearest" || nearest_mode == "floor")
                && "This version of TensorRT only supports floor nearest_mode!",
            ErrorCode::kUNSUPPORTED_NODE);

        // The existence of a fourth input means a shape was passed as the resize parameter
        if (inputs.size() == 4)
        {
            auto* resizeShape = &convertToTensor(inputs.at(3), ctx);
            layer->setInput(1, *resizeShape);
            layer->setResizeMode(resizeMode);
            RETURN_FIRST_OUTPUT(layer);
        }
    }

    // Resizes that use scale factors have the same import logic between opsets
    auto scales = ctx->getOpsetVersion() >= 11 ? inputs.at(2) : inputs.at(1);
    ASSERT(scales.is_weights() && "Resize scales must be an initializer!", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights scales_weights = scales.weights();
    ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
    float const* scaleValues = static_cast<float const*>(scales_weights.values);
    layer->setResizeMode(resizeMode);
    layer->setScales(scaleValues, inputRank);
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

    const auto getInitialInputValue = [&ctx, &initialStateShape, &inputs](size_t inputIdx) -> nvinfer1::ITensor* {
        if (inputs.size() > inputIdx && inputs.at(inputIdx))
        {
            return &convertToTensor(inputs.at(inputIdx), ctx);
        }
        return constantOfShape(ctx,
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
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse
    // iterator must be concatenated.
    nvinfer1::ITensor* iterationInput{nullptr};
    if (direction == "forward")
    {
        iterationInput = unsqueezeTensor(ctx, *loop->addIterator(*input)->getOutput(0), std::vector<int>{0});
    }
    else if (direction == "reverse")
    {
        nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
        reverseIterator->setReverse(true);
        iterationInput = unsqueezeTensor(ctx, *reverseIterator->getOutput(0), std::vector<int>{0});
    }
    else
    {
        ASSERT(direction == "bidirectional", ErrorCode::kINVALID_NODE);
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);
        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        std::array<nvinfer1::ITensor*, 2> tensors{{unsqueezeTensor(ctx, *forward->getOutput(0), std::vector<int>{0}),
            unsqueezeTensor(ctx, *reverse->getOutput(0), std::vector<int>{0})}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());

    // H(t-1)
    nvinfer1::IRecurrenceLayer* hiddenState = loop->addRecurrence(*initialHidden);
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

    hiddenState->setInput(1, *Ht);
    LOG_VERBOSE("H(t) -> " << Ht->getDimensions());

    std::vector<TensorOrWeights> outputs{};
    // Y = concatenation of all H(t) for each element of the sequence
    // singlePassShape = (1, batchSize, hiddenSize)
    nvinfer1::ITensor* singlePassShape
        = ctx->network()
              ->addElementWise(*initialStateShape(),
                  *addConstant(ctx, std::vector<int>{numDirections, 1, 1}, ::ONNX_NAMESPACE::TensorProto_DataType_INT32,
                       nvinfer1::Dims{1, 3})
                       ->getOutput(0),
                  nvinfer1::ElementWiseOperation::kDIV)
              ->getOutput(0);
    outputs.emplace_back(
        concatenateRNNOutputs(ctx, loop, singlePassShape, getAxisLength(ctx, input, 0), Ht, numDirections));
    // Yh = last value of H(t)
    outputs.emplace_back(
        loop->addLoopOutput(*hiddenState->getOutput(0), nvinfer1::LoopOutput::kLAST_VALUE)->getOutput(0));

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
    const int nbInputs = node.input().size();
    const int nbScanInputs = attrs.get<int>("num_scan_inputs");
    // The number of state variables on the input and output is the same.
    const int nbStateVars = nbInputs - nbScanInputs;
    const int nbScanOutputs = node.output().size() - nbStateVars;

    std::vector<int> defaultScanInputArgs(nbScanInputs);
    std::fill(defaultScanInputArgs.begin(), defaultScanInputArgs.end(), 0);
    std::vector<int> scanInputAxes(attrs.get("scan_input_axes", defaultScanInputArgs));
    const std::vector<int> scanInputDirections(attrs.get("scan_input_directions", defaultScanInputArgs));

    std::vector<int> defaultScanOutputArgs(nbScanOutputs);
    std::fill(defaultScanOutputArgs.begin(), defaultScanOutputArgs.end(), 0);
    std::vector<int> scanOutputAxes(attrs.get("scan_output_axes", defaultScanOutputArgs));
    const std::vector<int> scanOutputDirections(attrs.get("scan_output_directions", defaultScanOutputArgs));

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
        stateVars.emplace_back(loop->addRecurrence(convertToTensor(inputs.at(i), ctx)));
        ctx->registerTensor(TensorOrWeights{stateVars.back()->getOutput(0)}, body.input(i).name());
    }
    for (int i = 0; i < nbScanInputs; ++i)
    {
        const int index = nbStateVars + i; // Scan Inputs are after the state variables.
        nvinfer1::IIteratorLayer* scanInput = loop->addIterator(convertToTensor(inputs.at(index), ctx));
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
    RETURN_FIRST_OUTPUT(ctx->network()->addShape(input));
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSIGMOID);
}

DEFINE_BUILTIN_OP_IMPORTER(Sin)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Sinh)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Size)
{
    // "data : T
    // An input tensor."
    const ShapeTensor shape = shapeOf(ctx, inputs.at(0));

    // "outputs a int64 scalar that equals to the total number of elements of the input tensor."
    const ShapeTensor size = product(ctx, shape, 0, shape.size, /*rank=*/0);

    return {{&size.tensor(ctx)}};
}

//! Return subscripts such that gather(concat(x,y),subscripts)
//! will return x with x[subcripts[i]] replaced by y[i].
ShapeTensor axesToInterlaceSubscripts(const std::vector<int64_t>& axes, int nbDims)
{
    ShapeTensor subscripts{iotaShapeVector(nbDims)};
    for (size_t i = 0; i < axes.size(); ++i)
    {
        subscripts.values[axes[i]] = nbDims + i;
    }
    return subscripts;
}

//! Decode a start or end index according to ONNX Slice rules:
//! "If the value passed to start or end is larger than the n (the number of
//! elements in this dimension), it represents n....  If a negative value is
//! passed for step, it represents slicing backward."
ShapeTensor decodeOnnxIndices(IImporterContext* ctx, const ShapeTensor& indices, const ShapeTensor& dims)
{
    // Oblique calculation implements the rules using only operations available in TensorRT.
    return sub(
        ctx, min(ctx, indices, dims), mul(ctx, dims, max(ctx, shapeVector(-1), min(ctx, shapeVector(0), indices))));
}

ShapeTensor computeSizes(IImporterContext* ctx, const ShapeTensor& starts, const ShapeTensor& ends,
    const ShapeTensor& steps, const ShapeTensor& dims)
{
    if (steps.isAll(1))
    {
        // The general formula in the else is correct,
        // but creates much debris for this common case.
        return sub(ctx, ends, starts);
    }
    else
    {
        // "If a negative value is passed for step, it represents slicing backward."
        // Compute ceil((end-start)/step) using only operations available in TensorRT.
        return sub(ctx, similar(dims, 0), floorDiv(ctx, sub(ctx, starts, ends), steps));
    }
}

DEFINE_BUILTIN_OP_IMPORTER(Slice)
{
    const int nbInputs = node.input().size();
    // "...it uses this information to slice the input data tensor."
    nvinfer1::ITensor& data = convertToTensor(inputs.at(0), ctx);
    const ShapeTensor dims = shapeOf(ctx, data);

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
        axes = nbInputs > 3 ? ShapeTensor(inputs.at(3)) : iotaShapeVector(dims.size);
        // Doesn't support dynamic axes currently.
        ASSERT(axes.valuesKnown(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(starts.size == axes.size, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(ends.size == axes.size, ErrorCode::kUNSUPPORTED_NODE);
        // "If steps are omitted, they are set to [1, ..., 1] of length len(starts)."
        steps = inputs.size() > 4 ? ShapeTensor(inputs.at(4)) : similar(starts, 1);
    }
    else
    {
        OnnxAttrs attrs(node, ctx);
        starts = ShapeTensor(1, attrs.get<std::vector<int64_t>>("starts"));
        ends = ShapeTensor(1, attrs.get<std::vector<int64_t>>("ends"));
        // "It's optional. If not present, will be treated as [0, 1, ..., len(starts) - 1]."
        axes = attrs.count("axes") ? ShapeTensor(1, attrs.get<std::vector<int64_t>>("axes"))
                                   : iotaShapeVector(starts.size);
        steps = similar(starts, 1);
    }

    // Decode axes.
    // Also inspect whether axes form an "iota" sequence 0, 1, 2, ....
    bool isIota = true;
    int j = 0;
    for (auto& axis : axes.values)
    {
        // "Accepted range is [-r, r-1] where r = rank(data)."
        const int r = dims.size;
        ASSERT(-r <= axis && axis < r, ErrorCode::kINVALID_VALUE);
        // "Negative value means counting dimensions from the back."
        if (axis < 0)
        {
            axis += r;
        }
        isIota &= axis == j;
        ++j;
    }
    // Check for duplicate axes.
    ASSERT(std::unordered_set<int64_t>(axes.values.begin(), axes.values.end()).size() == axes.values.size(),
        ErrorCode::kINVALID_NODE);

    if (axes.size < dims.size || !isIota)
    {
        // axes specify a subset of the dimensions, or out of order.
        // Convert starts/ends/steps to complete in-order form.
        const ShapeTensor subscripts{axesToInterlaceSubscripts(axes.values, dims.size)};
        starts = interlace(ctx, similar(dims, 0), starts, subscripts);
        ends = interlace(ctx, dims, ends, subscripts);
        steps = interlace(ctx, similar(dims, 1), steps, subscripts);
    }

    // "If a negative value is passed for any of the start or end indices,
    // it represents number of elements before the end of that dimension."
    starts = decodeOnnxIndices(ctx, starts, dims);
    ends = decodeOnnxIndices(ctx, ends, dims);

    // TensorRT uses sizes of the output dimensions instead of ends.
    const ShapeTensor sizes = computeSizes(ctx, starts, ends, steps, dims);

    nvinfer1::ISliceLayer* slice = addSlice(ctx, data, starts, sizes, steps);

    RETURN_FIRST_OUTPUT(slice);
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    OnnxAttrs attrs(node, ctx);
    // "input : T"
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const ShapeTensor dims = shapeOf(ctx, input);

    // "axis : int (default is 1)"
    int axis = attrs.get("axis", 1);

    // "Negative value means counting dimensions from the back.
    // Accepted range is [-r, r-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, dims.size));

    // "The input does not need to explicitly be a 2D vector; rather, it will be coerced into one."
    auto* flattened = flattenTensor(ctx, input, axis);
    auto* softMax = ctx->network()->addSoftMax(*flattened);
    // ONNX softmax is always on second dimension.
    softMax->setAxes(1 << 1);

    // Reshape back to original shape
    auto* reshapeLayer = addShuffle(ctx, *softMax->getOutput(0), dims);
    RETURN_FIRST_OUTPUT(reshapeLayer);
}

DEFINE_BUILTIN_OP_IMPORTER(Softplus)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS);
}

DEFINE_BUILTIN_OP_IMPORTER(Softsign)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTSIGN);
}

DEFINE_BUILTIN_OP_IMPORTER(SpaceToDepth)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node, ctx);
    int block_size = attrs.get<int>("blocksize");
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int ndim_spatial = dims.nbDims - 1;
    nvinfer1::Dims new_shape1;
    new_shape1.nbDims = dims.nbDims + ndim_spatial;
    new_shape1.d[0] = dims.d[0];
    for (int i = 0; i < ndim_spatial; ++i)
    {
        ASSERT(dims.d[1 + i] % block_size == 0, ErrorCode::kINVALID_NODE);
        new_shape1.d[1 + 2 * i + 0] = dims.d[1 + i] / block_size;
        new_shape1.d[1 + 2 * i + 1] = block_size;
    }
    layer->setReshapeDimensions(new_shape1);
    nvinfer1::Permutation perm;
    perm.order[ndim_spatial] = 0;
    for (int i = 0; i < ndim_spatial; ++i)
    {
        perm.order[ndim_spatial + 1 + i] = 1 + 2 * i + 0;
        perm.order[i] = 1 + 2 * i + 1;
    }
    layer->setSecondTranspose(perm);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    nvinfer1::Dims new_shape2;
    new_shape2.nbDims = dims.nbDims - ndim_spatial;
    new_shape2.d[0] = dims.d[ndim_spatial];
    for (int i = 0; i < ndim_spatial; ++i)
    {
        new_shape2.d[0] *= dims.d[i];
        new_shape2.d[1 + i] = dims.d[ndim_spatial + 1 + i];
    }
    tensor_ptr = reshapeTensor(ctx, *tensor_ptr, new_shape2);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    const int numOutputs = node.output().size();

    // "input : T
    // The tensor to split"
    nvinfer1::ITensor& inputTensor = convertToTensor(inputs.at(0), ctx);
    const ShapeTensor inputDims = shapeOf(ctx, inputTensor);

    // "axis : int (default is 0)
    // Which axis to split on."
    OnnxAttrs attrs(node, ctx);
    int axis = attrs.get<int>("axis", 0);

    // "A negative value means counting dimensions from the back.
    // Accepted range is [-rank, rank-1] where r = rank(input)."
    TRT_CHECK(convertAxis(axis, inputDims.size));

    ShapeTensor subscripts = iotaShapeVector(inputDims.size);
    subscripts.values[axis] = inputDims.size;

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

    const ShapeTensor zeros = similar(inputDims, 0);
    const ShapeTensor ones = similar(inputDims, 1);
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
        outputs.emplace_back(slice->getOutput(0));
    }

    return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt)
{
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
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
    auto* squeezed = squeezeTensor(ctx, data, axes);

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
    return unaryHelper(ctx, inputs.at(0), nvinfer1::UnaryOperation::kTAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh)
{
    RETURN_FIRST_OUTPUT(ctx->network()->addActivation(inputs.at(0).tensor(), nvinfer1::ActivationType::kTANH));
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
    const ShapeTensor inputDims = shapeOf(ctx, input);

    // "repeats : T1
    // 1D int64 tensor of the same length as input's dimension number,
    // includes numbers of repeated copies along input's dimensions.
    const ShapeTensor repeats{inputs.at(1)};

    ShapeTensor outputShape = mul(ctx, inputDims, repeats);
    nvinfer1::ISliceLayer* tile = addSlice(ctx, input, similar(inputDims, 0), outputShape, similar(inputDims, 1));
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
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }

    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(*tensorPtr, nvinfer1::TopKOperation::kMAX, k, axisMask);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* values = layer->getOutput(0);
    nvinfer1::ITensor* indices = layer->getOutput(1);

    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int> axes{1};
        values = squeezeTensor(ctx, *values, axes);
        ASSERT(values, ErrorCode::kUNSUPPORTED_NODE);
        indices = squeezeTensor(ctx, *indices, axes);
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
        nvinfer1::ITensor* output_tensor = transposeTensor(ctx, input.tensor(), perm, false);
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
    auto* expanded = unsqueezeTensor(ctx, data, axes);

    ASSERT(expanded && "Failed to unsqueeze tensor!", ErrorCode::kUNSUPPORTED_NODE);

    return {{expanded}};
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
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
        // Linear resize support 1-D, 2-D and 3D resize.
        ASSERT((nbDims >= 1) && (nbDims <= 3), ErrorCode::kUNSUPPORTED_NODE);
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }
    // Add resize layer
    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    layer->setScales(scale_factors.data(), nbDims);
    layer->setResizeMode(resizeMode);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Where)
{
    nvinfer1::ITensor* condition = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* x = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor* y = &convertToTensor(inputs.at(2), ctx);

    broadcastTensors(ctx, x, y, condition);

    nvinfer1::Dims cDims = condition->getDimensions();
    nvinfer1::Dims xDims = x->getDimensions();
    nvinfer1::Dims yDims = y->getDimensions();

    ASSERT(cDims.nbDims == xDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(cDims.nbDims == yDims.nbDims, ErrorCode::kUNSUPPORTED_NODE);

    auto* layer = ctx->network()->addSelect(*condition, *x, *y);

    RETURN_FIRST_OUTPUT(layer);
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
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Shuffle)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    nvinfer1::Permutation perm1 = attrs.get<nvinfer1::Permutation>("first_perm");
    nvinfer1::Permutation perm2 = attrs.get<nvinfer1::Permutation>("second_perm");

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(input);
    layer->setFirstTranspose(perm1);
    layer->setSecondTranspose(perm2);

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
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_MaxAverageBlendPool)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kINVALID_NODE);
    auto& input = inputs.at(0).tensor();

    OnnxAttrs attrs(node, ctx);
    int nbSpatialDims = attrs.get<nvinfer1::Dims>("kernel_shape").nbDims;
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding(true);
    getKernelParams(ctx, node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    float blend = attrs.get<float>("blend");

    nvinfer1::IPoolingLayer* layer
        = ctx->network()->addPoolingNd(input, nvinfer1::PoolingType::kMAX_AVERAGE_BLEND, kernel_size);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setAverageCountExcludesPadding(exclude_padding);
    layer->setPaddingMode(paddingMode);

    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);

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
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(TRT_Resize)
{
    ASSERT(inputs.at(0).is_tensor(), nvonnxparser::ErrorCode::kUNSUPPORTED_NODE);
    auto& input = inputs.at(0).tensor();

    nvinfer1::IResizeLayer* layer;
    layer = ctx->network()->addResize(input);

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
