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
#include "onnx2trt_utils.hpp"

#include <algorithm> // For std::min, std::max
#include <numeric>   // For std::iota
#include <sstream>   // For std::stringstream
#include <cstring>

namespace onnx2trt
{

string_map<NodeImporter>& getBuiltinOpImporterMap()
{
    static string_map<NodeImporter> builtin_op_importers;
    return builtin_op_importers;
}

namespace
{

bool registerBuiltinOpImporter(std::string op, NodeImporter const& importer)
{
    bool inserted = getBuiltinOpImporterMap().insert({op, importer}).second;
    assert(inserted);
    return inserted;
}

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


DEFINE_BUILTIN_OP_IMPORTER(Abs)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acos)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kACOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Acosh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kACOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(Add)
{
    bool isAnyOperandInt32 = check_for_int32(inputs);
    if (inputs.at(0).is_tensor() != inputs.at(1).is_tensor() && !isAnyOperandInt32)
    {
      return scaleHelper(
          ctx, node, inputs, ScaleOp::kSHIFT);
    }
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM, true);
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
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Asinh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Atan)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATAN);
}


DEFINE_BUILTIN_OP_IMPORTER(Atanh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATANH);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::DimsNCHW new_shape(dims.d[0], dims.d[1], dims.d[2], 1);
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    // Support for opset10 ceil_mode
    CeilingPoolDim ceilingPool;
    // Ceiling and dialations added in opset 10
    if (ctx->getOpsetVersion() >= 10)
    {
        OnnxAttrs attrs(node);
        const auto ceil_mode = attrs.get<int>("ceil_mode", 0);
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT(dilations[i] == 1, ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
        if (ceil_mode != 0) // Need to set pooling formula to use ceiling instead of floor
        {
            ctx->network()->setPoolingOutputDimensionsFormula(&ceilingPool);
        }
    }

    ASSERT(dims.nbDims >= 4, ErrorCode::kUNSUPPORTED_NODE);

    int nbSpatialDims = dims.nbDims - 2;
    ASSERT(nbSpatialDims == 2 || nbSpatialDims == 3, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;

    bool exclude_padding(true);
    get_kernel_params(node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    nvinfer1::IPoolingLayer* pooling_layer
        = ctx->network()->addPoolingNd(*tensor_ptr, nvinfer1::PoolingType::kAVERAGE, kernel_size);
    nvinfer1::ILayer* layer = pooling_layer;
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    pooling_layer->setStrideNd(strides);
    pooling_layer->setAverageCountExcludesPadding(exclude_padding);
    pooling_layer->setPaddingMode(paddingMode);

    pooling_layer->setPrePadding(beg_padding);
    pooling_layer->setPostPadding(end_padding);
    // Note: Average pooling requires special care with asymmetric padding
    //       because the need to exclude padding pixels from the average
    //       means we can't just use a pre-padding layer.
    nvinfer1::DimsHW pre_crop(0, 0), post_crop(0, 0);
    for (int d = 0; d < 2; ++d)
    {
        if (end_padding.d[d] == beg_padding.d[d])
        {
            // Symmetric padding, nothing special needed
        }
        else if (end_padding.d[d] == beg_padding.d[d] + 1)
        {
            // Pad symmetrically such that we get one more output element at
            // the beginning, and then crop it off after the pooling operation.
            beg_padding.d[d] += strides.d[d];
            pre_crop.d[d] = 1;
        }
        else
        {
            bool supported_form_of_asymmetric_padding_for_AveragePool = false;
            ASSERT(supported_form_of_asymmetric_padding_for_AveragePool, ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization)
{
    // Scale, bias, mean, and variance must be initializers
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    auto mean_weights = inputs.at(3).weights();
    auto variance_weights = inputs.at(4).weights();
    OnnxAttrs attrs(node);
    float eps = attrs.get<float>("epsilon", 1e-5f);
    ASSERT(scale_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && bias_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && mean_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT
            && variance_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT,
        ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::Dims new_shape({4, {dims.d[0], dims.d[1], dims.d[2], 1}});
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    int nchan = dims.d[1];
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
        return addScale(ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
    }
    else
    {
        auto scaledTensor = addScale(ctx, *tensor_ptr, nvinfer1::ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
        // Un-expand spatial dims back to 1D
        tensor_ptr = &convertToTensor(scaledTensor.value().at(0), ctx);
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        return {{tensor_ptr}};
    }

}

DEFINE_BUILTIN_OP_IMPORTER(Cast) {
    // Get input node.
    OnnxAttrs attrs(node);
    auto cast_dtype = attrs.get<int32_t>("to");
    auto * tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto trt_dtype = tensor_ptr->getType();
    // TensorRT currently only supports the following conversion: FP16 -> FP32.
    ASSERT(trt_dtype == nvinfer1::DataType::kHALF && cast_dtype == ::ONNX_NAMESPACE::TensorProto::FLOAT,
          ErrorCode::kUNSUPPORTED_NODE);
    // Add the layer.
    nvinfer1::IIdentityLayer* layer = ctx->network()->addIdentity(inputs.at(0).tensor());
    layer->setPrecision(nvinfer1::DataType::kFLOAT);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCEIL);
}

DEFINE_BUILTIN_OP_IMPORTER(Clip)
{
    OnnxAttrs attrs(node);
    // beta is the upper bound.
    float alpha = attrs.get("min", std::numeric_limits<float>::lowest());
    float beta = attrs.get("max", std::numeric_limits<float>::max());
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kCLIP, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Concat)
{
    std::vector<nvinfer1::ITensor*> tensors;
    for (auto& input : inputs)
    {
        tensors.push_back(&convertToTensor(input, ctx));
    }
    OnnxAttrs attrs(node);
    int axis = attrs.get<int>("axis");
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convert_axis(axis, nbDims));
    auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setAxis(axis);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Constant)
{
    // TODO: This silently fails if the dtype is not supported
    OnnxAttrs attrs(node);
    return {{attrs.get<ShapedWeights>("value")}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv)
{
    // Convolution Weights must be an initializer
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto kernel_weights = inputs.at(1).weights();
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::Dims new_shape({4, {dims.d[0], dims.d[1], dims.d[2], 1}});
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
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
    get_kernel_params(
        node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding, &dilations);

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
    OnnxAttrs attrs(node);
    int ngroup = attrs.get("group", 1);
    ASSERT(kernel_weights.shape.d[1] * ngroup == nchan, ErrorCode::kINVALID_NODE);
    layer->setNbGroups(ngroup);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
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
        nvinfer1::Dims new_shape({4, {dims.d[0], dims.d[1], dims.d[2], 1}});
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
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
    OnnxAttrs attrs(node);
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
    int nchan = dims.d[1];

    ASSERT(kernel_weights.shape.d[0] == nchan, ErrorCode::kINVALID_NODE);
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

    get_kernel_params(node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding,
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
        auto_gen_input_output_padding(dims, output_shape, kernel_size, strides, dilations, nbSpatialDims, beg_padding,
            end_padding, output_padding, paddingMode);
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

    nvinfer1::DimsHW output_padding_HW;
    if (output_padding.nbDims == 2)
    {
        output_padding_HW = nvinfer1::DimsHW(output_padding.d[0], output_padding.d[1]);
    }
    else
    {
        output_padding_HW = nvinfer1::DimsHW(output_padding.d[1], output_padding.d[2]);
    }

    if (output_padding_HW != nvinfer1::DimsHW(0, 0))
    {
        tensor_ptr = ctx->network()->addPadding(*tensor_ptr, nvinfer1::DimsHW(), output_padding_HW)->getOutput(0);
    }

    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Cos)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOS);
}

DEFINE_BUILTIN_OP_IMPORTER(Cosh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOSH);
}

DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace)
{
    // The Tensor has to be of shape NCHW
    ASSERT(inputs.at(0).shape().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node);
    int block_size = attrs.get<int>("blocksize");
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int ndim_spatial = dims.nbDims - 2;
    nvinfer1::Dims new_shape1;
    new_shape1.nbDims = dims.nbDims + ndim_spatial;
    new_shape1.d[0] = dims.d[0];
    new_shape1.d[ndim_spatial + 1] = dims.d[1];
    for (int i = 1; i <= ndim_spatial; ++i)
    {
        ASSERT(new_shape1.d[ndim_spatial + 1] % block_size == 0, ErrorCode::kINVALID_NODE);
        new_shape1.d[ndim_spatial + 1] /= block_size;
        new_shape1.d[i] = block_size;
        new_shape1.d[ndim_spatial + 1 + i] = dims.d[1 + i];
    }
    layer->setReshapeDimensions(new_shape1);
    nvinfer1::Permutation perm;
    perm.order[0] = 0;
    perm.order[1] = ndim_spatial + 1;
    for (int i = 1; i < ndim_spatial + 1; ++i)
    {
        perm.order[2 * i] = ndim_spatial + 1 + i;
        perm.order[2 * i + 1] = i;
    }
    layer->setSecondTranspose(perm);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    nvinfer1::Dims new_shape2;
    new_shape2.nbDims = dims.nbDims - ndim_spatial;
    new_shape2.d[0] = dims.d[0];
    new_shape2.d[1] = dims.d[1];
    for (int i = 1; i <= ndim_spatial; ++i)
    {
        new_shape2.d[1 + i] = dims.d[2 * i] * dims.d[1 + 2 * i];
    }
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape2);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensor_ptr}};
}

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div)
{
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kDIV, true);
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
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Exp)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten)
{
    OnnxAttrs attrs(node);
    int axis = attrs.get("axis", 1);
    nvinfer1::Dims dims = inputs.at(0).shape();
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    int dim0 = 1;
    int dim1 = 1;
    for (int i = 0; i < axis; i++)
    {
        dim0 *= dims.d[i];
    }
    for (int i = axis; i < dims.nbDims; i++)
    {
        dim1 *= dims.d[i];
    }
    nvinfer1::Dims new_shape{2, {dim0, dim1}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Gather)
{
    nvinfer1::ITensor* data = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* indices = &convertToTensor(inputs.at(1), ctx);
    OnnxAttrs attrs(node);
    int axis = attrs.get<int>("axis", 0);
    nvinfer1::Dims dataDims = data->getDimensions();
    int nbDims = dataDims.nbDims;

    // Indicies must have at least one dimension in TRT 6.0.
    bool expandIndices = indices->getDimensions().nbDims == 0;
    if (expandIndices)
    {
        nvinfer1::Dims oneD {1, {1}};
        indices = reshape_tensor(ctx, *indices, oneD);
    }

    TRT_CHECK(convert_axis(axis, nbDims));

    auto* layer = ctx->network()->addGather(*data, *indices, axis);

    auto* layerOutput = layer->getOutput(0);

    // Extra dimension would have been inserted on the axis we are gathering on. Remove it here.
    if (expandIndices)
    {
        nvinfer1::Dims oldDims = layerOutput->getDimensions();
        nvinfer1::Dims newDims {nbDims-1, {1}};
        for (int i = 0; i < newDims.nbDims; i++)
        {
            if (i >= axis)
            {
                newDims.d[i] = oldDims.d[i+1];
            }
            else
            {
                newDims.d[i] = oldDims.d[i];
            }
        }

        layerOutput = reshape_tensor(ctx, *layerOutput, newDims);
    }

    return {{layerOutput}};
}

// Adds a constant scalar to the network in the form of a constant layer.
template <typename ScalarType>
nvinfer1::IConstantLayer* addConstantScalar(IImporterContext* ctx, ScalarType scalar, ShapedWeights::DataType type)
{
    ShapedWeights scalarWeights = ctx->createTempWeights(type, nvinfer1::Dims{0, {}});
    static_cast<ScalarType*>(scalarWeights.values)[0] = scalar;
    return ctx->network()->addConstant(scalarWeights.shape, scalarWeights);
}

DEFINE_BUILTIN_OP_IMPORTER(Floor)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kFLOOR);
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);
    nvinfer1::ITensor& inputC = convertToTensor(inputs.at(2), ctx);

    // Use FC if it is likely to be faster - which is usually when no Shuffles are required.
    bool canUseFC = inputs.at(0).is_tensor() && inputs.at(1).is_weights() && inputs.at(2).is_weights() && alpha == 1.f
        && beta == 1.f && inputs.at(0).tensor().getDimensions().nbDims == 3 && inputs.at(1).weights().shape.nbDims == 2
        && inputs.at(2).weights().shape.nbDims == 1;
    if (canUseFC)
    {

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


    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputASqueezed, opA, *inputB, opB);
    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        broadcast_tensors(ctx, alphaConstantTensor, matmulTensor);
        nvinfer1::IElementWiseLayer* scaledMatmul = ctx->network()->addElementWise(
            *alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        matmulTensor = scaledMatmul->getOutput(0);
    }
    // Scale C if needed.
    nvinfer1::ITensor* biasTensor = &inputC;

    if (beta != 1.f)
    {
        nvinfer1::IConstantLayer* betaConstant
            = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
        broadcast_tensors(ctx, betaConstantTensor, biasTensor);
        nvinfer1::IElementWiseLayer* scaledBias
            = ctx->network()->addElementWise(*betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
        biasTensor = scaledBias->getOutput(0);
    }
    // A*B may be lower rank than C in TRT, so need to squeeze C.
    if (ctx->getOpsetVersion() < 7 && !attrs.get("broadcast", false))
    {
        nvinfer1::Dims squeezeDims = squeeze_leading_dims(biasTensor->getDimensions());
        biasTensor = reshape_tensor(ctx, *biasTensor, squeezeDims);
    }
    broadcast_tensors(ctx, matmulTensor, biasTensor);
    nvinfer1::IElementWiseLayer* biasAdd
        = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
    return {{biasAdd->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    ASSERT(dims.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::DimsHW kernel_size(dims.d[2], dims.d[3]);
    RETURN_FIRST_OUTPUT(ctx->network()->addPooling(tensor, nvinfer1::PoolingType::kAVERAGE, kernel_size));
}

// TODO: GlobalLpPool: pow(reduce_mean(pow(abs(x), p)), 1./p)

DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor.getDimensions();
    ASSERT(dims.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::DimsHW kernel_size(dims.d[2], dims.d[3]);
    RETURN_FIRST_OUTPUT(ctx->network()->addPooling(tensor, nvinfer1::PoolingType::kMAX, kernel_size));
}

DEFINE_BUILTIN_OP_IMPORTER(HardSigmoid)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha", 0.2f);
    float beta = attrs.get<float>("beta", 0.5f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kHARD_SIGMOID, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Identity)
{
    RETURN_IDENTITY(inputs.at(0));
}

DEFINE_BUILTIN_OP_IMPORTER(ImageScaler)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs{node};
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
    auto scale_weights = inputs.at(1).weights();
    auto bias_weights = inputs.at(2).weights();
    OnnxAttrs attrs(node);
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

    RETURN_FIRST_OUTPUT(ctx->network()->addPluginV2(&tensor_ptr, 1, *plugin));
}

DEFINE_BUILTIN_OP_IMPORTER(LeakyRelu)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha", 0.01f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kLEAKY_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(Log)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DECLARE_BUILTIN_OP_IMPORTER(Softmax);
DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax)
{
    auto result = importSoftmax(ctx, node, inputs);
    if (result.is_error())
    {
        return result;
    }
    else
    {
        auto& input = result.value().at(0);
        return apply_unary_function(ctx, input, nvinfer1::UnaryOperation::kLOG);
    }
}

DEFINE_BUILTIN_OP_IMPORTER(LRN)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    OnnxAttrs attrs(node);
    int size = attrs.get<int>("size");
    float alpha = attrs.get<float>("alpha", 0.0001);
    float beta = attrs.get<float>("beta", 0.75);
    float bias = attrs.get<float>("bias", 1.0);
    RETURN_FIRST_OUTPUT(ctx->network()->addLRN(tensor, size, alpha, beta, bias));
}

DEFINE_BUILTIN_OP_IMPORTER(LSTM) {
    // Input
    nvinfer1::ITensor& raw_input = convertToTensor(inputs.at(0), ctx);
    ASSERT(3 == raw_input.getDimensions().nbDims && "Input tensor must be 3 dimensional",
           ErrorCode::kINVALID_NODE);
    ASSERT((raw_input.getType() == nvinfer1::DataType::kFLOAT ||
            raw_input.getType() == nvinfer1::DataType::kHALF) &&
           "Only fp16 and fp32 inputs are supported",
           ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::DataType input_type = raw_input.getType();
    const int32_t max_seq_len = raw_input.getDimensions().d[0];
    const int32_t batch_size = raw_input.getDimensions().d[1];

    // Attributes
    OnnxAttrs attrs(node);
    const std::string direction_str = attrs.get<std::string>("direction", "forward");
    ASSERT((direction_str == "forward" || direction_str == "bidirectional") &&
           "Reverse LSTM unsupported",
           ErrorCode::kUNSUPPORTED_NODE);
    const nvinfer1::RNNDirection direction = (direction_str == "forward") ?
                                             nvinfer1::RNNDirection::kUNIDIRECTION :
                                             nvinfer1::RNNDirection::kBIDIRECTION;
    const int num_directions = (direction_str == "forward") ? 1 : 2;
    // There are three distinct uses of an activation function within the LSTM equations
    // One for the input/forget/output gates, one for the cell state, and one for the output
    // RNNv2 only supports the default choice for each, listed here (duplicated for bidirectional)
    std::vector<std::string> default_activations = {"Sigmoid", "Tanh", "Tanh"};
    if (num_directions == 2) {
        default_activations.insert(default_activations.end(), {"Sigmoid", "Tanh", "Tanh"});
    }
    const std::vector<std::string> activations =
        attrs.get<std::vector<std::string>>("activations", default_activations);
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
    for (int i = 3; i < node.input_size(); i++) {
        const std::string& input_name = node.input(i);
        if (input_name == "B") {
            has_bias = true;
        } else if (input_name == "sequence_lens") {
            sequence_lens = &(convertToTensor(inputs.at(i), ctx));
            ASSERT(sequence_lens &&
                   sequence_lens->getType() == nvinfer1::DataType::kINT32 &&
                   "Failed to process sequence_lens (sequence_lens must be int32)",
                   ErrorCode::kINVALID_NODE);
        } else if (input_name == "initial_h" || input_name == "initial_c") {
            nvinfer1::ITensor* output = nullptr;
            if (inputs.at(i).is_weights()) {
                /* constant->shuffle bug (NVBug 2650549), so we do the transpose manually */
                ShapedWeights weights = inputs.at(i).weights();
                const int dtype_size = get_dtype_size(weights.type);
                const size_t len = num_directions * batch_size * hidden_size * dtype_size;
                auto* source = reinterpret_cast<unsigned char*>(weights.values);
                std::vector<unsigned char> buffer;
                buffer.resize(len);
                for (int i = 0; i < num_directions; i++) {
                    for (int j = 0; j < batch_size; j++) {
                        for (int k = 0; k < hidden_size; k++) {
                            for (int b = 0; b < dtype_size; b++) {
                                int src_idx = i*batch_size*hidden_size*dtype_size +
                                              j*hidden_size*dtype_size + k*dtype_size + b;
                                int buf_idx = j*num_directions*hidden_size*dtype_size +
                                              i*hidden_size*dtype_size + k*dtype_size + b;
                                buffer.at(buf_idx) = source[src_idx];
                            }
                        }
                    }
                }
                std::memcpy(weights.values, static_cast<void*>(buffer.data()), len);
                const nvinfer1::Dims new_dims = {3, {batch_size, num_directions, hidden_size}};
                output = ctx->network()->addConstant(new_dims, weights)->getOutput(0);
                ASSERT(output &&
                       "Failed to convert initial_h or initial_c weights to constant layer",
                       ErrorCode::kINTERNAL_ERROR);
            } else {
                /* TODO: Once NVBug 2650549 is fixed, we can use just this path instead */
                /* nvinfer1::ITensor& source = convertToTensor(inputs.at(i), ctx); */
                nvinfer1::ITensor& source = inputs.at(i).tensor();
                auto* shuffle_layer = ctx->network()->addShuffle(source);
                ASSERT(shuffle_layer && "Failed to create initial_h shuffle layer",
                       ErrorCode::kINTERNAL_ERROR);
                shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});
                output = shuffle_layer->getOutput(0);
            }
            ASSERT(output->getType() == input_type &&
                   "initial_h and initial_c datatype must match input",
                   ErrorCode::kINVALID_NODE);
            if (input_name == "initial_h") {
                    initial_h = output;
            } else {
                    initial_c = output;
            }
        } else if (input_name == "P") {
            ASSERT(false && "Peephole connections not supported", ErrorCode::kUNSUPPORTED_NODE);
        }
    }

    // Input Shuffle Layer
    auto* input_shuffle_layer = ctx->network()->addShuffle(raw_input);
    ASSERT(input_shuffle_layer && "Failed to create input shuffle layer",
           ErrorCode::kINTERNAL_ERROR);
    input_shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});

    // RNNv2 Layer
    nvinfer1::ITensor& input_seqs = *(input_shuffle_layer->getOutput(0));
    const nvinfer1::RNNOperation op = nvinfer1::RNNOperation::kLSTM;
    const int32_t layer_count = 1;
    auto* layer = ctx->network()->addRNNv2(input_seqs, layer_count, hidden_size, max_seq_len, op);
    ASSERT(layer && "Failed to create RNNv2 layer", ErrorCode::kINTERNAL_ERROR);
    layer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    layer->setDirection(direction);
    if (sequence_lens) {
        layer->setSequenceLengths(*sequence_lens);
    }
    if (initial_h) {
        layer->setHiddenState(*initial_h);
    }
    if (initial_c) {
        layer->setCellState(*initial_c);
    }

    // Weights
    ASSERT(inputs.at(1).is_weights() && "W must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(2).is_weights() && "R must be constant", ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights gate_weights = inputs.at(1).weights();
    ShapedWeights rcur_weights = inputs.at(2).weights();

    nvinfer1::DataType gate_weights_type, rcur_weights_type;
    ASSERT(convert_dtype(gate_weights.type, &gate_weights_type) && "Bad datatype in W",
           ErrorCode::kINTERNAL_ERROR);
    ASSERT(convert_dtype(rcur_weights.type, &rcur_weights_type) && "Bad datatype in R",
           ErrorCode::kINTERNAL_ERROR);
    ASSERT(input_type == gate_weights_type && "W datatype must match X",
           ErrorCode::kINVALID_NODE);
    ASSERT(input_type == rcur_weights_type && "R datatype must match X",
           ErrorCode::kINVALID_NODE);

    ShapedWeights bias_weights;
    if (has_bias) {
        ASSERT(inputs.at(3).is_weights() && "B must be constant",
               ErrorCode::kUNSUPPORTED_NODE);
        bias_weights = inputs.at(3).weights();
        nvinfer1::DataType bias_weights_type;
        ASSERT(convert_dtype(bias_weights.type, &bias_weights_type) && "Bad datatype in B",
               ErrorCode::kINTERNAL_ERROR);
        ASSERT(input_type == bias_weights_type && "B datatype must match X",
               ErrorCode::kINVALID_NODE);
    }

    const int data_size = (input_type == nvinfer1::DataType::kFLOAT) ? 4 : 2;
    const int input_size = gate_weights.shape.d[2];

    auto weightBuilder = [input_type, data_size, hidden_size, ctx]
                         (int layer_index, ShapedWeights& src, int stride, int idx)
    {
        nvinfer1::Weights w;
        int direction_offset = data_size * layer_index * 4 * hidden_size * stride;
        int gate_offset = data_size * hidden_size * stride * idx;
        w.type   = input_type;
        w.values = reinterpret_cast<void*>(
            reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
        w.count  = hidden_size * stride;
        return w;
    };

    // RNNv2 requires that a bias be set, even if none is provided
    auto zeroes = ctx->createTempWeights(gate_weights.type, nvinfer1::Dims{1, {hidden_size}});
    std::memset(zeroes.values, 0, data_size * hidden_size);

    auto biasBuilder = [input_type, data_size, hidden_size, has_bias, zeroes]
                       (int layer_index, ShapedWeights& src, int idx)
    {
        nvinfer1::Weights b;
        int direction_offset = data_size * layer_index * 8 * hidden_size;
        int gate_offset = data_size * hidden_size * idx;
        b.type = input_type;
        if (has_bias) {
            b.values = reinterpret_cast<void*>(
                reinterpret_cast<unsigned char*>(src.values) + direction_offset + gate_offset);
        } else {
            b.values = zeroes.values;
        }
        b.count = hidden_size;
        return b;
    };

    for (int layer_index = 0; layer_index < num_directions; layer_index++) {
        nvinfer1::Weights W_i = weightBuilder(layer_index, gate_weights, input_size,  0);
        nvinfer1::Weights W_o = weightBuilder(layer_index, gate_weights, input_size,  1);
        nvinfer1::Weights W_f = weightBuilder(layer_index, gate_weights, input_size,  2);
        nvinfer1::Weights W_c = weightBuilder(layer_index, gate_weights, input_size,  3);
        nvinfer1::Weights R_i = weightBuilder(layer_index, rcur_weights, hidden_size, 0);
        nvinfer1::Weights R_o = weightBuilder(layer_index, rcur_weights, hidden_size, 1);
        nvinfer1::Weights R_f = weightBuilder(layer_index, rcur_weights, hidden_size, 2);
        nvinfer1::Weights R_c = weightBuilder(layer_index, rcur_weights, hidden_size, 3);

        bool isW = true;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, W_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, W_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, W_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, W_c);
        isW = false;
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, R_i);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, R_o);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, R_f);
        layer->setWeightsForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, R_c);

        nvinfer1::Weights B_wi = biasBuilder(layer_index, bias_weights, 0);
        nvinfer1::Weights B_wo = biasBuilder(layer_index, bias_weights, 1);
        nvinfer1::Weights B_wf = biasBuilder(layer_index, bias_weights, 2);
        nvinfer1::Weights B_wc = biasBuilder(layer_index, bias_weights, 3);
        nvinfer1::Weights B_ri = biasBuilder(layer_index, bias_weights, 4);
        nvinfer1::Weights B_ro = biasBuilder(layer_index, bias_weights, 5);
        nvinfer1::Weights B_rf = biasBuilder(layer_index, bias_weights, 6);
        nvinfer1::Weights B_rc = biasBuilder(layer_index, bias_weights, 7);

        isW = true;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, B_wi);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_wo);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_wf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, B_wc);
        isW = false;
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kINPUT,  isW, B_ri);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kOUTPUT, isW, B_ro);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kFORGET, isW, B_rf);
        layer->setBiasForGate(layer_index, nvinfer1::RNNGateType::kCELL,   isW, B_rc);
    }

    // Outputs
    ASSERT(layer->getNbOutputs() == 3, ErrorCode::kINTERNAL_ERROR);
    ASSERT(node.output_size() <= 3, ErrorCode::kINVALID_NODE);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < node.output_size(); i++) {
        auto* shuffle_layer = ctx->network()->addShuffle(*(layer->getOutput(i)));
        ASSERT(shuffle_layer && "Failed to create output shuffle layer",
               ErrorCode::kINTERNAL_ERROR);
        shuffle_layer->setFirstTranspose(nvinfer1::Permutation{1,0,2});
        if (i == 0) {
            nvinfer1::Dims Y_dims{4, {max_seq_len, batch_size, num_directions, hidden_size}};
            shuffle_layer->setReshapeDimensions(Y_dims);
            shuffle_layer->setSecondTranspose(nvinfer1::Permutation{0,2,1,3});
        }
        outputs.emplace_back(shuffle_layer->getOutput(0));
    }
    return {outputs};
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul)
{
    nvinfer1::ITensor* inputA = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::ITensor* inputB = &convertToTensor(inputs.at(1), ctx);

    broadcast_tensors(ctx, inputA, inputB);

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
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    ASSERT(dims.nbDims >= 2, ErrorCode::kINVALID_NODE);

    bool need_to_expand_dims = (dims.nbDims == 3);
    if (need_to_expand_dims)
    {
        // Expand spatial dims from 1D to 2D
        nvinfer1::DimsNCHW new_shape(dims.d[0], dims.d[1], dims.d[2], 1);
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    int nbSpatialDims = dims.nbDims - 2;

    // Support for opset10 ceil_mode
    CeilingPoolDim ceilingPool;
    // Ceiling and dialations added in opset 10
    if (ctx->getOpsetVersion() >= 10)
    {
        OnnxAttrs attrs(node);
        const auto ceil_mode = attrs.get<int>("ceil_mode", 0);
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT(dilations[i] == 1, ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
        if (ceil_mode != 0) // Need to set pooling formula to use ceiling instead of floor
        {
            ctx->network()->setPoolingOutputDimensionsFormula(&ceilingPool);
        }
    }
    ASSERT(nbSpatialDims == 2 || nbSpatialDims == 3, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;
    bool exclude_padding;
    get_kernel_params(node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);
    nvinfer1::IPoolingLayer* layer
        = ctx->network()->addPoolingNd(*tensor_ptr, nvinfer1::PoolingType::kMAX, kernel_size);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setStrideNd(strides);
    layer->setPaddingMode(paddingMode);
    layer->setPrePadding(beg_padding);
    layer->setPostPadding(end_padding);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();

    if (need_to_expand_dims)
    {
        // Un-expand spatial dims back to 1D
        nvinfer1::Dims new_shape{3, {dims.d[0], dims.d[1], dims.d[2]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensor_ptr}};
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
    bool isAnyOperandInt32 = check_for_int32(inputs);
    if (inputs.at(0).is_tensor() != inputs.at(1).is_tensor() && !isAnyOperandInt32)
    {
        return scaleHelper(
            ctx, node, inputs, ScaleOp::kSCALE);
    }
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPROD, true);
}

DEFINE_BUILTIN_OP_IMPORTER(Neg)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::DimsHW beg_padding, end_padding;
    OnnxAttrs attrs(node);
    auto mode = attrs.get<std::string>("mode", "constant");
    float value = attrs.get<float>("value", 0.f);
    ASSERT(mode == "constant" && value == 0, ErrorCode::kUNSUPPORTED_NODE);
    if (attrs.count("paddings"))
    {
        // TODO: This is a WAR for old versions of ONNX and should be removed in future
        auto onnx_padding = attrs.get<std::vector<int>>("paddings");
        ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 && onnx_padding[2] == 0 && onnx_padding[3] == 0,
            ErrorCode::kUNSUPPORTED_NODE);
        beg_padding.h() = onnx_padding[4];
        end_padding.h() = onnx_padding[5];
        beg_padding.w() = onnx_padding[6];
        end_padding.w() = onnx_padding[7];
        RETURN_FIRST_OUTPUT(ctx->network()->addPadding(tensor, beg_padding, end_padding));
    }
    auto onnx_padding = attrs.get<std::vector<int>>("pads");
    ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 && onnx_padding[4] == 0 && onnx_padding[5] == 0,
        ErrorCode::kUNSUPPORTED_NODE);
    beg_padding.h() = onnx_padding[2];
    beg_padding.w() = onnx_padding[3];
    end_padding.h() = onnx_padding[6];
    end_padding.w() = onnx_padding[7];
    RETURN_FIRST_OUTPUT(ctx->network()->addPadding(tensor, beg_padding, end_padding));
}

DEFINE_BUILTIN_OP_IMPORTER(ParametricSoftplus)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSOFTPLUS, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Pow)
{
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    bool isAnyOperandInt32 = check_for_int32(inputs);
    if (inputs.at(0).is_tensor() != inputs.at(1).is_tensor() && !isAnyOperandInt32)
    {
    return scaleHelper(
        ctx, node, inputs, ScaleOp::kPOWER);
    }
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kPOW, true);
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu)
{
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    const auto& shape1 = inputs.at(0).shape();
    nvinfer1::ITensor* slopes{};
    if (inputs.at(1).is_tensor())
    {
        if (inputs.at(1).shape().nbDims < shape1.nbDims)
        {
            nvinfer1::IShuffleLayer* reshapeLayer = ctx->network()->addShuffle(inputs.at(1).tensor());
            ASSERT(reshapeLayer, ErrorCode::kUNSUPPORTED_NODE);
            reshapeLayer->setReshapeDimensions(expand_dims(inputs.at(1).shape(), shape1.nbDims));
            slopes = reshapeLayer->getOutput(0);
        }
        else
        {
            slopes = &convertToTensor(inputs.at(1), ctx);
        }
        const auto& shape2 = slopes->getDimensions();
        ASSERT(shape1.nbDims == shape2.nbDims, ErrorCode::kUNSUPPORTED_NODE);
        for (int i = 0; i < shape1.nbDims; ++i)
        {
            ASSERT(shape1.d[i] == shape2.d[i] || shape2.d[i] == 1, ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    else
    {
        auto weights = inputs.at(1).weights();
        if (inputs.at(1).shape().nbDims < shape1.nbDims)
        {
            weights.shape = expand_dims(weights.shape, shape1.nbDims);
        }
        auto constantLayer = ctx->network()->addConstant(weights.shape, weights);
        ASSERT(constantLayer, ErrorCode::kUNSUPPORTED_NODE);
        slopes = constantLayer->getOutput(0);
    }
    ASSERT(input.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(slopes->getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(ctx->network()->addParametricReLU(input, *slopes));
}

DEFINE_BUILTIN_OP_IMPORTER(Reciprocal)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

NodeImportResult reduceTensor(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, TensorOrWeights input,
    nvinfer1::ReduceOperation operation)
{
    nvinfer1::ITensor& tensor = convertToTensor(input, ctx);
    OnnxAttrs attrs(node);
    bool keepdims = attrs.get("keepdims", 1);
    int ndim = tensor.getDimensions().nbDims;
    std::vector<int> axes;
    if (attrs.count("axes"))
    {
        axes = attrs.get<std::vector<int>>("axes");
    }
    else
    {
        axes.resize(ndim);
        std::iota(axes.begin(), axes.end(), 0);
    }

    uint32_t axisMask = 0;
    for (int axis : axes)
    {
        TRT_CHECK(convert_axis(axis, ndim));
        axisMask |= 1 << axis;
    }

    RETURN_FIRST_OUTPUT(ctx->network()->addReduce(tensor, operation, axisMask, keepdims));
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceL1)
{
    NodeImportResult abs_result = apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
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
    return apply_unary_function(ctx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp)
{
    // TODO: Abstract this sequence with a function or macro
    auto exp_result = apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
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
    return apply_unary_function(ctx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
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
    auto input = inputs.at(0);
    nvinfer1::Dims new_shape;
    bool newshape_is_set = false;

    if (ctx->getOpsetVersion() >= 5)
    {
        ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
        auto new_shape_input = inputs.at(1);

        if (new_shape_input.is_weights())
        {
            ShapedWeights new_shape_weights = new_shape_input.weights();
            ASSERT(new_shape_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
            ASSERT(new_shape_weights.type == ::ONNX_NAMESPACE::TensorProto::INT64, ErrorCode::kINVALID_NODE);
            int64_t const* new_shape_ptr = static_cast<int64_t const*>(new_shape_weights.values);
            new_shape.nbDims = new_shape_weights.shape.d[0];
            std::copy(new_shape_ptr, new_shape_ptr + new_shape.nbDims, new_shape.d);
            newshape_is_set = true;
        }
    }
    else
    {
        OnnxAttrs attrs(node);
        new_shape = attrs.get<nvinfer1::Dims>("shape");
        newshape_is_set = true;
    }

    // Copy dimensions from the input
    if (newshape_is_set)
    {
        for (int i = 0; i < new_shape.nbDims; i++)
        {
            if (new_shape.d[i] == 0)
            {
                new_shape.d[i] = input.shape().d[i];
            }
        }
    }

    int infer_dim = -1;
    if (input.is_weights())
    {
        auto weights = input.weights();
        TRT_CHECK(get_infer_dim(infer_dim, new_shape));
        if (infer_dim >= 0)
        {
            // Update the dim to the correct value
            int new_dim = get_shape_size(weights.shape) / (-1 * get_shape_size(new_shape));
            new_shape.d[infer_dim] = new_dim;
        }
        ASSERT(get_shape_size(new_shape) == get_shape_size(weights.shape), ErrorCode::kUNSUPPORTED_NODE);
        weights.shape = new_shape;
        return {{weights}};
    }
    else
    {
        nvinfer1::ITensor& tensor = input.tensor();
        if (newshape_is_set)
        {
            TRT_CHECK(get_infer_dim(infer_dim, new_shape));
            if (infer_dim >= 0)
            {
                // Update the dim to the correct value
                int new_dim = get_shape_size(tensor.getDimensions()) / (-1 * get_shape_size(new_shape));
                new_shape.d[infer_dim] = new_dim;
            }
        }

        nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);

        if (!newshape_is_set)
        {
            layer->setInput(1, inputs.at(1).tensor());
        }
        else
        {
            layer->setReshapeDimensions(new_shape);
        }
        
        nvinfer1::ITensor* tensor_ptr = layer->getOutput(0);
        return {{tensor_ptr}};
    }
}



DEFINE_BUILTIN_OP_IMPORTER(ScaledTanh)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha");
    float beta = attrs.get<float>("beta");
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSCALED_TANH, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Selu)
{
  OnnxAttrs attrs(node);
  float alpha = attrs.get("alpha", 1.6732f);
  float beta = attrs.get("gamma", 1.0507f);
  return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSELU, &alpha, &beta);
}

DEFINE_BUILTIN_OP_IMPORTER(Shape)
{
    auto shape = inputs.at(0).shape();
    // Avoid TRT err: Unused Input
    ctx->network()->addIdentity(inputs.at(0).tensor());
    nvinfer1::Dims weight_dims;
    weight_dims.nbDims = 1;
    weight_dims.d[0] = shape.nbDims;
    // Note: Should technically be int64, but int32 allows for TRT compatibility
    auto weights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, weight_dims);
    std::copy(&shape.d[0], &shape.d[0] + shape.nbDims, static_cast<int32_t*>(const_cast<void*>(weights.values)));
    return {{weights}};
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid)
{
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kSIGMOID);
}

DEFINE_BUILTIN_OP_IMPORTER(Sin)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kSIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Sinh)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kSINH);
}

DEFINE_BUILTIN_OP_IMPORTER(Size)
{
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    auto shape = tensor_ptr->getDimensions();
    // Avoid TRT err: Unused Input
    ctx->network()->addIdentity(*tensor_ptr);
    nvinfer1::Dims weight_dims;
    weight_dims.nbDims = 1;
    weight_dims.d[0] = shape.nbDims;
    // Note: Should technically be int64, but int32 allows for TRT compatibility
    auto weights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto::INT32, weight_dims);
    int32_t size = get_shape_size(shape);
    *static_cast<int32_t*>(const_cast<void*>(weights.values)) = size;
    return {{weights}};
}

// TRT-7031: add tests
DEFINE_BUILTIN_OP_IMPORTER(Slice)
{
    // If opset version >= 10 slice paramerters are weights instead of attributes
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    std::vector<int32_t> starts;
    std::vector<int32_t> ends;
    std::vector<int32_t> axes;
    std::vector<int32_t> steps;
    if (ctx->getOpsetVersion() >= 10)
    {
        ASSERT(node.input().size() >= 3 && node.input().size() <= 5, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

        slice_array(inputs.at(1), starts);
        slice_array(inputs.at(2), ends);
        if (node.input().size() == 4)
        {
            auto const& input_name = node.input(3);
            ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);

            ASSERT(input_name == "axes" ||  input_name == "steps", ErrorCode::kUNSUPPORTED_NODE);
            if (input_name == "axes")
            {
                slice_array(inputs.at(3), axes);
            }
            else if (input_name == "steps")
            {
                slice_array(inputs.at(3), steps);
            }
        }
        else if (node.input().size() == 5)
        {
            ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
            ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
            slice_array(inputs.at(3), axes);
            slice_array(inputs.at(4), steps);
        }
        if (axes.size() == 0)
        {
            axes = std::vector<int32_t>(starts.size(), 0);
            for (size_t i = 0; i < axes.size(); i++)
            {
                axes[i] = i;
            }
        }
        if (steps.size() == 0)
        {
            steps = std::vector<int32_t>(starts.size(), 1);
        }
    }
    else
    {
        OnnxAttrs attrs(node);
        std::vector<int64_t> start_attribute_vector = attrs.get<std::vector<int64_t>>("starts");
        std::vector<int32_t> start_attr(start_attribute_vector.begin(), start_attribute_vector.end());
        starts = start_attr;
        std::vector<int64_t> end_attribute_vector = attrs.get<std::vector<int64_t>>("ends");
        std::vector<int32_t> end_attr(end_attribute_vector.begin(), end_attribute_vector.end());
        ends = end_attr;
        std::vector<int64_t> axis_attribute_vector = attrs.get<std::vector<int64_t>>("axes");
        std::vector<int32_t> axis_attr(axis_attribute_vector.begin(), axis_attribute_vector.end());
        axes = axis_attr;
        steps = std::vector<int32_t>(starts.size(), 1);
    }

    const nvinfer1::Dims dims = tensor.getDimensions();
    const int nbDims = dims.nbDims;
    auto makeDims = [nbDims](int initVal) -> nvinfer1::Dims {
        nvinfer1::Dims result{nbDims, {}, {}};
        std::fill_n(&result.d[0], nbDims, initVal);
        return result;
    };
    nvinfer1::Dims sliceStart = makeDims(0);
    nvinfer1::Dims sliceEnd = dims;
    nvinfer1::Dims sliceSize = dims;
    nvinfer1::Dims sliceStride = makeDims(1); // ONNX has support for strides before opset 10

    for (size_t i = 0; i < axes.size(); i++)
    {
        int axis = axes[i];

        // Negative axis conversion
        TRT_CHECK(convert_axis(axis, nbDims));

        // Special pass through for no-ops (slice across the whole dimension, [:])
        if (starts[i] == 0 && ends[i] >= dims.d[axis] && steps[i] == 1)
        {
            continue;
        }

        // Check if slice is valid
        ASSERT(steps[i] != 0, ErrorCode::kINVALID_VALUE);
        sliceStride.d[axis] = steps[i];

        int upperlimit = dims.d[axis];
        int lowerlimit = 0;
        if (steps[i] < 0)
        {
            upperlimit = dims.d[axis] - 1;
            lowerlimit = -1;
        }

        // Calculate start index
        // Support for negative indexing
        if (starts[i] < 0)
        {
            sliceStart.d[axis] = std::max(dims.d[axis] + static_cast<int>(starts[i]), lowerlimit);
        }
        else
        {
            sliceStart.d[axis] = std::min(static_cast<int>(starts[i]), upperlimit);
        }

        // Calculate end index
        // Support for negative indexing
        if (ends[i] < 0)
        {
            // Differs from start because starts is inclusive and ends is exclusive
            sliceEnd.d[axis] = std::max(dims.d[axis] + static_cast<int>(ends[i]), lowerlimit);
        }
        else
        {
            sliceEnd.d[axis] = std::min(static_cast<int>(ends[i]), upperlimit);
        }

        sliceSize.d[axis] = std::max(
            static_cast<int>(std::ceil(static_cast<float>(sliceEnd.d[axis] - sliceStart.d[axis]) / steps[i])), 0);
    }

    // If entire slice op was a no-op, simply return the input tensor
    if (sliceSize == makeDims(0))
    {
        return {{&tensor}};
    }
    // TensorRT cannot handle size 0 slices
    else
    {
        for (int i = 0; i < nbDims; i++)
        {
            ASSERT(sliceSize.d[i] != 0, ErrorCode::kINVALID_VALUE);
        }
    }

    RETURN_FIRST_OUTPUT(ctx->network()->addSlice(tensor, sliceStart, sliceSize, sliceStride));
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax)
{
    OnnxAttrs attrs(node);
    int axis = attrs.get("axis", 1);
    int nbDims = inputs.at(0).shape().nbDims;
    TRT_CHECK(convert_axis(axis, nbDims));
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims shape = tensor_ptr->getDimensions();

    // Work around dynamic input shapes like [-1, -1, X] which cannot be reshape to 2D
    if (shape.nbDims == axis + 1)
    {
        auto* layer = ctx->network()->addSoftMax(*tensor_ptr);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
        // Set softmax bitmask to the last dimension
        layer->setAxes(1 << axis);
        tensor_ptr = layer->getOutput(0);
        return {{tensor_ptr}};
    }
    // Reshape the tensor to 2D and do softmax on the second dimension
    ASSERT(tensor_ptr = convert_tensor_to_2d(ctx, *tensor_ptr, axis), ErrorCode::kUNSUPPORTED_NODE);
    auto* layer = ctx->network()->addSoftMax(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    // Set softmax bitmask to the second dimension
    layer->setAxes(1 << 1);
    tensor_ptr = layer->getOutput(0);
    // Reshape the tensor back to original shape
    ASSERT(tensor_ptr = reshape_tensor(ctx, *tensor_ptr, shape), ErrorCode::kUNSUPPORTED_NODE);
    return {{tensor_ptr}};
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
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node);
    int block_size = attrs.get<int>("blocksize");
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int ndim_spatial = dims.nbDims - 2;
    nvinfer1::Dims new_shape1;
    new_shape1.nbDims = dims.nbDims + ndim_spatial;
    new_shape1.d[0] = dims.d[0];
    new_shape1.d[1] = dims.d[1];
    for (int i = 1; i <= ndim_spatial; ++ i)
    {
        ASSERT(dims.d[1 + i] % block_size == 0, ErrorCode::kINVALID_NODE);
        new_shape1.d[2 * i] = dims.d[1 + i] / block_size;
        new_shape1.d[2 * i + 1] = block_size;
    }
    layer->setReshapeDimensions(new_shape1);
    nvinfer1::Permutation perm;
    perm.order[0] = 0;
    perm.order[ndim_spatial + 1] = 1;

    for (int i = 1; i <= ndim_spatial; ++ i)
    {
        perm.order[ndim_spatial + 1 + i] = 2 * i ;
        perm.order[i] = 2 * i + 1;
    }
    layer->setSecondTranspose(perm);
    tensor_ptr = layer->getOutput(0);
    dims = tensor_ptr->getDimensions();
    nvinfer1::Dims new_shape2;
    new_shape2.nbDims = dims.nbDims - ndim_spatial;
    new_shape2.d[0] = dims.d[0];
    new_shape2.d[1] = dims.d[1];
    for (int i = 1; i <= ndim_spatial; ++i)
    {
        new_shape2.d[1] *= dims.d[i + 1];
        new_shape2.d[1 + i] = dims.d[ndim_spatial + 1 + i];
    }
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape2);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
    return {{tensor_ptr}};
}

// TODO: Legacy op for pre-1.0 ONNX spec; can be removed at some point
DEFINE_BUILTIN_OP_IMPORTER(SpatialBN)
{
    return importBatchNormalization(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Split)
{
    ASSERT(inputs.size() == 1, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor* tensor_ptr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    int nbDims = dims.nbDims;
    OnnxAttrs attrs(node);
    int axis = attrs.get<int>("axis", 0);
    TRT_CHECK(convert_axis(axis, nbDims));
    std::vector<int> output_lengths;
    int noutput = node.output().size();
    std::vector<int> start_index (noutput, 0);
    if (attrs.count("split"))
    {
        output_lengths = attrs.get<std::vector<int>>("split");
        ASSERT(static_cast<int>(output_lengths.size()) == noutput, ErrorCode::kINVALID_NODE);
    }
    else
    {
        ASSERT(dims.d[axis] == -1 || dims.d[axis] % noutput == 0, ErrorCode::kINVALID_NODE);
        output_lengths.assign(noutput, dims.d[axis] / noutput);
    }
    for (size_t i = 1; i < output_lengths.size(); i++)
    {
        start_index[i] = start_index[i - 1] + output_lengths[i - 1];
    }

    nvinfer1::Dims sliceStart = makeDims(nbDims, 0);
    nvinfer1::Dims sliceSize = dims;
    nvinfer1::Dims sliceStride = makeDims(nbDims, 1);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < noutput; ++i)
    {
        sliceStart.d[axis] = start_index[i];
        sliceSize.d[axis] = output_lengths[i];
        auto const layer = ctx->network()->addSlice(*tensor_ptr, sliceStart, sliceSize, sliceStride);
        // std::vector<int> size_vecto(dims.d, dims.d+dims.nbDims);
        if (std::any_of(sliceSize.d, sliceSize.d + sliceSize.nbDims, [](int i){return i == -1;})){
            layer->setInput(1, dimension_to_tensor(ctx, sliceStart));
            layer->setInput(2, dimension_to_tensor(ctx, sliceSize));
            layer->setInput(3, dimension_to_tensor(ctx, sliceStride));
        }
        outputs.push_back(layer->getOutput(0));
    }
    return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt)
{
    return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
}

DEFINE_BUILTIN_OP_IMPORTER(Squeeze)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims old_shape = tensor.getDimensions();
    int ndim_in = old_shape.nbDims;
    OnnxAttrs attrs(node);
    auto axes = attrs.get<std::vector<int>>("axes");
    for (auto& axis : axes)
    {
        TRT_CHECK(convert_axis(axis, ndim_in));
    }
    std::set<int> axes_set(axes.begin(), axes.end());
    int ndim_out = ndim_in - axes_set.size();
    ASSERT(ndim_out <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims new_shape;
    new_shape.nbDims = ndim_out;

    // Squeeze into a constant
    if (new_shape.nbDims == 0)
    {
        nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
        layer->setReshapeDimensions(new_shape);
        RETURN_FIRST_OUTPUT(layer);
    }

    nvinfer1::Permutation perm;

    // Align dynamic dimensions in input with the first permutation and 0s in new_shape
    // i: from 0 to new_shape.nbDims
    // j: from 0 to number of axes
    for (int i = 0, j = 0; (i + j) < old_shape.nbDims; )
    {
        if (!axes_set.count(i + j))
        {
            new_shape.d[i] = 0;
            perm.order[i] = i + j;
            i ++;
        }
        else
        {
            ASSERT((old_shape.d[i + j] == 1 || old_shape.d[i + j] == -1), ErrorCode::kINVALID_NODE);
            perm.order[j + ndim_out] = i + j;
            j ++;
        }
    }

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setFirstTranspose(perm);
    layer->setReshapeDimensions(new_shape);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Sub)
{
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUB, true);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum)
{
    return elementwiseHelper(ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Tan)
{
    return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kTAN);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh)
{
    RETURN_FIRST_OUTPUT(ctx->network()->addActivation(inputs.at(0).tensor(), nvinfer1::ActivationType::kTANH));
}

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu)
{
    OnnxAttrs attrs(node);
    float alpha = attrs.get<float>("alpha", 1.f);
    return activationHelper(ctx, node, inputs, nvinfer1::ActivationType::kTHRESHOLDED_RELU, &alpha);
}

DEFINE_BUILTIN_OP_IMPORTER(TopK)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    ASSERT(tensor.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node);
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

    int nbDims = tensor.getDimensions().nbDims;
    TRT_CHECK(convert_axis(axis, nbDims));
    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(tensor, nvinfer1::TopKOperation::kMAX, k, axisMask);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    return {{layer->getOutput(0), layer->getOutput(1)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose)
{
    TensorOrWeights input = inputs.at(0);
    OnnxAttrs attrs(node);
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
        nvinfer1::ITensor* output_tensor = transpose_tensor(ctx, input.tensor(), perm, false);
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
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims old_shape = tensor.getDimensions();
    int ndim_in = old_shape.nbDims;
    OnnxAttrs attrs(node);
    auto axes = attrs.get<std::vector<int>>("axes");
    std::set<int> axes_set(axes.begin(), axes.end());
    int ndim_out = ndim_in + axes_set.size();
    ASSERT(ndim_out <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Dims new_shape; 
    new_shape.nbDims = ndim_out;
    nvinfer1::Permutation perm;

    // Append 1 to new_shape for each axes and permute them into the right position
    // Align dynamic dimensions in input and 0s in new_shape
    // i: from 0 to old_shape.nbDims
    // j: from 0 to number of axes
    for (int i = 0, j = 0 ; (i + j) < ndim_in + (int)axes_set.size(); )
    {
        if (axes_set.count(i+j))
        {
            perm.order[i + j] = ndim_in + j;
            new_shape.d[ndim_in + j] = 1;
            j ++;
        }
        else
        {
            perm.order[i + j] = i;
            old_shape.d[i] < 0 ? new_shape.d[i] = 0 : new_shape.d[i] =old_shape.d[i];
            i ++;
        }
    }

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setReshapeDimensions(new_shape);
    layer->setSecondTranspose(perm);
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Resize)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    int input_dims = input.getDimensions().nbDims;
    ASSERT(input_dims > 0, ErrorCode::kUNSUPPORTED_NODE);

    // Add resize layer
    nvinfer1::IResizeLayer* layer = ctx->network()->addResize(input);

    // Retrive and validate scale factors.
    // Scale factors include batch dimensions as well.
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
    auto scales = inputs.at(1);
    // Support for scales as weights
    ASSERT(scales.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights scales_weights = scales.weights();
    ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales_weights.count() == static_cast<size_t>(input_dims), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
    // Get floating point scale factors.
    float const* scales_ptr = static_cast<float const*>(scales_weights.values);
    layer->setScales(scales_ptr, input_dims);

    // Set resize mode
    OnnxAttrs attrs(node);
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
    if (mode == "linear")
    {
        // Linear resize support 1-D, 2-D and 3D resize.
        ASSERT((input_dims >= 1) && (input_dims <= 3), ErrorCode::kUNSUPPORTED_NODE);
        resizeMode = nvinfer1::ResizeMode::kLINEAR;
    }
    layer->setResizeMode(resizeMode);

    // Set other attributes. ONNX spec does not have this attribute yet.
    // Default: False. Set it any way.
    layer->setAlignCorners(false);

    // Return layer output
    RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    const int nbDims = tensor.getDimensions().nbDims;
    ASSERT(nbDims > 0, ErrorCode::kUNSUPPORTED_NODE);
    OnnxAttrs attrs(node);
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
        if (!attrs.count("scales"))
        {
            assert(false && "attr has no scales");
        }
        else
        {
            // Get scale factors from OnnxAttrs.
            auto scales = attrs.get<std::vector<float>>("scales");
            // Scale factors has batch dimension.
            ASSERT(scales.size() == static_cast<size_t>(nbDims), ErrorCode::kUNSUPPORTED_NODE);
            for (int i = 0; i < nbDims; i++)
            {
                scale_factors[i] = scales[i];
            }
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

} // namespace

} // namespace onnx2trt
