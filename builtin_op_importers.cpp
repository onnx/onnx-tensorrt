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
#include "plugin.hpp"
#include "FancyActivation.hpp"
#include "ResizeNearest.hpp"
#include "Split.hpp"
#include "InstanceNormalization.hpp"

namespace onnx2trt {

namespace {

NodeImportResult
combineTensorsElementwise(IImporterContext* ctx,
                          std::vector<TensorOrWeights>& inputs,
                          nvinfer1::ElementWiseOperation binary_op) {
  ASSERT(!inputs.empty(), ErrorCode::kINVALID_NODE);
  std::vector<nvinfer1::ITensor*> input_tensors;
  for( auto& input : inputs ) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    input_tensors.push_back(&input.tensor());
  }
  nvinfer1::ITensor* combined = input_tensors.at(0);
  if( input_tensors.size() == 1 ) {
    // Note: Single input must be wrapped in identity to avoid messing up network outputs
    return {{identity(ctx, combined)}};
  }
  for( size_t i=1; i<input_tensors.size(); ++i ) {
    nvinfer1::ITensor* tensor = input_tensors.at(i);
    auto* layer = ctx->network()->addElementWise(
      *combined, *tensor, binary_op);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    combined = layer->getOutput(0);
  }
  return {{combined}};
}

// Returns false if the transpose does not require any data movement (i.e., it's equivalent to a reshape)
bool is_transpose_required(nvinfer1::Dims const& shape,
                           nvinfer1::Permutation const& perm) {
  int ndim = shape.nbDims;
  int prev_significant_dim = 0;
  for( int dst_i=0; dst_i<ndim; ++dst_i ) {
    int src_i = perm.order[dst_i];
    if( shape.d[src_i] != 1 ) {
      if( src_i < prev_significant_dim ) {
        return false;
      }
      prev_significant_dim = src_i;
    }
  }
  return true;
}

// Note: perm should not include the batch dim
nvinfer1::ITensor*
transpose_tensor(IImporterContext* ctx,
                 nvinfer1::ITensor& tensor,
                 nvinfer1::Permutation const& perm,
                 bool permute_dim_types=true) {
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
  if( !layer ) {
    return nullptr;
  }
  nvinfer1::Dims shape = tensor.getDimensions();
  nvinfer1::Dims new_shape;
  new_shape.nbDims = shape.nbDims;
  for( int i=0; i<new_shape.nbDims; ++i ) {
    new_shape.d[i] = 0; // 0 => copy from source
    new_shape.type[i] = shape.type[permute_dim_types ? perm.order[i] : i];
  }
  // TODO: Why do we get incorrect results when this condition is removed?
  if( !is_transpose_required(shape, perm) ) {
    layer->setFirstTranspose(perm);
  }
  layer->setReshapeDimensions(new_shape);
  return layer->getOutput(0);
}

nvinfer1::ITensor*
move_tensor_dimension(IImporterContext* ctx,
                      nvinfer1::ITensor& tensor,
                      int from, int to) {
  int ndim = tensor.getDimensions().nbDims;
  assert(0 <= from && from < ndim);
  assert(0 <= to   && to   < ndim);
  std::vector<int> vperm;
  vperm.reserve(ndim);
  for( int i=0; i<ndim; ++i ) {
    vperm.push_back(i);
  }
  vperm.erase(vperm.begin() + from);
  vperm.insert(vperm.begin() + to, from);
  nvinfer1::Permutation perm;
  std::copy(vperm.begin(), vperm.end(), perm.order);
  return transpose_tensor(ctx, tensor, perm, false);
}

nvinfer1::ITensor*
reshape_tensor(IImporterContext* ctx,
               nvinfer1::ITensor& tensor,
               nvinfer1::Dims shape) {
  if( shape == tensor.getDimensions() ) {
    return &tensor;
  }
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
  if( !layer ) {
    return nullptr;
  }
  layer->setReshapeDimensions(shape);
  return layer->getOutput(0);
}

nvinfer1::ITensor*
flatten_tensor(IImporterContext* ctx,
               nvinfer1::ITensor& tensor,
               int axis=0) {
  nvinfer1::Dims shape = tensor.getDimensions();
  nvinfer1::Dims new_shape = shape;
  for( int i=axis+1; i<shape.nbDims; ++i ) {
    new_shape.d[axis] *= shape.d[i];
    new_shape.d[i] = 1;
  }
  return reshape_tensor(ctx, tensor, new_shape);
}

} // namespace

string_map<NodeImporter>& getBuiltinOpImporterMap() {
  static string_map<NodeImporter> builtin_op_importers;
  return builtin_op_importers;
}

namespace {

bool registerBuiltinOpImporter(std::string op,
                               NodeImporter const& importer) {
  bool inserted = getBuiltinOpImporterMap().insert({op, importer}).second;
  assert(inserted);
  return inserted;
}

#define DECLARE_BUILTIN_OP_IMPORTER(op) \
  NodeImportResult import##op(IImporterContext* ctx, \
                              ::ONNX_NAMESPACE::NodeProto const& node, \
                              std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op) \
  NodeImportResult import##op(IImporterContext* ctx, \
                          ::ONNX_NAMESPACE::NodeProto const& node, \
                          std::vector<TensorOrWeights>& inputs); \
  static const bool op##_registered_builtin_op = \
      registerBuiltinOpImporter(#op, import##op); \
  const void* op##_registered_builtin_op_ptr = &op##_registered_builtin_op; \
  NodeImportResult import##op(IImporterContext* ctx, \
                              ::ONNX_NAMESPACE::NodeProto const& node, \
                              std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer) \
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE); \
  return {{layer->getOutput(0)}}

#define RETURN_IDENTITY(input) do { \
  TensorOrWeights output = identity(ctx, input); \
  ASSERT(output, ErrorCode::kUNSUPPORTED_NODE); \
  return {{output}}; \
} while(0)

DEFINE_BUILTIN_OP_IMPORTER(Abs) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Add) {
  // TODO: Check "broadcast" and "axis" attributes (also needed for Mul etc.)
  nvinfer1::ILayer* layer;
  if( inputs.at(0).is_tensor() && inputs.at(1).is_tensor() ) {
    layer = ctx->network()->addElementWise(
      inputs.at(0).tensor(), inputs.at(1).tensor(),
      nvinfer1::ElementWiseOperation::kSUM);
  } else if( inputs.at(0).is_weights() && inputs.at(1).is_weights() ) {
    // TODO: Add weights together (i.e., constant fold)
    //return Error(ONNX2TRT_UNSUPPORTED_NODE, "Both inputs are weights");
    return MAKE_ERROR("Both inputs are weights",
                      ErrorCode::kUNSUPPORTED_NODE);
  } else {
    auto& tensor = (inputs.at(0).is_tensor() ?
                    inputs.at(0).tensor() :
                    inputs.at(1).tensor());
    auto shift_weights = (inputs.at(0).is_weights() ?
                          inputs.at(0).weights() :
                          inputs.at(1).weights());
    auto scale_weights = ShapedWeights::empty(shift_weights.type);
    auto power_weights = ShapedWeights::empty(shift_weights.type);
    nvinfer1::ScaleMode mode = get_scale_mode(shift_weights.shape);
    if( mode == nvinfer1::ScaleMode::kELEMENTWISE ) {
      // TODO: TRT doesn't support including the batch dim in elementwise,
      //       but we can't do a more specific assertion here yet because
      //       the input tensor's shape may have been padded to WAR TRT's
      //       shape issues.
      ASSERT(get_shape_size(shift_weights.shape) ==
             get_shape_size(tensor.getDimensions()),
             ErrorCode::kUNSUPPORTED_NODE);
    } else if( mode == nvinfer1::ScaleMode::kCHANNEL ) {
      // TRT does not currently support full broadcasting
      OnnxAttrs attrs(node);
      ASSERT(attrs.count("broadcast"), ErrorCode::kUNSUPPORTED_NODE);
      bool broadcast = attrs.get<int>("broadcast");
      ASSERT(broadcast, ErrorCode::kINVALID_NODE);
      int axis = attrs.get<int>("axis", -1);
      if( axis < 0 ) {
        axis += tensor.getDimensions().nbDims; // Support negative indexing
      }
      ASSERT(axis == 1, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(shift_weights.shape.d[0] == tensor.getDimensions().d[0],
             ErrorCode::kUNSUPPORTED_NODE);
    }
    layer = ctx->network()->addScale(
      tensor, mode, shift_weights, scale_weights, power_weights);
  }
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  // TODO: Check ONNX defaults for these
  nvinfer1::DimsHW kernel_size(1, 1), strides(1, 1), beg_padding(0, 0), end_padding(0, 0);
  get_kernel_params(node, get_DimsHW_from_CHW(tensor.getDimensions()),
                    &kernel_size, &strides, &beg_padding, &end_padding);
  nvinfer1::IPoolingLayer* pooling_layer = ctx->network()->addPooling(
    tensor, nvinfer1::PoolingType::kAVERAGE, kernel_size);
  nvinfer1::ILayer* layer = pooling_layer;
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  pooling_layer->setStride(strides);
  // Note: Average pooling requires special care with asymmetric padding
  //       because the need to exclude padding pixels from the average
  //       means we can't just use a pre-padding layer.
  nvinfer1::DimsHW pre_crop(0, 0), post_crop(0, 0);
  for( int d=0; d<2; ++d ) {
    if( end_padding.d[d] == beg_padding.d[d] ) {
      // Symmetric padding, nothing special needed
    } else if( end_padding.d[d] == beg_padding.d[d] + 1 ) {
      // Pad symmetrically such that we get one more output element at
      // the beginning, and then crop it off after the pooling operation.
      beg_padding.d[d] += strides.d[d];
      pre_crop.d[d] = 1;
    } else {
      throw std::invalid_argument(
        "Unsupported form of asymmetric padding for AveragePool op");
    }
  }
  pooling_layer->setPadding(beg_padding);
  if( pre_crop  != nvinfer1::DimsHW(0, 0) ||
      post_crop != nvinfer1::DimsHW(0, 0) ) {
    layer = ctx->network()->addPadding(*pooling_layer->getOutput(0),
                                       -pre_crop, -post_crop);
  }
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(BatchNormalization) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  auto scale_weights    = inputs.at(1).weights();
  auto bias_weights     = inputs.at(2).weights();
  auto mean_weights     = inputs.at(3).weights();
  auto variance_weights = inputs.at(4).weights();
  OnnxAttrs attrs(node);
  float eps = attrs.get<float>("epsilon", 1e-5f);
  // TODO: Check if ONNX "spatial" attribute is important (maybe changes mean and variance broadcasting?)
  ASSERT(scale_weights.type    == ::ONNX_NAMESPACE::TensorProto::FLOAT &&
         bias_weights.type     == ::ONNX_NAMESPACE::TensorProto::FLOAT &&
         mean_weights.type     == ::ONNX_NAMESPACE::TensorProto::FLOAT &&
         variance_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT,
         ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = tensor.getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int nchan = dims.d[0];
  nvinfer1::Dims weights_shape{1, {nchan}};
  ASSERT(scale_weights.shape    == weights_shape, ErrorCode::kINVALID_NODE);
  ASSERT(bias_weights.shape     == weights_shape, ErrorCode::kINVALID_NODE);
  ASSERT(mean_weights.shape     == weights_shape, ErrorCode::kINVALID_NODE);
  ASSERT(variance_weights.shape == weights_shape, ErrorCode::kINVALID_NODE);
  auto combined_scale_weights = ctx->createTempWeights(scale_weights.type, scale_weights.shape);
  auto combined_bias_weights  = ctx->createTempWeights(bias_weights.type,  bias_weights.shape);
  size_t nweight = nchan;
  // Fold the weights together into a single bias and scale
  for( size_t i=0; i<nweight; ++i ) {
    float scale    = (static_cast<float const*>(scale_weights.values))[i];
    float bias     = (static_cast<float const*>(bias_weights.values))[i];
    float mean     = (static_cast<float const*>(mean_weights.values))[i];
    float variance = (static_cast<float const*>(variance_weights.values))[i];
    float& combined_scale_ref = const_cast<float*>(
        static_cast<float const*>(combined_scale_weights.values))[i];
    float& combined_bias_ref  = const_cast<float*>(
        static_cast<float const*>(combined_bias_weights.values))[i];
    combined_scale_ref = scale / sqrtf(variance + eps);
    combined_bias_ref  = bias - mean * combined_scale_ref;
  }
  auto dummy_power_weights = ShapedWeights::empty(scale_weights.type);
  auto* layer = ctx->network()->addScale(
    tensor, nvinfer1::ScaleMode::kCHANNEL,
    combined_bias_weights, combined_scale_weights, dummy_power_weights);
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Ceil) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::CEIL),
                                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Clip) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  // TODO: Need to use numeric_limits<half> etc for different input types?
  float minval = attrs.get("min", std::numeric_limits<float>::lowest());
  float maxval = attrs.get("max", std::numeric_limits<float>::max());
  RETURN_FIRST_OUTPUT(
    ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::CLIP,
                                             minval, maxval),
                   {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Concat) {
  std::vector<nvinfer1::ITensor*> tensors;
  for( auto& input : inputs ) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensors.push_back(&input.tensor());
  }
  OnnxAttrs attrs(node);
  int axis = attrs.get<int>("axis");
  if( axis < 0 ) {
    // Support negative indexing
    axis += inputs.at(0).shape().nbDims;
  }
  if( axis == 1 ) {
    RETURN_FIRST_OUTPUT(
      ctx->network()->addConcatenation(tensors.data(), tensors.size()));
  } else {
    using namespace nvinfer1::plugin;
    RETURN_FIRST_OUTPUT(
        ctx->addPlugin(new NvPlugin(createConcatPlugin(axis, false)), tensors));
  }
}

DEFINE_BUILTIN_OP_IMPORTER(Constant) {
  // TODO: This silently fails if the dtype is not supported
  OnnxAttrs attrs(node);
  return {{attrs.get<ShapedWeights>("value")}};
}

DEFINE_BUILTIN_OP_IMPORTER(Conv) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  auto kernel_weights = inputs.at(1).weights();
  ASSERT(kernel_weights.shape.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Weights bias_weights;
  if( inputs.size() == 3 ) {
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    auto shaped_bias_weights = inputs.at(2).weights();
    ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
    ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[0], ErrorCode::kINVALID_NODE);
    bias_weights = shaped_bias_weights;
  } else {
    bias_weights = ShapedWeights::empty(kernel_weights.type);
  }
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = kernel_weights.shape.d[2];
  kernel_size.w() = kernel_weights.shape.d[3];
  // TODO: Check ONNX defaults for these
  nvinfer1::DimsHW strides(1, 1);
  nvinfer1::DimsHW beg_padding(0, 0), end_padding(0, 0);
  nvinfer1::DimsHW dilations(1, 1);
  get_kernel_params(node, get_DimsHW_from_CHW(tensor_ptr->getDimensions()),
                    &kernel_size, &strides,
                    &beg_padding, &end_padding, &dilations);
  ASSERT(kernel_size.h() == kernel_weights.shape.d[2], ErrorCode::kINVALID_NODE);
  ASSERT(kernel_size.w() == kernel_weights.shape.d[3], ErrorCode::kINVALID_NODE);
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int nchan = dims.d[0];
  int noutput = kernel_weights.shape.d[0]; // Note: Weights order is KCRS
  if( beg_padding != end_padding ) {
    auto* layer = ctx->network()->addPadding(*tensor_ptr, beg_padding, end_padding);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    tensor_ptr = layer->getOutput(0);
  }
  nvinfer1::IConvolutionLayer* layer = ctx->network()->addConvolution(
    *tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  layer->setStride(strides);
  if( beg_padding == end_padding ) {
    layer->setPadding(beg_padding);
  }
  layer->setDilation(dilations);
  OnnxAttrs attrs(node);
  int ngroup = attrs.get("group", 1);
  ASSERT(kernel_weights.shape.d[1] * ngroup == nchan, ErrorCode::kINVALID_NODE);
  layer->setNbGroups(ngroup);
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  auto kernel_weights = inputs.at(1).weights();
  ASSERT(kernel_weights.shape.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Weights bias_weights;
  if( inputs.size() == 3 ) {
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    auto shaped_bias_weights = inputs.at(2).weights();
    ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
    ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[1], ErrorCode::kINVALID_NODE);
    bias_weights = shaped_bias_weights;
  } else {
    bias_weights = ShapedWeights::empty(kernel_weights.type);
  }
  OnnxAttrs attrs(node);
  nvinfer1::DimsHW input_shape  = get_DimsHW_from_CHW(tensor.getDimensions());
  nvinfer1::DimsHW output_shape;
  if( attrs.count("output_shape") ) {
    output_shape = attrs.get<nvinfer1::DimsHW>("output_shape");
  } else { // TODO: Add support for the new "output_padding" attribute
    ASSERT(attrs.get("auto_pad", std::string("VALID")) == "VALID",
           ErrorCode::kINVALID_NODE);
  }
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = kernel_weights.shape.d[2];
  kernel_size.w() = kernel_weights.shape.d[3];
  // TODO: Check ONNX defaults for these
  nvinfer1::DimsHW strides(1, 1);
  nvinfer1::DimsHW beg_padding(0, 0), end_padding(0, 0);
  nvinfer1::DimsHW dilations(1, 1);
  // Note: output_shape/input_shape are swapped here so that the padding
  //       calculations operate as if it were a regular forward convolution.
  get_kernel_params(node, output_shape,
                    &kernel_size, &strides,
                    &beg_padding, &end_padding, &dilations, &input_shape);
  ASSERT(kernel_size.h() == kernel_weights.shape.d[2], ErrorCode::kINVALID_NODE);
  ASSERT(kernel_size.w() == kernel_weights.shape.d[3], ErrorCode::kINVALID_NODE);
  nvinfer1::Dims dims = tensor.getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int nchan = dims.d[0];
  int ngroup = attrs.get("group", 1);
  int noutput = kernel_weights.shape.d[1] * ngroup; // Note: Weights order is CKRS
  nvinfer1::IDeconvolutionLayer* deconv_layer = ctx->network()->addDeconvolution(
    tensor, noutput, kernel_size, kernel_weights, bias_weights);
  nvinfer1::ILayer* layer = deconv_layer;
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  deconv_layer->setStride(strides);
  if( beg_padding == end_padding ) {
    deconv_layer->setPadding(beg_padding);
  } else {
    layer = ctx->network()->addPadding(*deconv_layer->getOutput(0),
                                       -beg_padding, -end_padding);
  }
  ASSERT(dilations.h() == 1 && dilations.w() == 1, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(kernel_weights.shape.d[0] == nchan, ErrorCode::kINVALID_NODE);
  deconv_layer->setNbGroups(ngroup);
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Div) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kDIV);
}

DEFINE_BUILTIN_OP_IMPORTER(Dropout) {
  RETURN_IDENTITY(inputs.at(0));
}

DEFINE_BUILTIN_OP_IMPORTER(Elu) {
  OnnxAttrs attrs(node);
  float alpha = attrs.get<float>("alpha", 1.f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::ELU, alpha),
                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Exp) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
}

DEFINE_BUILTIN_OP_IMPORTER(Flatten) {
  //nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  OnnxAttrs attrs(node);
  int axis = attrs.get("axis", 1);
  // Note: Flattening to shape=[batch, n] is currently the only sensible operation
  ASSERT(axis == 1, ErrorCode::kUNSUPPORTED_NODE);
  // Note: TRT implicitly flattens the input to FC layers, and in fact
  //       requires that it still has 4D shape, so in this case we
  //       simply ignore the reshape.
  nvinfer1::Dims dims = inputs.at(0).shape();
  // Note: Batch dim is implicit in TRT
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  //return {{inputs.at(0)}};
  // Note: This identity layer is added instead of just skipping the node so
  //       that the output(s) of the network don't get messed up (e.g.,
  //       connected directly to an input, which is not allowed in TRT).
//RETURN_IDENTITY(inputs.at(0));
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = flatten_tensor(ctx, inputs.at(0).tensor());
  ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  return {{tensor_ptr}};
  // TODO: Use this once reshape issues are solved in TRT
  //nvinfer1::Dims new_shape;
  //new_shape.nbDims = 1;
  //new_shape.d[0] = get_shape_size(dims);
  //new_shape.type[0] = nvinfer1::DimensionType::kCHANNEL;
  //auto* layer = ctx->network()->addShuffle(inputs.at(0).tensor());
  //ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  //layer->setReshapeDimensions(new_shape);
  //RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Floor) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::FLOOR),
                                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm) {
  // Note: Currently this only supports A=tensor, B=weights, C=biases
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  auto weights = inputs.at(1).weights();
  auto biases  = inputs.at(2).weights();
  OnnxAttrs attrs(node);
  float alpha = attrs.get("alpha", 1.f);
  float beta  = attrs.get("beta",  1.f);
  bool broadcast = attrs.get("broadcast", false);
  bool trans_a = attrs.get("transA", false);
  bool trans_b = attrs.get("transB", false);
  ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  // TODO: Once TRT supports arbitrary reshape, this will be needed (also in MatMul)
  //if( dims.nbDims == 1 ) {
  //  // TensorRT requires CHW input for FullyConnected layers
  //  nvinfer1::DimsCHW new_shape(dims.d[0], 1, 1);
  //  auto* layer = ctx->network()->addShuffle(tensor);
  //  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  //  layer->setReshapeDimensions(new_shape);
  //  tensor_ptr = layer->getOutput(0);
  //  dims = tensor_ptr->getDimensions();
  //}
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int ninput = dims.d[0] * dims.d[1] * dims.d[2];
  ASSERT(!trans_a, ErrorCode::kUNSUPPORTED_NODE);
  if( !trans_b ) {
    auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
    ASSERT(transposeWeights(weights, {1, 0}, &new_weights),
           ErrorCode::kUNSUPPORTED_NODE);
    weights = new_weights;
  }
  ASSERT(weights.shape.d[1] == ninput, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(alpha == 1.f, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(beta  == 1.f, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(broadcast, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(biases.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
  int nrow = biases.shape.d[0];
  ASSERT(weights.shape.d[0] == nrow, ErrorCode::kINVALID_NODE);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addFullyConnected(*tensor_ptr, nrow, weights, biases));
}

DEFINE_BUILTIN_OP_IMPORTER(GlobalAveragePool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  nvinfer1::Dims dims = tensor.getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::DimsHW kernel_size(dims.d[1], dims.d[2]);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addPooling(
      tensor, nvinfer1::PoolingType::kAVERAGE, kernel_size));
}

DEFINE_BUILTIN_OP_IMPORTER(HardSigmoid) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float alpha = attrs.get<float>("alpha", 0.2f);
  float beta  = attrs.get<float>("beta",  0.5f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
          new FancyActivationPlugin(FancyActivationPlugin::HARD_SIGMOID, alpha, beta),
          {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(InstanceNormalization) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  auto scale_weights = inputs.at(1).weights();
  auto bias_weights  = inputs.at(2).weights();
  OnnxAttrs attrs(node);
  float epsilon = attrs.get("epsilon", 1e-5f);
  // HACK TODO: Values < ~1e-4 were found to cause corrupt output in a RTST model. Need to suss this out.
  epsilon = std::max(epsilon, 1e-4f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
        new InstanceNormalizationPlugin(epsilon, scale_weights, bias_weights),
        {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(LeakyRelu) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float alpha = attrs.get<float>("alpha", 0.01f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
         new FancyActivationPlugin(FancyActivationPlugin::LEAKY_RELU, alpha),
         {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Log) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kLOG);
}

DECLARE_BUILTIN_OP_IMPORTER(Softmax);
DEFINE_BUILTIN_OP_IMPORTER(LogSoftmax) {
  auto result = importSoftmax(ctx, node, inputs);
  if( result.is_error() ) {
    return result;
  } else {
    auto& input = result.value().at(0);
    return apply_unary_function(ctx, input, nvinfer1::UnaryOperation::kLOG);
  }
}

DEFINE_BUILTIN_OP_IMPORTER(LRN) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  OnnxAttrs attrs(node);
  int   size  = attrs.get<int>("size");
  float alpha = attrs.get<float>("alpha");
  float beta  = attrs.get<float>("beta");
  float bias  = attrs.get<float>("bias");
  RETURN_FIRST_OUTPUT(
    ctx->network()->addLRN(tensor, size, alpha, beta, bias));
}

DEFINE_BUILTIN_OP_IMPORTER(MatMul) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  // Note: Currently this only supports A=tensor, B=weights
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  auto weights = inputs.at(1).weights();
  auto biases  = ShapedWeights::empty(weights.type);
  OnnxAttrs attrs(node);
  ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = tensor.getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  //if( !trans_b ) {
  {
    auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
    ASSERT(transposeWeights(weights, {1, 0}, &new_weights),
           ErrorCode::kUNSUPPORTED_NODE);
    weights = new_weights;
  }
    //}
  int ninput = dims.d[0] * dims.d[1] * dims.d[2];
  ASSERT(weights.shape.d[1] == ninput, ErrorCode::kINVALID_NODE);
  int nrow = weights.shape.d[0];
  RETURN_FIRST_OUTPUT(
    ctx->network()->addFullyConnected(tensor, nrow, weights, biases));
}

DEFINE_BUILTIN_OP_IMPORTER(Max) {
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  // TODO: Check ONNX defaults for these
  nvinfer1::DimsHW kernel_size(1, 1), strides(1, 1), beg_padding(0, 0), end_padding(0, 0);
  get_kernel_params(node, get_DimsHW_from_CHW(tensor_ptr->getDimensions()),
                    &kernel_size, &strides, &beg_padding, &end_padding);
  if( beg_padding != end_padding ) {
    auto* layer = ctx->network()->addPadding(*tensor_ptr, beg_padding, end_padding);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    tensor_ptr = layer->getOutput(0);
  }
  nvinfer1::IPoolingLayer* layer = ctx->network()->addPooling(
    *tensor_ptr, nvinfer1::PoolingType::kMAX, kernel_size);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  layer->setStride(strides);
  if( beg_padding == end_padding ) {
    layer->setPadding(beg_padding);
  }
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Min) {
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul) {
  nvinfer1::ILayer* layer;
  if( inputs.at(0).is_tensor() && inputs.at(1).is_tensor() ) {
    layer = ctx->network()->addElementWise(
      inputs.at(0).tensor(), inputs.at(1).tensor(),
      nvinfer1::ElementWiseOperation::kPROD);
  } else if( inputs.at(0).is_weights() && inputs.at(1).is_weights() ) {
    // TODO: Add weights together (i.e., constant fold)
    //         Implement a general binary op function supporting Add/Mul/etc. with all paths including const folding
    ASSERT(false, ErrorCode::kUNSUPPORTED_NODE);
  } else {
    auto& tensor = (inputs.at(0).is_tensor() ?
                    inputs.at(0).tensor() :
                    inputs.at(1).tensor());
    auto scale_weights = (inputs.at(0).is_weights() ?
                          inputs.at(0).weights() :
                          inputs.at(1).weights());
    auto shift_weights = ShapedWeights::empty(scale_weights.type);
    auto power_weights = ShapedWeights::empty(scale_weights.type);
    nvinfer1::ScaleMode mode = get_scale_mode(scale_weights.shape);
    if( mode == nvinfer1::ScaleMode::kELEMENTWISE ) {
      // TODO: TRT doesn't support including the batch dim in elementwise,
      //       but we can't do a more specific assertion here yet because
      //       the input tensor's shape may have been padded to WAR TRT's
      //       shape issues.
      ASSERT(get_shape_size(scale_weights.shape) == get_shape_size(tensor.getDimensions()),
             ErrorCode::kUNSUPPORTED_NODE);
    } else if( mode == nvinfer1::ScaleMode::kCHANNEL ) {
      // TRT does not currently support full broadcasting
      OnnxAttrs attrs(node);
      ASSERT(attrs.count("broadcast"), ErrorCode::kUNSUPPORTED_NODE);
      bool broadcast = attrs.get<int>("broadcast");
      ASSERT(broadcast, ErrorCode::kINVALID_NODE);
      int axis = attrs.get<int>("axis", -1);
      if( axis < 0 ) {
        axis += tensor.getDimensions().nbDims; // Support negative indexing
      }
      ASSERT(axis == 1, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(scale_weights.shape.d[0] == tensor.getDimensions().d[0],
             ErrorCode::kUNSUPPORTED_NODE);
    }
    layer = ctx->network()->addScale(
      tensor, mode, shift_weights, scale_weights, power_weights);
  }
  RETURN_FIRST_OUTPUT(layer);
}

DEFINE_BUILTIN_OP_IMPORTER(Neg) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kNEG);
}

DEFINE_BUILTIN_OP_IMPORTER(Pad) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  nvinfer1::DimsHW beg_padding, end_padding;
  OnnxAttrs attrs(node);
  auto mode = attrs.get<std::string>("mode", "constant");
  float value = attrs.get<float>("value", 0.f);
  ASSERT(mode == "constant" && value == 0, ErrorCode::kUNSUPPORTED_NODE);
  if( attrs.count("paddings") ) {
    // TODO: This is a WAR for old versions of ONNX and should be removed in future
    auto onnx_padding = attrs.get<std::vector<int>>("paddings");
    ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 &&
           onnx_padding[2] == 0 && onnx_padding[3] == 0,
           ErrorCode::kUNSUPPORTED_NODE);
    beg_padding.h() = onnx_padding[4];
    end_padding.h() = onnx_padding[5];
    beg_padding.w() = onnx_padding[6];
    end_padding.w() = onnx_padding[7];
    RETURN_FIRST_OUTPUT(
      ctx->network()->addPadding(tensor, beg_padding, end_padding));
  }
  auto onnx_padding = attrs.get<std::vector<int>>("pads");
  ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(onnx_padding[0] == 0 && onnx_padding[1] == 0 &&
         onnx_padding[4] == 0 && onnx_padding[5] == 0,
         ErrorCode::kUNSUPPORTED_NODE);
  beg_padding.h() = onnx_padding[2];
  beg_padding.w() = onnx_padding[3];
  end_padding.h() = onnx_padding[6];
  end_padding.w() = onnx_padding[7];
  RETURN_FIRST_OUTPUT(
    ctx->network()->addPadding(tensor, beg_padding, end_padding));
}

DEFINE_BUILTIN_OP_IMPORTER(Pow) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kPOW);
}

DEFINE_BUILTIN_OP_IMPORTER(PRelu) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ShapedWeights weights = inputs.at(1).weights();
  ASSERT(weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT,
         ErrorCode::kUNSUPPORTED_NODE);
  // TODO: Add support for per-channel scale factor
  nvinfer1::Dims scalar_shape{1, {1}};
  ASSERT(weights.shape == scalar_shape, ErrorCode::kUNSUPPORTED_NODE);
  float alpha = *reinterpret_cast<float const*>(weights.values);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
         new FancyActivationPlugin(FancyActivationPlugin::LEAKY_RELU, alpha),
         {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Reciprocal) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kRECIP);
}

DEFINE_BUILTIN_OP_IMPORTER(Relu) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addActivation(
      inputs.at(0).tensor(), nvinfer1::ActivationType::kRELU));
}

DEFINE_BUILTIN_OP_IMPORTER(Reshape) {
  auto input = inputs.at(0);
  nvinfer1::Dims new_shape;
  if( ctx->getOpsetVersion() >= 5 ) {
    auto new_shape_input = inputs.at(1);
    ASSERT(new_shape_input.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ShapedWeights new_shape_weights = new_shape_input.weights();
    ASSERT(new_shape_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
    ASSERT(new_shape_weights.type == ::ONNX_NAMESPACE::TensorProto::INT64,
           ErrorCode::kINVALID_NODE);
    int64_t const* new_shape_ptr =
      static_cast<int64_t const*>(new_shape_weights.values);
    new_shape.nbDims = new_shape_weights.shape.d[0];
    std::copy(new_shape_ptr, new_shape_ptr + new_shape.nbDims, new_shape.d);
  } else {
    OnnxAttrs attrs(node);
    new_shape = attrs.get<nvinfer1::Dims>("shape");
  }
  if( input.is_weights() ) {
    auto weights = input.weights();
    ASSERT(get_shape_size(new_shape) == get_shape_size(weights.shape),
           ErrorCode::kUNSUPPORTED_NODE);
    weights.shape = new_shape;
    return {{weights}};
  } else {
    enum { BATCH_DIM = 0 };
    nvinfer1::ITensor& tensor = input.tensor();
    new_shape = set_dims_CHW(remove_dim(new_shape, BATCH_DIM));
    if( new_shape.nbDims == 1 ) {
      // Note: TRT implicitly flattens the input to FC layers, and in fact
      //       requires that it still has 4D shape, so in this case we
      //       simply ignore the reshape.
      RETURN_IDENTITY(inputs.at(0));
    } else {
      // TODO: May be supportable in future TRT releases
      ASSERT(new_shape.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
      ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
      layer->setReshapeDimensions(new_shape);
      ASSERT(get_shape_size(layer->getOutput(0)->getDimensions()) ==
             get_shape_size(input.shape()), ErrorCode::kUNSUPPORTED_NODE);
      RETURN_FIRST_OUTPUT(layer);
    }
  }
}

//DEFINE_BUILTIN_OP_IMPORTER(RNN) {
//  OnnxAttrs attrs(node);
//  std::string direction_str = attrs.get("direction", "forward");
//  ASSERT(direction_str == "forward" || direction_str == "bidirectional",
//         ErrorCode::kUNSUPPORTED_NODE);
//  nvinfer1::RNNDirection direction = (direction_str == "forward" ?
//                                      nvinfer1::RNNDirection::kUNIDIRECTION :
//                                      nvinfer1::RNNDirection::kBIDIRECTION);
//  int hidden_size = attrs.get<int>("hidden_size");
//  std::vector<std::string> default_activation_strs = {"TanH", "TanH"};
//  auto activation_strs = attrs.get("activations", default_activation_strs);
//  ASSERT(activation_strs.size() == 1 || activation_strs.size() == 2,
//         ErrorCode::kINVALID_NODE);
//  if( activation_strs.size() == 2 ) {
//    ASSERT(activation_strs.at(1) == activation_strs.at(0),
//           ErrorCode::kUNSUPPORTED_NODE);
//  }
//  std::string activation_str = activation_strs.at(0);
//  ASSERT(activation_str == "TanH" || activation_str == "Relu",
//         ErrorCode::kUNSUPPORTED_NODE);
//  nvinfer1::RNNOperation op = (activation_str == "TanH" ?
//                                nvinfer1::RNNOperation::kTANH :
//                                nvinfer1::RNNOperation::kRELU);
//  nvinfer1::RNNMode mode = nvinfer1::RnnMode::kLINEAR;
//  int do_output_sequence = attrs.get("output_sequence", 0);
//  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
//  int layer_count = 1;
//  int max_sequence_length = 64; // TODO: How to specify this?
//
//  // TODO: weights = concatenate inputs.at(1).weights() and inputs.at(2).weights() over slowest dim
//  //       biases  = inputs.at(3).weights(); [OPTIONAL, default 0]
//
//  auto* layer = ctx->network()->addRNN(
//    inputs.at(0).tensor(), layer_count, hidden_size, max_seq_length,
//    op, mode, direction, weights, biases);
//
//  // TODO: Looks like we need to transpose the outputs from [1, T, N, dir, C] to [1, T, dir, N, C]
//  //       Return {{output 0, output 1}}, but take care of outputs being optional (i.e., check how many outputs there are, as well as output_sequence)
//}

DEFINE_BUILTIN_OP_IMPORTER(Upsample) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float height_scale = attrs.get<float>("height_scale");
  float width_scale  = attrs.get<float>("width_scale");
  auto mode = attrs.get<std::string>("mode", "nearest");
  ASSERT(mode == "nearest", ErrorCode::kUNSUPPORTED_NODE);
  auto scale = {height_scale, width_scale};
  RETURN_FIRST_OUTPUT(ctx->addPlugin(new ResizeNearestPlugin(scale),
                                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Selu) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float alpha = attrs.get("alpha", 1.6732f);
  float gamma = attrs.get("gamma", 1.0507f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
          new FancyActivationPlugin(FancyActivationPlugin::SELU, alpha, gamma),
          {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addActivation(
      inputs.at(0).tensor(), nvinfer1::ActivationType::kSIGMOID));
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  int axis = attrs.get("axis", 1);
  ASSERT(axis != 0, ErrorCode::kUNSUPPORTED_NODE); // Don't support axis = the batch dim
  int ndim = inputs.at(0).shape().nbDims;
  if( axis < 0 ) {
    axis += ndim; // Negative indexing
  } else {
    --axis; // Don't include the batch dim
  }
  ASSERT(0 <= axis && axis < ndim, ErrorCode::kINVALID_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  nvinfer1::Dims shape = tensor_ptr->getDimensions();
  ASSERT(tensor_ptr = flatten_tensor(ctx, *tensor_ptr, axis), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(tensor_ptr = move_tensor_dimension(ctx, *tensor_ptr, axis, 0), ErrorCode::kUNSUPPORTED_NODE);
  auto* layer = ctx->network()->addSoftMax(*tensor_ptr);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  tensor_ptr = layer->getOutput(0);
  ASSERT(tensor_ptr = move_tensor_dimension(ctx, *tensor_ptr, 0, axis), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(tensor_ptr = reshape_tensor(ctx, *tensor_ptr, shape), ErrorCode::kUNSUPPORTED_NODE);
  return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Softplus) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::SOFTPLUS),
                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Softsign) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::SOFTSIGN),
                     {&inputs.at(0).tensor()}));
}

// TODO: Legacy op for pre-1.0 ONNX spec; can be removed at some point
DEFINE_BUILTIN_OP_IMPORTER(SpatialBN) {
  return importBatchNormalization(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Split) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = inputs.at(0).shape();
  OnnxAttrs attrs(node);
  int axis = attrs.get<int>("axis");
  ASSERT(axis != 0, ErrorCode::kUNSUPPORTED_NODE);
  if( axis < 0 ) {
    axis += dims.nbDims;
    // HACK TODO: This is a (bad) WAR for the fact that the input dims may
    // have been padded to 4D and we no longer know how many dims there were
    // originally.
    axis = 0;
  } else {
    --axis; // Don't include batch dim
  }
  ASSERT(0 <= axis && axis < dims.nbDims, ErrorCode::kINVALID_NODE);
  std::vector<int> output_lengths;
  int noutput = node.output().size();
  if( attrs.count("split") ) {
    output_lengths = attrs.get<std::vector<int>>("split");
    ASSERT((int)output_lengths.size() == noutput, ErrorCode::kINVALID_NODE);
  } else {
    ASSERT(dims.d[axis] % noutput == 0, ErrorCode::kINVALID_NODE);
    output_lengths.assign(noutput, dims.d[axis] / noutput);
  }
  nvinfer1::IPluginLayer* layer =
      ctx->addPlugin(new SplitPlugin(axis, output_lengths),
                     {&inputs.at(0).tensor()});
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(layer->getNbOutputs() == noutput, ErrorCode::kINTERNAL_ERROR);
  std::vector<TensorOrWeights> outputs;
  for( int i=0; i<noutput; ++i ) {
    outputs.push_back(layer->getOutput(i));
  }
  return outputs;
}

DEFINE_BUILTIN_OP_IMPORTER(Sqrt) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kSQRT);
}

DEFINE_BUILTIN_OP_IMPORTER(Sub) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kSUB);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum) {
  return combineTensorsElementwise(
    ctx, inputs, nvinfer1::ElementWiseOperation::kSUM);
}

DEFINE_BUILTIN_OP_IMPORTER(Tanh) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addActivation(
      inputs.at(0).tensor(), nvinfer1::ActivationType::kTANH));
}

DEFINE_BUILTIN_OP_IMPORTER(ThresholdedRelu) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float alpha = attrs.get<float>("alpha", 1.f);
  RETURN_FIRST_OUTPUT(
      ctx->addPlugin(
         new FancyActivationPlugin(FancyActivationPlugin::THRESHOLDED_RELU, alpha),
         {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Transpose) {
  TensorOrWeights input = inputs.at(0);
  OnnxAttrs attrs(node);
  int ndim = input.shape().nbDims;
  ASSERT(ndim <= nvinfer1::Dims::MAX_DIMS, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Permutation default_perm; // Default is to reverse dims
  for( int i=0; i<ndim; ++i ) {
    default_perm.order[i] = ndim - 1 - i;
  }
  nvinfer1::Permutation perm = attrs.get("perm", default_perm);
  if( input.is_tensor() ) {
    // TRT doesn't support moving the batch dim
    ASSERT(perm.order[0] == 0, ErrorCode::kUNSUPPORTED_NODE);
    perm = remove_first_dim(perm);
    // Note: Dimension types kept unchanged in order to avoid TRT complaining about CHW order
    nvinfer1::ITensor* output_tensor =
        transpose_tensor(ctx, input.tensor(), perm, false);
    ASSERT(output_tensor, ErrorCode::kUNSUPPORTED_NODE);
    return {{output_tensor}};
  } else {
    auto weights = input.weights();
    auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
    ASSERT(transposeWeights(weights, perm, &new_weights),
           ErrorCode::kUNSUPPORTED_NODE);
    weights = new_weights;
    return {{weights}};
  }
}

} // namespace

} // namespace onnx2trt
