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

#include <numeric> // For std::iota

namespace onnx2trt {

namespace {

enum { BATCH_DIM = 0 };

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
  if( !(0 <= from && from < ndim) ) { return nullptr; }
  if( !(0 <= to   && to   < ndim) ) { return nullptr; }
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

NodeImportResult
addScale(IImporterContext*   ctx,
         nvinfer1::ITensor&  tensor_,
         nvinfer1::ScaleMode mode,
         nvinfer1::Weights   shift,
         nvinfer1::Weights   scale,
         nvinfer1::Weights   power) {
  nvinfer1::ITensor* tensor_ptr = &tensor_;
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  bool need_to_expand_dims = (dims.nbDims != 3);
  nvinfer1::Dims orig_shape = dims;
  if( need_to_expand_dims ) {
    // Expand or squash dims to 3D
    nvinfer1::Dims new_shape = dims;
    while( new_shape.nbDims < 3 ) {
      new_shape.d[new_shape.nbDims++] = 1;
    }
    while( new_shape.nbDims > 3 ) {
      new_shape.d[2] *= new_shape.d[--new_shape.nbDims];
    }
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
#endif // NV_TENSORRT_MAJOR >= 4

  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  // Fill in dtype for any unused (dummy) weights
  nvinfer1::DataType* dtype_ptr = nullptr;
  if( shift.count ) {
    dtype_ptr = &shift.type;
  }
  if( scale.count ) {
    ASSERT(!dtype_ptr || *dtype_ptr == scale.type,
           ErrorCode::kUNSUPPORTED_NODE);
    dtype_ptr = &scale.type;
  }
  if( power.count ) {
    ASSERT(!dtype_ptr || *dtype_ptr == power.type,
           ErrorCode::kUNSUPPORTED_NODE);
    dtype_ptr = &power.type;
  }
  ASSERT(dtype_ptr, ErrorCode::kINTERNAL_ERROR);
  shift.type = *dtype_ptr;
  scale.type = *dtype_ptr;
  power.type = *dtype_ptr;
  auto* layer = ctx->network()->addScale(
      *tensor_ptr, mode, shift, scale, power);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  tensor_ptr = layer->getOutput(0);

#if NV_TENSORRT_MAJOR >= 4
  if( need_to_expand_dims ) {
    // Un-expand spatial dims back to 1D
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, orig_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
}

// Explicit broadcasting for ONNX opset < 7
// This function adds extra dimensions to the end of rhs's shape in order to
// line up the dimensions based on the specified broadcasting axis.
Status applyLegacyBinaryOpBroadcasting(IImporterContext* ctx,
                                       ::ONNX_NAMESPACE::NodeProto const& node,
                                       TensorOrWeights& lhs,
                                       TensorOrWeights& rhs) {
  int lhs_ndim = lhs.shape().nbDims;
  int rhs_ndim = rhs.shape().nbDims;
  OnnxAttrs attrs(node);
  bool broadcasting_on = (attrs.count("axis") && attrs.count("broadcast") &&
                          attrs.get<int>("broadcast"));
  if (rhs_ndim >= lhs_ndim || !broadcasting_on) {
    return Status::success();
  }
  int axis = attrs.get<int>("axis");
  if( axis < 0 ) {
    axis += 1 + lhs_ndim; // Support negative indexing
  }
  // Note: axis=0 still means the batch dim here
  if( rhs.is_tensor() ) {
    // Batch dims of tensors must be aligned
    ASSERT(axis == BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
  } else { // rhs is weights
    if( axis == BATCH_DIM ) {
      // Weights must broadcast across the batch dim
      ASSERT(rhs.shape().d[0] == 1, ErrorCode::kUNSUPPORTED_NODE);
    }
    axis -= 1; // Shift batch dim to align with tensors
  }
  int num_dims_to_add = lhs_ndim - (axis + rhs_ndim);
  ASSERT(num_dims_to_add >= 0, ErrorCode::kINVALID_NODE);
  if (num_dims_to_add == 0) {
    return Status::success();
  }
  nvinfer1::Dims new_shape = rhs.shape();
  for (int i=0; i<num_dims_to_add; ++i) {
    new_shape.d[new_shape.nbDims++] = 1;
  }
  if (rhs.is_weights()) {
    rhs.weights().shape = new_shape;
  } else {
    ASSERT(rhs.reset_tensor(reshape_tensor(ctx, rhs.tensor(), new_shape)),
           ErrorCode::kUNSUPPORTED_NODE);
  }
  return Status::success();
}

NodeImportResult
combineTensorsElementwise(IImporterContext* ctx,
                          ::ONNX_NAMESPACE::NodeProto const& node,
                          std::vector<TensorOrWeights>& inputs,
                          nvinfer1::ElementWiseOperation binary_op,
                          bool legacy_binary_op_broadcasting=false) {
  ASSERT(!inputs.empty(), ErrorCode::kINVALID_NODE);
  if (ctx->getOpsetVersion() < 7 && legacy_binary_op_broadcasting) {
    ASSERT(inputs.size() == 2, ErrorCode::kINTERNAL_ERROR);
    TRT_CHECK(applyLegacyBinaryOpBroadcasting(ctx, node, inputs[0], inputs[1]));
  }
  std::vector<nvinfer1::ITensor*> input_tensors;
  int ndim_max = -1;
  int tensors_ndim_max = -1;
  for( auto input : inputs ) {
    ndim_max = std::max(ndim_max, input.shape().nbDims);
    // Note: Tensor dims always exclude the batch dim, but weights may not
    if( input.is_tensor() ) {
      tensors_ndim_max = std::max(tensors_ndim_max, input.shape().nbDims);
    }
  }
  for( auto input : inputs ) {
    nvinfer1::ITensor* tensor_ptr;
#if NV_TENSORRT_MAJOR < 4
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensor_ptr = &input.tensor();
#else
    if( input.is_weights() ) {
      auto weights = input.weights();
      // Note: TRT supports broadcasting, but ranks must match
      if( input.shape().nbDims < ndim_max ) {
        weights.shape = expand_dims(weights.shape, ndim_max);
      }
      if (weights.shape.nbDims == tensors_ndim_max + 1) {
        // The weights contain a batch dim, which must be removed
        // Note: TRT Constant layer has implicit batch dim of 1
        ASSERT(weights.shape.d[BATCH_DIM] == 1, ErrorCode::kUNSUPPORTED_NODE);
        weights.shape = remove_dim(weights.shape, BATCH_DIM);
      }
      auto* layer = ctx->network()->addConstant(weights.shape, weights);
      ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
      tensor_ptr = layer->getOutput(0);
    } else {
      tensor_ptr = &input.tensor();
      // Note: TRT supports broadcasting, but ranks must match
      // We can't expand the dims because the batch dim is always the left-most
      ASSERT(input.shape().nbDims == tensors_ndim_max,
             ErrorCode::kUNSUPPORTED_NODE);
    }
#endif
    input_tensors.push_back(tensor_ptr);
  }
  nvinfer1::ITensor* combined = input_tensors.at(0);
  if( input_tensors.size() == 1 ) {
    // Note: Single input must be wrapped in identity to avoid messing up network outputs
    return {{identity(ctx, combined)}};
  }
  for( size_t i=1; i<input_tensors.size(); ++i ) {
    nvinfer1::ITensor* tensor = input_tensors.at(i);
    ASSERT(tensor->getDimensions().nbDims == combined->getDimensions().nbDims,
           ErrorCode::kUNSUPPORTED_NODE);
    auto* layer = ctx->network()->addElementWise(
      *combined, *tensor, binary_op);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    combined = layer->getOutput(0);
  }
  return {{combined}};
}

// Note: As of TRT 4, ElementWise + Constant is preferred over Scale layer
#if NV_TENSORRT_MAJOR < 4
Status check_broadcast_attrs(IImporterContext* ctx, OnnxAttrs const& attrs,
                             nvinfer1::Dims const& dims) {
  if (ctx->getOpsetVersion() < 7) {
    ASSERT(attrs.count("broadcast"), ErrorCode::kUNSUPPORTED_NODE);
    bool broadcast = attrs.get<int>("broadcast");
    ASSERT(broadcast || dims.nbDims == 1, ErrorCode::kINVALID_NODE);
    int axis = attrs.get<int>("axis", -1);
    if( axis < 0 ) {
      axis += dims.nbDims; // Support negative indexing
    } else {
      ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
      axis -= 1; // Remove batch dim
    }
    ASSERT(axis == 0, ErrorCode::kUNSUPPORTED_NODE);
  }
  return Status::success();
}

enum ScaleOp {
  kSHIFT,
  kSCALE,
  kPOWER,
};

NodeImportResult importScaleOp(IImporterContext* ctx,
                               ::ONNX_NAMESPACE::NodeProto const& node,
                               TensorOrWeights& input0,
                               TensorOrWeights& input1,
                               ScaleOp op) {
  auto* tensor_ptr = (input0.is_tensor() ?
                      &input0.tensor() :
                      &input1.tensor());
  auto weights = (input0.is_weights() ?
                        input0.weights() :
                        input1.weights());
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  // Note: ONNX opset >= 7 uses Numpy-style broadcasting, so dims are padded
  // at the end with ones for broadcasting.
  weights.shape = squeeze_trailing_dims(weights.shape);
  nvinfer1::ScaleMode mode = get_scale_mode(weights.shape);
  if( mode == nvinfer1::ScaleMode::kELEMENTWISE ) {
    // TODO: TRT doesn't support including the batch dim in elementwise,
    //       but we can't do a more specific assertion here yet because
    //       the input tensor's shape may have been padded to WAR TRT's
    //       shape issues.
    ASSERT(get_shape_size(weights.shape) == get_shape_size(dims),
           ErrorCode::kUNSUPPORTED_NODE);
  } else if( mode == nvinfer1::ScaleMode::kCHANNEL ) {
    OnnxAttrs attrs(node);
    // TRT does not currently support full broadcasting
    TRT_CHECK(check_broadcast_attrs(ctx, attrs, dims));
    ASSERT(weights.shape.d[0] == dims.d[0],
           ErrorCode::kUNSUPPORTED_NODE);
  }
  nvinfer1::Weights shift_weights = {};
  nvinfer1::Weights scale_weights = {};
  nvinfer1::Weights power_weights = {};
  switch( op ) {
  case kSHIFT: shift_weights = weights; break;
  case kSCALE: scale_weights = weights; break;
  case kPOWER: power_weights = weights; break;
  }
  return addScale(
      ctx, *tensor_ptr, mode, shift_weights, scale_weights, power_weights);
}
#endif // NV_TENSORRT_MAJOR < 4

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

#define IGNORE_UNUSED_GLOBAL(x) \
  static void _ignore_unused2_##x(); \
  static void _ignore_unused1_##x() { (void)_ignore_unused2_##x; (void)x; } \
  static void _ignore_unused2_##x() { (void)_ignore_unused1_##x; } \
  struct SwallowSemicolon##x {}

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
  IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op); \
  NodeImportResult import##op(IImporterContext* ctx, \
                              ::ONNX_NAMESPACE::NodeProto const& node, \
                              std::vector<TensorOrWeights>& inputs)

#define RETURN_FIRST_OUTPUT(layer) do { \
  nvinfer1::ILayer* layer_ptr = layer; \
  ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE); \
  return {{layer_ptr->getOutput(0)}}; \
} while(0)

#define RETURN_IDENTITY(input) do { \
  TensorOrWeights output = identity(ctx, input); \
  ASSERT(output, ErrorCode::kUNSUPPORTED_NODE); \
  return {{output}}; \
} while(0)

DEFINE_BUILTIN_OP_IMPORTER(Abs) {
  return apply_unary_function(ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
}

DEFINE_BUILTIN_OP_IMPORTER(Add) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  // Note: As of TRT 4, ElementWise + Constant is preferred over Scale
#if NV_TENSORRT_MAJOR < 4
  if (inputs.at(0).is_tensor() != inputs.at(1).is_tensor()) {
    // One input is a tensor, the other is weights
    return importScaleOp(
        ctx, node, inputs.at(0), inputs.at(1), ScaleOp::kSHIFT);
  }
#endif
  return combineTensorsElementwise(
      ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM, true);
}

DEFINE_BUILTIN_OP_IMPORTER(AveragePool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  bool need_to_expand_dims = (dims.nbDims == 2);
  if( need_to_expand_dims ) {
    // Expand spatial dims from 1D to 2D
    nvinfer1::DimsCHW new_shape(dims.d[0], dims.d[1], 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::DimsHW kernel_size(1, 1), strides(1, 1), beg_padding(0, 0), end_padding(0, 0);
  get_kernel_params(node, get_DimsHW_from_CHW(dims),
                    &kernel_size, &strides, &beg_padding, &end_padding);
  nvinfer1::IPoolingLayer* pooling_layer = ctx->network()->addPooling(
    *tensor_ptr, nvinfer1::PoolingType::kAVERAGE, kernel_size);
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
      bool supported_form_of_asymmetric_padding_for_AveragePool = false;
      ASSERT(supported_form_of_asymmetric_padding_for_AveragePool,
             ErrorCode::kUNSUPPORTED_NODE);
    }
  }
  pooling_layer->setPadding(beg_padding);
  if( pre_crop  != nvinfer1::DimsHW(0, 0) ||
      post_crop != nvinfer1::DimsHW(0, 0) ) {
    layer = ctx->network()->addPadding(*pooling_layer->getOutput(0),
                                       -pre_crop, -post_crop);
  }
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  if( need_to_expand_dims ) {
    // Un-expand spatial dims back to 1D
    nvinfer1::Dims new_shape{2, {dims.d[0], dims.d[1]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
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
  return addScale(ctx, tensor, nvinfer1::ScaleMode::kCHANNEL,
                  combined_bias_weights, combined_scale_weights, {});
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
#if NV_TENSORRT_MAJOR >= 4
    ASSERT(input.tensor().getType() != nvinfer1::DataType::kINT32,
           ErrorCode::kUNSUPPORTED_NODE);
#endif // NV_TENSORRT_MAJOR >= 4
    tensors.push_back(&input.tensor());
  }
  OnnxAttrs attrs(node);
  int axis = attrs.get<int>("axis");
  if( axis < 0 ) {
    // Support negative indexing
    axis += inputs.at(0).shape().nbDims;
  } else {
    ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
    axis -= 1; // Remove batch dim
  }
#if NV_TENSORRT_MAJOR >= 4
  auto* layer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  layer->setAxis(axis);
  RETURN_FIRST_OUTPUT(layer);
#else // NV_TENSORRT_MAJOR < 4
  if( axis == 0 ) {
    RETURN_FIRST_OUTPUT(
      ctx->network()->addConcatenation(tensors.data(), tensors.size()));
  } else {
    ASSERT(inputs.at(0).shape().nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
    using namespace nvinfer1::plugin;
    RETURN_FIRST_OUTPUT(
        ctx->addPlugin(
            new NvPlugin(createConcatPlugin(1 + axis, false)), tensors));
  }
#endif // NV_TENSORRT_MAJOR < 4
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
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  bool need_to_expand_dims = (dims.nbDims == 2);
  if( need_to_expand_dims ) {
    // Expand spatial dims from 1D to 2D
    nvinfer1::DimsCHW new_shape(dims.d[0], dims.d[1], 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
  if( kernel_weights.shape.nbDims == 3 ) {
    kernel_weights.shape.nbDims = 4;
    kernel_weights.shape.d[3] = 1;
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
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
  nvinfer1::DimsHW strides(1, 1);
  nvinfer1::DimsHW beg_padding(0, 0), end_padding(0, 0);
  nvinfer1::DimsHW dilations(1, 1);
  get_kernel_params(node, get_DimsHW_from_CHW(dims),
                    &kernel_size, &strides,
                    &beg_padding, &end_padding, &dilations);
  ASSERT(kernel_size.h() == kernel_weights.shape.d[2], ErrorCode::kINVALID_NODE);
  ASSERT(kernel_size.w() == kernel_weights.shape.d[3], ErrorCode::kINVALID_NODE);
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
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  if( need_to_expand_dims ) {
    // Un-expand spatial dims back to 1D
    nvinfer1::Dims new_shape{2, {dims.d[0], dims.d[1]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(ConvTranspose) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  auto kernel_weights = inputs.at(1).weights();
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  bool need_to_expand_dims = (dims.nbDims == 2);
  if( need_to_expand_dims ) {
    // Expand spatial dims from 1D to 2D
    nvinfer1::DimsCHW new_shape(dims.d[0], dims.d[1], 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
  if( kernel_weights.shape.nbDims == 3 ) {
    kernel_weights.shape.nbDims = 4;
    kernel_weights.shape.d[3] = 1;
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(kernel_weights.shape.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Weights bias_weights;
  if( inputs.size() == 3 ) {
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    auto shaped_bias_weights = inputs.at(2).weights();
    ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
    ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[1],
           ErrorCode::kINVALID_NODE);
    bias_weights = shaped_bias_weights;
  } else {
    bias_weights = ShapedWeights::empty(kernel_weights.type);
  }
  OnnxAttrs attrs(node);
  nvinfer1::DimsHW input_shape = get_DimsHW_from_CHW(dims);
  nvinfer1::DimsHW output_shape;
  bool enable_padding_trick = true;
  if( attrs.count("output_shape") ) {
    output_shape = attrs.get<nvinfer1::DimsHW>("output_shape");
  } else {
    ASSERT(attrs.get("auto_pad", std::string("VALID")) == "VALID",
           ErrorCode::kINVALID_NODE);
    // Can't use asym->sym padding trick if no output shape provided
    enable_padding_trick = false;
  }
  nvinfer1::DimsHW kernel_size;
  kernel_size.h() = kernel_weights.shape.d[2];
  kernel_size.w() = kernel_weights.shape.d[3];
  nvinfer1::DimsHW strides(1, 1);
  nvinfer1::DimsHW beg_padding(0, 0), end_padding(0, 0);
  nvinfer1::DimsHW dilations(1, 1);
  // Note: output_shape/input_shape are swapped here so that the padding
  //       calculations operate as if it were a regular forward convolution.
  get_kernel_params(node, output_shape,
                    &kernel_size, &strides,
                    &beg_padding, &end_padding, &dilations, &input_shape,
                    enable_padding_trick);
  ASSERT(kernel_size.h() == kernel_weights.shape.d[2], ErrorCode::kINVALID_NODE);
  ASSERT(kernel_size.w() == kernel_weights.shape.d[3], ErrorCode::kINVALID_NODE);
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int nchan = dims.d[0];
  int ngroup = attrs.get("group", 1);
  int noutput = kernel_weights.shape.d[1] * ngroup; // Note: Weights order is CKRS
  nvinfer1::IDeconvolutionLayer* deconv_layer = ctx->network()->addDeconvolution(
    *tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);
  nvinfer1::ILayer* layer = deconv_layer;
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  deconv_layer->setStride(strides);
  if( !attrs.count("output_shape") && attrs.count("output_padding") ) {
    auto output_padding = attrs.get<nvinfer1::DimsHW>("output_padding");
    end_padding.h() -= output_padding.h();
    end_padding.w() -= output_padding.w();
  }
  if( beg_padding == end_padding ) {
    deconv_layer->setPadding(beg_padding);
  } else {
    layer = ctx->network()->addPadding(*deconv_layer->getOutput(0),
                                       -beg_padding, -end_padding);
  }
  ASSERT(dilations.h() == 1 && dilations.w() == 1, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(kernel_weights.shape.d[0] == nchan, ErrorCode::kINVALID_NODE);
  deconv_layer->setNbGroups(ngroup);
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  if( need_to_expand_dims ) {
    // Un-expand spatial dims back to 1D
    nvinfer1::Dims new_shape{2, {dims.d[0], dims.d[1]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
}

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(DepthToSpace) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  int block_size = attrs.get<int>("blocksize");
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  int ndim_spatial = dims.nbDims - 1;
  nvinfer1::Dims new_shape1;
  new_shape1.nbDims = dims.nbDims + ndim_spatial;
  new_shape1.d[ndim_spatial] = dims.d[0];
  for( int i=0; i<ndim_spatial; ++i ) {
    ASSERT(new_shape1.d[ndim_spatial] % block_size == 0, ErrorCode::kINVALID_NODE);
    new_shape1.d[ndim_spatial] /= block_size;
    new_shape1.d[i] = block_size;
    new_shape1.d[ndim_spatial + 1 + i] = dims.d[1 + i];
  }
  layer->setReshapeDimensions(new_shape1);
  nvinfer1::Permutation perm;
  perm.order[0] = ndim_spatial;
  for( int i=0; i<ndim_spatial; ++i ) {
    perm.order[1 + 2*i + 0] = ndim_spatial + 1 + i;
    perm.order[1 + 2*i + 1] = i;
  }
  layer->setSecondTranspose(perm);
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
  nvinfer1::Dims new_shape2;
  new_shape2.nbDims = dims.nbDims - ndim_spatial;
  new_shape2.d[0] = dims.d[0];
  for( int i=0; i<ndim_spatial; ++i ) {
    new_shape2.d[1 + i] = dims.d[1 + 2*i + 0] * dims.d[1 + 2*i + 1];
  }
  tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape2);
  ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  return {{tensor_ptr}};
}
#endif // NV_TENSORRT_MAJOR >= 4

DECLARE_BUILTIN_OP_IMPORTER(Mul);
DEFINE_BUILTIN_OP_IMPORTER(Div) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  // Note: As of TRT 4, ElementWise + Constant is preferred over Scale
#if NV_TENSORRT_MAJOR < 4
  if( inputs.at(0).is_tensor() && inputs.at(1).is_weights() ) {
    auto weights = inputs.at(1).weights();
    auto inv_weights = ctx->createTempWeights(weights.type, weights.shape);
    auto status = apply_unary_function(
        weights, &inv_weights, nvinfer1::UnaryOperation::kRECIP);
    if (status.is_error()) {
      return status;
    }
    std::vector<TensorOrWeights> new_inputs = {inputs.at(0), inv_weights};
    return importMul(ctx, node, new_inputs);
  }
#endif
  return combineTensorsElementwise(
      ctx, node, inputs, nvinfer1::ElementWiseOperation::kDIV, true);
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
  OnnxAttrs attrs(node);
  int axis = attrs.get("axis", 1);
  // Note: Flattening to shape=[batch, n] is currently the only sensible
  //       operation, because we can't remove or merge into the batch dim.
  ASSERT(axis == 1, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = inputs.at(0).shape();
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr;
#if NV_TENSORRT_MAJOR < 4
  // Note: TRT3 requires that the shape remain 3D (CHW)
  tensor_ptr = flatten_tensor(ctx, inputs.at(0).tensor());
#else // NV_TENSORRT_MAJOR >= 4
  nvinfer1::Dims new_shape{1, {(int)get_shape_size(dims)}};
  tensor_ptr = reshape_tensor(ctx, inputs.at(0).tensor(), new_shape);
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Floor) {
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(ctx->addPlugin(new FancyActivationPlugin(FancyActivationPlugin::FLOOR),
                                     {&inputs.at(0).tensor()}));
}

DEFINE_BUILTIN_OP_IMPORTER(Gemm) {
  // Note: Currently this only supports A=tensor, B=weights, C=biases
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor *tensor_ptr = &inputs.at(0).tensor();
  auto weights = inputs.at(1).weights();
  auto biases = inputs.at(2).weights();
  OnnxAttrs attrs(node);
  float alpha = attrs.get("alpha", 1.f);
  float beta = attrs.get("beta", 1.f);
  bool trans_a = attrs.get("transA", false);
  bool trans_b = attrs.get("transB", false);
  if (ctx->getOpsetVersion() < 7) {
    ASSERT(attrs.get("broadcast", false), ErrorCode::kUNSUPPORTED_NODE);
  }
  ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  // Note: TRT requires 3D input for FC layers, so we expand the dims
  bool need_to_expand_dims = (dims.nbDims == 1);
  if (need_to_expand_dims) {
    nvinfer1::DimsCHW new_shape(dims.d[0], 1, 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  int ninput = dims.d[0] * dims.d[1] * dims.d[2];
  ASSERT(!trans_a, ErrorCode::kUNSUPPORTED_NODE);
  if (!trans_b) {
    auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
    ASSERT(transposeWeights(weights, {1, 0}, &new_weights),
           ErrorCode::kUNSUPPORTED_NODE);
    weights = new_weights;
  }
  ASSERT(weights.shape.d[1] == ninput, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(alpha == 1.f, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(beta == 1.f, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(biases.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
  int nrow = biases.shape.d[0];
  ASSERT(weights.shape.d[0] == nrow, ErrorCode::kINVALID_NODE);
  auto *layer =
      ctx->network()->addFullyConnected(*tensor_ptr, nrow, weights, biases);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  // Un-expand the dims back to the original shape
  if (need_to_expand_dims) {
    nvinfer1::Dims new_shape{1, {dims.d[0]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
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

// TODO: GlobalLpPool: pow(reduce_mean(pow(abs(x), p)), 1./p)

DEFINE_BUILTIN_OP_IMPORTER(GlobalMaxPool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  nvinfer1::Dims dims = tensor.getDimensions();
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::DimsHW kernel_size(dims.d[1], dims.d[2]);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addPooling(
      tensor, nvinfer1::PoolingType::kMAX, kernel_size));
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

DEFINE_BUILTIN_OP_IMPORTER(Identity) {
  RETURN_IDENTITY(inputs.at(0));
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
#if NV_TENSORRT_MAJOR >= 4
  if( inputs.at(0).is_tensor() && inputs.at(1).is_tensor() &&
      inputs.at(0).shape().nbDims >= 2 && inputs.at(1).shape().nbDims >= 2) {
    nvinfer1::ITensor& tensor_a = inputs.at(0).tensor();
    nvinfer1::ITensor& tensor_b = inputs.at(1).tensor();
    RETURN_FIRST_OUTPUT(
        ctx->network()->addMatrixMultiply(tensor_a, false, tensor_b, false));
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(inputs.at(0).is_tensor(),  ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  // Note: Currently this only supports A=tensor, B=weights
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  auto weights = inputs.at(1).weights();
  auto biases  = ShapedWeights::empty(weights.type);
  OnnxAttrs attrs(node);
  ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  // Note: TRT requires 3D input for FC layers, so we expand the dims
  bool need_to_expand_dims = (dims.nbDims == 1);
  if( need_to_expand_dims ) {
    nvinfer1::DimsCHW new_shape(dims.d[0], 1, 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  //if( !trans_b ) {
  {
    auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
    ASSERT(transposeWeights(weights, {1, 0}, &new_weights),
           ErrorCode::kUNSUPPORTED_NODE);
    weights = new_weights;
  }
  int ninput = dims.d[0] * dims.d[1] * dims.d[2];
  ASSERT(weights.shape.d[1] == ninput, ErrorCode::kINVALID_NODE);
  int nrow = weights.shape.d[0];
  auto* layer = ctx->network()->addFullyConnected(
      *tensor_ptr, nrow, weights, biases);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  // Un-expand the dims back to the original shape
  if( need_to_expand_dims ) {
    nvinfer1::Dims new_shape{1, {dims.d[0]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
}

DEFINE_BUILTIN_OP_IMPORTER(Max) {
  return combineTensorsElementwise(
    ctx, node, inputs, nvinfer1::ElementWiseOperation::kMAX);
}

DEFINE_BUILTIN_OP_IMPORTER(MaxPool) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  ASSERT(dims.nbDims >= 2, ErrorCode::kINVALID_NODE);
#if NV_TENSORRT_MAJOR >= 4
  bool need_to_expand_dims = (dims.nbDims == 2);
  if( need_to_expand_dims ) {
    // Expand spatial dims from 1D to 2D
    nvinfer1::DimsCHW new_shape(dims.d[0], dims.d[1], 1);
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    dims = tensor_ptr->getDimensions();
  }
#endif // NV_TENSORRT_MAJOR >= 4
  ASSERT(dims.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::DimsHW kernel_size(1, 1), strides(1, 1), beg_padding(0, 0), end_padding(0, 0);
  get_kernel_params(node, get_DimsHW_from_CHW(dims),
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
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
#if NV_TENSORRT_MAJOR >= 4
  if( need_to_expand_dims ) {
    // Un-expand spatial dims back to 1D
    nvinfer1::Dims new_shape{2, {dims.d[0], dims.d[1]}};
    tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
    ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  }
#endif // NV_TENSORRT_MAJOR >= 4
  return {{tensor_ptr}};
}

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(Mean) {
  auto sum_result = combineTensorsElementwise(
    ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
  if( sum_result.is_error() ) {
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
  auto* constant_layer = ctx->network()->addConstant(
      scale_weights.shape, scale_weights);
  ASSERT(constant_layer, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& scale_constant = *constant_layer->getOutput(0);
  RETURN_FIRST_OUTPUT(
      ctx->network()->addElementWise(
          sum_tensor, scale_constant, nvinfer1::ElementWiseOperation::kPROD));
}
#endif // NV_TENSORRT_MAJOR >= 4

DEFINE_BUILTIN_OP_IMPORTER(Min) {
  return combineTensorsElementwise(
    ctx, node, inputs, nvinfer1::ElementWiseOperation::kMIN);
}

DEFINE_BUILTIN_OP_IMPORTER(Mul) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  // Note: As of TRT 4, ElementWise + Constant is preferred over Scale
#if NV_TENSORRT_MAJOR < 4
  if (inputs.at(0).is_tensor() != inputs.at(1).is_tensor()) {
    // One input is a tensor, the other is weights
    return importScaleOp(
        ctx, node, inputs.at(0), inputs.at(1), ScaleOp::kSCALE);
  }
#endif
  return combineTensorsElementwise(
      ctx, node, inputs, nvinfer1::ElementWiseOperation::kPROD, true);
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
  // Note: As of TRT 4, ElementWise + Constant is preferred over Scale
#if NV_TENSORRT_MAJOR < 4
  if (inputs.at(0).is_tensor() && inputs.at(1).is_weights()) {
    return importScaleOp(
        ctx, node, inputs.at(0), inputs.at(1), ScaleOp::kPOWER);
  }
#endif
  return combineTensorsElementwise(
      ctx, node, inputs, nvinfer1::ElementWiseOperation::kPOW, true);
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

#if NV_TENSORRT_MAJOR >= 4
NodeImportResult reduceTensor(IImporterContext* ctx,
                              ::ONNX_NAMESPACE::NodeProto const& node,
                              TensorOrWeights input,
                              nvinfer1::ReduceOperation operation) {
  ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = input.tensor();
  OnnxAttrs attrs(node);
  bool keepdims = attrs.get("keepdims", 1);
  int ndim = tensor.getDimensions().nbDims;
  std::vector<int> axes;
  if( attrs.count("axes") ) {
    axes = attrs.get<std::vector<int>>("axes");
  } else {
    axes.resize(ndim);
    std::iota(axes.begin(), axes.end(), 0);
  }
  uint32_t axis_mask = 0;
  for( int axis : axes ) {
    ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
    if( axis < 0 ) {
      axis += ndim; // Support negative indexing
    } else {
      --axis; // Don't include batch dim
    }
    axis_mask |= 1 << axis;
  }
  RETURN_FIRST_OUTPUT(
      ctx->network()->addReduce(tensor, operation, axis_mask, keepdims));
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceL1) {
  NodeImportResult abs_result = apply_unary_function(
      ctx, inputs.at(0), nvinfer1::UnaryOperation::kABS);
  if( abs_result.is_error() ) {
    return abs_result;
  }
  TensorOrWeights abs_input = abs_result.value().at(0);
  return reduceTensor(ctx, node, abs_input, nvinfer1::ReduceOperation::kSUM);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSum);
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSum) {
  auto sum_result = importReduceSum(ctx, node, inputs);
  if( sum_result.is_error() ) {
    return sum_result;
  }
  TensorOrWeights sum_input = sum_result.value().at(0);
  return apply_unary_function(ctx, sum_input, nvinfer1::UnaryOperation::kLOG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceLogSumExp) {
  // TODO: Abstract this sequence with a function or macro
  auto exp_result = apply_unary_function(
      ctx, inputs.at(0), nvinfer1::UnaryOperation::kEXP);
  if( exp_result.is_error() ) {
    return exp_result;
  }
  auto exp_inputs = exp_result.value();
  return importReduceLogSum(ctx, node, exp_inputs);
}
DECLARE_BUILTIN_OP_IMPORTER(ReduceSumSquare);
DEFINE_BUILTIN_OP_IMPORTER(ReduceL2) {
  auto sum_sqr_result = importReduceSumSquare(ctx, node, inputs);
  if( sum_sqr_result.is_error() ) {
    return sum_sqr_result;
  }
  TensorOrWeights sum_sqr = sum_sqr_result.value().at(0);
  return apply_unary_function(ctx, sum_sqr, nvinfer1::UnaryOperation::kSQRT);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMax) {
  return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMAX);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMean) {
  return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kAVG);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceMin) {
  return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kMIN);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceProd) {
  return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kPROD);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSum) {
  return reduceTensor(ctx, node, inputs.at(0), nvinfer1::ReduceOperation::kSUM);
}
DEFINE_BUILTIN_OP_IMPORTER(ReduceSumSquare) {
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  auto* sqr_layer = ctx->network()->addElementWise(
      tensor, tensor, nvinfer1::ElementWiseOperation::kPROD);
  ASSERT(sqr_layer, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* sqr_tensor_ptr = sqr_layer->getOutput(0);
  return reduceTensor(
      ctx, node, sqr_tensor_ptr, nvinfer1::ReduceOperation::kSUM);
}

#endif // NV_TENSORRT_MAJOR >= 4

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
    ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
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
    nvinfer1::ITensor& tensor = input.tensor();
    new_shape = set_dims_CHW(remove_dim(new_shape, BATCH_DIM));
    int infer_dim = -1;
    for (int i = 0; i < new_shape.nbDims; ++i) {
      if (new_shape.d[i] < 0) {
        // -1 bears special meaning, which means the current dimension can
        // be inferred while keepin the total number of elements the same.
        // https://github.com/onnx/onnx/blob/9b9f595107e3fc0295d50f6294d43879df17552f/onnx/defs/tensor/defs.cc#L73-L75
        ASSERT(new_shape.d[i] == -1, ErrorCode::kUNSUPPORTED_NODE);
        // We can only one dimension that has -1
        ASSERT(infer_dim == -1, ErrorCode::kUNSUPPORTED_NODE);
        infer_dim = i;
      }
    }
    if (infer_dim < 0) {
      ASSERT(get_shape_size(new_shape) ==
                 get_shape_size(tensor.getDimensions()),
             ErrorCode::kUNSUPPORTED_NODE);
    }
#if NV_TENSORRT_MAJOR < 4
    if( new_shape.nbDims == 1 ) {
      // Note: TRT implicitly flattens the input to FC layers, and in fact
      //       requires that it still has 4D shape, so in this case we
      //       simply ignore the reshape.
      RETURN_IDENTITY(inputs.at(0));
    } else {
      ASSERT(new_shape.nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
      ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
      layer->setReshapeDimensions(new_shape);
      ASSERT(get_shape_size(layer->getOutput(0)->getDimensions()) ==
             get_shape_size(input.shape()), ErrorCode::kUNSUPPORTED_NODE);
      RETURN_FIRST_OUTPUT(layer);
    }
#else // NV_TENSORRT_MAJOR >= 4
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    layer->setReshapeDimensions(new_shape);
    ASSERT(get_shape_size(layer->getOutput(0)->getDimensions()) ==
           get_shape_size(input.shape()), ErrorCode::kUNSUPPORTED_NODE);
    RETURN_FIRST_OUTPUT(layer);
#endif // NV_TENSORRT_MAJOR >= 4
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

DEFINE_BUILTIN_OP_IMPORTER(Shape) {
  auto shape = inputs.at(0).shape();
  if( inputs.at(0).is_tensor() ) {
    shape = insert_dim(shape, BATCH_DIM, -1);
  }
  nvinfer1::Dims weight_dims;
  weight_dims.nbDims = 1;
  weight_dims.d[0] = shape.nbDims;
  // Note: Should technically be int64, but int32 allows for TRT compatibility
  auto weights = ctx->createTempWeights(
      ::ONNX_NAMESPACE::TensorProto::INT32, weight_dims);
  std::copy(&shape.d[0], &shape.d[0] + shape.nbDims,
            static_cast<int32_t*>(const_cast<void*>(weights.values)));
  return {{weights}};
}

DEFINE_BUILTIN_OP_IMPORTER(Sigmoid) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(
    ctx->network()->addActivation(
      inputs.at(0).tensor(), nvinfer1::ActivationType::kSIGMOID));
}

DEFINE_BUILTIN_OP_IMPORTER(Size) {
  // Can't support tensors because we don't know the batch dim until runtime
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  auto shape = inputs.at(0).shape();
  nvinfer1::Dims weight_dims;
  weight_dims.nbDims = 1;
  weight_dims.d[0] = 1;
  // Note: Should technically be int64, but int32 allows for TRT compatibility
  auto weights = ctx->createTempWeights(
      ::ONNX_NAMESPACE::TensorProto::INT32, weight_dims);
  int32_t size = get_shape_size(shape);
  *static_cast<int32_t*>(const_cast<void*>(weights.values)) = size;
  return {{weights}};
}

DEFINE_BUILTIN_OP_IMPORTER(Softmax) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  int axis = attrs.get("axis", 1);
  ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
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

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(SpaceToDepth) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor* tensor_ptr = &inputs.at(0).tensor();
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(*tensor_ptr);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  int block_size = attrs.get<int>("blocksize");
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  int ndim_spatial = dims.nbDims - 1;
  nvinfer1::Dims new_shape1;
  new_shape1.nbDims = dims.nbDims + ndim_spatial;
  new_shape1.d[0] = dims.d[0];
  for( int i=0; i<ndim_spatial; ++i ) {
    ASSERT(dims.d[1 + i] % block_size == 0, ErrorCode::kINVALID_NODE);
    new_shape1.d[1 + 2*i + 0] = dims.d[1 + i] / block_size;
    new_shape1.d[1 + 2*i + 1] = block_size;
  }
  layer->setReshapeDimensions(new_shape1);
  nvinfer1::Permutation perm;
  perm.order[ndim_spatial] = 0;
  for( int i=0; i<ndim_spatial; ++i ) {
    perm.order[ndim_spatial + 1 + i] = 1 + 2*i + 0;
    perm.order[i] = 1 + 2*i + 1;
  }
  layer->setSecondTranspose(perm);
  tensor_ptr = layer->getOutput(0);
  dims = tensor_ptr->getDimensions();
  nvinfer1::Dims new_shape2;
  new_shape2.nbDims = dims.nbDims - ndim_spatial;
  new_shape2.d[0] = dims.d[ndim_spatial];
  for( int i=0; i<ndim_spatial; ++i ) {
    new_shape2.d[0] *= dims.d[i];
    new_shape2.d[1 + i] = dims.d[ndim_spatial + 1 + i];
  }
  tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape2);
  ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
  dims = tensor_ptr->getDimensions();
  return {{tensor_ptr}};
}
#endif // NV_TENSORRT_MAJOR >= 4

// TODO: Legacy op for pre-1.0 ONNX spec; can be removed at some point
DEFINE_BUILTIN_OP_IMPORTER(SpatialBN) {
  return importBatchNormalization(ctx, node, inputs);
}

DEFINE_BUILTIN_OP_IMPORTER(Split) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(inputs.size() == 1, ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims dims = inputs.at(0).shape();
  OnnxAttrs attrs(node);
  int axis = attrs.get<int>("axis", 0);
  ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
  if( axis < 0 ) {
    axis += dims.nbDims;
#if NV_TENSORRT_MAJOR < 4
    // HACK TODO: This is a (bad) WAR for the fact that the input dims may
    // have been padded to 4D and we no longer know how many dims there were
    // originally.
    axis = 0;
#endif // NV_TENSORRT_MAJOR < 4
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

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(Squeeze) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  nvinfer1::Dims old_shape = tensor.getDimensions();
  int ndim_in = old_shape.nbDims;
  OnnxAttrs attrs(node);
  auto axes = attrs.get<std::vector<int>>("axes");
  // Note: Can't handle batch dim as it is implicit in TRT
  for( auto& axis : axes ) {
    ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
    --axis;
  }
  std::set<int> axes_set(axes.begin(), axes.end());
  int ndim_out = ndim_in - axes_set.size();
  ASSERT(ndim_out <= nvinfer1::Dims::MAX_DIMS,
         ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims new_shape;
  new_shape.nbDims = ndim_out;
  for( int i=0,j=0; i<old_shape.nbDims; ++i ) {
    if( !axes_set.count(i) ) {
      new_shape.d[j++] = old_shape.d[i];
    } else {
      ASSERT(old_shape.d[i] == 1, ErrorCode::kINVALID_NODE);
    }
  }
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  layer->setReshapeDimensions(new_shape);
  ASSERT(get_shape_size(layer->getOutput(0)->getDimensions()) ==
         get_shape_size(old_shape), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(layer);
}
#endif // NV_TENSORRT_MAJOR >= 4

DECLARE_BUILTIN_OP_IMPORTER(Add);
DEFINE_BUILTIN_OP_IMPORTER(Sub) {
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  // Note: As of TRT 4, ElementWise + Constant is preferred over Scale
#if NV_TENSORRT_MAJOR < 4
  if( inputs.at(0).is_tensor() && inputs.at(1).is_weights() ) {
    auto weights = inputs.at(1).weights();
    auto neg_weights = ctx->createTempWeights(weights.type, weights.shape);
    auto status = apply_unary_function(
        weights, &neg_weights, nvinfer1::UnaryOperation::kNEG);
    if (status.is_error()) {
      return status;
    }
    std::vector<TensorOrWeights> new_inputs = {inputs.at(0), neg_weights};
    return importAdd(ctx, node, new_inputs);
  }
#endif
  return combineTensorsElementwise(
      ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUB, true);
}

DEFINE_BUILTIN_OP_IMPORTER(Sum) {
  return combineTensorsElementwise(
    ctx, node, inputs, nvinfer1::ElementWiseOperation::kSUM);
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

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(TopK) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  ASSERT(tensor.getType() != nvinfer1::DataType::kINT32,
         ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  ASSERT(attrs.count("k"), ErrorCode::kINVALID_NODE);
  int k    = attrs.get<int>("k");
  int axis = attrs.get("axis", -1);
  ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
  if( axis < 0 ) {
    nvinfer1::Dims dims = tensor.getDimensions();
    axis += dims.nbDims;
  } else {
    --axis; // Don't include batch dim
  }
  uint32_t axis_mask = 1 << axis;
  auto* layer = ctx->network()->addTopK(
      tensor, nvinfer1::TopKOperation::kMAX, k, axis_mask);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  return {{layer->getOutput(0), layer->getOutput(1)}};
}
#endif // NV_TENSORRT_MAJOR >= 4

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
    ASSERT(perm.order[BATCH_DIM] == BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
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

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(Unsqueeze) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  nvinfer1::Dims old_shape = tensor.getDimensions();
  int ndim_in = old_shape.nbDims;
  OnnxAttrs attrs(node);
  auto axes = attrs.get<std::vector<int>>("axes");
  // Note: Can't handle batch dim as it is implicit in TRT
  for( auto& axis : axes ) {
    ASSERT(axis != BATCH_DIM, ErrorCode::kUNSUPPORTED_NODE);
    --axis;
  }
  std::set<int> axes_set(axes.begin(), axes.end());
  int ndim_out = ndim_in + axes_set.size();
  ASSERT(ndim_out <= nvinfer1::Dims::MAX_DIMS,
         ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::Dims new_shape;
  new_shape.nbDims = ndim_out;
  for( int i=0,j=0; j<new_shape.nbDims; ++j ) {
    if( !axes_set.count(j) ) {
      new_shape.d[j] = old_shape.d[i++];
    } else {
      new_shape.d[j] = 1;
    }
  }
  nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
  ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
  layer->setReshapeDimensions(new_shape);
  ASSERT(get_shape_size(layer->getOutput(0)->getDimensions()) ==
         get_shape_size(old_shape), ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(layer);
}
#endif // NV_TENSORRT_MAJOR >= 4

DEFINE_BUILTIN_OP_IMPORTER(Upsample) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  ASSERT(tensor.getDimensions().nbDims == 3, ErrorCode::kUNSUPPORTED_NODE);
  OnnxAttrs attrs(node);
  float height_scale, width_scale;
  if( !attrs.count("scales") ) {
    height_scale = attrs.get<float>("height_scale");
    width_scale  = attrs.get<float>("width_scale");
  } else {
    auto scales = attrs.get<std::vector<float>>("scales");
    ASSERT(scales.size() == 4, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales[0] == 1 && scales[1] == 1, ErrorCode::kUNSUPPORTED_NODE);
    height_scale = scales[2];
    width_scale  = scales[3];
  }
  auto scale = {height_scale, width_scale};
  auto mode = attrs.get<std::string>("mode", "nearest");
  ASSERT(mode == "nearest", ErrorCode::kUNSUPPORTED_NODE);
  RETURN_FIRST_OUTPUT(ctx->addPlugin(new ResizeNearestPlugin(scale),
                                     {&inputs.at(0).tensor()}));
}

} // namespace

} // namespace onnx2trt
