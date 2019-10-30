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

#include "onnx2trt_utils.hpp"

std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape) {
  if( shape.nbDims == 0 ) {
    return stream;
  }
  stream << "(" << shape.d[0];
  for( int i=1; i<shape.nbDims; ++i ) {
    stream << ", " << shape.d[i];
  }
  stream << ")";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype) {
  switch( dtype ) {
  case nvinfer1::DataType::kFLOAT: return stream << "float32";
  case nvinfer1::DataType::kHALF:  return stream << "float16";
  case nvinfer1::DataType::kINT8:  return stream << "int8";
  case nvinfer1::DataType::kINT32: return stream << "int32";
  default: throw std::runtime_error("Unknown dtype");
  }
}

std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm) {
  int ndims = nvinfer1::Dims::MAX_DIMS;
  stream << "(" << perm.order[0];
  for( int i=1; i<ndims; ++i ) {
    stream << ", " << perm.order[i];
  }
  stream << ")";
  return stream;
}

namespace onnx2trt
{

NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha, float* beta)
{
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

    return {{layer->getOutput(0)}};
}

NodeImportResult addScale(IImporterContext* ctx, nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode,
    nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power)
{
    nvinfer1::ITensor* tensor_ptr = &tensor_;
    nvinfer1::Dims dims = tensor_ptr->getDimensions();

    // Scale layer expects inputs to be 4D.
    int expectedNbDims = 4;
    bool need_to_expand_dims = (dims.nbDims != expectedNbDims);
    nvinfer1::Dims orig_shape = dims;
    if (need_to_expand_dims)
    {
        // Expand or squash dims to 4D
        nvinfer1::Dims new_shape = dims;
        while (new_shape.nbDims < expectedNbDims)
        {
            new_shape.d[new_shape.nbDims++] = 1;
        }
        while (new_shape.nbDims > expectedNbDims)
        {
            new_shape.d[3] *= new_shape.d[--new_shape.nbDims];
        }
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
    }

    ASSERT(dims.nbDims == expectedNbDims, ErrorCode::kUNSUPPORTED_NODE);

    // Fill in dtype for any unused (dummy) weights
    nvinfer1::DataType* dtype_ptr = nullptr;
    if (shift.count)
    {
        dtype_ptr = &shift.type;
    }
    if (scale.count)
    {
        ASSERT(!dtype_ptr || *dtype_ptr == scale.type, ErrorCode::kUNSUPPORTED_NODE);
        dtype_ptr = &scale.type;
    }
    if (power.count)
    {
        ASSERT(!dtype_ptr || *dtype_ptr == power.type, ErrorCode::kUNSUPPORTED_NODE);
        dtype_ptr = &power.type;
    }
    ASSERT(dtype_ptr, ErrorCode::kINTERNAL_ERROR);
    shift.type = *dtype_ptr;
    scale.type = *dtype_ptr;
    power.type = *dtype_ptr;
    auto* layer = ctx->network()->addScale(*tensor_ptr, mode, shift, scale, power);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    tensor_ptr = layer->getOutput(0);

    if (need_to_expand_dims)
    {
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, orig_shape);
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
    }

    return {{tensor_ptr}};
}

void auto_gen_input_output_padding(nvinfer1::Dims input_dims, nvinfer1::Dims output_shape, nvinfer1::Dims kernel_size,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& beg_padding,
    nvinfer1::Dims& end_padding, nvinfer1::Dims& output_padding, nvinfer1::PaddingMode paddingMode)
{
    // When auto_pad == NONSET or VALID, input padding is explict
    // explicit output shape may require output padding
    if (paddingMode == nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN)
    {
        nvinfer1::Dims expected_output_shape;
        for (int i = 0; i < nbSpatialDims; i++)
        {
            expected_output_shape.d[i] = (input_dims.d[2 + i] - 1) * strides.d[i]
                + (kernel_size.d[i] - 1) * dilations.d[i] + 1 - beg_padding.d[i] - end_padding.d[i];
            output_padding.d[i] = output_shape.d[i] - expected_output_shape.d[i];
        }
    }
    else
    {
        // When auto_pad == SAME_UPPER or SAME_LOWER, output padding is explict
        // explicit output shape may require input padding
        nvinfer1::Dims total_padding = makeDims(nbSpatialDims, 0);
        for (int i = 0; i < nbSpatialDims; i++)
        {
            total_padding.d[i] = (input_dims.d[2 + i] - 1) * strides.d[i] + (kernel_size.d[i] - 1) * dilations.d[i] + 1
                + output_padding.d[i] - output_shape.d[i];
            if (paddingMode == nvinfer1::PaddingMode::kSAME_UPPER)
            {
                beg_padding.d[i] = total_padding.d[i] - (total_padding.d[i] / 2);
                end_padding.d[i] = total_padding.d[i] / 2;
            }
            else
            {
                beg_padding.d[i] = total_padding.d[i] / 2;
                end_padding.d[i] = total_padding.d[i] - (total_padding.d[i] / 2);
            }
        }
    }
}

Status applyLegacyBinaryOpBroadcasting(IImporterContext* ctx,
                                       ::ONNX_NAMESPACE::NodeProto const& node,
                                       TensorOrWeights& lhs,
                                       TensorOrWeights& rhs) {
  int lhs_ndim = lhs.shape().nbDims;
  int rhs_ndim = rhs.shape().nbDims;
  OnnxAttrs attrs(node);
  bool broadcasting_on = (attrs.count("axis") && attrs.count("broadcast") &&
                          attrs.get<int>("broadcast"));
  if (rhs_ndim >= lhs_ndim || !broadcasting_on) 
  {
    return Status::success();
  }
  int axis = attrs.get<int>("axis");
  if( axis < 0 ) 
  {
    axis += lhs_ndim; // Support negative indexing
  }
  int num_dims_to_add_at_end = lhs_ndim - rhs_ndim - axis;
  ASSERT(num_dims_to_add_at_end >= 0, ErrorCode::kINVALID_NODE);

  nvinfer1::Dims new_shape;
  new_shape.nbDims = 0;

  for (int i = 0; i < axis; i++)
  {
    new_shape.d[new_shape.nbDims++] = 1;
  }

  for (int i = 0; i < rhs_ndim; i++)
  {
    new_shape.d[new_shape.nbDims++] = rhs.shape().d[i];
  }

  for (int i=0; i<num_dims_to_add_at_end; ++i) 
  {
    new_shape.d[new_shape.nbDims++] = 1;
  }

  if (rhs.is_weights()) 
  {
    rhs.weights().shape = new_shape;
  } 
  else 
  {
    ASSERT(rhs.reset_tensor(reshape_tensor(ctx, rhs.tensor(), new_shape)),
           ErrorCode::kUNSUPPORTED_NODE);
  }
  return Status::success();
}

NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op)
{
    nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    ASSERT(tensor.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
    // Get attributes.
    OnnxAttrs attrs(node);
    int keepdims = attrs.get("keepdims", 1);
    int axis = attrs.get("axis", 0);

    // Insert a TopK layer with k set to 1.
    int nbDims = tensor.getDimensions().nbDims;
    TRT_CHECK(convert_axis(axis, nbDims));

    uint32_t axisMask = 1 << axis;
    nvinfer1::ITopKLayer* layer = ctx->network()->addTopK(tensor, op, 1, axisMask);
    ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
    // We don't care about the TopK values, just the indices.
    nvinfer1::ITensor* indices = layer->getOutput(1);
    indices->setType(nvinfer1::DataType::kINT32);
    if (keepdims)
    {
        // The default behavior of the TopK layer is to keepdims.
        return {{indices}};
    }
    else
    {
        // Otherwise, we need to squeeze the axis dimension
        std::vector<int> axes{axis};
        indices = squeezeTensor(ctx, *indices, axes);
        return {{indices}};
    }
}

void broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, const int nbDims)
{

    nvinfer1::Dims inputDims = t1->getDimensions();
    int nbInputDims = inputDims.nbDims;
    if (nbInputDims >= nbDims)
    {
        return;
    }

    int numLeadingDims =  nbDims - nbInputDims;

    // Perform broadcasting by unsqueezing input tensor on axes [0...numLeadingDims - 1]
    if (nbInputDims > 0)
    {
        std::vector<int> axes(numLeadingDims);
        std::iota(axes.begin(), axes.end(), 0);
        t1 = unsqueezeTensor(ctx, *t1, axes);
    }
    // Handle scalar inputs by reshaping to shape of rank numLeadingDims.
    else
    {
        auto* reshapeLayer = ctx->network()->addShuffle(*t1);
        reshapeLayer->setReshapeDimensions(makeDims(numLeadingDims, 1));
        t1 = reshapeLayer->getOutput(0);
    }
}

void broadcastTensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2)
{
    const int t1Dims = t1->getDimensions().nbDims;
    const int t2Dims = t2->getDimensions().nbDims;

    if (t1Dims == t2Dims)
    {
        return;
    }

    if (t1Dims > t2Dims)
    {
        broadcastTensors(ctx, t2, t1Dims);
    }
    else
    {
        broadcastTensors(ctx, t1, t2Dims);
    }
}

Status check_broadcast_attrs(IImporterContext* ctx, OnnxAttrs const& attrs, nvinfer1::Dims const& dims)
{
    if (ctx->getOpsetVersion() < 7)
    {
        ASSERT(attrs.count("broadcast"), ErrorCode::kUNSUPPORTED_NODE);
        bool broadcast = attrs.get<int>("broadcast");
        ASSERT(broadcast || dims.nbDims == 1, ErrorCode::kINVALID_NODE);
        int axis = attrs.get<int>("axis", -1);
        int nbDims = dims.nbDims;
        TRT_CHECK(convert_axis(axis, nbDims));
        ASSERT(axis == 0, ErrorCode::kUNSUPPORTED_NODE);
    }
    return Status::success();
}

bool check_for_input(::ONNX_NAMESPACE::NodeProto const& node, std::string const& input_node)
{
  for (auto input : node.input())
  {
    if (input_node == input)
    {
      return true;
    }
  }
  return false;
}

bool check_for_int32(std::vector<TensorOrWeights>const & inputs)
{
    bool isInt32 = false;
    for (auto input : inputs)
    {
        if (input.is_weights())
        {
            nvinfer1::DataType dt;
            convert_dtype(input.weights().type, &dt);
            isInt32 |= dt == nvinfer1::DataType::kINT32;
        }
        else
        {
            isInt32 |= input.tensor().getType() == nvinfer1::DataType::kINT32;
        }
    }
    return isInt32;
}

bool check_for_scale(std::vector<TensorOrWeights>const & inputs)
{
    bool useScale = true;
    // Inputs cannot be int32.
    useScale &= !(check_for_int32(inputs));
    // One input must be a tensor and the other one must be a weight.
    useScale &= inputs.at(0).is_tensor() != inputs.at(1).is_tensor();
    // Tensor input must be at least 4D.
    if (useScale)
    {
        const nvinfer1::ITensor& tensor = inputs.at(0).is_tensor() ? inputs.at(0).tensor() : inputs.at(1).tensor();
        useScale &= tensor.getDimensions().nbDims >= 4;
    }
    return useScale;
}

Status convert_axis(int& axis, int nbDims)
{
  // Support negative indexing
  if (axis < 0)
  {
    axis += nbDims;
  }
  ASSERT(axis >= 0 && axis < nbDims, ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

bool convert_dtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype)
{
  switch( onnx_dtype )
  {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:   *trt_dtype = nvinfer1::DataType::kFLOAT; break;
  case ::ONNX_NAMESPACE::TensorProto::INT8:    *trt_dtype = nvinfer1::DataType::kINT8;  break;
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16: *trt_dtype = nvinfer1::DataType::kHALF;  break;
  // See ShapedWeights.cpp for sanity check if all values can be safetly downcasted to INT32
  case ::ONNX_NAMESPACE::TensorProto::INT64:   *trt_dtype = nvinfer1::DataType::kINT32; break;
  case ::ONNX_NAMESPACE::TensorProto::INT32:   *trt_dtype = nvinfer1::DataType::kINT32; break;
  default:
    cerr << "Unsupported ONNX data type: " << get_dtype_name(onnx_dtype)
         << " (" << std::to_string(onnx_dtype) << ")" << endl;
    return false;
  }
  return true;
}

bool convert_input_dtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype)
{
  switch( onnx_dtype )
  {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:   *trt_dtype = nvinfer1::DataType::kFLOAT; break;
  case ::ONNX_NAMESPACE::TensorProto::INT8:    *trt_dtype = nvinfer1::DataType::kINT8;  break;
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16: *trt_dtype = nvinfer1::DataType::kHALF;  break;
  case ::ONNX_NAMESPACE::TensorProto::INT32:   *trt_dtype = nvinfer1::DataType::kINT32; break;
  default:
    cerr << "Unsupported ONNX data type: " << get_dtype_name(onnx_dtype)
         << " (" << std::to_string(onnx_dtype) << ")" << endl;
    return false;
  }
  return true;
}

bool convert_onnx_weights(::ONNX_NAMESPACE::TensorProto const& onnx_tensor, onnx2trt::ShapedWeights* weights)
{
    nvinfer1::Dims shape;
    shape.nbDims = onnx_tensor.dims().size();
    std::copy(onnx_tensor.dims().begin(), onnx_tensor.dims().end(), shape.d);
    auto const& onnx_tensor_type = onnx_tensor.data_type();
    void* data_ptr; // TODO: See if can make const*
    size_t nbytes;
    if( onnx_tensor.raw_data().size() > 0 )
    {
        data_ptr = (void*)onnx_tensor.raw_data().data();
        nbytes = onnx_tensor.raw_data().size();
    }
    else if( onnx_tensor.float_data().size() > 0 )
    {
        assert(onnx_tensor_type == ::ONNX_NAMESPACE::TensorProto::FLOAT);
        data_ptr = (void*)onnx_tensor.float_data().data();
        nbytes = onnx_tensor.float_data().size() * sizeof(float);
    }
    else if( onnx_tensor.int32_data().size() > 0 )
    {
        // TODO: Need special handling for int8 or float16 stored as int32_data
        assert(get_dtype_size(onnx_tensor_type) == 4);
        data_ptr = (void*)onnx_tensor.int32_data().data();
        nbytes = onnx_tensor.int32_data().size() * sizeof(int32_t);
    }
    else if( onnx_tensor.int64_data().size() > 0 )
    {
        assert(onnx_tensor_type == ::ONNX_NAMESPACE::TensorProto::INT64);
        data_ptr = (void*)onnx_tensor.int64_data().data();
        nbytes = onnx_tensor.int64_data().size() * sizeof(int64_t);
    }
    else
    {
        // Unsupported ONNX tensor format!
        return false;
    }

    // nvinfer1::DataType trt_dtype;
    // convert_dtype(onnx_tensor_type, &trt_dtype);


    onnx2trt::ShapedWeights trt_weights(onnx_tensor_type, data_ptr, shape);
    (void)nbytes;
    assert(trt_weights.size_bytes() == nbytes);
    *weights = trt_weights;
    return true;
}

nvinfer1::ITensor& convert_output_weight_to_tensor(TensorOrWeights& input, IImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return input.tensor();
    }
    else
    {
        const ShapedWeights& weights = input.weights();
        nvinfer1::Dims tensor_shape = weights.shape;
        return *(ctx->network()->addConstant(tensor_shape, weights)->getOutput(0));
    }

}

nvinfer1::ITensor* convert_tensor_to_2d(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis = 0)
{
    nvinfer1::Dims shape = tensor.getDimensions();
    nvinfer1::Dims new_shape = makeDims(2, 1);
    for (int i = 0; i < axis; ++i)
    {
        new_shape.d[0] *= shape.d[i];
    }
    for (int i = axis; i < shape.nbDims; ++i)
    {
        new_shape.d[1] *= shape.d[i];
    }
    return reshape_tensor(ctx, tensor, new_shape);
}

bool convert_weight_descriptor(onnxTensorDescriptorV1 const &desc, onnx2trt::ShapedWeights *weights)
{
  nvinfer1::Dims shape;
  shape.nbDims = desc.dimensions;
  // Special case for scalars
  if( shape.nbDims == 0 ) {
    shape.nbDims = 1;
    shape.d[0] = 1;
    shape.type[0] = nvinfer1::DimensionType::kCHANNEL;
  } else {
    std::copy(desc.shape, desc.shape + desc.dimensions, shape.d);
  }

  size_t element_count = 1;
  for (int i = 0; i < shape.nbDims; ++i) {
    element_count *= shape.d[i];
  }

  void* data_ptr;
  size_t nbytes;
  int32_t dtype;
  data_ptr = (void*)(desc. buffer);
  if (desc.dataType == ONNXIFI_DATATYPE_FLOAT32) {
    dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT;
    nbytes = element_count * sizeof(float);
  } else if (desc.dataType == ONNXIFI_DATATYPE_FLOAT16) {
    dtype = ::ONNX_NAMESPACE::TensorProto::FLOAT16;
    nbytes = element_count * sizeof(float) / 2;
  } else if (desc.dataType == ONNXIFI_DATATYPE_INT32) {
    dtype = ::ONNX_NAMESPACE::TensorProto::INT32;
    nbytes = element_count * sizeof(int32_t);
  } else if (desc.dataType == ONNXIFI_DATATYPE_INT64) {
    dtype = ::ONNX_NAMESPACE::TensorProto::INT64;
    nbytes = element_count * sizeof(int64_t);
  } else {
    // Unsupported format
    return false;
  }

  onnx2trt::ShapedWeights trt_weights(dtype, data_ptr, shape);
  (void)nbytes;
  assert(trt_weights.size_bytes() == nbytes);
  *weights = trt_weights;
  return true;
}

nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx)
{
    if (input.is_tensor())
    {
        return input.tensor();
    }
    else
    {
        // Handle non-tensor indices input by adding a new constant layer to the network.
        const ShapedWeights& weights = input.weights();
        return *(ctx->network()->addConstant(weights.shape, weights)->getOutput(0));
    }

}

nvinfer1::ITensor& dimension_to_tensor(IImporterContext* ctx, nvinfer1::Dims dims)
{
    ShapedWeights temp_weights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_INT32, nvinfer1::Dims{1, {dims.nbDims}});
    std::vector<int> data(dims.d, dims.d+dims.nbDims);
    std::copy(data.begin(), data.end(), static_cast<int*>(temp_weights.values));
    TensorOrWeights w(temp_weights);
    nvinfer1::ITensor& input = convertToTensor(w, ctx);
    return input;
}


int div_ceil(int n, int d) 
{
  return (n - 1) / d + 1;
}

NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op)
{
    ASSERT(!inputs.empty(), ErrorCode::kINVALID_NODE);
    std::vector<nvinfer1::ITensor*> inputTensors;
    int maxNbDims = -1;
    for (auto input : inputs)
    {
        maxNbDims = std::max(maxNbDims, input.shape().nbDims);
    }

    for (auto input: inputs)
    {
        nvinfer1::ITensor* tensor_ptr;
        if (input.shape().nbDims == 0 && input.is_weights())
        {
          if (input.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32 || input.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
          {
            int32_t index = static_cast<int32_t*>(input.weights().values)[0];
            tensor_ptr = addConstantScalar<int32_t>(ctx, static_cast<int32_t>(index), ::ONNX_NAMESPACE::TensorProto::INT32)->getOutput(0);
          }
          else
          {
            float index = static_cast<float*>(input.weights().values)[0];
            tensor_ptr = addConstantScalar<float>(ctx, static_cast<float>(index), ::ONNX_NAMESPACE::TensorProto::FLOAT)->getOutput(0);
          }
        }
        else
        {
            tensor_ptr = &convertToTensor(input, ctx);
        }
        // Broadcast all input tensors to size of maxNbDims
        broadcastTensors(ctx, tensor_ptr, maxNbDims);
        ASSERT(tensor_ptr->getDimensions().nbDims == maxNbDims && "Failed to broadcast tensors elementwise!",
               ErrorCode::kUNSUPPORTED_NODE);
        inputTensors.push_back(tensor_ptr);
    }

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
        ASSERT(tensor->getDimensions().nbDims == combined->getDimensions().nbDims, ErrorCode::kUNSUPPORTED_NODE);
        auto* layer = ctx->network()->addElementWise(*combined, *tensor, binary_op);
        ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
        combined = layer->getOutput(0);
    }
    return {{combined}};
}

nvinfer1::ITensor* flattenTensorDynamic(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    auto* shape = ctx->network()->addShape(tensor)->getOutput(0);

    // Gather "start indicies" in a loop and multiply them together to get the "start shape"
    // Do the same for the "axis indices" and then concat them together.

    nvinfer1::ITensor* startTensor;
    nvinfer1::ITensor* axisTensor;

    nvinfer1::Dims singleDim = makeDims(1, 1);

    // If axis is 0, startTensor is a single 1.
    if (axis == 0)
    {
        startTensor = &makeShapeTensor(ctx, singleDim);
    }
    else
    {      
        std::vector<int> startIndex {0};
        auto* startLayer = ctx->network()->addGather(*shape, *addConstant(ctx, startIndex, ::ONNX_NAMESPACE::TensorProto::INT32, singleDim)->getOutput(0), 0);
        startTensor = startLayer->getOutput(0);

        for (int i = 1; i < axis; i++)
        {
          startIndex = {i};
          auto* beforeLayer = ctx->network()->addGather(*shape, *addConstant(ctx, startIndex, ::ONNX_NAMESPACE::TensorProto::INT32, singleDim)->getOutput(0), 0);
          auto* before = beforeLayer->getOutput(0);
          startTensor = ctx->network()->addElementWise(*startTensor, *before, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
        }
    }

    std::vector<int> axisIndex {axis};
    auto* axisLayer = ctx->network()->addGather(*shape, *addConstant(ctx, axisIndex, ::ONNX_NAMESPACE::TensorProto::INT32, singleDim)->getOutput(0), 0);
    axisTensor = axisLayer->getOutput(0);

    for (int i = axis + 1; i < dims.nbDims; i++)
    {
        axisIndex = {i};
        auto* afterLayer = ctx->network()->addGather(*shape, *addConstant(ctx, axisIndex, ::ONNX_NAMESPACE::TensorProto::INT32, singleDim)->getOutput(0), 0);
        auto* after = afterLayer->getOutput(0);
        axisTensor = ctx->network()->addElementWise(*axisTensor, *after, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    }

    std::vector<nvinfer1::ITensor*> tensors{startTensor, axisTensor};

    auto* concatLayer = ctx->network()->addConcatenation(tensors.data(), tensors.size());
    auto* reshapeLayer = ctx->network()->addShuffle(tensor);
    reshapeLayer->setInput(1, *concatLayer->getOutput(0));

    return reshapeLayer->getOutput(0);
}

nvinfer1::ITensor* flattenTensorStatic(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    nvinfer1::Dims flattenDims = makeDims(2, 1);

    if (axis != 0)
    {
        flattenDims.d[0] = std::accumulate(dims.d, dims.d + axis, 1, std::multiplies<int>());
    }

    flattenDims.d[1] = std::accumulate(dims.d + axis, dims.d + dims.nbDims, 1, std::multiplies<int>());

    auto* flattenLayer = ctx->network()->addShuffle(tensor);
    flattenLayer->setReshapeDimensions(flattenDims);
    return flattenLayer->getOutput(0);
}


nvinfer1::ITensor* flattenTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis)
{
    if (isDynamic(tensor.getDimensions()))
    {
        return flattenTensorDynamic(ctx, tensor, axis);
    }

    return flattenTensorStatic(ctx, tensor, axis);
}

bool isDynamic(nvinfer1::Dims const& dims)
{
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) {return dim == -1;});
}

nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields)
{
    const auto mPluginRegistry = getPluginRegistry();

    const auto pluginCreator = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "ONNXTRT_NAMESPACE");

    if (!pluginCreator)
    {
      return nullptr;
    }

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return pluginCreator->createPlugin(nodeName.c_str(), &fc);
}

bool is_transpose_required(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm)
{
    int ndim = shape.nbDims;
    int prev_significant_dim = 0;
    for (int dst_i = 0; dst_i < ndim; ++dst_i)
    {
        int src_i = perm.order[dst_i];
        int dim_i = shape.d[src_i];
        if (dim_i != 1)
        {
            // For transposes on dynamically shaped tensors, we must return true.
            if (dim_i == -1)
            {
                return true;
            }
            else if (src_i < prev_significant_dim)
            {
                return true;
            }
            prev_significant_dim = src_i;
        }
    }
    return false;
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
        auto& inpShape = *ctx->network()->addShape(*inpTensor)->getOutput(0);
        auto& axisValue = *addConstantScalar(ctx, axis, ::ONNX_NAMESPACE::TensorProto_DataType_INT32, shape)->getOutput(0);
        return ctx->network()->addGather(inpShape, axisValue, 0)->getOutput(0);
    }
}

int get_conv_output_size(int input_size, int filter_size,
                         int stride, int dilation_rate,
                         int total_padding) 
{
  // This is based on the CUDNN formula
  int effective_input_size  = input_size + total_padding;
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  return div_ceil(effective_input_size - (effective_filter_size - 1), stride);
}

const char* get_dtype_name(int32_t onnx_dtype) {
  switch( onnx_dtype ) {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:      return "FLOAT";
  case ::ONNX_NAMESPACE::TensorProto::UINT8:      return "UINT8";
  case ::ONNX_NAMESPACE::TensorProto::INT8:       return "INT8";
  case ::ONNX_NAMESPACE::TensorProto::UINT16:     return "UINT16";
  case ::ONNX_NAMESPACE::TensorProto::INT16:      return "INT16";
  case ::ONNX_NAMESPACE::TensorProto::INT32:      return "INT32";
  case ::ONNX_NAMESPACE::TensorProto::INT64:      return "INT64";
  case ::ONNX_NAMESPACE::TensorProto::STRING:     return "STRING";
  case ::ONNX_NAMESPACE::TensorProto::BOOL:       return "BOOL";
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16:    return "FLOAT16";
  case ::ONNX_NAMESPACE::TensorProto::DOUBLE:     return "DOUBLE";
  case ::ONNX_NAMESPACE::TensorProto::UINT32:     return "UINT32";
  case ::ONNX_NAMESPACE::TensorProto::UINT64:     return "UINT64";
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:  return "COMPLEX64";
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return "COMPLEX128";
  default: return "<UNKNOWN>";
  }
}

int get_dtype_size(int32_t onnx_dtype) {
  switch( onnx_dtype ) {
  case ::ONNX_NAMESPACE::TensorProto::FLOAT16:    return 2;
  case ::ONNX_NAMESPACE::TensorProto::FLOAT:      return 4;
  case ::ONNX_NAMESPACE::TensorProto::DOUBLE:     return 8;
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX64:  return 8;
  case ::ONNX_NAMESPACE::TensorProto::COMPLEX128: return 16;
  case ::ONNX_NAMESPACE::TensorProto::UINT8:      return 1;
  case ::ONNX_NAMESPACE::TensorProto::INT8:       return 1;
  case ::ONNX_NAMESPACE::TensorProto::UINT16:     return 2;
  case ::ONNX_NAMESPACE::TensorProto::INT16:      return 2;
  case ::ONNX_NAMESPACE::TensorProto::UINT32:     return 4;
  case ::ONNX_NAMESPACE::TensorProto::INT32:      return 4;
  case ::ONNX_NAMESPACE::TensorProto::UINT64:     return 8;
  case ::ONNX_NAMESPACE::TensorProto::INT64:      return 8;
  // TODO: Add BOOL if necessary...
    // TODO: Some sort of error handling
  default: return -1;//throw std::invalid_argument("Unsupported TRT data type: " +
                     //                  std::to_string((int)trt_dtype));
  }
}

Status get_infer_dim(int& infer_dim, nvinfer1::Dims const& new_shape)
{
  for (int i = 0; i < new_shape.nbDims; ++i)
  {
    if (new_shape.d[i] < 0)
    {
      // -1 bears special meaning, which means the current dimension can
      // be inferred while keepin the total number of elements the same.
      // https://github.com/onnx/onnx/blob/9b9f595107e3fc0295d50f6294d43879df17552f/onnx/defs/tensor/defs.cc#L73-L75
      ASSERT(new_shape.d[i] == -1, ErrorCode::kUNSUPPORTED_NODE);
      // We can only one dimension that has -1
      ASSERT(infer_dim == -1, ErrorCode::kUNSUPPORTED_NODE);
      infer_dim = i;
    }
  }
  return Status::success();
}

void get_kernel_params(::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::Dims* kernel_size,
                       nvinfer1::Dims* strides,
                       nvinfer1::Dims* beg_padding,
                       nvinfer1::Dims* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       bool & count_exclude_padding,
                       nvinfer1::Dims* dilations,
                       nvinfer1::Dims* output_padding) {
  const int nbSpatialDims = kernel_size->nbDims;
  OnnxAttrs attrs(onnx_node);
  if( attrs.count("kernel_shape") ) {
    auto const* onnx_kernel_size = attrs.at("kernel_shape");
    setAttr(kernel_size, onnx_kernel_size, nbSpatialDims, 1);
  }
  if( attrs.count("strides") ) {
    auto const* onnx_strides = attrs.at("strides");
    setAttr(strides, onnx_strides, nbSpatialDims, 1);
  }
  if( dilations && attrs.count("dilations") ) {
    auto const* onnx_dilations = attrs.at("dilations");
    setAttr(dilations, onnx_dilations, nbSpatialDims, 1);
  }
  if( attrs.count("count_include_pad")){
    auto const* include_pad = attrs.at("count_include_pad");
    int val = include_pad->i();
    val == 1 ? count_exclude_padding=false : count_exclude_padding=true;
  }
  //For ConvTranspose Layer
  if( attrs.count("output_padding") ) {
    *output_padding = attrs.get<nvinfer1::Dims>("output_padding");
  }

  paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
  auto onnx_auto_pad = attrs.get("auto_pad", std::string("NOTSET"));
  if( onnx_auto_pad == "VALID" || onnx_auto_pad == "NOTSET" ) {
    if( attrs.count("pads") ) {
      auto onnx_padding = attrs.get<std::vector<int>>("pads");
      int ndim = onnx_padding.size() / 2;
      for(int i = 0; i < nbSpatialDims; ++i){
        if(i < ndim){
          beg_padding->d[i] = onnx_padding.at(i);
          end_padding->d[i] = onnx_padding.at(i + ndim);
        } else {
          beg_padding->d[i] = 0;
          end_padding->d[i] = 0;
        }
      }
    }
  } 
  else 
  {
    // If auto_pad is SAME_LOWER or SAME_UPPER, input padding should be calculated
    // "pads" attribute should not be specified
    assert(!attrs.count("pads"));
    // Note: ONNX is always NCHW ordering
    if( onnx_auto_pad == "SAME_LOWER" ) 
    {
    paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
    } 
    else if( onnx_auto_pad == "SAME_UPPER" ) 
    {
    paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } 
    else 
    {
      throw std::invalid_argument("Unexpected auto_pad value: " +
                                  onnx_auto_pad);
    }
  }
}

nvinfer1::ScaleMode get_scale_mode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape)
{
  if (weights_shape.nbDims == 1)
  {
    if (weights_shape.d[0] == 1)
    {
      return nvinfer1::ScaleMode::kUNIFORM;
    } 
    // Check for channel wide scale - assume tensor shape is NCHW.
    else if (weights_shape.d[0] == tensor_shape.d[1])
    {
      return nvinfer1::ScaleMode::kCHANNEL;
    }
  } 
  return nvinfer1::ScaleMode::kELEMENTWISE;
}

nvinfer1::Dims makeDims(int nbDims, int val)
{
    nvinfer1::Dims dims;
    dims.nbDims = nbDims;
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

nvinfer1::ITensor& makeShapeTensor(IImporterContext* ctx, nvinfer1::Dims dims)
{
    auto tempWeights = ctx->createTempWeights(::ONNX_NAMESPACE::TensorProto_DataType_INT32, makeDims(1, dims.nbDims));
    for(int i = 0; i < dims.nbDims; i++)
    {
        static_cast<int32_t*>(tempWeights.values)[i] = dims.d[i];
    }
    auto valueWeights = TensorOrWeights{tempWeights};
    return convertToTensor(valueWeights, ctx);
}

NodeImportResult poolingHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs, nvinfer1::PoolingType type)
{
    nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);
    nvinfer1::Dims dims = tensorPtr->getDimensions();
    bool needToExpandDims = (dims.nbDims == 3);
    if (needToExpandDims)
    {
        // Expand spatial dims from 1D to 2D
        std::vector<int>axes {3};
        tensorPtr = unsqueezeTensor(ctx, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensorPtr->getDimensions();
    }

    OnnxAttrs attrs(node);
    int nbSpatialDims = attrs.at("kernel_shape")->ints().size();   
    ASSERT((nbSpatialDims == 1 && needToExpandDims) || nbSpatialDims == 2 || nbSpatialDims == 3,
      ErrorCode::kUNSUPPORTED_NODE);

    // Support for opset10 ceil_mode
    CeilingPoolDim ceilingPool;
    // Ceiling and dialations added in opset 10
    if (ctx->getOpsetVersion() >= 10)
    {
        const auto ceil_mode = attrs.get<int>("ceil_mode", 0);
        const auto dilations = attrs.get<std::vector<int>>("dilations", std::vector<int>(2, 1));
        for (size_t i = 0; i < dilations.size(); i++)
            ASSERT(dilations[i] == 1, ErrorCode::kUNSUPPORTED_NODE); // Do not support pooling dilations currently
        nvinfer1::Dims spatialDims = makeDims(nbSpatialDims, 1);
        std::copy(&dims.d[2], &dims.d[2]+nbSpatialDims, &spatialDims.d[0]);
        if (ceil_mode != 0) // Need to set pooling formula to use ceiling instead of floor
        {
            ASSERT(!isDynamic(spatialDims) && "Cannot set output formula for an input with dynamic spatial dimensions!", ErrorCode::kUNSUPPORTED_NODE);
            ctx->network()->setPoolingOutputDimensionsFormula(&ceilingPool);
        }
    }

    nvinfer1::Dims kernel_size = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims strides = makeDims(nbSpatialDims, 1);
    nvinfer1::Dims beg_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::Dims end_padding = makeDims(nbSpatialDims, 0);
    nvinfer1::PaddingMode paddingMode;

    bool exclude_padding(true);

    // Populate pooling parameters.
    get_kernel_params(node, &kernel_size, &strides, &beg_padding, &end_padding, paddingMode, exclude_padding);

    nvinfer1::IPoolingLayer* poolingLayer = ctx->network()->addPoolingNd(*tensorPtr, type, kernel_size);
    poolingLayer->setStrideNd(strides);
    // This member is ignored in maxpooling
    poolingLayer->setAverageCountExcludesPadding(exclude_padding);
    poolingLayer->setPaddingMode(paddingMode);
    poolingLayer->setPrePadding(beg_padding);
    poolingLayer->setPostPadding(end_padding);

    // Note: Average pooling requires special care with asymmetric padding
    //       because the need to exclude padding pixels from the average
    //       means we can't just use a pre-padding layer.
    if (type == nvinfer1::PoolingType::kAVERAGE)
    {
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
    }

    tensorPtr = poolingLayer->getOutput(0);
    dims = tensorPtr->getDimensions();
    if (needToExpandDims)
    {
        // Un-expand spatial dims back to 1D
        std::vector<int>axes{3};
        tensorPtr = squeezeTensor(ctx, *tensorPtr, axes);
        ASSERT(tensorPtr, ErrorCode::kUNSUPPORTED_NODE);
    }
    return {{tensorPtr}};
}

nvinfer1::ITensor* reshape_tensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape)
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

    //layer->setInput(1,*inputShapeTensor);
    layer->setReshapeDimensions(shape);

    return layer->getOutput(0);
}


NodeImportResult scaleHelper(IImporterContext* ctx,
                               ::ONNX_NAMESPACE::NodeProto const& node,
                               std::vector<TensorOrWeights>& inputs,
                               ScaleOp op) {
  nvinfer1::ITensor* tensor_ptr = (inputs.at(0).is_tensor() ?
                                  &inputs.at(0).tensor() :
                                  &inputs.at(1).tensor());
  ShapedWeights weights = (inputs.at(0).is_weights() ?
                          inputs.at(0).weights() :
                          inputs.at(1).weights());
  nvinfer1::Dims dims = tensor_ptr->getDimensions();
  // Note: ONNX opset >= 7 uses Numpy-style broadcasting, so dims are padded
  // at the end with ones for broadcasting.
  weights.shape = squeeze_trailing_dims(weights.shape);
  nvinfer1::ScaleMode mode = get_scale_mode(weights.shape, dims);
  if (mode == nvinfer1::ScaleMode::kELEMENTWISE)
  {
    nvinfer1::ElementWiseOperation elementwise_op = {};
    switch (op)
    {
      case kSHIFT: elementwise_op = nvinfer1::ElementWiseOperation::kSUM; break;
      case kSCALE: elementwise_op = nvinfer1::ElementWiseOperation::kPROD; break;
      case kPOWER: elementwise_op = nvinfer1::ElementWiseOperation::kPOW; break;
    }
    // If shapes do not entirely match up, an elementwise layer is needed instead
    // to support full broadcasting.
    if (get_shape_size(weights.shape) != get_shape_size(dims))
    {
      return elementwiseHelper(ctx,
                          node,
                          inputs,
                          elementwise_op);
    }
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

void setAttr(nvinfer1::Dims * trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal){
  assert(trtAttr->nbDims == nbSpatialDims);
  int ndim = onnxAttr->ints().size();
  for(int i = 0; i < nbSpatialDims; ++i){
      if(i < ndim){
        trtAttr->d[i] = onnxAttr->ints(i);
      } else {
        trtAttr->d[i] = defaultVal;
      }
  }
}

Status slice_array(TensorOrWeights weights, std::vector<int32_t>& weight_vector)
{
    ASSERT(weights.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT((weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32) || (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64),
        ErrorCode::kINVALID_NODE);
    weight_vector.resize(weights.weights().count());
    if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        onnx2trt::convertINT64(weights.weights().values, weights.weights().count(), weight_vector);
    }
    else
    {
        auto array_start = static_cast<int32_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weight_vector.begin());
    }
    return Status(ErrorCode::kSUCCESS);
}

nvinfer1::ITensor* squeezeStaticTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    std::set<int> axesSet(axes.begin(), axes.end());
    std::vector<int> shape{dims.d, dims.d + dims.nbDims};

    for (const auto& axis : axesSet)
    {
        shape.erase(shape.begin() + axis);
    }

    nvinfer1::Dims newShape{dims.nbDims - static_cast<int>(axesSet.size())};
    std::copy(shape.begin(), shape.end(), newShape.d);

    auto* unsqueezeLayer = ctx->network()->addShuffle(tensor);
    unsqueezeLayer->setReshapeDimensions(newShape);
    return unsqueezeLayer->getOutput(0);
}

nvinfer1::ITensor* squeezeDynamicTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{
    int rank = tensor.getDimensions().nbDims;
    std::set<int> axisSet(axes.begin(), axes.end());

    std::vector<int> gatherIndices{};
    for (int i = 0; i < rank; ++i)
    {
        if (!axisSet.count(i))
        {
            gatherIndices.emplace_back(i);
        }
    }

    nvinfer1::IShapeLayer* shape = ctx->network()->addShape(tensor);

    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);

    if (gatherIndices.size() > 0)
    {
        nvinfer1::Dims gatherIndicesShape{1, static_cast<int>(gatherIndices.size())};
        nvinfer1::IGatherLayer* newShape = ctx->network()->addGather(*shape->getOutput(0), *addConstant(ctx, gatherIndices, ::ONNX_NAMESPACE::TensorProto::INT32, gatherIndicesShape)->getOutput(0), 0);
        layer->setInput(1, *newShape->getOutput(0));
    }
    else
    {
        layer->setReshapeDimensions(nvinfer1::Dims{0});
    }

    return layer->getOutput(0);
}

nvinfer1::ITensor* squeezeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{
    if (isDynamic(tensor.getDimensions()))
    {
        return squeezeDynamicTensor(ctx, tensor, axes);
    }
    return squeezeStaticTensor(ctx, tensor, axes);
}

nvinfer1::ITensor* transpose_tensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm, 
                                    bool permute_dim_types = true)
{
    nvinfer1::IShuffleLayer* layer = ctx->network()->addShuffle(tensor);
    if (!layer)
    {
        return nullptr;
    }
    nvinfer1::Dims shape = tensor.getDimensions();
    // If a transpose is required, add transpose property to the shuffle layer.
    if (is_transpose_required(shape, perm))
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
    }
    return layer->getOutput(0);
}

NodeImportResult unaryHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::UnaryOperation op)
{
    nvinfer1::ITensor& input = convertToTensor(inputs.at(0), ctx);
    nvinfer1::IUnaryLayer* layer = ctx->network()->addUnary(input, op);
    return {{layer->getOutput(0)}};
}

nvinfer1::ITensor* unsqueezeStaticTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    std::set<int> axesSet(axes.begin(), axes.end());
    std::vector<int> shape{dims.d, dims.d + dims.nbDims};

    for (const auto& axis : axesSet)
    {
        shape.insert(shape.begin() + axis, 1);
    }

    nvinfer1::Dims newShape{dims.nbDims + static_cast<int>(axesSet.size())};
    std::copy(shape.begin(), shape.end(), newShape.d);

    auto* unsqueezeLayer = ctx->network()->addShuffle(tensor);
    unsqueezeLayer->setReshapeDimensions(newShape);
    return unsqueezeLayer->getOutput(0);
}

nvinfer1::ITensor* unsqueezeDynamicTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{
    nvinfer1::Dims dims = tensor.getDimensions();
    int nbDims = dims.nbDims;
    std::set<int> axesSet(axes.begin(), axes.end());
    int newNbDims = nbDims + axesSet.size();

    // Ensure we're not unsqueezing past the maximum allowed dimensions
    if (newNbDims > nvinfer1::Dims::MAX_DIMS)
    {
        return nullptr;
    }

    auto singleDim = makeDims(1, 1);
    auto* singleShape = &makeShapeTensor(ctx, singleDim);

    nvinfer1::ITensor* tensorShape{nullptr};
    if (tensor.getDimensions().nbDims > 0)
    {
        tensorShape = ctx->network()->addShape(tensor)->getOutput(0);
    }

    std::vector<nvinfer1::ITensor*> shapeTensors;

    for (int axesCounter = 0, shapeCounter = 0; (axesCounter + shapeCounter < newNbDims);)
    {
        if (axesSet.count(axesCounter + shapeCounter))
        {
            shapeTensors.push_back(singleShape);
            axesCounter++;
        }
        else
        {
            // This branch should never be taken when the input tensor is a scalar.
            std::vector<int> index{shapeCounter};
            auto* gatherLayer = ctx->network()->addGather(*tensorShape, *addConstant(ctx, index, ::ONNX_NAMESPACE::TensorProto::INT32, singleDim)->getOutput(0), 0);
            shapeTensors.push_back(gatherLayer->getOutput(0));
            shapeCounter++;
        }
    }

    auto* newShapeLayer = ctx->network()->addConcatenation(shapeTensors.data(), shapeTensors.size());

    auto* unsqueezeLayer = ctx->network()->addShuffle(tensor);
    unsqueezeLayer->setInput(1, *newShapeLayer->getOutput(0));

    return unsqueezeLayer->getOutput(0);
}

nvinfer1::ITensor* unsqueezeTensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, const std::vector<int>& axes)
{

    if (isDynamic(tensor.getDimensions()))
    {
        return unsqueezeDynamicTensor(ctx, tensor, axes);
    }
    return unsqueezeStaticTensor(ctx, tensor, axes);
}

void update_padded_values(std::vector<float>&pad_values, const nvinfer1::DimsHW beg_padding,
  const nvinfer1::DimsHW end_padding, const nvinfer1::Dims padded_shape, const float pad_value)
{
  int pad_h = padded_shape.d[1];
  int pad_w = padded_shape.d[2];
  int num_elements = pad_values.size();

  // Handle H padding. First beg_padding.h * pad_w and last end_padding.h * pad_w
  // elements need to be updated to pad_value
  if (beg_padding.h() != 0)
  {
    int end = beg_padding.h() * pad_w;
    for (int i = 0; i < end; i++)
    {
      pad_values[i] = pad_value;
    }
  }
  if (end_padding.h() != 0)
  {
    for (int start = (pad_h - end_padding.h()) * pad_w; 
        start < num_elements; start++)
    {
      pad_values[start] = pad_value;
    }

  }
  // Handle W padding. First beg_padding.w() and last end_padding.w() 
  // elements of each row needs to be updated to pad_value
  if (beg_padding.w() != 0)
  {
    for (int h_dim = 0; h_dim < pad_h; h_dim++)
    {
      for (int w_dim = 0; w_dim < beg_padding.w(); w_dim++)
      {
        int row_base_index = h_dim*pad_h;
        pad_values[row_base_index + w_dim] = pad_value;
      }
    }
  }
  if (end_padding.w() != 0)
  {
    for (int h_dim = 0; h_dim < pad_h; h_dim++)
    {
      for (int w_dim = pad_w - end_padding.w();
          w_dim < pad_w; w_dim++)
      {
        int row_base_index = h_dim*pad_h;
        pad_values[row_base_index + w_dim] = pad_value;
      }
    }
  }
}

int64_t volume(const nvinfer1::Dims& dims)
{
    std::for_each(dims.d, dims.d + dims.nbDims, [](int d){ assert(d >= 0 && "volume makes no sense for dynamic shapes");});
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>{});
}

Status weightsToVector(TensorOrWeights weights, std::vector<int64_t>* weightVector)
{
    ASSERT(weights.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT((weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT32) || (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64),
        ErrorCode::kINVALID_NODE);
    weightVector->resize(weights.weights().count());
    if (weights.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
    {
        auto array_start = static_cast<int64_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    else
    {
        auto array_start = static_cast<int32_t*>(weights.weights().values);
        std::copy(array_start, array_start + weights.weights().count(), weightVector->begin());
    }
    return Status(ErrorCode::kSUCCESS);
}

} // namespace onnx2trt
