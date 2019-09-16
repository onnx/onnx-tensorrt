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
#include "trt_utils.hpp"
#include "OnnxAttrs.hpp"

#include <onnx/onnx_pb.h>
#include <onnx/onnxifi.h>
#include <NvInfer.h>

#include <iostream>
using std::cerr;
using std::endl;

class CeilingPoolDim:public nvinfer1::IOutputDimensionsFormula
{
public:
    nvinfer1::DimsHW compute(nvinfer1::DimsHW inputDims, nvinfer1::DimsHW kernelSize,
    nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, nvinfer1::DimsHW dilation, const char* layerName) const
    {
      nvinfer1::DimsHW outputDims;
      for (int dimension = 0; dimension < inputDims.nbDims; dimension++)
      {
        outputDims.d[dimension] = static_cast<int>(ceil((inputDims.d[dimension] + padding.d[dimension] * 2.0 - kernelSize.d[dimension]) / stride.d[dimension] + 1.0));
      }
      return outputDims;
  }
};

std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape);

std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype);

std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm);


namespace onnx2trt
{

enum ScaleOp
{
    kSHIFT,
    kSCALE,
    kPOWER,
};

// Helper function to import ONNX activation nodes into TRT
NodeImportResult activationHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ActivationType op, float* alpha = nullptr, float* beta = nullptr);

// Helper function to add a Scale layer into TRT
NodeImportResult addScale(IImporterContext* ctx, nvinfer1::ITensor& tensor_, nvinfer1::ScaleMode mode,
    nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power);

// Helper function to auto-generate the output padding given the attributes for certain ONNX nodes
void auto_gen_input_output_padding(nvinfer1::Dims input_dims, nvinfer1::Dims output_shape, nvinfer1::Dims kernel_size,
    nvinfer1::Dims strides, nvinfer1::Dims dilations, const int nbSpatialDims, nvinfer1::Dims& beg_padding,
    nvinfer1::Dims& end_padding, nvinfer1::Dims& output_padding, nvinfer1::PaddingMode paddingMode);

// Helper function for handling tensor broadcasting for opsets < 7
Status applyLegacyBinaryOpBroadcasting(IImporterContext* ctx,
                                       ::ONNX_NAMESPACE::NodeProto const& node,
                                       TensorOrWeights& lhs,
                                       TensorOrWeights& rhs);

// Helper function to import ArgMin and ArgMax nodes into TRT
NodeImportResult argMinMaxHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::TopKOperation op);

// Helper function to broadcast two tensors to the larger shape
void broadcast_tensors(IImporterContext* ctx, nvinfer1::ITensor*& t1, nvinfer1::ITensor*& t2);

// Helper function for parsing brodacsting attributes
Status check_broadcast_attrs(IImporterContext* ctx, OnnxAttrs const& attrs, nvinfer1::Dims const& dims);

// Helper function to check if node is connected to a grpah input
bool check_for_input(::ONNX_NAMESPACE::NodeProto const& node, std::string const& input_node);

// Helper function to check if node inputs are INT32 type
bool check_for_int32(std::vector<TensorOrWeights>const & inputs);

// Helper function to convert an ONNX axis into a TRT axis (supports negative indexing)
Status convert_axis(int& axis, int nbDims);

// Helper function to convert an ONNX datatype to a TRT datatype with INT64 downcasting
bool convert_dtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert an ONNX datatype to a TRT datatype without INT64 downcasting
bool convert_input_dtype(int32_t onnx_dtype, nvinfer1::DataType* trt_dtype);

// Helper function to convert ONNX weights to TRT weights
bool convert_onnx_weights(::ONNX_NAMESPACE::TensorProto const& onnx_tensor, onnx2trt::ShapedWeights* weights);

// Helper function to convert an weight graph output to tensor
nvinfer1::ITensor& convert_output_weight_to_tensor(TensorOrWeights& input, IImporterContext* ctx);

// Helper function to squeeze a tensor into two dimensions
nvinfer1::ITensor* convert_tensor_to_2d(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis);

// Helper function to convert ONNX weight descriptors to TRT weights
bool convert_weight_descriptor(onnxTensorDescriptorV1 const &desc, onnx2trt::ShapedWeights *weights);

// Helper functinon to convert weights to tensors
nvinfer1::ITensor& convertToTensor(TensorOrWeights& input, IImporterContext* ctx);

// Helper function to convert a 1D shape into a tensor of the same size
nvinfer1::ITensor& dimension_to_tensor(IImporterContext* ctx, nvinfer1::Dims dims);

// Helper function to calculate the ceil division of two integers.
int div_ceil(int n, int d);

// Helper function to import elementwise nodes into TRT
NodeImportResult elementwiseHelper(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::ElementWiseOperation binary_op,
    bool legacy_binary_op_broadcasting = false);

// Helper functino to flatten a tensor on a specified axis
nvinfer1::ITensor* flatten_tensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, int axis);

// Helper function to import a plugin from TensorRT's plugin registry given the name and version.
nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields);

// Returns false if the transpose does not require any data movement (i.e., it's equivalent to a reshape)
inline bool is_transpose_required(nvinfer1::Dims const& shape, nvinfer1::Permutation const& perm);

// Helper function to calculate the output size of a convolution operation
int get_conv_output_size(int input_size, int filter_size,
                         int stride, int dilation_rate,
                         int total_padding);

// Helper function to get the name of an ONNX data type
const char* get_dtype_name(int32_t onnx_dtype);

// Helper function to get the size of an ONNX data type
int get_dtype_size(int32_t onnx_dtype);

// Helper function to help extract the index of a potential -1 dimension in the reshape node
Status get_infer_dim(int& infer_dim, nvinfer1::Dims const& new_shape);

// Helper function to extract kernel parameters given a node's attributes
void get_kernel_params(::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::Dims* kernel_size,
                       nvinfer1::Dims* strides,
                       nvinfer1::Dims* beg_padding,
                       nvinfer1::Dims* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       bool& count_exclude_padding,
                       nvinfer1::Dims* dilations=nullptr,
                       nvinfer1::Dims* output_padding=nullptr);

// Helper function to get the scale mode for TRT's scale layer given the shapes of the inputs
nvinfer1::ScaleMode get_scale_mode(nvinfer1::Dims const& weights_shape, nvinfer1::Dims const& tensor_shape);

// Helper function to create a Dims object with the specified number of dims and value
nvinfer1::Dims makeDims(int nbDims, int val);

// Helper function to reshape a tensor to a specified size
nvinfer1::ITensor* reshape_tensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Dims shape);

// Helper function to convert ONNX node to a scale layer
NodeImportResult scaleHelper(IImporterContext* ctx,
                               ::ONNX_NAMESPACE::NodeProto const& node,
                               std::vector<TensorOrWeights>& inputs,
                               ScaleOp op);

// Helper function to set ONNX node attributes
void setAttr(nvinfer1::Dims * trtAttr, ::ONNX_NAMESPACE::AttributeProto const* onnxAttr, int nbSpatialDims, int defaultVal);

// Helper function to transpose a tensor given a permutation
nvinfer1::ITensor* transpose_tensor(IImporterContext* ctx, nvinfer1::ITensor& tensor, nvinfer1::Permutation const& perm, 
                                    bool permute_dim_types);

// Helper function for slice
Status slice_array(TensorOrWeights weights, std::vector<int32_t>& weight_vector);

// Helper function to import unary operations into TRT
NodeImportResult unaryHelper(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node,
    std::vector<TensorOrWeights>& inputs, nvinfer1::UnaryOperation op);

// Helper function to update padding values.
void update_padded_values(std::vector<float>&pad_values, const nvinfer1::DimsHW beg_padding,
  const nvinfer1::DimsHW end_padding, const nvinfer1::Dims padded_shape, const float pad_value);

} // namespace onnx2trt
