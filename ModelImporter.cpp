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

#include "ModelImporter.hpp"
#include "toposort.hpp"
#include "onnx_utils.hpp"
#include "onnx2trt_utils.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <limits>

namespace onnx2trt {

//Status const& ModelImporter::setInput(const char* name, nvinfer1::ITensor* input) {
//  _importer_ctx.setUserInput(name, input);
//  _last_error = Status::success();
//  return _last_error;
//}
//
//Status const& ModelImporter::setOutput(const char* name, nvinfer1::ITensor** output) {
//  _importer_ctx.setUserOutput(name, output);
//  _last_error = Status::success();
//  return _last_error;
//}

Status importInput(ImporterContext* importer_ctx,
                   ::ONNX_NAMESPACE::ValueInfoProto const& input,
                   nvinfer1::ITensor** tensor) {
  auto const& onnx_tensor_type = input.type().tensor_type();
  nvinfer1::DataType trt_dtype;
  ASSERT(convert_dtype(onnx_tensor_type.elem_type(), &trt_dtype),
         ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(onnx_tensor_type.shape().dim().size() > 0,
         ErrorCode::kUNSUPPORTED_NODE);
  auto trt_dims = convert_dims(onnx_tensor_type.shape().dim());
  nvinfer1::ITensor* user_input = importer_ctx->getUserInput(input.name().c_str());
  if( user_input ) {
    ASSERT(user_input, ErrorCode::kINVALID_VALUE);
    // Note: We intentionally don't check dimensions/dtype here so that users
    //       can change the input shape/type if they want to.
    //ASSERT(trt_dims  == user_input->getDimensions(), ErrorCode::kINVALID_VALUE);
    //ASSERT(trt_dtype == user_input->getType(),       ErrorCode::kINVALID_VALUE);
    *tensor = user_input;
    return Status::success();
  }
#if NV_TENSORRT_MAJOR < 4
  // WAR for TRT not supporting < 3 input dims
  for( int i=trt_dims.nbDims; i<3; ++i ) {
    // Pad with unitary dims
    ++trt_dims.nbDims;
    trt_dims.d[i] = 1;
    trt_dims.type[i] = (i == 0 ?
                        nvinfer1::DimensionType::kCHANNEL :
                        nvinfer1::DimensionType::kSPATIAL);
  }
  ASSERT(trt_dims.nbDims <= 3, ErrorCode::kUNSUPPORTED_NODE);
#endif // NV_TENSORRT_MAJOR < 4
  ASSERT(*tensor = importer_ctx->network()->addInput(
           input.name().c_str(), trt_dtype, trt_dims),
         ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

Status importInputs(ImporterContext* importer_ctx,
                    ::ONNX_NAMESPACE::GraphProto const& graph,
                    string_map<TensorOrWeights>* tensors,
                    uint32_t weights_count,
                    onnxTensorDescriptorV1 const* weight_descriptors) {
  // The weights may come from two sources:
  // either Initializer list in onnx graph
  // or User specified weight through onnxifi
  string_map<::ONNX_NAMESPACE::TensorProto const*> initializer_map;
  for( ::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer() ) {
    ASSERT(!initializer_map.count(initializer.name()), ErrorCode::kINVALID_GRAPH);
    initializer_map.insert({initializer.name(), &initializer});
  }
  ASSERT(weights_count == 0 || initializer_map.empty(),
         ErrorCode::kINVALID_VALUE);
  ASSERT(weights_count == 0 || weight_descriptors, ErrorCode::kINVALID_VALUE);
  string_map<onnxTensorDescriptorV1 const*> weight_map;
  for (uint32_t i = 0; i < weights_count; ++i) {
    onnxTensorDescriptorV1 const* desc = weight_descriptors + i;
    ASSERT(weight_map.emplace(desc->name, desc).second,
           ErrorCode::kINVALID_VALUE);
  }
  for( ::ONNX_NAMESPACE::ValueInfoProto const& input : graph.input() ) {
    TensorOrWeights tensor;
    if( initializer_map.count(input.name()) ) {
      ::ONNX_NAMESPACE::TensorProto const& initializer = *initializer_map.at(input.name());
      ShapedWeights weights;
      ASSERT(convert_onnx_weights(initializer, &weights),
             ErrorCode::kUNSUPPORTED_NODE);
      tensor = weights;
    } else if (weight_map.count(input.name())) {
      onnxTensorDescriptorV1 const& weight_desc = *weight_map.at(input.name());
      ShapedWeights weights;
      // We only support grabbing weight from CPU memory now
      ASSERT(weight_desc.memoryType == ONNXIFI_MEMORY_TYPE_CPU,
             ErrorCode::kINVALID_VALUE);

      ASSERT(convert_weight_descriptor(weight_desc, &weights),
             ErrorCode::kUNSUPPORTED_NODE);
      tensor = weights;
    } else {
      nvinfer1::ITensor* tensor_ptr;
      TRT_CHECK(importInput(importer_ctx, input, &tensor_ptr));
      tensor = tensor_ptr;
    }
    ASSERT(!tensors->count(input.name()), ErrorCode::kINVALID_GRAPH);
    tensors->insert({input.name(), tensor});
  }
  return Status::success();
}

NodeImportResult ModelImporter::importNode(::ONNX_NAMESPACE::NodeProto const& node,
                                           std::vector<TensorOrWeights>& inputs) {
  if( !_op_importers.count(node.op_type()) ) {
    return MAKE_ERROR("No importer registered for op: " + node.op_type(),
                      ErrorCode::kUNSUPPORTED_NODE);
  }
  NodeImporter const& node_importer = _op_importers.at(node.op_type());

  std::vector<TensorOrWeights> outputs;
  GET_VALUE(node_importer(&_importer_ctx, node, inputs), &outputs);
  ASSERT(outputs.size() <= (size_t)node.output().size(), ErrorCode::kINTERNAL_ERROR);
  for( size_t i=0; i<outputs.size(); ++i ) {
    std::string node_output_name = node.output(i);
    TensorOrWeights& output = outputs.at(i);
    if( output ) {
      if( output.is_tensor() ) {
        output.tensor().setName(node_output_name.c_str());
      }
      //// TODO: Remove when done testing
      //cout << "Imported " << node.op_type()
      //     << " output tensor '" << node_output_name;
      //if( output.is_tensor() ) {
      //  cout << "' (" << output.tensor().getType()  << ")";
      //}
      //cout << endl;
    }
  }
  return outputs;
}

Status deserialize_onnx_model(void const* serialized_onnx_model,
                              size_t      serialized_onnx_model_size,
                              bool is_serialized_as_text,
                              ::ONNX_NAMESPACE::ModelProto* model) {
  google::protobuf::io::ArrayInputStream raw_input(serialized_onnx_model,
                                                   serialized_onnx_model_size);
  if( is_serialized_as_text ) {
    ASSERT(google::protobuf::TextFormat::Parse(&raw_input, model),
           ErrorCode::kMODEL_DESERIALIZE_FAILED);
  } else {
    google::protobuf::io::CodedInputStream coded_input(&raw_input);
    // Note: This WARs the very low default size limit (64MB)
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max() / 4);
    ASSERT(model->ParseFromCodedStream(&coded_input),
           ErrorCode::kMODEL_DESERIALIZE_FAILED);
  }
  return Status::success();
}

Status deserialize_onnx_model(int fd,
                              bool is_serialized_as_text,
                              ::ONNX_NAMESPACE::ModelProto* model) {
  google::protobuf::io::FileInputStream raw_input(fd);
  if( is_serialized_as_text ) {
    ASSERT(google::protobuf::TextFormat::Parse(&raw_input, model),
           ErrorCode::kMODEL_DESERIALIZE_FAILED);
  } else {
    google::protobuf::io::CodedInputStream coded_input(&raw_input);
    // Note: This WARs the very low default size limit (64MB)
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max()/4);
    ASSERT(model->ParseFromCodedStream(&coded_input),
           ErrorCode::kMODEL_DESERIALIZE_FAILED);
  }
  return Status::success();
}

bool ModelImporter::supportsModel(void const *serialized_onnx_model,
                                  size_t serialized_onnx_model_size) {
  ::ONNX_NAMESPACE::ModelProto model;

  bool is_serialized_as_text = false;
  Status status =
      deserialize_onnx_model(serialized_onnx_model, serialized_onnx_model_size,
                             is_serialized_as_text, &model);
  if (status.is_error()) {
    _errors.push_back(status);
    return false;
  }
  for (const auto &node : model.graph().node()) {
    if (!this->supportsOperator(node.op_type().c_str())) {
      return false;
    }
  }
  return true;
}

bool ModelImporter::supportsOperator(const char* op_name) const {
  return _op_importers.count(op_name);
}

bool ModelImporter::parseWithWeightDescriptors(
    void const *serialized_onnx_model, size_t serialized_onnx_model_size,
    uint32_t weight_count, onnxTensorDescriptorV1 const *weight_descriptors) {
  _current_node = -1;
  // TODO: This function (and its overload below) could do with some cleaning,
  //       particularly wrt error handling.
  // Note: We store a copy of the model so that weight arrays will persist
  _onnx_models.emplace_back();
  ::ONNX_NAMESPACE::ModelProto &model = _onnx_models.back();
  bool is_serialized_as_text = false;
  Status status =
      deserialize_onnx_model(serialized_onnx_model, serialized_onnx_model_size,
                             is_serialized_as_text, &model);
  if (status.is_error()) {
    _errors.push_back(status);
    return false;
  }
  status = this->importModel(model, weight_count, weight_descriptors);
  if (status.is_error()) {
    status.setNode(_current_node);
    _errors.push_back(status);
    return false;
  }
  return true;
}

bool ModelImporter::parse(void const *serialized_onnx_model,
                          size_t serialized_onnx_model_size) {
  return this->parseWithWeightDescriptors(
      serialized_onnx_model, serialized_onnx_model_size, 0, nullptr);
}

Status
ModelImporter::importModel(::ONNX_NAMESPACE::ModelProto const &model,
                           uint32_t weight_count,
                           onnxTensorDescriptorV1 const *weight_descriptors) {
  _importer_ctx.clearOpsets();
  for( int i=0; i<model.opset_import().size(); ++i ) {
    std::string domain  = model.opset_import(i).domain();
    int64_t     version = model.opset_import(i).version();
    _importer_ctx.addOpset(domain, version);
  }
  ::ONNX_NAMESPACE::GraphProto const& graph = model.graph();
  string_map<TensorOrWeights> tensors;
  TRT_CHECK(importInputs(&_importer_ctx, graph, &tensors, weight_count,
                         weight_descriptors));
  std::vector<size_t> topological_order;
  ASSERT(toposort(graph.node(), &topological_order), ErrorCode::kINVALID_GRAPH);
  for( size_t node_idx : topological_order ) {
    _current_node = node_idx;
    ::ONNX_NAMESPACE::NodeProto const& node = graph.node(node_idx);
    std::vector<TensorOrWeights> inputs;
    for( auto const& input_name : node.input() ) {
      ASSERT(tensors.count(input_name), ErrorCode::kINVALID_GRAPH);
      inputs.push_back(tensors.at(input_name));
    }
    std::vector<TensorOrWeights> outputs;
    GET_VALUE(this->importNode(node, inputs), &outputs);
    for( size_t i=0; i<outputs.size(); ++i ) {
      std::string node_output_name = node.output(i);
      TensorOrWeights& output = outputs.at(i);
      // Note: This condition is to allow ONNX outputs to be ignored
      if( output ) {
        ASSERT(!tensors.count(node_output_name), ErrorCode::kINVALID_GRAPH);

        tensors.insert({node_output_name, output});
      }
    }
    if( node.output().size() > 0 ) {
      std::stringstream ss;
      ss << node.output(0) << ":"
         << node.op_type() << " -> "
         << outputs.at(0).shape();
      _importer_ctx.logger().log(
          nvinfer1::ILogger::Severity::kINFO, ss.str().c_str());
    }
  }
  _current_node = -1;
  // Mark outputs defined in the ONNX model (unless tensors are user-requested)
  for( ::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output() ) {
    ASSERT(tensors.count(output.name()), ErrorCode::kINVALID_GRAPH);
    ASSERT(tensors.at(output.name()).is_tensor(), ErrorCode::kUNSUPPORTED_GRAPH);
    nvinfer1::ITensor* output_tensor_ptr = &tensors.at(output.name()).tensor();
    if( output_tensor_ptr->isNetworkInput() ) {
      // HACK WAR for TRT not allowing input == output
      // TODO: Does this break things by changing the name of the input tensor?
      output_tensor_ptr->setName(("__" + output.name()).c_str());
      output_tensor_ptr = &identity(&_importer_ctx, output_tensor_ptr).tensor();
      ASSERT(output_tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
      output_tensor_ptr->setName(output.name().c_str());
    }
    nvinfer1::ITensor** user_output = _importer_ctx.getUserOutput(output.name().c_str());
    if( !user_output ) {
      _importer_ctx.network()->markOutput(*output_tensor_ptr);
      nvinfer1::DataType output_trt_dtype;
      ASSERT(convert_dtype(
                 output.type().tensor_type().elem_type(), &output_trt_dtype),
             ErrorCode::kUNSUPPORTED_NODE);
#if NV_TENSORRT_MAJOR >= 4
      // For INT32 data type, output type must match tensor type
      ASSERT(output_tensor_ptr->getType() != nvinfer1::DataType::kINT32 ||
             output_trt_dtype == nvinfer1::DataType::kINT32,
             ErrorCode::kUNSUPPORTED_NODE);
#endif // NV_TENSORRT_MAJOR >= 4
      // Note: Without this, output type is always float32
      output_tensor_ptr->setType(output_trt_dtype);
    }
  }
  // Return user-requested output tensors
  for( auto user_output_entry : _importer_ctx.getUserOutputs() ) {
    std::string         user_output_name = user_output_entry.first;
    nvinfer1::ITensor** user_output_ptr  = user_output_entry.second;
    ASSERT(tensors.count(user_output_name), ErrorCode::kINVALID_VALUE);
    TensorOrWeights user_output = tensors.at(user_output_name);
    ASSERT(user_output.is_tensor(), ErrorCode::kINVALID_VALUE);
    *user_output_ptr = &user_output.tensor();
  }
  return Status::success();
}

} // namespace onnx2trt
