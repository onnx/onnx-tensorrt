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
#include "OnnxAttrs.hpp"
#include "onnx2trt_utils.hpp"
#include "onnx_utils.hpp"
#include "toposort.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <limits>
#include <unordered_set>

namespace onnx2trt
{

Status setTensorLocations(
    IImporterContext* ctx, const std::vector<std::string>& tensors, const std::vector<std::string>& locations)
{
    ASSERT(tensors.size() >= locations.size(), nvonnxparser::ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < locations.size(); ++i)
    {
        std::string tensor = tensors.at(i);
        std::string location = locations.at(i);
        nvinfer1::TensorLocation loc
            = location == "device" ? nvinfer1::TensorLocation::kDEVICE : nvinfer1::TensorLocation::kHOST;

        if (ctx->tensorLocations().count(tensor) > 0)
        {
            ASSERT(ctx->tensorLocations()[tensor] == loc, nvonnxparser::ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            ctx->tensorLocations()[tensor] = loc;
        }
    }

    return Status::success();
}

template <typename T>
Status setStringMap(
    IImporterContext* ctx, const std::vector<std::string>& tensors, const std::vector<T>& data, string_map<T>& map)
{
    ASSERT(tensors.size() >= data.size(), nvonnxparser::ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < data.size(); ++i)
    {
        std::string name = tensors.at(i);
        T dataName = data.at(i);
        if (map.count(name) > 0)
        {
            ASSERT(map[name] == dataName, nvonnxparser::ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            map[name] = dataName;
        }
    }
    return Status::success();
}

Status parseGraph(
    IImporterContext* ctx, const ::ONNX_NAMESPACE::GraphProto& graph, bool deserializingINetwork, int* currentNode)
{
    // Import initializers.
    for (const ::ONNX_NAMESPACE::TensorProto& initializer : graph.initializer())
    {
        LOG_VERBOSE("Importing initializer: " << initializer.name());
        ShapedWeights weights;
        ASSERT(convertOnnxWeights(initializer, &weights, ctx), ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerTensor(TensorOrWeights{std::move(weights)}, initializer.name());
    }

    std::vector<size_t> topoOrder;
    ASSERT(toposort(graph.node(), &topoOrder), ErrorCode::kINVALID_GRAPH);

    const string_map<NodeImporter>& opImporters = getBuiltinOpImporterMap();
    for (const auto& nodeIndex : topoOrder)
    {
        if (currentNode)
        {
            *currentNode = nodeIndex;
        }
        const auto& node = graph.node(nodeIndex);
        LOG_VERBOSE("Parsing node: " << node.name() << " [" << node.op_type() << "]");

        // Assemble node inputs. These may come from outside the subgraph.
        std::vector<TensorOrWeights> nodeInputs;
        std::stringstream ssInputs{};
        ssInputs << node.name() << " [" << node.op_type() << "] inputs: ";
        for (const auto& inputName : node.input())
        {
            // Empty input names indicate optional inputs which have not been supplied.
            if (inputName.empty())
            {
                nodeInputs.emplace_back(nullptr);
                ssInputs << "[optional input, not set], ";
            }
            else
            {
                LOG_VERBOSE("Searching for input: " << inputName);
                ASSERT(ctx->tensors().count(inputName), ErrorCode::kINVALID_GRAPH);
                nodeInputs.push_back(ctx->tensors().at(inputName));
                ssInputs << "[" << inputName << " -> " << nodeInputs.back().shape() << "], ";
            }
        }
        LOG_VERBOSE(ssInputs.str());

        // Dispatch to appropriate converter.
        if (!opImporters.count(node.op_type()))
        {
            return MAKE_ERROR("No importer registered for op: " + node.op_type(), ErrorCode::kUNSUPPORTED_NODE);
        }
        const NodeImporter& importFunc = opImporters.at(node.op_type());
        std::vector<TensorOrWeights> outputs;

        const int nbLayersBefore = ctx->network()->getNbLayers();
        GET_VALUE(importFunc(ctx, node, nodeInputs), &outputs);

        ctx->registerLayer(ctx->network()->getLayer(nbLayersBefore), node.name());

        if (deserializingINetwork)
        {
            OnnxAttrs attrs(node, ctx);

            // Tensor locations, dynamic ranges and layer precisions will be set after parsing the network
            std::vector<std::string> outputsLocation = attrs.get<std::vector<std::string>>("trt_outputs_loc", {});
            std::vector<std::string> outputsVec(node.output().begin(), node.output().end());
            std::vector<std::string> layerName{node.name()};
            TRT_CHECK(setTensorLocations(ctx, outputsVec, outputsLocation));

            auto outputsRangeMin = attrs.get<std::vector<float>>("trt_outputs_range_min", {});
            TRT_CHECK(setStringMap<float>(ctx, outputsVec, outputsRangeMin, ctx->tensorRangeMins()));
            auto outputsRangeMax = attrs.get<std::vector<float>>("trt_outputs_range_max", {});
            TRT_CHECK(setStringMap<float>(ctx, outputsVec, outputsRangeMax, ctx->tensorRangeMaxes()));

            if (attrs.count("trt_layer_precision"))
            {
                std::vector<nvinfer1::DataType> layerPrecision{attrs.get<nvinfer1::DataType>("trt_layer_precision")};
                TRT_CHECK(setStringMap<nvinfer1::DataType>(ctx, layerName, layerPrecision, ctx->layerPrecisions()));
            }
        }

        // Set output names and register outputs with the context.
        std::stringstream ssOutputs{};
        ssOutputs << node.name() << " [" << node.op_type() << "] outputs: ";
        for (int i = 0; i < node.output().size(); ++i)
        {
            const auto& outputName = node.output(i);
            auto& output = outputs.at(i);
            ssOutputs << "[" << outputName << " -> " << output.shape() << "], ";
            // Note: This condition is to allow ONNX outputs to be ignored
            // Always register output weights (even empty ones) as it may be mapped to an unused input
            if ((output || output.is_weights()) && !outputName.empty())
            {
                ctx->registerTensor(std::move(output), outputName);
            }
        }
        LOG_VERBOSE(ssOutputs.str());
    }
    return Status::success();
}

Status importInput(ImporterContext* ctx, ::ONNX_NAMESPACE::ValueInfoProto const& input, nvinfer1::ITensor** tensor)
{
    auto const& onnxDtype = input.type().tensor_type();
    nvinfer1::DataType trtDtype;
    ASSERT_INPUT(convertDtype(onnxDtype.elem_type(), &trtDtype), ErrorCode::kUNSUPPORTED_NODE, input.name());
    nvinfer1::Dims trt_dims;
    ASSERT_INPUT(convertOnnxDims(onnxDtype.shape().dim(), trt_dims), ErrorCode::kUNSUPPORTED_GRAPH, input.name());
    nvinfer1::ITensor* userInput = ctx->getUserInput(input.name().c_str());
    if (userInput)
    {
        ASSERT_INPUT(userInput, ErrorCode::kINVALID_VALUE, input.name());
        // Note: We intentionally don't check dimensions/dtype here so that users can change the input shape/type if
        // they want to.
        *tensor = userInput;
        return Status::success();
    }

    LOG_VERBOSE(
        "Adding network input: " << input.name() << " with dtype: " << trtDtype << ", dimensions: " << trt_dims);
    ASSERT_INPUT(*tensor = ctx->network()->addInput(input.name().c_str(), trtDtype, trt_dims),
        ErrorCode::kUNSUPPORTED_NODE, input.name());
    return Status::success();
}

Status importInputs(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph,
    string_map<TensorOrWeights>* tensors, uint32_t weights_count, onnxTensorDescriptorV1 const* weight_descriptors)
{
    // The weights may come from two sources:
    // either Initializer list in onnx graph
    // or User specified weight through onnxifi
    ASSERT(weights_count == 0 || weight_descriptors, ErrorCode::kINVALID_VALUE);
    string_map<onnxTensorDescriptorV1 const*> weight_map;

    for (uint32_t i = 0; i < weights_count; ++i)
    {
        onnxTensorDescriptorV1 const* desc = weight_descriptors + i;
        ASSERT(weight_map.emplace(desc->name, desc).second, ErrorCode::kINVALID_VALUE);
    }

    // Initializers are not really network inputs, so they need to be excluded.
    std::unordered_set<std::string> initializers{};
    for (const ::ONNX_NAMESPACE::TensorProto& initializer : graph.initializer())
    {
        initializers.emplace(initializer.name());
    }

    for (const ::ONNX_NAMESPACE::ValueInfoProto& input : graph.input())
    {
        TensorOrWeights tensor;
        if (weight_map.count(input.name()))
        {
            const onnxTensorDescriptorV1& weight_desc = *weight_map.at(input.name());
            ShapedWeights weights;
            // We only support grabbing weight from CPU memory now
            ASSERT(weight_desc.memoryType == ONNXIFI_MEMORY_TYPE_CPU, ErrorCode::kINVALID_VALUE);
            ASSERT(convertWeightDescriptor(weight_desc, &weights, ctx), ErrorCode::kUNSUPPORTED_NODE);
            tensor = weights;
        }
        else if (!initializers.count(input.name()))
        {
            nvinfer1::ITensor* tensor_ptr;
            TRT_CHECK(importInput(ctx, input, &tensor_ptr));
            tensor = tensor_ptr;
        }
        ctx->registerTensor(std::move(tensor), input.name());
    }

    return Status::success();
}

Status deserialize_onnx_model(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    bool is_serialized_as_text, ::ONNX_NAMESPACE::ModelProto* model)
{
    google::protobuf::io::ArrayInputStream raw_input(serialized_onnx_model, serialized_onnx_model_size);
    if (is_serialized_as_text)
    {
        ASSERT(google::protobuf::TextFormat::Parse(&raw_input, model), ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    else
    {
        google::protobuf::io::CodedInputStream coded_input(&raw_input);
        // Note: This WARs the very low default size limit (64MB)
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
        ASSERT(model->ParseFromCodedStream(&coded_input), ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    return Status::success();
}

Status deserialize_onnx_model(int fd, bool is_serialized_as_text, ::ONNX_NAMESPACE::ModelProto* model)
{
    google::protobuf::io::FileInputStream raw_input(fd);
    if (is_serialized_as_text)
    {
        ASSERT(google::protobuf::TextFormat::Parse(&raw_input, model), ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    else
    {
        google::protobuf::io::CodedInputStream coded_input(&raw_input);
        // Note: This WARs the very low default size limit (64MB)
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
        ASSERT(model->ParseFromCodedStream(&coded_input), ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    return Status::success();
}

bool ModelImporter::supportsModel(
    void const* serialized_onnx_model, size_t serialized_onnx_model_size, SubGraphCollection_t& sub_graph_collection)
{

    ::ONNX_NAMESPACE::ModelProto model;
    bool is_serialized_as_text = false;
    Status status
        = deserialize_onnx_model(serialized_onnx_model, serialized_onnx_model_size, is_serialized_as_text, &model);

    if (status.is_error())
    {
        _errors.push_back(status);
        return false;
    }

    bool allSupported{true};

    // Parse the graph and see if we hit any parsing errors
    allSupported = parse(serialized_onnx_model, serialized_onnx_model_size);

    int error_node = -1;
    std::string input_node{};

    if (!allSupported)
    {
        int nerror = getNbErrors();
        for (int i = 0; i < nerror; ++i)
        {
            nvonnxparser::IParserError const* error = getError(i);
            if (error->node() != -1)
            {
                error_node = error->node();
                allSupported = false;
            }
            // The node that we failed on is one of the input nodes (-1). Get the name of the input node
            // that we failed on and remove all nodes that spawn out of it.
            else
            {
                // Node name is extracted through error->file as all errors thrown on input nodes are wrapped
                // around MAKE_INPUT_ERROR.
                input_node = error->file();
            }
        }
    }

    auto checkForInput = [&input_node](::ONNX_NAMESPACE::NodeProto const& node) {
        for (auto input : node.input())
        {
            if (input_node == input)
            {
                return true;
            }
        }
        return false;
    };

    bool newSubGraph(true);
    // Sort and partition supported subgraphs
    std::vector<size_t> topological_order;
    if (!toposort(model.graph().node(), &topological_order))
    {
        cout << "Failed to sort model topologically, exiting ..." << endl;
        return false;
    }

    auto* ctx = &_importer_ctx;

    for (int node_idx : topological_order)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = model.graph().node(node_idx);

        // Add the node to the subgraph if:
        //     1. Importer function is registered for the operator type
        //     2. It is NOT directly connected to an unsupported input
        //     3. Parsing did NOT hit an error on the node
        //     4. Any shape tensor output is coming from a supported node
        bool registered = supportsOperator(node.op_type().c_str());
        bool containsInput = (input_node.empty()) ? false : checkForInput(node);
        bool containsIndex = node_idx == error_node;
        auto const tensor = node.output()[0];
        bool supportedShapeTensorOutput = ctx->unsupportedShapeTensors().count(tensor) == 0 ? true : false;

        if (registered && !containsInput && !containsIndex && supportedShapeTensorOutput)
        {
            if (newSubGraph)
            {
                // If it is the beginning of a new subGraph, we start a new vector
                sub_graph_collection.emplace_back();
                // Mark all new graphs as "unknown"
                sub_graph_collection.back().second = false;
                newSubGraph = false;
            }
            // We add the new node to the last graph
            sub_graph_collection.back().first.emplace_back(node_idx);
        }
        else
        {
            // This is not a supported node, reset newSubGraph
            newSubGraph = true;
            allSupported = false;
        }
    }

    // Only mark the subgraph as supported if there is one supported subgraph.
    if (allSupported)
    {
        sub_graph_collection.back().second = true;
    }
    return allSupported;
}

bool ModelImporter::supportsOperator(const char* op_name) const
{
    return _op_importers.count(op_name);
}

bool ModelImporter::parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    uint32_t weight_count, onnxTensorDescriptorV1 const* weight_descriptors)
{
    _current_node = -1;
    // TODO: This function (and its overload below) could do with some cleaning,
    //       particularly wrt error handling.
    // Note: We store a copy of the model so that weight arrays will persist
    _onnx_models.emplace_back();
    ::ONNX_NAMESPACE::ModelProto& model = _onnx_models.back();
    bool is_serialized_as_text = false;
    Status status
        = deserialize_onnx_model(serialized_onnx_model, serialized_onnx_model_size, is_serialized_as_text, &model);
    if (status.is_error())
    {
        _errors.push_back(status);
        return false;
    }
    status = this->importModel(model, weight_count, weight_descriptors);
    if (status.is_error())
    {
        status.setNode(_current_node);
        _errors.push_back(status);
        return false;
    }
    return true;
}

bool ModelImporter::parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size)
{
    return this->parseWithWeightDescriptors(serialized_onnx_model, serialized_onnx_model_size, 0, nullptr);
}

void removeShapeTensorCasts(IImporterContext* ctx)
{
    // Removes any casts on shape tensors, as TensorRT does not support them.
    for (int i = 0, e = ctx->network()->getNbLayers(); i < e; ++i)
    {
        nvinfer1::ILayer* layer = ctx->network()->getLayer(i);
        constexpr nvinfer1::DataType SHAPE_TENSOR_TYPE = nvinfer1::DataType::kINT32;
        if (layer->getNbOutputs() > 0 && layer->getOutput(0)->isShapeTensor())
        {
            layer->resetPrecision();
            layer->resetOutputType(0);
            layer->setPrecision(SHAPE_TENSOR_TYPE);
            layer->setOutputType(0, SHAPE_TENSOR_TYPE);
            nvinfer1::ITensor& t = *layer->getOutput(0);
            // Set type only if necessary, to avoid TensorRT warnings
            // about setting type of non-input/output tensors.
            if (t.getType() != SHAPE_TENSOR_TYPE)
            {
                t.setType(SHAPE_TENSOR_TYPE);
            }
            // Some layers do not support shape tensor outputs. Keep track of these tensor names
            // for supportsModel().
            auto type = layer->getType();
            auto elementwiseOp = layer->getType() == nvinfer1::LayerType::kELEMENTWISE ? (static_cast<nvinfer1::IElementWiseLayer*>(layer))->getOperation() : nvinfer1::ElementWiseOperation::kSUM;
            auto reduceOp = layer->getType() == nvinfer1::LayerType::kREDUCE ? (static_cast<nvinfer1::IReduceLayer*>(layer))->getOperation() : nvinfer1::ReduceOperation::kSUM;

            if (!supportsShapeTensor(type, elementwiseOp, reduceOp))
            {
                auto name = layer->getOutput(0)->getName();
                ctx->unsupportedShapeTensors().insert(name);
                LOG_ERROR("Found " << name << " as a shape tensor output from a layer that does not support it!");
            }
        }
    }
}

Status ModelImporter::importModel(
    ::ONNX_NAMESPACE::ModelProto const& model, uint32_t weight_count, onnxTensorDescriptorV1 const* weight_descriptors)
{
    ASSERT(!_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure the network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag.", ErrorCode::kINVALID_VALUE);
    auto* ctx = &_importer_ctx;
    _importer_ctx.clearOpsets();
    // Initialize plugin registry
    initLibNvInferPlugins(static_cast<void*>(&ctx->logger()), "ONNXTRT_NAMESPACE");
    for (int i = 0; i < model.opset_import().size(); ++i)
    {
        std::string domain = model.opset_import(i).domain();
        int64_t version = model.opset_import(i).version();
        // TensorRT requires an ONNX graph to be generated with at least ai.onnx version 7.
        // ONNX spec says that the default domain is either an empty string or is "ai.onnx".
        if ((domain.empty() || domain == "ai.onnx") && version < 7)
        {
            LOG_WARNING(
                "TensorRT supports ONNX graphs generated with at least opset 7. Models using older opsets are not "
                "guaranteed to work.");
        }
        _importer_ctx.addOpset(domain, version);
    }
    ::ONNX_NAMESPACE::GraphProto const& graph = model.graph();
    // Create a dummy tensors so that we can reserve output names. If the output names are encountered elsewhere
    // in the graph, the ctx will know to make the names unique.
    for (const ::ONNX_NAMESPACE::ValueInfoProto& output : graph.output())
    {
        _importer_ctx.registerTensor(TensorOrWeights{}, output.name());
    }

    _current_node = -1;
    TRT_CHECK(importInputs(&_importer_ctx, graph, &_importer_ctx.tensors(), weight_count, weight_descriptors));
    TRT_CHECK(parseGraph(&_importer_ctx, graph, model.producer_name() == "TensorRT", &_current_node));

    // Mark outputs defined in the ONNX model (unless tensors are user-requested)
    for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
    {
        ASSERT(_importer_ctx.tensors().count(output.name()), ErrorCode::kINVALID_GRAPH);
        ASSERT(_importer_ctx.tensors().at(output.name()).is_tensor(), ErrorCode::kUNSUPPORTED_GRAPH);
        nvinfer1::ITensor* output_tensor_ptr = &_importer_ctx.tensors().at(output.name()).tensor();
        LOG_VERBOSE("Marking " << output_tensor_ptr->getName() << " as output: " << output.name());
        output_tensor_ptr->setName(output.name().c_str());
        if (output_tensor_ptr->isNetworkInput())
        {
            // HACK WAR for TRT not allowing input == output
            // TODO: Does this break things by changing the name of the input tensor?
            output_tensor_ptr->setName(("__" + output.name()).c_str());
            output_tensor_ptr = &identity(&_importer_ctx, output_tensor_ptr).tensor();
            ASSERT(output_tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
            output_tensor_ptr->setName(output.name().c_str());
        }
        nvinfer1::ITensor** user_output = _importer_ctx.getUserOutput(output.name().c_str());
        if (!user_output)
        {
            _importer_ctx.network()->markOutput(*output_tensor_ptr);
            nvinfer1::DataType output_trt_dtype;
            ASSERT(
                convertDtype(output.type().tensor_type().elem_type(), &output_trt_dtype), ErrorCode::kUNSUPPORTED_NODE);
            // For INT32 data type, output type must match tensor type
            ASSERT(output_tensor_ptr->getType() != nvinfer1::DataType::kINT32
                    || output_trt_dtype == nvinfer1::DataType::kINT32,
                ErrorCode::kUNSUPPORTED_NODE);
            // Note: Without this, output type is always float32
            output_tensor_ptr->setType(output_trt_dtype);
        }
    }
    // Return user-requested output tensors
    for (auto user_output_entry : _importer_ctx.getUserOutputs())
    {
        std::string user_output_name = user_output_entry.first;
        nvinfer1::ITensor** user_output_ptr = user_output_entry.second;
        ASSERT(_importer_ctx.tensors().count(user_output_name), ErrorCode::kINVALID_VALUE);
        TensorOrWeights user_output = _importer_ctx.tensors().at(user_output_name);
        ASSERT(user_output.is_tensor(), ErrorCode::kINVALID_VALUE);
        *user_output_ptr = &user_output.tensor();
    }

    if (model.producer_name() == "TensorRT")
    {
        // iterate over all tensors in the network and add them to "tensors" map
        string_map<nvinfer1::ITensor*> tensors;
        string_map<nvinfer1::ILayer*> layers;
        for (int idx = 0; idx < _importer_ctx.network()->getNbInputs(); ++idx)
        {
            nvinfer1::ITensor* tensor = _importer_ctx.network()->getInput(idx);
            if (tensor != nullptr)
            {
                tensors[tensor->getName()] = tensor;
            }
        }
        for (int idx = 0; idx < _importer_ctx.network()->getNbOutputs(); ++idx)
        {
            nvinfer1::ITensor* tensor = _importer_ctx.network()->getOutput(idx);
            if (tensor != nullptr)
            {
                tensors[tensor->getName()] = tensor;
            }
        }
        for (int layerIdx = 0; layerIdx < _importer_ctx.network()->getNbLayers(); ++layerIdx)
        {
            nvinfer1::ILayer* layer = _importer_ctx.network()->getLayer(layerIdx);
            for (int idx = 0; idx < layer->getNbInputs(); ++idx)
            {
                nvinfer1::ITensor* tensor = layer->getInput(idx);
                if (tensor != nullptr)
                {
                    tensors[tensor->getName()] = tensor;
                }
            }
            for (int idx = 0; idx < layer->getNbOutputs(); ++idx)
            {
                nvinfer1::ITensor* tensor = layer->getOutput(idx);
                if (tensor != nullptr)
                {
                    tensors[tensor->getName()] = tensor;
                }
            }
            layers[layer->getName()] = layer;
        }

        // Set locations for all tensors
        for (auto const& tensor : ctx->tensorLocations())
        {
            ASSERT(tensors.count(tensor.first) > 0, nvonnxparser::ErrorCode::kINVALID_GRAPH);
            tensors.at(tensor.first)->setLocation(tensor.second);
        }
        // Set dynamic range for all tensors
        for (auto const& tensor : ctx->tensorRangeMins())
        {
            // if there's a min range, there must be a max range as well
            ASSERT(tensors.count(tensor.first) > 0, nvonnxparser::ErrorCode::kINVALID_GRAPH);
            if (!std::isnan(tensor.second))
            {
                tensors.at(tensor.first)->setDynamicRange(tensor.second, ctx->tensorRangeMaxes().at(tensor.first));
            }
        }
        // Set precisions for all layers
        for (auto const& layer : ctx->layerPrecisions())
        {
            ASSERT(layers.count(layer.first) > 0, nvonnxparser::ErrorCode::kINVALID_GRAPH);
            layers.at(layer.first)->setPrecision(layer.second);
        }
    }

    removeShapeTensorCasts(ctx);
    return Status::success();
}

bool ModelImporter::parseFromFile(const char* onnxModelFile, int verbosity)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    ::ONNX_NAMESPACE::ModelProto onnx_model;

    bool is_binary = ParseFromFile_WAR(&onnx_model, onnxModelFile);
    if (!is_binary && !ParseFromTextFile(&onnx_model, onnxModelFile))
    {
        cerr << "Failed to parse ONNX model from file" << onnxModelFile << endl;
        return EXIT_FAILURE;
    }

    if (verbosity >= (int) nvinfer1::ILogger::Severity::kWARNING)
    {
        int64_t opset_version = (onnx_model.opset_import().size() ? onnx_model.opset_import(0).version() : 0);
        cout << "----------------------------------------------------------------" << endl;
        cout << "Input filename:   " << onnxModelFile << endl;
        cout << "ONNX IR version:  " << onnx_ir_version_string(onnx_model.ir_version()) << endl;
        cout << "Opset version:    " << opset_version << endl;
        cout << "Producer name:    " << onnx_model.producer_name() << endl;
        cout << "Producer version: " << onnx_model.producer_version() << endl;
        cout << "Domain:           " << onnx_model.domain() << endl;
        cout << "Model version:    " << onnx_model.model_version() << endl;
        cout << "Doc string:       " << onnx_model.doc_string() << endl;
        cout << "----------------------------------------------------------------" << endl;
    }

    { //...Read input file, parse it
        std::ifstream onnx_file(onnxModelFile, std::ios::binary | std::ios::ate);
        std::streamsize file_size = onnx_file.tellg();
        onnx_file.seekg(0, std::ios::beg);
        std::vector<char> onnx_buf(file_size);
        if (!onnx_file.read(onnx_buf.data(), onnx_buf.size()))
        {
            cerr << "ERROR: Failed to read from file " << onnxModelFile << endl;
            return false;
        }
        if (!parse(onnx_buf.data(), onnx_buf.size()))
        {
            int nerror = getNbErrors();
            for (int i = 0; i < nerror; ++i)
            {
                nvonnxparser::IParserError const* error = getError(i);
                if (error->node() != -1)
                {
                    ::ONNX_NAMESPACE::NodeProto const& node = onnx_model.graph().node(error->node());
                    cerr << "While parsing node number " << error->node() << " [" << node.op_type();
                    if (node.output().size() && verbosity >= (int) nvinfer1::ILogger::Severity::kVERBOSE)
                    {
                        cerr << " -> \"" << node.output(0) << "\"";
                    }
                    cerr << "]:" << endl;
                    if (verbosity >= (int) nvinfer1::ILogger::Severity::kVERBOSE)
                    {
                        cout << "--- Begin node ---" << endl;
                        cout << node << endl;
                        cout << "--- End node ---" << endl;
                    }
                }
                cerr << "ERROR: " << error->file() << ":" << error->line() << " In function " << error->func() << ":\n"
                     << "[" << static_cast<int>(error->code()) << "] " << error->desc() << endl;
            }
            return false;
        }

        if (verbosity >= (int) nvinfer1::ILogger::Severity::kVERBOSE)
        {
            cout << " ----- Parsing of ONNX model " << onnxModelFile << " is Done ---- " << endl;
        }
    } //...End Reading input file, parsing it
    return true;
}

} // namespace onnx2trt
