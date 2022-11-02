/*
 * SPDX-License-Identifier: Apache-2.0
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
#include <functional>
#include <unordered_set>
#include <sys/stat.h>

namespace onnx2trt
{


// Helper for deserializing INetwork
Status setTensorLocations(
    IImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<std::string> const& locations)
{
    ASSERT( (tensors.size() >= locations.size()) && "The size of tensors misaligns with the size of the attribute trt_outputs_loc.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < locations.size(); ++i)
    {
        std::string tensor = tensors.at(i);
        std::string location = locations.at(i);
        nvinfer1::TensorLocation loc
            = location == "device" ? nvinfer1::TensorLocation::kDEVICE : nvinfer1::TensorLocation::kHOST;

        if (ctx->tensorLocations().count(tensor) > 0)
        {
            ASSERT( (ctx->tensorLocations()[tensor] == loc) && "The tensor location cannot be changed.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            ctx->tensorLocations()[tensor] = loc;
        }
    }

    return Status::success();
}

// Helper for deserializing INetwork
template <typename T>
Status setStringMap(
    IImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<T> const& data, string_map<T>& map)
{
    ASSERT((tensors.size() >= data.size())
            && "The size of tensors misaligns with the size of the attribute trt_outputs_range_min/max.",
        nvonnxparser::ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < data.size(); ++i)
    {
        std::string name = tensors.at(i);
        T dataName = data.at(i);
        if (map.count(name) > 0)
        {
            ASSERT( (map[name] == dataName) && "The order of tensorRangeMin/Max in context misaligns with the order of the attribute trt_outputs_range_min/max.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            map[name] = dataName;
        }
    }
    return Status::success();
}

//! Make error explanation from TensorRT error recorder.
static std::string makeErrorExplanation(IImporterContext* ctx,  std::string const& nodeName)
{
    std::ostringstream result;
    result << "Invalid Node - " << nodeName;
    if (auto* errorRecorder = ctx->getErrorRecorder())
    {
        // Append information that might help the user understand the error.
        int32_t const nbErrors = errorRecorder->getNbErrors();
        for (int32_t i = 0; i < nbErrors; ++i)
        {
            result << "\n" << errorRecorder->getErrorDesc(i);
        }
    }
    return result.str();
}

//! Make error explanation from an exception.
static std::string makeErrorExplanation(std::exception const& e, std::string const& nodeName)
{
    std::ostringstream result;
    result << "Invalid Node - " << nodeName << "\n" << e.what();
    return result.str();
}

Status parseGraph(
    IImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, bool deserializingINetwork, int* currentNode)
{
    // Import initializers.
    for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
    {
        LOG_VERBOSE("Importing initializer: " << initializer.name());
        ShapedWeights weights;
        ASSERT(convertOnnxWeights(initializer, &weights, ctx) && "Failed to import initializer.", ErrorCode::kUNSUPPORTED_NODE);
        ctx->registerTensor(TensorOrWeights{std::move(weights)}, initializer.name());
    }

    std::vector<size_t> topoOrder;
    ASSERT(toposort(graph.node(), &topoOrder) && "Failed to sort the model topologically.", ErrorCode::kINVALID_GRAPH);

    string_map<NodeImporter> const& opImporters = getBuiltinOpImporterMap();
    for (auto const& nodeIndex : topoOrder)
    {
        if (currentNode)
        {
            *currentNode = nodeIndex;
        }
        auto const& node = graph.node(nodeIndex);
        std::string const& nodeName = getNodeName(node);
        LOG_VERBOSE("Parsing node: " << nodeName << " [" << node.op_type() << "]");

        // Assemble node inputs. These may come from outside the subgraph.
        std::vector<TensorOrWeights> nodeInputs;
        std::ostringstream ssInputs{};
        ssInputs << nodeName << " [" << node.op_type() << "] inputs: ";
        for (auto const& inputName : node.input())
        {
            // Empty input names indicate optional inputs which have not been supplied.
            if (inputName.empty())
            {
                // Push back null input as place holder.
                nodeInputs.emplace_back(nullptr);
                ssInputs << "[optional input, not set], ";
            }
            else
            {
                LOG_VERBOSE("Searching for input: " << inputName);
                ASSERT( (ctx->tensors().count(inputName)) && "Node input was not registered.", ErrorCode::kINVALID_GRAPH);
                nodeInputs.push_back(ctx->tensors().at(inputName));
                ssInputs << "[" << inputName << " -> " << nodeInputs.back().shape() << "["
                         << nodeInputs.back().getType() << "]"
                         << "], ";
            }
        }
        LOG_VERBOSE(ssInputs.str());

        // Dispatch to appropriate converter.
        NodeImporter const* importFunc{nullptr};
        if (opImporters.count(node.op_type()))
        {
            importFunc = &opImporters.at(node.op_type());
        }
        else
        {
            LOG_INFO("No importer registered for op: " << node.op_type() << ". Attempting to import as plugin.");
            importFunc = &opImporters.at("FallbackPluginImporter");
        }
        std::vector<TensorOrWeights> outputs;

        try
        {
            GET_VALUE((*importFunc)(ctx, node, nodeInputs), &outputs);
        }
        catch (std::exception const& e)
        {
            return MAKE_ERROR(makeErrorExplanation(e, nodeName), ErrorCode::kINVALID_NODE);
        }
        if (ctx->hasError())
        {
            return MAKE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE);
        }

        for (auto const& output : outputs)
        {
            if (output.is_tensor())
            {
                // check that we can resolve output dims
                // in the future we may have a network/layer.validate() which will help with that as well
                output.tensor().getDimensions();

                if (ctx->hasError())
                {
                    return MAKE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE);
                }
            }
        }

        if (deserializingINetwork)
        {
            OnnxAttrs attrs(node, ctx);

            // Tensor locations, dynamic ranges and layer precisions will be set after parsing the network
            std::vector<std::string> outputsLocation = attrs.get<std::vector<std::string>>("trt_outputs_loc", {});
            std::vector<std::string> outputsVec(node.output().begin(), node.output().end());
            std::vector<std::string> layerName{nodeName};
            CHECK(setTensorLocations(ctx, outputsVec, outputsLocation));

            auto outputsRangeMin = attrs.get<std::vector<float>>("trt_outputs_range_min", {});
            CHECK(setStringMap<float>(ctx, outputsVec, outputsRangeMin, ctx->tensorRangeMins()));
            auto outputsRangeMax = attrs.get<std::vector<float>>("trt_outputs_range_max", {});
            CHECK(setStringMap<float>(ctx, outputsVec, outputsRangeMax, ctx->tensorRangeMaxes()));

            if (attrs.count("trt_layer_precision"))
            {
                std::vector<nvinfer1::DataType> layerPrecision{attrs.get<nvinfer1::DataType>("trt_layer_precision")};
                CHECK(setStringMap<nvinfer1::DataType>(ctx, layerName, layerPrecision, ctx->layerPrecisions()));
            }
        }

        // Set output names and register outputs with the context.
        std::ostringstream ssOutputs{};
        ssOutputs << nodeName << " [" << node.op_type() << "] outputs: ";
        for (int32_t i = 0; i < node.output().size(); ++i)
        {
            auto const& outputName = node.output(i);
            auto& output = outputs.at(i);
            ssOutputs << "[" << outputName << " -> " << output.shape() << "[" << output.getType() << "]"
                      << "], ";
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

Status importInput(ImporterContext* ctx, ::ONNX_NAMESPACE::ValueInfoProto const& input, nvinfer1::ITensor** tensor,
    std::vector<NamedDimension>& namedDims)
{
    auto const& onnxDtype = input.type().tensor_type();
    nvinfer1::DataType trtDtype;
    ASSERT_INPUT(convertDtype(onnxDtype.elem_type(), &trtDtype) && "Failed to convert ONNX date type to TensorRT data type.", ErrorCode::kUNSUPPORTED_NODE, input.name());
    nvinfer1::Dims trt_dims;
    size_t const oldNbNamedDimensions = namedDims.size();
    ASSERT_INPUT(convertOnnxDims(onnxDtype.shape().dim(), trt_dims, namedDims) && "Failed to convert ONNX dimensions to TensorRT dimensions.", ErrorCode::kUNSUPPORTED_GRAPH, input.name());
    nvinfer1::ITensor* userInput = ctx->getUserInput(input.name().c_str());
    if (userInput)
    {
        ASSERT_INPUT(userInput && "User input is missing.", ErrorCode::kINVALID_VALUE, input.name());
        // Intentionally don't check dimensions/dtype here so that users can change the input shape/type if
        // they want to. However, equalities implied by dimension names are nonetheless respected.
        *tensor = userInput;
    }
    else
    {
        LOG_VERBOSE(
            "Adding network input: " << input.name() << " with dtype: " << trtDtype << ", dimensions: " << trt_dims);
        *tensor = ctx->network()->addInput(input.name().c_str(), trtDtype, trt_dims);
        ASSERT_INPUT(*tensor && "Failed to add input to the network.", ErrorCode::kUNSUPPORTED_NODE, input.name());
    }

    // Fill in field `tensor` for any dimensions that had names in the ONNX.
    for (auto i = oldNbNamedDimensions; i < namedDims.size(); ++i)
    {
        namedDims[i].tensor = *tensor;
    }
    return Status::success();
}

static Status setDimensionNames(ImporterContext* ctx, std::vector<NamedDimension>& namedDims)
{
    for (auto const& namedDim : namedDims)
    {
        namedDim.tensor->setDimensionName(namedDim.index, namedDim.dimParam.c_str());
    }
    return Status::success();
}

Status importInputs(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph,
    string_map<TensorOrWeights>* tensors)
{
    // The weights come from the Initializer list in onnx graph
    // Initializers are not really network inputs, so they need to be excluded.
    std::unordered_set<std::string> initializers{};
    for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
    {
        initializers.emplace(initializer.name());
    }

    std::vector<NamedDimension> namedDims;
    for (::ONNX_NAMESPACE::ValueInfoProto const& input : graph.input())
    {
        TensorOrWeights tensor;
        if (!initializers.count(input.name()))
        {
            nvinfer1::ITensor* tensor_ptr{nullptr};
            CHECK(importInput(ctx, input, &tensor_ptr, namedDims));
            tensor = tensor_ptr;
        }
        ctx->registerTensor(std::move(tensor), input.name());
    }

    return setDimensionNames(ctx, namedDims);
}

Status deserialize_onnx_model(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    bool is_serialized_as_text, ::ONNX_NAMESPACE::ModelProto* model)
{
    google::protobuf::io::ArrayInputStream raw_input(serialized_onnx_model, serialized_onnx_model_size);
    if (is_serialized_as_text)
    {
        ASSERT( (google::protobuf::TextFormat::Parse(&raw_input, model)) && "Failed to parse the ONNX model.", ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    else
    {
        google::protobuf::io::CodedInputStream coded_input(&raw_input);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        // Starting Protobuf 3.11 accepts only single parameter.
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
        // Note: This WARs the very low default size limit (64MB)
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
#endif
        ASSERT( (model->ParseFromCodedStream(&coded_input)) && "Failed to parse the ONNX model.", ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    return Status::success();
}

Status deserialize_onnx_model(int32_t fd, bool is_serialized_as_text, ::ONNX_NAMESPACE::ModelProto* model)
{
    google::protobuf::io::FileInputStream raw_input(fd);
    if (is_serialized_as_text)
    {
        ASSERT( (google::protobuf::TextFormat::Parse(&raw_input, model)) && "Failed to parse the ONNX model.", ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    else
    {
        google::protobuf::io::CodedInputStream coded_input(&raw_input);
        // Note: This WARs the very low default size limit (64MB)
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        // Starting Protobuf 3.11 accepts only single parameter.
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
        // Note: This WARs the very low default size limit (64MB)
        coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()/4);
#endif
        ASSERT( (model->ParseFromCodedStream(&coded_input)) && "Failed to parse the ONNX model.", ErrorCode::kMODEL_DESERIALIZE_FAILED);
    }
    return Status::success();
}

bool ModelImporter::supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    SubGraphCollection_t& sub_graph_collection, char const* model_path)
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

    if (model_path)
    {
        _importer_ctx.setOnnxFileLocation(model_path);
    }

    bool allSupported{true};

    // Parse the graph and see if we hit any parsing errors
    allSupported = parse(serialized_onnx_model, serialized_onnx_model_size);

    int32_t error_node = -1;
    std::string input_node{};

    if (!allSupported)
    {
        int32_t nerror = getNbErrors();
        for (int32_t i = 0; i < nerror; ++i)
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
    auto* ctx = &_importer_ctx;
    auto checkForInput = [&input_node, &ctx](::ONNX_NAMESPACE::NodeProto const& node) {
        for (auto input : node.input())
        {
            if (input_node == input || ctx->loopTensors()[input_node] == input)
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
        LOG_VERBOSE("Failed to sort model topologically, exiting ...");
        return false;
    }

    for (int32_t node_idx : topological_order)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = model.graph().node(node_idx);
        // Add the node to the subgraph if:
        //     1. There is an importer function registered for the operator type
        //     2. It is not directly connected to an unsupported input
        //     3. The importer function did not throw an assertion
        bool registered = supportsOperator(node.op_type().c_str());
        bool unsupportedInput = (input_node.empty()) ? false : checkForInput(node);
        bool unsuccessfulParse = node_idx == error_node;
        if (registered && !unsupportedInput && !unsuccessfulParse)
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

// This funciton is used by ONNXRT to partition out unsupported nodes
bool ModelImporter::supportsOperator(char const* op_name) const
{
    auto is = [op_name](char const* name) { return std::strcmp(op_name, name) == 0; };

    // Mark these following plugins as supported
    if (is("EfficientNMS_TRT") || is("PyramidROIAlign_TRT") || is("MultilevelCropAndResize_TRT")
        || is("DisentangledAttention_TRT"))
    {
        return true;
    }
    // Disable nodes that rely on DDS as ONNXRuntime does not support it at the moment
    if (is("NonMaxSuppression") || is("NonZero") || is("RoiAlign"))
    {
        return false;
    }
    return _op_importers.count(op_name);
}

bool ModelImporter::parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size)
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
    status = this->importModel(model);
    if (status.is_error())
    {
        status.setNode(_current_node);
        _errors.push_back(status);
        return false;
    }
    return true;
}

bool ModelImporter::parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path)
{
    auto* const ctx = &_importer_ctx;
    if (ctx->network()->getNbLayers() > 0)
    {
        LOG_ERROR("Parse was called with a non-empty network definition");
        return false;
    }
    if (model_path)
    {
        _importer_ctx.setOnnxFileLocation(model_path);
    }
    return this->parseWithWeightDescriptors(serialized_onnx_model, serialized_onnx_model_size);
}

Status ModelImporter::importModel(
    ::ONNX_NAMESPACE::ModelProto const& model)
{
    ASSERT(!_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure the network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag.", ErrorCode::kINVALID_VALUE);
    auto* ctx = &_importer_ctx;
    _importer_ctx.clearOpsets();
#if ENABLE_STD_PLUGIN
    // Initialize plugin registry
    initLibNvInferPlugins(static_cast<void*>(&ctx->logger()), "");
#endif // ENABLE_STD_PLUGIN
    for (int32_t i = 0; i < model.opset_import().size(); ++i)
    {
        std::string domain = model.opset_import(i).domain();
        int64_t version = model.opset_import(i).version();
        // TensorRT requires an ONNX graph to be generated with at least ai.onnx version 7.
        // ONNX spec says that the default domain is either an empty string or is "ai.onnx".
        if ((domain.empty() || domain == "ai.onnx") && version < 7)
        {
            LOG_WARNING("TensorRT supports ONNX graphs generated with at least opset 7. Models using older opsets are not guaranteed to work.");
        }
        _importer_ctx.addOpset(domain, version);
    }
    ::ONNX_NAMESPACE::GraphProto const& graph = model.graph();
    // Create a dummy tensors so that we can reserve output names. If the output names are encountered elsewhere
    // in the graph, the ctx will know to make the names unique.
    for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
    {
        _importer_ctx.registerTensor(TensorOrWeights{}, output.name());
    }

    _current_node = -1;
    CHECK(importInputs(&_importer_ctx, graph, &_importer_ctx.tensors()));
    CHECK(parseGraph(&_importer_ctx, graph, model.producer_name() == "TensorRT", &_current_node));

    _current_node = -1;
    // Mark outputs defined in the ONNX model (unless tensors are user-requested)
    for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
    {
        ASSERT((_importer_ctx.tensors().count(output.name())) && "The output tensor was not registered.",
            ErrorCode::kINVALID_GRAPH);
        nvinfer1::ITensor* output_tensor_ptr
            = &convertToTensor(_importer_ctx.tensors().at(output.name()), &_importer_ctx);
        LOG_VERBOSE("Marking " << output_tensor_ptr->getName() << " as output: " << output.name());
        output_tensor_ptr->setName(output.name().c_str());

        if (output_tensor_ptr->isNetworkInput())
        {
            // HACK WAR for TRT not allowing input == output
            // TODO: Does this break things by changing the name of the input tensor?
            output_tensor_ptr->setName(("__" + output.name()).c_str());
            output_tensor_ptr = &identity(&_importer_ctx, output_tensor_ptr).tensor();
            ASSERT(output_tensor_ptr && "Failed to add an Identity layer.", ErrorCode::kUNSUPPORTED_NODE);
            output_tensor_ptr->setName(output.name().c_str());
        }

        nvinfer1::ITensor** user_output = _importer_ctx.getUserOutput(output.name().c_str());
        if (!user_output)
        {
            _importer_ctx.network()->markOutput(*output_tensor_ptr);
            nvinfer1::DataType output_trt_dtype;
            ASSERT(convertDtype(output.type().tensor_type().elem_type(), &output_trt_dtype) && "Failed to convert ONNX date type to TensorRT data type.", ErrorCode::kUNSUPPORTED_NODE);
            // For INT32 data type, output type must match tensor type
            ASSERT( (output_tensor_ptr->getType() != nvinfer1::DataType::kINT32
                    || output_trt_dtype == nvinfer1::DataType::kINT32) && "For INT32 tensors, the output type must also be INT32.",
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
        ASSERT( (_importer_ctx.tensors().count(user_output_name)) && "The user-requested output was not registered.", ErrorCode::kINVALID_VALUE);
        TensorOrWeights user_output = _importer_ctx.tensors().at(user_output_name);
        ASSERT( (user_output.is_tensor()) && "The user-requested output must be a tensor.", ErrorCode::kINVALID_VALUE);
        *user_output_ptr = &user_output.tensor();
    }

    if (model.producer_name() == "TensorRT")
    {
        // iterate over all tensors in the network and add them to "tensors" map
        string_map<nvinfer1::ITensor*> tensors;
        string_map<nvinfer1::ILayer*> layers;
        for (int32_t idx = 0; idx < _importer_ctx.network()->getNbInputs(); ++idx)
        {
            nvinfer1::ITensor* tensor = _importer_ctx.network()->getInput(idx);
            if (tensor != nullptr)
            {
                tensors[tensor->getName()] = tensor;
            }
        }
        for (int32_t idx = 0; idx < _importer_ctx.network()->getNbOutputs(); ++idx)
        {
            nvinfer1::ITensor* tensor = _importer_ctx.network()->getOutput(idx);
            if (tensor != nullptr)
            {
                tensors[tensor->getName()] = tensor;
            }
        }
        for (int32_t layerIdx = 0; layerIdx < _importer_ctx.network()->getNbLayers(); ++layerIdx)
        {
            nvinfer1::ILayer* layer = _importer_ctx.network()->getLayer(layerIdx);
            for (int32_t idx = 0; idx < layer->getNbInputs(); ++idx)
            {
                nvinfer1::ITensor* tensor = layer->getInput(idx);
                if (tensor != nullptr)
                {
                    tensors[tensor->getName()] = tensor;
                }
            }
            for (int32_t idx = 0; idx < layer->getNbOutputs(); ++idx)
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
            ASSERT( (tensors.count(tensor.first) > 0) && "The tensor does not have an assigned location.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
            tensors.at(tensor.first)->setLocation(tensor.second);
        }
        // Set dynamic range for all tensors
        for (auto const& tensor : ctx->tensorRangeMins())
        {
            // if there's a min range, there must be a max range as well
            ASSERT( (tensors.count(tensor.first) > 0) && "The tensor does not have an assigned location.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
            if (!std::isnan(tensor.second))
            {
                tensors.at(tensor.first)->setDynamicRange(tensor.second, ctx->tensorRangeMaxes().at(tensor.first));
            }
        }
        // Set precisions for all layers
        for (auto const& layer : ctx->layerPrecisions())
        {
            ASSERT( (layers.count(layer.first) > 0) && "The layer does not have an assigned precision.", nvonnxparser::ErrorCode::kINVALID_GRAPH);
            layers.at(layer.first)->setPrecision(layer.second);
        }
    }

    return Status::success();
}

bool ModelImporter::parseFromFile(char const* onnxModelFile, int32_t verbosity)
{
    auto* ctx = &_importer_ctx;

    // Define S_ISREG macro for Windows
#if !defined(S_ISREG)
# define S_ISREG(mode) (((mode) & S_IFMT) == S_IFREG)
#endif

    struct stat sb;
    if (stat(onnxModelFile, &sb) == 0 && !S_ISREG(sb.st_mode))
    {
	LOG_ERROR("Input is not a regular file: " << onnxModelFile);
	return false;
    }

    GOOGLE_PROTOBUF_VERIFY_VERSION;
    ::ONNX_NAMESPACE::ModelProto onnx_model;

    bool const is_binary = ParseFromFile_WAR(&onnx_model, onnxModelFile);
    if (!is_binary && !ParseFromTextFile(&onnx_model, onnxModelFile))
    {
        LOG_ERROR("Failed to parse ONNX model from file: " << onnxModelFile);
        return false;
    }

    // Keep track of the absolute path to the ONNX file.
    _importer_ctx.setOnnxFileLocation(onnxModelFile);

    int64_t const opset_version = (onnx_model.opset_import().size() ? onnx_model.opset_import(0).version() : 0);
    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("Input filename:   " << onnxModelFile);
    LOG_INFO("ONNX IR version:  " << onnx_ir_version_string(onnx_model.ir_version()));
    LOG_INFO("Opset version:    " << opset_version);
    LOG_INFO("Producer name:    " << onnx_model.producer_name());
    LOG_INFO("Producer version: " << onnx_model.producer_version());
    LOG_INFO("Domain:           " << onnx_model.domain());
    LOG_INFO("Model version:    " << onnx_model.model_version());
    LOG_INFO("Doc string:       " << onnx_model.doc_string());
    LOG_INFO("----------------------------------------------------------------");

    { //...Read input file, parse it
        std::ifstream onnx_file(onnxModelFile, std::ios::binary | std::ios::ate);
        auto const file_size = onnx_file.tellg();
        onnx_file.seekg(0, std::ios::beg);
        std::vector<char> onnx_buf(file_size);
        if (!onnx_file.read(onnx_buf.data(), onnx_buf.size()))
        {
            LOG_ERROR("Failed to read from file: " << onnxModelFile);
            return false;
        }
        if (!parse(onnx_buf.data(), onnx_buf.size()))
        {
            int32_t const nerror = getNbErrors();
            for (int32_t i = 0; i < nerror; ++i)
            {
                nvonnxparser::IParserError const* error = getError(i);
                if (error->node() != -1)
                {
                    ::ONNX_NAMESPACE::NodeProto const& node = onnx_model.graph().node(error->node());
                    LOG_ERROR("While parsing node number " << error->node() << " [" << node.op_type() << " -> \"" << node.output(0) << "\"" << "]:");
                    LOG_ERROR("--- Begin node ---");
                    LOG_ERROR(pretty_print_onnx_to_string(node));
                    LOG_ERROR("--- End node ---");
                }
                LOG_ERROR("ERROR: " << error->file() << ":" << error->line() << " In function " << error->func() << ":\n"
                     << "[" << static_cast<int>(error->code()) << "] " << error->desc());
            }
            return false;
        }
    } //...End Reading input file, parsing it
    return true;
}

} // namespace onnx2trt
