/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ModelImporter.hpp"
#include "OnnxAttrs.hpp"
#include "importerUtils.hpp"
#include "onnxProtoUtils.hpp"
#include "toposort.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <functional>
#include <limits>
#include <sys/stat.h>
#include <unordered_set>

namespace onnx2trt
{

// Helper class and object to shutdown protobuf library upon library unload.
class ProtobufShutter
{
public:
    ~ProtobufShutter()
    {
        google::protobuf::ShutdownProtobufLibrary();
    }
};

static ProtobufShutter protobufShutter;

// Helper for deserializing INetwork
Status setTensorLocations(
    ImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<std::string> const& locations)
{
    ASSERT((tensors.size() >= locations.size())
            && "The size of tensors misaligns with the size of the attribute trt_outputs_loc.",
        nvonnxparser::ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < locations.size(); ++i)
    {
        std::string tensor = tensors.at(i);
        std::string location = locations.at(i);
        nvinfer1::TensorLocation loc
            = location == "device" ? nvinfer1::TensorLocation::kDEVICE : nvinfer1::TensorLocation::kHOST;

        if (ctx->tensorLocations().count(tensor) > 0)
        {
            ASSERT((ctx->tensorLocations()[tensor] == loc) && "The tensor location cannot be changed.",
                nvonnxparser::ErrorCode::kINVALID_GRAPH);
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
    ImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<T> const& data, StringMap<T>& map)
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
static std::string makeErrorExplanation(ImporterContext* ctx, std::string const& nodeName)
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
    result << "Exception occurred in - " << nodeName << "\n" << e.what();
    return result.str();
}

Status parseNode(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx, bool deserializingINetwork)
{
    // For nodes that contain subgraphs (Ifs, Loops, Scans, LocalFunctions), ensure that the recursion depth is
    // limited to a set amount. Recursion depth is tracked by the size of ctx->mBaseNameScopeStack().
    size_t const kMAX_NESTED_SUBGRAPHS = 24;
    if (ctx->getNestedDepth() > kMAX_NESTED_SUBGRAPHS)
    {
        ASSERT(false && "ONNX graph contains nested structures that exceed the maximum allowed by TensorRT!",
            ErrorCode::kUNSUPPORTED_GRAPH);
    }
    StringMap<NodeImporter> const& opImporters = getBuiltinOpImporterMap();
    std::string const& nodeName = getNodeName(node);
    std::string const& nodeType = node.op_type();
    LOG_VERBOSE("Parsing node: " << nodeName << " [" << nodeType << "]");

    // Assemble node inputs. These may come from outside the subgraph.
    std::vector<TensorOrWeights> nodeInputs;
    std::ostringstream ssInputs{};
    ssInputs << nodeName << " [" << nodeType << "] inputs: ";
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
            ASSERT_NODE((ctx->tensors().count(inputName)), "Node input was not registered.", node, nodeIdx,
                ErrorCode::kINVALID_GRAPH);
            nodeInputs.push_back(ctx->tensors().at(inputName));
            ssInputs << "[" << inputName << " -> " << nodeInputs.back().shape() << "[" << nodeInputs.back().getType()
                     << "]" << "], ";
        }
    }
    LOG_VERBOSE(ssInputs.str());

    // Dispatch to appropriate converter.
    NodeImporter const* importFunc{nullptr};
    if (opImporters.count(nodeType))
    {
        importFunc = &opImporters.at(nodeType);
    }
    else if (ctx->localFunctions().count(nodeType))
    {
        LOG_INFO("Found regisitered local function: " << nodeType << ". Importing as a local function.");
        importFunc = &opImporters.at("LocalFunctionImporter");
    }
    else
    {
        LOG_INFO("No importer registered for op: " << nodeType << ". Attempting to import as plugin.");
        importFunc = &opImporters.at("FallbackPluginImporter");
    }
    std::vector<TensorOrWeights> outputs;

    try
    {
        GET_VALUE((*importFunc)(ctx, node, nodeIdx, nodeInputs), &outputs);
    }
    catch (std::exception const& e)
    {
        return MAKE_NODE_ERROR(makeErrorExplanation(e, nodeName), ErrorCode::kINTERNAL_ERROR, node, nodeIdx);
    }
    if (ctx->hasError())
    {
        return MAKE_NODE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE, node, nodeIdx);
    }

    ctx->addLayerOutputTensors(nodeName, outputs);
    for (auto const& output : outputs)
    {
        if (output.is_tensor())
        {
            // check that we can resolve output dims
            // in the future we may have a network/layer.validate() which will help with that as well
            output.tensor().getDimensions();
            // If output dimensions cannot be resolved the error will be captured by the ErrorRecorder.
            if (ctx->hasError())
            {
                return MAKE_NODE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE, node, nodeIdx);
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
        CHECK_STATUS(setTensorLocations(ctx, outputsVec, outputsLocation));

        auto outputsRangeMin = attrs.get<std::vector<float>>("trt_outputs_range_min", {});
        CHECK_STATUS(setStringMap<float>(ctx, outputsVec, outputsRangeMin, ctx->tensorRangeMins()));
        auto outputsRangeMax = attrs.get<std::vector<float>>("trt_outputs_range_max", {});
        CHECK_STATUS(setStringMap<float>(ctx, outputsVec, outputsRangeMax, ctx->tensorRangeMaxes()));

        if (attrs.count("trt_layer_precision"))
        {
            std::vector<nvinfer1::DataType> layerPrecision{attrs.get<nvinfer1::DataType>("trt_layer_precision")};
            CHECK_STATUS(setStringMap<nvinfer1::DataType>(ctx, layerName, layerPrecision, ctx->layerPrecisions()));
        }
    }

    ASSERT_NODE((node.output().size() <= static_cast<int32_t>(outputs.size())),
        "Node has more output tensors than TRT expected, expected output size is "
            << outputs.size() << ", actual output size is " << node.output().size() << ".",
        node, nodeIdx, ErrorCode::kINVALID_GRAPH);

    // Set output names and register outputs with the context.
    std::ostringstream ssOutputs{};
    ssOutputs << nodeName << " [" << node.op_type() << "] outputs: ";
    for (int32_t i = 0; i < node.output().size(); ++i)
    {
        auto const& outputName = node.output(i);
        auto& output = outputs.at(i);
        ssOutputs << "[" << outputName << " -> " << output.shape() << "[" << output.getType() << "]" << "], ";
        // Note: This condition is to allow ONNX outputs to be ignored
        // Always register output weights (even empty ones) as it may be mapped to an unused input
        if ((output || output.is_weights()) && !outputName.empty())
        {
            ctx->registerTensor(std::move(output), outputName);
        }
        // UINT8 is only allowed as network inputs and outputs. Therefore any node that produces an UINT8-typed
        // output that is not also a graph output is unsupported.
        if (output.getType() == "UINT8")
        {
            bool legalUINT8 = false;
            for (auto const& graphOutput : ctx->getGraphOutputNames())
            {
                if (graphOutput.name() == outputName)
                {
                    legalUINT8 = true;
                }
            }
            ASSERT_NODE(legalUINT8, "TensorRT does not support UINT8 types for intermediate tensors!", node, nodeIdx,
                ErrorCode::kUNSUPPORTED_NODE);
        }
    }
    LOG_VERBOSE(ssOutputs.str());
    return Status::success();
}

void parseNodeStaticCheck(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors, size_t const nodeIndex)
{
    StringMap<OpStaticErrorChecker> const& opCheckers = getOpStaticErrorCheckerMap();
    StringMap<NodeImporter> const& opImporters = getBuiltinOpImporterMap();
    std::string const& nodeName = getNodeName(node);
    std::string const& nodeType = node.op_type();
    LOG_VERBOSE("Static check for parsing node: " << nodeName << " [" << nodeType << "]");

    // Dispatch to appropriate static error checker.
    OpStaticErrorChecker const* checkerFunc{nullptr};
    if (opImporters.count(nodeType))
    {
        if (!opCheckers.count(nodeType))
        {
            std::string errorMsg = "No static checker was found for " + nodeType;
            errors.push_back(MAKE_NODE_ERROR(errorMsg, ErrorCode::kINTERNAL_ERROR, node, nodeIndex));
            return;
        }
        checkerFunc = &opCheckers.at(nodeType);
    }
    else if (opCheckers.count(nodeType))
    {
        checkerFunc = &opCheckers.at(nodeType);
    }
    else if (ctx->localFunctions().count(nodeType))
    {
        LOG_INFO("Found regisitered local function: " << nodeType << ". Checking as a local function.");
        checkerFunc = &opCheckers.at("LocalFunctionImporter");
    }
    else
    {
        LOG_INFO("No checker registered for op: " << nodeType << ". Attempting to check as plugin.");
        checkerFunc = &opCheckers.at("FallbackPluginImporter");
    }
    (*checkerFunc)(ctx, node, errors, nodeIndex);
}

Status parseGraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, std::vector<Status>& errors,
    bool deserializingINetwork, int* currentNode)
{
    // Import initializers.
    try
    {
        for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
        {
            LOG_VERBOSE("Importing initializer: " << initializer.name());
            ShapedWeights weights;
            ASSERT(
                ctx->getWeightsContext().convertOnnxWeights(initializer, &weights) && "Failed to import initializer.",
                ErrorCode::kUNSUPPORTED_NODE);
            ctx->registerTensor(TensorOrWeights{std::move(weights)}, initializer.name());
        }
    }
    catch (const std::exception& e)
    {
        ASSERT(false && "Failed to import initialzer", ErrorCode::kINVALID_GRAPH);
    }

    // Keep track of graph outputs in the context to validate UINT8 nodes
    for (const auto& output : graph.output())
    {
        ctx->getGraphOutputNames().push_back(output);
    }

    std::vector<size_t> topoOrder;
    ASSERT(toposort(graph.node(), &topoOrder) && "Failed to sort the model topologically.", ErrorCode::kINVALID_GRAPH);

    for (auto const& nodeIndex : topoOrder)
    {
        if (currentNode)
        {
            *currentNode = nodeIndex;
        }
        parseNodeStaticCheck(ctx, graph.node(nodeIndex), errors, nodeIndex);
        if (errors.size() == 0)
        {
            // At most one dynamic error will be returned.
            auto status = parseNode(ctx, graph.node(nodeIndex), nodeIndex, deserializingINetwork);
            if (status.is_error())
            {
                errors.push_back(status);
            }
        }
    }
    if (errors.size() != 0)
    {
        auto result = errors.back();
        errors.pop_back(); // this error will be added back to the list in ModelImporter::parseWithWeightDescriptors.
        return result;
    }
    return Status::success();
}

std::vector<Status> importInput(ImporterContext* ctx, ::ONNX_NAMESPACE::ValueInfoProto const& input,
    nvinfer1::ITensor** tensor, std::vector<NamedDimension>& namedDims)
{
    std::vector<Status> errorList{};
    auto const& onnxDtype = input.type().tensor_type();
    nvinfer1::DataType trtDtype{nvinfer1::DataType::kFLOAT};
    CHECK_INPUT(
        convertDtype(onnxDtype.elem_type(), &trtDtype) && "Failed to convert ONNX date type to TensorRT data type.",
        ErrorCode::kUNSUPPORTED_NODE, input.name(), errorList);
    nvinfer1::Dims trt_dims;
    size_t const oldNbNamedDimensions = namedDims.size();
    CHECK_INPUT(convertOnnxDims(onnxDtype.shape().dim(), trt_dims, namedDims)
            && "Failed to convert ONNX dimensions to TensorRT dimensions.",
        ErrorCode::kUNSUPPORTED_GRAPH, input.name(), errorList);

    LOG_VERBOSE(
        "Adding network input: " << input.name() << " with dtype: " << trtDtype << ", dimensions: " << trt_dims);
    if (errorList.empty())
    {
        *tensor = ctx->network()->addInput(input.name().c_str(), trtDtype, trt_dims);
        CHECK_INPUT(
            *tensor && "Failed to add input to the network.", ErrorCode::kUNSUPPORTED_NODE, input.name(), errorList);
    }

    // Fill in field `tensor` for any dimensions that had names in the ONNX.
    for (auto i = oldNbNamedDimensions; i < namedDims.size(); ++i)
    {
        namedDims[i].tensor = *tensor;
    }
    return errorList;
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
    StringMap<TensorOrWeights>* tensors, std::vector<Status>& errors)
{
    // The weights come from the Initializer list in onnx graph
    // Initializers are not really network inputs, so they need to be excluded.
    std::unordered_set<std::string> initializers{};
    for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
    {
        initializers.emplace(initializer.name());
    }

    std::vector<NamedDimension> namedDims;
    std::vector<Status> statusList{};
    for (::ONNX_NAMESPACE::ValueInfoProto const& input : graph.input())
    {
        TensorOrWeights tensor;
        if (!initializers.count(input.name()))
        {
            nvinfer1::ITensor* tensor_ptr{nullptr};
            std::vector<Status> status = importInput(ctx, input, &tensor_ptr, namedDims);
            statusList.insert(statusList.end(), status.begin(), status.end());
            tensor = tensor_ptr;
            if (statusList.empty() && tensor_ptr->getType() == nvinfer1::DataType::kINT64)
            {
                LOG_WARNING("Make sure input " << input.name() << " has Int64 binding.");
            }
        }
        ctx->registerTensor(std::move(tensor), input.name());
    }
    if (!statusList.empty())
    {
        errors.insert(errors.end(), statusList.begin(), statusList.end());
        return statusList[0];
    }
    return setDimensionNames(ctx, namedDims);
}

Status importLocalFunctions(ImporterContext* ctx, ::ONNX_NAMESPACE::ModelProto const& model)
{
    for (auto const& localFunction : model.functions())
    {
        ctx->localFunctions().insert({localFunction.name(), localFunction});
    }
    return Status::success();
}

// Internal helper function used for ONNXRT-TRT EP to filter out DDS nodes
bool isDDSOp(char const* op_name)
{
    auto is = [op_name](char const* name) { return std::strcmp(op_name, name) == 0; };
    if (is("NonMaxSuppression") || is("NonZero") || is("RoiAlign"))
    {
        return true;
    }
    return false;
}

std::pair<bool, ModelImporter::SubGraphSupportVector_t> ModelImporter::doSupportsModel(
    void const* serialized_onnx_model, size_t serialized_onnx_model_size, char const* model_path)
{
    ::ONNX_NAMESPACE::ModelProto model;
    Status status = deserializeOnnxModel(serialized_onnx_model, serialized_onnx_model_size, &model);

    if (status.is_error())
    {
        mErrors.push_back(status);
        return std::make_pair<bool, SubGraphSupportVector_t>(false, {});
    }

    if (model_path)
    {
        mImporterCtx.setOnnxFileLocation(model_path);
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
    auto* ctx = &mImporterCtx;
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
        return std::make_pair<bool, SubGraphSupportVector_t>(false, {});
    }

    SubGraphSupportVector_t supportVector;
    for (int32_t node_idx : topological_order)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = model.graph().node(node_idx);
        // Add the node to the subgraph if:
        //     1. It is not a node that requires DDS
        //     2. It is not directly connected to an unsupported input
        //     3. The importer function did not throw an assertion
        bool unsupportedDDS = isDDSOp(node.op_type().c_str());
        bool unsupportedInput = (input_node.empty()) ? false : checkForInput(node);
        bool unsuccessfulParse = node_idx == error_node;
        if (!unsupportedDDS && !unsupportedInput && !unsuccessfulParse)
        {
            if (newSubGraph)
            {
                // If it is the beginning of a new subGraph, we start a new vector
                supportVector.emplace_back();
                // Mark all new graphs as "unknown"
                supportVector.back().second = false;
                newSubGraph = false;
            }
            // We add the new node to the last graph
            supportVector.back().first.emplace_back(node_idx);
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
        supportVector.back().second = true;
    }
    return std::make_pair(allSupported, std::move(supportVector));
}

bool ModelImporter::supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    SubGraphCollection_t& sub_graph_collection, char const* model_path) noexcept
{
    ONNXTRT_TRY
    {
        std::pair<bool, SubGraphSupportVector_t> result
            = doSupportsModel(serialized_onnx_model, serialized_onnx_model_size, model_path);
        bool supports = result.first;
        SubGraphSupportVector_t supportVector = result.second;

        sub_graph_collection.clear();

        // SubGraphCollection uses size_t, while SubGraphSupportVector_t uses int64_t
        for (const auto& pair : supportVector)
        {
            bool subgraphSupports = pair.second;

            std::vector<int64_t> const& subgraphNodes = pair.first;
            std::vector<size_t> subgraphNodesRet(subgraphNodes.begin(), subgraphNodes.end());

            // Create a new pair and add it to vector b
            sub_graph_collection.push_back(std::make_pair(subgraphNodesRet, subgraphSupports));
        }

        return supports;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

bool ModelImporter::supportsModelV2(
    void const* serialized_onnx_model, size_t serialized_onnx_model_size, char const* model_path) noexcept
{
    ONNXTRT_TRY
    {
        std::pair<bool, SubGraphSupportVector_t> result
            = doSupportsModel(serialized_onnx_model, serialized_onnx_model_size, model_path);
        bool supports = result.first;
        SubGraphSupportVector_t supportVector = result.second;

        mSubGraphSupportVector.resize(supportVector.size());
        std::copy(supportVector.begin(), supportVector.end(), mSubGraphSupportVector.begin());

        return supports;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

int64_t ModelImporter::getNbSubgraphs() noexcept
{
    ONNXTRT_TRY
    {
        return mSubGraphSupportVector.size();
    }
    ONNXTRT_CATCH_RECORD
    return 0;
}

bool ModelImporter::isSubgraphSupported(int64_t const index) noexcept
{
    ONNXTRT_TRY
    {
        std::ostringstream errorMessage;
        errorMessage << "Query index " << index
                     << " exceeds subgraph support vector (size = " << mSubGraphSupportVector.size()
                     << "). Have you called supports_model_v2?";
        ONNXTRT_CHECK(mSubGraphSupportVector.size() > static_cast<uint64_t>(index),
            MAKE_ERROR(errorMessage.str(), ErrorCode::kINVALID_VALUE));
        return mSubGraphSupportVector[index].second;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

int64_t* ModelImporter::getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept
{
    ONNXTRT_TRY
    {
        std::ostringstream errorMessage;
        errorMessage << "Query index " << index
                     << " exceeds subgraph support vector (size = " << mSubGraphSupportVector.size()
                     << "). Have you called supports_model_v2?";
        ONNXTRT_CHECK(mSubGraphSupportVector.size() > static_cast<uint64_t>(index),
            MAKE_ERROR(errorMessage.str(), ErrorCode::kINVALID_VALUE));
        subgraphLength = mSubGraphSupportVector[index].first.size();
        return mSubGraphSupportVector[index].first.data();
    }
    ONNXTRT_CATCH_RECORD

    subgraphLength = 0;
    return nullptr;
}

bool ModelImporter::supportsOperator(char const* op_name) const noexcept
{
    ONNXTRT_TRY
    {
        return _op_importers.count(op_name);
    }
    ONNXTRT_CATCH_RECORD

    return false;
}

bool ModelImporter::parseWithWeightDescriptors(
    void const* serialized_onnx_model, size_t serialized_onnx_model_size) noexcept
{
    ONNXTRT_TRY
    {
        mCurrentNode = -1;
        // TODO: This function (and its overload below) could do with some cleaning,
        //       particularly wrt error handling.
        // Note: We store a copy of the model so that weight arrays will persist
        mONNXModels.emplace_back();
        ::ONNX_NAMESPACE::ModelProto& model = mONNXModels.back();
        Status status = deserializeOnnxModel(serialized_onnx_model, serialized_onnx_model_size, &model);
        if (status.is_error())
        {
            mErrors.push_back(status);
            return false;
        }
        status = this->importModel(model);
        if (status.is_error())
        {
            mErrors.push_back(status);
            return false;
        }
        return true;
    }
    ONNXTRT_CATCH_RECORD

    return false;
}

bool ModelImporter::parse(
    void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path) noexcept
{
    ONNXTRT_TRY
    {
        auto* const ctx = &mImporterCtx;

        if (ctx->network()->getNbLayers() > 0)
        {
            LOG_ERROR("Parse was called with a non-empty network definition");
            return false;
        }
        if (model_path)
        {
            mImporterCtx.setOnnxFileLocation(model_path);
        }
        return this->parseWithWeightDescriptors(serialized_onnx_model, serialized_onnx_model_size);
    }
    ONNXTRT_CATCH_RECORD

    return false;
}

Status ModelImporter::importModel(::ONNX_NAMESPACE::ModelProto const& model) noexcept
{
    ONNXTRT_TRY
    {
        auto* ctx = &mImporterCtx;
        mImporterCtx.clearOpsets();
        // Add domain import limit for security reasons
        int32_t const MAX_DOMAINS = 1024;
        ASSERT(model.opset_import().size() <= MAX_DOMAINS
                && "Model contains more than 1024 domains! Parsing will halt for security reasons.",
            ErrorCode::kUNSUPPORTED_GRAPH);
        for (int32_t i = 0; i < model.opset_import().size(); ++i)
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
            mImporterCtx.addOpset(domain, version);
        }
        ::ONNX_NAMESPACE::GraphProto const& graph = model.graph();
        // Create a dummy tensors so that we can reserve output names. If the output names are encountered elsewhere
        // in the graph, the ctx will know to make the names unique.
        for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
        {
            mImporterCtx.registerTensor(TensorOrWeights{}, output.name());
        }

        // Import LocalFunctions
        CHECK_STATUS(importLocalFunctions(&mImporterCtx, model));

        // Propagate OnnxParserFlags down to the importer context.
        mImporterCtx.setFlags(getFlags());

        mCurrentNode = -1;
        importInputs(&mImporterCtx, graph, &mImporterCtx.tensors(), mErrors);
        CHECK_STATUS(parseGraph(&mImporterCtx, graph, mErrors, model.producer_name() == "TensorRT", &mCurrentNode));

        mCurrentNode = -1;
        // Mark outputs defined in the ONNX model (unless tensors are user-requested)
        for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
        {
            ASSERT((mImporterCtx.tensors().count(output.name())) && "The output tensor was not registered.",
                ErrorCode::kINVALID_GRAPH);
            nvinfer1::ITensor* output_tensor_ptr
                = &convertToTensor(mImporterCtx.tensors().at(output.name()), &mImporterCtx);
            LOG_VERBOSE("Marking " << output_tensor_ptr->getName() << " as output: " << output.name());
            output_tensor_ptr->setName(output.name().c_str());

            if (output_tensor_ptr->isNetworkInput())
            {
                // HACK WAR for TRT not allowing input == output
                // TODO: Does this break things by changing the name of the input tensor?
                output_tensor_ptr->setName(("__" + output.name()).c_str());
                output_tensor_ptr = &identity(&mImporterCtx, output_tensor_ptr).tensor();
                ASSERT(output_tensor_ptr && "Failed to add an Identity layer.", ErrorCode::kUNSUPPORTED_NODE);
                output_tensor_ptr->setName(output.name().c_str());
            }

            mImporterCtx.network()->markOutput(*output_tensor_ptr);
            nvinfer1::DataType output_trt_dtype;
            ASSERT(convertDtype(output.type().tensor_type().elem_type(), &output_trt_dtype)
                    && "Failed to convert ONNX date type to TensorRT data type.",
                ErrorCode::kUNSUPPORTED_NODE);
            // For INT32 data type, output type must match tensor type
            ASSERT((output_tensor_ptr->getType() != nvinfer1::DataType::kINT32
                       || output_trt_dtype == nvinfer1::DataType::kINT32)
                    && "For INT32 tensors, the output type must also be INT32.",
                ErrorCode::kUNSUPPORTED_NODE);
            // Note: Without this, output type is always float32
            output_tensor_ptr->setType(output_trt_dtype);
            if (output_trt_dtype == nvinfer1::DataType::kINT64)
            {
                LOG_WARNING("Make sure output " << output.name() << " has Int64 binding.");
            }
        }

        if (model.producer_name() == "TensorRT")
        {
            // iterate over all tensors in the network and add them to "tensors" map
            StringMap<nvinfer1::ITensor*> tensors;
            StringMap<nvinfer1::ILayer*> layers;
            for (int32_t idx = 0; idx < mImporterCtx.network()->getNbInputs(); ++idx)
            {
                nvinfer1::ITensor* tensor = mImporterCtx.network()->getInput(idx);
                if (tensor != nullptr)
                {
                    tensors[tensor->getName()] = tensor;
                }
            }
            for (int32_t idx = 0; idx < mImporterCtx.network()->getNbOutputs(); ++idx)
            {
                nvinfer1::ITensor* tensor = mImporterCtx.network()->getOutput(idx);
                if (tensor != nullptr)
                {
                    tensors[tensor->getName()] = tensor;
                }
            }
            for (int32_t layerIdx = 0; layerIdx < mImporterCtx.network()->getNbLayers(); ++layerIdx)
            {
                nvinfer1::ILayer* layer = mImporterCtx.network()->getLayer(layerIdx);
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
                ASSERT((tensors.count(tensor.first) > 0) && "The tensor does not have an assigned location.",
                    nvonnxparser::ErrorCode::kINVALID_GRAPH);
                tensors.at(tensor.first)->setLocation(tensor.second);
            }
            // Set dynamic range for all tensors
            for (auto const& tensor : ctx->tensorRangeMins())
            {
                // if there's a min range, there must be a max range as well
                ASSERT((tensors.count(tensor.first) > 0) && "The tensor does not have an assigned location.",
                    nvonnxparser::ErrorCode::kINVALID_GRAPH);
                if (!std::isnan(tensor.second))
                {
                    tensors.at(tensor.first)->setDynamicRange(tensor.second, ctx->tensorRangeMaxes().at(tensor.first));
                }
            }
            // Avoid setting layer precision if graph is strongly typed.
            if (!ctx->network()->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
            {
                // Set precisions for all layers.
                for (auto const& layer : ctx->layerPrecisions())
                {
                    ASSERT((layers.count(layer.first) > 0) && "The layer does not have an assigned precision.",
                        nvonnxparser::ErrorCode::kINVALID_GRAPH);
                    layers.at(layer.first)->setPrecision(layer.second);
                }
            }
        }

        // Regenerate the plugin library list
        mPluginLibraryList = ctx->getUsedVCPluginLibraries();
        mPluginLibraryListCStr.clear();
        mPluginLibraryListCStr.reserve(mPluginLibraryList.size());
        for (auto const& s : mPluginLibraryList)
        {
            mPluginLibraryListCStr.push_back(s.c_str());
        }

        return Status::success();
    }
    ONNXTRT_CATCH_RECORD
    return Status{ErrorCode::kINTERNAL_ERROR};
}

bool ModelImporter::parseFromFile(char const* onnxModelFile, int32_t verbosity) noexcept
{
    auto* ctx = &mImporterCtx;

    // Define S_ISREG macro for Windows
#if !defined(S_ISREG)
#define S_ISREG(mode) (((mode) & S_IFMT) == S_IFREG)
#endif

    struct stat sb;
    if (stat(onnxModelFile, &sb) == 0 && !S_ISREG(sb.st_mode))
    {
        LOG_ERROR("Input is not a regular file: " << onnxModelFile);
        return false;
    }

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // Own the ONNX model for weights to persist.
    mONNXModels.emplace_back();
    ::ONNX_NAMESPACE::ModelProto& onnxModel = mONNXModels.back();
    bool const fileLoadSuccess = ParseFromFileAsBinary(&onnxModel, onnxModelFile);
    if (!fileLoadSuccess)
    {
        LOG_ERROR("Failed to parse ONNX model from file: " << onnxModelFile << "!");
        return false;
    }

    // Keep track of the absolute path to the ONNX file.
    mImporterCtx.setOnnxFileLocation(onnxModelFile);

    int64_t const opset_version = (onnxModel.opset_import().size() ? onnxModel.opset_import(0).version() : 0);
    LOG_INFO("----------------------------------------------------------------");
    LOG_INFO("Input filename:   " << onnxModelFile);
    LOG_INFO("ONNX IR version:  " << onnxIRVersionAsString(onnxModel.ir_version()));
    LOG_INFO("Opset version:    " << opset_version);
    LOG_INFO("Producer name:    " << onnxModel.producer_name());
    LOG_INFO("Producer version: " << onnxModel.producer_version());
    LOG_INFO("Domain:           " << onnxModel.domain());
    LOG_INFO("Model version:    " << onnxModel.model_version());
    LOG_INFO("Doc string:       " << onnxModel.doc_string());
    LOG_INFO("----------------------------------------------------------------");

    // Set currentNode count to -1
    mCurrentNode = -1;
    Status status = this->importModel(onnxModel);
    if (status.is_error())
    {
        mErrors.push_back(status);
    }

    int32_t const numErrors = getNbErrors();
    for (int32_t i = 0; i < numErrors; ++i)
    {
        nvonnxparser::IParserError const* error = getError(i);
        if (error->node() != -1)
        {
            ::ONNX_NAMESPACE::NodeProto const& node = onnxModel.graph().node(error->node());
            LOG_ERROR("While parsing node number " << error->node() << " [" << node.op_type() << " -> \""
                                                   << node.output(0) << "\"" << "]:");
            LOG_ERROR("--- Begin node ---" << "\n" << node);
            LOG_ERROR("--- End node ---");
        }
        LOG_ERROR("ERROR: " << error->file() << ":" << error->line() << " In function " << error->func() << ":\n"
                            << "[" << static_cast<int>(error->code()) << "] " << error->desc());
    }
    return numErrors == 0;
}

char const* const* ModelImporter::getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept
{
    nbPluginLibs = mPluginLibraryListCStr.size();
    return (nbPluginLibs > 0) ? mPluginLibraryListCStr.data() : nullptr;
}

} // namespace onnx2trt
