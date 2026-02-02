/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ModelImporter.hpp"
#include "OnnxAttrs.hpp"
#include "Status.hpp"
#include "errorHelpers.hpp"
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
void setTensorLocations(
    ImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<std::string> const& locations)
{
    ONNXTRT_CHECK(tensors.size() >= locations.size(),
        "The size of tensors misaligns with the size of the attribute trt_outputs_loc.", ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < locations.size(); ++i)
    {
        std::string tensor = tensors.at(i);
        std::string location = locations.at(i);
        nvinfer1::TensorLocation loc
            = location == "device" ? nvinfer1::TensorLocation::kDEVICE : nvinfer1::TensorLocation::kHOST;

        if (ctx->tensorLocations().count(tensor) > 0)
        {
            ONNXTRT_CHECK(ctx->tensorLocations()[tensor] == loc, "The tensor location cannot be changed.",
                ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            ctx->tensorLocations()[tensor] = loc;
        }
    }
}

// Helper for deserializing INetwork
template <typename T>
void setStringMap(
    ImporterContext* ctx, std::vector<std::string> const& tensors, std::vector<T> const& data, StringMap<T>& map)
{
    ONNXTRT_CHECK(tensors.size() >= data.size(),
        "The size of tensors misaligns with the size of the attribute trt_outputs_range_min/max.",
        ErrorCode::kINVALID_GRAPH);
    for (size_t i = 0; i < data.size(); ++i)
    {
        std::string name = tensors.at(i);
        T dataName = data.at(i);
        if (map.count(name) > 0)
        {
            ONNXTRT_CHECK(map[name] == dataName,
                "The order of tensorRangeMin/Max in context misaligns with the order of the attribute "
                "trt_outputs_range_min/max.",
                ErrorCode::kINVALID_GRAPH);
        }
        else
        {
            map[name] = dataName;
        }
    }
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

void ModelImporter::logModelInfo()
{
    auto* ctx = &mImporterCtx;
    int64_t const opset_version = (mOnnxModel.opset_import().size() ? mOnnxModel.opset_import(0).version() : 0);
    auto fileName = ctx->getWeightsContext().getOnnxFileLocation();
    LOG_INFO("----------------------------------------------------------------");
    if (!fileName.empty())
    {
        LOG_INFO("Input filename:   " << fileName);
    }
    LOG_INFO("ONNX IR version:  " << onnxIRVersionAsString(mOnnxModel.ir_version()));
    LOG_INFO("Opset version:    " << opset_version);
    LOG_INFO("Producer name:    " << mOnnxModel.producer_name());
    LOG_INFO("Producer version: " << mOnnxModel.producer_version());
    LOG_INFO("Domain:           " << mOnnxModel.domain());
    LOG_INFO("Model version:    " << mOnnxModel.model_version());
    LOG_INFO("Doc string:       " << mOnnxModel.doc_string());
    LOG_INFO("----------------------------------------------------------------");
}

void ModelImporter::logErrors()
{
    auto* ctx = &mImporterCtx;
    int32_t const numErrors = getNbErrors();
    for (int32_t i = 0; i < numErrors; ++i)
    {
        nvonnxparser::IParserError const* error = getError(i);
        if (error->node() != -1)
        {
            auto const& graph = mOnnxModel.graph();
            if (error->node() >= 0 && error->node() < graph.node_size())
            {
                ::ONNX_NAMESPACE::NodeProto const& node = graph.node(error->node());
                LOG_ERROR("While parsing node number " << error->node() << " [" << node.op_type() << " -> \""
                                                       << node.output(0) << "\"" << "]:");
                LOG_ERROR("--- Begin node ---" << "\n" << node);
                LOG_ERROR("--- End node ---");
            }
            else
            {
                LOG_ERROR("While parsing node number "
                    << error->node() << " (index out of bounds, graph size: " << graph.node_size() << ")");
            }
        }
        LOG_ERROR("ERROR: " << error->file() << ":" << error->line() << " In function " << error->func() << ":\n"
                            << "[" << static_cast<int>(error->code()) << "] " << error->desc());
    }
}

void ModelImporter::reportSubgraphs()
{
    int32_t error_node = -1;
    std::string input_node{};

    bool allSupported = getNbErrors() == 0;

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
    if (!toposort(mOnnxModel.graph().node(), &topological_order))
    {
        LOG_ERROR("Failed to sort model topologically");
        return;
    }

    mSubGraphSupportVector.clear();
    for (int32_t node_idx : topological_order)
    {
        ::ONNX_NAMESPACE::NodeProto const& node = mOnnxModel.graph().node(node_idx);
        // Add the node to the subgraph if:
        //     1. It is not directly connected to an unsupported input
        //     2. The importer function did not throw an assertion
        bool unsupportedInput = !input_node.empty() && checkForInput(node);
        bool unsuccessfulParse = node_idx == error_node;
        if (!unsupportedInput && !unsuccessfulParse)
        {
            if (newSubGraph)
            {
                // If it is the beginning of a new subGraph, we start a new vector
                mSubGraphSupportVector.emplace_back();
                // Mark all new graphs as "unknown"
                mSubGraphSupportVector.back().second = false;
                newSubGraph = false;
            }
            // We add the new node to the last graph
            mSubGraphSupportVector.back().first.emplace_back(node_idx);
        }
        else
        {
            // This is not a supported node, reset newSubGraph
            newSubGraph = true;
            allSupported = false;
        }
    }
    // Only one subgraph, mark it as true.
    if (allSupported && mSubGraphSupportVector.size() == 1)
    {
        mSubGraphSupportVector.back().second = true;
    }
}

bool isNodeInPluginRegistry(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node)
{
    OnnxAttrs attrs(node, ctx);
    std::string const pluginVersion{attrs.get<std::string>("plugin_version", "1")};
    std::string const pluginNamespace{attrs.get<std::string>("plugin_namespace", "")};
    LOG_VERBOSE("Checking if node can be treated as plugin: " << node.op_type() << ", plugin_version: " << pluginVersion
                                                           << ", plugin_namespace: " << pluginNamespace);
    nvinfer1::IPluginCreatorInterface* creator
        = importPluginCreator(ctx, node.op_type(), pluginVersion, pluginNamespace);
    return creator;
}

void parseNode(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx, bool deserializingINetwork)
{
    // For nodes that contain subgraphs (Ifs, Loops, Scans, LocalFunctions), ensure that the recursion depth is
    // limited to a set amount. Recursion depth is tracked by the size of ctx->mBaseNameScopeStack().
    size_t const kMAX_NESTED_SUBGRAPHS = 24;
    if (ctx->getNestedDepth() > kMAX_NESTED_SUBGRAPHS)
    {
        ONNXTRT_THROW(MAKE_ERROR("ONNX graph contains nested structures that exceed the maximum allowed by TensorRT!",
            ErrorCode::kUNSUPPORTED_GRAPH));
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
            ONNXTRT_CHECK_NODE((ctx->tensors().count(inputName)), "Node input was not registered.", node, nodeIdx,
                ErrorCode::kINVALID_GRAPH);
            nodeInputs.push_back(ctx->tensors().at(inputName));
            ssInputs << "[" << inputName << " -> " << nodeInputs.back().shape() << "[" << nodeInputs.back().getType()
                     << "]"
                     << "], ";
        }
    }
    LOG_VERBOSE(ssInputs.str());

    // UINT8 weights that are not Q/DQ inputs will be converted to INT32
    // If the UINT8 quantization flag is enabled, constants with UINT8 will also be permitted.
    uint32_t uint8AsymmetricQuantizationFlag = 1U
        << static_cast<uint32_t>(nvonnxparser::OnnxParserFlag::kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA);
    bool allowUint8Quantization = ctx->getFlags() & uint8AsymmetricQuantizationFlag;

    bool skipUInt8Conversion = (node.op_type() == "QuantizeLinear" || node.op_type() == "DequantizeLinear"
        || (allowUint8Quantization && node.op_type() == "Constant"));
    skipUInt8Conversion
        |= (node.op_type() == "TRT_MXFP8QuantizeLinear" || node.op_type() == "TRT_MXFP8DequantizeLinear");

        if (!skipUInt8Conversion)
        {
            for (auto& nodeInput : nodeInputs)
            {
                if (nodeInput.is_weights()
                    && nodeInput.weights().type == static_cast<int32_t>(::ONNX_NAMESPACE::TensorProto::UINT8))
                {
                    auto weights = nodeInput.weights();
                    LOG_WARNING("UINT8 data " << weights.name << " is being converted to INT32.");
                    auto uint8_data = static_cast<uint8_t*>(weights.values);
                    int32_t* int32_data = ctx->getWeightsContext().convertUINT8(uint8_data, weights.shape);
                    auto int32ShapedWeights = ShapedWeights(
                        static_cast<int32_t>(::ONNX_NAMESPACE::TensorProto::INT32), int32_data, weights.shape);
                    ctx->tensors()[std::string(weights.name)] = int32ShapedWeights;
                    nodeInput = int32ShapedWeights;
                }
            }
        }

    // Dispatch to appropriate converter.
    NodeImporter const* importFunc{nullptr};

    // if the ENABLE_PLUGIN_OVERRIDE flag is set, then let the plugin override the standard ONNX operator
    bool const pluginOverriding
        = ctx->getFlags() & (1U << static_cast<uint32_t>(nvonnxparser::OnnxParserFlag::kENABLE_PLUGIN_OVERRIDE));
    OnnxAttrs attrs(node, ctx);
    bool const isPluginNode = attrs.count("plugin_namespace");

    if (pluginOverriding || isPluginNode)
    {
        if (isNodeInPluginRegistry(ctx, node))
        {
            LOG_VERBOSE("Found registered plugin: " << nodeType << ". Importing this node as a plugin.");
            importFunc = &opImporters.at("FallbackPluginImporter");
        }
    }

    if (!importFunc)
    {
        if (opImporters.count(nodeType))
        {
            importFunc = &opImporters.at(nodeType);
        }
        // To be consistent with operator override, local function will be preferred over plugin if the flag is unset.
        else if (ctx->localFunctions().count(nodeType))
        {
            LOG_VERBOSE("Found registered local function: " << nodeType << ". Importing as a local function.");
            importFunc = &opImporters.at("LocalFunctionImporter");
        }
        else if (!pluginOverriding && !isPluginNode) // avoid to check Node in plugin registry again
        {
            LOG_VERBOSE("No importer registered for op: " << nodeType << ". Attempting to import as plugin.");
            importFunc = &opImporters.at("FallbackPluginImporter");
        }
    }
    if (!importFunc)
    {
        ONNXTRT_THROW(
            MAKE_NODE_ERROR("No importer registered for op: " + nodeType, ErrorCode::kUNSUPPORTED_NODE, node, nodeIdx));
    }

    std::vector<TensorOrWeights> outputs;
    try
    {
        outputs = (*importFunc)(ctx, node, nodeIdx, nodeInputs);
    }
    catch (OnnxTrtException& e)
    {
        throw e;
    }
    catch (std::exception& e)
    {
        ONNXTRT_THROW(MAKE_NODE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE, node, nodeIdx));
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
                ONNXTRT_THROW(
                    MAKE_NODE_ERROR(makeErrorExplanation(ctx, nodeName), ErrorCode::kINVALID_NODE, node, nodeIdx));
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
        setTensorLocations(ctx, outputsVec, outputsLocation);

        auto outputsRangeMin = attrs.get<std::vector<float>>("trt_outputs_range_min", {});
        setStringMap<float>(ctx, outputsVec, outputsRangeMin, ctx->tensorRangeMins());
        auto outputsRangeMax = attrs.get<std::vector<float>>("trt_outputs_range_max", {});
        setStringMap<float>(ctx, outputsVec, outputsRangeMax, ctx->tensorRangeMaxes());

        if (attrs.count("trt_layer_precision"))
        {
            std::vector<nvinfer1::DataType> layerPrecision{attrs.get<nvinfer1::DataType>("trt_layer_precision")};
            setStringMap<nvinfer1::DataType>(ctx, layerName, layerPrecision, ctx->layerPrecisions());
        }
    }

    int32_t nonEmptyOutputs
        = std::count_if(node.output().begin(), node.output().end(), [](std::string const& str) { return !str.empty(); });
    ONNXTRT_CHECK_NODE(nonEmptyOutputs == static_cast<int32_t>(outputs.size()),
        "Node has more output tensors than TRT expected, expected output size is "
            << outputs.size() << ", actual output size is " << nonEmptyOutputs << ".",
        node, nodeIdx, ErrorCode::kINVALID_GRAPH);

    // Set output names and register outputs with the context.
    std::ostringstream ssOutputs{};
    ssOutputs << nodeName << " [" << node.op_type() << "] outputs: ";
    for (int32_t i = 0, trtCnt = 0; i < node.output().size(); ++i)
    {
        auto const& outputName = node.output(i);
        // Empty strings denote null-tensor outputs. Ignore these.
        if (outputName.empty())
        {
            continue;
        }
        auto& output = outputs.at(trtCnt);
        ssOutputs << "[" << outputName << " -> " << output.shape() << "[" << output.getType() << "]"
                  << "], ";
        // Note: This condition is to allow ONNX outputs to be ignored
        // Always register output weights (even empty ones) as it may be mapped to an unused input
        if ((output || output.is_weights()) && !outputName.empty())
        {
            ctx->registerTensor(std::move(output), outputName);
        }
        // UINT8 is only allowed as network inputs, network outputs, and constants for QDQ nodes unless the UINT8
        // quantization flag is set. If the UINT8 quantization flag is set, then UINT8 is also permitted as a
        // QuantizeLinear output or Gather output (when they feed into a dequantize node). Other than the cases listed,
        // any non-constant node that produces an UINT8-typed output that is not also a graph output is unsupported.
        if (output.getType() == "UINT8")
        {
            bool legalUINT8 = node.op_type() == "Constant"
                || (allowUint8Quantization && (node.op_type() == "Gather" || node.op_type() == "QuantizeLinear"));
            for (auto const& graphOutput : ctx->getGraphOutputNames())
            {
                if (graphOutput.name() == outputName)
                {
                    legalUINT8 = true;
                    break;
                }
            }
            ONNXTRT_CHECK_NODE(legalUINT8,
                "TensorRT does not support UINT8 types for intermediate tensors. For UINT8 quantization, the "
                "kIMPORT_UINT8_QUANTIZATION flag must be set. (DLA version >= 3.16 only)",
                node, nodeIdx, ErrorCode::kUNSUPPORTED_NODE);
        }
        trtCnt++;
    }
    LOG_VERBOSE(ssOutputs.str());
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
        // Let plugin take precedence over local function. So first check if this can be dispatched to a plugin.
        if (isNodeInPluginRegistry(ctx, node))
        {
            LOG_VERBOSE("Found registered plugin: " << nodeType << ". Importing local function as a plugin.");
            checkerFunc = &opCheckers.at("FallbackPluginImporter");
        }
        else
        {
            LOG_VERBOSE("Found registered local function: " << nodeType << ". Importing as a local function.");
            checkerFunc = &opCheckers.at("LocalFunctionImporter");
        }
    }
    else
    {
        LOG_VERBOSE("No checker registered for op: " << nodeType << ". Attempting to check as plugin.");
        checkerFunc = &opCheckers.at("FallbackPluginImporter");
    }
    (*checkerFunc)(ctx, node, errors, nodeIndex);
}

void parseGraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, std::vector<Status>& errors,
    bool deserializingINetwork, int32_t* currentNode, int32_t subgraphParentIdx)
{
    // Import initializers.
    try
    {
        for (::ONNX_NAMESPACE::TensorProto const& initializer : graph.initializer())
        {
            LOG_VERBOSE("Importing initializer: " << initializer.name());
            ShapedWeights weights;
            ONNXTRT_CHECK(ctx->getWeightsContext().convertOnnxWeights(initializer, &weights),
                "Failed to import initializer: " << initializer.name(), ErrorCode::kUNSUPPORTED_NODE);
            ctx->registerTensor(TensorOrWeights{std::move(weights)}, initializer.name());
        }
    }
    catch (const std::exception& e)
    {
        ONNXTRT_THROW(MAKE_ERROR(std::string("Failed to import initializer: ") + e.what(), ErrorCode::kINVALID_GRAPH));
    }

    // Keep track of graph outputs in the context to validate UINT8 nodes
    for (const auto& output : graph.output())
    {
        ctx->getGraphOutputNames().push_back(output);
    }

    std::vector<size_t> topoOrder;
    ONNXTRT_CHECK(
        toposort(graph.node(), &topoOrder), "Failed to sort the model topologically.", ErrorCode::kINVALID_GRAPH);

    for (auto const& nodeIndex : topoOrder)
    {
        if (currentNode)
        {
            *currentNode = nodeIndex;
        }
        // When parsing a subgraph, use the parent node index for error reporting.
        // This ensures errors in subgraphs (e.g., Loop body) report the correct main-graph node index.
        size_t const nodeIdxForErrors = (subgraphParentIdx >= 0) ? static_cast<size_t>(subgraphParentIdx) : nodeIndex;
        parseNodeStaticCheck(ctx, graph.node(nodeIndex), errors, nodeIdxForErrors);

        int32_t numPrevLayers = ctx->network()->getNbLayers();
        // Parse the node
        if (errors.size() == 0)
        {
            // At most one dynamic error will be returned.
            parseNode(ctx, graph.node(nodeIndex), nodeIdxForErrors, deserializingINetwork);
        }

        // Validate DLA support if kREPORT_CAPABILITY_DLA flag is set.
        if (ctx->getDLACapabilityMode())
        {
            ctx->checkDLASupport(numPrevLayers, graph.node(nodeIndex), nodeIdxForErrors);
        }
    }

    // Static check still reports error through the error vector by design
    if (errors.size() != 0)
    {
        auto result = errors.back();
        errors.pop_back(); // this error will be added back to the list in ModelImporter::parseWithWeightDescriptors.

        ONNXTRT_THROW(result);
    }
}

// Still returns a vector<Status> since CHECK_INPUT doesn't immediately return
std::vector<Status> importInput(ImporterContext* ctx, ::ONNX_NAMESPACE::ValueInfoProto const& input,
    nvinfer1::ITensor** tensor, std::vector<NamedDimension>& namedDims)
{
    std::vector<Status> errorList{};
    auto const& onnxDtype = input.type().tensor_type();
    nvinfer1::DataType trtDtype{nvinfer1::DataType::kFLOAT};
    CHECK_INPUT(
        convertDtype(onnxDtype.elem_type(), &trtDtype) && "Failed to convert ONNX date type to TensorRT data type.",
        ErrorCode::kUNSUPPORTED_NODE, input.name(), errorList);
    // If convertOnnxDims fails, trt_dims may not be modified. Also CHECK_INPUT won't return immediately.
    // So we need to initialize trt_dims to avoid illegal access in the following log verbose.
    nvinfer1::Dims trt_dims{};
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

namespace
{
//! \return true if \p inputName is a minus sign followed by digits.
//! I.e., it matches the regex `^\-\d+$`.
[[nodiscard]] bool isNegativeInteger(std::string_view inputName) noexcept
{
    // Check if the string starts with '-' and has more than one character
    if (inputName.size() > 1 && inputName.front() == '-')
    {
        // Use std::all_of to check if all characters after '-' are digits
        return std::all_of(inputName.begin() + 1, inputName.end(),
            [](char c) {return std::isdigit(static_cast<unsigned char>(c));});
    }
    return false;
}
} // namespace

static void setDimensionNames(ImporterContext* ctx, std::vector<NamedDimension>& namedDims)
{
    for (auto const& namedDim : namedDims)
    {
        std::string const name = isNegativeInteger(namedDim.dimParam) ?
            "trt_dynamic_dim_" + std::to_string(namedDim.index) : namedDim.dimParam;
        namedDim.tensor->setDimensionName(namedDim.index, name.c_str());
    }
}

void importInputs(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, StringMap<TensorOrWeights>* tensors,
    std::vector<Status>& errors)
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
        return;
    }
    setDimensionNames(ctx, namedDims);
}

void importLocalFunctions(ImporterContext* ctx, ::ONNX_NAMESPACE::ModelProto const& model)
{
    for (auto const& localFunction : model.functions())
    {
        ctx->localFunctions().insert({localFunction.name(), localFunction});
    }
}

bool ModelImporter::supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
    SubGraphCollection_t& sub_graph_collection, char const* model_path) noexcept
{
    ONNXTRT_TRY
    {
        bool supports = parse(serialized_onnx_model, serialized_onnx_model_size, model_path);
        reportSubgraphs();
        sub_graph_collection.clear();

        // SubGraphCollection uses size_t, while SubGraphSupportVector_t uses int64_t
        for (const auto& pair : mSubGraphSupportVector)
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
        bool supports = parse(serialized_onnx_model, serialized_onnx_model_size, model_path);
        reportSubgraphs();
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
                     << "). Have you called supports_model_v2 or parse?";
        ONNXTRT_CHECK(mSubGraphSupportVector.size() > static_cast<uint64_t>(index), errorMessage.str(),
            ErrorCode::kINVALID_VALUE);
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
                     << "). Have you called supports_model_v2 or parse?";
        ONNXTRT_CHECK(mSubGraphSupportVector.size() > static_cast<uint64_t>(index), errorMessage.str(),
            ErrorCode::kINVALID_VALUE);
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
        return parse(serialized_onnx_model, serialized_onnx_model_size, nullptr);
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

        deserializeOnnxModel(serialized_onnx_model, serialized_onnx_model_size, &mOnnxModel);
        try
        {
            this->importModel();
        }
        catch (OnnxTrtException& e)
        {
            mErrors.push_back(e.getStatus());
        }
        catch (std::exception& e)
        {
            mErrors.push_back(MAKE_ERROR(e.what(), ErrorCode::kINTERNAL_ERROR));
        }

        logErrors();
        reportSubgraphs();

        return mErrors.size() == 0;
    }
    ONNXTRT_CATCH_RECORD

    return false;
}

void ModelImporter::importModel()
{
    auto* ctx = &mImporterCtx;
    mImporterCtx.clearOpsets();
    // Add domain import limit for security reasons
    int32_t const MAX_DOMAINS = 1024;
    ONNXTRT_CHECK(mOnnxModel.opset_import().size() <= MAX_DOMAINS,
        "Model contains more than 1024 domains! Parsing will halt for security reasons.",
        ErrorCode::kUNSUPPORTED_GRAPH);
    for (int32_t i = 0; i < mOnnxModel.opset_import().size(); ++i)
    {
        std::string domain = mOnnxModel.opset_import(i).domain();
        int64_t version = mOnnxModel.opset_import(i).version();
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

    logModelInfo();

    ::ONNX_NAMESPACE::GraphProto const& graph = mOnnxModel.graph();
    // Create a dummy tensors so that we can reserve output names. If the output names are encountered elsewhere
    // in the graph, the ctx will know to make the names unique.
    for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
    {
        mImporterCtx.registerTensor(TensorOrWeights{}, output.name());
    }

    // Import LocalFunctions
    importLocalFunctions(&mImporterCtx, mOnnxModel);

    // Propagate OnnxParserFlags down to the importer context.
    mImporterCtx.setFlags(getFlags());

    if (getFlag(nvonnxparser::OnnxParserFlag::kREPORT_CAPABILITY_DLA))
    {
        ONNXTRT_CHECK(ctx->getBuilderConfig(),
            "A valid IBuilderConfig must be set when the kREPORT_CAPABILITY_DLA flag is set.",
            ErrorCode::kINVALID_VALUE);
    }

    mCurrentNode = -1;
    importInputs(&mImporterCtx, graph, &mImporterCtx.tensors(), mErrors);
    parseGraph(&mImporterCtx, graph, mErrors, mOnnxModel.producer_name() == "TensorRT", &mCurrentNode);

    mCurrentNode = -1;
    // Mark outputs defined in the ONNX model (unless tensors are user-requested)
    for (::ONNX_NAMESPACE::ValueInfoProto const& output : graph.output())
    {
        ONNXTRT_CHECK((mImporterCtx.tensors().count(output.name())),
            "The output tensor " << output.name() << " was not registered.", ErrorCode::kINVALID_GRAPH);
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
            ONNXTRT_CHECK(output_tensor_ptr, "Failed to add an Identity layer.", ErrorCode::kUNSUPPORTED_NODE);
            output_tensor_ptr->setName(output.name().c_str());
        }

        mImporterCtx.network()->markOutput(*output_tensor_ptr);
        nvinfer1::DataType output_trt_dtype;

        ONNXTRT_CHECK(convertDtype(output.type().tensor_type().elem_type(), &output_trt_dtype),
            "Failed to convert ONNX date type to TensorRT data type.", ErrorCode::kUNSUPPORTED_NODE);
        // For INT32 data type, output type must match tensor type
        ONNXTRT_CHECK((output_tensor_ptr->getType() != nvinfer1::DataType::kINT32
                          || output_trt_dtype == nvinfer1::DataType::kINT32),
            "For INT32 tensors, the output type must also be INT32.", ErrorCode::kUNSUPPORTED_NODE);
        // Note: Without this, output type is always float32
        output_tensor_ptr->setType(output_trt_dtype);
        if (output_trt_dtype == nvinfer1::DataType::kINT64)
        {
            LOG_WARNING("Make sure output " << output.name() << " has Int64 binding.");
        }
    }

    if (mOnnxModel.producer_name() == "TensorRT")
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
            ONNXTRT_CHECK((tensors.count(tensor.first) > 0), "The tensor does not have an assigned location.",
                nvonnxparser::ErrorCode::kINVALID_GRAPH);
            tensors.at(tensor.first)->setLocation(tensor.second);
        }
        // Set dynamic range for all tensors
        for (auto const& tensor : ctx->tensorRangeMins())
        {
            // if there's a min range, there must be a max range as well
            ONNXTRT_CHECK((tensors.count(tensor.first) > 0), "The tensor does not have its dynamic range set.",
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
                ONNXTRT_CHECK((layers.count(layer.first) > 0), "The layer does not have an assigned precision.",
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
}

bool ModelImporter::parseFromFile(char const* onnxModelFile, int32_t verbosity) noexcept
{
    ONNXTRT_TRY
    {
        ONNXTRT_CHECK(onnxModelFile, "Input file cannot be empty.", ErrorCode::kINVALID_VALUE);

        // Define S_ISREG macro for Windows
#if !defined(S_ISREG)
#define S_ISREG(mode) (((mode) & S_IFMT) == S_IFREG)
#endif

        struct stat sb;
        ONNXTRT_CHECK(stat(onnxModelFile, &sb) == 0 && S_ISREG(sb.st_mode),
            "Input file cannot be found, or is not a regular file: " << onnxModelFile, ErrorCode::kINVALID_VALUE);

        GOOGLE_PROTOBUF_VERIFY_VERSION;

        // Own the ONNX model for weights to persist.
        ONNXTRT_CHECK(ParseFromFileAsBinary(&mOnnxModel, onnxModelFile),
            "Cannot read from input file: " << onnxModelFile, ErrorCode::kINVALID_VALUE);

        // Keep track of the absolute path to the ONNX file.
        mImporterCtx.setOnnxFileLocation(onnxModelFile);

        // Prevent failure of importModel from early-exiting
        try
        {
            this->importModel();
        }
        catch (OnnxTrtException& e)
        {
            mErrors.push_back(e.getStatus());
        }
        catch (std::exception& e)
        {
            mErrors.push_back(MAKE_ERROR(e.what(), ErrorCode::kINTERNAL_ERROR));
        }

        logErrors();
        reportSubgraphs();

        return mErrors.size() == 0;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

char const* const* ModelImporter::getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept
{
    nbPluginLibs = mPluginLibraryListCStr.size();
    return (nbPluginLibs > 0) ? mPluginLibraryListCStr.data() : nullptr;
}

bool ModelImporter::loadModelProto(
    void const* serializedOnnxModel, size_t serializedOnnxModelSize, char const* modelPath) noexcept
{
    ONNXTRT_TRY
    {
        auto* ctx = &mImporterCtx;
        if (modelPath)
        {
            ctx->setOnnxFileLocation(modelPath);
        }

        deserializeOnnxModel(serializedOnnxModel, serializedOnnxModelSize, &mOnnxModel);

        // Populate map of initializers for loadInitializers()
        for (::ONNX_NAMESPACE::TensorProto const& initializer : mOnnxModel.graph().initializer())
        {
            ctx->getWeightsContext().initializerMap().insert({initializer.name(), &initializer});
        }
        return true;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

bool ModelImporter::loadInitializer(char const* name, void const* data, size_t size) noexcept
{
    ONNXTRT_TRY
    {
        auto* ctx = &mImporterCtx;
        if (mOnnxModel.ByteSizeLong() == 0)
        {
            LOG_ERROR("An ONNX model has not been loaded yet - cannot load initializer.");
            return false;
        }

        return ctx->getWeightsContext().loadExternalInit(name, data, size);
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

bool ModelImporter::parseModelProto() noexcept
{
    ONNXTRT_TRY
    {
        auto* const ctx = &mImporterCtx;

        if (mOnnxModel.ByteSizeLong() == 0)
        {
            LOG_ERROR("An ONNX model has not been loaded yet - cannot parse empty model.");
            return false;
        }
        // Prevent failure of importModel from early-exiting
        try
        {
            this->importModel();
        }
        catch (OnnxTrtException& e)
        {
            mErrors.push_back(e.getStatus());
        }
        catch (std::exception& e)
        {
            mErrors.push_back(MAKE_ERROR(e.what(), ErrorCode::kINTERNAL_ERROR));
        }
        logErrors();
        reportSubgraphs();

        return mErrors.size() == 0;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}

bool ModelImporter::setBuilderConfig(const nvinfer1::IBuilderConfig* const builderConfig) noexcept
{
    ONNXTRT_TRY
    {
        auto* const ctx = &mImporterCtx;
        if (!builderConfig)
        {
            LOG_ERROR("Provided builder config is nullptr.");
            return false;
        }
        ctx->setBuilderConfig(builderConfig);
        return true;
    }
    ONNXTRT_CATCH_RECORD
    return false;
}
} // namespace onnx2trt
