/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvOnnxParser.h"
#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "TensorOrWeights.hpp"

#include <NvInfer.h>
#include <functional>
#include <onnx/onnx_pb.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>

namespace onnx2trt
{

class IImporterContext;

// TODO: Find ABI-safe alternative approach for this:
//         Can't use std::vector
//         Can't use ::onnx::NodeProto
//         Can't use std::function
typedef ValueOrStatus<std::vector<TensorOrWeights>> NodeImportResult;
typedef std::function<NodeImportResult(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)>
    NodeImporter;

template <typename T>
using StringMap = std::unordered_map<std::string, T>;

class IImporterContext
{
public:
    virtual nvinfer1::INetworkDefinition* network() = 0;
    virtual StringMap<TensorOrWeights>& tensors() = 0;
    virtual StringMap<nvinfer1::TensorLocation>& tensorLocations() = 0;
    virtual StringMap<float>& tensorRangeMins() = 0;
    virtual StringMap<float>& tensorRangeMaxes() = 0;
    virtual StringMap<nvinfer1::DataType>& layerPrecisions() = 0;
    virtual std::unordered_set<std::string>& unsupportedShapeTensors() = 0;
    virtual StringMap<std::string>& loopTensors() = 0;
    virtual void setOnnxFileLocation(std::string location) = 0;
    virtual std::string getOnnxFileLocation() = 0;
    virtual void registerTensor(TensorOrWeights tensor, std::string const& basename, bool const checkUniqueName = false) = 0;

    //! Register a layer, which ensures it has a unique name.
    //! If node!=nullptr, set the metadata for the layer to the node's name.
    virtual void registerLayer(nvinfer1::ILayer* layer, std::string const& basename, ::ONNX_NAMESPACE::NodeProto const* node) = 0;

    //! Short form of register layer to use when the basename is the node's name.
    virtual void registerLayer(nvinfer1::ILayer* layer, ::ONNX_NAMESPACE::NodeProto const& node) = 0;

    virtual ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims shape, uint8_t value = 0) = 0;
    virtual int64_t getOpsetVersion(const char* domain = "") const = 0;
    virtual nvinfer1::ILogger& logger() = 0;
    virtual bool hasError() const = 0;
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const = 0;
    virtual nvinfer1::IConstantLayer* getConstantLayer(const char* name) const = 0;

    virtual void setFlags(nvonnxparser::OnnxParserFlags const& onnxParserFlags) = 0;
    virtual nvonnxparser::OnnxParserFlags getFlags() const = 0;

    //! Push a new scope for base names (ONNX names).
    virtual void pushBaseNameScope() = 0;

    //! Revert actions of registerTensor for names in the top scope and pop it.
    virtual void popBaseNameScope() = 0;

    //! Declare the given node requires a plugin library for the given pluginName, which is provided by the
    //! logical library name pluginLib (should correspond to the DLL/DSO name with suffix and "lib" prefix stripped,
    //! e.g. nvinfer_vc_plugin for libnvinfer_vc_plugin.so.8).
    virtual void addUsedVCPluginLibrary(
        ::ONNX_NAMESPACE::NodeProto const& node, char const* pluginName, char const* pluginLib)
        = 0;

    // Returns a list of strings corresponding to paths to the used VC plugins on disk.  May throw on error.
    virtual std::vector<std::string> getUsedVCPluginLibraries() = 0;

    //! Returns a map of FunctionProto names : Function protos.
    virtual StringMap<::ONNX_NAMESPACE::FunctionProto>& localFunctions() = 0;

    //! Return current list of local functions and corresponding attributes
    virtual std::vector<std::pair<std::string, StringMap<::ONNX_NAMESPACE::AttributeProto const*>>>& localFunctionStack() = 0;

    //! Return output names of the ONNX graph
    virtual std::vector<::ONNX_NAMESPACE::ValueInfoProto>& getGraphOutputNames() = 0;

protected:
    virtual ~IImporterContext() {}
};

} // namespace onnx2trt
