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
    virtual void registerTensor(TensorOrWeights tensor, const std::string& basename, bool const checkUniqueName = false)
        = 0;
    virtual void registerLayer(nvinfer1::ILayer* layer, const std::string& basename) = 0;
    virtual ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims shape, uint8_t value = 0) = 0;
    virtual int64_t getOpsetVersion(const char* domain = "") const = 0;
    virtual nvinfer1::ILogger& logger() = 0;
    virtual bool hasError() const = 0;
    virtual nvinfer1::IErrorRecorder* getErrorRecorder() const = 0;
    virtual nvinfer1::IConstantLayer* getConstantLayer(const char* name) const = 0;

    //! Push a new scope for base names (ONNX names).
    virtual void pushBaseNameScope() = 0;

    //! Revert actions of registerTensor for names in the top scope and pop it.
    virtual void popBaseNameScope() = 0;
protected:
    virtual ~IImporterContext() {}
};

} // namespace onnx2trt
