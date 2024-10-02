/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "errorHelpers.hpp"
#include "onnxOpCheckers.hpp"
#include "onnxOpImporters.hpp"
#include <stdexcept>

namespace onnx2trt
{

void parseNode(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    bool deserializingINetwork = false);

void parseNodeStaticCheck(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors, size_t const nodeIndex);

void parseGraph(ImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, std::vector<Status>& errors,
    bool deserializingINetwork = false, int32_t* currentNode = nullptr);

class ModelImporter : public nvonnxparser::IParser
{
    using SubGraphSupport_t = std::pair<std::vector<int64_t>, bool>;
    using SubGraphSupportVector_t = std::vector<SubGraphSupport_t>;

protected:
    StringMap<NodeImporter> _op_importers;
    virtual void importModel(::ONNX_NAMESPACE::ModelProto const& model);

private:
    ImporterContext mImporterCtx;
    std::vector<std::string> mPluginLibraryList; // Array of strings containing plugin libs
    std::vector<char const*>
        mPluginLibraryListCStr; // Array of C-strings corresponding to the strings in mPluginLibraryList
    std::list<::ONNX_NAMESPACE::ModelProto> mONNXModels; // Needed for ownership of weights
    SubGraphSupportVector_t mSubGraphSupportVector;
    int mCurrentNode;
    mutable std::vector<Status> mErrors; // Marked as mutable so that errors could be reported from const functions
    nvonnxparser::OnnxParserFlags mOnnxParserFlags{
        1U << static_cast<uint32_t>(
            nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM)}; // kNATIVE_INSTANCENORM is ON by default.
    std::pair<bool, SubGraphSupportVector_t> doSupportsModel(
        void const* serialized_onnx_model, size_t serialized_onnx_model_size, char const* model_path = nullptr);

public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger) noexcept
        : _op_importers(getBuiltinOpImporterMap())
        , mImporterCtx(network, logger)
    {
    }
    bool parseWithWeightDescriptors(
        void const* serialized_onnx_model, size_t serialized_onnx_model_size) noexcept override;
    bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        const char* model_path = nullptr) noexcept override;

    bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) noexcept override;
    bool supportsModelV2(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        char const* model_path = nullptr) noexcept override;

    int64_t getNbSubgraphs() noexcept override;
    bool isSubgraphSupported(int64_t const index) noexcept override;
    int64_t* getSubgraphNodes(int64_t const index, int64_t& subgraphLength) noexcept override;

    bool supportsOperator(const char* op_name) const noexcept override;

    void setFlags(nvonnxparser::OnnxParserFlags onnxParserFlags) noexcept override
    {
        mOnnxParserFlags = onnxParserFlags;
    }
    nvonnxparser::OnnxParserFlags getFlags() const noexcept override
    {
        return mOnnxParserFlags;
    }

    void clearFlag(nvonnxparser::OnnxParserFlag onnxParserFlag) noexcept override
    {
        ONNXTRT_TRY
        {
            mOnnxParserFlags &= ~(1U << static_cast<uint32_t>(onnxParserFlag));
        }
        ONNXTRT_CATCH_RECORD
    }

    void setFlag(nvonnxparser::OnnxParserFlag onnxParserFlag) noexcept override
    {
        ONNXTRT_TRY
        {
            mOnnxParserFlags |= 1U << static_cast<uint32_t>(onnxParserFlag);
        }
        ONNXTRT_CATCH_RECORD
    }

    bool getFlag(nvonnxparser::OnnxParserFlag onnxParserFlag) const noexcept override
    {
        ONNXTRT_TRY
        {
            auto flag = 1U << static_cast<uint32_t>(onnxParserFlag);
            return static_cast<bool>(mOnnxParserFlags & flag);
        }
        ONNXTRT_CATCH_RECORD
        return false;
    }

    int32_t getNbErrors() const noexcept override
    {
        return mErrors.size();
    }
    nvonnxparser::IParserError const* getError(int32_t index) const noexcept override
    {
        ONNXTRT_TRY
        {
            return &mErrors.at(index);
        }
        ONNXTRT_CATCH_RECORD
        return nullptr;
    }
    void clearErrors() noexcept override
    {
        mErrors.clear();
    }

    nvinfer1::ITensor const* getLayerOutputTensor(char const* name, int64_t i) noexcept override
    {
        ONNXTRT_TRY
        {
            if (!name)
            {
                throw std::invalid_argument("name is a nullptr");
            }
            return mImporterCtx.findLayerOutputTensor(name, i);
        }
        ONNXTRT_CATCH_RECORD
        return nullptr;
    }

    bool parseFromFile(char const* onnxModelFile, int32_t verbosity) noexcept override;

    virtual char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept override;
};

} // namespace onnx2trt
