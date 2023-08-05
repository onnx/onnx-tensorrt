/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "builtin_op_importers.hpp"

namespace onnx2trt
{

Status parseNode(IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, bool deserializingINetwork = false);

Status parseGraph(IImporterContext* ctx, ::ONNX_NAMESPACE::GraphProto const& graph, bool deserializingINetwork = false,
    int32_t* currentNode = nullptr);

class ModelImporter : public nvonnxparser::IParser
{
protected:
    StringMap<NodeImporter> _op_importers;
    virtual Status importModel(::ONNX_NAMESPACE::ModelProto const& model);

private:
    ImporterContext mImporterCtx;
    std::vector<std::string> mPluginLibraryList; // Array of strings containing plugin libs
    std::vector<char const*>
        mPluginLibraryListCStr; // Array of C-strings corresponding to the strings in mPluginLibraryList
    std::list<::ONNX_NAMESPACE::ModelProto> mONNXModels; // Needed for ownership of weights
    int mCurrentNode;
    std::vector<Status> mErrors;
    nvonnxparser::OnnxParserFlags mOnnxParserFlags{0};

public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : _op_importers(getBuiltinOpImporterMap())
        , mImporterCtx(network, logger)
    {
    }
    bool parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) override;
    bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path = nullptr) override;
    bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) override;

    bool supportsOperator(const char* op_name) const override;

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
        mOnnxParserFlags &= ~(1U << static_cast<uint32_t>(onnxParserFlag));
    }

    void setFlag(nvonnxparser::OnnxParserFlag onnxParserFlag) noexcept override
    {
        mOnnxParserFlags |= 1U << static_cast<uint32_t>(onnxParserFlag);
    }

    bool getFlag(nvonnxparser::OnnxParserFlag onnxParserFlag) const noexcept override
    {
        auto flag = 1U << static_cast<uint32_t>(onnxParserFlag);
        return static_cast<bool>(mOnnxParserFlags & flag);
    }

    void destroy() override
    {
        delete this;
    }
    int32_t getNbErrors() const override
    {
        return mErrors.size();
    }
    nvonnxparser::IParserError const* getError(int32_t index) const override
    {
        assert(0 <= index && index < (int32_t) mErrors.size());
        return &mErrors[index];
    }
    void clearErrors() override
    {
        mErrors.clear();
    }

    bool parseFromFile(char const* onnxModelFile, int32_t verbosity) override;

    virtual char const* const* getUsedVCPluginLibraries(int64_t& nbPluginLibs) const noexcept override;
};

} // namespace onnx2trt
