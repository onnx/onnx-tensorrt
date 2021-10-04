/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ImporterContext.hpp"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "builtin_op_importers.hpp"
#include "utils.hpp"

namespace onnx2trt
{

Status parseGraph(IImporterContext* ctx, const ::ONNX_NAMESPACE::GraphProto& graph, bool deserializingINetwork = false, int* currentNode = nullptr);

class ModelImporter : public nvonnxparser::IParser
{
protected:
    string_map<NodeImporter> _op_importers;
    virtual Status importModel(::ONNX_NAMESPACE::ModelProto const& model);

private:
    ImporterContext _importer_ctx;
    std::list<::ONNX_NAMESPACE::ModelProto> _onnx_models; // Needed for ownership of weights
    int _current_node;
    std::vector<Status> _errors;

public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : _op_importers(getBuiltinOpImporterMap())
        , _importer_ctx(network, logger)
    {
    }
    bool parseWithWeightDescriptors(void const* serialized_onnx_model, size_t serialized_onnx_model_size) override;
    bool parse(void const* serialized_onnx_model, size_t serialized_onnx_model_size, const char* model_path = nullptr) override;
    bool supportsModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        SubGraphCollection_t& sub_graph_collection, const char* model_path = nullptr) override;

    bool supportsOperator(const char* op_name) const override;
    void destroy() override
    {
        delete this;
    }
    int getNbErrors() const override
    {
        return _errors.size();
    }
    nvonnxparser::IParserError const* getError(int index) const override
    {
        assert(0 <= index && index < (int) _errors.size());
        return &_errors[index];
    }
    void clearErrors() override
    {
        _errors.clear();
    }

    //...LG: Move the implementation to .cpp
    bool parseFromFile(const char* onnxModelFile, int verbosity) override;
};

} // namespace onnx2trt
