/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ImporterContext.hpp"

namespace onnx2trt
{

void ImporterContext::pushBaseNameScope()
{
    mBaseNameScopeStack.push_back({});
}

void ImporterContext::popBaseNameScope()
{
    auto& tensorMap = tensors();
    for (auto& binding : mBaseNameScopeStack.back())
    {
        if (binding.second.first)
        {
            tensorMap.erase(binding.first);
        }
        else
        {
            tensorMap.at(binding.first) = std::move(binding.second.second);
        }
    }
    mBaseNameScopeStack.pop_back();
}

void ImporterContext::registerTensor(TensorOrWeights tensor, std::string const& basename, bool const checkUniqueName)
{
    // TRT requires unique tensor names.
    std::string const& uniqueName = generateUniqueName(mTensorNames, basename);

    if (tensor)
    {
        if (tensor.is_tensor())
        {
            tensor.tensor().setName(uniqueName.c_str());
            // Logging macro refers to ctx.
            auto* ctx = this;
            LOG_VERBOSE("Registering tensor: " << uniqueName << " for ONNX tensor: " << basename);
        }
        else if (tensor.is_weights())
        {
            auto const& weights = tensor.weights();
            if (tensor.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
            {
                tensor = ShapedWeights{::ONNX_NAMESPACE::TensorProto::INT32,
                    convertINT64(reinterpret_cast<int64_t*>(weights.values), weights.shape, this), weights.shape};
            }
            // It may be possible for nested subgraphs to have different values for the same initializer.
            // For multiple name scopes - use unique name to keep track of weights.
            if (!mBaseNameScopeStack.empty())
            {
                tensor.weights().setName(uniqueName.c_str());
            }
            else
            {
                tensor.weights().setName(basename.c_str());
            }
        }
    }

    std::string const& nameToCheck = checkUniqueName ? uniqueName : basename;

    auto const p = this->tensors().emplace(nameToCheck, TensorOrWeights{});
    bool nameIsDuplicate = false;
    if (!mBaseNameScopeStack.empty())
    {
        // Remember original binding so it can be restored when scope is popped.
        auto const q
            = mBaseNameScopeStack.back().emplace(nameToCheck, std::make_pair(p.second, std::move(p.first->second)));
        // Check that scope did not already have a binding for basename.
        nameIsDuplicate = !q.second;
    }
    else
    {
        // The condition here accounts for ModelImporter::importModel reserving
        // output names by registering null tensors.
        nameIsDuplicate = !p.second && !p.first->second.isNullTensor();
    }
    if (nameIsDuplicate)
    {
        throw std::runtime_error("ONNX graph has duplicate tensor name: " + nameToCheck);
    }
    p.first->second = std::move(tensor);
}

void ImporterContext::registerLayer(nvinfer1::ILayer* layer, std::string const& basename)
{
    // No layer will be added for Constant nodes in ONNX.
    if (layer)
    {
        std::string const name = basename.empty() ? layer->getName() : basename;
        std::string const& uniqueName = generateUniqueName(mLayerNames, name);

        auto* ctx = this; // To enable logging.
        LOG_VERBOSE("Registering layer: " << uniqueName << " for ONNX node: " << basename);

        layer->setName(uniqueName.c_str());
        if (layer->getType() == nvinfer1::LayerType::kCONSTANT)
        {
            if (basename != uniqueName && mConstantLayers.find(uniqueName) != mConstantLayers.end())
            {
                LOG_ERROR("Constant layer: " << uniqueName << " can be a duplicate of: " << basename);
                assert(!"Internal error: duplicate constant layers for the same weights");
            }
            mConstantLayers.insert({uniqueName, static_cast<nvinfer1::IConstantLayer*>(layer)});
        }
    }
}

} // namespace onnx2trt
