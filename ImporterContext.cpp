/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ImporterContext.hpp"
#include "NvInferVersion.h"
#include "importerUtils.hpp"
#include "weightUtils.hpp"
#include <sstream>

#if !defined(_WIN32)
#include <dlfcn.h>
#if defined(__linux__)
#include <link.h>
#endif
#else // defined(_WIN32)
#include <windows.h>
#endif // !defined(_WIN32)

#define RT_ASSERT(cond)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            throw std::runtime_error("Assertion " #cond " failed!");                                                   \
        }                                                                                                              \
    } while (0)

#define STRINGIFY(x) #x
#define LITERAL(x) STRINGIFY(x)

namespace
{

//! Translates a "logical" library name into an OS-dependent DSO or DLL name
std::string getOSLibraryName(char const* logicalName)
{
    std::stringstream libName;
#if defined(_WIN32)
    libName << logicalName << ".dll";
#else
    libName << "lib" << logicalName << ".so." << NV_TENSORRT_MAJOR;
#endif
    return libName.str();
}

//! Platform-agnostic wrapper around dynamic libraries.
class DynamicLibrary
{
public:
    explicit DynamicLibrary(std::string const& name)
        : mLibName{name}
    {
#if defined(_WIN32)
        mHandle = LoadLibraryA(name.c_str());
#else  // defined(_WIN32)
        int32_t flags{RTLD_LAZY};
        mHandle = dlopen(name.c_str(), flags);
#endif // defined(_WIN32)

        if (mHandle == nullptr)
        {
            std::string errorStr{};
#if !defined(_WIN32)
            errorStr = std::string{" due to "} + std::string{dlerror()};
#endif
            throw std::runtime_error("Unable to open library: " + name + errorStr);
        }
    }

    DynamicLibrary(DynamicLibrary const&) = delete;
    DynamicLibrary(DynamicLibrary const&&) = delete;

    ~DynamicLibrary()
    {
        try
        {
#if defined(_WIN32)
            RT_ASSERT(static_cast<bool>(FreeLibrary(static_cast<HMODULE>(mHandle))));
#else
            RT_ASSERT(dlclose(mHandle) == 0);
#endif
        }
        catch (...)
        {
            std::cerr << "Unable to close library: " << mLibName << std::endl;
        }
    }

    //!
    //! Retrieve a function symbol from the loaded library.
    //!
    //! \return the loaded symbol on success
    //! \throw std::invalid_argument if loading the symbol failed.
    //!
    template <typename Signature>
    std::function<Signature> symbolAddress(char const* name)
    {
        if (mHandle == nullptr)
        {
            throw std::runtime_error("Handle to library is nullptr.");
        }
        void* ret;
#if defined(_MSC_VER)
        ret = static_cast<void*>(GetProcAddress(static_cast<HMODULE>(mHandle), name));
#else
        ret = dlsym(mHandle, name);
#endif
        if (ret == nullptr)
        {
            std::string const kERROR_MSG(mLibName + ": error loading symbol: " + std::string(name));
            throw std::invalid_argument(kERROR_MSG);
        }
        return reinterpret_cast<Signature*>(ret);
    }

    std::string getFullPath() const
    {
        RT_ASSERT(mHandle != nullptr);
#if defined(__linux__)
        link_map* linkMap = nullptr;
        auto const err = dlinfo(mHandle, RTLD_DI_LINKMAP, &linkMap);
        RT_ASSERT(err == 0 && linkMap != nullptr && linkMap->l_name != nullptr);
        return std::string{linkMap->l_name};
#elif defined(_WIN32)
        constexpr int32_t kMAX_PATH_LEN{4096};
        std::string path(kMAX_PATH_LEN, '\0'); // since C++11, std::string storage is guaranteed to be contiguous
        auto const pathLen = GetModuleFileNameA(static_cast<HMODULE>(mHandle), &path[0], kMAX_PATH_LEN);
        RT_ASSERT(GetLastError() == ERROR_SUCCESS);
        path.resize(pathLen);
        path.shrink_to_fit();
        return path;
#else
        RT_ASSERT(!"Unsupported operation: getFullPath()");
#endif
    }

private:
    std::string mLibName{}; //!< Name of the DynamicLibrary
    void* mHandle{};        //!< Handle to the DynamicLibrary
};

//! Translates an OS-dependent DSO/DLL name into a path on the filesystem
inline std::string getOSLibraryPath(std::string const& osLibName)
{
    DynamicLibrary lib{osLibName};
    return lib.getFullPath();
}

} // namespace

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
    std::string const& uniqueName = generateUniqueName(mTensorNames, mSuffixCounter, basename);

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

void ImporterContext::registerLayer(
    nvinfer1::ILayer* layer, std::string const& basename, ::ONNX_NAMESPACE::NodeProto const* node)
{
    // No layer will be added for Constant nodes in ONNX.
    if (layer)
    {
        std::string const name = basename.empty() ? layer->getName() : basename;
        std::string const& uniqueName = generateUniqueName(mLayerNames, mSuffixCounter, basename);

        auto* ctx = this; // To enable logging.
        if (node != nullptr)
        {
            LOG_VERBOSE("Registering layer: " << uniqueName << " for ONNX node: " << basename);
        }
        else
        {
            LOG_VERBOSE("Registering layer: " << uniqueName << " required by ONNX-TRT");
        }

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
    // Set metadata only if the layer is associated with an ONNX node.
    // Skip constant layers because constants are represented as initializers in ONNX and should not be associated
    // with any ONNX node.
    if (node != nullptr && layer != nullptr && layer->getType() != nvinfer1::LayerType::kCONSTANT)
    {
        processMetadata(this, *node, layer);
    }
}

void ImporterContext::registerLayer(nvinfer1::ILayer* layer, ::ONNX_NAMESPACE::NodeProto const& node)
{
    registerLayer(layer, getNodeName(node), &node);
}

void ImporterContext::registerAttention(
    nvinfer1::IAttention* attention, std::string const& basename, ::ONNX_NAMESPACE::NodeProto const* node)
{
    if (attention)
    {
        std::string const name = basename.empty() ? attention->getName() : basename;
        std::string const& uniqueName = generateUniqueName(mLayerNames, mSuffixCounter, basename);

        auto* ctx = this; // Logging macro uses ctx.
        if (node != nullptr)
        {
            LOG_VERBOSE("Registering attention: " << uniqueName << " for ONNX node: " << basename);
        }
        else
        {
            LOG_VERBOSE("Registering attention: " << uniqueName << " required by ONNX-TRT");
        }

        attention->setName(uniqueName.c_str());
    }

    // Set metadata if the attention is associated with an ONNX node.
    if (node != nullptr && attention != nullptr)
    {
        processMetadata(this, *node, attention);
    }
}

void ImporterContext::registerAttention(nvinfer1::IAttention* attention, ::ONNX_NAMESPACE::NodeProto const& node)
{
    registerAttention(attention, getNodeName(node), &node);
}

void ImporterContext::addUsedVCPluginLibrary(
    ::ONNX_NAMESPACE::NodeProto const& node, char const* pluginName, char const* pluginLib)
{
    auto* ctx = this; // For logging
    auto osPluginLibName = getOSLibraryName(pluginLib);
    LOG_VERBOSE("Node " << getNodeName(node) << " requires plugin " << pluginName << " which is provided by "
                        << osPluginLibName);
    mLogicalVCPluginLibraries.insert(osPluginLibName);
}

std::vector<std::string> ImporterContext::getUsedVCPluginLibraries()
{
    auto* ctx = this; // For logging
#if defined(_WIN32) || defined(__linux__)
    std::vector<std::string> ret;
    ret.reserve(mLogicalVCPluginLibraries.size());
    for (auto const& l : mLogicalVCPluginLibraries)
    {
        auto osLibPath = getOSLibraryPath(l);
        LOG_VERBOSE("Library " << l << " located on filesystem as " << osLibPath);
        ret.emplace_back(std::move(osLibPath));
    }
    return ret;
#else
    LOG_WARNING("getUsedVCPluginLibraries not implemented on platform!");
    return {};
#endif
}

void ImporterContext::checkDLASupport(int32_t numPrevLayers, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIndex)
{
    for (int32_t i = numPrevLayers; i < network()->getNbLayers(); ++i)
    {
        auto* layer = network()->getLayer(i);
        if (!mBuilderConfig->canRunOnDLA(layer))
        {
            ONNXTRT_THROW(MAKE_NODE_ERROR("DLA validation failed for layer: " + std::string(layer->getName()), ErrorCode::kUNSUPPORTED_NODE, node, nodeIndex));
        }
    }
}

[[nodiscard]] bool ImporterContext::getDLACapabilityMode() const
{
    return mOnnxParserFlags & (1U << static_cast<uint32_t>(nvonnxparser::OnnxParserFlag::kREPORT_CAPABILITY_DLA));
}

} // namespace onnx2trt

