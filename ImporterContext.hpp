/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvOnnxParser.h"
#include "ShapedWeights.hpp"
#include "Status.hpp"
#include "TensorOrWeights.hpp"
#include "WeightsContext.hpp"
#include "onnxErrorRecorder.hpp"
#include <fstream>
#include <functional>
#include <list>
#include <onnx/onnx_pb.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnx2trt
{

template <typename T>
using StringMap = std::unordered_map<std::string, T>;

class ErrorRecorderWrapper
{
public:
    ErrorRecorderWrapper(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : mNetwork(network)
    {
        if (mNetwork)
        {
            mUserErrorRecorder = mNetwork->getErrorRecorder();
            mOnnxErrorRecorder = ONNXParserErrorRecorder::create(logger, mUserErrorRecorder);
            if (mOnnxErrorRecorder)
            {
                if (mUserErrorRecorder)
                {
                    mUserErrorRecorder->incRefCount();
                }
                mNetwork->setErrorRecorder(mOnnxErrorRecorder);
            }
        }
    }

    ~ErrorRecorderWrapper()
    {
        if (mNetwork && mOnnxErrorRecorder)
        {
            if (mUserErrorRecorder)
            {
                mNetwork->setErrorRecorder(mUserErrorRecorder);
                mUserErrorRecorder->decRefCount();
            }
            ONNXParserErrorRecorder::destroy(mOnnxErrorRecorder);
        }
    }

    bool hasError() const
    {
        return mOnnxErrorRecorder != nullptr && mOnnxErrorRecorder->getNbErrors() != 0;
    }

    //! Return recorder used by hasError().
    nvinfer1::IErrorRecorder* getErrorRecorder() const
    {
        return mOnnxErrorRecorder ? mOnnxErrorRecorder : nullptr;
    }

private:
    nvinfer1::INetworkDefinition* mNetwork{nullptr};
    ONNXParserErrorRecorder* mOnnxErrorRecorder{nullptr};
    nvinfer1::IErrorRecorder* mUserErrorRecorder{nullptr};
};

class ImporterContext
{
    nvinfer1::INetworkDefinition* mNetwork;
    nvinfer1::ILogger* mLogger;
    nvinfer1::IBuilderConfig const* mBuilderConfig{nullptr};
    //! WeightsContext object to hold ownership of ONNX weights and any temporary weights created by the Parser.
    WeightsContext mWeightsContext;
    StringMap<int64_t> mOpsets;
    //! All tensors in the graph mapped to their names.
    StringMap<TensorOrWeights> mTensors;
    StringMap<nvinfer1::TensorLocation> mTensorLocations;
    StringMap<float> mTensorRangeMins;
    StringMap<float> mTensorRangeMaxes;
    StringMap<nvinfer1::DataType> mLayerPrecisions;
    //! Set to keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    std::set<std::string> mTensorNames;
    //! Set to keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    std::set<std::string> mLayerNames;
    //! An increasing suffix counter used to uniquify layer names.
    int64_t mSuffixCounter{0};
    //! Set to keep track of how many times a refittable name created by the parser shows up, to avoid duplicate naming in TRT.
    //! Currently tracks the following nodes:
    //!     1. BatchNorm - Parser pre-combines scales and bias weights for the IScaleLayer.
    //!     2. ConstantOfShape - The value of the ConstantOfShape does not have a name, so the parser needs to create one for it.
    std::set<std::string> mTempRefittableWeights;
    //! An increasing suffix counter used to uniquify refittable weight names created by the parser.
    int64_t mTempRefittableWeightsSuffixCounter{0};
    //! Set to hold output tensor names of layers that produce shape tensor outputs but do not
    //! natively support them.
    std::unordered_set<std::string> mUnsupportedShapeTensors;
    //! Container to map subgraph tensors to their original outer graph names.
    StringMap<std::string> mLoopTensors;
    //! Error recorder to control TRT errors.
    std::unique_ptr<ErrorRecorderWrapper> mErrorWrapper;
    StringMap<nvinfer1::IConstantLayer*> mConstantLayers;
    bool mConvertINT64Logged{false};
    bool mConvertINT64OutOfBoundsLogged{false};
    bool mConvertDoubleLogged{false};
    bool mConvertDoubleOutOfBoundsLogged{false};
    //! OnnxParserFlags specified by the parser.
    nvonnxparser::OnnxParserFlags mOnnxParserFlags;
    StringMap<std::vector<nvinfer1::ITensor const*>> mNodeNameToTensor;

    //! Logical library names for VC plugin libraries. This gets translated to library paths
    //! when getUsedVCPluginLibraries() is called.
    std::set<std::string> mLogicalVCPluginLibraries;

    //! Stack of names defined by nested ONNX graphs, with information about how to
    //! restore their associated values when popping back to the surrounding scope.
    //!
    //! The stack is empty when processing the top-level ONNX graph.
    //! back() corresponds to the innermost ONNX graph being processed.
    //!
    //! For each entry {name, {bool, TensorOrWeights}}:
    //!
    //! * If the bool is true, the name was newly introduced by the scope.
    //!
    //! * If the bool is false, the name shadows a name in a surrounding scope,
    //!   and TensorOrWeights was the name's value before being shadowed.
    //!
    std::vector<StringMap<std::pair<bool, TensorOrWeights>>> mBaseNameScopeStack;

    //! Map holding FunctionProtos
    StringMap<::ONNX_NAMESPACE::FunctionProto> mLocalFunctions;

    //! Data type to keep track of a local function in mLocalFunctionStack.
    //! It is a tuple of three elements: (1) function name (2) node name and (3) function attributes.
    struct LocalFunctionMetadata
    {
        std::string functionName;
        std::string nodeName;
        StringMap<::ONNX_NAMESPACE::AttributeProto const*> attrs;
    };

    //! Vector to hold current local function names and attributes
    std::vector<LocalFunctionMetadata> mLocalFunctionStack;

    //! Vector to hold the local function names at each error
    std::vector<std::vector<std::string>> mLocalFunctionErrors;

    //! Vector to hold expected graph outputs
    std::vector<::ONNX_NAMESPACE::ValueInfoProto> mGraphOutputNames;

public:
    ImporterContext(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : mNetwork(network)
        , mLogger(logger)
        , mWeightsContext(WeightsContext(logger))
        , mErrorWrapper(std::make_unique<ErrorRecorderWrapper>(mNetwork, logger))
    {
    }
    nvinfer1::INetworkDefinition* network()
    {
        assert(mNetwork != nullptr);
        return mNetwork;
    }
    WeightsContext& getWeightsContext()
    {
        return mWeightsContext;
    }
    StringMap<TensorOrWeights>& tensors()
    {
        return mTensors;
    }
    StringMap<nvinfer1::TensorLocation>& tensorLocations()
    {
        return mTensorLocations;
    }
    StringMap<float>& tensorRangeMins()
    {
        return mTensorRangeMins;
    }
    StringMap<float>& tensorRangeMaxes()
    {
        return mTensorRangeMaxes;
    }
    StringMap<nvinfer1::DataType>& layerPrecisions()
    {
        return mLayerPrecisions;
    }
    std::unordered_set<std::string>& unsupportedShapeTensors()
    {
        return mUnsupportedShapeTensors;
    }
    StringMap<std::string>& loopTensors()
    {
        return mLoopTensors;
    }
    // Pass file location down to WeightsContext as all external weight handling logic is done in that class.
    void setOnnxFileLocation(std::string location)
    {
        mWeightsContext.setOnnxFileLocation(location);
    }
    void pushBaseNameScope();

    void popBaseNameScope();

    // This actually handles weights as well, but is named this way to be consistent with the tensors()
    void registerTensor(TensorOrWeights tensor, std::string const& basename, bool const checkUniqueName = false);

    void registerLayer(nvinfer1::ILayer* layer, std::string const& basename, ::ONNX_NAMESPACE::NodeProto const* node);
    void registerLayer(nvinfer1::ILayer* layer, ::ONNX_NAMESPACE::NodeProto const& node);

    void registerAttention(
        nvinfer1::IAttention* attention, std::string const& basename, ::ONNX_NAMESPACE::NodeProto const* node);
    void registerAttention(nvinfer1::IAttention* attention, ::ONNX_NAMESPACE::NodeProto const& node);

    nvinfer1::ILogger& logger()
    {
        return *mLogger;
    }

    // Register an unique name for the created weights
    ShapedWeights createNamedTempWeights(ShapedWeights::DataType type, nvinfer1::Dims shape, bool refittable = false)
    {
        if (refittable)
        {
            return mWeightsContext.createNamedTempWeights(
                type, shape, mTempRefittableWeights, mTempRefittableWeightsSuffixCounter, /*refittable=*/true);
        }
        return mWeightsContext.createNamedTempWeights(type, shape, mTensorNames, mSuffixCounter);
    }

    void clearOpsets()
    {
        mOpsets.clear();
    }
    void addOpset(std::string domain, int64_t version)
    {
        mOpsets.emplace(domain, version);
    }
    int64_t getOpsetVersion(const char* domain = "") const
    {
        if (mOpsets.empty())
        {
            return 1;
        }
        else if (mOpsets.size() == 1)
        {
            return mOpsets.begin()->second;
        }
        else if (mOpsets.count(domain))
        {
            return mOpsets.at(domain);
        }
        else
        {
            domain = "ai.onnx";
            assert(mOpsets.count(domain));
            return mOpsets.at(domain);
        }
    }
    bool hasError() const noexcept
    {
        return mErrorWrapper != nullptr && mErrorWrapper->hasError();
    }

    nvinfer1::IErrorRecorder* getErrorRecorder() const noexcept
    {
        return mErrorWrapper ? mErrorWrapper->getErrorRecorder() : nullptr;
    }
    nvinfer1::IConstantLayer* getConstantLayer(const char* name) const
    {
        if (name == nullptr)
        {
            return nullptr;
        }
        auto const iter = mConstantLayers.find(name);
        if (iter == mConstantLayers.end())
        {
            return nullptr;
        }
        return iter->second;
    }

    void setFlags(nvonnxparser::OnnxParserFlags const& onnxParserFlags)
    {
        mOnnxParserFlags = onnxParserFlags;
    }
    nvonnxparser::OnnxParserFlags getFlags() const
    {
        return mOnnxParserFlags;
    }

    virtual void addUsedVCPluginLibrary(
        ::ONNX_NAMESPACE::NodeProto const& node, char const* pluginName, char const* pluginLib);

    virtual std::vector<std::string> getUsedVCPluginLibraries();

    bool isConvertINT64Logged()
    {
        return mConvertINT64Logged;
    }
    void setConvertINT64Logged(bool logged)
    {
        mConvertINT64Logged = logged;
    }
    bool isConvertINT64OutOfBoundsLogged()
    {
        return mConvertINT64OutOfBoundsLogged;
    }
    void setConvertINT64OutOfBoundsLogged(bool logged)
    {
        mConvertINT64OutOfBoundsLogged = logged;
    }
    bool isConvertDoubleLogged()
    {
        return mConvertDoubleLogged;
    }
    void setConvertDoubleLogged(bool logged)
    {
        mConvertDoubleLogged = logged;
    }
    bool isConvertDoubleOutOfBoundsLogged()
    {
        return mConvertDoubleOutOfBoundsLogged;
    }
    void setConvertDoubleOutOfBoundsLogged(bool logged)
    {
        mConvertDoubleOutOfBoundsLogged = logged;
    }
    StringMap<::ONNX_NAMESPACE::FunctionProto>& localFunctions()
    {
        return mLocalFunctions;
    }
    std::vector<LocalFunctionMetadata>& localFunctionStack()
    {
        return mLocalFunctionStack;
    }
    std::vector<std::vector<std::string>>& localFunctionErrors()
    {
        return mLocalFunctionErrors;
    }
    std::vector<::ONNX_NAMESPACE::ValueInfoProto>& getGraphOutputNames()
    {
        return mGraphOutputNames;
    }
    nvinfer1::ITensor const* findLayerOutputTensor(std::string name, int64_t i)
    {
        auto it = mNodeNameToTensor.find(name);
        if (it == mNodeNameToTensor.end())
        {
            return nullptr;
        }
        auto tensors = it->second;
        return i < static_cast<int64_t>(tensors.size()) ? tensors.at(i) : nullptr;
    }
    void addLayerOutputTensors(std::string name, std::vector<TensorOrWeights> const& outputs)
    {
        static std::unordered_set<std::string> duplicateNames;
        if (mNodeNameToTensor.find(name) != mNodeNameToTensor.end())
        {
            // Log once (only if insertion succeeded) for each unique name.
            auto result = duplicateNames.insert(name);
            if (result.second)
            {
                auto* ctx = this; // For logging
                LOG_WARNING(
                    "A node named " << name
                                    << " already exists, the output tensors of this new instance will not be queryable.");
            }
            return;
        }
        for (auto const& output : outputs)
        {
            if (output.is_tensor())
            {
                mNodeNameToTensor[name].push_back(static_cast<nvinfer1::ITensor const*>(&(output.tensor())));
            }
        }
    }
    size_t getNestedDepth()
    {
        return mBaseNameScopeStack.size();
    }

    // Returns if the underlying network was created with the KSTRONGLY_TYPED flag.
    bool isStronglyTyped()
    {
        assert(mNetwork != nullptr);
        return mNetwork->getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    }
    void setBuilderConfig(nvinfer1::IBuilderConfig const* builderConfig)
    {
        mBuilderConfig = builderConfig;
    }
    nvinfer1::IBuilderConfig const* getBuilderConfig() const
    {
        return mBuilderConfig;
    }

    // Called after each node is parsed to validate that added layers are supported on DLA.
    // When kREPORT_CAPABILITY_DLA flag is set, it's expected that the parser will run layer validation.
    void checkDLASupport(int32_t numPrevLayers, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIndex);

    [[nodiscard]] bool getDLACapabilityMode() const;
};

typedef std::vector<TensorOrWeights> NodeOutputs;
typedef std::function<NodeOutputs(ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, size_t const nodeIdx,
    std::vector<TensorOrWeights>& inputs)>
    NodeImporter;

typedef std::function<void(
    ImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<Status>& errors, size_t const nodeIndex)>
    OpStaticErrorChecker;

} // namespace onnx2trt

