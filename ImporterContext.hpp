/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx2trt.hpp"
#include "onnx2trt_utils.hpp"
#include "onnxErrorRecorder.hpp"
#include "onnx/common/stl_backports.h"
#include <list>
#include <unordered_map>

namespace onnx2trt
{

class ErrorRecorderWrapper
{
public:
    ErrorRecorderWrapper(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : mNetwork(network)
        , mLogger(logger)
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
    nvinfer1::ILogger* mLogger{nullptr};
    ONNXParserErrorRecorder* mOnnxErrorRecorder{nullptr};
    nvinfer1::IErrorRecorder* mUserErrorRecorder{nullptr};
};

class ImporterContext final : public IImporterContext
{
    nvinfer1::INetworkDefinition* mNetwork;
    nvinfer1::ILogger* mLogger;
    std::list<std::vector<uint8_t>> mTempBufs;
    StringMap<nvinfer1::ITensor*> mUserInputs;
    StringMap<nvinfer1::ITensor**> mUserOutputs;
    StringMap<int64_t> mOpsets;
    StringMap<TensorOrWeights> mTensors; // All tensors in the graph mapped to their names.
    StringMap<nvinfer1::TensorLocation> mTensorLocations;
    StringMap<float> mTensorRangeMins;
    StringMap<float> mTensorRangeMaxes;
    StringMap<nvinfer1::DataType> mLayerPrecisions;
    std::set<std::string> mTensorNames; // Keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    std::set<std::string> mLayerNames; // Keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    int64_t mSuffixCounter{0}; // increasing suffix counter used to uniquify layer names.
    std::unordered_set<std::string> mUnsupportedShapeTensors; // Container to hold output tensor names of layers that produce shape tensor outputs but do not natively support them.
    StringMap<std::string> mLoopTensors; // Container to map subgraph tensors to their original outer graph names.
    std::string mOnnxFileLocation;       // Keep track of the directory of the parsed ONNX file
    std::unique_ptr<ErrorRecorderWrapper> mErrorWrapper; // error recorder to control TRT errors
    StringMap<nvinfer1::IConstantLayer*> mConstantLayers;

public:
    ImporterContext(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : mNetwork(network)
        , mLogger(logger)
        , mErrorWrapper(ONNX_NAMESPACE::make_unique<ErrorRecorderWrapper>(mNetwork, logger))
    {
    }
    nvinfer1::INetworkDefinition* network() override
    {
        return mNetwork;
    }
    StringMap<TensorOrWeights>& tensors() override
    {
        return mTensors;
    }
    StringMap<nvinfer1::TensorLocation>& tensorLocations() override
    {
        return mTensorLocations;
    }
    StringMap<float>& tensorRangeMins() override
    {
        return mTensorRangeMins;
    }
    StringMap<float>& tensorRangeMaxes() override
    {
        return mTensorRangeMaxes;
    }
    StringMap<nvinfer1::DataType>& layerPrecisions() override
    {
        return mLayerPrecisions;
    }
    std::unordered_set<std::string>& unsupportedShapeTensors() override
    {
        return mUnsupportedShapeTensors;
    }
    StringMap<std::string>& loopTensors() override
    {
        return mLoopTensors;
    }
    void setOnnxFileLocation(std::string location) override
    {
        mOnnxFileLocation = location;
    }
    std::string getOnnxFileLocation() override
    {
        return mOnnxFileLocation;
    }
    // This actually handles weights as well, but is named this way to be consistent with the tensors()
    void registerTensor(TensorOrWeights tensor, const std::string& basename) override
    {
        // TRT requires unique tensor names.
        const std::string uniqueName = generateUniqueName(mTensorNames, basename);

        if (tensor)
        {
            auto* ctx = this; // To enable logging.
            if (tensor.is_tensor())
            {
                tensor.tensor().setName(uniqueName.c_str());

                LOG_VERBOSE("Registering tensor: " << uniqueName << " for ONNX tensor: " << basename);
            }
            else if (tensor.is_weights())
            {
                const auto& weights = tensor.weights();
                if (tensor.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
                {
                    tensor = ShapedWeights{::ONNX_NAMESPACE::TensorProto::INT32,
                        convertINT64(reinterpret_cast<int64_t*>(weights.values), weights.shape, ctx), weights.shape};
                }
                tensor.weights().setName(basename.c_str());
            }

        }
        // Overwrite previous tensors registered with the same name (this only happens when there are subgraphs,
        // and in that case, overwriting is the desired behavior).
        this->tensors()[basename] = std::move(tensor);
    }

    void registerLayer(nvinfer1::ILayer* layer, const std::string& basename) override
    {
        // No layer will be added for Constant nodes in ONNX.
        if (layer)
        {
            const std::string name = basename.empty() ? layer->getName() : basename;
            const std::string uniqueName = generateUniqueName(mLayerNames, name);

            auto* ctx = this; // To enable logging.
            LOG_VERBOSE("Registering layer: " << uniqueName << " for ONNX node: " << basename);

            layer->setName(uniqueName.c_str());
            if (layer->getType() == nvinfer1::LayerType::kCONSTANT)
            {
                if (basename != uniqueName)
                {
                    LOG_ERROR("Constant layer: " << uniqueName << " can be a duplicate of: " << basename);
                    assert(!"Internal error: duplicate constant layers for the same weights");
                }
                mConstantLayers.insert({uniqueName, static_cast<nvinfer1::IConstantLayer*>(layer)});
            }
        }
    }

    nvinfer1::ILogger& logger() override
    {
        return *mLogger;
    }

    ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims shape, uint8_t value = 0) override
    {
        ShapedWeights weights(type, nullptr, shape);
        // Need special logic for handling scalars.
        if (shape.nbDims == 0)
        {
            mTempBufs.push_back(std::vector<uint8_t>(getDtypeSize(type), value));
        }
        else
        {
            mTempBufs.push_back(std::vector<uint8_t>(weights.size_bytes(), value));
        }
        weights.values = mTempBufs.back().data();
        return weights;
    }

    bool setUserInput(const char* name, nvinfer1::ITensor* input)
    {
        mUserInputs[name] = input;
        return true;
    }
    bool setUserOutput(const char* name, nvinfer1::ITensor** output)
    {
        mUserOutputs[name] = output;
        return true;
    }
    nvinfer1::ITensor* getUserInput(const char* name)
    {
        if (!mUserInputs.count(name))
        {
            return nullptr;
        }
        else
        {
            return mUserInputs.at(name);
        }
    }
    nvinfer1::ITensor** getUserOutput(const char* name)
    {
        if (!mUserOutputs.count(name))
        {
            return nullptr;
        }
        else
        {
            return mUserOutputs.at(name);
        }
    }
    StringMap<nvinfer1::ITensor**> const& getUserOutputs() const
    {
        return mUserOutputs;
    }
    void clearOpsets()
    {
        mOpsets.clear();
    }
    void addOpset(std::string domain, int64_t version)
    {
        mOpsets.emplace(domain, version);
    }
    int64_t getOpsetVersion(const char* domain = "") const override
    {
        if (mOpsets.empty())
        {
            return 1;
        }
        else if (mOpsets.size() == 1)
        {
            return mOpsets.begin()->second;
        }
        else
        {
            assert(mOpsets.count(domain));
            return mOpsets.at(domain);
        }
    }
    bool hasError() const noexcept override
    {
        return mErrorWrapper != nullptr && mErrorWrapper->hasError();
    }

    nvinfer1::IErrorRecorder* getErrorRecorder() const noexcept override
    {
        return mErrorWrapper ? mErrorWrapper->getErrorRecorder() : nullptr;
    }
    nvinfer1::IConstantLayer* getConstantLayer(const char* name) const final
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

private:
    std::string generateUniqueName(std::set<std::string>& namesSet, const std::string& basename)
    {
        std::string candidate = basename;

        while (namesSet.find(candidate) != namesSet.end())
        {
            candidate = basename + "_" + std::to_string(mSuffixCounter);
            ++mSuffixCounter;
        }

        namesSet.insert(candidate);

        return candidate;
    }
};

} // namespace onnx2trt
