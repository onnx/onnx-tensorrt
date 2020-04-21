/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "onnx2trt.hpp"
#include "onnx2trt_utils.hpp"

#include <list>
#include <unordered_map>

namespace onnx2trt
{

class ImporterContext final : public IImporterContext
{
    nvinfer1::INetworkDefinition* _network;
    nvinfer1::ILogger* _logger;
    std::list<std::vector<uint8_t>> _temp_bufs;
    StringMap<nvinfer1::ITensor*> _user_inputs;
    StringMap<nvinfer1::ITensor**> _user_outputs;
    StringMap<int64_t> _opsets;
    StringMap<TensorOrWeights> mTensors; // All tensors in the graph mapped to their names.
    StringMap<nvinfer1::TensorLocation> mTensorLocations;
    StringMap<float> mTensorRangeMins;
    StringMap<float> mTensorRangeMaxes;
    StringMap<nvinfer1::DataType> mLayerPrecisions;
    StringMap<size_t>
        mTensorNameCounts; // Keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    StringMap<size_t>
        mLayerNameCounts; // Keep track of how many times a tensor name shows up, to avoid duplicate naming in TRT.
    std::unordered_set<std::string> mUnsupportedShapeTensors; // Container to hold any shape tensors that are the output of layers that do not support shape tensors.
public:
    ImporterContext(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : _network(network)
        , _logger(logger)
    {
    }
    virtual nvinfer1::INetworkDefinition* network() override
    {
        return _network;
    }
    virtual StringMap<TensorOrWeights>& tensors() override
    {
        return mTensors;
    }
    virtual StringMap<nvinfer1::TensorLocation>& tensorLocations() override
    {
        return mTensorLocations;
    }
    virtual StringMap<float>& tensorRangeMins() override
    {
        return mTensorRangeMins;
    }
    virtual StringMap<float>& tensorRangeMaxes() override
    {
        return mTensorRangeMaxes;
    }
    virtual StringMap<nvinfer1::DataType>& layerPrecisions() override
    {
        return mLayerPrecisions;
    }
    virtual std::unordered_set<std::string>& unsupportedShapeTensors() override
    {
        return mUnsupportedShapeTensors;
    }

    // This actually handles weights as well, but is named this way to be consistent with the tensors()
    virtual void registerTensor(TensorOrWeights tensor, const std::string& basename) override
    {
        // TRT requires unique tensor names.
        const std::string uniqueName
            = mTensorNameCounts[basename] ? (basename + "_" + std::to_string(mTensorNameCounts[basename])) : basename;
        ++mTensorNameCounts[basename];

        if (tensor)
        {
            auto* ctx = this; // To enable logging.
            if (tensor.is_tensor())
            {
                tensor.tensor().setName(uniqueName.c_str());

                LOG_VERBOSE("Registering tensor: " << uniqueName << " for ONNX tensor: " << basename);
            }
            else if (tensor.is_weights() && tensor.weights().type == ::ONNX_NAMESPACE::TensorProto::INT64)
            {
                const auto& weights = tensor.weights();
                tensor = ShapedWeights{::ONNX_NAMESPACE::TensorProto::INT32,
                    convertINT64(reinterpret_cast<int64_t*>(weights.values), weights.shape, ctx), weights.shape};
            }
        }
        // Overwrite previous tensors registered with the same name (this only happens when there are subgraphs,
        // and in that case, overwriting is the desired behavior).
        this->tensors()[basename] = std::move(tensor);
    }

    virtual void registerLayer(nvinfer1::ILayer* layer, const std::string& basename) override
    {
        // No layer will be added for Constant nodes in ONNX.
        if (layer)
        {
            const std::string name = basename.empty() ? layer->getName() : basename;
            const std::string uniqueName
                = mLayerNameCounts[name] ? (name + "_" + std::to_string(mLayerNameCounts[name])) : name;
            ++mLayerNameCounts[name];

            auto* ctx = this; // To enable logging.
            LOG_VERBOSE("Registering layer: " << name << " for ONNX node: " << basename);

            layer->setName(uniqueName.c_str());
        }
    }

    virtual nvinfer1::ILogger& logger() override
    {
        return *_logger;
    }

    virtual ShapedWeights createTempWeights(ShapedWeights::DataType type, nvinfer1::Dims shape) override
    {
        ShapedWeights weights(type, nullptr, shape);
        // Need special logic for handling scalars.
        if (shape.nbDims == 0)
        {
            _temp_bufs.push_back(std::vector<uint8_t>(getDtypeSize(type)));
        }
        else
        {
            _temp_bufs.push_back(std::vector<uint8_t>(weights.size_bytes()));
        }
        weights.values = _temp_bufs.back().data();
        return weights;
    }

    bool setUserInput(const char* name, nvinfer1::ITensor* input)
    {
        _user_inputs[name] = input;
        return true;
    }
    bool setUserOutput(const char* name, nvinfer1::ITensor** output)
    {
        _user_outputs[name] = output;
        return true;
    }
    nvinfer1::ITensor* getUserInput(const char* name)
    {
        if (!_user_inputs.count(name))
        {
            return nullptr;
        }
        else
        {
            return _user_inputs.at(name);
        }
    }
    nvinfer1::ITensor** getUserOutput(const char* name)
    {
        if (!_user_outputs.count(name))
        {
            return nullptr;
        }
        else
        {
            return _user_outputs.at(name);
        }
    }
    StringMap<nvinfer1::ITensor**> const& getUserOutputs() const
    {
        return _user_outputs;
    }
    void clearOpsets()
    {
        _opsets.clear();
    }
    void addOpset(std::string domain, int64_t version)
    {
        _opsets.emplace(domain, version);
    }
    virtual int64_t getOpsetVersion(const char* domain = "") const override
    {
        if (_opsets.empty())
        {
            return 1;
        }
        else if (_opsets.size() == 1)
        {
            return _opsets.begin()->second;
        }
        else
        {
            assert(_opsets.count(domain));
            return _opsets.at(domain);
        }
    }
};

} // namespace onnx2trt
