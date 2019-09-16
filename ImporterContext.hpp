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

#include <list>
#include <unordered_map>

namespace onnx2trt {

class ImporterContext final : public IImporterContext {
  nvinfer1::INetworkDefinition* _network;
  nvinfer1::ILogger* _logger;
  std::list<std::vector<uint8_t>> _temp_bufs;
  std::unordered_map<std::string, nvinfer1::ITensor*>  _user_inputs;
  std::unordered_map<std::string, nvinfer1::ITensor**> _user_outputs;
  std::unordered_map<std::string, int64_t> _opsets;
public:
  ImporterContext(nvinfer1::INetworkDefinition* network,
                  nvinfer1::ILogger* logger)
    : _network(network), _logger(logger) {}

  virtual nvinfer1::INetworkDefinition* network() override
  {
    return _network;
  }

  nvinfer1::ILogger& logger() { return *_logger; }

  virtual ShapedWeights createTempWeights(ShapedWeights::DataType type,
                                          nvinfer1::Dims shape) override
  {
    ShapedWeights weights(type, nullptr, shape);
    _temp_bufs.push_back(std::vector<uint8_t>(weights.size_bytes()));
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
    if( !_user_inputs.count(name) ) {
      return nullptr;
    } else {
      return _user_inputs.at(name);
    }
  }

  nvinfer1::ITensor** getUserOutput(const char* name)
  {
    if( !_user_outputs.count(name) ) {
      return nullptr;
    } else {
      return _user_outputs.at(name);
    }
  }

  std::unordered_map<std::string, nvinfer1::ITensor**> const& getUserOutputs() const
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

  virtual int64_t getOpsetVersion(const char* domain="") const override 
  {
    if (_opsets.empty()) {
      return 1;
    } else if (_opsets.size() == 1) {
      return _opsets.begin()->second;
    } else {
      assert(_opsets.count(domain));
      return _opsets.at(domain);
    }
  }
};

} // namespace onnx2trt
