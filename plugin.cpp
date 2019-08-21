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

#include "plugin.hpp"
#include "serialize.hpp"
#include <NvInferPlugin.h>

namespace onnx2trt {

// ========================= Plugin =====================

  void Plugin::serializeBase(void*& buffer) {
    serialize_value(&buffer, _input_dims);
    serialize_value(&buffer, _max_batch_size);
    serialize_value(&buffer, _data_type);
    serialize_value(&buffer, _data_format);
  }

  void Plugin::deserializeBase(void const*& serialData, size_t& serialLength) {
    deserialize_value(&serialData, &serialLength, &_input_dims);
    deserialize_value(&serialData, &serialLength, &_max_batch_size);
    deserialize_value(&serialData, &serialLength, &_data_type);
    deserialize_value(&serialData, &serialLength, &_data_format);
  }

  size_t Plugin::getBaseSerializationSize() {
    return (serialized_size(_input_dims) +
            serialized_size(_max_batch_size) +
            serialized_size(_data_type) +
            serialized_size(_data_format));
  }

  bool Plugin::supportsFormat(nvinfer1::DataType type,
                              nvinfer1::PluginFormat format) const {
    return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)  &&
            (format == nvinfer1::PluginFormat::kNCHW));
  }

  void Plugin::configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                   const nvinfer1::Dims* outputDims, int nbOutputs,
                                   nvinfer1::DataType type,
                                   nvinfer1::PluginFormat format,
                                   int maxBatchSize)  {
    _data_type = type;
    _data_format = format;
    _input_dims.assign(inputDims, inputDims + nbInputs);
    _max_batch_size = maxBatchSize;
  }

// ========================= PluginV2 =====================

  void PluginV2::serializeBase(void*& buffer) const {
    serialize_value(&buffer, _input_dims);
    serialize_value(&buffer, _max_batch_size);
    serialize_value(&buffer, _data_type);
    serialize_value(&buffer, _data_format);
  }

  void PluginV2::deserializeBase(void const*& serialData, size_t& serialLength) {
    deserialize_value(&serialData, &serialLength, &_input_dims);
    deserialize_value(&serialData, &serialLength, &_max_batch_size);
    deserialize_value(&serialData, &serialLength, &_data_type);
    deserialize_value(&serialData, &serialLength, &_data_format);
  }

  size_t PluginV2::getBaseSerializationSize() const {
    return (serialized_size(_input_dims) +
            serialized_size(_max_batch_size) +
            serialized_size(_data_type) +
            serialized_size(_data_format));
  }

  bool PluginV2::supportsFormat(nvinfer1::DataType type,
                              nvinfer1::PluginFormat format) const {
    return ((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)  &&
            (format == nvinfer1::PluginFormat::kNCHW));
  }

  void PluginV2::configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                                   const nvinfer1::Dims* outputDims, int nbOutputs,
                                   nvinfer1::DataType type,
                                   nvinfer1::PluginFormat format,
                                   int maxBatchSize)  {
    _data_type = type;
    _data_format = format;
    _input_dims.assign(inputDims, inputDims + nbInputs);
    _max_batch_size = maxBatchSize;
  }

// ========================= PluginAdapter =====================

  int PluginAdapter::getNbOutputs() const {
    return _pluginV2->getNbOutputs();
  }
  nvinfer1::Dims PluginAdapter::getOutputDimensions(int index,
                                                    const nvinfer1::Dims *inputDims,
                                                    int nbInputs)  {
    return _pluginV2->getOutputDimensions(index, inputDims, nbInputs);
  }
  void PluginAdapter::serialize(void* buffer) const {
    return _pluginV2->serialize(buffer);
  }
  size_t PluginAdapter::getSerializationSize() const {
    return _pluginV2->getSerializationSize();
  }
  bool PluginAdapter::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const
  {
    return _pluginV2->supportsFormat(type, format);
  }
  void PluginAdapter::configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs,
                                          const nvinfer1::Dims *outputDims, int nbOutputs,
                                          nvinfer1::DataType type,
                                          nvinfer1::PluginFormat format,
                                          int maxBatchSize) {
    return _pluginV2->configureWithFormat(inputDims, nbInputs,
                                   outputDims, nbOutputs,
                                   type, format, maxBatchSize);
  }
  size_t PluginAdapter::getWorkspaceSize(int maxBatchSize) const {
    return _pluginV2->getWorkspaceSize(maxBatchSize);
  }
  int PluginAdapter::initialize() { return _pluginV2->initialize(); }

  void PluginAdapter::terminate() {
    if (_pluginV2) {
      _pluginV2->terminate();
    }
  }

  void PluginAdapter::destroy() {
    return _pluginV2->destroy();
  }

  nvinfer1::IPluginV2* PluginAdapter::clone() const {
    return _pluginV2->clone();
  }

  const char* PluginAdapter::getPluginVersion() const {
    return _pluginV2->getPluginVersion();
  }

  void PluginAdapter::setPluginNamespace(const char* pluginNamespace) {
    return _pluginV2->setPluginNamespace(pluginNamespace);
  }

  const char* PluginAdapter::getPluginNamespace() const {
      return _pluginV2->getPluginNamespace();
  }

  int PluginAdapter::enqueue(int batchSize,
                             const void *const *inputs, void **outputs,
                             void *workspace, cudaStream_t stream) {
    return _pluginV2->enqueue(batchSize, inputs, outputs, workspace, stream);
  }

} // namespace onnx2trt
