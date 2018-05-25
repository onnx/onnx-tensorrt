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

  void Plugin::serializeBase(void*& buffer)  {
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

  size_t Plugin::getBaseSerializationSize()  {
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

// ========================= PluginAdapter =====================

  int PluginAdapter::getNbOutputs() const {
    return _plugin->getNbOutputs();
  }
  nvinfer1::Dims PluginAdapter::getOutputDimensions(int index,
                                                    const nvinfer1::Dims *inputDims,
                                                    int nbInputs)  {
    return _plugin->getOutputDimensions(index, inputDims, nbInputs);
  }
  void PluginAdapter::serialize(void* buffer)  {
    return _plugin->serialize(buffer);
  }
  size_t PluginAdapter::getSerializationSize() {
    return _plugin->getSerializationSize();
  }
  bool PluginAdapter::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const 
  {
    if (_ext)
      return _ext->supportsFormat(type, format);
    else
      return (type == nvinfer1::DataType::kFLOAT &&
              format == nvinfer1::PluginFormat::kNCHW);
  }
  void PluginAdapter::configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs,
                                          const nvinfer1::Dims *outputDims, int nbOutputs,
                                          nvinfer1::DataType type,
                                          nvinfer1::PluginFormat format,
                                          int maxBatchSize) {
    if (_ext)
      return _ext->configureWithFormat(inputDims, nbInputs,
                                       outputDims, nbOutputs,
                                       type, format, maxBatchSize);
    else
      return _plugin->configure(inputDims, nbInputs,
                                outputDims, nbOutputs,
                                maxBatchSize);
  }
  size_t PluginAdapter::getWorkspaceSize(int maxBatchSize) const {
    return _plugin->getWorkspaceSize(maxBatchSize);
  }
  int PluginAdapter::initialize() { return _plugin->initialize(); }

  void PluginAdapter::terminate() {
    if (_plugin) {
      _plugin->terminate();
    }
  }

  int PluginAdapter::enqueue(int batchSize,
                             const void *const *inputs, void **outputs,
                             void *workspace, cudaStream_t stream) {
    return _plugin->enqueue(batchSize, inputs, outputs, workspace, stream);
  }

// ========================= NvPlugin =====================

  const char* NvPlugin::getPluginType() const {
    using namespace nvinfer1;
    switch( _plugin->getPluginType() ) {
    case PluginType::kFASTERRCNN:         return "FasterRCNN";
    case PluginType::kNORMALIZE:          return "Normalize";
    case PluginType::kPERMUTE:            return "Permute";
    case PluginType::kPRIORBOX:           return "SSDPriorBox";
    case PluginType::kSSDDETECTIONOUTPUT: return "SSDDetectionOutput";
    case PluginType::kCONCAT:             return "Concat";
    case PluginType::kPRELU:              return "PRelu";
    case PluginType::kYOLOREORG:          return "YoloReorg";
    case PluginType::kYOLOREGION:         return "YoloRegion";
    default: return "Unknown";
    }
  }

  void NvPlugin::destroy() {
    if (_plugin) {
#if NV_TENSORRT_MAJOR >= 4 // WAR for double-free issue with NvPlugins in TRT 3
      _plugin->destroy();
#endif // NV_TENSORRT_MAJOR >= 4
      _plugin = 0;
    }
    delete this;
  }

} // namespace onnx2trt
