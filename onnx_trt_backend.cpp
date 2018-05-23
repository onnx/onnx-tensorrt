#include "NvOnnxParser.h"
#include "onnx/onnxifi.h"
#include <NvInfer.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <unordered_map>

namespace {

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};
template <typename T> inline std::shared_ptr<T> infer_object(T *obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

// Logger for GIE info/warning/errors
class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;
  std::ostream *_ostream;

public:
  TRT_Logger(Severity verbosity = Severity::kWARNING,
             std::ostream &ostream = std::cout)
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char *msg) override {
    if (severity <= _verbosity) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
      const char *sevstr =
          (severity == Severity::kINTERNAL_ERROR
               ? "    BUG"
               : severity == Severity::kERROR
                     ? "  ERROR"
                     : severity == Severity::kWARNING
                           ? "WARNING"
                           : severity == Severity::kINFO ? "   INFO"
                                                         : "UNKNOWN");
      (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
    }
  }
};

onnxStatus CheckShape(const nvinfer1::Dims &dims,
                      const onnxTensorDescriptor &input) {
  bool matched = false;
  if (input.dimensions != dims.nbDims + 1) {
    matched = true;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (input.shape[i + 1] != dims.d[i]) {
        matched = false;
        break;
      }
    }
  }
  return matched ? ONNXIFI_STATUS_SUCCESS : ONNXIFI_STATUS_MISMATCHING_SHAPE;
}

size_t GetTensorFootprint(const onnxTensorDescriptor &input) {
  size_t acc = 1;
  for (int i = 0; i < input.dimensions; ++i) {
    acc *= input.shape[i];
  }
  size_t multiplier = 1;
  switch (input.dataType) {
  case ONNXIFI_DATATYPE_FLOAT16:
    multiplier = sizeof(float) / 2;
    break;
  case ONNXIFI_DATATYPE_FLOAT32:
    multiplier = sizeof(float);
    break;
  case ONNXIFI_DATATYPE_INT8:
    multiplier = sizeof(int8_t);
    break;
  case ONNXIFI_DATATYPE_INT16:
    multiplier = sizeof(int16_t);
    break;
  case ONNXIFI_DATATYPE_INT32:
    multiplier = sizeof(int32_t);
    break;
  case ONNXIFI_DATATYPE_UINT8:
    multiplier = sizeof(uint8_t);
    break;
  case ONNXIFI_DATATYPE_UINT16:
    multiplier = sizeof(uint16_t);
    break;
  case ONNXIFI_DATATYPE_UINT32:
    multiplier = sizeof(uint32_t);
    break;
  default:
    multiplier = 0;
  }
  return acc * multiplier;
}

struct OnnxTensorRTBackendID {
  int device_id{0};
};

class OnnxTensorRTBackendRep {
public:
  OnnxTensorRTBackendRep() {
    trt_builder_ = infer_object(nvinfer1::createInferBuilder(trt_logger_));
    trt_network_ = infer_object(trt_builder_->createNetwork());
    parser_ =
        infer_object(nvonnxparser::createParser(*trt_network_, trt_logger_));
  }

  onnxStatus ImportModel(void const *serialized_onnx_model,
                         size_t serialized_onnx_model_size) {
    auto succeeded =
        parser_->parse(serialized_onnx_model, serialized_onnx_model_size);
    if (!succeeded) {
      const auto num_errors = parser_->getNbErrors();
      if (num_errors > 0) {
        const auto *error = parser_->getError(num_errors - 1);
        switch (error->code()) {
        case nvonnxparser::ErrorCode::kMEM_ALLOC_FAILED:
          return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
        case nvonnxparser::ErrorCode::kMODEL_DESERIALIZE_FAILED:
          return ONNXIFI_STATUS_INVALID_PROTOBUF;
        case nvonnxparser::ErrorCode::kINVALID_VALUE:
          return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
        case nvonnxparser::ErrorCode::kINVALID_GRAPH:
        case nvonnxparser::ErrorCode::kINVALID_NODE:
          return ONNXIFI_STATUS_INVALID_MODEL;
        case nvonnxparser::ErrorCode::kUNSUPPORTED_NODE:
        case nvonnxparser::ErrorCode::kUNSUPPORTED_GRAPH:
          return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
        default:
          return ONNXIFI_STATUS_INTERNAL_ERROR;
        }
      }
    } else {
      return ONNXIFI_STATUS_SUCCESS;
    }
  }

  nvinfer1::ICudaEngine *buildCudaEngine() {
    return trt_builder_->buildCudaEngine(*trt_network_);
  }

  size_t batch_size() const { return batch_size_; }

private:
  TRT_Logger trt_logger_;
  std::shared_ptr<nvinfer1::IBuilder> trt_builder_{nullptr};
  std::shared_ptr<nvinfer1::INetworkDefinition> trt_network_{nullptr};
  std::shared_ptr<nvonnxparser::IParser> parser_{nullptr};
  // TODO: configerable max batch size
  size_t batch_size_{10};
};

class GraphRep {
public:
  GraphRep(OnnxTensorRTBackendRep *backendrep) {
    assert(backendrep);
    trt_engine_ = infer_object(backendrep->buildCudaEngine());
    batch_size_ = backendrep->batch_size();
  }

  ~GraphRep() { ClearDeviceBuffers(); }

  onnxStatus InitIO(uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors);

  onnxStatus Run();

private:
  void ClearDeviceBuffers();

  onnxStatus CheckAndBindTensor(const nvinfer1::Dims &dims,
                                const onnxTensorDescriptor &tensor);

  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> trt_executor_{nullptr};
  std::vector<void *> bindings_;
  std::unordered_map<std::string, const onnxTensorDescriptor *> input_map_;
  std::unordered_map<std::string, const onnxTensorDescriptor *> output_map_;
  std::unordered_map<std::string, void *> device_buffers_;
  size_t max_batch_size_{0};
  size_t batch_size_{0};
};

void GraphRep::ClearDeviceBuffers() {
  for (auto kv : device_buffers_) {
    cudaFree(kv.second);
  }
  device_buffers_.clear();
}

onnxStatus GraphRep::CheckAndBindTensor(const nvinfer1::Dims &dims,
                                        const onnxTensorDescriptor &tensor) {
  // Check memory type
  if (tensor.memoryType != ONNXIFI_MEMORY_TYPE_CPU ||
      tensor.memoryType != ONNXIFI_MEMORY_TYPE_CUDA_BUFFER) {
    return ONNXIFI_STATUS_INVALID_DATATYPE;
  }
  // Check input shape
  auto ret = CheckShape(dims, tensor);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    return ret;
  }

  // For CPU tensor, we need to create a device memory and the bind. For CUDA
  // tensor, we can bind directly
  if (tensor.memoryType == ONNXIFI_MEMORY_TYPE_CPU) {
    void *cuda_buffer;
    size_t footprint = GetTensorFootprint(tensor);
    if (!footprint) {
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }
    if (cudaMalloc(&cuda_buffer, footprint) != cudaSuccess) {
      return ONNXIFI_STATUS_NO_DEVICE_MEMORY;
    }
    device_buffers_.emplace(tensor.name, cuda_buffer);
    bindings_.push_back(cuda_buffer);
  } else {
    bindings_.push_back((void *)(tensor.buffer));
  }

  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus GraphRep::InitIO(uint32_t inputsCount,
                            const onnxTensorDescriptor *inputDescriptors,
                            uint32_t outputsCount,
                            const onnxTensorDescriptor *outputDescriptors) {
  ClearDeviceBuffers();
  // Setup the input/output bindings and decide batch size
  for (auto i = 0; i < inputsCount; ++i) {
    if (!inputDescriptors[i].name) {
      return ONNXIFI_STATUS_INVALID_NAME;
    }
    // We only support NCHW
    if (inputDescriptors[i].dimensions != 4) {
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }
    if (i == 0) {
      batch_size_ = inputDescriptors[i].shape[0];
    } else {
      if (batch_size_ != inputDescriptors[i].shape[0]) {
        return ONNXIFI_STATUS_INVALID_SHAPE;
      }
    }
    input_map_.emplace(std::string(inputDescriptors[i].name),
                       inputDescriptors + i);
  }

  // We don't support the case when batch size is larger than max batch size
  // yet, but this is not a hard constraint.
  if (batch_size_ > max_batch_size_) {
    return ONNXIFI_STATUS_NO_DEVICE_RESOURCES;
  }

  for (auto i = 0; i < outputsCount; ++i) {
    if (!outputDescriptors[i].name) {
      return ONNXIFI_STATUS_INVALID_NAME;
    }
    // We only support NCHW
    if (outputDescriptors[i].dimensions != 4) {
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }
    if (batch_size_ != outputDescriptors[i].shape[0]) {
      return ONNXIFI_STATUS_INVALID_SHAPE;
    }
    output_map_.emplace(std::string(outputDescriptors[i].name),
                        outputDescriptors + i);
  }

  int nbindings = trt_engine_->getNbBindings();
  for (int b = 0; b < nbindings; ++b) {
    nvinfer1::Dims dims = trt_engine_->getBindingDimensions(b);
    // Check data type consistency
    auto binding_datatype = trt_engine_->getBindingDataType(b);
    if (binding_datatype != nvinfer1::DataType::kFLOAT) {
      return ONNXIFI_STATUS_MISMATCHING_DATATYPE;
    }

    if (trt_engine_->bindingIsInput(b)) {
      const auto it = input_map_.find(trt_engine_->getBindingName(b));
      if (it == input_map_.end()) {
        return ONNXIFI_STATUS_INVALID_NAME;
      }
      if (auto ret =
              CheckAndBindTensor(dims, *it->second) != ONNXIFI_STATUS_SUCCESS) {
        return ret;
      }
    } else {
      // output
      const auto it = output_map_.find(trt_engine_->getBindingName(b));
      if (it == output_map_.end()) {
        return ONNXIFI_STATUS_INVALID_NAME;
      }
      if (auto ret =
              CheckAndBindTensor(dims, *it->second) != ONNXIFI_STATUS_SUCCESS) {
        return ret;
      }
    }
  }

  trt_executor_ = infer_object(trt_engine_->createExecutionContext());
  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus GraphRep::Run() {
  // Copy input if necessary
  // TODO: cache tensor footprint
  for (auto kv : device_buffers_) {
    auto it = input_map_.find(kv.first);
    if (it != input_map_.end()) {
      cudaMemcpy(kv.second, (void *)(it->second->buffer),
                 GetTensorFootprint(*it->second), cudaMemcpyHostToDevice);
    } else if (output_map_.find(kv.first) == output_map_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }
  }

  // Run TensorRT
  // TODO(add async api)
  cudaStream_t stream = 0;
  trt_executor_->enqueue(batch_size_, bindings_.data(), stream, nullptr);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  // Copy output if necessary
  for (auto kv : device_buffers_) {
    auto it = output_map_.find(kv.first);
    if (it != output_map_.end()) {
      cudaMemcpy((void *)(it->second->buffer), kv.second,
                 GetTensorFootprint(*it->second), cudaMemcpyDeviceToHost);
    } else if (input_map_.find(kv.first) == input_map_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }
  }
  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendIDs)(onnxBackendID *backendIDs, size_t *numBackends) {
  *backendIDs = (onnxBackendID)(new OnnxTensorRTBackendID());
  *numBackends = 1;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackendID)(onnxBackendID backendID) {
  auto *backend_id = reinterpret_cast<OnnxTensorRTBackendID *>(backendID);
  if (!backend_id) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  delete backend_id;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendInfo)(onnxBackendID backendID, onnxBackendInfo infoType,
                        void *infoValue, size_t *infoValueSize) {
  if (!infoValueSize) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
#define SET_STRING(str)                                                        \
  {                                                                            \
    snprintf((char *)(infoValue), *infoValueSize, str);                        \
    *infoValueSize = strlen(str) + 1;                                          \
  }

#define SET_UINT64(x)                                                          \
  {                                                                            \
    if (*infoValueSize < sizeof(uint64_t)) {                                   \
      return ONNXIFI_STATUS_INVALID_POINTER;                                   \
    }                                                                          \
    *(uint64_t *)(infoValue) = x;                                              \
    *infoValueSize = sizeof(uint64_t);                                         \
  }
  if (infoType == ONNXIFI_BACKEND_NAME) {
    SET_STRING("TensorRT");
  } else if (infoType == ONNXIFI_BACKEND_VENDOR) {
    SET_STRING("Nvidia");
  } else if (infoType == ONNXIFI_BACKEND_VERSION) {
    SET_STRING("1.0.0");
  } else if (infoType == ONNXIFI_BACKEND_EXTENSIONS) {
    *infoValueSize = 0;
  } else if (infoType == ONNXIFI_BACKEND_DEVICE) {
    SET_STRING("gpu");
  } else if (infoType == ONNXIFI_BACKEND_DEVICE_TYPE) {
    SET_UINT64(ONNXIFI_DEVICE_TYPE_GPU);
  } else if (infoType == ONNXIFI_BACKEND_CAPABILITIES) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_INIT_PROPERTIES) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_MEMORY_TYPES) {
    SET_UINT64(ONNXIFI_MEMORY_TYPE_CUDA_BUFFER);
  } else if (infoType == ONNXIFI_BACKEND_MEMORY_SIZE) {
    size_t free, total;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
      return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
    }
    SET_UINT64(uint64_t(total));
  }
  // Dummy numbers
  else if (infoType == ONNXIFI_BACKEND_MAX_GRAPH_SIZE) {
    SET_UINT64(1000000UL);
  } else if (infoType == ONNXIFI_BACKEND_MAX_GRAPH_COUNT) {
    SET_UINT64(1UL);
  } else if (infoType == ONNXIFI_BACKEND_MACS_FP32) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_MACS_FP16) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_MEMORY_BANDWIDTH) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH) {
    SET_UINT64(0UL);
  } else if (infoType == ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH) {
    SET_UINT64(0UL);
  } else {
    return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
  }
  return ONNXIFI_STATUS_SUCCESS;
#undef RETURN_STRING
#undef SET_UINT64
}

// NB: Why not have onnxModel as const char*?  and we should set the name to
// onnxGraph And we don't have ir_version and opset info here, which are needed
// for model check
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendCompatibility)(onnxBackendID backendID, size_t onnxModelSize,
                                 const void *onnxModel) {
  if (!onnxModel) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (onnxModelSize == 0) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  // NB: not ideal case. We CHECK model by actually trying to run the
  // conversion. However, this might be the case for other vendors
  OnnxTensorRTBackendRep backendrep;
  return backendrep.ImportModel(onnxModel, onnxModelSize);
}

// NB: Passing arguments to backend is tricky. And we need more documentation
// for it I didn't put any arguments here for now.
// TODO: submit arguments for
// - setMaxBatchSize (size_t)
// - setMaxWorkspaceSize (size_t)
// - setHalf2Mode (bool)
// - setInt8Mode (bool)
// - setDebugSync (bool)
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitBackend)(onnxBackendID backendID, const uint64_t *auxPropertiesList,
                     onnxBackend *backend) {
  *backend = (onnxBackend)(new OnnxTensorRTBackendRep());
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackend)(onnxBackend backend) {
  auto *backendrep = reinterpret_cast<OnnxTensorRTBackendRep *>(backend);
  if (!backendrep) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }
  delete backendrep;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitGraph)(onnxBackend backend, size_t onnxModelSize,
                   const void *onnxModel, uint32_t weightsCount,
                   const onnxTensorDescriptor *weightDescriptors,
                   onnxGraph *graph) {
  auto *backendrep = reinterpret_cast<OnnxTensorRTBackendRep *>(backend);
  if (!backendrep) {
    return ONNXIFI_STATUS_INVALID_BACKEND;
  }
  if (!onnxModel) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }
  if (onnxModelSize == 0) {
    return ONNXIFI_STATUS_INVALID_SIZE;
  }

  // Parse the model
  // TODO: Ignore the weightDescriptors for now and rely on initialization list
  auto ret = backendrep->ImportModel(onnxModel, onnxModelSize);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    return ret;
  }

  // Create the TRT engine
  // TODO: error handling
  *graph = (onnxGraph)(new GraphRep(backendrep));
  return ONNXIFI_STATUS_SUCCESS;
}

// NB: in the context of TRT, this step will setup the input/output bindings for
// ICudaEngine
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxSetGraphIO)(onnxGraph graph, uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors) {
  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }
  if (!inputDescriptors || !outputDescriptors) {
    return ONNXIFI_STATUS_INVALID_POINTER;
  }

  return graph_rep->InitIO(inputsCount, inputDescriptors, outputsCount,
                           outputDescriptors);
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxRunGraph)(onnxGraph graph, const onnxMemoryFence *inputFence,
                  onnxMemoryFence *outputFence) {
  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }

  return graph_rep->Run();
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseGraph)(onnxGraph graph) {

  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNXIFI_STATUS_INVALID_GRAPH;
  }
  delete graph_rep;
  return ONNXIFI_STATUS_SUCCESS;
}
