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
                      const onnxTensorDescriptor &desc,
                      bool allow_same_size) {
  bool matched = false;
  if (desc.dimensions == dims.nbDims + 1) {
    matched = true;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (desc.shape[i + 1] != dims.d[i]) {
        return ONNXIFI_STATUS_MISMATCHING_SHAPE;
      }
    }
  } else if (allow_same_size && desc.dimensions > 1) {
    size_t dim_size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      dim_size *= dims.d[i];
    }
    size_t desc_size = 1;
    // Skip the first dim which is batch size
    for (int i = 1; i < desc.dimensions; ++i) {
      desc_size *= desc.shape[i];
    }
    matched = (dim_size == desc_size) ? true : false;
    if (!matched) {
      std::cerr << "mismatched output " << desc.name << ": " << desc_size
                << " vs " << dim_size << std::endl;
    }
  }


  return matched ? ONNXIFI_STATUS_SUCCESS : ONNXIFI_STATUS_MISMATCHING_SHAPE;
}

size_t GetTensorFootprint(const onnxTensorDescriptor &input) {
  size_t acc = 1;
  for (unsigned i = 0; i < input.dimensions; ++i) {
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
  OnnxTensorRTBackendID(int i) : device_id(i) {}
  int device_id{0};
};

class OnnxTensorRTEvent {
public:
  OnnxTensorRTEvent(cudaStream_t s) : stream_(s) {
    if (cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) !=
        cudaSuccess) {
      throw std::runtime_error("Cannot create cudaEvent");
    }
  }

  ~OnnxTensorRTEvent() { cudaEventDestroy(event_); }

  onnxStatus Signal() {
    return (cudaEventRecord(event_, stream_) == cudaSuccess)
               ? ONNXIFI_STATUS_SUCCESS
               : ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  onnxStatus Wait() {
    return (cudaEventSynchronize(event_) == cudaSuccess)
               ? ONNXIFI_STATUS_SUCCESS
               : ONNXIFI_STATUS_INTERNAL_ERROR;
  }

private:
  cudaStream_t stream_{0};
  cudaEvent_t event_;
};

class CudaDeviceGuard {
public:
  CudaDeviceGuard(int backend_id) {
    if (cudaGetDevice(&saved_device_) != cudaSuccess) {
      throw std::runtime_error("Cannot run cudaGetDevice");
    }
    if (saved_device_ != backend_id) {
      if (cudaSetDevice(backend_id) != cudaSuccess) {
        throw std::runtime_error("Cannot run cudaSetDevice");
      }
      need_restore_ = true;
    }
  }

  ~CudaDeviceGuard() {
    if (need_restore_) {
      cudaSetDevice(saved_device_);
    }
  }

private:
  int saved_device_{-1};
  bool need_restore_{false};
};
class OnnxTensorRTBackendRep {
public:
  OnnxTensorRTBackendRep(const OnnxTensorRTBackendID &backend_id)
      : device_id_(backend_id.device_id) {
    trt_builder_ = infer_object(nvinfer1::createInferBuilder(trt_logger_));
    trt_builder_->setMaxBatchSize(max_batch_size_);
    trt_builder_->setMaxWorkspaceSize(max_workspace_size_);
    trt_network_ = infer_object(trt_builder_->createNetwork());
    parser_ = infer_object(
        nvonnxparser::createParser(trt_network_.get(), trt_logger_));
    CudaDeviceGuard guard(device_id_);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("Cannot create cudaStream");
    }
  }

  ~OnnxTensorRTBackendRep() { cudaStreamDestroy(stream_); }

  int device_id() const { return device_id_; }
  cudaStream_t stream() const { return stream_; }

  onnxStatus ImportModel(void const *serialized_onnx_model,
                         size_t serialized_onnx_model_size,
                         uint32_t weight_count,
                         onnxTensorDescriptor const *weight_descriptors) {
    auto succeeded = parser_->parseWithWeightDescriptors(
        serialized_onnx_model, serialized_onnx_model_size, weight_count,
        weight_descriptors);
    if (!succeeded) {
      const auto num_errors = parser_->getNbErrors();
      if (num_errors > 0) {
        const auto *error = parser_->getError(num_errors - 1);
        std::cerr << "Parsing error: " << error->desc() << " at "
                  << error->file() << ":" << error->line() << " ("
                  << error->func() << ")." << std::endl;
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
    }

    return ONNXIFI_STATUS_SUCCESS;
  }

  nvinfer1::ICudaEngine *buildCudaEngine() {
    return trt_builder_->buildCudaEngine(*trt_network_);
  }

  size_t max_batch_size() const { return max_batch_size_; }

private:
  TRT_Logger trt_logger_;
  cudaStream_t stream_;
  std::shared_ptr<nvinfer1::IBuilder> trt_builder_{nullptr};
  std::shared_ptr<nvinfer1::INetworkDefinition> trt_network_{nullptr};
  std::shared_ptr<nvonnxparser::IParser> parser_{nullptr};
  // TODO: configerable max batch size
  int device_id_{0};
  size_t max_batch_size_{128};
  size_t max_workspace_size_{1024UL*1024UL*1024UL*2UL};
};

class GraphRep {
public:
  GraphRep(OnnxTensorRTBackendRep *backendrep)
      : device_id_(backendrep->device_id()),
        max_batch_size_(backendrep->max_batch_size()),
        stream_(backendrep->stream()) {
    if (cudaSetDevice(device_id_) != cudaSuccess) {
      throw std::runtime_error("Cannot set CUDA device");
    }
    trt_engine_ = infer_object(backendrep->buildCudaEngine());
    max_batch_size_ = backendrep->max_batch_size();
  }

  ~GraphRep() { ClearDeviceBuffers(); }

  onnxStatus InitIO(uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors);

  onnxStatus Run();

  cudaStream_t stream() const { return stream_; }

private:
  void ClearDeviceBuffers();

  onnxStatus CheckAndBindTensor(const nvinfer1::Dims &dims,
                                const onnxTensorDescriptor &tensor,
                                bool is_output);

  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> trt_executor_{nullptr};
  std::vector<void *> bindings_;
  std::unordered_map<std::string, const onnxTensorDescriptor *> input_map_;
  std::unordered_map<std::string, const onnxTensorDescriptor *> output_map_;
  std::unordered_map<std::string, void *> device_buffers_;
  int device_id_{0};
  size_t max_batch_size_{0};
  size_t batch_size_{0};
  cudaStream_t stream_;
};

void GraphRep::ClearDeviceBuffers() {
  for (auto kv : device_buffers_) {
    cudaFree(kv.second);
  }
  device_buffers_.clear();
}

onnxStatus GraphRep::CheckAndBindTensor(const nvinfer1::Dims &dims,
                                        const onnxTensorDescriptor &tensor,
                                        bool is_output) {
  // Check memory type
  if (tensor.memoryType != ONNXIFI_MEMORY_TYPE_CPU &&
      tensor.memoryType != ONNXIFI_MEMORY_TYPE_CUDA_BUFFER) {
    return ONNXIFI_STATUS_INVALID_DATATYPE;
  }
  // Check tensor shape
  auto ret = CheckShape(dims, tensor, is_output);
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
  CudaDeviceGuard guard(device_id_);
  ClearDeviceBuffers();
  // Setup the input/output bindings and decide batch size
  for (unsigned i = 0; i < inputsCount; ++i) {
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
    std::cerr << "Adding input " << i << ": " << inputDescriptors[i].name
              << ", type: " << inputDescriptors[i].memoryType << std::endl;
    input_map_.emplace(std::string(inputDescriptors[i].name),
                       inputDescriptors + i);
  }

  // We don't support the case when batch size is larger than max batch size
  // yet, but this is not a hard constraint.
  if (batch_size_ > max_batch_size_) {
    return ONNXIFI_STATUS_NO_DEVICE_RESOURCES;
  }

  for (unsigned i = 0; i < outputsCount; ++i) {
    if (!outputDescriptors[i].name) {
      return ONNXIFI_STATUS_INVALID_NAME;
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
      std::cerr << "Input: " << trt_engine_->getBindingName(b)
                << ", Dim: " << dims.d[0] << ", " << dims.d[1] << ", "
                << dims.d[2] << std::endl;
      const auto it = input_map_.find(trt_engine_->getBindingName(b));
      if (it == input_map_.end()) {
        return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
      }
      if (auto ret =
              CheckAndBindTensor(dims, *it->second, false) != ONNXIFI_STATUS_SUCCESS) {
        return ret;
      }
    } else {
      // output: for output, we enforce 4D dim although it can be in 2D, we do
      // an implicit reshape in `CheckAndBindTensor`
      const auto it = output_map_.find(trt_engine_->getBindingName(b));
      if (it == output_map_.end()) {
        return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
      }
      if (auto ret =
              CheckAndBindTensor(dims, *it->second, true) != ONNXIFI_STATUS_SUCCESS) {
        return ret;
      }
    }
  }

  trt_executor_ = infer_object(trt_engine_->createExecutionContext());
  return ONNXIFI_STATUS_SUCCESS;
}

onnxStatus GraphRep::Run() {
  CudaDeviceGuard guard(device_id_);
  // Copy input if necessary
  // TODO: cache tensor footprint
  for (auto kv : device_buffers_) {
    auto it = input_map_.find(kv.first);
    if (it != input_map_.end()) {
      cudaMemcpyAsync(kv.second, (void *)(it->second->buffer),
                      GetTensorFootprint(*it->second), cudaMemcpyHostToDevice,
                      stream_);
    } else if (output_map_.find(kv.first) == output_map_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }
  }

  // Run TensorRT
  trt_executor_->enqueue(batch_size_, bindings_.data(), stream_, nullptr);

  // Copy output if necessary
  for (auto kv : device_buffers_) {
    auto it = output_map_.find(kv.first);
    if (it != output_map_.end()) {
      cudaMemcpyAsync((void *)(it->second->buffer), kv.second,
                      GetTensorFootprint(*it->second), cudaMemcpyDeviceToHost,
                      stream_);
    } else if (input_map_.find(kv.first) == input_map_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }
  }
  return ONNXIFI_STATUS_SUCCESS;
}

template <class F> onnxStatus OnnxifiTryCatch(F &&tryBlock) {
  try {
    return tryBlock();
  } catch (const std::bad_alloc &e) {
    std::cerr << "Allocation failed: " << e.what() << std::endl;
    return ONNXIFI_STATUS_NO_SYSTEM_MEMORY;
  } catch (const std::exception &e) {
    std::cerr << "Internal Error: " << e.what() << std::endl;
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }
}
} // namespace

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendIDs)(onnxBackendID *backendIDs, size_t *numBackends) {
  return OnnxifiTryCatch([&] {
    if (!numBackends) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    int nDevices{0};
    cudaGetDeviceCount(&nDevices);
    if (!backendIDs) {
      *numBackends = nDevices;
      return ONNXIFI_STATUS_FALLBACK;
    } else {
      size_t len = (*numBackends < nDevices) ? (*numBackends) : nDevices;
      std::vector<std::unique_ptr<OnnxTensorRTBackendID>> vtmp;
      for (size_t i = 0; i < len; ++i) {
        vtmp.emplace_back(new OnnxTensorRTBackendID(i));
      }
      for (size_t i = 0; i < len; ++i) {
        backendIDs[i] = (onnxBackendID)(vtmp[i].release());
      }
      return (*numBackends < nDevices) ? ONNXIFI_STATUS_FALLBACK
                                       : ONNXIFI_STATUS_SUCCESS;
    }
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackendID)(onnxBackendID backendID) {
  return OnnxifiTryCatch([&] {
    auto *backend_id = reinterpret_cast<OnnxTensorRTBackendID *>(backendID);
    if (!backend_id) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    delete backend_id;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendInfo)(onnxBackendID backendID, onnxBackendInfo infoType,
                        void *infoValue, size_t *infoValueSize) {
  return OnnxifiTryCatch([&] {
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
    switch (infoType) {
    case ONNXIFI_BACKEND_NAME:
      SET_STRING("TensorRT");
      break;
    case ONNXIFI_BACKEND_VENDOR:
      SET_STRING("Nvidia");
      break;
    case ONNXIFI_BACKEND_VERSION:
      SET_STRING("1.0.0");
      break;
    case ONNXIFI_BACKEND_EXTENSIONS:
      *infoValueSize = 0;
      break;
    case ONNXIFI_BACKEND_DEVICE:
      SET_STRING("gpu");
      break;
    case ONNXIFI_BACKEND_DEVICE_TYPE:
      SET_UINT64(ONNXIFI_DEVICE_TYPE_GPU);
      break;
    case ONNXIFI_BACKEND_CAPABILITIES:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_INIT_PROPERTIES:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_MEMORY_TYPES:
      SET_UINT64(ONNXIFI_MEMORY_TYPE_CUDA_BUFFER);
      break;
    case ONNXIFI_BACKEND_MEMORY_SIZE: {
      size_t free, total;
      if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        return ONNXIFI_STATUS_BACKEND_UNAVAILABLE;
      }
      SET_UINT64(uint64_t(total));
      break;
    }
    // TODO: Dummy numbers below
    case ONNXIFI_BACKEND_MAX_GRAPH_SIZE:
      SET_UINT64(1000000UL);
      break;
    case ONNXIFI_BACKEND_MAX_GRAPH_COUNT:
      SET_UINT64(1UL);
      break;
    case ONNXIFI_BACKEND_MACS_FP32:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_MACS_FP16:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_MEMORY_BANDWIDTH:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH:
      SET_UINT64(0UL);
      break;
    case ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH:
      SET_UINT64(0UL);
      break;
    default:
      return ONNXIFI_STATUS_UNSUPPORTED_PARAMETER;
    }
    return ONNXIFI_STATUS_SUCCESS;
#undef SET_STRING
#undef SET_UINT64
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendCompatibility)(onnxBackendID backendID, size_t onnxModelSize,
                                 const void *onnxModel) {
  return OnnxifiTryCatch([&] {
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }

    TRT_Logger trt_logger;
    auto parser = infer_object(nvonnxparser::createParser(nullptr, trt_logger));
    if (parser->supportsModel(onnxModel, onnxModelSize)) {
      return ONNXIFI_STATUS_SUCCESS;
    } else {
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  });
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
  auto ret = OnnxifiTryCatch([&] {
    auto *backend_id = reinterpret_cast<OnnxTensorRTBackendID *>(backendID);
    if (!backend_id) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    *backend = (onnxBackend)(new OnnxTensorRTBackendRep(*backend_id));
    return ONNXIFI_STATUS_SUCCESS;
  });
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    *backend = NULL;
  }
  return ret;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseBackend)(onnxBackend backend) {
  return OnnxifiTryCatch([&] {
    auto *backendrep = reinterpret_cast<OnnxTensorRTBackendRep *>(backend);
    if (!backendrep) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    delete backendrep;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxInitEvent)(onnxBackend backend, onnxEvent *event) {
  auto ret = OnnxifiTryCatch([&] {
    if (!event) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    auto *backendrep = reinterpret_cast<OnnxTensorRTBackendRep *>(backend);
    if (!backendrep) {
      return ONNXIFI_STATUS_INVALID_BACKEND;
    }
    *event = reinterpret_cast<onnxEvent>(
        new OnnxTensorRTEvent(backendrep->stream()));
    return ONNXIFI_STATUS_SUCCESS;
  });
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    *event = NULL;
  }
  return ret;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxSignalEvent)(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    return trt_event->Signal();
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxWaitEvent)(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    return trt_event->Wait();
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseEvent)(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto *trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    delete trt_event;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxInitGraph)(onnxBackend backend, size_t onnxModelSize,
                   const void *onnxModel, uint32_t weightsCount,
                   const onnxTensorDescriptor *weightDescriptors,
                   onnxGraph *graph) {
  auto ret = OnnxifiTryCatch([&] {
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
    auto ret = backendrep->ImportModel(onnxModel, onnxModelSize, weightsCount,
                                       weightDescriptors);
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      return ret;
    }

    // Create the TRT engine
    *graph = (onnxGraph)(new GraphRep(backendrep));
    return ONNXIFI_STATUS_SUCCESS;
  });
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    *graph = NULL;
  }
  return ret;
}

// NB: in the context of TRT, this step will setup the input/output bindings for
// ICudaEngine
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxSetGraphIO)(onnxGraph graph, uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptor *outputDescriptors) {
  return OnnxifiTryCatch([&] {
    auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
    if (!graph_rep) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    if (!inputDescriptors || !outputDescriptors) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    return graph_rep->InitIO(inputsCount, inputDescriptors, outputsCount,
                             outputDescriptors);
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxRunGraph)(onnxGraph graph, const onnxMemoryFence *inputFence,
                  onnxMemoryFence *outputFence) {
  return OnnxifiTryCatch([&] {
    auto *trt_event = reinterpret_cast<OnnxTensorRTEvent *>(inputFence->event);
    auto ret = trt_event->Wait();
    if (ret != ONNXIFI_STATUS_SUCCESS) {
      return ret;
    }
    auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
    if (!graph_rep) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }

    ret = graph_rep->Run();
    auto output_event = new OnnxTensorRTEvent(graph_rep->stream());
    outputFence->event = reinterpret_cast<onnxEvent>(output_event);
    outputFence->type = ONNXIFI_SYNCHRONIZATION_EVENT;
    output_event->Signal();
    return ret;
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
ONNXIFI_SYMBOL_NAME(onnxReleaseGraph)(onnxGraph graph) {
  return OnnxifiTryCatch([&] {
    auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
    if (!graph_rep) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    delete graph_rep;
    return ONNXIFI_STATUS_SUCCESS;
  });
}
