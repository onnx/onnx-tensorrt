#include "NvOnnxParser.h"
#include "common.hpp"
#include "onnx/onnxifi.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <atomic>
#include <ctime>
#include <mutex>
#include <thrust/device_vector.h>
#include <unordered_map>


#define BACKEND_NAME          "TensorRT"
#define BACKEND_VENDOR        "Nvidia"
#define BACKEND_VERSION       "1.0.0"
#define BACKEND_EXTENSIONS    ""
#define BACKEND_IR_VERSION    "3"
#define BACKEND_OPSET_VERSION "ai.onnx:7"


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
                      const onnxTensorDescriptorV1 &desc,
                      bool allow_same_size) {
  bool matched = false;
  if (desc.dimensions == static_cast<uint32_t>(dims.nbDims) + 1) {
    matched = true;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (desc.shape[i + 1] != static_cast<uint64_t>(dims.d[i])) {
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
    for (uint32_t i = 1; i < desc.dimensions; ++i) {
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

size_t GetTensorFootprint(const onnxTensorDescriptorV1 &input) {
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
    std::lock_guard<std::mutex> guard(mutex_);
    if (fired_) {
      return ONNXIFI_STATUS_INVALID_STATE;
    }

    if (cudaEventRecord(event_, stream_) == cudaSuccess) {
      fired_ = true;
      return ONNXIFI_STATUS_SUCCESS;
    } else {
      return ONNXIFI_STATUS_INTERNAL_ERROR;
    }
  }

  onnxStatus Wait() {
    std::lock_guard<std::mutex> guard(mutex_);
    return (cudaEventSynchronize(event_) == cudaSuccess)
               ? ONNXIFI_STATUS_SUCCESS
               : ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  onnxStatus CheckState(onnxEventState *state) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!fired_) {
      *state = ONNXIFI_EVENT_STATE_NONSIGNALLED;
      return ONNXIFI_STATUS_SUCCESS;
    }

    auto rt = cudaEventQuery(event_);
    if (rt == cudaErrorNotReady) {
      *state = ONNXIFI_EVENT_STATE_NONSIGNALLED;
      return ONNXIFI_STATUS_SUCCESS;
    } else if (rt == cudaSuccess) {
      *state = ONNXIFI_EVENT_STATE_SIGNALLED;
      return ONNXIFI_STATUS_SUCCESS;
    } else {
      *state = ONNXIFI_EVENT_STATE_INVALID;
      return ONNXIFI_STATUS_INVALID_STATE;
    }
  }

private:
  std::mutex mutex_;
  std::atomic<bool> fired_{false};
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
        nvonnxparser::createParser(*trt_network_, trt_logger_));
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
                         onnxTensorDescriptorV1 const *weight_descriptors) {
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
          return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
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
  size_t max_workspace_size_{1024UL * 1024UL * 1024UL * 2UL};
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
                    const onnxTensorDescriptorV1 *inputDescriptors,
                    uint32_t outputsCount,
                    const onnxTensorDescriptorV1 *outputDescriptors);

  onnxStatus Run();

  cudaStream_t stream() const { return stream_; }

private:
  void ClearDeviceBuffers();

  onnxStatus CheckAndBindTensor(const nvinfer1::Dims &dims,
                                const onnxTensorDescriptorV1 &tensor,
                                bool is_output);

  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> trt_executor_{nullptr};
  std::vector<void *> bindings_;
  std::unordered_map<std::string, const onnxTensorDescriptorV1 *> input_map_;
  std::unordered_map<std::string, const onnxTensorDescriptorV1 *> output_map_;
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
                                        const onnxTensorDescriptorV1 &tensor,
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
                            const onnxTensorDescriptorV1 *inputDescriptors,
                            uint32_t outputsCount,
                            const onnxTensorDescriptorV1 *outputDescriptors) {
  CudaDeviceGuard guard(device_id_);
  ClearDeviceBuffers();
  // Setup the input/output bindings and decide batch size
  for (unsigned i = 0; i < inputsCount; ++i) {
    if (inputDescriptors[i].tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }
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
    if (outputDescriptors[i].tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }
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
      if (auto ret = CheckAndBindTensor(dims, *it->second, false) !=
                     ONNXIFI_STATUS_SUCCESS) {
        return ret;
      }
    } else {
      // output: for output, we enforce 4D dim although it can be in 2D, we do
      // an implicit reshape in `CheckAndBindTensor`
      const auto it = output_map_.find(trt_engine_->getBindingName(b));
      if (it == output_map_.end()) {
        return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
      }
      if (auto ret = CheckAndBindTensor(dims, *it->second, true) !=
                     ONNXIFI_STATUS_SUCCESS) {
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

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendIDs(onnxBackendID *backendIDs, size_t *numBackends) {
  return OnnxifiTryCatch([&] {
    if (!numBackends) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    int nDevices_int{0};
    cudaGetDeviceCount(&nDevices_int);
    size_t nDevices{static_cast<size_t>(nDevices_int)};
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
onnxReleaseBackendID(onnxBackendID backendID) {
  return OnnxifiTryCatch([&] {
    auto *backend_id = reinterpret_cast<OnnxTensorRTBackendID *>(backendID);
    if (!backend_id) {
      return ONNXIFI_STATUS_INVALID_ID;
    }
    delete backend_id;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

static onnxStatus setUIntInfo(
  void* valuePtr,
  size_t *valueSizePtr,
  uint64_t value)
{
  onnxStatus status = ONNXIFI_STATUS_FALLBACK;
  if (valuePtr != nullptr && *valueSizePtr >= sizeof(uint64_t)) {
    *static_cast<uint64_t*>(valuePtr) = value;
    status = ONNXIFI_STATUS_SUCCESS;
  }
  *valueSizePtr = sizeof(uint64_t);
  return status;
}

static onnxStatus setStringInfo(
  void* valuePtr,
  size_t *valueSizePtr,
  const char* value,
  size_t valueSize)
{
  onnxStatus status = ONNXIFI_STATUS_FALLBACK;
  if (valuePtr != nullptr && *valueSizePtr >= valueSize) {
    memcpy(valuePtr, value, valueSize);
    status = ONNXIFI_STATUS_SUCCESS;
  }
  *valueSizePtr = valueSize;
  return status;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendInfo(onnxBackendID backendID, onnxBackendInfo infoType,
                   void *infoValue, size_t *infoValueSize) {
  return OnnxifiTryCatch([&] {
    if (infoValueSize == nullptr) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }

    if (backendID == nullptr) {
      return ONNXIFI_STATUS_INVALID_ID;
    }

    const int cudaDeviceId =
      static_cast<OnnxTensorRTBackendID*>(backendID)->device_id;

    switch (infoType) {
      case ONNXIFI_BACKEND_ONNXIFI_VERSION:
        return setUIntInfo(infoValue, infoValueSize,
          UINT64_C(0x0000000100000000));
      case ONNXIFI_BACKEND_NAME:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_NAME, strlen(BACKEND_NAME));
      case ONNXIFI_BACKEND_VENDOR:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_VENDOR, strlen(BACKEND_VENDOR));
      case ONNXIFI_BACKEND_VERSION:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_VERSION, strlen(BACKEND_VERSION));
      case ONNXIFI_BACKEND_EXTENSIONS:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_EXTENSIONS, strlen(BACKEND_EXTENSIONS));
      case ONNXIFI_BACKEND_DEVICE:
      {
        cudaDeviceProp deviceProperties = { 0 };
        cudaError_t cudaError =
          cudaGetDeviceProperties(&deviceProperties, cudaDeviceId);
        switch (cudaError) {
          case cudaSuccess:
            break;
          case cudaErrorInvalidDevice:
            return ONNXIFI_STATUS_INVALID_ID;
          default:
            return ONNXIFI_STATUS_INTERNAL_ERROR;
        }
        return setStringInfo(infoValue, infoValueSize,
          deviceProperties.name,
          strnlen(deviceProperties.name, sizeof(deviceProperties.name)));
      }
      case ONNXIFI_BACKEND_DEVICE_TYPE:
        return setUIntInfo(infoValue, infoValueSize,
          ONNXIFI_DEVICE_TYPE_GPU);
      case ONNXIFI_BACKEND_ONNX_IR_VERSION:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_IR_VERSION, strlen(BACKEND_IR_VERSION));
      case ONNXIFI_BACKEND_OPSET_VERSION:
        return setStringInfo(infoValue, infoValueSize,
          BACKEND_OPSET_VERSION, strlen(BACKEND_OPSET_VERSION));
      case ONNXIFI_BACKEND_CAPABILITIES:
        return setUIntInfo(infoValue, infoValueSize, 0);
      case ONNXIFI_BACKEND_INIT_PROPERTIES:
        return setUIntInfo(infoValue, infoValueSize, 0);
      case ONNXIFI_BACKEND_MEMORY_TYPES:
        return setUIntInfo(infoValue, infoValueSize,
          ONNXIFI_MEMORY_TYPE_CPU | ONNXIFI_MEMORY_TYPE_CUDA_BUFFER);
      case ONNXIFI_BACKEND_GRAPH_INIT_PROPERTIES:
        return setUIntInfo(infoValue, infoValueSize, 0);
      case ONNXIFI_BACKEND_SYNCHRONIZATION_TYPES:
        return setUIntInfo(infoValue, infoValueSize,
          ONNXIFI_SYNCHRONIZATION_EVENT);
      case ONNXIFI_BACKEND_CPU_MEMORY_READ_BANDWIDTH:
      case ONNXIFI_BACKEND_CPU_MEMORY_WRITE_BANDWIDTH:
        /* Assume PCI Express 3.0 x16 */
        return setUIntInfo(infoValue, infoValueSize, UINT64_C(16519104985));
      case ONNXIFI_BACKEND_MAX_GRAPH_COUNT:
        return setUIntInfo(infoValue, infoValueSize, UINT64_MAX);
      case ONNXIFI_BACKEND_MEMORY_SIZE:
      case ONNXIFI_BACKEND_MAX_GRAPH_SIZE:
      case ONNXIFI_BACKEND_PCI_BUS_ID:
      case ONNXIFI_BACKEND_PCI_DEVICE_ID:
      case ONNXIFI_BACKEND_PCI_DOMAIN_ID:
      case ONNXIFI_BACKEND_MACS_FP32:
      case ONNXIFI_BACKEND_MACS_FP16:
      case ONNXIFI_BACKEND_MEMORY_BANDWIDTH:
      {
        cudaDeviceProp deviceProperties = { 0 };
        cudaError_t cudaError =
          cudaGetDeviceProperties(&deviceProperties, cudaDeviceId);
        switch (cudaError) {
          case cudaSuccess:
            break;
          case cudaErrorInvalidDevice:
            return ONNXIFI_STATUS_INVALID_ID;
          default:
            return ONNXIFI_STATUS_INTERNAL_ERROR;
        }
        switch (infoType) {
          case ONNXIFI_BACKEND_MEMORY_SIZE:
          case ONNXIFI_BACKEND_MAX_GRAPH_SIZE:
            return setUIntInfo(infoValue, infoValueSize,
              static_cast<uint64_t>(deviceProperties.totalGlobalMem));
          case ONNXIFI_BACKEND_MEMORY_BANDWIDTH:
            return setUIntInfo(infoValue, infoValueSize,
              static_cast<uint64_t>(deviceProperties.memoryClockRate) *
              static_cast<uint64_t>(deviceProperties.memoryBusWidth) *
              /*
               * clock rate: kHZ -> HZ (multiply by 1000)
               * bus width: bits -> bytes (divide by 8)
               * 2x DDR factor (multiply by 2)
               */
              UINT64_C(250));
          case ONNXIFI_BACKEND_PCI_BUS_ID:
            return setUIntInfo(infoValue, infoValueSize,
              static_cast<uint64_t>(deviceProperties.pciBusID));
          case ONNXIFI_BACKEND_PCI_DEVICE_ID:
            return setUIntInfo(infoValue, infoValueSize,
              static_cast<uint64_t>(deviceProperties.pciDeviceID));
          case ONNXIFI_BACKEND_PCI_DOMAIN_ID:
            return setUIntInfo(infoValue, infoValueSize,
              static_cast<uint64_t>(deviceProperties.pciDomainID));
          case ONNXIFI_BACKEND_MACS_FP32:
          {
            /*
             * See "32-bit floating-point add, multiply, multiply-add" in
             * "Throughput of Native Arithmetic Instructions" table in
             * CUDA Programming Guide. Multiply by 2 because we could FMA
             * as two FLOPs.
             */
            uint64_t flopsPerCycle = 0;
            switch (deviceProperties.major) {
              case 3:
                /* Kepler */
                flopsPerCycle = 192 * 2;
                break;
              case 5:
                /* Maxwell */
                flopsPerCycle = 128 * 2;
                break;
              case 6:
                /* Pascal */
                switch (deviceProperties.minor) {
                  case 0:
                    flopsPerCycle = 64 * 2;
                    break;
                  case 1:
                    flopsPerCycle = 128 * 2;
                    break;
                  case 2:
                    flopsPerCycle = 128 * 2;
                    break;
                }
                break;
              case 7:
                /* Volta */
                if (deviceProperties.minor == 0) {
                  flopsPerCycle = 64 * 2;
                }
                break;
            }
            if (flopsPerCycle == 0) {
              return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
            }
            return setUIntInfo(infoValue, infoValueSize,
              UINT64_C(1000) /* KHz -> Hz */ *
              static_cast<uint64_t>(deviceProperties.clockRate) *
              static_cast<uint64_t>(deviceProperties.multiProcessorCount) *
              flopsPerCycle);
          }
          case ONNXIFI_BACKEND_MACS_FP16:
          {
            /*
             * See "16-bit floating-point add, multiply, multiply-add" and
             * "32-bit floating-point add, multiply, multiply-add" in
             * "Throughput of Native Arithmetic Instructions" table in
             * CUDA Programming Guide. Use the maximum among 16-bit and 32-bit
             * throughput. Multiply by 2 because we could FMA as two FLOPs.
             */
            uint64_t flopsPerCycle = 0;
            switch (deviceProperties.major) {
              case 3:
                /* Kepler */
                flopsPerCycle = 192 * 2;
                break;
              case 5:
                /* Maxwell */
                if (deviceProperties.minor == 3) {
                  /* Maxwell-based Tegra supports FP16 at 2x rate */
                  flopsPerCycle = 256 * 2;
                } else {
                  flopsPerCycle = 128 * 2;
                }
                break;
              case 6:
                /* Pascal */
                switch (deviceProperties.minor) {
                  case 0:
                    /* Use FP16 */
                    flopsPerCycle = 128 * 2;
                    break;
                  case 1:
                    /* Use FP32 */
                    flopsPerCycle = 128 * 2;
                    break;
                  case 2:
                    /* Use FP16 */
                    flopsPerCycle = 256 * 2;
                    break;
                }
                break;
              case 7:
                /* Volta */
                if (deviceProperties.minor == 0) {
                  /*
                   * Tensor Core:
                   * - 8 Tensor Cores per multiprocessor
                   * - 64 FMA/cycle on each Tensor Core
                   * - 2 FLOPs / FMA
                   */
                  flopsPerCycle = 8 * 64 * 2;
                }
                break;
            }
            if (flopsPerCycle == 0) {
              return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
            }
            return setUIntInfo(infoValue, infoValueSize,
              UINT64_C(1000) /* KHz -> Hz */ *
              static_cast<uint64_t>(deviceProperties.clockRate) *
              static_cast<uint64_t>(deviceProperties.multiProcessorCount) *
              flopsPerCycle);
          }
          default:
            return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
        }
      }
      case ONNXIFI_BACKEND_CUDA_INDEX:
        return setUIntInfo(infoValue, infoValueSize,
          static_cast<uint64_t>(cudaDeviceId));
      default:
        return ONNXIFI_STATUS_UNSUPPORTED_ATTRIBUTE;
    }
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetBackendCompatibility(onnxBackendID backendID, size_t onnxModelSize,
                            const void *onnxModel) {
  return OnnxifiTryCatch([&] {
    if (!onnxModel) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (onnxModelSize == 0) {
      return ONNXIFI_STATUS_INVALID_SIZE;
    }

    TRT_Logger trt_logger;
    std::shared_ptr<nvinfer1::IBuilder> trt_builder = infer_object(nvinfer1::createInferBuilder(trt_logger));
    std::shared_ptr<nvinfer1::INetworkDefinition> trt_network = infer_object(trt_builder->createNetwork());
    auto parser = infer_object(nvonnxparser::createParser(*trt_network, trt_logger));
    SubGraphCollection_t subgraphcollection;
    if (parser->supportsModel(onnxModel, onnxModelSize, subgraphcollection)) {
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxInitBackend(onnxBackendID backendID, const uint64_t *auxPropertiesList,
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
onnxReleaseBackend(onnxBackend backend) {
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
onnxInitEvent(onnxBackend backend, onnxEvent *event) {
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
onnxSignalEvent(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    return trt_event->Signal();
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxWaitEvent(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    return trt_event->Wait();
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxGetEventState(onnxEvent event, onnxEventState *state) {
  return OnnxifiTryCatch([&] {
    if (!state) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    *state = ONNXIFI_EVENT_STATE_INVALID;
    auto trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    return trt_event->CheckState(state);
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxReleaseEvent(onnxEvent event) {
  return OnnxifiTryCatch([&] {
    auto *trt_event = reinterpret_cast<OnnxTensorRTEvent *>(event);
    if (!trt_event) {
      return ONNXIFI_STATUS_INVALID_EVENT;
    }
    delete trt_event;
    return ONNXIFI_STATUS_SUCCESS;
  });
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxInitGraph(
    onnxBackend backend, const uint64_t *auxPropertiesList,
    size_t onnxModelSize, const void *onnxModel, uint32_t weightsCount,
    const onnxTensorDescriptorV1 *weightDescriptors, onnxGraph *graph) {
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

    for (auto i = 0U; i < weightsCount; ++i) {
      if (weightDescriptors[i].tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1) {
        return ONNXIFI_STATUS_UNSUPPORTED_TAG;
      }
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI onnxSetGraphIO(
    onnxGraph graph, uint32_t inputsCount,
    const onnxTensorDescriptorV1 *inputDescriptors, uint32_t outputsCount,
    const onnxTensorDescriptorV1 *outputDescriptors) {
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

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
onnxRunGraph(onnxGraph graph, const onnxMemoryFenceV1 *inputFence,
             onnxMemoryFenceV1 *outputFence) {
  return OnnxifiTryCatch([&] {
    if (!inputFence || !outputFence) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
    if (inputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1 ||
        outputFence->tag != ONNXIFI_TAG_MEMORY_FENCE_V1) {
      return ONNXIFI_STATUS_UNSUPPORTED_TAG;
    }
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
onnxReleaseGraph(onnxGraph graph) {
  return OnnxifiTryCatch([&] {
    auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
    if (!graph_rep) {
      return ONNXIFI_STATUS_INVALID_GRAPH;
    }
    delete graph_rep;
    return ONNXIFI_STATUS_SUCCESS;
  });
}
