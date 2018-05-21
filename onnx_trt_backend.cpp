#include "onnx/onnxifi.h"
#include "NvOnnxParser.h"
#include <NvInfer.h>
#include <thrust/device_vector.h>
#include <ctime>
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

/*onnxStatus CheckShape(const nvinfer1::Dims &dims,
                      const onnxTensorDescriptor &input) {
  bool matched = false;
  if (input.dimensions != dims.nbDims) {
    matched = true;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (input.shape[i] != dims.d[i]) {
        matched = false;
        break;
      }
    }
  }
  return matched ? ONNX_STATUS_SUCCESS : ONNX_STATUS_INVALID_SHAPE;
}*/

struct OnnxTensorRTBackendID {
  int device_id{0};
};

class OnnxTensorRTBackendRep {
public:
  OnnxTensorRTBackendRep() {
    trt_builder_ = infer_object(nvinfer1::createInferBuilder(trt_logger_));
    trt_network_ = infer_object(trt_builder_->createNetwork());
    parser_ = infer_object(nvonnxparser::createParser(*trt_network_, trt_logger_));
  }

  int ImportModel(void const* serialized_onnx_model, size_t serialized_onnx_model_size) {
    auto succeeded = parser_->parse(serialized_onnx_model, serialized_onnx_model_size);
    if (!succeeded) {
      const auto num_errors = parser_->getNbErrors();
      if (num_errors > 0) {
        const auto* error = parser_->getError(num_errors - 1);
        switch(error->code()) {
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
  //TODO: configerable max batch size
  size_t batch_size_{10};
};

class GraphRep {
public:
  GraphRep(OnnxTensorRTBackendRep *backendrep) {
    assert(backendrep);
    trt_engine_ =
        infer_object(backendrep->buildCudaEngine());
    batch_size_ = backendrep->batch_size();
  }

  /*
  onnxStatus InitIO(uint32_t inputsCount,
                    const onnxTensorDescriptor *inputDescriptors,
                    uint32_t outputsCount,
                    onnxTensorDescriptor *outputDescriptors);

  onnxStatus Run();
*/
private:
  /*
  void PrepareInputData(const onnxTensorDescriptor &input,
                        thrust::device_vector<float> *dv_input);
  void PrepareOutputData(onnxTensorDescriptor *output,
                         thrust::device_vector<float> *dv_output);
  */
  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::unordered_map<std::string, const onnxTensorDescriptor *> input_map_;
  std::unordered_map<std::string, const onnxTensorDescriptor *> output_map_;
  std::vector<thrust::device_vector<float>> dv_inputs_;
  std::vector<thrust::device_vector<float>> dv_outputs_;
  std::vector<void *> bindings_;
  std::vector<onnxTensorDescriptor *> output_tensors_;
  size_t batch_size_{0};
};

/*
void GraphRep::PrepareInputData(const onnxTensorDescriptor &input,
                                thrust::device_vector<float> *dv_input) {
  size_t total_size = 1;
  for (int i = 0; i < input.dimensions; ++i) {
    total_size *= input.shape[i];
  }

  dv_input->resize(total_size);
  float *d_input = thrust::raw_pointer_cast(&(*dv_input)[0]);
  // NB: Marat, is this the right way of using onnxTensorDescriptor.buffer?
  // We only support float in TRT for now
  CHECK_CUDART(cudaMemcpy(d_input, (float *)(input.buffer),
                          dv_input->size() * sizeof(float),
                          cudaMemcpyHostToDevice));
  bindings_.push_back(((void*)(d_input));
}

void GraphRep::PrepareOutputData(onnxTensorDescriptor *output,
                                 thrust::device_vector<float> *dv_output) {
  size_t total_size = 1;
  for (int i = 0; i < output->dimensions; ++i) {
    total_size *= output->shape[i];
  }
  dv_output->resize(total_size);
  float *d_output = thrust::raw_pointer_cast(&(*dv_output)[0]);
  bindings_.push_back((void *)(d_output));
  output_tensors_.push_back(output);
}

onnxStatus GraphRep::InitIO(uint32_t inputsCount,
                            const onnxTensorDescriptor *inputDescriptors,
                            uint32_t outputsCount,
                            onnxTensorDescriptor *outputDescriptors) {
  // Setup the input/output bindings
  for (auto i = 0; i < inputsCount; ++i) {
    if (!inputDescriptors[i].name) {
      return ONNX_STATUS_INVALID_NAME;
    }
    input_map_.emplace(std::string(inputDescriptors[i].name),
                       inputDescriptors + i);
  }
  for (auto i = 0; i < outputsCount; ++i) {
    if (!outputDescriptors[i].name) {
      return ONNX_STATUS_INVALID_NAME;
    }
    output_map_.emplace(std::string(outputDescriptors[i].name),
                        outputDescriptors + i);
  }

  dv_inputs.resize(inputsCount);
  dv_outputs.resize(outputsCount);
  int nbindings = trt_engine->getNbBindings();
  int input_idx = 0;
  int output_idx = 0;
  for (int b = 0; b < nbindings; ++b) {
    nvinfer1::Dims dims = trt_engine_->getBindingDimensions(b);
    if (trt_engine_->bindingIsInput(b)) {
      const auto it = input_map_.find(trt_engine->getBindingName(b));
      if (it == input_map_.end()) {
        return ONNX_STATUS_INVALID_NAME;
      }
      const auto &tensor_input = *(it->second);

      // Check data type consistency
      // NB: We don't have a error value for data type mismatch?
      auto binding_datatype = trt_engine_->getBindingDataType(b);
      if (binding_datatype != nvinfer1::DataType::kFLOAT) {
        return ONNX_STATUS_INTERNAL_ERROR;
      }

      // Check input shape
      auto ret = CheckShape(dims, tensor_input);
      if (ret != ONNX_STATUS_SUCCESS) {
        return ret;
      }

      PrepareInputData(tensor_input, &dv_inputs.at(input_idx));
      ++input_idx;
    } else {
      // output
      const auto it = output_map_.find(trt_engine->getBindingName(b));
      if (it == output_map_.end()) {
        return ONNX_STATUS_INVALID_NAME;
      }

      const auto &tensor_output = *(it->second);

      // Check data type consistency
      // NB: We don't have a error value for data type mismatch?
      auto binding_datatype = trt_engine_->getBindingDataType(b);
      if (binding_datatype != nvinfer1::DataType::kFLOAT) {
        return ONNX_STATUS_INTERNAL_ERROR;
      }

      // Check input shape
      auto ret = CheckShape(dims, tensor_output);
      if (ret != ONNX_STATUS_SUCCESS) {
        return ret;
      }

      PrepareInputData(tensor_output, &bindings_, &dv_outputs.at(input_idx));
      ++output_idx;
    }
  }

  return ONNX_STATUS_SUCCESS;
}

onnxStatus GraphRep::Run() {
  auto trt_executor = infer_object(trt_engine_->createExecutionContext());

  // Run TensorRT
  // TODO(add async api)
  cudaStream_t stream = 0;
  trt_executor->enqueue(batch_size_, bindings_.data(), stream, nullptr);
  CHECK_CUDART(cudaStreamSynchronize(stream));

  // Fill the output tensors
  int idx = 0;
  for (const auto &dv : dv_outputs_) {
    auto *output_tensor = output_tensors_.at(idx++);
    thrust::host_vector<float> hv_output = dv;
    memcpy((void *)(output_tensor->buffer), hv_output.data(),
           sizeof(float) * hv_output.size());
  }

  return ONNX_STATUS_SUCCESS;
}*/

} // namespace

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI ONNXIFI_SYMBOL_NAME(
    onnxGetBackendIDs)(onnxBackendID *backendIDs, size_t *numBackends) {
  *backendIDs = (onnxBackendID)(new OnnxTensorRTBackendID());
  *numBackends = 1;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  ONNXIFI_SYMBOL_NAME(onnxReleaseBackendID)(
    onnxBackendID backendID) {
  auto *backend_id = reinterpret_cast<OnnxTensorRTBackendID*>(backendID);
  if (!backend_id) {
    return ONNXIFI_STATUS_INVALID_ID;
  }
  delete backend_id;
  return ONNXIFI_STATUS_SUCCESS;
}

ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  ONNXIFI_SYMBOL_NAME(onnxGetBackendInfo)(
    onnxBackendID backendID,
    onnxBackendInfo infoType,
    void* infoValue,
    size_t* infoValueSize) {
    if (!infoValueSize) {
      return ONNXIFI_STATUS_INVALID_POINTER;
    }
#define SET_STRING(str)                                                     \
  {                                                                            \
    snprintf((char *)(infoValue), *infoValueSize, str);                          \
    *infoValueSize = strlen(str) + 1;                                          \
  }

#define SET_UINT64(x) \
 { \
   if(*infoValueSize < sizeof(uint64_t)) { \
     return ONNXIFI_STATUS_INVALID_POINTER; \
   } \
   *(uint64_t*)(infoValue) = x; \
   *infoValueSize = sizeof(uint64_t); \
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
ONNXIFI_PUBLIC ONNXIFI_CHECK_RESULT onnxStatus ONNXIFI_ABI
  ONNXIFI_SYMBOL_NAME(onnxInitBackend)(
    onnxBackendID backendID,
    const uint64_t* auxPropertiesList,
    onnxBackend* backend) {
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
  auto ret =  backendrep->ImportModel(onnxModel, onnxModelSize);
  if (ret != ONNXIFI_STATUS_SUCCESS) {
    return ret;
  }

  // Create the TRT engine
  // TODO: error handling
  *graph = (onnxGraph)(new GraphRep(backendrep));
  return ONNXIFI_STATUS_SUCCESS;
}

/*
// NB: Similarly, here we should take in a model instead of a graph. We still
// can bypass the initilization_list and use onnxTensorDescriptors as an
// optimization
// NB: the returned handle here is described as a opaque graph handle. However,
// in the context of TRT, it is a runtime engine for that graph. We might need
// to generalize this in the documentation, call this graph execution engine or
// something
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxInitGraph)(
    onnxBackend backend, size_t onnxModelSize, const void *onnxModel,
    uint32_t weightsCount, const onnxTensorDescriptor *weightDescriptors,
    onnxGraph *graph) {
  auto *backendrep = reinterpret_cast<OnnxTensorRTBackendRep *>(backend);
  if (!backendrep) {
    return ONNX_STATUS_INVALID_BACKEND;
  }
  if (!onnxModel) {
    return ONNX_STATUS_INVALID_POINTER;
  }
  if (onnxModelSize == 0) {
    return ONNX_STATUS_INVALID_SIZE;
  }

  // Parse
  onnx::ModelProto onnx_model;
  imemstream stream(reinterpret_cast<char const *>(onnxModel), onnxModelSize);
  if (!onnx_model.ParseFromIstream(&stream)) {
    return ONNX_STATUS_INVALID_PROTOBUF;
  }

  // Check (we might want to by pass this check if the model only has one node)
  try {
    onnx::checker::check_model(onnx_model);
  } catch (const onnx::checker::ValidationError &) {
    return ONNX_STATUS_INVALID_MODEL;
  }

  // Load the model
  // TODO: Ignore the weightDescriptors for now and rely on initialization list
  // in the protobuf
  onnx2trt::Status status = backendrep->ImportModel(onnx_model);
  if (status.is_error()) {
    switch (status.code()) {
    case ONNX2TRT_STATUS_MEM_ALLOC_FAILED:
      return ONNX_STATUS_NO_SYSTEM_MEMORY;
    case ONNX2TRT_STATUS_UNSUPPORTED_NODE:
      return ONNX_STATUS_UNSUPPORTED_OPERATOR;
    case ONNX2TRT_STATUS_INVALID_NODE:
    case ONNX2TRT_STATUS_INVALID_GRAPH:
      return ONNX_STATUS_UNSUPPORTED_PARAMETER;
    default:
      return ONNX_STATUS_INTERNAL_ERROR;
    }
  }

  // Create the TRT engine
  // TODO: error handling
  *graph = (onnxGraph)(new GraphRep(backendrep));

  return ONNX_STATUS_SUCCESS;
}

// NB: in the context of TRT, this step will setup the input/output bindings for
// ICudaEngine
// NB2: In addition, I think outputDescriptors should not be a const pointer as
// we are going to fill the output data
ONNX_PUBLIC onnxStatus ONNX_ABI ONNX_SYMBOL_NAME(onnxSetGraphIO)(
    onnxGraph graph, uint32_t inputsCount,
    const onnxTensorDescriptor *inputDescriptors, uint32_t outputsCount,
    onnxTensorDescriptor *outputDescriptors) {
  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNX_STATUS_INVALID_GRAPH;
  }
  if (!inputDescriptors || !outputDescriptors) {
    return ONNX_STATUS_INVALID_POINTER;
  }

  return graph_rep->InitIO(inputsCount, inputDescriptors, outputsCount,
                           outputDescriptors);
}

ONNX_PUBLIC onnxStatus ONNX_ABI
ONNX_SYMBOL_NAME(onnxRunGraph)(onnxGraph graph) {
  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNX_STATUS_INVALID_GRAPH;
  }

  return graph_rep->Run();
}

ONNX_PUBLIC onnxStatus ONNX_ABI
ONNX_SYMBOL_NAME(onnxReleaseGraph)(onnxGraph graph) {
  auto *graph_rep = reinterpret_cast<GraphRep *>(graph);
  if (!graph_rep) {
    return ONNX_STATUS_INVALID_GRAPH;
  }
  delete graph_rep;
  return ONNX_STATUS_SUCCESS;
}
*/
