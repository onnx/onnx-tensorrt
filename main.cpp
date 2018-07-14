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

#include "NvOnnxParser.h"
#include "onnx_utils.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <unistd.h> // For ::getopt
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <ctime>
#include <fcntl.h> // For ::open
#include <limits>

struct InferDeleter {
  template<typename T>
  void operator()(T* obj) const {
    if( obj ) {
      obj->destroy();
    }
  }
};
template<typename T>
inline std::shared_ptr<T> infer_object(T* obj) {
  if( !obj ) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

// Logger for GIE info/warning/errors
class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;
  std::ostream* _ostream;
public:
  TRT_Logger(Severity verbosity=Severity::kWARNING,
             std::ostream& ostream=std::cout)
    : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char* msg) override {
    if( severity <= _verbosity ) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               std::gmtime(&rawtime));
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" :
                            severity == Severity::kERROR          ? "  ERROR" :
                            severity == Severity::kWARNING        ? "WARNING" :
                            severity == Severity::kINFO           ? "   INFO" :
                            "UNKNOWN");
      (*_ostream) << "[" << buf << " " << sevstr << "] "
                  << msg
                  << std::endl;
    }
  }
};

bool ParseFromFile_WAR(google::protobuf::Message* msg,
                       const char*                filename) {
  int fd = ::open(filename, O_RDONLY);
  google::protobuf::io::FileInputStream raw_input(fd);
  raw_input.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&raw_input);
  // Note: This WARs the very low default size limit (64MB)
  coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::max()/4);
  return msg->ParseFromCodedStream(&coded_input);
}

bool ParseFromTextFile(google::protobuf::Message* msg,
                       const char*                filename) {
  int fd = ::open(filename, O_RDONLY);
  google::protobuf::io::FileInputStream raw_input(fd);
  raw_input.SetCloseOnDelete(true);
  return google::protobuf::TextFormat::Parse(&raw_input, msg);
}

std::string onnx_ir_version_string(int64_t ir_version=::ONNX_NAMESPACE::IR_VERSION) {
  int onnx_ir_major = ir_version / 1000000;
  int onnx_ir_minor = ir_version % 1000000 / 10000;
  int onnx_ir_patch = ir_version % 10000;
  return (std::to_string(onnx_ir_major) + "." +
          std::to_string(onnx_ir_minor) + "." +
          std::to_string(onnx_ir_patch));
}

void print_usage() {
  cout << "ONNX to TensorRT model parser" << endl;
  cout << "Usage: onnx2trt onnx_model.pb" << "\n"
       << "                [-o engine_file.trt]  (output TensorRT engine)" << "\n"
       << "                [-t onnx_model.pbtxt] (output ONNX text file without weights)" << "\n"
       << "                [-T onnx_model.pbtxt] (output ONNX text file with weights)" << "\n"
       << "                [-b max_batch_size (default 32)]" << "\n"
       << "                [-w max_workspace_size_bytes (default 1 GiB)]" << "\n"
       << "                [-d model_data_type_bit_depth] (32 => float32, 16 => float16)" << "\n"
       << "                [-l] (list layers and their shapes)" << "\n"
       << "                [-g] (debug mode)" << "\n"
       << "                [-v] (increase verbosity)" << "\n"
       << "                [-q] (decrease verbosity)" << "\n"
       << "                [-V] (show version information)" << "\n"
       << "                [-h] (show help)" << endl;
}

void print_version() {
  cout << "Parser built against:" << endl;
  cout << "  ONNX IR version:  " << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << endl;
  cout << "  TensorRT version: "
       << NV_TENSORRT_MAJOR << "."
       << NV_TENSORRT_MINOR << "."
       << NV_TENSORRT_PATCH << endl;
}

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::string engine_filename;
  std::string text_filename;
  std::string full_text_filename;
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
  int model_dtype_nbits = 32;
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  bool print_layer_info = false;
  bool debug_builder = false;

  int arg = 0;
  while( (arg = ::getopt(argc, argv, "o:b:w:t:T:d:lgvqVh")) != -1 ) {
    switch (arg){
    case 'o':
      if( optarg ) { engine_filename = optarg; break; }
      else { cerr << "ERROR: -o flag requires argument" << endl; return -1; }
    case 't':
      if( optarg ) { text_filename = optarg; break; }
      else { cerr << "ERROR: -t flag requires argument" << endl; return -1; }
    case 'T':
      if( optarg ) { full_text_filename = optarg; break; }
      else { cerr << "ERROR: -T flag requires argument" << endl; return -1; }
    case 'b':
      if( optarg ) { max_batch_size = atoll(optarg); break; }
      else { cerr << "ERROR: -b flag requires argument" << endl; return -1; }
    case 'w':
      if( optarg ) { max_workspace_size = atoll(optarg); break; }
      else { cerr << "ERROR: -w flag requires argument" << endl; return -1; }
    case 'd':
      if( optarg ) { model_dtype_nbits = atoi(optarg); break; }
      else { cerr << "ERROR: -d flag requires argument" << endl; return -1; }
    case 'l': print_layer_info = true; break;
    case 'g': debug_builder = true; break;
    case 'v': ++verbosity; break;
    case 'q': --verbosity; break;
    case 'V': print_version(); return 0;
    case 'h': print_usage(); return 0;
    }
  }
  int num_args = argc - optind;
  if( num_args != 1 ) {
    print_usage();
    return -1;
  }
  std::string onnx_filename = argv[optind];

  nvinfer1::DataType model_dtype;
  if(      model_dtype_nbits == 32 ) { model_dtype = nvinfer1::DataType::kFLOAT; }
  else if( model_dtype_nbits == 16 ) { model_dtype = nvinfer1::DataType::kHALF; }
  //else if( model_dtype_nbits ==  8 ) { model_dtype = nvinfer1::DataType::kINT8; }
  else {
    cerr << "ERROR: Invalid model data type bit depth: " << model_dtype_nbits << endl;
    return -2;
  }

  ::ONNX_NAMESPACE::ModelProto onnx_model;
  bool is_binary = ParseFromFile_WAR(&onnx_model, onnx_filename.c_str());
  if( !is_binary && !ParseFromTextFile(&onnx_model, onnx_filename.c_str()) ) {
    cerr << "Failed to parse ONNX model" << endl;
    return -3;
  }

  if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    int64_t opset_version = (onnx_model.opset_import().size() ?
                             onnx_model.opset_import(0).version() : 0);
    cout << "----------------------------------------------------------------" << endl;
    cout << "Input filename:   " << onnx_filename << endl;
    cout << "ONNX IR version:  " << onnx_ir_version_string(onnx_model.ir_version()) << endl;
    cout << "Opset version:    " << opset_version << endl;
    cout << "Producer name:    " << onnx_model.producer_name() << endl;
    cout << "Producer version: " << onnx_model.producer_version() << endl;
    cout << "Domain:           " << onnx_model.domain() << endl;
    cout << "Model version:    " << onnx_model.model_version() << endl;
    cout << "Doc string:       " << onnx_model.doc_string() << endl;
    cout << "----------------------------------------------------------------" << endl;
  }

  if( onnx_model.ir_version() > ::ONNX_NAMESPACE::IR_VERSION ) {
    cerr << "WARNING: ONNX model has a newer ir_version ("
         << onnx_ir_version_string(onnx_model.ir_version())
         << ") than this parser was built against ("
         << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << ")." << endl;
  }

  if( !text_filename.empty() ) {
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
      cout << "Writing ONNX model (without weights) as text to " << text_filename << endl;
    }
    std::ofstream onnx_text_file(text_filename.c_str());
    std::string onnx_text = pretty_print_onnx_to_string(onnx_model);
    onnx_text_file.write(onnx_text.c_str(), onnx_text.size());
  }
  if( !full_text_filename.empty() ) {
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
      cout << "Writing ONNX model (with weights) as text to " << full_text_filename << endl;
    }
    std::string full_onnx_text;
    google::protobuf::TextFormat::PrintToString(onnx_model, &full_onnx_text);
    std::ofstream full_onnx_text_file(full_text_filename.c_str());
    full_onnx_text_file.write(full_onnx_text.c_str(), full_onnx_text.size());
  }

  TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
  auto trt_builder = infer_object(nvinfer1::createInferBuilder(trt_logger));
  auto trt_network = infer_object(trt_builder->createNetwork());
  auto trt_parser  = infer_object(nvonnxparser::createParser(
                                      trt_network.get(), trt_logger));

  // TODO: Fix this for the new API
  //if( print_layer_info ) {
  //  parser->setLayerInfoStream(&std::cout);
  //}
  (void)print_layer_info;

  if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "Parsing model" << endl;
  }

  {
    std::ifstream onnx_file(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);
    if( !onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) {
      cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
      return -4;
    }
    if( !trt_parser->parse(onnx_buf.data(), onnx_buf.size()) ) {
      int nerror = trt_parser->getNbErrors();
      for( int i=0; i<nerror; ++i ) {
        nvonnxparser::IParserError const* error = trt_parser->getError(i);
        if( error->node() != -1 ) {
          ::ONNX_NAMESPACE::NodeProto const& node =
            onnx_model.graph().node(error->node());
          cerr << "While parsing node number " << error->node()
               << " [" << node.op_type();
          if( node.output().size() ) {
            cerr << " -> \"" << node.output(0) << "\"";
          }
          cerr << "]:" << endl;
          if( verbosity >= (int)nvinfer1::ILogger::Severity::kINFO ) {
            cerr << "--- Begin node ---" << endl;
            cerr << node << endl;
            cerr << "--- End node ---" << endl;
          }
        }
        cerr << "ERROR: "
             << error->file() << ":" << error->line()
             << " In function " << error->func() << ":\n"
             << "[" << static_cast<int>(error->code()) << "] " << error->desc()
             << endl;
      }
      return -5;
    }
  }

  bool fp16 = trt_builder->platformHasFastFp16();

  if( !engine_filename.empty() ) {
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
      cout << "Building TensorRT engine, FP16 available:"<< fp16 << endl;
      cout << "    Max batch size:     " << max_batch_size << endl;
      cout << "    Max workspace size: " << max_workspace_size / (1024. * 1024) << " MiB" << endl;
    }
    trt_builder->setMaxBatchSize(max_batch_size);
    trt_builder->setMaxWorkspaceSize(max_workspace_size);
    if( fp16 && model_dtype == nvinfer1::DataType::kHALF) {
      trt_builder->setHalf2Mode(true);
    } else if( model_dtype == nvinfer1::DataType::kINT8 ) {
      // TODO: Int8 support
      //trt_builder->setInt8Mode(true);
      cerr << "ERROR: Int8 mode not yet supported" << endl;
      return -5;
    }
    trt_builder->setDebugSync(debug_builder);
    auto trt_engine = infer_object(trt_builder->buildCudaEngine(*trt_network.get()));

    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
      cout << "Writing TensorRT engine to " << engine_filename << endl;
    }
    auto engine_plan = infer_object(trt_engine->serialize());
    std::ofstream engine_file(engine_filename.c_str());
    engine_file.write((char*)engine_plan->data(), engine_plan->size());
    engine_file.close();
  }

  if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "All done" << endl;
  }
  return 0;
}
