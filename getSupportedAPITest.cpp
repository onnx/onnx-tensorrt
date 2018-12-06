
#include <iostream>
#include <fstream>
#include <unistd.h> // For ::getopt

#include "NvOnnxParserTypedefs.h"
#include "NvOnnxParser.h"
#include "onnx_utils.hpp"
#include "api_utils.h"


using std::cout;
using std::cerr;
using std::endl;

template<typename inferenceModel, typename parserInstance>
SubGraphCollection_t GetCapability(inferenceModel& onnx_model, parserInstance& trt_parser, std::vector<char> onnx_buf) {

  SubGraphCollection_t supportedSubGraphsCollection;
  
  if (trt_parser->supportsModel(onnx_buf.data(), onnx_buf.size(), supportedSubGraphsCollection)) {
    cout << "The model is 100% supported by the TensorRT";
  } else {
    cout << "The model contains unsupported Nodes. It has been partitioned to a set of supported subGraphs:";
    cout << "There are "<<supportedSubGraphsCollection.size()<<" supported subGraphs: "<<endl;
    cout << "{ ";
    for (auto subGraph: supportedSubGraphsCollection) {
      cout << "\t{";
      for (auto idx: subGraph) cout <<"\t"<< idx <<","<<onnx_model.graph().node(idx).op_type();
      cout << "\t}"<<endl;
    }
    cout << "\t}"<<endl;
  }
  cout << endl;
  
  return supportedSubGraphsCollection;
}



int main(int argc, char* argv[]) {
    cout << " TensorRT NetworkSupport API example "<<endl;

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string engine_filename;
    std::string text_filename;
    std::string full_text_filename;
    size_t max_batch_size = 32;
    size_t max_workspace_size = 1 << 30;
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

    int num_args = argc - optind;
    if( num_args != 1 ) {
	print_usage();
	return -1;
    }
    std::string onnx_filename = argv[optind];

    TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
    auto trt_builder = infer_object(nvinfer1::createInferBuilder(trt_logger));
    auto trt_network = infer_object(trt_builder->createNetwork());
    auto trt_parser  = infer_object(nvonnxparser::createParser(trt_network.get(), trt_logger));

    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
	cout << "Parsing model" << endl;
    }

    std::ifstream onnx_file(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);

    if( !onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) {
	cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
	return 1;
    }

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    ParseFromFile_WAR(&onnx_model, onnx_filename.c_str());

    SubGraphCollection_t SubGraphCollection;
    try {
	cout << "---------------------------------------------------------------"<<endl;
	SubGraphCollection = GetCapability(onnx_model, trt_parser, onnx_buf);
    } catch (const std::exception &e) {
	std::cerr << "Internal Error: " << e.what() << std::endl;
	return 1;
    }
    
    if( !engine_filename.empty() ) {
	trt_builder->setMaxBatchSize(max_batch_size);
	trt_builder->setMaxWorkspaceSize(max_workspace_size);

	auto trt_engine = infer_object(trt_builder->buildCudaEngine(*trt_network.get()));
	
	if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
	    cout << "Writing TensorRT engine to " << engine_filename << endl;
	}
	auto engine_plan = infer_object(trt_engine->serialize());
	std::ofstream engine_file(engine_filename.c_str(), std::ios::binary);
	engine_file.write((char*)engine_plan->data(), engine_plan->size());
	engine_file.close();
    }
	
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
	cout << "All done" << endl;
    }
    return 0;
}
