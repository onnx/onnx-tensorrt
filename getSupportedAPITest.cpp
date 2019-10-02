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

#include <iostream>
#include <fstream>
#include <unistd.h> // For ::getopt
#include <string>
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "onnx_utils.hpp"
#include "common.hpp"

using std::cout;
using std::cerr;
using std::endl;

void print_usage() {
  cout << "This program will determine whether or not an ONNX model is compatible with TensorRT. " 
       << "If it isn't, a list of supported subgraphs and unsupported operations will be printed." << endl;
  cout << "Usage: getSupportedAPITest -m onnx_model.pb" << endl;
  cout << "Optional argument: -e TRT_engine" << endl;
}

void printSubGraphs(SubGraphCollection_t& subGraphs, ::ONNX_NAMESPACE::ModelProto onnx_model)
{
    if (subGraphs.size() != 1)
    {
        cout << "The model contains unsupported Nodes. It has been partitioned to a set of supported subGraphs." << endl;
        cout << "There are "<< subGraphs.size() << " supported subGraphs: " << endl;
        cout << "NOTE: Due to some limitations with the parser, the support of specific subgraphs may not have been determined."
        << " Please refer to the printed subgraphs to see if they are truly supported or not." << endl;
    }
    else 
    {
        cout << "The model is fully supported by TensorRT. Printing the parsed graph:" << endl;
    }

    for (auto subGraph: subGraphs) 
    {
        cout << "\t{";
        for (auto idx: subGraph.first) cout << "\t" << idx << "," <<onnx_model.graph().node(idx).op_type();
        cout << "\t}\t - ";
        if (subGraph.second)
        {
            cout << "Fully supported" << endl;
        }
        else
        {
            cout << "UNKNOWN whether this is fully supported." << endl; 
        }
    }
}


int main(int argc, char* argv[]) {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string engine_filename;
    std::string text_filename;
    std::string full_text_filename;
    std::string onnx_filename;
    int c;
    size_t max_batch_size = 32;
    size_t max_workspace_size = 1 << 30;
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    while ((c = getopt (argc, argv, "m:e:")) != -1)
    {
        switch(c)
        {
            case 'm':
                    onnx_filename = optarg;
                    break;
            case 'e':
                    engine_filename = optarg;
                    break;
        }
    }

    if (onnx_filename.empty())
    {
        print_usage();
        return -1;
    }

    common::TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);

    auto trt_builder = common::infer_object(nvinfer1::createInferBuilder(trt_logger));

    auto trt_network = common::infer_object(trt_builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto trt_parser  = common::infer_object(nvonnxparser::createParser(*trt_network, trt_logger));

    initLibNvInferPlugins(&trt_logger, "");

    cout << "Parsing model: " << onnx_filename << endl;
    
    std::ifstream onnx_file(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);

    if( !onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) {
        cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
        return -1;
    }

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    if (!common::ParseFromFile_WAR(&onnx_model, onnx_filename.c_str()))
    {
        cout << "Failure while parsing ONNX file" << endl;
        return -1;
    }

    SubGraphCollection_t SubGraphCollection;

    // supportsModel() parses the graph and returns a list of supported subgraphs.
    if (!trt_parser->supportsModel(onnx_buf.data(), onnx_buf.size(), SubGraphCollection))
    {
        cout << "Model cannot be fully parsed by TensorRT!" << endl;
        printSubGraphs(SubGraphCollection, onnx_model);
        return -1;
    }

    printSubGraphs(SubGraphCollection, onnx_model);
    
    // If -e was specified, create and save the TensorRT engine to disk.
    // Note we do not call trt_parser->parse() here since it's already done above in parser->supportsModel()
    if( !engine_filename.empty() ) {
        trt_builder->setMaxBatchSize(max_batch_size);
        trt_builder->setMaxWorkspaceSize(max_workspace_size);

        cout << "input name: " << trt_network->getInput(0)->getName() << endl;
        cout << "output name: " << trt_network->getOutput(0)->getName() << endl;
        cout << "num layers: " << trt_network->getNbLayers() << endl;
        cout << "outputs: " << trt_network->getNbOutputs() << endl;

        auto trt_engine = common::infer_object(trt_builder->buildCudaEngine(*trt_network.get()));
    
        if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
            cout << "Writing TensorRT engine to " << engine_filename << endl;
        }
        auto engine_plan = common::infer_object(trt_engine->serialize());
        std::ofstream engine_file(engine_filename.c_str(), std::ios::binary);
        engine_file.write(reinterpret_cast<const char*>(engine_plan->data()), engine_plan->size());
        engine_file.close();
    }
    
    if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        cout << "All done" << endl;
    }
    return 0;
}
