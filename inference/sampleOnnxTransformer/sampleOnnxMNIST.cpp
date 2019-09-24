/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
//#include <time.h>
#include <sys/time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
using namespace nvinfer1;
using namespace std;

static const int BATCH_SIZE = 16;
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const std::string gSampleName = "TensorRT.sample_onnx_transformer";

samplesCommon::Args gArgs;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();


    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 25);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }
    
    //samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

bool getEngine(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    ICudaEngine** pEngine) // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();

    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 25);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }
    
    //samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
    *pEngine = builder->buildCudaEngine(*network);
    assert(pEngine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    //engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}
void doInference(IExecutionContext& context, float* input, float* output1, float* output2, float* output3, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 4);//1 input and 3 output
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex{};
    int outputIndex[3];
        int i = 0;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
        {    inputIndex = b;}
        else
        {    outputIndex[i] = b;
            i++;}
    }

cudaEvent_t start2, stop2;
cudaEventCreate(&start2);
cudaEventCreate(&stop2);





    struct timeval start; 
    struct timeval end; 
    float time_use;
    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * BATCH_SIZE*128 * sizeof(float)));
    // 253 931 0
    gettimeofday(&start,NULL);   
    CHECK(cudaMalloc(&buffers[outputIndex[1]], batchSize * 128*BATCH_SIZE*512 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex[0]], batchSize * BATCH_SIZE*128 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex[2]], batchSize * BATCH_SIZE*128 * sizeof(float)));
    gettimeofday(&end,NULL);   
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("enqueue time_use is %.10f\n",time_use);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * BATCH_SIZE*128 * sizeof(float), cudaMemcpyHostToDevice, stream));
    gettimeofday(&start,NULL);   
cudaEventRecord(start2, 0);
    context.enqueue(batchSize, buffers, stream, nullptr);
cudaEventRecord(stop2, 0);
cudaEventSynchronize(stop2);
float elapsedTime=0;
cudaEventElapsedTime(&elapsedTime, start2, stop2);
printf("sds-cuda-time  %f <ms>\n",  elapsedTime);
cudaEventDestroy(start2);
cudaEventDestroy(stop2);
    gettimeofday(&end,NULL);   
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("enqueue time_use is %.10f\n",time_use);
    gettimeofday(&start,NULL);   
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex[1]], batchSize * 128*BATCH_SIZE*512 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex[0]], batchSize * 128*BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output3, buffers[outputIndex[2]], batchSize * 128*BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gettimeofday(&end,NULL);   
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("enqueue time_use is %.10f\n",time_use);
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex[0]]));
    CHECK(cudaFree(buffers[outputIndex[1]]));
    CHECK(cudaFree(buffers[outputIndex[2]]));
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


vector<string> split(const string& str, const string& delim) {  
    vector<string> res;  
    if("" == str) return res;  
    char * strs = new char[str.length() + 1] ; 
    strcpy(strs, str.c_str());   

    char * d = new char[delim.length() + 1];  
    strcpy(d, delim.c_str());  

    char *p = strtok(strs, d);  
    while(p) {  
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);  
    }   
    return res;  
}  


int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/transformer/", "data/transformer/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    // create a TensorRT model from the onnx model and serialize it to a stream
    //IHostMemory* trtModelStream{nullptr};

    //if (!onnxToTRTModel("fair_gec_onnx.onnx", 1, trtModelStream))
     //   gLogger.reportFail(sampleTest);

    //assert(trtModelStream != nullptr);

    // read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H * INPUT_W];
    int num = rand() % 10;
    readPGMFile(locateFile(std::to_string(num) + ".pgm", gArgs.dataDirs), fileData);

    // print an ascii representation
    gLogInfo << "Input:\n";
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    gLogInfo << std::endl;

    float data[BATCH_SIZE*128];
    #if 0
    for (int i = 0; i < 1*128; i++)
        data[i] = 1;
    //Keeping the Secret of Genetic Testing
    data[121] = 8119;
    data[122] = 7;
    data[123] = 10465;
    data[124] = 12;
    data[125] = 26009;
    data[126] = 16832;
    data[127] = 2;
   #endif 
//    ifstream fin( "../data/transformer/yusong.txt" );  
//
    
    
    ifstream fin( "/home/odin/shendasai/TensorRT-5.1.5.0/data/transformer/yusong.txt" );  
    string  s;  
      


    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    ICudaEngine* engine = nullptr;
    //if (!getEngine("sds_del_input1_253_input0float.onnx", 1, &engine))
    if (!getEngine("sds_batch16_fusion.onnx", 1, &engine))
        gLogger.reportFail(sampleTest);
    assert(engine != nullptr);
    //trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    // run inference
    float prob1[BATCH_SIZE*128];
    float prob2[BATCH_SIZE*128*512];
    float prob3[BATCH_SIZE*128];

    // tag time
    struct timeval start;
    struct timeval end;
    float time_use = 0;
    
    while ( getline(fin,s) )
    {    
      cout  <<  " Read from file:  "  <<  s  <<  endl; 
      for(int b = 0; b < BATCH_SIZE; b++)
      {
      vector<string> strs = split(s, string(" "));
      int size_ = strs.size();
      int j = 0;
      for(j= 0; j < 128-size_; j++)
      {
        data[b*128+j] = 1;
      }
      for(int i = 0; i < size_;i++)
      {
        data[b*128 +j + i] = atoi(strs[i].c_str());
      } 
      //here leak one!!
      if(!getline(fin,s))
          return -1;
    }
    gettimeofday(&start,NULL);   
    doInference(*context, data, prob1, prob2, prob3, 1);
    gettimeofday(&end,NULL);   
    time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("time_use is %.10f\n",time_use);
}
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    float val{0.0f};
    int idx{0};

    //Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
    }

    gLogInfo << "Output:\n";
    for (int i = 0; i < 128; i++)
    {
        gLogInfo << " Prob1 " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob1[i] << "\n ";
    }
    for (int i = 0; i < 128; i++)
    {
        gLogInfo << " Prob2 " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob2[i] << "\n ";
    }
    for (int i = 0; i < 128; i++)
    {
        gLogInfo << " Prob3 " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob3[i] << "\n ";
    }
    gLogInfo << std::endl;

    bool pass{true};
    //bool pass{idx == num && val > 0.9f};

    return gLogger.reportTest(sampleTest, pass);
}
