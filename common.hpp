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
#pragma once

#include <onnx/onnx_pb.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <ctime>
#include <fcntl.h> // For ::open
#include <limits>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <unistd.h>
// Namespace for common functions used throughout onnx-trt
using namespace::google::protobuf::io;

using namespace std;

namespace common
{
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

  // Logger for TensorRT info/warning/errors
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

  inline bool ParseFromFile_WAR(google::protobuf::Message* msg,
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

  
  inline bool SerializeToFile_WAR(google::protobuf::Message* msg,
                         const char*                filename) {
                         #if 0
    using namespace google::protobuf::io;
    int fd = open(filename, O_CREAT | O_RDWR);
    if (fd == -1) {
        return false;
    }
    //ZeroCopyOutputStream *raw_output = new FileOutputStream(fd1);
    google::protobuf::io::ZeroCopyOutputStream *raw_output = new google::protobuf::io::FileOutputStream(fd);
    
    google::protobuf::io::CodedOutputStream coded_output(raw_output);

    //coded_output.SetTotalBytesLimit(std::numeric_limits<int>::max(),
              //                     std::numeric_limits<int>::max()/4);
    
    int ret = msg->SerializeToCodedStream(&coded_output);
    //delete(raw_output);
    //close(fd);
    #endif
#if 1
    //����һ: ʹ��SerializePartialToOstream�����л�,ע��ios::binary�Զ�������д���ļ�
    fstream fserial(filename, ios::out | ios::trunc | ios::binary);    
    if (!msg->SerializePartialToOstream(&fserial))
    {
       cerr<<"Failed to serial address book data!\n";
       return false;
    }
    cout<<"Serial address book data successfully!\n";
    fserial.close();
    fserial.clear();
    #endif
#if 0
    int fd  = _open(filename, O_WRONLY |O_CREAT| O_BINARY, S_IREAD|S_IWRITE);    
    if( -1 == fd )
    {
        cerr<<"Create addressbook.data failed!\n";
        return false;
    }
    char tmpArr[2000000000];
    memset(tmpArr,0,sizeof(tmpArr));
    ZeroCopyOutputStream *raw_output = new ArrayOutputStream(tmpArr,msg->ByteSize()+1);    
    CodedOutputStream* coded_output = new CodedOutputStream(raw_output);    
    if( !msg->SerializeToCodedStream( coded_output ))
    {
        cerr<<"Fail to serial addressbook data!\n";
        return false;
    }    
    _write(fd,tmpArr,msg->ByteSize()+1);
    cout<<"serial address successfully!\n";
    delete coded_output;
    delete raw_output;        
    close(fd);  

    #endif
    return true;
  }

  inline bool ParseFromTextFile(google::protobuf::Message* msg,
                         const char*                filename) {
    int fd = ::open(filename, O_RDONLY);
    google::protobuf::io::FileInputStream raw_input(fd);
    raw_input.SetCloseOnDelete(true);
    return google::protobuf::TextFormat::Parse(&raw_input, msg);
  }

  inline std::string onnx_ir_version_string(int64_t ir_version=::ONNX_NAMESPACE::IR_VERSION) {
    int onnx_ir_major = ir_version / 1000000;
    int onnx_ir_minor = ir_version % 1000000 / 10000;
    int onnx_ir_patch = ir_version % 10000;
    return (std::to_string(onnx_ir_major) + "." +
            std::to_string(onnx_ir_minor) + "." +
            std::to_string(onnx_ir_patch));
  }

  inline void print_version() {
    std::cout << "Parser built against:" << std::endl;
    std::cout << "  ONNX IR version:  " << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << std::endl;
    std::cout << "  TensorRT version: "
         << NV_TENSORRT_MAJOR << "."
         << NV_TENSORRT_MINOR << "."
         << NV_TENSORRT_PATCH << std::endl;
  }
} // namespace common
