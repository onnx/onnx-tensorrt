 /*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <onnx/onnx_pb.h>
#include <sstream>

#pragma once

namespace
{

//! Describes occurrence of a named dimension.
class NamedDimension
{
public:
    //! TensorRT tensor.
    nvinfer1::ITensor* tensor;

    //! Index of tensor dimension to be named.
    int32_t index;

    //! ONNX "dim param" that is the name of the dimension.
    std::string dimParam;

    //! Construct a NamedDimension where the tensor will be filled in later.
    NamedDimension(int32_t index_, const std::string& dimParam_)
        : tensor(nullptr)
        , index(index_)
        , dimParam(dimParam_)
    {
    }
};

template <typename OnnxDims>
bool convertOnnxDims(OnnxDims const& onnxDims, nvinfer1::Dims& trtDims, std::vector<NamedDimension>& namedDims)
{
    std::vector<int32_t> onnxDimsVec;
    for (const auto& onnxDim : onnxDims)
    {
        // For empty dimensions, the ONNX specification says it's a dynamic dimension
        if (!onnxDim.has_dim_value() && !onnxDim.has_dim_param())
        {
            onnxDimsVec.emplace_back(-1);
        }
        else
        {
            if (!onnxDim.dim_param().empty())
            {
                namedDims.emplace_back(static_cast<int32_t>(onnxDimsVec.size()), onnxDim.dim_param());
            }
            const int32_t dim = onnxDim.dim_param() == "" ? (onnxDim.dim_value() >= 0 ? onnxDim.dim_value() : -1) : -1;
            onnxDimsVec.emplace_back(dim);
        }
    }
    trtDims.nbDims = onnxDimsVec.size();
    assert(trtDims.nbDims <= nvinfer1::Dims::MAX_DIMS);
    std::copy(onnxDimsVec.begin(), onnxDimsVec.end(), trtDims.d);
    return true;
}

// Removes raw data from the text representation of an ONNX model
void remove_raw_data_strings(std::string& s)
{
    std::string::size_type beg = 0;
    const std::string key = "raw_data: \"";
    const std::string sub = "...";
    while ((beg = s.find(key, beg)) != std::string::npos)
    {
        beg += key.length();
        std::string::size_type end = beg - 1;
        // Note: Must skip over escaped end-quotes
        while (s[(end = s.find("\"", ++end)) - 1] == '\\')
        {
        }
        if (end - beg > 128)
        { // Only remove large data strings
            s.replace(beg, end - beg, "...");
        }
        beg += sub.length();
    }
}

// Removes float_data, int32_data etc. from the text representation of an ONNX model
std::string remove_repeated_data_strings(std::string& s)
{
    std::istringstream iss(s);
    std::ostringstream oss;
    bool is_repeat = false;
    for (std::string line; std::getline(iss, line);)
    {
        if (line.find("float_data:") != std::string::npos || line.find("int32_data:") != std::string::npos
            || line.find("int64_data:") != std::string::npos)
        {
            if (!is_repeat)
            {
                is_repeat = true;
                oss << line.substr(0, line.find(":") + 1) << " ...\n";
            }
        }
        else
        {
            is_repeat = false;
            oss << line << "\n";
        }
    }
    return oss.str();
}

} // anonymous namespace

inline std::string pretty_print_onnx_to_string(::google::protobuf::Message const& message)
{
    std::string s;
    ::google::protobuf::TextFormat::PrintToString(message, &s);
    remove_raw_data_strings(s);
    s = remove_repeated_data_strings(s);
    return s;
}

inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::ModelProto const& message)
{
    stream << pretty_print_onnx_to_string(message);
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::NodeProto const& message)
{
    stream << pretty_print_onnx_to_string(message);
    return stream;
}

//...
//...Consider moving all of the below functions into a stand alone
//...

inline bool ParseFromFile_WAR(google::protobuf::Message* msg, const char* filename)
{

    std::ifstream stream(filename, std::ios::in | std::ios::binary);
    if (!stream)
    {
        std::cerr << "Could not open file " << std::string(filename) << std::endl;
        return false;
    }
    google::protobuf::io::IstreamInputStream rawInput(&stream);

    google::protobuf::io::CodedInputStream coded_input(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    // Starting Protobuf 3.11 accepts only single parameter.
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
    // Note: This WARs the very low default size limit (64MB)
    coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
#endif
    return msg->ParseFromCodedStream(&coded_input);
}

inline bool ParseFromTextFile(google::protobuf::Message* msg, const char* filename)
{
    std::ifstream stream(filename, std::ios::in);
    if (!stream)
    {
        std::cerr << "Could not open file " << std::string(filename) << std::endl;
        return false;
    }

    google::protobuf::io::IstreamInputStream rawInput(&stream);

    return google::protobuf::TextFormat::Parse(&rawInput, msg);
}

inline std::string onnx_ir_version_string(int64_t ir_version = ::ONNX_NAMESPACE::IR_VERSION)
{
    int onnx_ir_major = ir_version / 1000000;
    int onnx_ir_minor = ir_version % 1000000 / 10000;
    int onnx_ir_patch = ir_version % 10000;
    return (std::to_string(onnx_ir_major) + "." + std::to_string(onnx_ir_minor) + "." + std::to_string(onnx_ir_patch));
}
