/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxProtoUtils.hpp"

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <onnx/onnx_pb.h>
#include <sstream>

namespace onnx2trt
{
void removeRawDataStrings(std::string& s)
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

std::string removeRepeatedDataStrings(std::string const& s)
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

std::string convertProtoToString(::google::protobuf::Message const& message)
{
    std::string s;
    ::google::protobuf::TextFormat::PrintToString(message, &s);
    removeRawDataStrings(s);
    s = removeRepeatedDataStrings(s);
    return s;
}

bool ParseFromFileAsBinary(google::protobuf::Message* msg, char const* filename)
{

    std::ifstream stream(filename, std::ios::in | std::ios::binary);
    if (!stream)
    {
        std::cerr << "Could not open file " << std::string(filename) << std::endl;
        return false;
    }
    google::protobuf::io::IstreamInputStream rawInput(&stream);

    google::protobuf::io::CodedInputStream codedInput(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    // Starting Protobuf 3.11 accepts only single parameter.
    codedInput.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
    // Note: This WARs the very low default size limit (64MB)
    codedInput.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
#endif
    return msg->ParseFromCodedStream(&codedInput);
}

std::string onnxIRVersionAsString(int64_t irVersion)
{
    int64_t verMajor = irVersion / 1000000;
    int64_t verMinor = irVersion % 1000000 / 10000;
    int64_t verPatch = irVersion % 10000;
    return (std::to_string(verMajor) + "." + std::to_string(verMinor) + "." + std::to_string(verPatch));
}

Status deserializeOnnxModel(void const* serializedModel, size_t serializedModelSize, ::ONNX_NAMESPACE::ModelProto* model)
{
    google::protobuf::io::ArrayInputStream rawInput(serializedModel, serializedModelSize);
    google::protobuf::io::CodedInputStream codedInput(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
    // Starting Protobuf 3.11 accepts only single parameter.
    codedInput.SetTotalBytesLimit(std::numeric_limits<int>::max());
#else
    // Note: This WARs the very low default size limit (64MB)
    codedInput.SetTotalBytesLimit(std::numeric_limits<int>::max(), std::numeric_limits<int>::max() / 4);
#endif
    ASSERT((model->ParseFromCodedStream(&codedInput)) && "Failed to parse the ONNX model.",
        ErrorCode::kMODEL_DESERIALIZE_FAILED);
    return Status::success();
}

} // namespace onnx2trt
