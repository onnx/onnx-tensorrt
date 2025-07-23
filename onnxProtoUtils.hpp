/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Status.hpp"
#include "errorHelpers.hpp"
#include <iostream>
#include <onnx/onnx_pb.h>
#include <sstream>

#include <fstream>
#include <string>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#if USE_LITE_PROTOBUF
#include <google/protobuf/message_lite.h>
#else // !USE_LITE_PROTOBUF
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#endif // USE_LITE_PROTOBUF

// This file contains the declaration of helper functions used for converting and working with Protobuf files.

namespace onnx2trt
{

// Removes raw data from the text representation of an ONNX model.
void removeRawDataStrings(std::string& s);

// Removes float_data, int32_data etc. from the text representation of an ONNX model.
std::string removeRepeatedDataStrings(std::string const& s);

// Returns the ONNX IR version as a string.
std::string onnxIRVersionAsString(int64_t ir_version = ::ONNX_NAMESPACE::IR_VERSION);

// Converts a raw protobuf::Message or protobuf::MessageLite into a string representation.
template <typename ProtoMessage>
std::string convertProtoToString(ProtoMessage const& message)
{
    std::string s{};
// Textformat available in full proto only. Return only the name when using protobuf-lite.
#if USE_LITE_PROTOBUF
    s = "Node name: " + message.name();
    return s;
#else
    ::google::protobuf::TextFormat::PrintToString(message, &s);
    removeRawDataStrings(s);
    s = removeRepeatedDataStrings(s);
    return s;
#endif // USE_LITE_PROTOBUF
}

// Deserializes an ONNX ModelProto passed in as a protobuf::Message or a protobuf::MessageLite.
template <typename ProtoMessage>
void deserializeOnnxModel(void const* serializedModel, size_t serializedModelSize, ProtoMessage* model)
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

    ONNXTRT_CHECK(model->ByteSizeLong() == 0, "A model was previously parsed!", ErrorCode::kMODEL_DESERIALIZE_FAILED);

    ONNXTRT_CHECK(model->ParseFromCodedStream(&codedInput), "Failed to parse the ONNX model.",
        ErrorCode::kMODEL_DESERIALIZE_FAILED);
}

// Helper function to dispatch to deserializeOnnxModel when user provides a path to the model.
template <typename ProtoMessage>
bool ParseFromFileAsBinary(ProtoMessage* msg, char const* filename)
{
    std::ifstream onnxFile(filename, std::ios::ate | std::ios::binary);
    if (!onnxFile)
    {
        std::cerr << "Could not open file " << std::string(filename) << std::endl;
        return false;
    }
    // Determine the file size
    auto fileSize = onnxFile.tellg();
    onnxFile.seekg(0, std::ios::beg);

    // Create buffer and read tne entire file to the buffer.
    std::vector<char> buffer(fileSize);
    if (!onnxFile.read(buffer.data(), fileSize))
    {
        std::cerr << "Error reading file: " << filename << std::endl;
        return false;
    }

    deserializeOnnxModel(buffer.data(), buffer.size(), msg);
    return true;
}

// ostream overload for printing NodeProtos.
inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::NodeProto const& message)
{
    stream << convertProtoToString(message);
    return stream;
}

} // namespace onnx2trt
