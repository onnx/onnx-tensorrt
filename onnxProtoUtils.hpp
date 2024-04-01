/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Status.hpp"
#include <iostream>
#include <onnx/onnx_pb.h>
#include <sstream>

// This file contains the declaration of helper functions used for converting and working with Protobuf files.

namespace onnx2trt
{

// Converts a raw protobuf message into a string representation.
std::string convertProtoToString(::google::protobuf::Message const& message);

// ostream overload for printing ModelProtos
inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::ModelProto const& message)
{
    stream << convertProtoToString(message);
    return stream;
}

// ostream overload for printing NodeProtos
inline std::ostream& operator<<(std::ostream& stream, ::ONNX_NAMESPACE::NodeProto const& message)
{
    stream << convertProtoToString(message);
    return stream;
}

// Removes raw data from the text representation of an ONNX model.
void removeRawDataStrings(std::string& s);

// Removes float_data, int32_data etc. from the text representation of an ONNX model.
std::string removeRepeatedDataStrings(std::string const& s);

// Parses a protobuf file from disk, reading as a binary file.
bool ParseFromFileAsBinary(google::protobuf::Message* msg, char const* filename);

std::string onnxIRVersionAsString(int64_t ir_version = ::ONNX_NAMESPACE::IR_VERSION);

Status deserializeOnnxModel(void const* serializedModel, size_t serializedModelSize, ::ONNX_NAMESPACE::ModelProto* model);

} // namespace onnx2trt
