/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvOnnxParser.h"
#include "ModelImporter.hpp"

extern "C" void* createNvOnnxParser_INTERNAL(void* network_, void* logger_, int version)
{
    auto network = static_cast<nvinfer1::INetworkDefinition*>(network_);
    auto logger = static_cast<nvinfer1::ILogger*>(logger_);
    return new onnx2trt::ModelImporter(network, logger);
}

extern "C" int getNvOnnxParserVersion()
{
    return NV_ONNX_PARSER_VERSION;
}
