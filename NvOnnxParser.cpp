/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "NvOnnxParser.h"
#include "ModelImporter.hpp"
#include "ModelRefitter.hpp"
#include "NvInferRuntime.h"

extern "C" void* createNvOnnxParser_INTERNAL(void* network_, void* logger_, int version) noexcept
{
    auto network = static_cast<nvinfer1::INetworkDefinition*>(network_);
    auto logger = static_cast<nvinfer1::ILogger*>(logger_);
    return new onnx2trt::ModelImporter(network, logger);
}

extern "C" void* createNvOnnxParserRefitter_INTERNAL(void* refitter_, void* logger_, int32_t version) noexcept
{
    auto refitter = static_cast<nvinfer1::IRefitter*>(refitter_);
    auto logger = static_cast<nvinfer1::ILogger*>(logger_);
    return new onnx2trt::ModelRefitter(refitter, logger);
}

extern "C" int getNvOnnxParserVersion() noexcept
{
    return NV_ONNX_PARSER_VERSION;
}
