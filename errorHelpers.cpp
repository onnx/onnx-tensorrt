/*
 * SPDX-License-Identifier: Apache-2.0
 */
#include "errorHelpers.hpp"

namespace onnx2trt
{
OnnxTrtException::OnnxTrtException(Status status)
    : mStatus(status)
{
}
Status OnnxTrtException::getStatus() const noexcept
{
    return mStatus;
}
char const* OnnxTrtException::what() const noexcept
{
    if (mMessage.empty())
    {
        mMessage = parserErrorStr(&mStatus);
    }
    return mMessage.c_str();
}

nvinfer1::ErrorCode errorCodeToTrtCode(ErrorCode const code)
{
    switch (code)
    {
    case ErrorCode::kSUCCESS: return nvinfer1::ErrorCode::kSUCCESS;

    case ErrorCode::kINTERNAL_ERROR:
    case ErrorCode::kMODEL_DESERIALIZE_FAILED:
    case ErrorCode::kREFIT_FAILED:
    {
        return nvinfer1::ErrorCode::kINTERNAL_ERROR;
    }

    case ErrorCode::kMEM_ALLOC_FAILED:
    {
        return nvinfer1::ErrorCode::kFAILED_ALLOCATION;
    }

    case ErrorCode::kINVALID_VALUE:
    case ErrorCode::kINVALID_GRAPH:
    case ErrorCode::kINVALID_NODE:
    case ErrorCode::kUNSUPPORTED_GRAPH:
    case ErrorCode::kUNSUPPORTED_NODE:
    case ErrorCode::kUNSUPPORTED_NODE_ATTR:
    case ErrorCode::kUNSUPPORTED_NODE_INPUT:
    case ErrorCode::kUNSUPPORTED_NODE_DATATYPE:
    case ErrorCode::kUNSUPPORTED_NODE_DYNAMIC:
    case ErrorCode::kUNSUPPORTED_NODE_SHAPE:
    {
        return nvinfer1::ErrorCode::kINVALID_ARGUMENT;
    }
    }
    return nvinfer1::ErrorCode::kINTERNAL_ERROR;
}
} // namespace onnx2trt
