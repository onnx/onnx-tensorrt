/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "Status.hpp"
#include <NvInferRuntime.h>
#include <exception>
#include <sstream>
#include <stdexcept>

#define ONNXTRT_TRY try

#define ONNXTRT_CATCH_RECORD                                                                                           \
    catch (OnnxTrtException & e)                                                                                       \
    {                                                                                                                  \
        Status status = e.getStatus();                                                                                 \
        mImporterCtx.getErrorRecorder()->reportError(errorCodeToTrtCode(status.code()), e.what());                     \
        mErrors.push_back(status);                                                                                     \
    }                                                                                                                  \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        mImporterCtx.getErrorRecorder()->reportError(nvinfer1::ErrorCode::kUNSPECIFIED_ERROR, e.what());               \
        mErrors.push_back(Status{ErrorCode::kINTERNAL_ERROR, e.what()});                                               \
    }

#define ONNXTRT_CATCH_LOG(logger)                                                                                      \
    catch (OnnxTrtException & e)                                                                                       \
    {                                                                                                                  \
        Status status = e.getStatus();                                                                                 \
        (logger)->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());                                         \
        mErrors.push_back(status);                                                                                     \
    }                                                                                                                  \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        (logger)->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());                                         \
        mErrors.push_back(Status{ErrorCode::kINTERNAL_ERROR, e.what()});                                               \
    }

#define ONNXTRT_THROW(status) throw OnnxTrtException(status)

#define ONNXTRT_CHECK(cond, desc, code)                                                                                \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        std::ostringstream ss;                                                                                         \
        ss << "Assertion failed: " << #cond << ": " << desc;                                                           \
        ONNXTRT_THROW(MAKE_ERROR(ss.str(), (code)));                                                                   \
    }

#define ONNXTRT_CHECK_NODE(cond, desc, node, nodeIdx, code)                                                            \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        std::ostringstream ss;                                                                                         \
        ss << "Assertion failed: " << #cond << ": " << desc;                                                           \
        ONNXTRT_THROW(MAKE_NODE_ERROR((ss.str()), (code), (node), (nodeIdx)));                                         \
    }

namespace onnx2trt
{
inline char const* errorCodeStr(ErrorCode code)
{
    switch (code)
    {
    case ErrorCode::kSUCCESS: return "SUCCESS";
    case ErrorCode::kINTERNAL_ERROR: return "INTERNAL_ERROR";
    case ErrorCode::kMEM_ALLOC_FAILED: return "MEM_ALLOC_FAILED";
    case ErrorCode::kMODEL_DESERIALIZE_FAILED: return "MODEL_DESERIALIZE_FAILED";
    case ErrorCode::kINVALID_VALUE: return "INVALID_VALUE";
    case ErrorCode::kINVALID_GRAPH: return "INVALID_GRAPH";
    case ErrorCode::kINVALID_NODE: return "INVALID_NODE";
    case ErrorCode::kUNSUPPORTED_GRAPH: return "UNSUPPORTED_GRAPH";
    case ErrorCode::kUNSUPPORTED_NODE: return "UNSUPPORTED_NODE";
    case ErrorCode::kUNSUPPORTED_NODE_ATTR: return "UNSUPPORTED_NODE_ATTR";
    case ErrorCode::kUNSUPPORTED_NODE_INPUT: return "UNSUPPORTED_NODE_INPUT";
    case ErrorCode::kUNSUPPORTED_NODE_DATATYPE: return "UNSUPPORTED_NODE_DATATYPE";
    case ErrorCode::kUNSUPPORTED_NODE_DYNAMIC: return "UNSUPPORTED_NODE_DYNAMIC";
    case ErrorCode::kUNSUPPORTED_NODE_SHAPE: return "UNSUPPORTED_NODE_SHAPE";
    case ErrorCode::kREFIT_FAILED: return "REFIT_FAILED";
    }
    return "UNKNOWN";
}

inline std::string const parserErrorStr(nvonnxparser::IParserError const* error)
{
    std::string const nodeInfo = "In node " + std::to_string(error->node()) + " with name: " + error->nodeName()
        + " and operator: " + error->nodeOperator() + " ";
    std::string const errorInfo
        = std::string("(") + error->func() + "): " + errorCodeStr(error->code()) + ": " + error->desc();
    if (error->code() == ErrorCode::kMODEL_DESERIALIZE_FAILED || error->code() == ErrorCode::kREFIT_FAILED)
    {
        return errorInfo.c_str();
    }
    return (nodeInfo + errorInfo).c_str();
}

nvinfer1::ErrorCode errorCodeToTrtCode(ErrorCode const code);

class OnnxTrtException : public std::exception
{
    Status mStatus;
    mutable std::string mMessage;

public:
    OnnxTrtException(Status status);

    Status getStatus() const noexcept;

    virtual char const* what() const noexcept override;

    virtual ~OnnxTrtException() {}
};

} // namespace onnx2trt
