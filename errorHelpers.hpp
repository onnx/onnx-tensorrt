/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "Status.hpp"
#include <NvInferRuntime.h>
#include <sstream>
#include <stdexcept>

#define ONNXTRT_TRY try

#define ONNXTRT_CATCH_RECORD                                                                                           \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        mImporterCtx.getErrorRecorder()->reportError(nvinfer1::ErrorCode::kINTERNAL_ERROR, e.what());                  \
        mErrors.push_back(Status{ErrorCode::kINTERNAL_ERROR, e.what()});                                               \
    }

#define ONNXTRT_CATCH_LOG(logger)                                                                                      \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        (logger)->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());                                         \
    }

namespace onnx2trt
{
inline void ONNXTRT_CHECK(bool cond, Status status)
{
    if (!cond)
    {
        std::ostringstream os;
        os << "[" << status.file() << ":" << status.func() << ":" << status.line() << "] ";
        os << "Error Code " << static_cast<int32_t>(status.code()) << ": " << status.desc();

        throw std::runtime_error(os.str());
    }
}
} // namespace onnx2trt
