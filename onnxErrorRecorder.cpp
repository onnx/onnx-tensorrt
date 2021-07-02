/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnxErrorRecorder.hpp"
#include <exception>

namespace onnx2trt
{


ONNXParserErrorRecorder* ONNXParserErrorRecorder::create(
    nvinfer1::ILogger* logger, nvinfer1::IErrorRecorder* otherRecorder)
{
    try
    {
        auto recorder = new ONNXParserErrorRecorder(logger, otherRecorder);
        if (recorder)
        {
            recorder->incRefCount();
        }
        return recorder;
    }
    catch (const std::exception& e)
    {
        logError(logger, e.what());
        return nullptr;
    }
}

void ONNXParserErrorRecorder::destroy(ONNXParserErrorRecorder*& recorder)
{
    if (recorder)
    {
        recorder->decRefCount();
        recorder = nullptr;
    }
}

void ONNXParserErrorRecorder::logError(nvinfer1::ILogger* logger, const char* str)
{
    if (logger)
    {
        logger->log(ILogger::Severity::kERROR, str);
    }
}

ONNXParserErrorRecorder::ONNXParserErrorRecorder(
    nvinfer1::ILogger* logger, nvinfer1::IErrorRecorder* otherRecorder)
    : mUserRecorder(otherRecorder)
    , mLogger(logger)
{
    if (mUserRecorder)
    {
        mUserRecorder->incRefCount();
    }
}

ONNXParserErrorRecorder::~ONNXParserErrorRecorder() noexcept
{
    if (mUserRecorder)
    {
        mUserRecorder->decRefCount();
    }
}

void ONNXParserErrorRecorder::clear() noexcept
{
    try
    {
        // grab a lock so that there is no addition while clearing.
        std::lock_guard<std::mutex> guard(mStackLock);
        mErrorStack.clear();
    }
    catch (const std::exception& e)
    {
        logError(mLogger, e.what());
    }
};

bool ONNXParserErrorRecorder::reportError(
    nvinfer1::ErrorCode val, nvinfer1::IErrorRecorder::ErrorDesc desc) noexcept
{
    try
    {
        std::lock_guard<std::mutex> guard(mStackLock);
        mErrorStack.push_back(errorPair(val, desc));
        if (mUserRecorder)
        {
            mUserRecorder->reportError(val, desc);
        }
        else
        {
            logError(mLogger, desc);
        }
    }
    catch (const std::exception& e)
    {
        logError(mLogger, e.what());
    }
    // All errors are considered fatal.
    return true;
}

nvinfer1::IErrorRecorder::RefCount ONNXParserErrorRecorder::incRefCount() noexcept
{
    // Atomically increment or decrement the ref counter.
    return ++mRefCount;
}

nvinfer1::IErrorRecorder::RefCount ONNXParserErrorRecorder::decRefCount() noexcept
{
    auto newVal = --mRefCount;
    if (newVal == 0)
    {
        delete this;
    }
    return newVal;
}

} // namespace onnx2trt
