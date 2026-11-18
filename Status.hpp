/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvOnnxParser.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef ENABLE_STD_PLUGIN
#define ENABLE_STD_PLUGIN 1
#endif // ENABLE_STD_PLUGIN


#ifndef USE_LITE_PROTOBUF
#define USE_LITE_PROTOBUF 0
#endif // USE_LITE_PROTOBUF

// Helper macro that wraps __FILE__ and returns a std::string of the filename at runtime.
#if defined(__GLIBCXX__) && defined(__aarch64__)
[[maybe_unused]] static std::string filename(std::string_view path)
{
    auto pos = path.rfind('/');
    if (pos == std::string_view::npos)
    {
        return std::string(path);
    }
    return std::string(path.substr(pos + 1));
}
#define ONNX2TRT_FILENAME (filename(__FILE__))
#else
#define ONNX2TRT_FILENAME (std::filesystem::path(__FILE__).filename().string())
#endif

// Logging macros. msg is a stream-insertion expression, e.g. LOG_VERBOSE("Shape: " << shape), so it cannot be
// parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define LOG(msg, severity)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        if ((severity) <= nvinfer1::ILogger::Severity::kWARNING)                                                       \
            ss << ONNX2TRT_FILENAME << ":" << __LINE__ << ": ";                                                        \
        ss << msg;                                                                                                     \
        ctx->logger().log(severity, ss.str().c_str());                                                                 \
    } while (0)
// NOLINTEND(bugprone-macro-parentheses)

#define LOG_VERBOSE(msg) LOG(msg, nvinfer1::ILogger::Severity::kVERBOSE)
#define LOG_INFO(msg) LOG(msg, nvinfer1::ILogger::Severity::kINFO)
#define LOG_WARNING(msg) LOG(msg, nvinfer1::ILogger::Severity::kWARNING)
#define LOG_ERROR(msg) LOG(msg, nvinfer1::ILogger::Severity::kERROR)

#define MAKE_ERROR(desc, code) onnx2trt::Status((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__)

// Append a Status built like MAKE_ERROR to error_list, constructing it in place.
#define ADD_ERROR(desc, code, error_list)                                                                              \
    (error_list).emplace_back((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__)

#define ASSERT(condition, error_code)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return MAKE_ERROR("Assertion failed: " #condition, (error_code));                                          \
        }                                                                                                              \
    } while (0)

#define MAKE_NODE_ERROR(desc, code, node, index)                                                                       \
    onnx2trt::Status(                                                                                                  \
        (code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), ((node).name()), ((node).op_type()))

// Append a Status built like MAKE_NODE_ERROR to error_list, constructing it in place.
#define ADD_NODE_ERROR(desc, code, node, index, error_list)                                                            \
    (error_list)                                                                                                       \
        .emplace_back(                                                                                                 \
            (code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), ((node).name()), ((node).op_type()))

// msg is a stream-insertion expression, so it cannot be parenthesized.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define ASSERT_NODE(condition, msg, node, index, error_code)                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::stringstream error;                                                                                   \
            error << "Assertion failed: " << #condition << ": " << msg;                                                \
            return MAKE_NODE_ERROR((error.str()), (error_code), node, index);                                          \
        }                                                                                                              \
    } while (0)
// NOLINTEND(bugprone-macro-parentheses)

#define MAKE_STATIC_ERROR(desc, code, node, index)                                                                     \
    onnx2trt::Status(                                                                                                  \
        (code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), ((node).name()), ((node).op_type()))

#define ADD_STATIC_ERROR(desc, code, node, index, error_list)                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        size_t stackSize = ctx->localFunctionStack().size();                                                           \
        std::vector<std::string> localFunctionStackString{};                                                           \
        std::vector<char const*> localFunctionStackChar{};                                                             \
        for (size_t i = 0; i < stackSize; i++)                                                                         \
        {                                                                                                              \
            auto const& func = ctx->localFunctionStack()[i];                                                           \
            localFunctionStackString.push_back(func.nodeName + " (" + func.functionName + ")");                        \
        }                                                                                                              \
        ctx->localFunctionErrors().push_back(localFunctionStackString);                                                \
        for (size_t i = 0; i < stackSize; i++)                                                                         \
        {                                                                                                              \
            localFunctionStackChar.push_back(ctx->localFunctionErrors().back()[i].c_str());                            \
        }                                                                                                              \
        (error_list)                                                                                                   \
            .emplace_back((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), ((node).name()),             \
                ((node).op_type()), localFunctionStackChar);                                                           \
    } while (0)

#define STATIC_CHECK(condition, error_code, node, error_list, index)                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            ADD_STATIC_ERROR(#condition, (error_code), node, index, error_list);                                       \
        }                                                                                                              \
    } while (0)

#define CHECK_INPUT(condition, error_code, name, error_list)                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            (error_list).emplace_back((error_code), "Assertion failed: " #condition, (name), __LINE__, __func__);      \
        }                                                                                                              \
    } while (0)

#define ASSERT_C(condition, error_code)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return error_code;                                                                                         \
        }                                                                                                              \
    } while (0)

#define GET_VALUE(value_or_error_, result_ptr)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        auto const& value_or_error = value_or_error_;                                                                  \
        if (value_or_error.is_error())                                                                                 \
        {                                                                                                              \
            return value_or_error.error();                                                                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            *(result_ptr) = value_or_error.value();                                                                    \
        }                                                                                                              \
    } while (0)

#define CHECK_STATUS(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        onnx2trt::Status status = call;                                                                                \
        if (!status.is_success())                                                                                      \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

// Nullptr check for added layers and tensors. All added layers and their output tensors
// should be non-null, so throw an exception here if it is null. This exception
// will be caught by the parseNode function.
template <typename T>
T* N_CHECK(T* inputPtr)
{
    if (!inputPtr)
    {
        throw std::runtime_error("Internal Error!");
    }
    return inputPtr;
}

namespace onnx2trt
{

template <typename T>
std::ostream& printSequence(std::ostream& stream, T const* begin, int count);

std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape);

std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm);

std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype);

using nvonnxparser::ErrorCode;

class Status : public nvonnxparser::IParserError
{
    ErrorCode mCode;
    std::string mDesc;
    std::string mFile;
    int32_t mLine;
    std::string mFunc;
    int32_t mNode;
    std::string mNodeName;
    std::string mNodeOperator;
    std::vector<char const*> mLocalFunctionStack;

public:
    static Status success()
    {
        return Status(ErrorCode::kSUCCESS);
    }
    Status() = default;
    explicit Status(ErrorCode code, std::string desc = "", std::string file = "", int32_t line = 0,
        std::string func = "", int32_t node = -1, std::string nodeName = "", std::string nodeOperator = "",
        std::vector<char const*> localFunctionStack = {})
        : mCode(code)
        , mDesc(std::move(desc))
        , mFile(std::move(file))
        , mLine(line)
        , mFunc(std::move(func))
        , mNode(node)
        , mNodeName(std::move(nodeName))
        , mNodeOperator(std::move(nodeOperator))
        , mLocalFunctionStack(std::move(localFunctionStack))
    {
    }
    [[nodiscard]] ErrorCode code() const override
    {
        return mCode;
    }
    [[nodiscard]] char const* desc() const override
    {
        return mDesc.c_str();
    }
    [[nodiscard]] char const* file() const override
    {
        return mFile.c_str();
    }
    [[nodiscard]] int line() const override
    {
        return mLine;
    }
    [[nodiscard]] char const* func() const override
    {
        return mFunc.c_str();
    }
    [[nodiscard]] int node() const override
    {
        return mNode;
    }
    [[nodiscard]] bool is_error() const
    {
        return mCode != ErrorCode::kSUCCESS;
    }
    [[nodiscard]] bool is_success() const
    {
        return mCode == ErrorCode::kSUCCESS;
    }
    void setNode(int32_t node)
    {
        mNode = node;
    }
    [[nodiscard]] char const* nodeName() const override
    {
        return mNodeName.c_str();
    }
    [[nodiscard]] char const* nodeOperator() const override
    {
        return mNodeOperator.c_str();
    }
    [[nodiscard]] char const* const* localFunctionStack() const override
    {
        return mLocalFunctionStack.data();
    }
    [[nodiscard]] int32_t localFunctionStackSize() const override
    {
        return mLocalFunctionStack.size();
    }
};

template <typename T>
class ValueOrStatus
{
    bool mIsError;
    T mValue;
    Status mError;

public:
    ValueOrStatus(T const& value)
        : mIsError(false)
        , mValue(value)
        , mError(Status::success())
    {
    }
    ValueOrStatus(T&& value)
        : mIsError(false)
        , mValue(std::move(value))
        , mError(Status::success())
    {
    }
    ValueOrStatus(Status const& error)
        : mIsError(true)
        , mError(error)
    {
    }
    ValueOrStatus(Status&& error)
        : mIsError(true)
        , mError(std::move(error))
    {
    }
    [[nodiscard]] bool is_error() const
    {
        return mIsError;
    }
    T const& value() const
    {
        assert(!mIsError);
        return mValue;
    }
    T& value()
    {
        assert(!mIsError);
        return mValue;
    }
    [[nodiscard]] Status const& error() const
    {
        assert(mIsError);
        return mError;
    }
};

} // namespace onnx2trt
