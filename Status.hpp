/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvOnnxParser.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>
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

// Logging macros
#define LOG(msg, severity)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        std::stringstream ss{};                                                                                        \
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)                                                         \
            ss << ONNX2TRT_FILENAME << ":" << __LINE__ << ": ";                                                        \
        ss << msg;                                                                                                     \
        ctx->logger().log(severity, ss.str().c_str());                                                                 \
    } while (0)

#define LOG_VERBOSE(msg) LOG(msg, nvinfer1::ILogger::Severity::kVERBOSE)
#define LOG_INFO(msg) LOG(msg, nvinfer1::ILogger::Severity::kINFO)
#define LOG_WARNING(msg) LOG(msg, nvinfer1::ILogger::Severity::kWARNING)
#define LOG_ERROR(msg) LOG(msg, nvinfer1::ILogger::Severity::kERROR)

#define MAKE_ERROR(desc, code) onnx2trt::Status((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__)

#define ASSERT(condition, error_code)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return MAKE_ERROR("Assertion failed: " #condition, (error_code));                                          \
        }                                                                                                              \
    } while (0)

#define MAKE_NODE_ERROR(desc, code, node, index)                                                                       \
    onnx2trt::Status((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), (node.name()), (node.op_type()))

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

#define MAKE_STATIC_ERROR(desc, code, node, index)                                                                     \
    onnx2trt::Status((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index), (node.name()), (node.op_type()))

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
        error_list.push_back(onnx2trt::Status((code), (desc), ONNX2TRT_FILENAME, __LINE__, __func__, (index),          \
            (node.name()), (node.op_type()), localFunctionStackChar));                                                 \
    } while (0)

#define STATIC_CHECK(condition, error_code, node, error_list, index)                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            ADD_STATIC_ERROR(#condition, (error_code), node, index, error_list);                                       \
        }                                                                                                              \
    } while (0)

#define MAKE_INPUT_ERROR(desc, code, name) Status((code), (desc), name, __LINE__, __func__)

#define CHECK_INPUT(condition, error_code, name, error_list)                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            error_list.push_back(MAKE_INPUT_ERROR("Assertion failed: " #condition, (error_code), (name)));             \
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
            *result_ptr = value_or_error.value();                                                                      \
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
std::ostream& printSequence(std::ostream& stream, const T* begin, int count);

std::ostream& operator<<(std::ostream& stream, nvinfer1::Dims const& shape);

std::ostream& operator<<(std::ostream& stream, nvinfer1::Permutation const& perm);

std::ostream& operator<<(std::ostream& stream, nvinfer1::DataType const& dtype);

using nvonnxparser::ErrorCode;

class Status : public nvonnxparser::IParserError
{
    ErrorCode _code;
    std::string _desc;
    std::string _file;
    int _line;
    std::string _func;
    int _node;
    std::string _nodeName;
    std::string _nodeOperator;
    std::vector<char const*> _localFunctionStack;

public:
    static Status success()
    {
        return Status(ErrorCode::kSUCCESS);
    }
    Status() {}
    explicit Status(ErrorCode code, std::string desc = "", std::string file = "", int line = 0, std::string func = "",
        int node = -1, std::string nodeName = "", std::string nodeOperator = "",
        std::vector<char const*> localFunctionStack = {})
        : _code(code)
        , _desc(desc)
        , _file(file)
        , _line(line)
        , _func(func)
        , _node(node)
        , _nodeName(nodeName)
        , _nodeOperator(nodeOperator)
        , _localFunctionStack(localFunctionStack)
    {
    }
    ErrorCode code() const override
    {
        return _code;
    }
    char const* desc() const override
    {
        return _desc.c_str();
    }
    char const* file() const override
    {
        return _file.c_str();
    }
    int line() const override
    {
        return _line;
    }
    char const* func() const override
    {
        return _func.c_str();
    }
    int node() const override
    {
        return _node;
    }
    bool is_error() const
    {
        return _code != ErrorCode::kSUCCESS;
    }
    bool is_success() const
    {
        return _code == ErrorCode::kSUCCESS;
    }
    void setNode(int node)
    {
        _node = node;
    }
    char const* nodeName() const override
    {
        return _nodeName.c_str();
    }
    char const* nodeOperator() const override
    {
        return _nodeOperator.c_str();
    }
    char const* const* localFunctionStack() const override
    {
        return _localFunctionStack.data();
    }
    int32_t localFunctionStackSize() const override
    {
        return _localFunctionStack.size();
    }
};

template <typename T>
class ValueOrStatus
{
    bool _is_error;
    T _value;
    Status _error;

public:
    ValueOrStatus(T const& value)
        : _is_error(false)
        , _value(value)
        , _error(Status::success())
    {
    }
    ValueOrStatus(T&& value)
        : _is_error(false)
        , _value(value)
        , _error(Status::success())
    {
    }
    ValueOrStatus(Status const& error)
        : _is_error(true)
        , _error(error)
    {
    }
    ValueOrStatus(Status&& error)
        : _is_error(true)
        , _error(error)
    {
    }
    bool is_error() const
    {
        return _is_error;
    }
    T const& value() const
    {
        assert(!_is_error);
        return _value;
    }
    T& value()
    {
        assert(!_is_error);
        return _value;
    }
    Status const& error() const
    {
        assert(_is_error);
        return _error;
    }
};

} // namespace onnx2trt
