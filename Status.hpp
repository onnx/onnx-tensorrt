/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "NvOnnxParser.h"

#include <cassert>
#include <string>

#ifndef ENABLE_STD_PLUGIN
#define ENABLE_STD_PLUGIN 1
#endif // ENABLE_STD_PLUGIN

#ifndef ENABLE_SAFE_PLUGIN
#define ENABLE_SAFE_PLUGIN 0
#endif // ENABLE_SAFE_PLUGIN

// Used to strip out build path information from debug prints
#if defined(SOURCE_LENGTH)
#define __FILENAME__ (__FILE__ + SOURCE_LENGTH)
#else
#define __FILENAME__ (__FILE__)
#endif

#define MAKE_ERROR(desc, code) onnx2trt::Status((code), (desc), __FILENAME__, __LINE__, __func__)

#define ASSERT(condition, error_code)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return MAKE_ERROR("Assertion failed: " #condition, (error_code));                                          \
        }                                                                                                              \
    } while (0)

#define MAKE_INPUT_ERROR(desc, code, name) Status((code), (desc), name, __LINE__, __func__)

#define ASSERT_INPUT(condition, error_code, name)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return MAKE_INPUT_ERROR("Assertion failed: " #condition, (error_code), (name));                            \
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

#define CHECK(call)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        Status status = call;                                                                                          \
        if (!status.is_success())                                                                                      \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

namespace onnx2trt
{

using nvonnxparser::ErrorCode;

class Status : public nvonnxparser::IParserError
{
    ErrorCode _code;
    std::string _desc;
    std::string _file;
    int _line;
    std::string _func;
    int _node;

public:
    static Status success()
    {
        return Status(ErrorCode::kSUCCESS);
    }
    Status()
    {
    }
    explicit Status(ErrorCode code, std::string desc = "", std::string file = "", int line = 0, std::string func = "",
        int node = -1)
        : _code(code)
        , _desc(desc)
        , _file(file)
        , _line(line)
        , _func(func)
        , _node(node)
    {
    }
    ErrorCode code() const override
    {
        return _code;
    }
    const char* desc() const override
    {
        return _desc.c_str();
    }
    const char* file() const override
    {
        return _file.c_str();
    }
    int line() const override
    {
        return _line;
    }
    const char* func() const override
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
