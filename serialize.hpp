/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstring>
#include <vector>
#include <cassert>
#include <type_traits>

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

template<typename T>
inline void serialize_value(void** buffer, T const& value);

template<typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size, T* value);

namespace {

template<typename T, class Enable=void>
struct Serializer {};

template<typename T>
struct Serializer<T, typename std::enable_if<
                       std::is_arithmetic<T>::value ||
                       std::is_enum<T>::value ||
                       std::is_pod<T>::value>::type> {
  static size_t serialized_size(T const& value) {
    return sizeof(T);
  }
  static void serialize(void** buffer, T const& value) {
    ::memcpy(*buffer, &value, sizeof(T));
    reinterpret_cast<char*&>(*buffer) += sizeof(T);
  }
  static void deserialize(void const** buffer, size_t* buffer_size, T* value) {
    assert(*buffer_size >= sizeof(T));
    ::memcpy(value, *buffer, sizeof(T));
    reinterpret_cast<char const*&>(*buffer) += sizeof(T);
    *buffer_size -= sizeof(T);
  }
};

template<>
struct Serializer<const char*> {
  static size_t serialized_size(const char* value) {
    return strlen(value) + 1;
  }
  static void serialize(void** buffer, const char* value) {
    ::strcpy(static_cast<char*>(*buffer), value);
    reinterpret_cast<char*&>(*buffer) += strlen(value) + 1;
  }
  static void deserialize(void const** buffer, size_t* buffer_size, const char** value) {
    *value = static_cast<char const*>(*buffer);
    size_t data_size = strnlen(*value, *buffer_size) + 1;
    assert(*buffer_size >= data_size);
    reinterpret_cast<char const*&>(*buffer) += data_size;
    *buffer_size -= data_size;
  }
};

template<typename T>
struct Serializer<std::vector<T>, typename std::enable_if<
                                    std::is_arithmetic<T>::value ||
                                    std::is_enum<T>::value ||
                                    std::is_pod<T>::value>::type> {
  static size_t serialized_size(std::vector<T> const& value) {
    return sizeof(value.size()) + value.size() * sizeof(T);
  }
  static void serialize(void** buffer, std::vector<T> const& value) {
    serialize_value(buffer, value.size());
    size_t nbyte = value.size() * sizeof(T);
    ::memcpy(*buffer, value.data(), nbyte);
    reinterpret_cast<char*&>(*buffer) += nbyte;
  }
  static void deserialize(void const** buffer, size_t* buffer_size, std::vector<T>* value) {
    size_t size;
    deserialize_value(buffer, buffer_size, &size);
    value->resize(size);
    size_t nbyte = value->size() * sizeof(T);
    assert(*buffer_size >= nbyte);
    ::memcpy(value->data(), *buffer, nbyte);
    reinterpret_cast<char const*&>(*buffer) += nbyte;
    *buffer_size -= nbyte;
  }
};

} // namespace

template<typename T>
inline size_t serialized_size(T const& value) {
  return Serializer<T>::serialized_size(value);
}

template<typename T>
inline void serialize_value(void** buffer, T const& value) {
  return Serializer<T>::serialize(buffer, value);
}

template<typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size, T* value) {
  return Serializer<T>::deserialize(buffer, buffer_size, value);
}


