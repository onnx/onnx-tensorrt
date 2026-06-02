/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <onnx/onnx-ml.pb.h>
#include <unordered_map>
#include <vector>

#include "ImporterContext.hpp"

class OnnxAttrs
{
    template <typename T>
    using string_map = std::unordered_map<std::string, T>;
    typedef string_map<::ONNX_NAMESPACE::AttributeProto const*> AttrMap;
    AttrMap mAttrs;
    onnx2trt::ImporterContext* mCtx;

public:
    explicit OnnxAttrs(::ONNX_NAMESPACE::NodeProto const& onnx_node, onnx2trt::ImporterContext* ctx)
        : mCtx{ctx}
    {
        for (auto const& attr : onnx_node.attribute())
        {
            mAttrs.insert({attr.name(), &attr});
        }
    }

    bool count(std::string const& key) const
    {
        return mAttrs.count(key);
    }

    ::ONNX_NAMESPACE::AttributeProto const* at(std::string key) const
    {
        if (auto it = mAttrs.find(key); it != mAttrs.end())
        {
            return it->second;
        }
        throw std::out_of_range("Attribute not found: " + key);
    }

    bool exists(std::string key) const
    {
        return mAttrs.count(key) != 0;
    }

    ::ONNX_NAMESPACE::AttributeProto::AttributeType type(std::string const& key) const
    {
        return this->at(key)->type();
    }

    template <typename T>
    T get(std::string const& key) const;

    template <typename T>
    T get(std::string const& key, T const& default_value) const
    {
        return mAttrs.count(key) ? this->get<T>(key) : default_value;
    }
};
