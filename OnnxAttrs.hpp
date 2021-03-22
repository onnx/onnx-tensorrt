/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <onnx/onnx_pb.h>
#include <unordered_map>
#include <vector>

#include "ImporterContext.hpp"

class OnnxAttrs
{
    template <typename T>
    using string_map = std::unordered_map<std::string, T>;
    typedef string_map<::ONNX_NAMESPACE::AttributeProto const*> AttrMap;
    AttrMap _attrs;
    onnx2trt::IImporterContext* mCtx;

public:
    explicit OnnxAttrs(::ONNX_NAMESPACE::NodeProto const& onnx_node, onnx2trt::IImporterContext* ctx)
        : mCtx{ctx}
    {
        for (auto const& attr : onnx_node.attribute())
        {
            _attrs.insert({attr.name(), &attr});
        }
    }

    bool count(const std::string& key) const
    {
        return _attrs.count(key);
    }

    ::ONNX_NAMESPACE::AttributeProto const* at(std::string key) const
    {
        if (!_attrs.count(key))
        {
            throw std::out_of_range("Attribute not found: " + key);
        }
        return _attrs.at(key);
    }

    ::ONNX_NAMESPACE::AttributeProto::AttributeType type(const std::string& key) const
    {
        return this->at(key)->type();
    }


    template <typename T>
    T get(const std::string& key) const;

    template <typename T>
    T get(const std::string& key, T const& default_value) const
    {
        return _attrs.count(key) ? this->get<T>(key) : default_value;
    }
};
