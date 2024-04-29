/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ShapedWeights.hpp"
#include "importerUtils.hpp"
#include <cstdint>
#include <cstring>
#include <limits>

namespace onnx2trt
{

size_t ShapedWeights::count() const
{
    assert(shape.nbDims >= 0);
    size_t c = 1;
    for (int32_t i = 0; i < this->shape.nbDims; ++i)
    {
        if (shape.d[i] == 0)
        {
            c = 0;
            break;
        }
        if (c > std::numeric_limits<size_t>::max() / shape.d[i])
        {
            throw std::runtime_error("Count of weights exceeds maximum!");
        }
        c *= this->shape.d[i];
    }
    return c;
}

ShapedWeights ShapedWeights::empty(DataType type)
{
    return ShapedWeights(type, nullptr, nvinfer1::Dims{1, {0}});
}

ShapedWeights::ShapedWeights(DataType type_, void* values_, nvinfer1::Dims shape_)
    : type(type_)
    , values(values_)
    , shape(shape_)
{
    // Note: this->shape.type[] is not used
}

size_t ShapedWeights::size_bytes() const
{
    return getTensorOrWeightsSizeBytes(this->count(), this->type);
}

ShapedWeights::operator bool() const
{
    return (bool) this->values;
}

ShapedWeights::operator nvinfer1::Weights() const
{
    nvinfer1::Weights w{};
    w.values = this->values;
    bool supported_type = convertDtype(this->type, &w.type);
    (void) supported_type;
    assert(supported_type);
    w.count = this->count();
    return w;
}

const char* ShapedWeights::getName() const
{
    return this->name;
}

void ShapedWeights::setName(const char* n)
{
    this->name = n;
}

} // namespace onnx2trt
