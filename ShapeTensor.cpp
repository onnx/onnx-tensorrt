/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "ShapeTensor.hpp"
#include "TensorOrWeights.hpp"
#include "onnx2trt_utils.hpp"
#include <algorithm>
#include <cassert>
#include <functional>

namespace onnx2trt
{

ShapeTensor::ShapeTensor(int rank_, int32_t size_)
    : rank(rank_)
    , size(size_)
{
    assert((rank_ == 0 || rank_ == 1) && "shape tensor must have rank 0 or 1");
    assert(size_ >= 0);
    assert(rank_ > 0 || size_ == 1);
}

ShapeTensor::ShapeTensor(int rank_, std::vector<int64_t>&& values_)
    : ShapeTensor(rank_, values_.size())
{
    values = std::move(values_);
}

ShapeTensor::ShapeTensor(TensorOrWeights& t)
{
    const nvinfer1::Dims d = t.shape();
    assert((d.nbDims == 0 || d.nbDims == 1) && "shape tensor must have rank 0 or 1");
    assert((d.nbDims == 0 || d.d[0] >= 0) && "shape tensor must have build-time shape");
    rank = d.nbDims;
    size = rank == 0 ? 1 : d.d[0];
    if (t.is_tensor())
    {
        mTensor = &t.tensor();
    }
    else
    {
        weightsToVector(t.weights(), &values);
    }
}

void ShapeTensor::assign(const nvinfer1::ILayer* layer)
{
    assert(layer);
    assert(!mTensor && "already assigned");
    mTensor = layer->getOutput(0);
    assert(mTensor);
#ifndef NDEBUG
    const nvinfer1::Dims d = mTensor->getDimensions();
    assert(rank == d.nbDims);
    assert(size == (rank == 0 ? 1 : d.d[0]));
#endif
}

bool ShapeTensor::isAll(int64_t x) const
{
    return valuesKnown() && std::all_of(values.begin(), values.end(), [x](int64_t y) { return x == y; });
}

nvinfer1::ITensor& ShapeTensor::tensor(IImporterContext* ctx) const
{
    if (!mTensor)
    {
        assert(valuesKnown());
        const nvinfer1::Dims dims{rank, {size}, {}};
        const nvinfer1::Weights w{nvinfer1::DataType::kINT32, convertINT64(values.data(), dims, ctx), size};
        mTensor = ctx->network()->addConstant(dims, w)->getOutput(0);
    }
    return *mTensor;
}

ShapeTensor iotaShapeVector(int32_t n)
{
    std::vector<int64_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    return ShapeTensor(1, std::move(values));
}

ShapeTensor similar(const ShapeTensor& exemplar, int64_t value)
{
    return ShapeTensor(exemplar.rank, std::vector<int64_t>(exemplar.size, value));
}

ShapeTensor fillShapeVector(int32_t size, int64_t value)
{
    return ShapeTensor(1, std::vector<int64_t>(size, value));
}

using nvinfer1::ElementWiseOperation;

//! Helper that implements an elementwise operations on two shape tensors x and y.
//! f must implement the operation on a pair of int64_t.
//! commutes should be true f is commutative.
//! rightIdentity should be the right identity value for f.
static ShapeTensor op(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y, ElementWiseOperation operation,
    bool commutative, int64_t rightIdentity, const std::function<int64_t(int64_t, int64_t)>&& f)
{
    assert(x.rank == y.rank);
    assert(x.size == 1 || y.size == 1 || x.size == y.size);
    if (y.isAll(rightIdentity) && y.size <= x.size)
    {
        return x;
    }
    if (commutative && x.isAll(rightIdentity) && x.size <= y.size)
    {
        return y;
    }
    ShapeTensor z(x.rank, std::max(x.size, y.size));
    if (x.valuesKnown() && y.valuesKnown())
    {
        z.values.resize(z.size);
        for (int i = 0; i < z.size; ++i)
        {
            // The % simulates broadcast rules.
            z.values[i] = f(x.values[i % x.size], y.values[i % y.size]);
        }
    }
    else
    {
        z.assign(ctx->network()->addElementWise(x.tensor(ctx), y.tensor(ctx), operation));
    }
    return z;
}

ShapeTensor add(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kSUM, true, 0, std::plus<int64_t>());
}

ShapeTensor sub(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kSUB, false, 0, std::minus<int64_t>());
}

ShapeTensor mul(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kPROD, true, 1, std::multiplies<int64_t>());
}

ShapeTensor min(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kMIN, true, std::numeric_limits<int64_t>::max(),
        [](int64_t x, int64_t y) { return std::min(x, y); });
}

ShapeTensor max(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kMAX, true, std::numeric_limits<int64_t>::min(),
        [](int64_t x, int64_t y) { return std::max(x, y); });
}

ShapeTensor floorDiv(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    return op(ctx, x, y, ElementWiseOperation::kFLOOR_DIV, false, 1, [](int64_t x, int64_t y) {
        assert(y != 0 && "divisor must be non-zero");
        const int64_t d = x / y;
        return d * y == x ? d : d - ((x < 0) ^ (y < 0));
    });
}

ShapeTensor product(IImporterContext* ctx, const ShapeTensor& x, int first, int last, int rank)
{
    assert(first <= last);
    ShapeTensor z(rank, std::vector<int64_t>(1, 1));
    for (int i = first; i < last; ++i)
    {
        z = mul(ctx, z, gather(ctx, x, ShapeTensor(rank, std::vector<int64_t>(1, i))));
    }
    return z;
}

ShapeTensor concat(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    assert(x.rank == 1);
    assert(y.rank == 1);
    // Optimizing the case x.size == 0 or y.size == 0 not only reduces clutter,
    // but is required because TensorRT does not allow empty tensors.
    if (x.size == 0)
    {
        return y;
    }
    if (y.size == 0)
    {
        return x;
    }
    ShapeTensor z(1, x.size + y.size);
    if (x.valuesKnown() && y.valuesKnown())
    {
        z.values.resize(z.size);
        auto p = std::copy(x.values.begin(), x.values.end(), z.values.begin());
        std::copy(y.values.begin(), y.values.end(), p);
    }
    else
    {
        nvinfer1::ITensor* const args[2] = {&x.tensor(ctx), &y.tensor(ctx)};
        z.assign(ctx->network()->addConcatenation(args, 2));
    }
    return z;
}

ShapeTensor gather(IImporterContext* ctx, const ShapeTensor& data, const ShapeTensor& indices)
{
    assert(data.rank == 1);
    ShapeTensor z(indices.rank, indices.size);
    // Optimizing the case indices.size == 0 is required, because TensorRT does not allow empty tensors.
    if ((data.valuesKnown() && indices.valuesKnown()) || indices.size == 0)
    {
        z.values.resize(z.size);
        std::transform(indices.values.begin(), indices.values.end(), z.values.begin(), [&](int64_t i) {
            assert(0 <= i);
            assert(static_cast<size_t>(i) < data.values.size());
            return data.values[i];
        });
    }
    else
    {
        z.assign(ctx->network()->addGather(data.tensor(ctx), indices.tensor(ctx), 0));
    }
    return z;
}

ShapeTensor shapeOf(IImporterContext* ctx, nvinfer1::ITensor& tensor)
{
    const nvinfer1::Dims d = tensor.getDimensions();
    ShapeTensor z(1, d.nbDims);
    if (isDynamic(d))
    {
        z.assign(ctx->network()->addShape(tensor));
    }
    else
    {
        z.values.resize(d.nbDims);
        std::copy_n(d.d, d.nbDims, z.values.begin());
    }
    return z;
}

ShapeTensor shapeOf(IImporterContext* ctx, TensorOrWeights& t)
{
    if (t.is_tensor())
    {
        return shapeOf(ctx, t.tensor());
    }
    else
    {
        const nvinfer1::Dims& d = t.weights().shape;
        return ShapeTensor(1, std::vector<int64_t>(d.d, d.d + d.nbDims));
    }
}

ShapeTensor convertTo1D(IImporterContext* ctx, const ShapeTensor& tensor)
{
    assert(tensor.rank == 0);
    assert(tensor.size == 1);
    ShapeTensor result(1, 1);
    if (tensor.valuesKnown())
    {
        result.values = tensor.values;
    }
    else
    {
        result.assign(addShuffle(ctx, tensor.tensor(ctx), shapeVector(1)));
    }
    return result;
}

//! If x has known values, return Dims with those values.
//! Otherwise return Dims with zeros.
static nvinfer1::Dims toDims(const ShapeTensor& x)
{
    nvinfer1::Dims d{x.size, {}, {}};
    if (x.valuesKnown())
    {
        assert(x.size <= nvinfer1::Dims::MAX_DIMS);
        std::copy(x.values.begin(), x.values.end(), d.d);
    }
    return d;
}

//! If x does not have known values, set layer input to tensor with value of x.
static void setShapeInputIfDynamic(IImporterContext* ctx, nvinfer1::ILayer* layer, int inputIndex, const ShapeTensor& x)
{
    if (!x.valuesKnown())
    {
        layer->setInput(inputIndex, x.tensor(ctx));
    }
}

nvinfer1::IShuffleLayer* addShuffle(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& reshapeDims)
{
    nvinfer1::IShuffleLayer* shuffle = ctx->network()->addShuffle(data);
    if (reshapeDims.valuesKnown())
    {
        shuffle->setReshapeDimensions(toDims(reshapeDims));
    }
    else
    {
        shuffle->setInput(1, reshapeDims.tensor(ctx));
    }
    return shuffle;
}

nvinfer1::ISliceLayer* addSlice(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& starts,
    const ShapeTensor& sizes, const ShapeTensor& strides)
{
    nvinfer1::ISliceLayer* slice = ctx->network()->addSlice(data, toDims(starts), toDims(sizes), toDims(strides));
    setShapeInputIfDynamic(ctx, slice, 1, starts);
    setShapeInputIfDynamic(ctx, slice, 2, sizes);
    setShapeInputIfDynamic(ctx, slice, 3, strides);
    return slice;
}

nvinfer1::IFillLayer* addFill(IImporterContext* ctx, const ShapeTensor& shape, nvinfer1::FillOperation op)
{
    nvinfer1::IFillLayer* fill = ctx->network()->addFill(toDims(shape), op);
    setShapeInputIfDynamic(ctx, fill, 0, shape);
    return fill;
}

std::ostream& operator<<(std::ostream& stream, const ShapeTensor& x)
{
    stream << "(";
    for (int i = 0; i < x.size; ++i)
    {
        stream << (i ? ", " : "");
        if (x.valuesKnown())
        {
            stream << x.values[i];
        }
        else
        {
            stream << "_";
        }
    }
    return stream << ")";
}

} // namespace onnx2trt
