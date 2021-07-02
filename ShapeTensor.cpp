/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

ShapeTensor::ShapeTensor(int rank_, std::vector<int64_t>&& values_)
    : mDepth(0)
    , mAllValuesKnown(true)
    , mRank(rank_)
    , mSize(values_.size())
    , mValues(std::move(values_))
{
    assert((rank_ == 0 || rank_ == 1) && "shape tensor must have rank 0 or 1");
    assert(rank_ > 0 || mValues.size() == 1);
}

ShapeTensor::ShapeTensor(TensorOrWeights& t)
    : mDepth(0)
{
    if (t.is_tensor())
    {
        *this = ShapeTensor(t.tensor());
    }
    else
    {
        const nvinfer1::Dims d = t.shape();
        assert(0 <= d.nbDims);
        assert(d.nbDims <= 1 && "shape tensor must be 0D or 1D");
        mRank = d.nbDims;
        mSize = d.nbDims == 0 ? 1 : d.d[0];
        weightsToVector(t.weights(), &mValues);
        mAllValuesKnown = true;
    }
}

static bool hasAllNonNegativeValues(const std::vector<int64_t>& values)
{
    return std::all_of(values.begin(), values.end(), [](int x) { return x >= 0; });
}

ShapeTensor::ShapeTensor(nvinfer1::ITensor& t, int depth)
    : mDepth(depth)
    , mRank(1)
    , mTensor(&t)
{
    const nvinfer1::Dims dims = t.getDimensions();

    switch (mDepth)
    {
    case 0:
        assert(t.getType() == nvinfer1::DataType::kINT32);
        mRank = dims.nbDims;
        if (mRank == 0)
        {
            mSize = 1;
        }
        else if (mRank == 1)
        {
            mSize = dims.d[0];
        }
        else
        {
            assert(mRank == -1);
        }
        break;

    case 1:
        if (dims.nbDims >= 0)
        {
            mSize = dims.nbDims;
            mValues.resize(dims.nbDims);
            std::copy_n(dims.d, dims.nbDims, mValues.begin());
            mAllValuesKnown = hasAllNonNegativeValues(mValues);
        }
        break;

    case 2:
        mSize = 1;
        if (dims.nbDims >= 0)
        {
            mValues = {dims.nbDims};
            mAllValuesKnown = hasAllNonNegativeValues(mValues);
        }
        break;

    case 3:
        // Applying IShapeLayer three times always yields a 1D vector containing 1.
        mDepth = 0;
        mSize = 1;
        mValues = {1};
        mAllValuesKnown = true;
        mTensor = nullptr;
        break;

    default:
        // Though depths greater than 3 could be handled the same as 3, they are
        // likely a sign of a problem.  Depths less than 0 make no sense.
        assert(0);
        break;
    }
}

ShapeTensor shapeVector(int64_t value)
{
    return ShapeTensor(1, std::vector<int64_t>({value}));
}

ShapeTensor shapeScalar(int64_t value)
{
    return ShapeTensor(0, std::vector<int64_t>({value}));
}

bool ShapeTensor::valueKnown(int k) const
{
    assert(0 <= k);
    assert(k < mSize);
    return allValuesKnown() || (mValues.size() == static_cast<size_t>(mSize) && mValues[k] >= 0);
}

bool ShapeTensor::isAll(int64_t x) const
{
    assert(mDepth >= 0 && "undefined tensor");
    return allValuesKnown() && std::all_of(begin(), end(), [x](int64_t y) { return x == y; });
}

nvinfer1::ITensor& ShapeTensor::tensor(IImporterContext* ctx) const
{
    assert(mDepth >= 0 && "undefined tensor");
    assert(mDepth <= 2);
    if (!mTensor || mDepth != 0)
    {
        // Need to create an ITensor representing *this.
        if (allValuesKnown())
        {
            // Create constant
            const nvinfer1::Dims dims{rank(), {size()}, {}};
            const nvinfer1::Weights w{nvinfer1::DataType::kINT32, convertINT64(mValues.data(), dims, ctx), size()};
            mTensor = ctx->network()->addConstant(dims, w)->getOutput(0);
            mDepth = 0;
        }
        else
        {
            assert(mTensor);
            for (; mDepth > 0; --mDepth)
            {
                mTensor = ctx->network()->addShape(*mTensor)->getOutput(0);
            }
        }
    }
    return *mTensor;
}

ShapeTensor iotaShapeVector(int32_t n)
{
    std::vector<int64_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    return ShapeTensor(1, std::move(values));
}

ShapeTensor similar(IImporterContext* ctx, const ShapeTensor& exemplar, int64_t value)
{
    return fillShapeVector(ctx, value, shapeOf(exemplar));
}

ShapeTensor fillShapeVector(IImporterContext* ctx, int64_t value, const ShapeTensor& count)
{
    assert(count.rank() == 1 && "implementation assumes 1D size");
    assert(count.size() == 1 && "implementation assumes 1D size of known size");
    if (count.allValuesKnown())
    {
        return ShapeTensor(1, std::vector<int64_t>(count[0], value));
    }
    else
    {
        nvinfer1::ISliceLayer* slice
            = addSlice(ctx, shapeVector(value).tensor(ctx), shapeVector(0), count, shapeVector(0));
        return ShapeTensor(*slice->getOutput(0));
    }
}

using nvinfer1::ElementWiseOperation;

//! Helper that implements an elementwise operations on two shape tensors x and y.
//! f must implement the operation on a pair of int64_t.
//! commutes should be true f is commutative.
//! rightIdentity should be the right identity value for f.
static ShapeTensor op(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y, ElementWiseOperation operation,
    bool commutative, int64_t rightIdentity, const std::function<int64_t(int64_t, int64_t)>&& f)
{
    assert(!x.rankKnown() || !y.rankKnown() || x.rank() == y.rank());
    if (x.sizeKnown() && y.sizeKnown())
    {
        assert(x.size() == 1 || y.size() == 1 || x.size() == y.size());
        if (y.isAll(rightIdentity) && y.size() <= x.size())
        {
            return x;
        }
        if (commutative && x.isAll(rightIdentity) && x.size() <= y.size())
        {
            return y;
        }
    }
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        std::vector<int64_t> values(std::max(x.size(), y.size()));
        for (size_t i = 0; i < values.size(); ++i)
        {
            // The % simulates broadcast rules.
            values[i] = f(x[i % x.size()], y[i % y.size()]);
        }
        return ShapeTensor(x.rank(), std::move(values));
    }
    return ShapeTensor(*ctx->network()->addElementWise(x.tensor(ctx), y.tensor(ctx), operation)->getOutput(0), 0);
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

ShapeTensor broadcast(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y)
{
    // max(x,y) works unless x or y is 0.
    // min(x,y,1) yields 0 if x or y is 0, and 1 otherwise.
    // So compute max(x,y)*min(x,y,1).
    return mul(ctx, max(ctx, x, y), min(ctx, x, min(ctx, y, similar(ctx, y, 1))));
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
    assert(!x.rankKnown() || x.rank() == 1);
    assert(!y.rankKnown() || y.rank() == 1);
    if (x.sizeKnown() && x.size() == 0)
    {
        return y;
    }
    if (y.sizeKnown() && y.size() == 0)
    {
        return x;
    }
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        std::vector<int64_t> values(x.size() + y.size());
        auto p = std::copy(x.begin(), x.end(), values.begin());
        std::copy(y.begin(), y.end(), p);
        return ShapeTensor(1, std::move(values));
    }

    nvinfer1::ITensor* const args[2] = {&x.tensor(ctx), &y.tensor(ctx)};
    return ShapeTensor(*ctx->network()->addConcatenation(args, 2)->getOutput(0));
}

ShapeTensor gather(IImporterContext* ctx, const ShapeTensor& data, const ShapeTensor& indices)
{
    assert(data.rank() == 1);
    if (indices.allValuesKnown()
        && std::all_of(indices.begin(), indices.end(), [&data](int i) { return data.valueKnown(i); }))
    {
        std::vector<int64_t> z(indices.size());
        std::transform(indices.begin(), indices.end(), z.begin(), [&data](int64_t i) {
            assert(0 <= i);
            assert(i < data.size());
            return data[i];
        });
        return ShapeTensor(indices.rank(), std::move(z));
    }
    return ShapeTensor(*ctx->network()->addGather(data.tensor(ctx), indices.tensor(ctx), 0)->getOutput(0));
}

ShapeTensor shapeOf(nvinfer1::ITensor& tensor)
{
    return ShapeTensor(tensor, 1);
}

ShapeTensor shapeOf(TensorOrWeights& t)
{
    if (t.is_tensor())
    {
        return shapeOf(t.tensor());
    }
    else
    {
        const nvinfer1::Dims& d = t.weights().shape;
        return ShapeTensor(1, std::vector<int64_t>(d.d, d.d + d.nbDims));
    }
}

ShapeTensor shapeOf(const ShapeTensor& t)
{
    assert(t.mDepth >= 0);
    if (t.mTensor)
    {
        return ShapeTensor(*t.mTensor, t.mDepth + 1);
    }
    else
    {
        assert(t.rankKnown());
        assert(t.sizeKnown());
        // ShapeTensor is either a scalar or vector.
        // shape of a scalar is an empty tensor.
        // shape of a vector is a one-element tensor containing the length of the vector.
        return t.rank() == 0 ? ShapeTensor(0, {}) : ShapeTensor(1, {t.size()});
    }
}

ShapeTensor convertTo1D(IImporterContext* ctx, const ShapeTensor& tensor)
{
    assert(tensor.rank() == 0);
    assert(tensor.size() == 1);
    if (tensor.valueKnown(0))
    {
        return shapeScalar(tensor[0]);
    }
    return ShapeTensor(*addShuffle(ctx, tensor.tensor(ctx), shapeVector(1))->getOutput(0));
}

//! If all values of x are known, return Dims with those values.
//! Otherwise return Dims with zeros.
static nvinfer1::Dims toDims(const ShapeTensor& x)
{
    nvinfer1::Dims d{-1, {}, {}};
    if (x.sizeKnown())
    {
        d.nbDims = x.size();
        if (x.allValuesKnown())
        {
            assert(x.size() <= nvinfer1::Dims::MAX_DIMS);
            std::copy(x.begin(), x.end(), d.d);
        }
    }
    return d;
}

//! If not all values in x are known, set layer input specifed by inputIndex
//! to tensor with value of x.
static void setShapeInputIfDynamic(IImporterContext* ctx, nvinfer1::ILayer* layer, int inputIndex, const ShapeTensor& x)
{
    if (!x.allValuesKnown())
    {
        layer->setInput(inputIndex, x.tensor(ctx));
    }
}

bool operator==(const ShapeTensor& x, const ShapeTensor& y)
{
    if (x.allValuesKnown() && y.allValuesKnown())
    {
        return x.mValues == y.mValues;
    }
    assert(x.mTensor || y.mTensor);
    return x.mTensor == y.mTensor && x.mDepth == y.mDepth;
}

nvinfer1::ITensor& reshape(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& newShape)
{
    const ShapeTensor oldShape = shapeOf(data);
    if (newShape == oldShape)
    {
        return data;
    }
    return *addShuffle(ctx, data, newShape)->getOutput(0);
}

nvinfer1::IShuffleLayer* addShuffle(
    IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& reshapeDims, bool zeroIsPlaceholder)
{
    nvinfer1::IShuffleLayer* shuffle = ctx->network()->addShuffle(data);
    if (reshapeDims.allValuesKnown())
    {
        shuffle->setReshapeDimensions(toDims(reshapeDims));
    }
    else
    {
        shuffle->setInput(1, reshapeDims.tensor(ctx));
    }
    shuffle->setZeroIsPlaceholder(zeroIsPlaceholder);
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
    for (int i = 0, e = x.size(); i < e; ++i)
    {
        stream << (i ? ", " : "");
        if (x.valueKnown(i))
        {
            stream << x[i];
        }
        else
        {
            stream << "_";
        }
    }
    if (x.size() == 1 && x.rank() == 1)
    {
        // Use Python convention to distinguish 1-element vector from a scalar.
        stream << ",";
    }
    return stream << ")";
}

} // namespace onnx2trt
