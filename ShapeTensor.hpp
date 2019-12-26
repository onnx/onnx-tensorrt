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

#pragma once

#include <NvInfer.h>
#include <cassert>
#include <iosfwd>
#include <vector>

namespace onnx2trt
{

class IImporterContext;
class TensorOrWeights;

//! Represents a 0D or 1D tensor of int64_t.
//! Unlike TensorRT, ShapeTensor allows empty tensors.
class ShapeTensor
{
public:
    //! Create undefined ShapeTensor.
    ShapeTensor() = default;

    //! Create ShapeTensor with known rank and size, but unknown values.
    //! If rank_ is 0, the size_t must be 1.
    ShapeTensor(int rank_, int32_t size_);

    //! Create ShapeTensor with known rank and values.
    ShapeTensor(int rank_, std::vector<int64_t>&& values_);

    //! Create ShapeTensor representing value of TensorOrWeights.
    ShapeTensor(TensorOrWeights& t);

    //! Number of dimensions (0 or 1), or -1 if undefined ShapeTensor.
    int8_t rank{-1};

    //! Number of values in the shape tensor, or -1 if undefined ShapeTensor.
    int32_t size{-1};

    //! Values of shape tensor if they are known, otherwise empty.
    std::vector<int64_t> values;

    //! True if values of the shape tensor are known.
    bool valuesKnown() const
    {
        return values.size() == static_cast<size_t>(size);
    }

    //! True if values of the shape tensor are known to be equal to given value.
    bool isAll(int64_t value) const;

    //! Get TensorRT tensor representation.
    nvinfer1::ITensor& tensor(IImporterContext* ctx) const;

    //! Set TensorRT tensor representation to layer->getOutput(0).
    //! Asserts that dimensions of the tensor agree with current rank and size.
    //! This is a low-level routine for use by min, max, mul, sub, etc.
    void assign(const nvinfer1::ILayer* layer);

private:
    //! Cached TensorRT representation, or null if not yet created.
    mutable nvinfer1::ITensor* mTensor{nullptr};
};

std::ostream& operator<<(std::ostream& stream, const ShapeTensor& x);

//! Create 1D ShapeTensor of length n filled with value.
ShapeTensor fillShapeVector(int32_t n, int64_t value);

//! Create 1D ShapeTensor of length 1 containing given value.
inline ShapeTensor shapeVector(int64_t value)
{
    return fillShapeVector(1, value);
}

//! Create 1D ShapeTensor with [0,n)
ShapeTensor iotaShapeVector(int32_t n);

//! Create ShapeTensor filled with value that has same shape as exemplar.
ShapeTensor similar(const ShapeTensor& exemplar, int64_t value);

//! Elementwise addition
ShapeTensor add(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Elementwise subtraction
ShapeTensor sub(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Elementwise multiplication
ShapeTensor mul(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Elementwise min
ShapeTensor min(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Elementwise max
ShapeTensor max(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Elementwise floor division
ShapeTensor floorDiv(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Return product of x[i] for i in [first..last), as 0D or one-element 1D tensor of given rank.
ShapeTensor product(IImporterContext* ctx, const ShapeTensor& x, int first, int last, int rank);

//! Gather where x is 1D tensor and y can be 0D or 1D
ShapeTensor gather(IImporterContext* ctx, const ShapeTensor& data, const ShapeTensor& indices);

//! Concatenation of two 1D tensors
ShapeTensor concat(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Return gather(concat(x,y),subscripts)
inline ShapeTensor interlace(
    IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y, const ShapeTensor& subscripts)
{
    return gather(ctx, concat(ctx, x, y), subscripts);
}

//! Return shape of a tensor.
ShapeTensor shapeOf(IImporterContext* ctx, nvinfer1::ITensor& tensor);

//! Return shape of the value represented by a TensorOrWeights.
ShapeTensor shapeOf(IImporterContext* ctx, TensorOrWeights& t);

//! Reshape 0D tensor to 1D tensor.
ShapeTensor convertTo1D(IImporterContext* ctx, const ShapeTensor& tensor);

//! Add an ISliceLayer.
nvinfer1::ISliceLayer* addSlice(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& starts,
    const ShapeTensor& sizes, const ShapeTensor& strides);

//! Add an IShuffleLayer.
nvinfer1::IShuffleLayer* addShuffle(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& reshapeDims);

//! Add an IFillLayer.
nvinfer1::IFillLayer* addFill(IImporterContext* ctx, const ShapeTensor& shape, nvinfer1::FillOperation op);

} // namespace onnx2trt
