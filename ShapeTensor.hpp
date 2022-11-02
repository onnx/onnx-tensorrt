/*
 * SPDX-License-Identifier: Apache-2.0
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
class ShapeTensor
{
public:
    //! Create undefined ShapeTensor.
    ShapeTensor() = default;

    //! Create ShapeTensor with known rank and int64_t values.
    ShapeTensor(int32_t rank_, std::vector<int64_t>&& values_);

    //! Create ShapeTensor with known rank and float values.
    ShapeTensor(int32_t rank_, std::vector<float>&& values_);

    //! Create ShapeTensor representing value of TensorOrWeights.
    ShapeTensor(IImporterContext* ctx, TensorOrWeights& t);

    //! Construct ShapeTensor equivalent to applying IShapeLayer depth times.
    //! The depth may be in [0,3].
    explicit ShapeTensor(nvinfer1::ITensor& t, int depth = 0);

    //! True if rank is known.
    bool rankKnown() const
    {
        return mRank != kRANK_UNKNOWN;
    }

    //! Number of dimensions.  Always 0 or 1.
    int32_t rank() const
    {
        assert(rankKnown());
        return mRank;
    }

    //! True if number of elements in tensor is known.
    bool sizeKnown() const
    {
        return mSize != kSIZE_UNKNOWN;
    }

    //! Number of elements in the tensor.  Asserts that sizeKnown()==true.
    int32_t size() const
    {
        assert(sizeKnown());
        return mSize;
    }

    //! True if tensor is known to be an empty vector.
    bool isEmpty() const
    {
        // No need to check rank because if rank is 0, then mSize==1,
        // and if rank is unknown, mSize = kSIZE_UNKNOWN.
        return mSize == 0;
    }

    //! True if all element values are known.
    bool allValuesKnown() const
    {
        return mAllValuesKnown;
    }

    //! True if all element values equal the given value.
    bool isAll(int64_t value) const;

    //! True if floating-point shape tensor.
    bool isFloat() const
    {
        return mIsFloat;
    }

    using const_iterator = std::vector<int64_t>::const_iterator;

    //! Iterator pointing to beginning of sequence of element values.
    //! Requires that allValuesKnown() is true.
    const_iterator begin() const
    {
        assert(mAllValuesKnown);
        return mValues.begin();
    }

    //! Iterator pointing to end of sequence of element values.
    //! Requires that allValuesKnown() is true.
    const_iterator end() const
    {
        assert(mAllValuesKnown);
        return mValues.end();
    }

    //! True if operator[](k) is valid.
    bool valueKnown(int k) const;

    //! Return kth value.
    //! For a 0D tensor, k must be 0.
    //! Requires that valueKnown(k) is true.
    int64_t operator[](int k) const
    {
        assert(valueKnown(k));
        return mValues[k];
    }

    //! Return true if x and y always have the same value.
    friend bool operator==(const ShapeTensor& x, const ShapeTensor& y);
    friend ShapeTensor shapeOf(const ShapeTensor& t);

    //! Get TensorRT tensor representation.
    nvinfer1::ITensor& tensor(IImporterContext* ctx) const;

private:
    //! Number of IShapeLayer to apply to mTensor to get ITensor representing value of *this.
    //! -1 for undefined *this, a value in [0,2] otherwise.
    //! 0: *this represents value of the tensor (always 0D or 1D)
    //! 1: *this represents shape of mTensor (always 1D)
    //! 2: *this represents rank of mTensor (always 1D tensor of length 1)
    mutable int8_t mDepth{-1};

    //! True if all values are known.
    bool mAllValuesKnown{false};

    static constexpr int kRANK_UNKNOWN = -1;
    static constexpr int kSIZE_UNKNOWN = -1;

    //! Rank of *this.
    //! Always -1, 0 or 1.
    int8_t mRank{kRANK_UNKNOWN};

    //! Number of elements in the tensor, or -1 if unknown.
    int32_t mSize{kSIZE_UNKNOWN};

    //! Must be non-null if mAllValuesKnown.
    mutable nvinfer1::ITensor* mTensor{nullptr};

    //! Values of elements if some might be known.
    //! mValues.size() is always zero or equal to mSize.
    //! When mAllValuesKnown==true, all the values in mValues are correct
    //! and mValues.size() == mSize.
    //! When mAllValuesKnown==false, only the non-negative values in mValues
    //! are guaranteed to be correct, and only so if mValues.size() == mSize.
    std::vector<int64_t> mValues{};

    bool mIsFloat{false};
};

//! Print ShapeTensor.  Unknown values are printed as _.
std::ostream& operator<<(std::ostream& stream, const ShapeTensor& x);

//! Create 1D ShapeTensor of length n filled with value.
//! count must be 1D ShapeTensor of size 1.
ShapeTensor fillShapeVector(IImporterContext* ctx, int64_t value, const ShapeTensor& count);

//! Create 1D ShapeTensor of length 1 containing given value.
ShapeTensor shapeVector(int64_t value);

//! Create 0D ShapeTensor containing the given value.
ShapeTensor shapeScalar(int64_t value);

//! Create 1D ShapeTensor containing [0,n).
ShapeTensor iotaShapeVector(int32_t n);

//! Create ShapeTensor filled with value that has same shape as exemplar.
//! The exemplar must be 1D.
ShapeTensor similar(IImporterContext* ctx, const ShapeTensor& exemplar, int64_t value);

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

//! Elementwise f, for a partial function f defined by:
//! f(x,x) = x
//! f(1,x) = x
//! f(x,1) = x
//! Undefined otherwise or if x < 0.
ShapeTensor broadcast(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Return product of x[i] for i in [first..last), as 0D or one-element 1D tensor of given rank.
ShapeTensor product(IImporterContext* ctx, const ShapeTensor& x, int first, int last, int rank);

//! Gather where data is 1D tensor and indices can be 0D or 1D
ShapeTensor gather(IImporterContext* ctx, const ShapeTensor& data, const ShapeTensor& indices);

//! Concatenation of two 1D tensors
ShapeTensor concat(IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y);

//! Cast to int32_t shape tensor.
ShapeTensor castToInt32(IImporterContext* ctx, ShapeTensor const& x);

//! Return gather(concat(x,y),subscripts)
inline ShapeTensor interlace(
    IImporterContext* ctx, const ShapeTensor& x, const ShapeTensor& y, const ShapeTensor& subscripts)
{
    return gather(ctx, concat(ctx, x, y), subscripts);
}

//! Return shape of a tensor.
ShapeTensor shapeOf(nvinfer1::ITensor& tensor);
ShapeTensor shapeOf(const ShapeTensor& tensor);
ShapeTensor shapeOf(TensorOrWeights& t);

//! Reshape 0D tensor to 1D tensor.
ShapeTensor convertTo1D(IImporterContext* ctx, const ShapeTensor& tensor);

//! Add an ISliceLayer.
nvinfer1::ISliceLayer* addSlice(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& starts,
    const ShapeTensor& sizes, const ShapeTensor& strides);

//! Add an IShuffleLayer.
//! If the result does not need to have its parameters changed, and
//! optimizing the no-op case away is okay, use function reshape instead.
//!
//! In general the default zeroIsPlaceholder=false should be used so
//! that reshaping to empty tensors works correctly.  Calling with
//! zeroIsPlaceholder=true should happen only when replicating the
//! semantics of the ONNX Reshape operator.
nvinfer1::IShuffleLayer* addShuffle(
    IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& reshapeDims, bool zeroIsPlaceholder = false);

//! Add an IFillLayer.
nvinfer1::IFillLayer* addFill(IImporterContext* ctx, const ShapeTensor& shape, nvinfer1::FillOperation op);

//! Reshape a tensor.
//!
//! Treats any zeros in newShape as dimensions, not placeholders.
//! Implementation note: does not insert shuffle if it's a no-op.
nvinfer1::ITensor& reshape(IImporterContext* ctx, nvinfer1::ITensor& data, const ShapeTensor& newShape);

} // namespace onnx2trt
