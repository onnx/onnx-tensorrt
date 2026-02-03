/*
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include "AttentionHelpers.hpp"
#include "ImporterContext.hpp"
#include "NvInfer.h"
#include "ShapeTensor.hpp"
#include "errorHelpers.hpp"
#include "importerUtils.hpp"
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace
{
//!
//! \brief Return true if `divident` is divisible by `divisor`.
//!
bool isDivisible(int64_t const divident, int64_t const divisor)
{
    return (divisor != 0) && ((divident % divisor) == 0);
}
} // namespace

namespace onnx2trt
{

//!
//! \brief Reshape and return the Q, K, or V tensor from the input tensor.
//!
//! \param qkvInput The input tensor. This can either be a 4D tensor (batchSize, numHeads, sequenceLength, headSize) or
//!                 a 3D tensor (batchSize, sequenceLength, hiddenSize=numHeads*headSize). If it is a 3D tensor,
//!                 permute and reshape to the 4D shape before returning. Otherwise, return the input tensor.
//! \param attrs The ONNX node attributes.
//! \param ctx The importer context.
//! \param isQ True if the input tensor is the Q tensor, false if it is the K or V tensor.
//! \return nvinfer1::ITensor& The Q, K, or V tensor.
//!
nvinfer1::ITensor& reshapeQKVTensor(
    TensorOrWeights& qkvInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const isQ, bool const needsReshape)
{
    if (needsReshape)
    {
        // qkvInput is a 3D tensor (batchSize, sequenceLength, hiddenSize=numHeads * headSize).
        // Get relevant dimensions.
        int64_t const numHeadsValue
            = isQ ? attrs.get<int64_t>("q_num_heads", 0) : attrs.get<int64_t>("kv_num_heads", 0);
        ONNXTRT_CHECK(numHeadsValue != 0,
            "q_num_heads and kv_num_heads attributes are not specified, which are required for 3D Q/K/V tensors",
            ErrorCode::kINVALID_NODE);
        ShapeTensor numHeads = shapeVector(numHeadsValue);

        ShapeTensor hiddenSize = gather(ctx, shapeOf(qkvInput), shapeVector(2));
        if (hiddenSize.allValuesKnown())
        {
            // Perform static check for divisibility.
            ONNXTRT_CHECK(isDivisible(hiddenSize[0], numHeads[0]),
                "hidden_size must be divisible by num_heads. Received hidden_size=" << hiddenSize[0]
                                                                                    << " and num_heads=" << numHeads,
                ErrorCode::kINVALID_NODE);
        }

        ShapeTensor headSize = floorDiv(ctx, hiddenSize, numHeads);

        // == Transform (batchSize, sequenceLength, hiddenSize) -> (batchSize, numHeads, sequenceLength, headSize) by ==
        // 1. Reshape to (batchSize, sequenceLength, numHeads, headSize).
        // Use (0, 0, numHeads, headSize) as a shorthand to propagate `batchSize` and `sequenceLength` from the input
        // tensor without instantiating them. Set `zeroIsPlaceholder` to enable this shorthand.
        ShapeTensor newShape = concat(ctx, fillShapeVector(ctx, 0, shapeVector(2)), concat(ctx, numHeads, headSize));
        nvinfer1::IShuffleLayer* shuffle
            = addShuffle(ctx, convertToTensor(qkvInput, ctx), newShape, /*zeroIsPlaceholder*/ true);

        // 2. Permute to (batchSize, numHeads, sequenceLength, headSize)
        shuffle->setSecondTranspose({0, 2, 1, 3});

        return *N_CHECK(shuffle->getOutput(0));
    }
    else
    {
        return convertToTensor(qkvInput, ctx);
    }
}

//!
//! \brief Scale the Q or K tensor by `sqrt(scale)`.
//!
//! `scale` is either provided as an attribute or set as the default value of `1/sqrt(headSize)`. `scale` is defined as
//! `QK^T -> QK^T * scale`, but we apply `Q -> Q * sqrt(scale)` and `K -> K * sqrt(scale)` for numerical stability.
//!
//! \param qkTensor The Q or K tensor to scale.
//! \param attrs The ONNX node attributes.
//! \param ctx The importer context.
//! \return nvinfer1::ITensor& The scaled Q or K tensor.
//!
nvinfer1::ITensor& scaleQKTensor(nvinfer1::ITensor& qkTensor, OnnxAttrs const& attrs, ImporterContext* ctx)
{
    nvinfer1::ITensor* sqrtScale = nullptr;

    if (attrs.count("scale"))
    {
        // Obtain the sqrt of scale as a constant (output of a constant layer).
        nvinfer1::IConstantLayer* constant
            = addConstantScalar(ctx, std::sqrt(attrs.get<float>("scale")), ::ONNX_NAMESPACE::TensorProto::FLOAT, 4);
        sqrtScale = castHelper(ctx, N_CHECK(constant)->getOutput(0), qkTensor.getType());
    }
    else
    {
        ShapeTensor headSize = gather(ctx, shapeOf(qkTensor), shapeScalar(3));
        nvinfer1::ITensor* headSizeF = castHelper(ctx, &headSize.tensor(ctx), qkTensor.getType());

        // By default, scale := 1/sqrt(headSize)
        nvinfer1::ITensor* sqrtHeadSize = getUnaryResult(ctx, *headSizeF, nvinfer1::UnaryOperation::kSQRT);
        nvinfer1::ITensor* scale = getUnaryResult(ctx, *sqrtHeadSize, nvinfer1::UnaryOperation::kRECIP);

        sqrtScale = getUnaryResult(ctx, *scale, nvinfer1::UnaryOperation::kSQRT);
        sqrtScale = unsqueezeTensor(ctx, *sqrtScale, {0, 1, 2, 3});
    }

    // Scale Q or K tensor by `sqrt(scale)`.
    return *getElementWiseResult(ctx, qkTensor, *sqrtScale, nvinfer1::ElementWiseOperation::kPROD);
}

nvinfer1::ITensor& convertToQTensor(
    TensorOrWeights& qInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape)
{
    return scaleQKTensor(reshapeQKVTensor(qInput, attrs, ctx, true /*isQ*/, needsReshape), attrs, ctx);
}

nvinfer1::ITensor& convertToKTensor(
    TensorOrWeights& kInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape)
{
    return scaleQKTensor(reshapeQKVTensor(kInput, attrs, ctx, false /*isQ*/, needsReshape), attrs, ctx);
}

nvinfer1::ITensor& convertToVTensor(
    TensorOrWeights& vInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape)
{
    return reshapeQKVTensor(vInput, attrs, ctx, false /*isQ*/, needsReshape);
}

nvinfer1::ITensor& convertToMaskTensor(TensorOrWeights& maskInput, ImporterContext* ctx)
{
    ONNXTRT_CHECK(maskInput.shape().nbDims <= 4,
        "Attention masks should have rank leq 4. Got mask with rank " << maskInput.shape().nbDims << ".",
        ErrorCode::kINVALID_NODE);

    if (maskInput.shape().nbDims == 4)
    {
        // Mask has rank 4. Directly return the mask tensor.
        return convertToTensor(maskInput, ctx);
    }
    else
    {
        // Mask has rank less than 4. Reshape to rank 4 by prepending dimensions.
        int32_t const numDimsToPrepend = 4 - maskInput.shape().nbDims;
        std::vector<int32_t> unsqueezeAxes(numDimsToPrepend);
        std::iota(unsqueezeAxes.begin(), unsqueezeAxes.end(), 0);

        return *unsqueezeTensor(ctx, convertToTensor(maskInput, ctx), unsqueezeAxes);
    }
}

nvinfer1::AttentionNormalizationOp parseNormalizationOp(OnnxAttrs const& attrs)
{
    std::string normalizationOp
        = attrs.get<std::string>("TRT_normalization_op", "softmax"); // Normalization op defaults to softmax.
    if (normalizationOp == "softmax")
    {
        return nvinfer1::AttentionNormalizationOp::kSOFTMAX;
    }
    else if (normalizationOp == "none")
    {
        return nvinfer1::AttentionNormalizationOp::kNONE;
    }
    else
    {
        ONNXTRT_CHECK(false, "Unsupported normalization op: " << normalizationOp, ErrorCode::kINVALID_NODE);
    }
}

nvinfer1::ITensor& reshapeOutputTensor(nvinfer1::ITensor& tensor, ImporterContext* ctx, bool const needsReshape)
{
    if (!needsReshape)
    {
        return tensor;
    }
    else
    {
        ShapeTensor numHeads = gather(ctx, shapeOf(tensor), shapeVector(1));
        ShapeTensor headSize = gather(ctx, shapeOf(tensor), shapeVector(3));
        ShapeTensor hiddenSize = mul(ctx, numHeads, headSize);

        // == Transform (batchSize, numHeads, sequenceLength, headSize) -> (batchSize, sequenceLength, hiddenSize) by ==
        // 1. Transpose the middle two dimensions: (batchSize, numHeads, sequenceLength, headSize) -> (batchSize,
        // sequenceLength, numHeads, headSize)
        // 2. Reshape to (batchSize, sequenceLength, hiddenSize).
        // Use (0, 0, hiddenSize) as a shorthand to propagate `batchSize` and `sequenceLength` from the input
        // tensor without instantiating them. Set `zeroIsPlaceholder` to enable this shorthand.
        ShapeTensor newShape = concat(ctx, fillShapeVector(ctx, 0, shapeVector(2)), hiddenSize);
        nvinfer1::IShuffleLayer* shuffle = addShuffle(ctx, tensor, newShape, /*zeroIsPlaceholder*/ true);
        shuffle->setFirstTranspose({0, 2, 1, 3});

        return *N_CHECK(shuffle->getOutput(0));
    }
}
} // namespace onnx2trt
