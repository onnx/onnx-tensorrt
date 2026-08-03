/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Helper functions used for importing the ONNX Attention operator.
 *
 */

#pragma once

#include "ImporterContext.hpp"
#include "OnnxAttrs.hpp"

namespace onnx2trt
{

//!
//! \brief Convert the input tensor to the Q (query) tensor accepted by TensorRT.
//!
//! This is a wrapper over \p convertToTensor with the following additional transformations:
//! 1) If the input is a 3D tensor with shape (batchSize, sequenceLength, hiddenSize=numHeads*headSize), permute and
//!    reshape to the 4D tensor (batchSize, numHeads, sequenceLength, headSize) expected by TensorRT.
//! 2) Obtain `scale` from the attribute if provided, otherwise use `1/sqrt(headSize)` as the default value. While
//!    `scale` is defined on the QK^T product, apply `sqrt(scale)` on the Q tensor for numerical stability.
//!
//! \param qInput The input tensor to convert.
//! \param attrs The attributes of the Attention node.
//! \param ctx The importer context.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted Q tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToQTensor(
    TensorOrWeights& qInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the Q tensor accepted by TensorRT with an explicit head count.
//!
//! This overload is used by operators that use a different attribute name than ONNX Attention for head count, while
//! keeping the same scale handling as \p convertToQTensor.
//!
//! \param qInput The input tensor to convert.
//! \param attrs The attributes of the source node.
//! \param ctx The importer context.
//! \param numHeads The query head count.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted Q tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToQTensor(TensorOrWeights& qInput, OnnxAttrs const& attrs, ImporterContext* ctx,
    int64_t numHeads, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the K (key) tensor accepted by TensorRT.
//!
//! This is a wrapper over \p convertToTensor with the following additional transformations:
//! 1) If the input is a 3D tensor with shape (batchSize, sequenceLength, hiddenSize=numHeads*headSize), permute and
//!    reshape to the 4D tensor (batchSize, numHeads, sequenceLength, headSize) expected by TensorRT.
//! 2) Obtain `scale` from the attribute if provided, otherwise use `1/sqrt(headSize)` as the default value. While
//!    `scale` is defined on the QK^T product, apply `sqrt(scale)` on the K tensor for numerical stability.
//!
//! \param kInput The input tensor to convert.
//! \param attrs The attributes of the Attention node.
//! \param ctx The importer context.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted K tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToKTensor(
    TensorOrWeights& kInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the K tensor accepted by TensorRT with an explicit head count.
//!
//! This overload is used by operators that use a different attribute name than ONNX Attention for head count, while
//! keeping the same scale handling as \p convertToKTensor.
//!
//! \param kInput The input tensor to convert.
//! \param attrs The attributes of the source node.
//! \param ctx The importer context.
//! \param numHeads The key head count.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted K tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToKTensor(TensorOrWeights& kInput, OnnxAttrs const& attrs, ImporterContext* ctx,
    int64_t numHeads, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the V (value) tensor accepted by TensorRT.
//!
//! This is a wrapper over \p convertToTensor with the following additional transformation:
//! 1) If the input is a 3D tensor with shape (batchSize, sequenceLength, hiddenSize=numHeads*headSize), permute and
//!    reshape to the 4D tensor (batchSize, numHeads, sequenceLength, headSize) expected by TensorRT.
//!
//! \param vInput The input tensor to convert.
//! \param attrs The attributes of the Attention node.
//! \param ctx The importer context.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted V tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToVTensor(
    TensorOrWeights& vInput, OnnxAttrs const& attrs, ImporterContext* ctx, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the V tensor accepted by TensorRT with an explicit head count.
//!
//! This overload is used by operators that use a different attribute name than ONNX Attention for head count.
//!
//! \param vInput The input tensor to convert.
//! \param ctx The importer context.
//! \param numHeads The value head count.
//! \param needsReshape True if the input tensor needs to be reshaped to 4D, false otherwise.
//! \return nvinfer1::ITensor& The converted V tensor with shape (batchSize, numHeads, sequenceLength, headSize).
//!
nvinfer1::ITensor& convertToVTensor(
    TensorOrWeights& vInput, ImporterContext* ctx, int64_t numHeads, bool const needsReshape = false);

//!
//! \brief Convert the input tensor to the mask tensor accepted by TensorRT.
//!
//! \precondition: The input tensor shape is ONNX-broadcastable to (batchSize, qNumHeads, QSequenceLength,
//! KVSequenceLength), where ONNX-broadcastable is defined as satisfying any one of the following:
//! 1) The input tensor has exactly the same shape as the target shape.
//! 2) The input tensor has the same rank (number of dimensions) as the target shape, and each dimension is either the
//!    same as the target shape or 1.
//! 3) The input tensor has less rank than the target shape, but it can have its shape
//!    prepended with dimensions of length 1 to satisfy 2).
//!
//! \param maskInput The input tensor to convert.
//! \param ctx The importer context.
//! \return nvinfer1::ITensor& The converted mask tensor that is TensorRT-broadcastable to (batchSize, qNumHeads,
//!         qSequenceLength, kvSequenceLength), where TensorRT-broadcastable is defined as satisfying properties 1) or
//!         2) above, but not 3).
//!
nvinfer1::ITensor& convertToMaskTensor(TensorOrWeights& maskInput, ImporterContext* ctx);

//!
//! \brief Parse the normalization op from the attributes.
//!
//! While ONNX does not support specifying normalization op (always softmax), users could use the custom attribute
//! \p TRT_normalization_op to set it for TensorRT.
//!
//! \param attrs The attributes of the Attention node.
//! \return nvinfer1::AttentionNormalizationOp The parsed normalization op.
//!
nvinfer1::AttentionNormalizationOp parseNormalizationOp(OnnxAttrs const& attrs);

//!
//! \brief Parse the causal mask kind from the \p causal_kind string attribute.
//!
//! Supported values are "none" (kNONE), "upper_left" (kUPPER_LEFT), and "lower_right" (kLOWER_RIGHT).
//! Defaults to "none" when the attribute is absent.
//!
//! \param attrs The attributes of the TRT_Attention node.
//! \return nvinfer1::CausalMaskKind The parsed causal mask kind.
//!
nvinfer1::CausalMaskKind parseCausalKind(OnnxAttrs const& attrs);

//!
//! \brief Parse an AttentionIOForm from a string attribute.
//!
//! Supported values are "padded_bhnd" (kPADDED_BHND) and "packed_nhd" (kPACKED_NHD).
//!
//! \param attrs The attributes of the node.
//! \param attrName The name of the attribute to parse.
//! \return nvinfer1::AttentionIOForm The parsed IO form.
//!
nvinfer1::AttentionIOForm parseIOForm(OnnxAttrs const& attrs, std::string const& attrName);

//!
//! \brief Reshape the output tensor to the original input rank.
//!
//! \param tensor The output tensor to reshape.
//! \param ctx The importer context.
//! \param needsReshape True if the output tensor needs to be reshaped back to 3d, false otherwise.
//! \return nvinfer1::ITensor& The reshaped output tensor.
nvinfer1::ITensor& reshapeOutputTensor(
    nvinfer1::ITensor& tensor, ImporterContext* ctx, bool const needsReshape = false);
} // namespace onnx2trt
