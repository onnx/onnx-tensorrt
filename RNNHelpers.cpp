/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RNNHelpers.hpp"
#include "LoopHelpers.hpp"
#include "onnx2trt_utils.hpp"
#include <array>

namespace onnx2trt
{

nvinfer1::ITensor* addRNNInput(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, std::vector<TensorOrWeights>& inputs, const std::string& direction)
{
    // In the forward/reverse cases, we only use a single iterator. In the bidirectional case, a forward and reverse
    // iterator must be concatenated.
    // Input dimensions: [1, B, E]
    nvinfer1::ITensor* iterationInput{nullptr};
    nvinfer1::ITensor* input = &convertToTensor(inputs.at(0), ctx);

    const int sequenceLenIndex = 4;
    bool isRagged = inputs.size() > sequenceLenIndex && inputs.at(sequenceLenIndex);

    if (direction == "forward")
    {
        iterationInput = unsqueezeTensor(ctx, node, *loop->addIterator(*input)->getOutput(0), std::vector<int>{0});

        if (isRagged)
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(sequenceLenIndex), ctx);
            auto maxLen = getAxisLength(ctx, input, 0);
            iterationInput = clearMissingSequenceElements(ctx, node, loop, seqLens, iterationInput, maxLen);
        }
    }
    else if (direction == "reverse")
    {
        nvinfer1::IIteratorLayer* reverseIterator = loop->addIterator(*input);
        reverseIterator->setReverse(true);
        iterationInput = unsqueezeTensor(ctx, node, *reverseIterator->getOutput(0), std::vector<int>{0});
        if (isRagged)
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(sequenceLenIndex), ctx);
            auto maxLen = getAxisLength(ctx, input, 0);
            iterationInput = clearMissingSequenceElements(ctx, node, loop, seqLens, iterationInput, maxLen, true);
        }
    }
    else if (direction == "bidirectional")
    {
        nvinfer1::IIteratorLayer* forward = loop->addIterator(*input);
        nvinfer1::IIteratorLayer* reverse = loop->addIterator(*input);
        reverse->setReverse(true);

        auto forwardInput = unsqueezeTensor(ctx, node, *forward->getOutput(0), std::vector<int>{0});
        auto reverseInput = unsqueezeTensor(ctx, node, *reverse->getOutput(0), std::vector<int>{0});
        if (isRagged)
        {
            nvinfer1::ITensor* seqLens = &convertToTensor(inputs.at(sequenceLenIndex), ctx);
            auto counter = addLoopCounter(ctx, loop);
            auto maxLen = getAxisLength(ctx, input, 0);
            forwardInput = clearMissingSequenceElements(ctx, node, loop, seqLens, forwardInput, maxLen, false, counter);
            reverseInput = clearMissingSequenceElements(ctx, node, loop, seqLens, reverseInput, maxLen, true, counter);
        }

        // Stack on the 0th axis to create a (numDirections, B, E) tensor.
        std::array<nvinfer1::ITensor*, 2> tensors{{forwardInput, reverseInput}};
        nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
        concat->setAxis(0);
        iterationInput = concat->getOutput(0);
    }
    if (iterationInput)
    {
        LOG_VERBOSE("Input shape: " << iterationInput->getDimensions());
    }
    return iterationInput;
}

nvinfer1::ITensor* clearMissingSequenceElements(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop,
    nvinfer1::ITensor* seqLens, nvinfer1::ITensor* toMask, nvinfer1::ITensor* maxLen, bool reverse,
    nvinfer1::ITensor* counter)
{
    nvinfer1::ITensor* zero
        = addConstantScalar(ctx, 0.f, ::ONNX_NAMESPACE::TensorProto::FLOAT, nvinfer1::Dims3(1, 1, 1))->getOutput(0);
    nvinfer1::ITensor* seqMask = getRaggedMask(ctx, node, loop, seqLens, maxLen, reverse, counter);
    return ctx->network()->addSelect(*seqMask, *toMask, *zero)->getOutput(0);
}

nvinfer1::ITensor* maskRNNHidden(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens,
    nvinfer1::ITensor* prevH, nvinfer1::ITensor* Ht, nvinfer1::ITensor* maxLen, bool reverse,
    nvinfer1::ITensor* counter)
{
    // maxLen must be provided if reverse is true
    // Forwards previous hidden state if invalid
    nvinfer1::ITensor* valid = getRaggedMask(ctx, node, loop, seqLens, maxLen, reverse, counter);
    return ctx->network()->addSelect(*valid, *Ht, *prevH)->getOutput(0);
}

nvinfer1::ITensor* maskBidirRNNHidden(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens,
    nvinfer1::ITensor* maxLen, nvinfer1::ITensor* Ht1, nvinfer1::ITensor* Ht, nvinfer1::ITensor* singlePassShape)
{
    // Splits hidden state into forward and backward states, masks each accordingly, then concatenates

    nvinfer1::ITensor* forwardStart
        = addConstant(ctx, std::vector<int32_t>{0, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
              ->getOutput(0);
    nvinfer1::ITensor* reverseStart
        = addConstant(ctx, std::vector<int32_t>{1, 0, 0}, ::ONNX_NAMESPACE::TensorProto::INT32, nvinfer1::Dims{1, 3})
              ->getOutput(0);

    nvinfer1::ISliceLayer* HtForwardLayer
        = ctx->network()->addSlice(*Ht, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
    HtForwardLayer->setInput(1, *forwardStart);
    HtForwardLayer->setInput(2, *singlePassShape);

    nvinfer1::ISliceLayer* HtBackwardLayer
        = ctx->network()->addSlice(*Ht, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
    HtBackwardLayer->setInput(1, *reverseStart);
    HtBackwardLayer->setInput(2, *singlePassShape);

    nvinfer1::ISliceLayer* Ht1ForwardLayer
        = ctx->network()->addSlice(*Ht1, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
    Ht1ForwardLayer->setInput(1, *forwardStart);
    Ht1ForwardLayer->setInput(2, *singlePassShape);

    nvinfer1::ISliceLayer* Ht1BackwardLayer
        = ctx->network()->addSlice(*Ht1, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 1, 1});
    Ht1BackwardLayer->setInput(1, *reverseStart);
    Ht1BackwardLayer->setInput(2, *singlePassShape);

    auto forwardHt = HtForwardLayer->getOutput(0);
    auto backwardHt = HtBackwardLayer->getOutput(0);
    auto forwardHt1 = Ht1ForwardLayer->getOutput(0);
    auto backwardHt1 = Ht1BackwardLayer->getOutput(0);

    auto counter = addLoopCounter(ctx, loop, 0);
    forwardHt = maskRNNHidden(ctx, node, loop, seqLens, forwardHt1, forwardHt, maxLen, false, counter);
    backwardHt = maskRNNHidden(ctx, node, loop, seqLens, backwardHt1, backwardHt, maxLen, true, counter);
    std::array<nvinfer1::ITensor*, 2> tensors{{forwardHt, backwardHt}};
    nvinfer1::IConcatenationLayer* concat = ctx->network()->addConcatenation(tensors.data(), 2);
    concat->setAxis(0);
    return concat->getOutput(0);
}

nvinfer1::ITensor* getRaggedMask(IImporterContext* ctx, const ::ONNX_NAMESPACE::NodeProto& node, nvinfer1::ILoop* loop, nvinfer1::ITensor* seqLens,
    nvinfer1::ITensor* maxLen, bool reverse, nvinfer1::ITensor* counter)
{
    // Returns a bool tensor which is true where the elements are valid (within the sequence) and false when outside the
    // sequence.
    // maxLen must be provided if reverse is true
    assert(!reverse || maxLen);

    if (!counter)
    {
        counter = addLoopCounter(ctx, loop, 0);
    }

    // Create Mask
    nvinfer1::ITensor* seqMask;
    if (reverse)
    {
        counter
            = ctx->network()
                  ->addElementWise(*unsqueezeTensor(ctx, node, *maxLen, {0}), *counter, nvinfer1::ElementWiseOperation::kSUB)
                  ->getOutput(0);
        seqMask
            = ctx->network()->addElementWise(*seqLens, *counter, nvinfer1::ElementWiseOperation::kLESS)->getOutput(0);
        seqMask = ctx->network()->addUnary(*seqMask, nvinfer1::UnaryOperation::kNOT)->getOutput(0);
    }
    else
    {
        seqMask
            = ctx->network()->addElementWise(*counter, *seqLens, nvinfer1::ElementWiseOperation::kLESS)->getOutput(0);
    }
    return unsqueezeTensor(ctx, node, *seqMask, std::vector<int>{0, 2});
}

} // namespace onnx2trt
