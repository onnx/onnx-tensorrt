/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bfloat16.hpp"
#include <bit>

namespace onnx2trt
{

BFloat16::operator float() const
{
    return std::bit_cast<float>(static_cast<uint32_t>(mRep) << 16);
}

BFloat16::BFloat16(float x)
{
    uint32_t bits = std::bit_cast<uint32_t>(x);

    // FP32 format: 1 sign bit, 8 bit exponent, 23 bit mantissa
    // BF16 format: 1 sign bit, 8 bit exponent, 7 bit mantissa

    // Mask for exponent
    constexpr uint32_t exponent = 0xFFU << 23;

    // Check if exponent is all 1s (NaN or infinite)
    if ((bits & exponent) != exponent)
    {
        // x is finite - round to even
        bits += 0x7FFFU + (bits >> 16 & 1);
    }

    mRep = static_cast<uint16_t>(bits >> 16);
}

} // namespace onnx2trt
