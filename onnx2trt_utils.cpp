/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "onnx2trt_utils.hpp"

namespace onnx2trt {

void get_kernel_params(::ONNX_NAMESPACE::NodeProto const& onnx_node,
                       nvinfer1::DimsHW const& input_shape,
                       nvinfer1::DimsHW* kernel_size,
                       nvinfer1::DimsHW* strides,
                       nvinfer1::DimsHW* beg_padding,
                       nvinfer1::DimsHW* end_padding,
                       nvinfer1::PaddingMode& paddingMode,
                       nvinfer1::DimsHW* dilations,
                       nvinfer1::DimsHW const* output_shape) {
  OnnxAttrs attrs(onnx_node);
  if( attrs.count("kernel_shape") ) {
    auto const* onnx_kernel_size = attrs.at("kernel_shape");
    int ndim = onnx_kernel_size->ints().size();
    kernel_size->h() = onnx_kernel_size->ints(0);
    kernel_size->w() = ndim > 1 ? onnx_kernel_size->ints(1) : 1;
  }
  if( attrs.count("strides") ) {
    auto const* onnx_strides = attrs.at("strides");
    int ndim = onnx_strides->ints().size();
    strides->h() = onnx_strides->ints(0);
    strides->w() = ndim > 1 ? onnx_strides->ints(1) : 1;
  }
  if( dilations && attrs.count("dilations") ) {
    auto const* onnx_dilations = attrs.at("dilations");
    int ndim = onnx_dilations->ints().size();
    dilations->h() = onnx_dilations->ints(0);
    dilations->w() = ndim > 1 ? onnx_dilations->ints(1) : 1;
  }
  paddingMode = nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
  auto onnx_auto_pad = attrs.get("auto_pad", std::string("NOTSET"));
  if( onnx_auto_pad == "VALID" || onnx_auto_pad == "NOTSET" ) {
    if( attrs.count("pads") ) {
      auto onnx_padding = attrs.get<std::vector<int>>("pads");
      int ndim = onnx_padding.size() / 2;
      int i = 0;
      beg_padding->h() = onnx_padding.at(i++);
      beg_padding->w() = ndim > 1 ? onnx_padding.at(i++) : 0;
      end_padding->h() = onnx_padding.at(i++);
      end_padding->w() = ndim > 1 ? onnx_padding.at(i++) : 0;
    }
  } else { // SAME_* padding
    assert(!attrs.count("pads"));
    // Note: ONNX is always NCHW ordering
    if( onnx_auto_pad == "SAME_LOWER" ) {
    paddingMode = nvinfer1::PaddingMode::kSAME_LOWER;
    } else if( onnx_auto_pad == "SAME_UPPER" ) {
    paddingMode = nvinfer1::PaddingMode::kSAME_UPPER;
    } else {
      throw std::invalid_argument("Unexpected auto_pad value: " +
                                  onnx_auto_pad);
    }
  }
}

} // namespace onnx2trt
