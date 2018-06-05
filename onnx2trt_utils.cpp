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
                       nvinfer1::DimsHW* dilations,
                       nvinfer1::DimsHW const* output_shape,
                       bool enable_padding_trick) {
  // TODO: Generalize this function to support 3D spatial data
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
  auto onnx_auto_pad = attrs.get("auto_pad", std::string("VALID"));
  if( onnx_auto_pad == "VALID" ) {
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
    int ih = input_shape.h();
    int iw = input_shape.w();
    assert(ih != 0 && iw != 0);
    int sh = strides->h();
    int sw = strides->w();
    int kh = kernel_size->h();
    int kw = kernel_size->w();
    int dh = dilations ? dilations->h() : 1;
    int dw = dilations ? dilations->w() : 1;
    kh += (kh - 1) * (dh - 1);
    kw += (kw - 1) * (dw - 1);
    int oh;
    int ow;
    if( output_shape ) {
      oh = output_shape->h();
      ow = output_shape->w();
    } else {
      oh = (ih - 1) / sh + 1;
      ow = (iw - 1) / sw + 1;
    }
    int ph = (oh - 1) * sh + kh - ih;
    int pw = (ow - 1) * sw + kw - iw;
    int ph_minor = ph >> 1; // Note: Rounds to -inf by design
    int ph_major = ph - ph_minor;
    int pw_minor = pw >> 1; // Note: Rounds to -inf by design
    int pw_major = pw - pw_minor;
    if( onnx_auto_pad == "SAME_LOWER" ) {
      beg_padding->h() = ph_major;
      beg_padding->w() = pw_major;
      end_padding->h() = ph_minor;
      end_padding->w() = pw_minor;
    } else if( onnx_auto_pad == "SAME_UPPER" ) {
      beg_padding->h() = ph_minor;
      beg_padding->w() = pw_minor;
      end_padding->h() = ph_major;
      end_padding->w() = pw_major;
    } else {
      throw std::invalid_argument("Unexpected auto_pad value: " +
                                  onnx_auto_pad);
    }
  }
  if( !enable_padding_trick ) {
    return;
  }
  // Check if asymmetric padding can be converted to symmetric padding such
  // that CUDNN/TRT will still produce the correct result. This is only
  // possible when beg_padding = end_padding + 1, and in the general case we
  // check if the output size remains unchanged under the transformation
  // (beg, end) --> (beg, beg).
  for( int dim=0; dim<2; ++dim ) {
    int beg_p = beg_padding->d[dim];
    int end_p = end_padding->d[dim];
    if( beg_p == end_p ) {
      continue;
    }
    int i = input_shape.d[dim];
    int k = kernel_size->d[dim];
    int s = strides->d[dim];
    int d = dilations ? dilations->d[dim] : 1;
    int osize_asymmetric = get_conv_output_size(i, k, s, d, beg_p + end_p);
    int osize_symmetric  = get_conv_output_size(i, k, s, d, beg_p + beg_p);
    if( osize_symmetric == osize_asymmetric ) {
      end_padding->d[d] = beg_padding->d[d];
    }
  }
}

} // namespace onnx2trt
