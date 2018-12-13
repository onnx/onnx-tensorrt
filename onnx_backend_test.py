 # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 # DEALINGS IN THE SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import onnx_tensorrt.backend as trt

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(trt, __name__)

# Ops that are not currently supported
backend_test.exclude(r'[a-z,_]*_cast_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_hardmax_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_slice_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_PReLU_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_tile_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_repeat_[a-z,_]*') # 'Tile' op
backend_test.exclude(r'[a-z,_]*_cos_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_sin_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_tan_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_acos_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_asin_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_atan_[a-z,_]*')
# No 3D convolutions or pooling
backend_test.exclude(r'[a-z,_]*_Conv3d_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_AvgPool3d_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_MaxPool3d_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_BatchNorm3d_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_averagepool_3d_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_maxpool_3d_[a-z,_]*')
# No boolean operations
backend_test.exclude(r'[a-z,_]*_and[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_not[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_or[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_xor[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_greater_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_less_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_equal_[a-z,_]*')
# No all-constant models
backend_test.exclude(r'test_constant_cuda')
# No arbitrary flatten
backend_test.exclude(r'test_flatten_axis0_cuda')
backend_test.exclude(r'test_flatten_axis2_cuda')
backend_test.exclude(r'test_flatten_axis3_cuda')
# No axis == batch dim
backend_test.exclude(r'test_logsoftmax_axis_0_cuda')
backend_test.exclude(r'test_softmax_axis_0_cuda')
# No fancy padding
backend_test.exclude(r'test_constant_pad_cuda')
backend_test.exclude(r'test_reflect_pad_cuda')
backend_test.exclude(r'test_edge_pad_cuda')
backend_test.exclude(r'[a-z,_]*_ReplicationPad[a-z,0-9,_]*')
# No slice of batch dim
backend_test.exclude(r'test_operator_chunk_cuda')
# No RNNs yet
backend_test.exclude(r'[a-z,_]*_rnn_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_lstm_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_gru_[a-z,_]*')
# No reduce over batch dim
backend_test.exclude(r'test_reduce_[a-z,0-9,_]*_default_axes_[a-z,_]*')

if 'TRAVIS' in os.environ:
    backend_test.exclude('(test_vgg19|test_vgg16)')
# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
