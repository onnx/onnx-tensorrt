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

# Include all of the nodes that we support. 
# Onnx native node tests
backend_test.include(r'.*test_abs.*')
backend_test.include(r'.*test_acos.*')
backend_test.include(r'.*test_acosh.*')
backend_test.include(r'.*test_add.*')
backend_test.include(r'.*test_argmax.*')
backend_test.include(r'.*test_argmin.*')
backend_test.include(r'.*test_asin.*')
backend_test.include(r'.*test_asinh.*')
backend_test.include(r'.*test_atan.*')
backend_test.include(r'.*test_atanh.*')
backend_test.include(r'.*test_averagepool.*')
backend_test.include(r'.*test_AvgPool.*')
backend_test.include(r'.*test_BatchNorm.*eval.*')
backend_test.include(r'.*test_ceil.*')
backend_test.include(r'.*test_clip.*')
backend_test.include(r'.*test_concat.*')
backend_test.include(r'.*test_constant.*') # May segfault
backend_test.include(r'.*test_Conv[1-3]d*')
backend_test.include(r'.*test_cos.*')
backend_test.include(r'.*test_cosh.*')
backend_test.include(r'.*test_depthtospace.*')
backend_test.include(r'.*test_div.*')
backend_test.include(r'.*test_dropout.*')
backend_test.include(r'.*test_ELU*')
backend_test.include(r'.*test_elu.*')
backend_test.include(r'.*test_Embedding*')
backend_test.include(r'.*test_exp.*')
backend_test.include(r'.*test_flatten.*')
backend_test.include(r'.*test_floor.*')
backend_test.include(r'.*test_gather.*')
backend_test.include(r'.*test_globalaveragepool.*')
backend_test.include(r'.*test_globalmaxpool.*')
backend_test.include(r'.*test_hardsigmoid.*')
backend_test.include(r'.*test_identity.*')
backend_test.include(r'.*test_LeakyReLU*')
backend_test.include(r'.*test_leakyrelu.*')
backend_test.include(r'.*test_Linear.*')
backend_test.include(r'.*test_log.*')
backend_test.include(r'.*test_logsoftmax.*')
backend_test.include(r'.*test_LogSoftmax.*')
backend_test.include(r'.*test_log_softmax.*')
backend_test.include(r'.*test_lrn.*')
backend_test.include(r'.*test_matmul.*')
backend_test.include(r'.*test_max.*')
backend_test.include(r'.*test_MaxPool[1-9]d.*')
backend_test.include(r'.*test_mean.*')
backend_test.include(r'.*test_min.*')
backend_test.include(r'.*test_mul.*')
backend_test.include(r'.*test_neg.*')
backend_test.include(r'.*test_operator_addmm.*')
backend_test.include(r'.*test_operator_basic.*')
backend_test.include(r'.*test_operator_clip.*')
backend_test.include(r'.*test_operator_concat2.*')
backend_test.include(r'.*test_operator_conv_.*')
backend_test.include(r'.*test_operator_exp.*')
backend_test.include(r'.*test_operator_flatten.*')
backend_test.include(r'.*test_operator_index.*')
backend_test.include(r'.*test_operator_max_.*')
backend_test.include(r'.*test_operator_maxpool.*')
backend_test.include(r'.*test_operator_min.*')
backend_test.include(r'.*test_operator_non_float_params.*')
backend_test.include(r'.*test_operator_params.*')
backend_test.include(r'.*test_operator_permute2.*')
backend_test.include(r'.*test_operator_pow.*')
backend_test.include(r'.*test_operator_reduced_mean_.*')
backend_test.include(r'.*test_operator_reduced_mean_keepdim.*')
backend_test.include(r'.*test_operator_reduced_sum_.*')
backend_test.include(r'.*test_operator_reduced_sum_keepdim.*')
backend_test.include(r'.*test_operator_selu.*')
backend_test.include(r'.*test_operator_sqrt.*')
backend_test.include(r'.*test_operator_symbolic_override_nested.*')
backend_test.include(r'.*test_operator_view.*')
backend_test.include(r'.*test_pow.*')
backend_test.include(r'.*test_PoissonNLLLLoss_no_reduce*')
backend_test.include(r'.*test_reciprocal.*')
backend_test.include(r'.*test_reduce.*')
backend_test.include(r'.*test_ReLU*')
backend_test.include(r'.*test_relu.*')
backend_test.include(r'.*test_selu.*')
backend_test.include(r'.*test_Sigmoid*')
backend_test.include(r'.*test_sigmoid.*')
backend_test.include(r'.*test_sin.*')
backend_test.include(r'.*test_sinh.*')
backend_test.include(r'.*test_Softmax*')
backend_test.include(r'.*test_softmax.*')
backend_test.include(r'.*test_Softmin*')
backend_test.include(r'.*test_Softplus*')
backend_test.include(r'.*test_softplus.*')
backend_test.include(r'.*test_softsign.*')
backend_test.include(r'.*test_sqrt.*')
backend_test.include(r'.*test_squeeze_cuda')
backend_test.include(r'.*test_sub.*')
backend_test.include(r'.*test_sum.*')
backend_test.include(r'.*test_tan.*')
backend_test.include(r'.*test_Tanh*')
backend_test.include(r'.*test_tanh.*')
backend_test.include(r'.*test_thresholdedrelu.*')
backend_test.include(r'.*test_transpose.*')
backend_test.include(r'.*test_unsqueeze.*')
backend_test.include(r'.*test_ZeroPad2d*')

# Onnx native model tests
backend_test.include(r'.*test_bvlc_alexnet.*')
backend_test.include(r'.*test_densenet121.*')
backend_test.include(r'.*test_inception_v1.*')
backend_test.include(r'.*test_inception_v2.*')
backend_test.include(r'.*test_resnet50.*')
backend_test.include(r'.*test_shufflenet.*')
backend_test.include(r'.*test_squeezenet.*')
backend_test.include(r'.*test_vgg19.*')
backend_test.include(r'.*test_zfnet512.*')

# exclude unenabled ops get pulled in with wildcards
# test_constant_pad gets pulled in with the test_constant* wildcard. Explicitly disable padding tests for now.
backend_test.exclude(r'.*test_constant_pad.*')
backend_test.exclude(r'.*test_constantofshape.*')
backend_test.exclude(r'.*test_expand.*')
# Operator MATMULINTEGER is not supported by TRT
backend_test.exclude(r'.*test_matmulinteger.*')
backend_test.exclude(r'.*test_maxpool.*')
backend_test.exclude(r'.*test_maxunpool.*')

# Known issues
backend_test.exclude(r'.*test_gemm.*')
backend_test.exclude(r'.*test_operator_mm.*')
backend_test.exclude(r'.*test_shape.*')
backend_test.exclude(r'.*test_size.*')
backend_test.exclude(r'.*test_operator_symbolic_override.*')


globals().update(backend_test
                .enable_report()
                .test_cases)

if __name__ == '__main__':
    unittest.main()
