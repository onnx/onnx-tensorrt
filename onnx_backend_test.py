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
backend_test.exclude(r'[a-z,_]*_prelu_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_tile_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_repeat_[a-z,_]*') # 'Tile' op
backend_test.exclude(r'[a-z,_]*_compress_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_eyelike_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_expand_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_constantofshape_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_isnan_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_maxunpool_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_mvn_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_nonzero_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_onehot_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_scan.*')
backend_test.exclude(r'[a-z,_]*_scatter_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_shrink_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_sign_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_tfidfvectorizer_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_where_[a-z,_]*')

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
# No operations on axis 0 == BATCH_DIM
backend_test.exclude(r'[a-z,_]*softmax_axis_0_cuda[a-z,_]*')
backend_test.exclude(r'[a-z,_]*argmax_default_axis[a-z,_]*')
backend_test.exclude(r'[a-z,_]*argmin_default_axis[a-z,_]*')
backend_test.exclude(r'[a-z,_]*test_concat.*_axis_0[a-z,_]*')
backend_test.exclude(r'[a-z,_]*gather_0[a-z,_]*')
backend_test.exclude(r'test_reduce_log_sum_asc_axes_cuda')
backend_test.exclude(r'test_reduce_log_sum_default_cuda')
backend_test.exclude(r'test_split_variable_parts_1d_cuda')
backend_test.exclude(r'test_split_variable_parts_default_axis_cuda')
backend_test.exclude(r'test_squeeze_cuda')
backend_test.exclude(r'test_unsqueeze_cuda')
backend_test.exclude(r'test_Embedding_cuda')
backend_test.exclude(r'test_Embedding_sparse_cuda')
backend_test.exclude(r'test_operator_index_cuda')
backend_test.exclude(r'[a-z,_]*transpose_all_permutations[a-z,_]*')
backend_test.exclude(r'test_transpose_default_cuda')
backend_test.exclude(r'test_PoissonNLLLLoss_no_reduce_cuda')
backend_test.exclude(r'test_operator_params_cuda')
# No slice of batch dim
backend_test.exclude(r'test_operator_chunk_cuda')
# No fancy padding
backend_test.exclude(r'test_constant_pad_cuda')
backend_test.exclude(r'test_reflect_pad_cuda')
backend_test.exclude(r'test_edge_pad_cuda')
backend_test.exclude(r'[a-z,_]*_ReplicationPad[a-z,0-9,_]*')
backend_test.exclude(r'[a-z,_]*ReflectionPad2d[a-z,_]*')
# No non-zero padding
backend_test.exclude(r'test_ConstantPad2d_cuda')
backend_test.exclude(r'test_operator_pad_cuda')
# No RNNs yet
backend_test.exclude(r'[a-z,_]*_rnn_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_lstm_[a-z,_]*')
backend_test.exclude(r'[a-z,_]*_gru_[a-z,_]*')
# No reduce over batch dim
backend_test.exclude(r'test_reduce_[a-z,0-9,_]*_default_axes_[a-z,_]*')
# No INT64 type inputs
backend_test.exclude(r'test_argmax_keepdims_example_cuda')
backend_test.exclude(r'test_argmax_keepdims_random_cuda')
backend_test.exclude(r'test_argmax_no_keepdims_example_cuda')
backend_test.exclude(r'test_argmax_no_keepdims_random_cuda')
backend_test.exclude(r'test_argmin_keepdims_example_cuda')
backend_test.exclude(r'test_argmin_keepdims_random_cuda')
backend_test.exclude(r'test_argmin_no_keepdims_example_cuda')
backend_test.exclude(r'test_argmin_no_keepdims_random_cuda')
backend_test.exclude(r'test_top_k_cuda')
backend_test.exclude(r'test_gather_1_cuda')
backend_test.exclude(r'test_reshape_extended_dims_cuda')
backend_test.exclude(r'test_reshape_negative_dim_cuda')
backend_test.exclude(r'test_reshape_one_dim_cuda')
backend_test.exclude(r'test_reshape_reduced_dims_cuda')
backend_test.exclude(r'test_reshape_reordered_dims_cuda')
backend_test.exclude(r'test_shape_cuda')
backend_test.exclude(r'test_shape_example_cuda')
backend_test.exclude(r'test_size_cuda')
backend_test.exclude(r'test_size_example_cuda')
backend_test.exclude(r'test_operator_non_float_params_cuda')
# No DOUBLE type inputs
backend_test.exclude(r'test_operator_add_broadcast_cuda')
backend_test.exclude(r'test_operator_add_size1_broadcast_cuda')
backend_test.exclude(r'test_operator_add_size1_right_broadcast_cuda')
backend_test.exclude(r'test_operator_add_size1_singleton_broadcast_cuda')
backend_test.exclude(r'test_operator_addconstant_cuda')
# Weights for Conv, ConvTranspose and Batchnorm MUST be nvinfer1::Weights type. 
# GEMMs 'B' tensor must be a nvinver1::Weights type.
backend_test.exclude(r'test_basic_conv_with_padding_cuda')
backend_test.exclude(r'test_basic_conv_without_padding_cuda')
backend_test.exclude(r'test_batchnorm_epsilon_cuda')
backend_test.exclude(r'test_batchnorm_example_cuda')
backend_test.exclude(r'test_conv_with_strides_no_padding_cuda')
backend_test.exclude(r'test_conv_with_strides_padding_cuda')
backend_test.exclude(r'test_convtranspose_1d_cuda')
backend_test.exclude(r'test_convtranspose_3d_cuda')
backend_test.exclude(r'test_convtranspose_cuda')
backend_test.exclude(r'test_convtranspose_kernel_shape_cuda')
backend_test.exclude(r'test_convtranspose_output_shape_cuda')
backend_test.exclude(r'test_convtranspose_pad_cuda')
backend_test.exclude(r'test_convtranspose_pads_cuda')
backend_test.exclude(r'test_convtranspose_with_kernel_cuda')
backend_test.exclude(r'test_gemm_broadcast_cuda')
backend_test.exclude(r'test_gemm_nobroadcast_cuda')
backend_test.exclude(r'test_instancenorm_epsilon_cuda')
backend_test.exclude(r'test_instancenorm_example_cuda')
backend_test.exclude(r'test_matmul_2d_cuda')
backend_test.exclude(r'test_operator_addmm_cuda')
backend_test.exclude(r'test_operator_mm_cuda')
# Can't handle 1D tensor inputs since we strip batch dimension:
backend_test.exclude(r'test_mul_bcast_cuda')
backend_test.exclude(r'test_sub_bcast_cuda')
backend_test.exclude(r'test_add_bcast_cuda')
backend_test.exclude(r'test_div_bcast_cuda')
backend_test.exclude(r'test_pow_bcast_array_cuda')
backend_test.exclude(r'test_pow_bcast_scalar_cuda')
# Do not support "Indices" output for maxpool
backend_test.exclude(r'[a-z,_]*maxpool_with_argmax[a-z,_]*')
# Incompatible padding
backend_test.exclude(r'test_averagepool_2d_same_lower_cuda')
# TODO: Fix these tests
backend_test.exclude(r'test_averagepool_2d_pads_count_include_pad_cuda')
backend_test.exclude(r'test_averagepool_2d_precomputed_pads_count_include_pad_cuda')
backend_test.exclude(r'test_logsoftmax_axis_1_cuda')
backend_test.exclude(r'test_softmax_axis_1_cuda')
backend_test.exclude(r'[a-z,_]*softmax_default_axis_cuda[a-z,_]*')

if 'TRAVIS' in os.environ:
    backend_test.exclude('(test_vgg19|test_vgg16)')
# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
