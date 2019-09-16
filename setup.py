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

import os
import argparse
from setuptools import setup, find_packages, Extension

__version__ = '0.1.0'

parser = argparse.ArgumentParser(description='Setup to build ONNX TensorRT parser')
parser.add_argument('action', nargs='*')

parser.add_argument('--build-lib', type=str,
                    help='A location of the build directory')

parser.add_argument('--include-dirs', type=str,
                    help='A location of the include directories, semicolon separated')



args = parser.parse_args()

print(args)

if args.build_lib == None:
    args.build_lib = 'build'

TRT_ROOT = os.getenv('TRT_ROOT')

if TRT_ROOT == None:
    INC_DIRS = []
else:
    INC_DIRS = [TRT_ROOT + '/include']

SWIG_OPTS = [
    '-c++',
    '-modern',
    '-builtin',
]

EXTRA_COMPILE_ARGS =  [
    '-std=c++11',
    '-DUNIX',
    '-D__UNIX',
    '-m64',
    '-fPIC',
    '-O2',
    '-w',
    '-fmessage-length=0',
    '-fno-strict-aliasing',
    '-D_FORTIFY_SOURCE=2',
    '-fstack-protector',
    '--param=ssp-buffer-size=4',
    '-Wformat',
    '-Werror=format-security',
    '-DNDEBUG',
    '-g',
    '-fwrapv',
    '-Wall',
    '-DSWIG',
]

EXTRA_LINK_ARGS = [
]



nv_onnx_parser_module = Extension(
    'onnx_tensorrt.parser._nv_onnx_parser_bindings',
    sources=['nv_onnx_parser_bindings.i'],
    swig_opts=SWIG_OPTS,
    extra_objects=[
        args.build_lib + '/libnvonnxparser.so',
    ],
    include_dirs=INC_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS)

setup(name='onnx_tensorrt',
      version=__version__,
      description='TensorRT backend for ONNX',
      author='NVIDIA Corporation',
      author_email='bbarsdell@nvidia.com',
      url='https://github.com/onnx/onnx-tensorrt',
      packages=find_packages(),
      ext_modules=[nv_onnx_parser_module, nv_onnx_runtime_module],
      install_requires=[
          "numpy>=1.8.1",
          "tensorrt>=3.0.0",
          "onnx>=1.0.1",
          "pycuda",
      ])
