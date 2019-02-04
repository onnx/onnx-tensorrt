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

import tensorrt as trt
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
from six import string_types


def _tensorrt_version():
    return [int(n) for n in trt.__version__.split('.')]

# If TensorRT major is >= 5, then we use new Python bindings
USE_PYBIND = _tensorrt_version()[0] >= 5


class Binding(object):
    def __init__(self, engine, idx_or_name, max_batch_size):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index  = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name  = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)

        if USE_PYBIND:
            dtype = engine.get_binding_dtype(self.index)
            dtype_map = {trt.DataType.FLOAT: np.float32,
                         trt.DataType.HALF:  np.float16,
                         trt.DataType.INT8:  np.int8}
            if hasattr(trt.DataType, 'INT32'):
                dtype_map[trt.DataType.INT32] = np.int32
        else:
            dtype = engine.get_binding_data_type(self.index)
            dtype_map = {trt.infer.DataType_kFLOAT: np.float32,
                         trt.infer.DataType_kHALF:  np.float16,
                         trt.infer.DataType_kINT8:  np.int8}
            if hasattr(trt.infer, 'DataType_kINT32'):
                dtype_map[trt.infer.DataType_kINT32] = np.int32

        self.dtype = dtype_map[dtype]

        if USE_PYBIND:
            shape = engine.get_binding_shape(self.index)
        else:
            shape = engine.get_binding_dimensions(self.index).shape()

        self.shape = (max_batch_size,) + tuple(shape)
        self._host_buf   = None
        self._device_buf = None
    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf
    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf
    def get_async(self, batch_size, stream):
        src = self.device_buffer[:batch_size]
        dst = self.host_buffer[:batch_size]
        src.get_async(stream, dst)
        return dst

def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine

        if USE_PYBIND:
            self.max_batch_size = self.engine.max_batch_size
            nbinding = self.engine.num_bindings
        else:
            self.max_batch_size = self.engine.get_max_batch_size()
            nbinding = self.engine.get_nb_bindings()

        bindings = [Binding(self.engine, i, self.max_batch_size)
                    for i in range(nbinding)]
        self.binding_addrs = [b.device_buffer.ptr for b in bindings]
        self.inputs  = [b for b in bindings if     b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]
        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()
    def __del__(self):
        if self.engine is not None:
            if USE_PYBIND:
                del self.engine
            else:
                self.engine.destroy()
    def run(self, inputs):
        if len(inputs) != len(self.inputs):
            raise ValueError("Wrong number of inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]
        batch_size = inputs[0].shape[0]
        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            if input_array.shape[0] != batch_size:
                raise ValueError("All inputs must have same batch size")
            # WAR for TRT requiring at least 2 dims (NC)
            min_dims = 2
            if _tensorrt_version()[0] < 4:
                min_dims = 4
            if input_array.ndim < min_dims:
                new_shape = (input_array.shape +
                             (1,) * (min_dims - input_array.ndim))
                input_array = input_array.reshape(new_shape)
            expected_shape = tuple(input_binding.shape[1:])
            given_shape    = tuple(input_array.shape[1:])
            if given_shape != expected_shape:
                raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                                 (i, expected_shape, given_shape))
            if input_array.dtype != input_binding.dtype:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                                 (i, input_binding.dtype, input_array.dtype))
            input_binding_array = input_binding.device_buffer[:batch_size]
            input_binding_array.set_async(input_array, self.stream)

        if USE_PYBIND:
            self.context.execute_async(
                batch_size, self.binding_addrs, self.stream.handle)
        else:
            self.context.enqueue(
                batch_size, self.binding_addrs, self.stream.handle, None)

        results = [output.get_async(batch_size, self.stream)
                   for output in self.outputs]
        self.stream.synchronize()
        return results
    def run_no_dma(self, batch_size):
        if USE_PYBIND:
            self.context.execute_async(
                batch_size, self.binding_addrs, self.stream.handle)
        else:
            self.context.enqueue(
                batch_size, self.binding_addrs, self.stream.handle, None)
        self.stream.synchronize()
