# SPDX-License-Identifier: Apache-2.0

import tensorrt as trt
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
from six import string_types

class Binding(object):
    def __init__(self, engine, idx_or_name):
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


        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL: np.bool}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)
        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False
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
    def get_async(self, stream):
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x

def check_input_validity(input_idx, input_array, input_binding):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        #TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                            (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i)
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
            del self.engine

    def run(self, inputs):
        # len(inputs) > len(self.inputs) with Shape operator, input is never used
        # len(inputs) == len(self.inputs) for other operators
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]


        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(
            self.binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream)
                   for output in self.outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self.outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):
        self.context.execute_async(
            batch_size, self.binding_addrs, self.stream.handle)
