# SPDX-License-Identifier: Apache-2.0

import tensorrt as trt
import numpy as np
from six import string_types
from cuda.bindings import runtime as cudart
from typing import Union, Optional

# RAII memory management classes
import ctypes
from cuda.bindings import driver as cuda, nvrtc


class ArrayWithOwner(np.ndarray):
    """Numpy array that holds a reference to its owner object"""
    def __new__(cls, input_array, owner):
        obj = np.asarray(input_array).view(cls)
        obj._owner = owner
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._owner = getattr(obj, '_owner', None)



def cuda_call(call):
    """Helper function to make CUDA calls and check for errors"""
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    err, res = call[0], call[1:]
    if err.value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                err.value, _cudaGetErrorEnum(err)
            )
        )
    if len(res) == 1:
        return res[0]
    elif len(res) == 0:
        return None
    else:
        return res


# Initialize CUDA
cuda_call(cudart.cudaFree(0))


class PinnedHostMem:
    """Pinned host memory allocation for faster GPU transfers"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        if dtype is None:
            dtype = np.dtype(np.uint8)
        else:
            dtype = np.dtype(dtype)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))

        self._host_ptr = host_mem
        self._host_size = size
        self._nbytes = nbytes
        self._dtype = dtype

    @property
    def array(self) -> np.ndarray:
        # Create view with proper memory ownership
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(self._dtype))
        host_array = np.ctypeslib.as_array(ctypes.cast(self._host_ptr, pointer_type), (self._host_size,))
        return ArrayWithOwner(host_array, self)

    @array.setter
    def array(self, data: Union[np.ndarray, bytes]):
        """Set the array data with proper bounds checking"""
        host_array = self.array  # Get the numpy array view
        if isinstance(data, np.ndarray):
            if data.size > self._host_size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self._host_size}"
                )
            np.copyto(host_array[:data.size], data.flat, casting='safe')
        else:
            assert self._dtype == np.uint8
            host_array[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        """Explicitly free pinned host memory"""
        if self._host_ptr is not None:
            try:
                cuda_call(cudart.cudaFreeHost(self._host_ptr))
                self._host_ptr = None
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __str__(self):
        return f"PinnedHost:\n{self.array}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        # Fallback cleanup - not guaranteed to be called
        self.free()


class DeviceMem:
    """Device-only memory allocation for cases where host memory is not needed"""
    def __init__(self, size: int):
        self._device_ptr = cuda_call(cudart.cudaMalloc(size))
        self._nbytes = size

    @property
    def device_ptr(self) -> int:
        """Device memory pointer"""
        return self._device_ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        """Explicitly free device memory"""
        if self._device_ptr is not None:
            try:
                cuda_call(cudart.cudaFree(self._device_ptr))
                self._device_ptr = None
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __str__(self):
        return f"Device:\n{self.device_ptr}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        # Fallback cleanup - not guaranteed to be called
        self.free()


class CudaStream:
    """RAII wrapper for CUDA stream"""
    def __init__(self):
        self._stream = cuda_call(cudart.cudaStreamCreate())

    @property
    def stream(self):
        return self._stream

    def free(self):
        """Explicitly free the CUDA stream"""
        if self._stream is not None:
            try:
                cuda_call(cudart.cudaStreamDestroy(self._stream))
                self._stream = None
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __del__(self):
        # Fallback cleanup - not guaranteed to be called
        self.free()


class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
        else:
            self.index = idx_or_name
            self.name  = engine.get_tensor_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT


        dtype = engine.get_tensor_dtype(self.name)
        dtype_map = {trt.DataType.FLOAT:  np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL:  np.bool_,}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64

        self.dtype = dtype_map[dtype]
        shape = engine.get_tensor_shape(self.name)

        self.shape = tuple(shape)
        self._host_buf = None
        self._device_mem = None

    @property
    def host_buffer(self):
        if self._host_buf is None:
            size = np.prod(self.shape)
            self._host_buf = PinnedHostMem(size, np.dtype(self.dtype))
        return self._host_buf.array

    @property
    def device_buffer(self):
        if self._device_mem is None:
            size = np.prod(self.shape)
            nbytes = size * np.dtype(self.dtype).itemsize
            self._device_mem = DeviceMem(nbytes)
        return self._device_mem.device_ptr
    def get_async(self, stream):
        cuda_call(cudart.cudaMemcpyAsync(self.host_buffer.ctypes.data, self.device_buffer, self.host_buffer.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        return self.host_buffer

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

        bindings = [Binding(self.engine, i)
                    for i in range(self.engine.num_io_tensors)]
        self.binding_addrs = [int(b.device_buffer) for b in bindings]
        self.inputs  = [b for b in bindings if     b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]

        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = CudaStream()

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
            cuda_call(cudart.cudaMemcpyAsync(input_binding.device_buffer, input_array.ctypes.data, input_array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream.stream))

        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            tensor_name = self.engine.get_tensor_name(i)
            if i < len(inputs) and self.engine.is_shape_inference_io(tensor_name):
                self.context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
            else:
                self.context.set_tensor_address(tensor_name, self.binding_addrs[i])

        self.context.execute_async_v3(self.stream.stream)

        results = [output.get_async(self.stream.stream)
                   for output in self.outputs]
        cuda_call(cudart.cudaStreamSynchronize(self.stream.stream))

        return results

    def run_no_dma(self):
        self.context.execute_async_v3(self.stream.stream)
