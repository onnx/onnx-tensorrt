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

from __future__ import print_function
from .tensorrt_engine import Engine
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
from onnx import helper as onnx_helper
import numpy as np
import sys
import six

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)

def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count

def _tensorrt_version():
    return [int(n) for n in trt.__version__.split('.')]

# If TensorRT major is >= 5, then we use new Python bindings
USE_PYBIND = _tensorrt_version()[0] >= 5

if USE_PYBIND:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if not USE_PYBIND:
    from . import parser
    from . import runtime as parser_runtime


class TensorRTBackendRep(BackendRep):
    def __init__(self, model, device, max_batch_size=32,
                 max_workspace_size=None, serialize_engine=False, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)

        if USE_PYBIND:
            self._logger = TRT_LOGGER
            self.builder = trt.Builder(self._logger)
        else:
            self._logger = trt.infer.ConsoleLogger(trt.infer.LogSeverity.WARNING)
            self.builder = trt.infer.create_infer_builder(self._logger)

        self.network = self.builder.create_network()

        if USE_PYBIND:
            self.parser = trt.OnnxParser(self.network, self._logger)
        else:
            self.parser = parser.create_parser(self.network, self._logger)

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model
        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
        if max_workspace_size is None:
            max_workspace_size = 1 << 28

        if USE_PYBIND:
            self.builder.max_batch_size = max_batch_size
            self.builder.max_workspace_size = max_workspace_size
        else:
            self.builder.set_max_batch_size(max_batch_size)
            self.builder.set_max_workspace_size(max_workspace_size)

        for layer in self.network:
            print(layer.name)

        print(self.network[-1].get_output(0).shape)

        trt_engine = self.builder.build_cuda_engine(self.network)
        if trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        if serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine)
        self.engine = Engine(trt_engine)
        self._output_shapes = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)
    def _serialize_deserialize(self, trt_engine):
        if USE_PYBIND:
            self.runtime = trt.Runtime(TRT_LOGGER)
        else:
            self.runtime = trt.infer.create_infer_runtime(self._logger)
            self.plugin_factory = parser_runtime.create_plugin_factory(self._logger)

        serialized_engine = trt_engine.serialize()
        del self.parser # Parser no longer needed for ownership of plugins

        if USE_PYBIND:
            trt_engine = self.runtime.deserialize_cuda_engine(
                serialized_engine)
        else:
            trt_engine = self.runtime.deserialize_cuda_engine(
                serialized_engine, self.plugin_factory)
        return trt_engine
    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.

        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        outputs = self.engine.run(inputs)
        output_names = [output.name for output in self.engine.outputs]
        for i, (name, array) in enumerate(zip(output_names, outputs)):
            output_shape = self._output_shapes[name]
            # HACK WAR for unknown output shape in run_node
            if output_shape == (-99,):
                # WAR for TRT requiring at least 2 dims (NC)
                min_dims = 2
                if _tensorrt_version()[0] < 4:
                    # WAR for TRT only supporting 4D (NCHW) tensors
                    min_dims = 4
                if array.ndim == min_dims:
                    npadding_dims = count_trailing_ones(array.shape)
                    if npadding_dims > 0:
                        outputs[i] = array.reshape(
                            array.shape[:-npadding_dims])
            else:
                # HACK WAR replace fixed batch dim with variable
                output_shape = (-1,) + output_shape[1:]
                outputs[i] = array.reshape(output_shape)
        outputs_tuple = namedtupledict('Outputs', output_names)(*outputs)
        return namedtupledict('Outputs', output_names)(*outputs)

def np2onnx_dtype(np_dtype):
    if np_dtype == np.dtype('float32'):
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.dtype('float16'):
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.dtype('int64'):
        return onnx.TensorProto.INT64
    elif np_dtype == np.dtype('int32'):
        return onnx.TensorProto.INT32
    elif np_dtype == np.dtype('int8'):
        return onnx.TensorProto.INT8
    else:
        raise TypeError("Unsupported data type:", np_dtype)

def make_node_test_model(node, inputs, use_weights=True):
    # HACK TODO: The output info is unknown here; not sure what the best solution is
    output_dtype = np.float32 # Dummy value only
    output_shape = [-99]      # Dummy value only
    graph_inputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(array.dtype), array.shape)
                    for name, array in zip(node.input, inputs)]
    graph_outputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(output_dtype), output_shape)
                     for name in node.output]
    if use_weights:
        # Add initializers for all inputs except the first
        initializers = [onnx_helper.make_tensor(
            name, np2onnx_dtype(array.dtype), array.shape, array.flatten().tolist())
                        for name, array in zip(node.input[1:], inputs[1:])]
    else:
        initializers = []
    graph = onnx_helper.make_graph(
           [node], "RunNodeGraph_" + node.op_type,
           graph_inputs, graph_outputs, initializer=initializers)
    model = onnx_helper.make_model(graph)
    return model

class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CUDA:0', **kwargs):
        """Build an engine from the given model.

        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        """
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, device, **kwargs)
    @classmethod
    def run_model(cls, model, inputs, device='CUDA:0', **kwargs):
        """Build and run an engine from the given model.

        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.

        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        return cls.prepare(model, device, **kwargs).run(inputs)
    @classmethod
    def run_node(cls, node, inputs, device='CUDA:0'):
        """Build and run an engine from the given node.

        node -- An ONNX node as a deserialized protobuf.

        Note: This function is intended for testing purposes only;
              use prepare() or run_model() for other purposes.
        """
        super(TensorRTBackend, cls).run_node(node, inputs, device)
        # HACK TODO: This is somewhat dodgy. We first try with weights for all
        #            inputs but the first, then we try again with no weights if
        #            the first try fails.
        model = make_node_test_model(node, inputs, use_weights=True)
        try: results = TensorRTBackend.prepare(model, device).run(inputs[:1])
        except RuntimeError:
            model = make_node_test_model(node, inputs, use_weights=False)
            results = TensorRTBackend.prepare(model, device).run(inputs)
        return results
    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        return device.type == DeviceType.CUDA

prepare         = TensorRTBackend.prepare
run_node        = TensorRTBackend.run_node
run_model       = TensorRTBackend.run_model
supports_device = TensorRTBackend.supports_device
