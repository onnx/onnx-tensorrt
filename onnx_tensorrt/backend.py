# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function
from .tensorrt_engine import Engine
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
from onnx import helper as onnx_helper
from onnx import numpy_helper
import numpy as np
import six

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        if isinstance(error_string, bytes):
            error_string = error_string.decode("utf-8")
        raise RuntimeError("cudaSetDevice: " + error_string)

def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTBackendRep(BackendRep):
    def __init__(self, model, device,
            max_workspace_size=None, serialize_engine=False, verbose=False, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self._logger = TRT_LOGGER
        self.builder = trt.Builder(self._logger)
        self.config = self.builder.create_builder_config()
        self.network = self.builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self._logger)
        self.shape_tensor_inputs = []
        self.serialize_engine = serialize_engine
        self.verbose = verbose
        self.dynamic = False

        if self.verbose:
            print(f'\nRunning {model.graph.name}...')
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
        if max_workspace_size is None:
            max_workspace_size = 1 << 28

        self.config.max_workspace_size = max_workspace_size

        num_inputs = self.network.num_inputs
        for idx in range(num_inputs):
            inp_tensor = self.network.get_input(idx)
            if inp_tensor.is_shape_tensor or -1 in inp_tensor.shape:
                self.dynamic = True
                break

        if self.verbose:
            for layer in self.network:
                print(layer)

            print(f'Output shape: {self.network[-1].get_output(0).shape}')

        if self.dynamic:
            if self.verbose:
                print("Found dynamic inputs! Deferring engine build to run stage")
        else:
            self._build_engine()
        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type

    def _build_engine(self, inputs=None):
        """
        Builds a TensorRT engine with a builder config.
        :param inputs: inputs to the model; if not None, this means we are building the engine at run time,
                       because we need to register optimization profiles for some inputs
        :type inputs: List of np.ndarray
        """

        if inputs:
            opt_profile = self.builder.create_optimization_profile()
            # Set optimization profiles for the input bindings that need them
            for i in range(self.network.num_inputs):
                inp_tensor = self.network.get_input(i)
                name = inp_tensor.name
                # Set profiles for shape tensors
                if inp_tensor.is_shape_tensor:
                    if inputs[i].ndim > 0:
                        val_list = inputs[i].tolist()
                        opt_profile.set_shape_input(name, val_list, val_list, val_list)
                    else:
                        opt_profile.set_shape_input(name, [inputs[i]], [inputs[i]], [inputs[i]])
                # Set profiles for dynamic execution tensors
                elif -1 in inp_tensor.shape:
                    opt_profile.set_shape(name, inputs[i].shape, inputs[i].shape, inputs[i].shape)

            self.config.add_optimization_profile(opt_profile)

        trt_engine = self.builder.build_engine(self.network, self.config)

        if trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        if self.serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine)
        self.engine = Engine(trt_engine)

    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def _serialize_deserialize(self, trt_engine):
        self.runtime = trt.Runtime(TRT_LOGGER)
        serialized_engine = trt_engine.serialize()
        del self.parser # Parser no longer needed for ownership of plugins
        trt_engine = self.runtime.deserialize_cuda_engine(
                serialized_engine)
        return trt_engine

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        if self.dynamic:
            self._build_engine(inputs)

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
                if self._output_dtype[name] == onnx.TensorProto.INT64 and array.dtype == np.int32:
                    casted_output = np.array(outputs[i], dtype=np.int64)
                    if np.equal(outputs[i], casted_output).all():
                        outputs[i] = np.array(outputs[i], dtype=np.int64)
                if self._output_dtype[name] == onnx.TensorProto.DOUBLE and array.dtype == np.float32:
                    casted_output = np.array(outputs[i], dtype=np.double)
                    if np.equal(outputs[i], casted_output).all():
                        outputs[i] = np.array(outputs[i], dtype=np.double)

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
    elif np_dtype == np.dtype('double'):
        return onnx.TensorProto.DOUBLE
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
