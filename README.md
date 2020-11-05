# TensorRT Backend For ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

For the list of recent changes, see the [changelog](Changelog.md).

## Supported TensorRT Versions

Development on the Master branch is for the latest version of [TensorRT 7.2.1](https://developer.nvidia.com/nvidia-tensorrt-download) with full-dimensions and dynamic shape support.

For previous versions of TensorRT, refer to their respective branches.

## Full Dimensions + Dynamic Shapes

Building INetwork objects in full dimensions mode with dynamic shape support requires calling the following API:

C++

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    builder->createNetworkV2(explicitBatch)

Python

    import tensorrt
    explicit_batch = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder.create_network(explicit_batch)

For examples of usage of these APIs see:
* [sampleONNXMNIST](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleOnnxMNIST)
* [sampleDynamicReshape](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleDynamicReshape)

## Supported Operators

Current supported ONNX operators are found in the [operator support matrix](operators.md).

# Installation

### Dependencies

 - [Protobuf >= 3.0.x](https://github.com/google/protobuf/releases)
 - [TensorRT 7.2.1](https://developer.nvidia.com/tensorrt)
 - [TensorRT 7.2.1 open source libaries (master branch)](https://github.com/NVIDIA/TensorRT/)

### Building

For building within docker, we recommend using and setting up the docker containers as instructed in the main (TensorRT repository)[https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment] to build the onnx-tensorrt library.

Once you have cloned the repository, you can build the parser libraries and executables by running:

    cd onnx-tensorrt
    mkdir build && cd build
    cmake .. -DTENSORRT_ROOT=<path_to_trt> && make -j
    // Ensure that you update your LD_LIBRARY_PATH to pick up the location of the newly built library:
    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

## Executable Usage

ONNX models can be converted to serialized TensorRT engines using the `onnx2trt` executable:

    onnx2trt my_model.onnx -o my_engine.trt

ONNX models can also be converted to human-readable text:

    onnx2trt my_model.onnx -t my_model.onnx.txt

ONNX models can also be optimized by ONNX's optimization libraries (added by [dsandler](https://gitlab-master.nvidia.com/dsandler)).
To optimize an ONNX model and output a new one use `-m` to specify the output model name and `-O` to specify a semicolon-separated list of optimization passes to apply:

    onnx2trt my_model.onnx -O "pass_1;pass_2;pass_3" -m my_model_optimized.onnx

See more all available optimization passes by running:

    onnx2trt -p

See more usage information by running:

    onnx2trt -h

### Python Modules

Python bindings for the ONNX-TensorRT parser are packaged in the shipped `.whl` files. Install them with

    python3 -m pip install <tensorrt_install_dir>/python/tensorrt-7.x.x.x-cp<python_ver>-none-linux_x86_64.whl

TensorRT 7.2.1 supports ONNX release 1.6.0. Install it with:

    python3 -m pip install onnx==1.6.0

The ONNX-TensorRT backend can be installed by running:

    python3 setup.py install

## ONNX-TensorRT Python Backend Usage

The TensorRT backend for ONNX can be used in Python as follows:

```python
import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("/path/to/model.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
```

## C++ Library Usage

The model parser library, libnvonnxparser.so, has its C++ API declared in this header:

    NvOnnxParser.h

### Tests

After installation (or inside the Docker container), ONNX backend tests can be run as follows:

Real model tests only:

    python onnx_backend_test.py OnnxBackendRealModelTest

All tests:

    python onnx_backend_test.py

You can use `-v` flag to make output more verbose.

## Pre-trained Models

Pre-trained models in ONNX format can be found at the [ONNX Model Zoo](https://github.com/onnx/models)