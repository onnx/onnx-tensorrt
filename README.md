# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

## Supported TensorRT Versions

Development on the Master branch is for the latest version of [TensorRT 7.0](https://developer.nvidia.com/nvidia-tensorrt-download) with full-dimensions and dynamic shape support.

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

 - [Protobuf >= 3.8.x](https://github.com/google/protobuf/releases)
 - [TensorRT 7.0](https://developer.nvidia.com/tensorrt)
 - [TensorRT 7.0 open source libaries (master branch)](https://github.com/NVIDIA/TensorRT/)

### Building

For building on master, we recommend following the instructions on the [master branch of TensorRT](https://github.com/NVIDIA/TensorRT/) as there are new dependencies that were introduced to support these new features.

To build on older branches refer to their respective READMEs.


## Executable usage

ONNX models can be converted to serialized TensorRT engines using the `onnx2trt` executable:

    onnx2trt my_model.onnx -o my_engine.trt

ONNX models can also be converted to human-readable text:

    onnx2trt my_model.onnx -t my_model.onnx.txt

ONNX models can also be optimized by ONNX's optimization libraries.
To optimize an ONNX model and output a new one use `-m` to specify the output model name and `-O` to specify a semicolon-separated list of optimization passes to apply:

    onnx2trt my_model.onnx -O "pass_1;pass_2;pass_3" -m my_model_optimized.onnx

See more all available optimization passes by running:

    onnx2trt -p

See more usage information by running:

    onnx2trt -h

### Python modules
Python bindings for the ONNX-TensorRT parser are packaged in the shipped `.whl` files. Install them with

    pip install <tensorrt_install_dir>/python/tensorrt-7.x.x.x-cp27-none-linux_x86_64.whl

TensorRT 7.0 supports ONNX release 1.6.0. Install it with:

    pip install onnx==1.6.0

## ONNX Python backend usage

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

## C++ library usage

The model parser library, libnvonnxparser.so, has its C++ API declared in this header:

    NvOnnxParser.h

### Docker image

#### Tar-Based TensorRT

Build the onnx_tensorrt Docker image using tar-based TensorRT by running:

    git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git
    cd onnx-tensorrt
    cp /path/to/TensorRT-7.x.x.tar.gz .
    docker build -f docker/onnx-tensorrt-tar.Dockerfile --tag=onnx-tensorrt:7.x.x .

#### Deb-Based TensorRT

Build the onnx_tensorrt Docker image using deb-based TensorRT by running:

    git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git
    cd onnx-tensorrt
    cp /path/to/nv-tensorrt-repo-ubuntu1x04-cudax.x-trt7.x.x.x-ga-yyyymmdd_1-1_amd64.deb .
    docker build -f docker/onnx-tensorrt-deb.Dockerfile --tag=onnx-tensorrt:7.x.x.x .

### Tests

After installation (or inside the Docker container), ONNX backend tests can be run as follows:

Real model tests only:

    python onnx_backend_test.py OnnxBackendRealModelTest

All tests:

    python onnx_backend_test.py

You can use `-v` flag to make output more verbose.

## Pre-trained models

Pre-trained models in ONNX format can be found at the [ONNX Model Zoo](https://github.com/onnx/models)
