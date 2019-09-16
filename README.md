# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

## Supported TensorRT Versions

Development on the Master branch is for the latest version of [TensorRT 6.0](https://developer.nvidia.com/nvidia-tensorrt-download) with full-dimensions and dynamic shape support.

For version 6.0 without full-dimensions support, clone and build from the [6.0 branch](https://github.com/onnx/onnx-tensorrt/tree/6.0)

For version 5.1, clone and build from the [5.1 branch](https://github.com/onnx/onnx-tensorrt/tree/5.1)

For versions < 5.1, clone and build from the [5.0 branch](https://github.com/onnx/onnx-tensorrt/tree/v5.0)


## Supported Operators

Current supported ONNX operators are found in the [operator support matrix](operators.md).

# Installation

### Dependencies

 - [Protobuf >= 3.8.x](https://github.com/google/protobuf/releases)
 - [TensorRT 6.0](https://developer.nvidia.com/tensorrt)
 - [TensorRT 6.0 open source libaries (master branch)](https://github.com/NVIDIA/TensorRT/)

### Building

For building on master, we recommend following the instructions on the [master branch of TensorRT](https://github.com/NVIDIA/TensorRT/) as there are new dependencies that were introduced to support these new features.

To build on older branches refer to their respective READMEs.


## Executable usage

ONNX models can be converted to serialized TensorRT engines using the `onnx2trt` executable:

    onnx2trt my_model.onnx -o my_engine.trt

ONNX models can also be converted to human-readable text:

    onnx2trt my_model.onnx -t my_model.onnx.txt

See more usage information by running:

    onnx2trt -h

### Python modules
Python bindings for the ONNX-TensorRT parser are packaged in the shipped `.whl` files. Install them with

    pip install <tensorrt_install_dir>/python/tensorrt-6.0.1.5-cp27-none-linux_x86_64.whl

TensorRT 6.0 supports ONNX release 1.5.0. Install it with:

    pip install onnx==1.5.0

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

Important typedefs required for parsing ONNX models are declared in this header:

    NvOnnxParserTypedefs.h

### Docker image

Build the onnx_tensorrt Docker image by running:

    cp /path/to/TensorRT-6.0.*.tar.gz .
    docker build -t onnx_tensorrt .

### Tests

After installation (or inside the Docker container), ONNX backend tests can be run as follows:

Real model tests only:

    python onnx_backend_test.py OnnxBackendRealModelTest

All tests:

    python onnx_backend_test.py

You can use `-v` flag to make output more verbose.

## Pre-trained models

Pre-trained models in ONNX format can be found at the [ONNX Model Zoo](https://github.com/onnx/models)