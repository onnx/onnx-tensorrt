# TensorRT backend for ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/sdk/#inference).

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

## Executable usage

ONNX models can be converted to serialized TensorRT engines using the `onnx2trt` executable:

    onnx2trt my_model.onnx -o my_engine.trt

ONNX models can also be converted to human-readable text:

    onnx2trt my_model.onnx -t my_model.onnx.txt

See more usage information by running:

    onnx2trt -h

## C++ library usage

The model parser library, libnvonnxparser.so, has a C++ API declared in this header:

    NvOnnxParser.h

TensorRT engines built using this parser must use the plugin factory provided in
libnvonnxparser_runtime.so, which has a C++ API declared in this header:

    NvOnnxParserRuntime.h

## Installation

### Dependencies

 - [Protobuf](https://github.com/google/protobuf/releases)
 - [TensorRT 3+](https://developer.nvidia.com/tensorrt)

### Download the code
Clone the code from GitHub. 

    git clone --recursive https://github.com/onnx/onnx-tensorrt.git

### Executable and libraries

Suppose your TensorRT library is located at `/opt/tensorrt`. Build the `onnx2trt` executable and the `libnvonnxparser*` libraries using CMake. Note that `onnx-tensorrt` will by default tell the CUDA compiler generate code for the latest SM version. If you are using a GPU with a lower SM version you can specify which SMs to build for by using the optional `-DGPU_ARCHS` flag. For example, if you are running TensorRT on an older Pascal card such as a GTX 1080, you can specify `-DGPU_ARCHS="61"` to generate CUDA code specifically for your card.

See [here](https://developer.nvidia.com/cuda-gpus) for finding what maximum compute capability your specific GPU supports.

    mkdir build
    cd build
    cmake .. -DTENSORRT_ROOT=/opt/tensorrt
    OR
    cmake .. -DTENSORRT_ROOT=/opt/tensorrt -DGPU_ARCHS="61"
    make -j8
    sudo make install

### Python modules
Python bindings for the ONNX-TensorRT parser in TensorRT versions >= 5.0 are packaged in the shipped `.whl` files. No extra install is necessary.

For earlier versions of TensorRT, the Python wrappers are built using SWIG.
Build the Python wrappers and modules by running:

    python setup.py build
    sudo python setup.py install

### Docker image

Build the onnx_tensorrt Docker image by running:

    cp /path/to/TensorRT-3.0.*.tar.gz .
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
