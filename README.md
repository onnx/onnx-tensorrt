<!--- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT Backend For ONNX

Parses ONNX models for execution with [TensorRT](https://developer.nvidia.com/tensorrt).

See also the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/).

For the list of recent changes, see the [changelog](docs/Changelog.md).

For a list of commonly seen issues and questions, see the [FAQ](docs/faq.md).

For business inquiries, please contact researchinquiries@nvidia.com

For press and other inquiries, please contact Hector Marinez at hmarinez@nvidia.com

## Supported TensorRT Versions

Development on the this branch is for the latest version of [TensorRT 10.15](https://developer.nvidia.com/nvidia-tensorrt-download) with full-dimensions and dynamic shape support.

For previous versions of TensorRT, refer to their respective branches.

## Supported Operators

Current supported ONNX operators are found in the [operator support matrix](docs/operators.md).

# Installation

### Dependencies

 - [TensorRT 10.15](https://developer.nvidia.com/tensorrt)
 - [TensorRT 10.15 open source libraries](https://github.com/NVIDIA/TensorRT/)
 - [Protobuf >= 3.20.3 (Optional)](https://github.com/google/protobuf/releases)

### Building

For building within Docker or on Windows, we recommend using the build instructions in the main [TensorRT repository](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment) to build the onnx-tensorrt library.

Once you have cloned the repository, you can build the parser libraries and executables by running:

    cd onnx-tensorrt
    mkdir build && cd build
    cmake .. -DTENSORRT_ROOT=<path_to_trt> && make -j
    # Ensure that you update your LD_LIBRARY_PATH to pick up the location of the newly built library:
    export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

Note that this project has a dependency on CUDA. By default the build will look in `/usr/local/cuda` for the CUDA toolkit installation. If your CUDA path is different, overwrite the default path by providing `-DCUDA_TOOLKIT_ROOT_DIR=<path_to_cuda_install>` in the CMake command.

To build with `protobuf-lite` support, add `-DUSE_ONNX_LITE_PROTO=1` to the end of the `cmake` command.

### InstanceNormalizaiton Performance

There are two implementations of InstanceNormalization that may perform differently depending on various parameters. By default, the parser will use the native TensorRT implementation of InstanceNorm. Users that want to benchmark using the plugin implementation of InstanceNorm can unset the parser flag `kNATIVE_INSTANCENORM` prior to parsing the model. Note that the plugin implementation cannot be used for building version compatible or hardware compatible engines, and attempting to do so will result in an error.

C++ Example:

    // Unset the kNATIVE_INSTANCENORM flag to use the plugin implementation.
    parser->unsetFlag(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);

Python Example:

    // Unset the NATIVE_INSTANCENORM flag to use the plugin implementation.
    parser.clear_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

## Executable Usage

There are currently two officially supported tools for users to quickly check if an ONNX model can parse and build into a TensorRT engine from an ONNX file.

For C++ users, there is the [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/opensource/trtexec) binary that is typically found in the `<tensorrt_root_dir>/bin` directory. The basic command of running an ONNX model is:

`trtexec --onnx=model.onnx`

Refer to the link or run `trtexec -h` for more information on CLI options.

For Python users, there is the [polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) tool. The basic command for running an onnx model is:

`polygraphy run model.onnx --trt`

Refer to the link or run `polygraphy run -h` for more information on CLI options.

### Python Modules

Python bindings for the ONNX-TensorRT parser are packaged in the shipped `.whl` files.

TensorRT 10.15 supports ONNX release 1.18.0. Install it with:

    python3 -m pip install onnx==1.18.0

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
