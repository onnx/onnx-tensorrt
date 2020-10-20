# ONNX-TensorRT Changelog

## TensorRT 7.2.1 Release - 2020-10-20

### Added
- Added support for parsing large models with external data
- Added API for interfacing with TensorRT's refit feature
- Updated `onnx_tensorrt` backend to support dynamic shapes
- Added support for 3D instance normalizations [#515](https://github.com/onnx/onnx-tensorrt/pull/515)
- Improved clarity on the resize modes TRT supports [#512](https://github.com/onnx/onnx-tensorrt/pull/521)
- Added Changelog

### Changed
- Unified docker usage between ONNX-TensorRT and TensorRT.

## Removed
- Removed deprecated docker files.
- Removed deprecated `setup.py`. 

