# ONNX-TensorRT Changelog

## 21.03 Container Release - 2021-03-09
### Added
- Added opset13 support for `SoftMax`, `LogSoftmax`, `Squeeze`, and `Unsqueeze`
- Added support for the `EyeLike` operator
- Added support for the `GatherElements` operator

### Fixes
### Removed

## 21.02 Container Release - 2021-01-18
### Added
 - Added support for the `ReverseSequence` operator [#590] - https://github.com/onnx/onnx-tensorrt/pull/590
 - Updated `parse()` and `supportsModel()` API calls with an optional `model_path` parameter to support models with external weights [#621](https://github.com/onnx/onnx-tensorrt/pull/621)
 - Added support for the `Celu` operator
 - Added support for the `CumSum` operator
 - Added support for the `LessOrEqual` operator
 - Added support for the `LpNormalization` operator
 - Added support for the `LpPool` operator
 - Added support for the `GreaterOrEqual` operator
 - Added support for dynamic inputs in `onnx_tensorrt` python backend
 - Added FAQ section for commonly asked questions

### Fixes
 - Fixed relative path imports for models with external weights [#619]https://github.com/onnx/onnx-tensorrt/pull/619
 - Fixed importing loops operators with no loop-carried depedencies [#619](https://github.com/onnx/onnx-tensorrt/pull/619)
 - Worked around unsupported BOOL concats through casting [#620](https://github.com/onnx/onnx-tensorrt/pull/620)
 - Fixed compilation error with GCC9 [#568](https://github.com/onnx/onnx-tensorrt/pull/568)

### Removed
 - Removed `onnx_tensorrt/config.py` as it is no longer needed

## 20.12 Container Release - 2020-12-17

### Added
 - Added `setup.py` to properly install `onnx_tensorrt` python backend
 - Added 4D transpose for ONNX weights [#557](https://github.com/onnx/onnx-tensorrt/pull/557)

### Fixes
 - Fixed slice computations for large slices [#558](https://github.com/onnx/onnx-tensorrt/pull/558)

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

