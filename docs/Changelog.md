<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-TensorRT Changelog

# TensorRT 10.5 GA Release - 2024-10-1
For more details, see the 10.5 GA release notes.

- Added support for real-valued `STFT` operations
- Improved error handling in `IParser`

# TensorRT 10.4 GA Release - 2024-9-5
For more details, see the 10.4 GA release notes.

- Added support for tensor `axes` for `Pad` operations
- Added support for `BlackmanWindow`, `HammingWindow`, and `HannWindow` operations
- Improved error handling in `IParserRefitter`
- Fixed kernel shape inference in multi-input convolutions

# TensorRT 10.3 GA Release - 2024-8-7
For more details, see the 10.3 GA release notes.

- Added support for tensor `axes` inputs for `Slice` nodes
- Updated `ScatterElements` importer to use an updated plugin

# TensorRT 10.2 GA Release - 2024-7-10 
For more details, see the 10.2 GA release notes.

- Improved error handling with new macros and classes
- Minor changes to op importers for `GRU` and `Squeeze`

# TensorRT 10.1 GA Release - 2024-6-10
For more details, see the 10.1 GA release notes.

- Added `supportsModelV2` API
- Added support for `DeformConv` operation
- Added support for `PluginV3` TensorRT Plugins
- Marked all IParser and IParserRefitter APIs as `noexcept`
- Shape inputs can be passed to custom ops supported by `IPluginV3`-based plugins by indicating the input indices to be interpreted as shape inputs by a node attribute named `tensorrt_plugin_shape_input_indices`.

# TensorRT 10.0 GA Release - 2024-4-25
For more details, see the 10.0 GA release notes.

- Added support for building with with `protobuf-lite`
- Fixed issue when parsing and refitting models with nested `BatchNormalization` nodes
- Added support for empty inputs in custom plugin nodes

# TensorRT 10.0 EA Release - 2024-4-1
For more details, see the 10.0 EA release notes.

- Added new class `IParserRefitter` that can be used to refit a TensorRT engine with the weights of an ONNX model
- `kNATIVE_INSTANCENORM` is now set to ON by default
- Added support for `IPluginV3` interfaces from TensorRT
- Added support for `INT4` quantization
- Added support for the `reduction` attribute in `ScatterElements`
- Added support for `wrap` padding mode in `Pad`

# TensorRT 9.3 GA Release - 2024-2-8
For more details, see the 9.3 GA release notes for the fixes since 9.2 GA.

- Added native support for `INT32` and `INT64` types for `ArgMin` and `ArgMax` nodes
- Fixed check for valid `zero_point` values in `QuantizeLinear` and `DequantizeLinear` nodes

# TensorRT 9.2 GA Release - 2023-11-8
For more details, see the 9.2 GA release notes for the fixes since 9.1 GA.

- Added support for `Hardmax`
- Fixed type inference for few operators to use native ONNX types

# TensorRT 9.1 GA Release - 2023-10-18
For more details, see the 9.1 GA release notes for the fixes since 9.0 GA.

- Added new `ErrorCode` enums to improve error logging
- Added new members to `IParserError` to improve error logging 
- Added static checkers when parsing nodes, resulting better reporting of errors

# TensorRT 9.0 GA Release - 2023-9-5
For more details, see the 9.0 GA release notes for the fixes since 9.0 EA.

- Added support for FP8 and BF16 datatypes.
- Fixed a bug that previously caused `If` nodes to fail import due to branch output size mismatch
- Improved support for importing ONNX Local Functions

# TensorRT 9.0 EA Release - 2023-8-4
For more details, see the 9.0 EA release notes for the fixes since 8.6 GA.

- Added support for INT64 data type. The ONNX parser no longer automatically casts INT64 to INT32.
- Added support for ONNX local functions when parsing ONNX models with the ONNX parser. 
- Breaking API Change: In TensorRT 9.0, due to the introduction of INT64 as a supported data type, ONNX models with INT64 I/O require INT64 bindings. Note that prior to this release, such models required INT32 bindings.
- Updated ONNX submodule to v1.14.0.

# TensorRT 8.6 GA Release - 2023-5-1
For more details, see the 8.6 GA release notes for the fixes since 8.6 EA.

- Renamed `kVERSION_COMPATIBLE` flag to `kNATIVE_INSTANCENORM`
- Added support for N-D `Trilu`
- Removed old LSTM importer
- Updated ONNX submodule to v1.13.1.

# TensorRT 8.6 EA Release - 2023-3-13

## Added

For more details, see the 8.6 EA release notes for new features added in TensorRT 8.6.

- Added support for `GroupNormalization`, `LayerNormalization`, `IsInf` operations
- Added support for INT32 input types for `Argmin`, `Argmax`, and `TopK`
- Added support for `ReverseSequence` operators with dynamic shapes
- Added support for `TopK` operators with dynamic `K` values
- Added `OnnxParserFlag` enum and `setFlag` interfaces to the ONNX parser to modify the default parsing behavior
- Added metadata tracking, now ONNX node metadata will be embedded into TensorRT layers

## Changed

- All cast operations will now use the new `CastLayer` over the pervious `IdentityLayer`. 

# TensorRT 8.5 GA Release - 2022-11-2

## Added

For more details, see the 8.5 GA release notes for new features added in TensorRT 8.5

- Added the `RandomNormal`, `RandomUniform`, `MeanVarianceNormalization`, `RoiAlign`, `Mod`, `Trilu`, `GridSample` and `NonZero` operations
- Added native support for the `NonMaxSuppression` operator
- Added support for importing ONNX networks with `UINT8` I/O types

## Fixed
- Fixed an issue with output padding with 1D deconv
- Fixed an issue when flattening 1D tensors
- Fixed an issue when parsing String attributes from TRT plugins
- Fixed an issue when importing `If` subgraphs with shared initializer names
- Fixied an issue when importing `Loop` subgraphs with `INT_MAX` trip counts

## Removed
- Removed `onnx2trt` binary. See the README.md for alternative binaries to run ONNX model with TensorRT.

## TensorRT 22.08 Release 2022-8-16
### Updated
- Updated TensorRT version to 8.4.2
- Updated ONNX submodule version to 1.12
- Updated operators support documentation

### Fixes
- Fixed handling of no-op `Flatten` operations
- Fixed `allowZero` logic in Reshape operation

### Deprecated
- Deprecated `onnx2trt` binary. This will be removed in the next release of TensorRT.

## TensorRT 8.4 GA Release - 2022-6-6

### Added 

For more details, see the 8.4 GA release notes for new features added in TensorRT 8.4

- Added native FP16 support for importing and manipulating FP16 initializers
- Added support for `Shrink`
- Added support for `Xor`
- Added dynamic shape support for `ArgMax` and `ArgMin`
- Added dynamic shape support for `Range` for floating point types

### Fixes
- Fixed an issue in tensor name scoping in ONNX models with nested subgraphs
- Fixed misc issues when dealing with empty tensors
- Fixed the operations in the `Celu` importer function
- Removed unnecessary reshapes in the `GEMM` importer function

## TensorRT 8.2 GA Release - 2021-11-23

### Added

See the 8.2 EA release notes for new features added in TensorRT 8.2.

### Fixes
- Removed duplicate constant layer checks that caused some performance regressions
- Fixed expand dynamic shape calculations
- Added parser-side checks for Scatter layer support 

## TensorRT 8.2 EA Release - 2021-10-04
### Added
- Added support for the following ONNX operators:
  - Einsum
  - IsNan
  - GatherND
  - Scatter
  - ScatterElements
  - ScatterND
  - Sign
  - Round

### Updated
- Updated `Gather` and `GatherElements` implementations to natively support negative indices
- Updated `Pad` layer to support ND padding, along with `edge` and `reflect` padding mode support
- Updated `If` layer with general performance improvements.

## TensorRT 8.0 Release - 2021-07-02
### Added
 - Rehauled resize operator, now fully supporting the following modes:
    - Coordinate Transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`
    - Modes: `nearest`, `linear`
    - Nearest Modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`
 - QuantizeLinear/DequantizeLinear updates:
   - Added support for tensor scales
   - Added support for per-axis quantization
 - Added support for multi-input ConvTranpose
 - Added support for generic 2D padding
 - Added experimental support for `NonMaxSuppression`

### Updated
 - Moved `RefitMap` API to core TensorRT.
 - Added Datatype column to [operators.md](https://github.com/onnx/onnx-tensorrt/blob/master/docs/operators.md)

## 21.05 Container Release - 2021-05-17
### Added
- Added library only build target [#659](https://github.com/onnx/onnx-tensorrt/pull/659)
- Added support for negative gather indices [#681](https://github.com/onnx/onnx-tensorrt/pull/681)
- Added support for `DOUBLE`-typed inputs and weights through downcast to float [#674](https://github.com/onnx/onnx-tensorrt/pull/674)
- Added support for optional plugin fields in FallbackPlugin path [#676](https://github.com/onnx/onnx-tensorrt/pull/676)

### Updated
- Updated license [#657](https://github.com/onnx/onnx-tensorrt/pull/657)

### Fixes
- Fixed index offset calculation in GatherElements [#675](https://github.com/onnx/onnx-tensorrt/pull/675)
- Clarified dynamic shape support for ReverseSequence

## 21.03 Container Release - 2021-03-09
### Added
- Added opset13 support for `SoftMax`, `LogSoftmax`, `Squeeze`, and `Unsqueeze`
- Added support for the `EyeLike` operator
- Added support for the `GatherElements` operator

### Fixes
### Removed

## 21.02 Container Release - 2021-01-18
### Added
 - Added support for the `ReverseSequence` operator [#590](https://github.com/onnx/onnx-tensorrt/pull/590)
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
