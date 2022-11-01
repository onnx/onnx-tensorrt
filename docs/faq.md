<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-TensorRT FAQ

For all uses we recommend installing the following tools:
* [ONNX-Graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
* [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)

## How do I import and run an ONNX model through TensorRT?

There are currently two officially supported tools for users to quickly check if an ONNX model can parse and build into a TensorRT engine from an ONNX file.

For C++ users, there is the [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) binary that is typically found in the `<tensorrt_root_dir>/bin` directory. The basic command of running an ONNX model is:

`trtexec --onnx=model.onnx`

Refer to the link or run `trtexec -h` for more information on CLI options.

For Python users, there is the [polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) tool. The basic command for running an onnx model is:

`polygraphy run model.onnx --trt`

Refer to the link or run `polygraphy run -h` for more information on CLI options.

## Common Assertion Errors

### `inputs.at(0) must be an initializer!` or `inputs.at(0).is_weights()`

This is a common error seen when importing some ONNX models into TensorRT. Currently for some TensorRT layers (such as TopK and Padding), some attributes are required to be graph-level constants. We’ve seen some examples in the past where some convertors will insert a subgraph to compute these constants rather than use an initializer for these nodes. In the majority of these cases, constant-folding these subgraphs will result in an ONNX model that can be imported by TensorRT. Polygraphy's surgeon tool provides a constant folding function. It can be run through:

`polygraphy surgeon sanitize model.onnx --fold-constants --output model_folded.onnx`

### `Network must have at least one output!`

This is a generic error which is seen when there was an issue parsing the ONNX model. To better root cause where the error has occurred, re-run the parsing step with verbose logging to better understand where the parser failed.

### `getPluginCreator() could not find Plugin <operator name> version 1`

This is an error stating that onnx-tensorrt does not have an import function defined for a particular operator. The TensorRT team is continuously working on improving the operator coverage in onnx-tensorrt. Feel free to open an issue on any unsupported operators that you come across in your models. For more information on how to write implementation for unsupported operators yourself, see the `custom layer support` section below.

## Custom Layer Support

Custom layer support in onnx-tensorrt is done through TensorRT plugins. Any custom plugins must be registered with TensorRT’s plugin registry in order for it to be visible to the onnx-tensorrt parser.

For writing a plugin for a custom ONNX operator, the quickest way to do so without modifying the parser code is by utilizing the `fallbackPluginImporter` function. As long as the inputs, outputs, and attributes of your custom operator are consistent with those of your plugin, the ONNX-TensorRT parser will do the mapping for you. You can refer to [this blog post](https://developer.nvidia.com/blog/estimating-depth-beyond-2d-using-custom-layers-on-tensorrt-and-onnx-models/) on how to write a plugin for a custom ONNX operator.

For writing a plugin for existing ONNX operators that requires modification of the parser code, you can refer to the InstanceNormalization import function and the [corresponding plugin implementation](https://github.com/NVIDIA/TensorRT/tree/main/plugin/instanceNormalizationPlugin) in the main TensorRT repository.

## Quantized Operator Support

As of the latest release version of TensorRT, the only two ONNX quantizing operators we support are the opset 11 definitions of `QuantizeLinear` and `DequantizeLinear`. We are currently working to expand the ability for TensorRT to import QAT and PTQ ONNX networks in future releases.
