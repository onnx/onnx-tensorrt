<!--- SPDX-License-Identifier: Apache-2.0 -->

# Supported ONNX Operators

TensorRT 10.0 supports operators in the inclusive range of opset 9 to opset 20. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md). More details and limitations are documented in the chart below.

TensorRT supports the following ONNX data types: DOUBLE, FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, FP8, INT8, INT4, UINT8, and BOOL

> Note: There is limited support for DOUBLE type. TensorRT will attempt to cast DOUBLE down to FLOAT, clamping values to `+-FLT_MAX` if necessary.
> Note: INT8, INT4, and FP8 are treated as `Quantized Types` in TensorRT, where support is available only through quantization from a floating-point type with higher precision.
> Note: UINT8 is only supported as network input or output tensor types.

## Operator Support Matrix

| Operator                  | Supported  | Supported Types | Restrictions                                                                                                           |
|---------------------------|------------|-----------------|------------------------------------------------------------------------------------------------------------------------|
| Abs                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Acos                      | Y          | FP32, FP16, BF16 |
| Acosh                     | Y          | FP32, FP16, BF16 |
| Add                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| AffineGrid                | N          |
| And                       | Y          | BOOL |
| ArgMax                    | Y          | FP32, FP16, BF16, INT32, INT64 |
| ArgMin                    | Y          | FP32, FP16, BF16, INT32, INT64 |
| Asin                      | Y          | FP32, FP16, BF16 |
| Asinh                     | Y          | FP32, FP16, BF16 |
| Atan                      | Y          | FP32, FP16, BF16 |
| Atanh                     | Y          | FP32, FP16, BF16 |
| AveragePool               | Y          | FP32, FP16, BF16 | 2D or 3D Pooling only. `dilations` must be empty or all ones                                                                                                              |
| BatchNormalization        | Y          | FP32, FP16, BF16 |
| Bernoulli                 | N          |
| BitShift                  | N          |
| BitwiseAnd                | N          |
| BitwiseNot                | N          |
| BitwiseOr                 | N          |
| BitwiseXor                | N          |
| BlackmanWindow            | N          |
| Cast                      | Y          | FP32, FP16, BF16, INT32, INT64, , UINT8, BOOL |                                                                                                       |
| CastLike                  | Y          | FP32, FP16, BF16, INT32, INT64, , UINT8, BOOL |                                                                                                       |
| Ceil                      | Y          | FP32, FP16, BF16 |
| Col2Im                    | N          |
| Celu                      | Y          | FP32, FP16, BF16 |
| CenterCropPad             | N          |
| Clip                      | Y          | FP32, FP16, BF16 |                                                                                        |
| Compress                  | N          |
| Concat                    | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| ConcatFromSequence        | N          |
| Constant                  | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| ConstantOfShape           | Y          | FP32 |
| Conv                      | Y          | FP32, FP16, BF16 |
| ConvInteger               | N          |
| ConvTranspose             | Y          | FP32, FP16, BF16 |
| Cos                       | Y          | FP32, FP16, BF16 |
| Cosh                      | Y          | FP32, FP16, BF16 |
| CumSum                    | Y          | FP32, FP16, BF16 | `axis` must be an initializer                                                                                                            |
| DFT                       | N          |
| DepthToSpace              | Y          | FP32, FP16, BF16, INT32, INT64 |
| DequantizeLinear          | Y          | INT8, FP8, INT4 | `x_zero_point` must be zero                                                                                    |
| Det                       | N          |
| Div                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Dropout                   | Y          | FP32, FP16, BF16 |
| DynamicQuantizeLinear     | N          |
| Einsum                    | Y          | FP32, FP16, BF16 |
| Elu                       | Y          | FP32, FP16, BF16 |
| Equal                     | Y          | FP32, FP16, BF16, INT32, INT64 |
| Erf                       | Y          | FP32, FP16, BF16 |
| Exp                       | Y          | FP32, FP16, BF16 |
| Expand                    | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| EyeLike                   | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Flatten                   | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Floor                     | Y          | FP32, FP16, BF16|
| Gather                    | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| GatherElements            | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| GatherND                  | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Gelu                      | Y          | FP32, FP16, BF16, INT8, INT32, INT64 |
| Gemm                      | Y          | FP32, FP16, BF16 |
| GlobalAveragePool         | Y          | FP32, FP16, BF16 |
| GlobalLpPool              | Y          | FP32, FP16, BF16 |
| GlobalMaxPool             | Y          | FP32, FP16, BF16 |
| Greater                   | Y          | FP32, FP16, BF16, INT32, INT64 |
| GreaterOrEqual            | Y          | FP32, FP16, BF16, INT32, INT64 |
| GridSample                | Y          | FP32, FP16 |
| GroupNormalization        | Y          | FP32, FP16, BF16 |
| GRU                       | Y          | FP32, FP16, BF16 | For bidirectional GRUs, activation functions must be the same for both the forward and reverse pass
| HammingWindow             | N          |
| HannWindow                | N          |
| HardSigmoid               | Y          | FP32, FP16, BF16 |
| HardSwish                 | Y          | FP32, FP16, BF16 |
| Hardmax                   | Y          | FP32, FP16, BF16 | `axis` dimension of input must be a build-time constant
| Identity                  | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| If                        | Y          | FP32, FP16, BF16, INT32, INT64, BOOL | Output tensors of the two conditional branches must have broadcastable shapes, and must have different names
| ImageScaler               | Y          | FP32, FP16, BF16|
| ImageDecoder              | N          |
| InstanceNormalization     | Y          | FP32, FP16, BF16 |
| IsInf                     | Y          | FP32, FP16, BF16 |
| IsNaN                     | Y          | FP32, FP16, BF16, INT32, INT64 |
| LayerNormalization        | Y          | FP32, FP16, BF16
| LeakyRelu                 | Y          | FP32, FP16, BF16 |
| Less                      | Y          | FP32, FP16, BF16, INT32, INT64 |
| LessOrEqual               | Y          | FP32, FP16, BF16, INT32, INT64 |
| Log                       | Y          | FP32, FP16, BF16 |
| LogSoftmax                | Y          | FP32, FP16, BF16 |
| Loop                      | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| LRN                       | Y          | FP32, FP16, BF16 |
| LSTM                      | Y          | FP32, FP16, BF16 | For bidirectional LSTMs, activation functions must be the same for both the forward and reverse pass
| LpNormalization           | Y          | FP32, FP16, BF16 |
| LpPool                    | Y          | FP32, FP16, BF16 | `dilations` must be empty or all ones
| MatMul                    | Y          | FP32, FP16, BF16 |
| MatMulInteger             | N          |
| Max                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| MaxPool                   | Y          | FP32, FP16, BF16 | 2D or 3D pooling only. `Indices` output tensor unsupported. `dilations` must be empty or all ones
| MaxRoiPool                | N          |
| MaxUnpool                 | N          |
| Mean                      | Y          | FP32, FP16, BF16, FP8, INT32, INT64 |
| MeanVarianceNormalization | Y          | FP32, FP16, BF16|
| MelWeightMatrix           | N          |
| Min                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Mish                      | Y          | FP32, FP16 |
| Mod                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Mul                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Multinomial               | N          |
| Neg                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| NegativeLogLikelihoodLoss | N          |
| NonMaxSuppression         | Y          | FP32, FP16 |
| NonZero                   | Y          | FP32, FP16
| Not                       | Y          | BOOL |
| OneHot                    | Y          | FP32, FP16, BF16, INT32, INT64, BOOL | `depth` must be a build-time constant
| Optional                  | N          |
| OptionalGetElement        | N          |
| OptionalHasElement        | N          |
| Or                        | Y          | BOOL |
| Pad                       | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ParametricSoftplus        | Y          | FP32, FP16, BF16 |
| Pow                       | Y          | FP32, FP16, BF16 |
| PRelu                     | Y          | FP32, FP16, BF16 |
| QLinearConv               | N          |
| QLinearMatMul             | N          |
| QuantizeLinear            | Y          | FP32, FP16, BF16 | `y_zero_point` must be 0                                                                   |
| RandomNormal              | Y          | FP32, FP16, BF16 | `seed` value is ignored by TensorRT
| RandomNormalLike          | Y          | FP32, FP16, BF16 | `seed` value is ignored by TensorRT
| RandomUniform             | Y          | FP32, FP16, BF16 | `seed` value is ignored by TensorRT
| RandomUniformLike         | Y          | FP32, FP16, BF16 | `seed` value is ignored by TensorRT
| Range                     | Y          | FP32, FP16, BF16, INT32, INT64 |
| Reciprocal                | Y          | FP32, FP16, BF16 |
| ReduceL1                  | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceL2                  | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceLogSum              | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceLogSumExp           | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceMax                 | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceMean                | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceMin                 | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceProd                | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceSum                 | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| ReduceSumSquare           | Y          | FP32, FP16, BF16, INT32, INT64 | `axes` must be an initializer |
| RegexFullMatch            | N          |
| Relu                      | Y          | FP32, FP16, BF16, INT32, INT64 |
| Reshape                   | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Resize                    | Y          | FP32, FP16, BF16 | Supported resize transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`.<br />Supported resize modes: `nearest`, `linear`.<br />Supported nearest modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`.<br />Supported aspect ratio policy: `stretch`.<br />When `scales` is a tensor input, `axes` must be an iota vector of length rank(input).<br />Antialiasing is not supported.|
| ReverseSequence           | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| RNN                       | Y          | FP32, FP16, BF16| For bidirectional RNNs, activation functions must be the same for both the forward and reverse pass
| RoiAlign                  | Y          | FP32, FP16 |
| Round                     | Y          | FP32, FP16, BF16 |
| STFT                      | N          |
| ScaledTanh                | Y          | FP32, FP16, BF16 |
| Scan                      | Y          | FP32, FP16, BF16|
| Scatter                   | Y          | FP32, FP16, BF16, INT32, INT64 |
| ScatterElements           | Y          | FP32, FP16, BF16, INT32, INT64 |
| ScatterND                 | Y          | FP32, FP16, BF16, INT32, INT64 | `reduction` is not supported
| Selu                      | Y          | FP32, FP16, BF16, |
| SequenceAt                | N          |
| SequenceConstruct         | N          |
| SequenceEmpty             | N          |
| SequenceErase             | N          |
| SequenceInsert            | N          |
| SequenceLength            | N          |
| SequenceMap               | N          |
| Shape                     | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Shrink                    | Y          | FP32, FP16, BF16, INT32, INT64 |
| Sigmoid                   | Y          | FP32, FP16, BF16 |
| Sign                      | Y          | FP32, FP16, BF16, INT32, INT64 |
| Sin                       | Y          | FP32, FP16, BF16 |
| Sinh                      | Y          | FP32, FP16, BF16 |
| Size                      | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Slice                     | Y          | FP32, FP16, BF16, INT32, INT64, BOOL | `axes` must be an initializer                                                                                                            |
| Softmax                   | Y          | FP32, FP16, BF16 |
| SoftmaxCrossEntropyLoss   | N          |
| Softplus                  | Y          | FP32, FP16, BF16 |
| Softsign                  | Y          | FP32, FP16, BF16 |
| SpaceToDepth              | Y          | FP32, FP16, BF16, INT32, INT64 |
| Split                     | Y          | FP32, FP16, BF16, INT32, INT64 BOOL |                                                                                                          |
| SplitToSequence           | N          |
| Sqrt                      | Y          | FP32, FP16, BF16 |
| Squeeze                   | Y          | FP32, FP16, BF16, INT32, INT64, BOOL | `axes` must be an initializer                                                                                                            |
| StringConcat              | N          |
| StringNormalizer          | N          |
| StringSplit               | N          |
| Sub                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Sum                       | Y          | FP32, FP16, BF16, INT32, INT64 |
| Tan                       | Y          | FP32, FP16, BF16 |
| Tanh                      | Y          | FP32, FP16, BF16 |
| TfIdfVectorizer           | N          |
| ThresholdedRelu           | Y          | FP32, FP16, BF16 |
| Tile                      | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| TopK                      | Y          | FP32, FP16, BF16, INT32, INT64 |
| Transpose                 | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Trilu                     | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Unique                    | N          |
| Unsqueeze                 | Y          | FP32, FP16, BF16, INT32, INT64, BOOL | `axes` must be a constant tensor                                                                                                         |
| Upsample                  | Y          | FP32, FP16, BF16 |
| Where                     | Y          | FP32, FP16, BF16, INT32, INT64, BOOL |
| Xor                       | Y          | BOOL
