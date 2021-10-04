<!--- SPDX-License-Identifier: Apache-2.0 -->

# Supported ONNX Operators

TensorRT 8.2 supports operators up to Opset 13. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

TensorRT supports the following ONNX data types: DOUBLE, FLOAT32, FLOAT16, INT8, and BOOL

> Note: There is limited support for INT32, INT64, and DOUBLE types. TensorRT will attempt to cast down INT64 to INT32 and DOUBLE down to FLOAT, clamping values to `+-INT_MAX` or `+-FLT_MAX` if necessary.

See below for the support matrix of ONNX operators in ONNX-TensorRT.

## Operator Support Matrix

| Operator                  | Supported  | Supported Types | Restrictions                                                                                                           |
|---------------------------|------------|-----------------|------------------------------------------------------------------------------------------------------------------------|
| Abs                       | Y          | FP32, FP16, INT32 |
| Acos                      | Y          | FP32, FP16 |
| Acosh                     | Y          | FP32, FP16 |
| Add                       | Y          | FP32, FP16, INT32 |
| And                       | Y          | BOOL | 
| ArgMax                    | Y          | FP32, FP16 |
| ArgMin                    | Y          | FP32, FP16 |
| Asin                      | Y          | FP32, FP16 |
| Asinh                     | Y          | FP32, FP16 |
| Atan                      | Y          | FP32, FP16 |
| Atanh                     | Y          | FP32, FP16 |
| AveragePool               | Y          | FP32, FP16, INT8, INT32 | 2D or 3D Pooling only                                                                                                                    |
| BatchNormalization        | Y          | FP32, FP16 |
| BitShift                  | N          |
| Cast                      | Y          | FP32, FP16, INT32, INT8, BOOL |                                                                                                       |
| Ceil                      | Y          | FP32, FP16 |
| Celu                      | Y          | FP32, FP16 |
| Clip                      | Y          | FP32, FP16, INT8 | `min` and `max` clip values must be initializers                                                                                         |
| Compress                  | N          |
| Concat                    | Y          | FP32, FP16, INT32, INT8, BOOL |
| ConcatFromSequence        | N          |
| Constant                  | Y          | FP32, FP16, INT32, INT8, BOOL |
| ConstantOfShape           | Y          | FP32 |
| Conv                      | Y          | FP32, FP16, INT8 | 2D or 3D convolutions only                                                                                                               |
| ConvInteger               | N          |
| ConvTranspose             | Y          | FP32, FP16, INT8 | 2D or 3D deconvolutions only\. Weights `W` must be an initializer                                                                        |
| Cos                       | Y          | FP32, FP16 |
| Cosh                      | Y          | FP32, FP16 |
| CumSum                    | Y          | FP32, FP16 | `axis` must be an initializer                                                                                                            |
| DepthToSpace              | Y          | FP32, FP16, INT32 |
| DequantizeLinear          | Y          | INT8 | `x_zero_point` must be zero                                                                                    |
| Det                       | N          |
| Div                       | Y          | FP32, FP16, INT32 |
| Dropout                   | Y          | FP32, FP16 |
| DynamicQuantizeLinear     | N          |
| Einsum                    | Y          | FP32, FP16 | Ellipsis and diagonal operations are not supported.
| Elu                       | Y          | FP32, FP16, INT8 |
| Equal                     | Y          | FP32, FP16, INT32 |
| Erf                       | Y          | FP32, FP16 |
| Exp                       | Y          | FP32, FP16 |
| Expand                    | Y          | FP32, FP16, INT32, BOOL |
| EyeLike                   | Y          | FP32, FP16, INT32, BOOL |
| Flatten                   | Y          | FP32, FP16, INT32, BOOL |
| Floor                     | Y          | FP32, FP16 |
| Gather                    | Y          | FP32, FP16, INT8, INT32 |
| GatherElements            | Y          | FP32, FP16, INT8, INT32 |
| GatherND                  | Y          | FP32, FP16, INT8, INT32 |
| Gemm                      | Y          | FP32, FP16, INT8 |
| GlobalAveragePool         | Y          | FP32, FP16, INT8 |
| GlobalLpPool              | Y          | FP32, FP16, INT8 |
| GlobalMaxPool             | Y          | FP32, FP16, INT8 |
| Greater                   | Y          | FP32, FP16, INT32 |
| GreaterOrEqual            | Y          | FP32, FP16, INT32 |
| GRU                       | Y          | FP32, FP16 | For bidirectional GRUs, activation functions must be the same for both the forward and reverse pass
| HardSigmoid               | Y          | FP32, FP16, INT8 |
| Hardmax                   | N          |
| Identity                  | Y          | FP32, FP16, INT32, INT8, BOOL |
| If                        | Y          | FP32, FP16, INT32, BOOL | Output tensors of the two conditional branches must have broadcastable shapes, and must have different names
| ImageScaler               | Y          | FP32, FP16 |
| InstanceNormalization     | Y          | FP32, FP16 | Scales `scale` and biases `B` must be initializers. Input rank must be >=3 & <=5                                                                                  |
| IsInf                     | N          |
| IsNaN                     | Y          | FP32, FP16, INT32 |
| LeakyRelu                 | Y          | FP32, FP16, INT8 |
| Less                      | Y          | FP32, FP16, INT32 |
| LessOrEqual               | Y          | FP32, FP16, INT32 |
| Log                       | Y          | FP32, FP16 |
| LogSoftmax                | Y          | FP32, FP16 |
| Loop                      | Y          | FP32, FP16, INT32, BOOL |
| LRN                       | Y          | FP32, FP16 |
| LSTM                      | Y          | FP32, FP16 | For bidirectional LSTMs, activation functions must be the same for both the forward and reverse pass
| LpNormalization           | Y          | FP32, FP16 |
| LpPool                    | Y          | FP32, FP16, INT8 |
| MatMul                    | Y          | FP32, FP16 |
| MatMulInteger             | N          |
| Max                       | Y          | FP32, FP16, INT32 |
| MaxPool                   | Y          | FP32, FP16, INT8 |
| MaxRoiPool                | N          |
| MaxUnpool                 | N          |
| Mean                      | Y          | FP32, FP16, INT32 |
| MeanVarianceNormalization | N          |
| Min                       | Y          | FP32, FP16, INT32 |
| Mod                       | N          |
| Mul                       | Y          | FP32, FP16, INT32 |
| Multinomial               | N          |
| Neg                       | Y          | FP32, FP16, INT32 |
| NegativeLogLikelihoodLoss | N          |
| NonMaxSuppression         | Y [EXPERIMENTAL] | FP32, FP16 | Inputs `max_output_boxes_per_class`, `iou_threshold`, and `score_threshold` must be initializers. Output has fixed shape and is padded to [`max_output_boxes_per_class`, 3].
| NonZero                   | N          |
| Not                       | Y          | BOOL |
| OneHot                    | N          |
| Or                        | Y          | BOOL |
| Pad                       | Y          | FP32, FP16, INT8, INT32 |
| ParametricSoftplus        | Y          | FP32, FP16, INT8 |
| Pow                       | Y          | FP32, FP16 |
| PRelu                     | Y          | FP32, FP16, INT8 |
| QLinearConv               | N          |
| QLinearMatMul             | N          |
| QuantizeLinear            | Y          | FP32, FP16 | `y_zero_point` must be 0                                                                   |
| RandomNormal              | N          |
| RandomNormalLike          | N          |
| RandomUniform             | Y          | FP32, FP16 |
| RandomUniformLike         | Y          | FP32, FP16 |
| Range                     | Y          | FP32, FP16, INT32 | Floating point inputs are only supported if `start`, `limit`, and `delta` inputs are initializers                                                 |
| Reciprocal                | N          |
| ReduceL1                  | Y          | FP32, FP16 |
| ReduceL2                  | Y          | FP32, FP16 |
| ReduceLogSum              | Y          | FP32, FP16 |
| ReduceLogSumExp           | Y          | FP32, FP16 |
| ReduceMax                 | Y          | FP32, FP16 |
| ReduceMean                | Y          | FP32, FP16 |
| ReduceMin                 | Y          | FP32, FP16 |
| ReduceProd                | Y          | FP32, FP16 |
| ReduceSum                 | Y          | FP32, FP16 |
| ReduceSumSquare           | Y          | FP32, FP16 |
| Relu                      | Y          | FP32, FP16, INT8 |
| Reshape                   | Y          | FP32, FP16, INT32, INT8, BOOL |
| Resize                    | Y          | FP32, FP16 | Supported resize transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`.<br />Supported resize modes: `nearest`, `linear`.<br />Supported nearest modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`   |
| ReverseSequence           | Y          | FP32, FP16 | Dynamic input shapes are unsupported
| RNN                       | Y          | FP32, FP16 | For bidirectional RNNs, activation functions must be the same for both the forward and reverse pass
| RoiAlign                  | N          |
| Round                     | Y          | FP32, FP16, INT8 |
| ScaledTanh                | Y          | FP32, FP16, INT8 |
| Scan                      | Y          | FP32, FP16 |
| Scatter                   | Y          | FP32, FP16, INT8, INT32 |
| ScatterElements           | Y          | FP32, FP16, INT8, INT32 |
| ScatterND                 | Y          | FP32, FP16, INT8, INT32 |
| Selu                      | Y          | FP32, FP16, INT8|
| SequenceAt                | N          |
| SequenceConstruct         | N          |
| SequenceEmpty             | N          |
| SequenceErase             | N          |
| SequenceInsert            | N          |
| SequenceLength            | N          |
| Shape                     | Y          | FP32, FP16, INT32, INT8, BOOL |
| Shrink                    | N          |
| Sigmoid                   | Y          | FP32, FP16, INT8 |
| Sign                      | Y          | FP32, FP16, INT8, INT32 |
| Sin                       | Y          | FP32, FP16 |
| Sinh                      | Y          | FP32, FP16 |
| Size                      | Y          | FP32, FP16, INT32, INT8, BOOL |
| Slice                     | Y          | FP32, FP16, INT32, INT8, BOOL | `axes` must be an initializer                                                                                                            |
| Softmax                   | Y          | FP32, FP16 |
| SoftmaxCrossEntropyLoss   | N          |
| Softplus                  | Y          | FP32, FP16, INT8 |
| Softsign                  | Y          | FP32, FP16, INT8 |
| SpaceToDepth              | Y          | FP32, FP16, INT32 |
| Split                     | Y          | FP32, FP16, INT32, BOOL | `split` must be an initializer                                                                                                           |
| SplitToSequence           | N          |
| Sqrt                      | Y          | FP32, FP16 |
| Squeeze                   | Y          | FP32, FP16, INT32, INT8, BOOL | `axes` must be an initializer                                                                                                            |
| StringNormalizer          | N          |
| Sub                       | Y          | FP32, FP16, INT32 |
| Sum                       | Y          | FP32, FP16, INT32 |
| Tan                       | Y          | FP32, FP16 |
| Tanh                      | Y          | FP32, FP16, INT8 |
| TfIdfVectorizer           | N          |
| ThresholdedRelu           | Y          | FP32, FP16, INT8 |
| Tile                      | Y          | FP32, FP16, INT32, BOOL |
| TopK                      | Y          | FP32, FP16 |
| Transpose                 | Y          | FP32, FP16, INT32, INT8, BOOL |
| Unique                    | N          |
| Unsqueeze                 | Y          | FP32, FP16, INT32, INT8, BOOL | `axes` must be a constant tensor                                                                                                         |
| Upsample                  | Y          | FP32, FP16 |
| Where                     | Y          | FP32, FP16, INT32, BOOL |
| Xor                       | N          |
