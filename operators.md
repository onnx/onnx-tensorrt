# Supported ONNX Operators

TensorRT 7.2 supports operators up to Opset 11. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

TensorRT supports the following ONNX data types: FLOAT32, FLOAT16, INT8, and BOOL

\*There is limited support for INT32 and INT64 types. TensorRT will attempt to cast down INT64 to INT32 where possible. If not possible, TensorRT will throw an error. See the [TensorRT layer support matrix](https://docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html#layers-precision-matrix) for more information on data type support.

## Operator Support Matrix

| Operator              | Supported? | Restrictions                                                                                                                           |
|-----------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Abs                   | Y          |
| Acos                  | Y          |
| Acosh                 | Y          |
| Add                   | Y          |
| And                   | Y          |
| ArgMax                | Y          |
| ArgMin                | Y          |
| Asin                  | Y          |
| Asinh                 | Y          |
| Atan                  | Y          |
| Atanh                 | Y          |
| AveragePool           | Y          | 2D or 3D Pooling only                                                                                                                  |
| BatchNormalization    | Y          |
| BitShift              | N          |
| Cast                  | Y          | Cast is only supported for TRT types                                                                                                   |
| Ceil                  | Y          |
| Clip                  | Y          | min and max clip values must be an initializer                                                                                         |
| Compress              | N          |
| Concat                | Y          |
| ConcatFromSequence N  |
| Constant              | Y          |
| ConstantOfShape       | Y          |
| Conv                  | Y          | 2D or 3D convolutions only                                                                                                             |
| ConvInteger           | N          |
| ConvTranspose         | Y          | 2D or 3D deconvolutions only\. Weights must be an initializer                                                                          |
| Cos                   | Y          |
| Cosh                  | Y          |
| CumSum                | N          |
| DepthToSpace          | Y          |
| DequantizeLinear      | Y          | Scales and zero\-point value must be initializers                                                                                      |
| Det                   | N          |
| Div                   | Y          |
| Dropout               | N          |
| Elu                   | Y          |
| Equal                 | Y          |
| Erf                   | Y          |
| Exp                   | Y          |
| Expand                | Y          |
| EyeLike               | N          |
| Flatten               | Y          |
| Floor                 | Y          |
| Gather                | Y          |
| GatherElements        | N          |
| GatherND              | N          |
| Gemm                  | Y          |
| GlobalAveragePool     | Y          |
| GlobalLpPool          | N          |
| GlobalMaxPool         | Y          |
| Greater               | Y          |
| GRU                   | Y          |
| HardSigmoid           | Y          |
| Hardmax               | N          |
| Identity              | Y          |
| If                    | N          |
| ImageScaler           | Y          |
| InstanceNormalization | Y          | Scales and biases must be an initializer                                                                                               |
| IsInf                 | N          |
| IsNaN                 | N          |
| LeakyRelu             | Y          |
| Less                  | Y          |
| Log                   | Y          |
| LogSoftmax            | Y          |
| Loop                  | Y          |
| LRN                   | Y          |
| LSTM                  | Y          |
| LpNormalization       | N          |
| LpPool                | N          |
| MatMul                | Y          |
| MatMulInteger         | N          |
| Max                   | Y          |
| MaxPool               | Y          |
| MaxRoiPool            | N          |
| MaxUnpool             | N          |
| Mean                  | Y          |
| Min                   | Y          |
| Mod                   | N          |
| Mul                   | Y          |
| Multinomial           | N          |
| Neg                   | Y          |
| NonMaxSuppression     | N          |
| NonZero               | N          |
| Not                   | Y          |
| OneHot                | N          |
| Or                    | Y          |
| Pad                   | Y          | Zero\-padding on last 2 dimensions only                                                                                                |
| ParametricSoftplus    | Y          |
| Pow                   | Y          |
| PRelu                 | Y          |
| QLinearConv           | N          |
| QLinearMatMul         | N          |
| QuantizeLinear        | Y          | Scales and zero\-point value must be initializers                                                                                      |
| RNN                   | N          |
| RandomNormal          | N          |
| RandomNormalLike      | N          |
| RandomUniform         | Y          |
| RandomUniformLike     | Y          |
| Range                 | Y          | Float inputs are only supported if start, limit and delta inputs are initializers                                                      |
| Reciprocal            | N          |
| ReduceL1              | Y          |
| ReduceL2              | Y          |
| ReduceLogSum          | Y          |
| ReduceLogSumExp       | Y          |
| ReduceMax             | Y          |
| ReduceMean            | Y          |
| ReduceMin             | Y          |
| ReduceProd            | Y          |
| ReduceSum             | Y          |
| ReduceSumSquare       | Y          |
| Relu                  | Y          |
| Reshape               | Y          |
| Resize                | Y          | Asymmetric coordinate transformation mode only\. Nearest or Linear resizing mode only\. "floor" mode only for resize\_mode attribute\. |
| ReverseSequence       | N          |
| RNN                   | Y          |
| RoiAlign              | N          |
| Round                 | N          |
| ScaledTanh            | Y          |
| Scan                  | Y          |
| Scatter               | N          |
| ScatterElements       | N          |
| ScatterND             | N          |
| Selu                  | Y          |
| SequenceAt            | N          |
| SequenceConstruct     | N          |
| SequenceEmpty         | N          |
| SequenceErase         | N          |
| SequenceInsert        | N          |
| SequenceLength        | N          |
| Shape                 | Y          |
| Shrink                | N          |
| Sigmoid               | Y          |
| Sign                  | N          |
| Sin                   | Y          |
| Sinh                  | Y          |
| Size                  | Y          |
| Slice                 | Y          | Slice axes must be an initializer                                                                                                      
| Softmax               | Y          |
| Softplus              | Y          |
| Softsign              | Y          |
| SpaceToDepth          | Y          |
| Split                 | Y          |
| SplitToSequence       | N          |
| Sqrt                  | Y          |
| Squeeze               | Y          |
| StringNormalizer      | N          |
| Sub                   | Y          |
| Sum                   | Y          |
| Tan                   | Y          |
| Tanh                  | Y          |
| TfIdfVectorizer       | N          |
| ThresholdedRelu       | Y          |
| Tile                  | Y          |
| TopK                  | Y          |
| Transpose             | Y          |
| Unique                | N          |
| Unsqueeze             | Y          |
| Upsample              | Y          |
| Where                 | Y          |
| Xor                   | N          |
