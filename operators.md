# Supported ONNX Operators

TensorRT 6.0 supports operators up to Opset 10. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

TensorRT supports the following ONNX data types: FLOAT32, FLOAT16, INT32, INT8, INT64*

\*TensorRT will attempt to cast down INT64 to INT32 where possible. If not possible, TensorRT will throw an error.

## Operator Support Matrix

| ONNX Operator                   |  Supported   | TRT Version   | Notes                                                                      |
|---------------------------------|--------------|---------------|----------------------------------------------------------------------------|
| Abs                             | Y            |               |                                                                            |
| Acos                            | Y            | 5.1           |                                                                            |
| Acosh                           | Y            | 5.1           |                                                                            |
| Add                             | Y            |               |                                                                            |
| And                             | N            | N/A           |                                                                            |
| Argmax                          | Y            | 4.0           |                                                                            |
| Argmin                          | Y            | 4.0           |                                                                            |
| Asin                            | Y            | 5.1           |                                                                            |
| Asinh                           | Y            | 5.1           |                                                                            |
| Atan                            | Y            | 5.1           |                                                                            |
| Atanh                           | Y            | 5.1           |                                                                            |
| AveragePool                     | Y            |               | 2D/3D pooling only.                                                        |
| BatchNormalization              | Y            |               |                                                                            |
| Cast                            | Y            | 5.1           | Only FP16->FP32 casts are supported                                        |
| Ceil                            | Y            |               |                                                                            |
| Clip                            | Y            |               |                                                                            |
| Compress                        | N            | N/A           |                                                                            |
| Concat                          | Y            |               |                                                                            |
| Constant                        | Y            |               |                                                                            |
| ConstantOfShape                 | N            | N/A           |                                                                            |
| Conv                            | Y            |               | 2D/3D convolution only. Convolution weights must be an initializer.        |
| ConvInteger                     | N            | N/A           |                                                                            |
| ConvTranspose                   | Y            |               | 2D/3D deconvolution only. Deconvolution weights must be an initializer.    |
| Cos                             | Y            | 5.1           |                                                                            |
| Cosh                            | Y            |               |                                                                            |
| DepthToSpace                    | Y            | 4.0           |                                                                            |
| DequantizeLinear                | N            | N/A           |                                                                            |
| Div                             | Y            |               |                                                                            |
| Dropout                         | Y            |               |                                                                            |
| Elu                             | Y            |               |                                                                            |
| Equal                           | N            | N/A           |                                                                            |
| Erf                             | N            | N/A           |                                                                            |
| Exp                             | Y            |               |                                                                            |
| Expand                          | N            | N/A           |                                                                            |
| EyeLike                         | N            | N/A           |                                                                            |
| Flatten                         | Y            |               |                                                                            |
| Floor                           | Y            |               |                                                                            |
| GRU                             | N            | N/A           |                                                                            |
| Gather                          | Y            | 4.0           |                                                                            |
| Gemm                            | Y            |               |                                                                            |
| GlobalAveragePool               | Y            |               | 2D/3D pooling only.                                                        |
| GlobalLpPool                    | N            | N/A           |                                                                            |
| GlobalMaxPool                   | Y            |               | 2D/3D pooling only.                                                        |
| Greater                         | N            | N/A           |                                                                            |
| HardSigmoid                     | Y            |               |                                                                            |
| Hardmax                         | N            | N/A           |                                                                            |
| Identity                        | Y            |               |                                                                            |
| If                              | N            | N/A           |                                                                            |
| InstanceNormalization           | Y            |               |                                                                            |
| IsInf                           | N            | N/A           |                                                                            |
| IsNaN                           | N            | N/A           |                                                                            |
| LRN                             | Y            |               |                                                                            |
| LSTM                            | N            | N/A           |                                                                            |
| LeakyRelu                       | Y            |               |                                                                            |
| Less                            | N            | N/A           |                                                                            |
| Log                             | Y            |               |                                                                            |
| LogSoftmax                      | Y            |               |                                                                            |
| Loop                            | N            | N/A           |                                                                            |
| LpNormalization                 | N            | N/A           |                                                                            |
| LpPool                          | N            | N/A           |                                                                            |
| MatMul                          | Y            |               |                                                                            |
| MatMulInteger                   | N            | N/A           |                                                                            |
| Max                             | Y            |               |                                                                            |
| MaxPool                         | Y            |               | 2D/3D pooling only.                                                        |
| MaxRoiPool                      | N            | N/A           |                                                                            |
| MaxUnpool                       | N            | N/A           |                                                                            |
| Mean                            | Y            | 4.0           |                                                                            |
| Min                             | Y            |               |                                                                            |
| Mod                             | N            | N/A           |                                                                            |
| Mul                             | Y            |               |                                                                            |
| Multinomial                     | N            | N/A           |                                                                            |
| Neg                             | Y            |               |                                                                            |
| NonMaxSuppression               | N            | N/A           |                                                                            |
| NonZero                         | N            | N/A           |                                                                            |
| Not                             | N            | N/A           |                                                                            |
| OneHot                          | N            | N/A           |                                                                            |
| Or                              | N            | N/A           |                                                                            |
| PRelu                           | Y            | 6.0           |                                                                            |
| Pad                             | Y            |               | Zero-constant padding only.                                                |
| Pow                             | Y            |               |                                                                            |
| QLinearConv                     | N            | N/A           |                                                                            |
| QLinearMatMul                   | N            | N/A           |                                                                            |
| QuantizeLinear                  | N            | N/A           |                                                                            |
| RNN                             | N            | N/A           |                                                                            |
| RandomNormal                    | N            | N/A           |                                                                            |
| RandomNormalLike                | N            | N/A           |                                                                            |
| RandomUniform                   | N            | N/A           |                                                                            |
| RandomUniformLike               | N            | N/A           |                                                                            |
| Reciprocal                      | Y            |               |                                                                            |
| ReduceL1                        | Y            | 4.0           |                                                                            |
| ReduceL2                        | Y            | 4.0           |                                                                            |
| ReduceLogSum                    | Y            | 4.0           |                                                                            |
| ReduceLogSumExp                 | Y            | 4.0           |                                                                            |
| ReduceMax                       | Y            | 4.0           |                                                                            |
| ReduceMean                      | Y            | 4.0           |                                                                            |
| ReduceMin                       | Y            | 4.0           |                                                                            |
| ReduceProd                      | Y            | 4.0           |                                                                            |
| ReduceSum                       | Y            | 4.0           |                                                                            |
| ReduceSumSquare                 | Y            | 4.0           |                                                                            |
| Relu                            | Y            |               |                                                                            |
| Resize                          | Y            | 6.0           |                                                                            |
| ReverseSequence                 | N            | N/A           |                                                                            |
| RoiAlign                        | N            | N/A           |                                                                            |
| Reshape                         | Y            |               |                                                                            |
| Scan                            | N            | N/A           |                                                                            |
| Scatter                         | N            | N/A           |                                                                            |
| Selu                            | Y            |               |                                                                            |
| Shape                           | Y            |               |                                                                            |
| Shrink                          | N            | N/A           |                                                                            |
| Sigmoid                         | Y            |               |                                                                            |
| Sign                            | N            | N/A           |                                                                            |
| Sin                             | Y            | 5.1           |                                                                            |
| Sinh                            | Y            | 5.1           |                                                                            |
| Size                            | N            | N/A           |                                                                            |
| Slice                           | Y            |               |                                                                            |
| Softmax                         | Y            |               |                                                                            |
| Softplus                        | Y            |               |                                                                            |
| Softsign                        | Y            |               |                                                                            |
| SpaceToDepth                    | Y            | 4.0           |                                                                            |
| Split                           | Y            |               |                                                                            |
| Sqrt                            | Y            |               |                                                                            |
| Squeeze                         | Y            | 4.0           |                                                                            |
| StringNormalizer                | N            | N/A           |                                                                            |
| Sub                             | Y            |               |                                                                            |
| Sum                             | Y            |               |                                                                            |
| Tan                             | Y            |               |                                                                            |
| Tanh                            | Y            | 5.1           |                                                                            |
| TfIdfVectorizer                 | N            | N/A           |                                                                            |
| ThresholdedRelu                 | Y            |               |                                                                            |
| Tile                            | N            | N/A           |                                                                            |
| TopK                            | Y            | 4.0           |                                                                            |
| Transpose                       | Y            |               |                                                                            |
| Unsqueeze                       | Y            |               |                                                                            |
| Upsample                        | Y            | 4.0           |                                                                            |
| Where                           | N            | N/A           |                                                                            |
| Xor                             | N            | N/A           |                                                                            |
