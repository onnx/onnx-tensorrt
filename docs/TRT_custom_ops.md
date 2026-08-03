<!--- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT Custom Operator Schemas

## TRT_QuantizedAttention

Custom op for quantized Attention based on ONNX [Attention-23](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Attention). The common attributes, inputs, and outputs between this op and the standard Attention op follow the same rules as in the standard spec; only the ones unique to this custom op are followed by a description. This op supports the same padded and packed/ragged tensor forms as `TRT_Attention`.

### Attributes

<dl>
<dt><tt>query_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the Q tensor and output. Controls Q independently of KV.</dd>
<dt><tt>kv_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the K and V tensors. Controls KV independently of Q.</dd>
<dt><tt>is_causal</tt></dt>
<dt><tt>kv_num_heads</tt></dt>
<dt><tt>q_num_heads</tt></dt>
<dt><tt>scale</tt></dt>
<dt><tt>normalization_quantize_to_type - INT</tt></dt>
<dd>fp8 (default) or int8. The quantization type for Q/DQ between normalization and BMM2.</dd>
<dt><tt>TRT_decomposable - INT</tt></dt>
<dd>Default is 0.</dd>
<dt><tt>TRT_normalization_op - STRING</tt></dt>
<dd>"softmax" (default) or "none".</dd>
</dl>

See the `TRT_Attention` tensor format notation for the meaning of `"padded_bhnd"` and `"packed_nhd"`.

### Inputs (5-7)

<dl>
<dt><tt>Q</tt> (index 0)</dt>
<dt><tt>K</tt> (index 1)</dt>
<dt><tt>V</tt> (index 2)</dt>
<dt><tt>attn_mask</tt> (optional, index 3)</dt>
<dt><tt>normalization_quantize_scale</tt> (optional, index 4)</dt>
<dd>Scalar tensor of fp32, fp16, or bf16 type. The quantization scale for Q/DQ between normalization and BMM2.</dd>
<dt><tt>query_lengths</tt> (optional, index 5) - INT32</dt>
<dd>1D tensor of shape [batchSize + 1] containing cumulative token counts for the query sequence, starting with 0, e.g. [0, len_0, len_0 + len_1, ...]. Must be provided when <tt>query_form</tt> is <tt>"packed_nhd"</tt>.</dd>
<dt><tt>kv_lengths</tt> (optional, index 6) - INT32</dt>
<dd>1D tensor whose shape depends on <tt>kv_form</tt>: when <tt>"packed_nhd"</tt>, shape is [batchSize + 1] containing cumulative token counts starting with 0, e.g. [0, len_0, len_0 + len_1, ...] and this input must be provided; when <tt>"padded_bhnd"</tt>, shape is [batchSize] containing per-batch sequence lengths and this input is optional, e.g. [len_0, len_1, ...].</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
</dl>

## TRT_Attention

Custom op for attention with packed (ragged) batching support. This op mirrors TensorRT's `IAttention` API directly:
Q, K, and V are passed to TensorRT without implicit reshaping or scaling. The caller must provide tensors in the
shape required by the selected IO forms.

**Tensor format notation:** `"padded_bhnd"` denotes 4D tensors with shape `[B, H, N, D]` and `"packed_nhd"` denotes 3D tensors with shape `[T, H, D]`, where B=batch, H=num\_heads, N=sequence\_length, D=head\_size, and T=total\_tokens (sum of all sequence lengths in the batch). This notation follows the ONNX Attention-23 spec; the TRT C++ API docs use `[b, d, s, h]` for the same layout (`d`=num\_heads, `s`=seq\_len, `h`=head\_size).

### Attributes

<dl>
<dt><tt>query_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the Q tensor and output. Controls Q independently of KV.</dd>
<dt><tt>kv_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the K and V tensors. Controls KV independently of Q.</dd>
<dt><tt>causal_kind - STRING</tt></dt>
<dd>"none" (default), "upper_left", or "lower_right". Sets the implicit causal mask kind.</dd>
<dt><tt>TRT_decomposable - INT</tt></dt>
<dd>0 (default) or 1. When set to 1, TensorRT may decompose the attention into multiple kernels if no fused kernel is supported.</dd>
<dt><tt>TRT_normalization_op - STRING</tt></dt>
<dd>"softmax" (default) or "none".</dd>
<dt><tt>nb_rank - INT</tt></dt>
<dd>Number of ranks for multi-device attention execution. Default is 1.</dd>
</dl>

The `query_form` and `kv_form` attributes are independent: Q and KV may use different forms (e.g., packed Q with padded KV for LLM context-phase attention).

### Inputs (3-6)

<dl>
<dt><tt>Q</tt> (index 0)</dt>
<dt><tt>K</tt> (index 1)</dt>
<dt><tt>V</tt> (index 2)</dt>
<dt><tt>attn_mask</tt> (optional, index 3)</dt>
<dd>Mask tensor with rank at most 4. A BOOL mask uses true values for positions that may attend; other mask values are added to the BMM1 output.</dd>
<dt><tt>query_lengths</tt> (optional, index 4) - INT32</dt>
<dd>1D tensor of shape [batchSize + 1] containing cumulative token counts for the query sequence, starting with 0, e.g. [0, len_0, len_0 + len_1, ...]. Must be provided when <tt>query_form</tt> is <tt>"packed_nhd"</tt>.</dd>
<dt><tt>kv_lengths</tt> (optional, index 5) - INT32</dt>
<dd>1D tensor whose shape depends on <tt>kv_form</tt>: when <tt>"packed_nhd"</tt>, shape is [batchSize + 1] containing cumulative token counts starting with 0, e.g. [0, len_0, len_0 + len_1, ...] and this input must be provided; when <tt>"padded_bhnd"</tt>, shape is [batchSize] containing per-batch sequence lengths and this input is optional, e.g. [len_0, len_1, ...].</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
<dd>Output tensor with the same shape, IO form, and data type as <tt>Q</tt>.</dd>
</dl>

### Restrictions

- `Q`, `K`, and `V` must already match the selected IO forms. Unlike ONNX `Attention`, `TRT_Attention` does not use `q_num_heads`, `kv_num_heads`, or `scale` to reshape or scale Q/K/V.
- `attn_mask` cannot be used together with `causal_kind` values other than `"none"`.
- When query and KV forms differ, Q and KV tensors may have different ranks. For example, with packed Q and padded KV: Q is `[T, numHeads, headSize]` while K/V are `[batchSize, numHeads, seqLen, headSize]`.

## TRT_KVCacheUpdate

Custom op for KV cache updates with packed update support. Based on the TensorScatter op used for KV cache updates in attention mechanisms. See the tensor format notation defined in the TRT\_Attention section above.

### Attributes

<dl>
<dt><tt>update_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the update tensor.</dd>
<dt><tt>mode - STRING</tt></dt>
<dd>"linear" (default).</dd>
<dt><tt>axis - INT</tt></dt>
<dd>Scatter axis in the past_cache tensor. Must be -2 (default), which is the sequence-length dimension in the [B, H, N, D] layout.</dd>
</dl>

### Inputs (2-4)

<dl>
<dt><tt>past_cache</tt> (index 0)</dt>
<dd>4D tensor of shape [B, H, N, D].</dd>
<dt><tt>update</tt> (index 1)</dt>
<dd>4D [B, H, N, D] for padded form or 3D [T, H, D] for packed form.</dd>
<dt><tt>write_indices</tt> (optional, index 2) - INT32</dt>
<dd>1D tensor of shape [batchSize].</dd>
<dt><tt>update_lengths</tt> (optional, index 3) - INT32</dt>
<dd>1D tensor of shape [batchSize + 1] containing cumulative token counts for the update sequence.</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
</dl>

### Restrictions

- `past_cache` must always be 4D `[B, H, N, D]`.
- `update` must be 4D `[B, H, N, D]` for padded form or 3D `[T, H, D]` for packed form.

## TRT_MoE

Custom op for Mixture-of-Experts (MoE) feed-forward layers. Maps to TensorRT's `IMoELayer` via `INetworkDefinition::addMoE`. Supports gated linear units with optional SiLU activation (SwiGLU), optional per-expert biases, and optional quantization of the down-projection activations.

### Attributes

<dl>
<dt><tt>activation_type - INT</tt></dt>
<dd>Activation function applied between the gate and up projections. 0 = no activation (kNONE), 1 = SiLU (kSILU). Default is 0.</dd>
<dt><tt>quantization_mode - INT</tt></dt>
<dd>0 = no quantization, 1 = static quantization, 2 = dynamic double quantization (dynamic_dblq). When non-zero, <tt>quantization_dtype</tt> must also be specified. Default is 0.</dd>
<dt><tt>quantization_dtype - INT</tt></dt>
<dd>TensorRT <tt>DataType</tt> enum value for the down-projection activation quantization type (e.g., kFP8, kFP4). Required when <tt>quantization_mode</tt> is non-zero. Default is kFP8.</dd>
<dt><tt>quantization_block_shape - LIST of INT (4 elements)</tt></dt>
<dd>Block shape <tt>[d0, d1, d2, d3]</tt> for dynamic double quantization. Required when <tt>quantization_mode</tt> is 2.</dd>
<dt><tt>dyn_q_output_scale_dtype - INT</tt></dt>
<dd>TensorRT <tt>DataType</tt> enum value for the output scale tensor in dynamic double quantization. Required when <tt>quantization_mode</tt> is 2. Default is kFLOAT.</dd>
<dt><tt>swiglu_limit - FLOAT</tt></dt>
<dd>Clamp limit for the SwiGLU activation. Only consulted when any of the three <tt>swiglu_*</tt> attributes are present. Default is +∞ (no clamping).</dd>
<dt><tt>swiglu_alpha - FLOAT</tt></dt>
<dd>Alpha scale for the SwiGLU activation. Default is 1.0.</dd>
<dt><tt>swiglu_beta - FLOAT</tt></dt>
<dd>Beta offset for the SwiGLU activation. Default is 0.0.</dd>
</dl>

### Inputs (6-10)

<dl>
<dt><tt>hiddenStates</tt> (index 0) - FP32/FP16/BF16</dt>
<dd>3D tensor of shape [batchSize, seqLen, hiddenSize]. Token embeddings to route through the MoE layer.</dd>
<dt><tt>selectedExpertsForTokens</tt> (index 1) - INT32</dt>
<dd>3D tensor of shape [batchSize, seqLen, topK]. Expert indices selected for each token.</dd>
<dt><tt>scoresForSelectedExperts</tt> (index 2) - FP32/FP16/BF16</dt>
<dd>3D tensor of shape [batchSize, seqLen, topK]. Routing weights for the selected experts.</dd>
<dt><tt>fcGateWeights</tt> (index 3) - FP32/FP16/BF16</dt>
<dd>3D tensor of shape [numExperts, hiddenSize, moeInterSize]. Gate projection weights.</dd>
<dt><tt>fcUpWeights</tt> (index 4) - FP32/FP16/BF16 </dt>
<dd>3D tensor of shape [numExperts, hiddenSize, moeInterSize]. Up projection weights.</dd>
<dt><tt>fcDownWeights</tt> (index 5) - FP32/FP16/BF16</dt>
<dd>3D tensor of shape [numExperts, moeInterSize, hiddenSize]. Down projection weights.</dd>
<dt><tt>fcGateBiases</tt> (optional, index 6) - FP32/FP16/BF16</dt>
<dd>Bias for the gate projection. Must be provided together with <tt>fcUpBiases</tt> and <tt>fcDownBiases</tt> — all three biases are required or none.</dd>
<dt><tt>fcUpBiases</tt> (optional, index 7) - FP32/FP16/BF16</dt>
<dd>Bias for the up projection. See <tt>fcGateBiases</tt>.</dd>
<dt><tt>fcDownBiases</tt> (optional, index 8) - FP32/FP16/BF16</dt>
<dd>Bias for the down projection. See <tt>fcGateBiases</tt>.</dd>
<dt><tt>fcDownActivationScale</tt> (optional, index 9) - FP8/FP4</dt>
<dd>Quantization scale for down-projection activations. Required when <tt>quantization_mode</tt> is non-zero.</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
<dd>3D tensor of shape [batchSize, seqLen, hiddenSize]. MoE layer output with the same shape and data type as <tt>hiddenStates</tt>.</dd>
</dl>

### Restrictions

- Biases must be provided for all three projections (gate, up, down) or for none.
- `fcDownActivationScale` (index 9) must be present when `quantization_mode` is non-zero; `quantization_dtype` must also be set in that case.
- `quantization_block_shape` and `dyn_q_output_scale_dtype` are required only when `quantization_mode` is 2.
