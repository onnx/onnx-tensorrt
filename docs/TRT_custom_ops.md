<!--- SPDX-License-Identifier: Apache-2.0 -->

# TensorRT Custom Operator Schemas

## TRT_QuantizedAttention

Custom op for quantized Attention based on ONNX [Attention-23](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Attention). The common attributes, inputs, and outputs between this op and the standard Attention op follow the same rules as in the standard spec; only the ones unique to this custom op are followed by a description.

### Attributes

<dl>
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

### Inputs (4 or 5)

<dl>
<dt><tt>Q</tt></dt>
<dt><tt>K</tt></dt>
<dt><tt>V</tt></dt>
<dt><tt>attn_mask</tt> (optional)</dt>
<dt><tt>normalization_quantize_scale</tt></dt>
<dd>Scalar tensor of fp32, fp16, or bf16 type. The quantization scale for Q/DQ between normalization and BMM2.</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
</dl>

## TRT_Attention

Custom op for attention with packed (ragged) batching support. Based on ONNX [Attention-23](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Attention), with additional attributes and inputs for the `kPACKED_NHD` IO form.

**Tensor format notation:** `"padded_bhnd"` denotes 4D tensors with shape `[B, H, N, D]` and `"packed_nhd"` denotes 3D tensors with shape `[T, H, D]`, where B=batch, H=num\_heads, N=sequence\_length, D=head\_size, and T=total\_tokens (sum of all sequence lengths in the batch). This notation follows the ONNX Attention-23 spec; the TRT C++ API docs use `[b, d, s, h]` for the same layout (`d`=num\_heads, `s`=seq\_len, `h`=head\_size).

### Attributes

<dl>
<dt><tt>query_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the Q tensor and output. Controls Q independently of KV.</dd>
<dt><tt>kv_form - STRING</tt></dt>
<dd>"padded_bhnd" (default) or "packed_nhd". Sets the IO form of the K and V tensors. Controls KV independently of Q.</dd>
<dt><tt>is_causal</tt></dt>
<dt><tt>scale</tt></dt>
<dt><tt>TRT_decomposable - INT</tt></dt>
<dd>Default is 0.</dd>
<dt><tt>TRT_normalization_op - STRING</tt></dt>
<dd>"softmax" (default) or "none".</dd>
<dt><tt>nb_rank - INT</tt></dt>
<dd>Default is 1.</dd>
</dl>

The `query_form` and `kv_form` attributes are independent: Q and KV may use different forms (e.g., packed Q with padded KV for LLM context-phase attention).

### Inputs (6-8)

<dl>
<dt><tt>Q</tt> (index 0)</dt>
<dt><tt>K</tt> (index 1)</dt>
<dt><tt>V</tt> (index 2)</dt>
<dt><tt>attn_mask</tt> (optional, index 3)</dt>
<dt><tt>past_key</tt> (optional, index 4) - unsupported</dt>
<dt><tt>past_value</tt> (optional, index 5) - unsupported</dt>
<dt><tt>query_lengths</tt> (optional, index 6) - INT32</dt>
<dd>1D tensor of shape [batchSize + 1] containing cumulative token counts for the query sequence.</dd>
<dt><tt>kv_lengths</tt> (optional, index 7) - INT32</dt>
<dd>1D tensor whose shape depends on <tt>kv_form</tt>: when <tt>"packed_nhd"</tt>, shape is [batchSize + 1] containing cumulative token counts; when <tt>"padded_bhnd"</tt>, shape is [batchSize] containing per-batch sequence lengths.</dd>
</dl>

### Outputs

<dl>
<dt><tt>Y</tt></dt>
</dl>

### Restrictions

- Attention mask (`attn_mask`) is not supported when `query_form` is `"packed_nhd"`.
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
