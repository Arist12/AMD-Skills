# Aiter Flash Attention Integration

## Tensor Layout (Critical)

aiter uses **`(batch, seqlen, nheads, headdim)`** — NOT `(batch, nheads, seqlen, headdim)`.

Most HuggingFace transformers models (including Gemma/PaliGemma) use `(batch, nheads, seqlen, headdim)` internally. You must transpose before and after calling aiter.

```python
# HuggingFace layout: (batch, nheads, seqlen, headdim)
# aiter layout:       (batch, seqlen, nheads, headdim)
q_aiter = query_states.transpose(1, 2).contiguous()
k_aiter = key_states.transpose(1, 2).contiguous()
v_aiter = value_states.transpose(1, 2).contiguous()

out = aiter.flash_attn_func(q_aiter, k_aiter, v_aiter, causal=is_causal,
                             softmax_scale=scaling)

# Transpose back and continue — output is already (batch, seqlen, nheads, headdim)
# which matches what GemmaAttention.forward expects after the attention call
attn_output = out  # already (batch, seqlen, nheads, headdim)
```

## `aiter.flash_attn_func` API

```python
aiter.flash_attn_func(
    q,                    # (batch, seqlen_q, nheads, headdim)
    k,                    # (batch, seqlen_k, nheads_k, headdim)
    v,                    # (batch, seqlen_k, nheads_k, headdim)
    dropout_p=0.0,        # 0.0 for inference
    softmax_scale=None,   # default: 1/sqrt(headdim)
    causal=False,         # True for autoregressive
    window_size=(-1,-1,0),# (-1,-1) = full context
    deterministic=True,
    return_lse=False,
)
# Returns: (batch, seqlen_q, nheads, headdim) — same layout as input q
```

Supports GQA/MQA natively: if `nheads_q` is a multiple of `nheads_k`, aiter handles the head
grouping internally. You do NOT need to call `repeat_kv()` before aiter — pass the original
un-repeated K and V.

## Drop-in Replacement for `eager_attention_forward`

For HuggingFace transformers models that use `eager_attention_forward` (like Gemma), replace:

```python
import aiter

AITER_AVAILABLE = hasattr(torch.version, "hip") and torch.version.hip is not None
try:
    if AITER_AVAILABLE:
        import aiter
except ImportError:
    AITER_AVAILABLE = False

def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs,
):
    if AITER_AVAILABLE:
        # aiter handles GQA natively — no need for repeat_kv
        q = query.transpose(1, 2).contiguous()   # (b, nheads, seq, hd) -> (b, seq, nheads, hd)
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        attn_output = aiter.flash_attn_func(
            q, k, v, dropout_p=dropout, softmax_scale=scaling, causal=True,
        )
        # attn_output is (b, seq, nheads, hd) — matches expected output layout
        return attn_output, None

    # Fallback: PyTorch SDPA (works on both NVIDIA and AMD)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key_states, value_states,
        attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
```

### Why this is faster

1. **No `repeat_kv`**: aiter handles GQA grouping in-kernel (saves a full tensor expansion)
2. **Fused kernel**: single CK kernel vs 3 separate ops (bmm + softmax + bmm)
3. **AMD-optimized**: CK flash attention is tuned for CDNA/RDNA wavefront width

## torch.compile Compatibility

aiter attention IS compilable — it does NOT cause graph breaks. You can safely use it inside
a `torch.compile`d function:

```python
@torch.compile(mode="max-autotune")
def compiled_forward(model, inputs):
    return model(inputs)  # aiter attention called inside — no graph break
```

If you see graph breaks, ensure you're using `aiter.flash_attn_func` (the function API), not
the class-based `aiter.FlashAttnFunc.apply` which may trigger custom autograd graph breaks.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong layout `(b, nheads, seq, hd)` | Shape mismatch error or wrong results | Transpose to `(b, seq, nheads, hd)` before calling |
| Calling `repeat_kv` before aiter | Wasted memory + slower | Remove `repeat_kv` — aiter handles GQA natively |
| Using `torch.ops.aiter.mha_fwd` directly | Returns tuple `(out, lse, ...)` | Use `aiter.flash_attn_func` which returns just the output tensor |
| Test crashes → declare "SDPA optimal" | No timing data collected | Fix the test code, don't abandon the optimization |
| Testing with tiny synthetic tensors | Results don't transfer to real model | Test with actual model dimensions |
