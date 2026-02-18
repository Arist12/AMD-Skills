# BSNH Layout Unification

## The Problem

HuggingFace transformer models use **BNSH** (Batch, Num_heads, Seq, Head_dim)
format internally, but AITER flash attention and fused RoPE take **BSNH**
(Batch, Seq, Num_heads, Head_dim). This mismatch causes 3 materializing
transpose operations per attention layer:

```python
# HuggingFace Gemma attention path (BNSH):
q = q.view(B, S, H, D).transpose(1, 2)  # BSNH -> BNSH (materializes)
q = apply_rope(q, cos, sin)              # RoPE on BNSH
attn_out = attention(q, k, v)            # attention on BNSH
attn_out = attn_out.transpose(1, 2)      # BNSH -> BSNH (materializes)
```

For a model with 18 layers × 10 denoise steps, this is **540 materializing
transpose+contiguous operations** per inference. At bsz=1 where operations
are memory-bandwidth-bound, this can account for 10-15% of total latency.

## The Solution

Keep tensors in **BSNH format throughout** the entire forward pass:

```python
# Optimized path (BSNH end-to-end):
q = q.view(B, S, H, D)                    # Already BSNH, zero-cost view
# AITER fused RoPE operates on SBHD format — zero-cost .view() of BSNH when B=1
q = aiter.rope_cached_fwd_impl(q, cos, sin)
# AITER flash attention takes BSNH directly
attn_out = aiter.flash_attn_func(q, k, v)
# Output is already BSNH — direct reshape to [B, S, H*D]
output = attn_out.reshape(B, S, H * D)
```

## Impact

Measured on OpenPI Pi0 (3.5B, 18 layers, 10 denoise steps, bsz=1, MI355X):

| Metric | Before (BNSH) | After (BSNH) | Change |
|--------|---------------|---------------|--------|
| Per-step latency | 4.38ms | 3.73ms | -14.8% |
| Transposes per inference | 540 | 0 | -100% |
| Total E2E (10 steps) | 43.76ms | 37.27ms | -14.8% |

## Requirements

- AITER flash attention (`aiter.flash_attn_func`) — takes BSNH natively
- AITER fused RoPE (`aiter.rope_cached_fwd_impl`) — takes SBHD, which is a
  zero-cost `.view()` of BSNH when batch=1
- All QKV projections output directly to BSNH via `.view(B, S, H, D)` instead
  of `.view(B, S, H, D).transpose(1, 2)`

## Implementation Notes

1. **GQA (Grouped Query Attention)**: AITER flash attention handles GQA natively
   via `num_kv_heads < num_query_heads`. No need for `repeat_kv()` — AITER does
   the head expansion internally.

2. **Output projection**: After attention, reshape directly `[B, S, H, D] -> [B, S, H*D]`
   without transpose. This is a zero-cost view operation.

3. **KV cache format**: If using static KV cache, store in BSNH format too. The
   cache update is a simple slice write without any layout conversion.

4. **Compatibility**: The BSNH path requires replacing HuggingFace's
   `eager_attention_forward` function. Keep the original as a fallback for
   non-ROCm platforms.
