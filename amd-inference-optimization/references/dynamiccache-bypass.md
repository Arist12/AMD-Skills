# DynamicCache Bypass for CUDAGraph Capture

## The Problem

HuggingFace's `transformers.cache_utils.DynamicCache` allocates KV cache tensors
dynamically during `forward()`. Each call appends new key/value states, growing
the cache. This is fundamentally incompatible with CUDAGraph capture, which
requires all tensor allocations to be fixed at capture time.

Symptoms when capturing with DynamicCache:
- `RuntimeError: Trying to resize storage that is not resizable`
- Tensor ownership errors (graph replays with stale buffer references)
- Silent correctness failures (output all zeros or NaN)

## The Solution: Static KV Buffers

Replace DynamicCache with pre-allocated static tensors of fixed shape. For models
with a prefix-suffix architecture (e.g., VLM + action expert), the KV cache shape
is known at init time.

### Pattern for Prefix-Suffix Models (e.g., OpenPI pi0)

```python
class StaticKVCache:
    """Fixed-shape KV cache for CUDAGraph compatibility."""

    def __init__(self, num_layers, num_kv_heads, max_seq_len, head_dim, device, dtype):
        self.key_cache = [
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                       device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                       device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.seq_len = 0

    def update(self, layer_idx, key_states, value_states):
        """Write KV states into pre-allocated buffers (no allocation)."""
        bsz, num_heads, seq_len, head_dim = key_states.shape
        end = self.seq_len + seq_len
        self.key_cache[layer_idx][:, :, self.seq_len:end, :] = key_states
        self.value_cache[layer_idx][:, :, self.seq_len:end, :] = value_states
        return self.key_cache[layer_idx][:, :, :end, :], \
               self.value_cache[layer_idx][:, :, :end, :]

    def advance(self, seq_len):
        """Call after filling prefix to advance the write pointer."""
        self.seq_len += seq_len

    def reset_suffix(self):
        """Reset suffix portion for next denoise step (keep prefix)."""
        pass  # Suffix overwrites same region each step
```

### Replacing DynamicCache in Forward Pass

The key change: instead of passing `past_key_values=None` and getting back a
DynamicCache, pass the StaticKVCache and have the model write into it:

```python
# BEFORE (DynamicCache — breaks graph capture):
outputs, past_kv = model(inputs, past_key_values=None, use_cache=True)
# past_kv is a DynamicCache with dynamically-grown tensors

# AFTER (StaticKVCache — graph-compatible):
static_kv = StaticKVCache(num_layers=18, num_kv_heads=1,
                          max_seq_len=700, head_dim=256,
                          device='cuda', dtype=torch.bfloat16)
outputs = fast_forward(model, inputs, static_kv)
# All KV writes go into pre-allocated buffers
```

### Why This Enables Full-Pipeline Graph Capture

With StaticKVCache:
1. **Prefix phase**: Fills layers 0..N of the cache. No dynamic allocation.
2. **Denoise loop**: Each step reads the full cache, writes suffix KV into the
   same buffer region. Memory addresses are fixed.
3. **Graph capture**: Can now capture the entire pipeline (prefix + denoise)
   because no tensor shapes change between iterations.

## When to Apply

- Required for **Step B** of the incremental capture strategy (full-pipeline graph)
- NOT needed for **Step A** (denoise-loop-only capture, where prefix is computed
  outside the graph and KV cache is passed as a static input)
- Only needed when the model uses HuggingFace's `DynamicCache` or similar
  growable cache implementations

## Common Mistakes

1. **Passing DynamicCache into graph capture**: Will silently corrupt outputs
2. **Using `use_cache=True` with standard HF forward**: Creates DynamicCache internally
3. **Not pre-computing max_seq_len**: Must know the total sequence length (prefix + suffix) at init
