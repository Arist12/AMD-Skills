# CUDAGraph / HIP Graph Strategy on ROCm

## Table of Contents
- [Why Manual CUDAGraph](#why-manual-cudagraph)
- [Complete Manual Capture Pattern](#complete-manual-capture-pattern)
- [Dynamo RNG Patch for ROCm](#dynamo-rng-patch-for-rocm)
- [Memory Pool Management](#memory-pool-management)
- [Common Pitfalls](#common-pitfalls)

## Why Manual CUDAGraph

On ROCm, there are two CUDAGraph approaches. Only manual capture works reliably:

| Approach | ROCm Status | Notes |
|----------|-------------|-------|
| `torch.compile(mode="reduce-overhead")` | **BROKEN** (65x slowdown) | Inductor's internal graph capture fails on ROCm |
| `inductor_config.triton.cudagraphs = True` | **BROKEN** | Same mechanism, same failure |
| Manual `torch.cuda.CUDAGraph()` | **Works** | Full-call capture + replay |

Manual capture gives the largest single latency reduction (typically 30-50% of total).

## Complete Manual Capture Pattern

```python
import torch

class CUDAGraphWrapper:
    """Wraps a compiled model with manual CUDAGraph capture and replay."""

    def __init__(self, model, example_inputs):
        self.model = model
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.pool = torch.cuda.graph_pool_handle()

    def warmup(self, *args):
        """Run model once to trigger torch.compile JIT compilation."""
        with torch.no_grad():
            _ = self.model(*args)
        torch.cuda.synchronize()

    def capture(self, *static_inputs):
        """Capture the model's forward pass as a CUDA graph."""
        self.static_inputs = static_inputs

        # Warm up before capture
        self.warmup(*static_inputs)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self.pool):
            self.static_output = self.model(*self.static_inputs)

        torch.cuda.synchronize()

    def replay(self, *new_inputs):
        """Copy new inputs and replay the captured graph."""
        assert self.graph is not None, "Must call capture() first"

        # Copy new data into static input tensors (same memory addresses)
        for static, new in zip(self.static_inputs, new_inputs):
            static.copy_(new)

        # Replay the captured graph
        self.graph.replay()

        return self.static_output.clone()


# Usage:
model = torch.compile(model, mode="default")  # NOT "reduce-overhead"

wrapper = CUDAGraphWrapper(model, example_inputs)
wrapper.capture(*static_inputs)

# Inference loop
for batch in data:
    output = wrapper.replay(*batch)
```

### Requirements for CUDAGraph Capture

- All input tensors must be **pre-allocated** with fixed shapes and memory addresses
- No dynamic shapes within the captured region
- No CPU-GPU synchronization inside the captured region
- No Python control flow that varies between iterations
- No in-place operations on tensors outside the graph's memory pool

## Dynamo RNG Patch for ROCm

### Problem

When using `torch.compile` with CUDAGraph capture on ROCm, Dynamo's `preserve_global_state` context manager calls `torch.cuda.get_rng_state()`, which is forbidden during CUDA graph capture:

```
RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture
```

### Solution

Patch `preserve_global_state` to skip CUDA RNG state during graph capture:

```python
import contextlib
import random as _py_random
import torch
import torch._dynamo.utils as dynamo_utils

def patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture():
    """Patch Dynamo to skip CUDA RNG state during graph capture on ROCm."""

    @contextlib.contextmanager
    def patched_preserve_global_state(tx):
        # Save Python and CPU RNG state (always safe)
        py_rng_state = _py_random.getstate()
        torch_rng_state = torch.random.get_rng_state()

        # Save CUDA RNG state ONLY when not capturing
        cuda_rng_state = None
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            cuda_rng_state = torch.cuda.get_rng_state()

        try:
            yield
        finally:
            # Restore Python and CPU RNG state
            _py_random.setstate(py_rng_state)
            torch.random.set_rng_state(torch_rng_state)

            # Restore CUDA RNG state only if we saved it
            if cuda_rng_state is not None and not torch.cuda.is_current_stream_capturing():
                torch.cuda.set_rng_state(cuda_rng_state)

    # Apply the patch
    dynamo_utils.preserve_global_state = patched_preserve_global_state


# Call BEFORE any torch.compile() calls:
patch_dynamo_preserve_global_state_for_rocm_cudagraph_capture()
```

### When to Apply

- Call before `torch.compile()` and before any CUDAGraph capture
- Only needed on ROCm (harmless on CUDA but unnecessary)
- Check with: `torch.version.hip is not None`

## Memory Pool Management

Use a private memory pool for graph capture to avoid interference with other allocations:

```python
# Create a private pool
pool = torch.cuda.graph_pool_handle()

# All graph captures share the same pool
with torch.cuda.graph(graph1, pool=pool):
    out1 = model1(*inputs1)

with torch.cuda.graph(graph2, pool=pool):
    out2 = model2(*inputs2)
```

### Memory Considerations

- CUDAGraph capture allocates all intermediate tensors upfront
- Peak memory during capture = peak memory during replay (no variation)
- Use `torch.cuda.memory_stats()` to verify memory usage after capture
- If memory is tight, capture with `torch.cuda.empty_cache()` beforehand

## Common Pitfalls

### 1. Dynamic Shapes
```python
# BAD: Shape changes between iterations
for seq_len in varying_lengths:
    output = graph_wrapper.replay(input[:seq_len])  # WILL CRASH

# GOOD: Pad to fixed shape
MAX_SEQ_LEN = 512
padded_input = F.pad(input, (0, 0, 0, MAX_SEQ_LEN - input.shape[0]))
output = graph_wrapper.replay(padded_input)
```

### 2. Capturing Before Compilation
```python
# BAD: Capture before warm-up triggers compilation
model = torch.compile(model, mode="default")
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(*inputs)  # May capture compilation overhead!

# GOOD: Warm up first
model = torch.compile(model, mode="default")
_ = model(*inputs)  # Trigger compilation
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(*inputs)  # Captures only the optimized graph
```

### 3. Sync Points Inside Capture
```python
# BAD: Sync inside captured region
with torch.cuda.graph(graph):
    output = model(*inputs)
    torch.cuda.synchronize()  # FORBIDDEN during capture

# GOOD: Sync only outside capture
with torch.cuda.graph(graph):
    output = model(*inputs)
torch.cuda.synchronize()  # OK here
```

### 4. Multiple torch.compile Regions
If the model has multiple `torch.compile` regions, capture the entire pipeline in one graph, not individual regions. Each graph capture/replay has fixed overhead.
