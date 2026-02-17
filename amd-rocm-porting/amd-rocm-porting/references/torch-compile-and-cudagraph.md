# torch.compile & CUDAGraph on ROCm

Complete configuration for torch.compile Inductor backend and CUDAGraph/HIP Graph capture on AMD
ROCm GPUs.

## Table of Contents

- [torch.compile Mode Selection](#torchcompile-mode-selection)
- [Inductor Configuration](#inductor-configuration)
- [Compile Safety Monkey-Patch](#compile-safety-monkey-patch)
- [Dynamo RNG Patch for HIP Graph](#dynamo-rng-patch-for-hip-graph)
- [Manual CUDAGraph Capture](#manual-cudagraph-capture)
- [Stream Capture Rules](#stream-capture-rules)
- [Stream Capture Detection](#stream-capture-detection)
- [Triton Kernel Considerations](#triton-kernel-considerations)
- [Debugging torch.compile on ROCm](#debugging-torchcompile-on-rocm)

## torch.compile Mode Selection

### The critical rule

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
compile_mode = "default" if is_rocm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

**Why**: `reduce-overhead` enables Inductor's internal CUDAGraph integration, which is broken on
ROCm. It causes Inductor to attempt automatic graph capture, resulting in up to **65x slowdown**
or hangs. Use `mode="default"` and capture graphs manually if needed.

### Compile options

```python
compile_options = {
    "mode": "default",       # NOT reduce-overhead
    "fullgraph": False,      # Allow graph breaks for flexibility
    "dynamic": True,         # Handle variable-length sequences
}
model = torch.compile(model, **compile_options)
```

- `fullgraph=False`: Allows graph breaks at unsupported operations. Safer for porting.
- `dynamic=True`: Generates code for variable shapes (avoids recompilation per shape).

### Environment variable override

```python
import os
default_mode = "default" if is_rocm else "reduce-overhead"
compile_mode = os.environ.get("TORCH_COMPILE_MODE", default_mode)
```

## Inductor Configuration

### Essential settings for ROCm

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# === CRITICAL: Disable broken features ===

# Inductor's internal cudagraph capture is broken on ROCm
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Memory planning triggers deep recursion on ROCm
inductor_config.memory_planning = False

# === GEMM backend selection ===

# ATen (rocBLAS) is faster than Triton for GEMM on AMD
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False

# === Triton kernel tuning ===

# Coordinate descent tuning for block sizes, num_warps
inductor_config.coordinate_descent_tuning = True

# Benchmark to pick fastest kernel per operation
inductor_config.benchmark_kernel = True

# === Fusion settings ===

# Reduce kernel launch overhead by fusing operations
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.aggressive_fusion = True
inductor_config.max_fusion_size = 128    # default is 64
inductor_config.shape_padding = True      # pad to power-of-2 for Triton

# Group small kernels together
inductor_config.group_fusion = True
inductor_config.triton.multi_kernel = 1

# === Inference optimization ===

# Freeze weights (constant-fold for inference)
inductor_config.freezing = True

# Reorder operations for memory locality
inductor_config.reorder_for_locality = True

# === Dynamo settings ===

dynamo_config.cache_size_limit = 128
dynamo_config.suppress_errors = False
```

### What each setting does

| Setting | Why on ROCm | Default | ROCm value |
|---|---|---|---|
| `triton.cudagraphs` | Inductor cudagraph capture broken | `True` | `False` |
| `triton.cudagraph_trees` | Same as above | `True` | `False` |
| `memory_planning` | Deep recursion crash | `True` | `False` |
| `max_autotune_gemm_backends` | rocBLAS faster than Triton GEMM | `"ATEN,TRITON"` | `"ATEN"` |
| `coordinate_descent_tuning` | Finds optimal Triton configs | `False` | `True` |
| `aggressive_fusion` | Fewer kernel launches | `False` | `True` |
| `freezing` | Weight constant-folding | `False` | `True` (inference) |

## Compile Safety Monkey-Patch

Prevent any code path from accidentally using `reduce-overhead`:

```python
_original_compile = torch.compile

def _safe_compile(model=None, **kwargs):
    """Override torch.compile to prevent reduce-overhead on ROCm."""
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_rocm and kwargs.get("mode") in (None, "reduce-overhead"):
        kwargs["mode"] = "default"
        print("[rocm-port] Overriding compile mode to 'default' (reduce-overhead broken on ROCm)")
    return _original_compile(model, **kwargs)

torch.compile = _safe_compile
```

## Dynamo RNG Patch for HIP Graph

### The problem

ROCm's HIP runtime forbids `torch.cuda.get_rng_state()` during stream capture.
`torch._dynamo.convert_frame.preserve_global_state` calls it unconditionally, causing
`RuntimeError` during CUDAGraph capture on ROCm.

### The fix

Skip CUDA RNG state save/restore when a stream capture is in progress:

```python
import functools
import torch
import torch._dynamo.convert_frame

def patch_dynamo_for_rocm_graph_capture():
    """
    Patch Dynamo's preserve_global_state to skip CUDA RNG during stream capture.
    Must be called BEFORE any CUDAGraph capture.
    """
    # Guard against double-patching
    if getattr(torch._dynamo.convert_frame, "_rocm_rng_patched", False):
        return
    torch._dynamo.convert_frame._rocm_rng_patched = True

    _orig = torch._dynamo.convert_frame.preserve_global_state

    def _preserve_skip_rng(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Save RNG state only when NOT capturing
            rng_state = None
            if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                try:
                    rng_state = torch.cuda.get_rng_state()
                except Exception:
                    pass

            try:
                return fn(*args, **kwargs)
            finally:
                # Restore only if we saved it
                if rng_state is not None:
                    try:
                        torch.cuda.set_rng_state(rng_state)
                    except Exception:
                        pass

        return wrapper

    torch._dynamo.convert_frame.preserve_global_state = _preserve_skip_rng
```

### When to apply

- Before any `torch.cuda.CUDAGraph()` capture
- Before any `torch.cuda.graph()` context manager
- Idempotent (safe to call multiple times due to guard)

## Manual CUDAGraph Capture

Since Inductor's automatic CUDAGraph is broken on ROCm, capture manually.

### Full workflow

```python
import torch

# Step 0: Apply patches
patch_dynamo_for_rocm_graph_capture()

# Step 1: Prepare model and inputs
model = torch.compile(model, mode="default")
model.eval()

# Static inputs (must be reused across capture and replay)
static_input = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)

# Step 2: Warm up (resolves all Dynamo compilation, Triton autotuning)
with torch.no_grad():
    for _ in range(5):
        _ = model(static_input)
torch.cuda.current_stream().synchronize()

# Step 3: Capture
pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph, pool=pool):
    static_output = model(static_input)

torch.cuda.current_stream().synchronize()

# Step 4: Replay (for benchmarking or inference)
# Copy new data into static_input, then replay
static_input.copy_(new_input)
graph.replay()
result = static_output  # updated in-place by graph replay
```

### Private pool isolation

Always use `graph_pool_handle()` to isolate graph memory from the regular allocator:

```python
pool = torch.cuda.graphs.graph_pool_handle()
```

This prevents interference between graph-captured allocations and normal allocations.

### Multiple graphs sharing a pool

```python
pool = torch.cuda.graphs.graph_pool_handle()
graph_a = torch.cuda.CUDAGraph()
graph_b = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph_a, pool=pool):
    out_a = model_a(input_a)

with torch.cuda.graph(graph_b, pool=pool):
    out_b = model_b(input_b)
```

## Stream Capture Rules

### Forbidden during capture

| Operation | Why | Workaround |
|---|---|---|
| `.item()` | Device→host sync | Cache before capture |
| `torch.cuda.synchronize()` | Device-wide sync | Use stream sync before capture |
| `print(tensor)` | Implicit `.item()` | Print shapes only |
| `if tensor.shape[0] > 1` | Dynamic branching | Pre-resolve during warmup |
| `torch.cuda.get_rng_state()` | HIP forbids during capture | Apply Dynamo RNG patch |
| Memory allocation of new sizes | Graph records fixed allocation pattern | Use same shapes always |

### Allowed during capture

| Operation | Notes |
|---|---|
| Kernel launches | Primary purpose of graphs |
| `torch.cuda.current_stream().synchronize()` | Stream-level only |
| In-place operations | `tensor.copy_()`, `tensor.fill_()` |
| Pre-allocated tensor writes | Static output tensors |

### Stream-level vs device-level sync

```python
# PREFERRED on ROCm (lower overhead)
torch.cuda.current_stream().synchronize()

# AVOID (device-wide sync, higher overhead)
torch.cuda.synchronize()
```

## Stream Capture Detection

Check if currently inside a graph capture to guard unsafe operations:

```python
def is_capturing():
    """Check if a HIP/CUDA graph capture is in progress."""
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False

# Usage: guard .item() calls
if not is_capturing():
    loss_val = loss.item()
    print(f"Loss: {loss_val}")
```

### Dynamo is_compiling detection

```python
def is_dynamo_compiling():
    """Check if Dynamo is currently tracing/compiling."""
    try:
        return bool(getattr(torch._dynamo, "is_compiling", lambda: False)())
    except Exception:
        return False
```

## Triton Kernel Considerations

### Numerical stability (tanh via exp)

ROCm Triton builds may lack a native `tanh` intrinsic. Implement via clamped exp:

```python
import triton
import triton.language as tl

@triton.jit
def gelu_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    # Clamp to avoid exp overflow → NaN
    inner = 0.7978845608 * (x + 0.044715 * x * x * x)
    inner = tl.maximum(tl.minimum(inner, 10.0), -10.0)
    exp_2x = tl.math.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)

    out = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)
```

### Float32 accumulation

Always accumulate in float32 for numerical stability, store back in bfloat16/float16:

```python
x = tl.load(ptr + offsets, mask=mask).to(tl.float32)  # load as f32
# ... computation in f32 ...
tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)  # store as bf16
```

### Block size selection for AMD

Prefer multiples of 64 (wavefront size):

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["n_elements"],
)
```

## Debugging torch.compile on ROCm

### Enable Inductor logging

```python
import logging
torch._logging.set_logs(inductor=logging.DEBUG)
# Or via environment variable
os.environ["TORCH_LOGS"] = "inductor"
```

### Export generated Triton code

```python
inductor_config.debug = True
inductor_config.trace.enabled = True
# Generated code saved to /tmp/torchinductor_*/
```

### Disable torch.compile for debugging

```python
# Quick disable to check if issue is compile-related
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Or
model_eager = model  # skip torch.compile()
```

### Common compile errors on ROCm

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: deep recursion` | `memory_planning=True` | Set `memory_planning=False` |
| `65x slowdown` | `reduce-overhead` mode | Use `mode="default"` |
| `Triton compilation error` | Missing ROCm Triton backend | Install `triton-rocm` |
| `CUDA error during capture` | RNG state access | Apply Dynamo RNG patch |
| `Graph break at unsupported op` | Op not supported by Inductor | Set `fullgraph=False` |
