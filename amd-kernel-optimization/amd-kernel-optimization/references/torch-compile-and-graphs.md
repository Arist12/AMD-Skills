# torch.compile and CUDAGraph on ROCm

## torch.compile Mode Selection

| Mode | What It Does | Known ROCm Considerations |
|------|-------------|--------------------------|
| `default` | Dynamo tracing + Triton codegen for elementwise fusion | Generally stable on ROCm; good starting point |
| `reduce-overhead` | Adds CUDAGraph capture to reduce launch overhead | Uses CUDAGraphs internally; ROCm HIP graph support has known stability issues — test carefully on your workload |
| `max-autotune` | Benchmarks multiple backends (including Triton GEMM) per op | Longer compile time; benchmark to see if the autotuning gains offset the compilation cost |

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

# "default" is a safe starting point on ROCm; benchmark other modes for your workload
compile_mode = "default" if is_rocm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

## Inductor Configuration for ROCm

These settings are a reasonable **starting point** for ROCm. Adjust based on your profiling results.

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# GEMM backend: ATen (rocBLAS) as starting point.
# Benchmark with max_autotune=True to see if Triton/hipBLASLt help for your shapes.
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False
inductor_config.coordinate_descent_tuning = False

# Fusion: enable for elementwise ops
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.aggressive_fusion = False  # experiment: may help or hurt

# Inductor CUDAGraphs: OFF as safe default on ROCm (can cause instability)
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Memory planning: OFF as safe default (known to cause recursion issues
# with large graphs on ROCm). Enable and test if your graph is small.
inductor_config.memory_planning = False

# Reorder for locality: generally helps memory access patterns
inductor_config.reorder_for_locality = True

# Dynamo cache: increase for models with dynamic shapes (KV cache, variable seq len)
dynamo_config.cache_size_limit = 128
```

## Compiling Through Vendor Ops

To get the best results from `torch.compile`, compile *through* vendor attention/GEMM ops rather than graph-breaking around them.

### Avoiding graph breaks

Graph breaks split the compiled graph into multiple fragments, increasing kernel launch overhead. Common causes on ROCm:

1. **Python-level vendor op wrappers** — Dynamo can't trace through Python control flow in vendor libraries
2. **Explicit `torch._dynamo.disable`** — Wrapping ops in `@dynamo.disable` forces a graph break

### Solution: use `torch.ops` path

Register vendor ops as torch custom ops and call them via `torch.ops.*`:

```python
# Python wrapper — may cause graph break:
from aiter import flash_attn_func
out = flash_attn_func(q, k, v)

# Direct torch.ops path — compile-friendly:
out = torch.ops.aiter.mha_fwd.default(q, k, v, ...)[0]
```

The `torch.ops` path lets Dynamo trace through the call without graph-breaking.

## CUDAGraph on ROCm

### Inductor-level CUDAGraphs

Inductor's built-in graph capture (`reduce-overhead` mode) may be unstable on ROCm. HIP graph support varies by ROCm version — test on your specific setup.

### Manual full-call CUDAGraph capture

Capturing an entire function call as one graph can reduce kernel launch overhead:

```python
import torch.cuda

# 1. Warmup (must run with exact same shapes)
for _ in range(5):
    output = model_fn(input_tensors)

# 2. Capture
pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    static_output = model_fn(input_tensors)

# 3. Replay in loop
for step in range(num_steps):
    graph.replay()
    result = static_output  # output aliases the captured memory
```

### The Dynamo RNG capture bug (ROCm-specific)

**Problem:** If Dynamo (torch.compile) traces a new frame during graph capture, it calls `torch.cuda.get_rng_state()`, which queries the CUDA generator seed. ROCm disallows this during capture:
```
RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture
```

**Fix:** Patch Dynamo's `preserve_global_state` to skip CUDA RNG state during capture:

```python
import torch._dynamo.convert_frame as _cf

def patch_dynamo_for_rocm_capture():
    """Skip CUDA RNG get_state during graph capture on ROCm."""
    import contextlib, functools, random as _py_random

    def _safe_preserve(fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            guards = _cf.GlobalStateGuard()
            prior_grad = torch.is_grad_enabled()

            with torch._C._PreserveDispatchKeyGuard():
                py_rng = _py_random.getstate()
                torch_rng = torch.random.get_rng_state()

                # KEY FIX: skip CUDA RNG if capturing
                cuda_rng = None
                if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                    cuda_rng = torch.cuda.get_rng_state()

                try:
                    return fn(*args, **kwargs)
                finally:
                    torch._C._set_grad_enabled(prior_grad)
                    _py_random.setstate(py_rng)
                    torch.random.set_rng_state(torch_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng)

        return _fn

    _cf.preserve_global_state = _safe_preserve
```

Call this patch **before** graph capture if using `torch.compile` + manual CUDAGraph together on ROCm.

## HIP Environment Variables

```bash
# Async kernel launches (keep 0 for production; set 1 only for debugging)
export HIP_LAUNCH_BLOCKING=0

# Suppress verbose AMD logging
export AMD_LOG_LEVEL=0

# Enable HIP compilation cache (avoids recompilation on restart)
export HIP_CACHE_ENABLED=1

# PyTorch memory allocator tuning for ROCm
# expandable_segments avoids fragmentation; useful for large models
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

## Profiling on ROCm

### torch.profiler (chrome trace)

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    output = model_fn(input_tensors)

prof.export_chrome_trace("trace.json")
```

Open `trace.json` in `chrome://tracing` or Perfetto to identify:
- Hot kernels (longest wall-clock time)
- Kernel launch gaps (CPU overhead between GPU kernels)
- Memory copies (host-device transfers)

For deeper hardware-level analysis, AMD also provides **`rocprofv3`** — a GPU profiler that collects hardware counters (occupancy, cache hit rates, memory bandwidth).

### Inductor compile logging

Wrap `torch._inductor.compile_fx.compile_fx` to log per-graph compilation timing, node counts, and op breakdowns. This helps identify:
- How many sub-graphs Dynamo creates (more = more graph breaks)
- Which ops are most common (optimization targets)
- Compilation time bottlenecks
