# torch.compile & CUDAGraph on ROCm

## Compile Mode (critical)

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
compile_mode = "default" if is_rocm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

`reduce-overhead` triggers Inductor's automatic CUDAGraph capture, which is **broken on ROCm** (up to 65x slowdown or hang). Gate with `is_rocm` — NVIDIA path unchanged.

## Inductor Configuration

```python
import torch._inductor.config as inductor_config

# CRITICAL: disable broken features on ROCm
inductor_config.triton.cudagraphs = False       # broken on ROCm
inductor_config.triton.cudagraph_trees = False  # broken on ROCm
inductor_config.memory_planning = False         # deep recursion crash on ROCm

# GEMM: ATen/rocBLAS faster than Triton on AMD
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False

# Tuning and fusion (improve AMD performance)
inductor_config.coordinate_descent_tuning = True
inductor_config.epilogue_fusion = True
inductor_config.aggressive_fusion = True
```

Apply before any `torch.compile()` call. Gate the whole block behind `if is_rocm:`.

## Compile Safety Monkey-Patch

Intercept any code path that hardcodes `reduce-overhead`:

```python
_orig_compile = torch.compile
def _safe_compile(model=None, **kwargs):
    if is_rocm and kwargs.get("mode") in (None, "reduce-overhead"):
        kwargs["mode"] = "default"
    return _orig_compile(model, **kwargs)
torch.compile = _safe_compile
```

## Dynamo RNG Patch (before any CUDAGraph capture)

ROCm forbids `torch.cuda.get_rng_state()` during stream capture. Patch Dynamo to skip it:

```python
import functools, torch._dynamo.convert_frame

def patch_dynamo_rng():
    if getattr(torch._dynamo.convert_frame, "_rocm_patched", False):
        return
    torch._dynamo.convert_frame._rocm_patched = True
    _orig = torch._dynamo.convert_frame.preserve_global_state
    def _skip_rng(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            rng = None
            if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                try: rng = torch.cuda.get_rng_state()
                except Exception: pass
            try: return fn(*args, **kwargs)
            finally:
                if rng is not None:
                    try: torch.cuda.set_rng_state(rng)
                    except Exception: pass
        return wrapper
    torch._dynamo.convert_frame.preserve_global_state = _skip_rng
```

## Manual CUDAGraph Capture

Since Inductor automatic capture is broken, capture manually:

```python
patch_dynamo_rng()
model = torch.compile(model, mode="default")
# 1. Warmup (resolves Dynamo tracing + Triton autotuning)
with torch.no_grad():
    for _ in range(5): _ = model(static_input)
torch.cuda.current_stream().synchronize()
# 2. Capture
pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    static_output = model(static_input)
# 3. Replay: copy new input, call graph.replay(), read static_output
```

## Stream Capture Rules

**Forbidden during capture**: `.item()`, `torch.cuda.synchronize()`, `print(tensor)`,
`torch.cuda.get_rng_state()`, dynamic shape branching, new-size memory allocation.

**Prefer**: `torch.cuda.current_stream().synchronize()` over `torch.cuda.synchronize()` (lower overhead on ROCm).

## Triton on ROCm

- Prefer block sizes that are multiples of **64** (AMD wavefront width): 512, 1024, 2048
- Always accumulate in `float32`, store back in `bfloat16`/`float16`
- Clamp inputs to `[-10, 10]` before `exp` to avoid tanh overflow NaN
