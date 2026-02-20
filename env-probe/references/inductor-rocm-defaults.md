# Inductor & Dynamo Defaults on ROCm Docker Images

## The Problem

AMD ROCm Docker images (e.g., `rocm/sgl-dev:*`) ship with PyTorch builds that override
inductor defaults at the system level. These overrides are invisible — they don't show up in
`pip list`, `rocm-smi`, or any standard environment inspection. They only manifest at runtime
when `torch.compile()` is called.

The most dangerous default is `max_autotune=True`, which causes torch.compile to benchmark
every GEMM operator across all available backends (ATEN, TRITON, CPP). For models with hundreds
of matmuls (e.g., a transformer with 10 unrolled denoise steps), this autotuning process
never finishes — the process hangs indefinitely with no error message and no timeout.

## How to Check Current Defaults

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# The critical ones
print(f"max_autotune:               {inductor_config.max_autotune}")
print(f"max_autotune_gemm_backends: {inductor_config.max_autotune_gemm_backends}")
print(f"triton.cudagraphs:          {inductor_config.triton.cudagraphs}")
print(f"triton.cudagraph_trees:     {inductor_config.triton.cudagraph_trees}")
print(f"memory_planning:            {inductor_config.memory_planning}")
print(f"cache_size_limit:           {dynamo_config.cache_size_limit}")
```

## Safe Defaults for ROCm

Apply before any `torch.compile()` call:

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# ── CRITICAL: prevent indefinite GEMM autotuning ──
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm_backends = "ATEN"  # rocBLAS directly, skip Triton/CPP

# ── CRITICAL: unstable on ROCm ──
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# ── CRITICAL: deep recursion crash on ROCm ──
inductor_config.memory_planning = False

# ── IMPORTANT: prevent recompilation loops ──
dynamo_config.cache_size_limit = 128
```

## Why Each Setting Matters

### `max_autotune = False`

| Default (ROCm Docker) | Recommended |
|------------------------|-------------|
| `True` | `False` |

When True, inductor benchmarks each GEMM across all backends. On NVIDIA this is annoying but
eventually finishes. On ROCm, the Triton backend compilation is much slower, and with hundreds
of GEMMs the total time becomes effectively infinite.

Setting to False means inductor picks the backend heuristically (which is ATEN/rocBLAS — the
right choice anyway).

### `max_autotune_gemm_backends = "ATEN"`

| Default (ROCm Docker) | Recommended |
|------------------------|-------------|
| `"ATEN,TRITON"` or `"ATEN,TRITON,CPP"` | `"ATEN"` |

Even with `max_autotune=False`, this setting controls which backends are *available*. On ROCm,
rocBLAS (through ATEN) consistently outperforms Triton for GEMMs by 35-55%. There's no benefit
to keeping Triton/CPP as GEMM backends.

### `triton.cudagraphs = False`

Triton-managed CUDAGraph capture is unstable on ROCm. It can produce incorrect results silently
or hang during capture. Use manual `torch.cuda.CUDAGraph()` capture instead if you need graph
replay.

### `triton.cudagraph_trees = False`

CUDAGraph trees (hierarchical graph management) are even less stable than flat cudagraphs on
ROCm. Always disable.

### `memory_planning = False`

Inductor's memory planning pass triggers deep recursion on ROCm with complex model graphs,
causing either a stack overflow crash or an indefinite hang. Disabling it means slightly higher
peak memory usage but reliable compilation.

### `dynamo_config.cache_size_limit = 128`

Default is often 8 or 16. Complex models with many control flow paths (like models with multiple
denoise steps) can exceed this, causing cache eviction and full recompilation. Each recompilation
re-traces the entire model. Set to 128 to prevent this.

## Docker Version History

| Docker Tag | Known Defaults | Notes |
|------------|---------------|-------|
| `rocm/sgl-dev:v0.5.8-rocm700-mi30x-20260129` | `max_autotune=True` | hipBLASLt solver bug present |
| `rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260210` | `max_autotune=True` | hipBLASLt solver bug, FP8 flash attn broken |
| `rocm/sgl-dev:v0.5.8.post1-rocm700-mi35x-20260219` | TBD (run env_probe.py) | May have fixes |

This table should be updated as new Docker versions are tested. Run `env_probe.py` in each
new Docker and record the findings.

## Environment Variable Overrides

Some inductor settings can be forced via environment variables, which take precedence over
Python-level configuration:

```bash
# These override Python config — check for them
TORCHINDUCTOR_MAX_AUTOTUNE=1     # Overrides inductor_config.max_autotune
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1  # Disables all inductor caches
TORCH_COMPILE_DEBUG=1             # Enables verbose compilation output
```

The env_probe.py script checks for these overrides.
