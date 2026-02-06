---
name: amd-porting-optimization
description: >
  Port NVIDIA-only PyTorch repositories to AMD ROCm GPUs, optimize performance with
  AMD-specific kernels, and benchmark iteratively to validate correctness and speedup.
  Use this skill when the user wants to: (1) make an NVIDIA-only codebase run on AMD GPUs,
  (2) optimize PyTorch code for AMD MI-series GPUs (MI100/MI200/MI300/MI350),
  (3) write benchmarks comparing eager vs optimized performance on AMD hardware.
  Follows an iterative workflow: port -> benchmark baseline -> optimize -> re-benchmark -> iterate.
---

# AMD GPU Porting & Optimization for PyTorch Repositories

An iterative workflow for porting NVIDIA-only PyTorch codebases to AMD ROCm GPUs and optimizing performance.

## Workflow Overview

```
Phase 1: Port (Make It Run)
    |
    v
Phase 2: Benchmark Baseline (Measure It)
    |
    v
Phase 3: Optimize (Make It Fast) ----+
    |                                 |
    v                                 |
Phase 4: Re-benchmark (Prove It) ----+  (iterate)
```

The key principle: **always measure before and after**. Write benchmark scripts first to establish a baseline, then optimize, then re-benchmark to quantify improvement. Iterate until satisfied.

---

## Phase 1: Compatibility Porting (Make It Run)

Goal: Get the codebase running on AMD GPUs with zero functional regressions.

### 1.1 Understand the PyTorch Device Abstraction

PyTorch's ROCm build maps `torch.cuda.*` APIs to HIP transparently. Most model code works unchanged:

```python
# These all work on AMD ROCm without modification:
torch.cuda.is_available()      # True on ROCm
tensor.cuda()                   # Allocates on AMD GPU via HIP
torch.cuda.synchronize()        # Syncs AMD GPU
torch.cuda.memory_allocated()   # Reports AMD GPU memory
```

**Do NOT** blindly find-and-replace `cuda` with `rocm` or `hip` in PyTorch code. The `torch.cuda` namespace is the correct API on both platforms.

### 1.2 Porting Checklist

Search the codebase for each category and apply fixes:

#### A. Vendor-Specific Libraries

| NVIDIA Library | AMD Replacement | Notes |
|---------------|-----------------|-------|
| `pynvml` / `nvidia-smi` | `amdsmi` / `rocm-smi` CLI | GPU monitoring, temperature, utilization |
| `flash-attn` (Tri Dao) | `aiter` (AMD) or `flash-attn-rocm` | Flash attention kernels |
| `apex` | Usually unnecessary | PyTorch native AMP covers most cases |
| `cupy` | `hipify` or rewrite | If used for custom CUDA kernels |
| `triton` | `triton` (works on ROCm) | Same package, auto-targets AMD |

Search patterns:
```
grep -rn "pynvml\|nvidia_smi\|nvml\|from flash_attn\|import apex" src/
```

#### B. Environment Variables

| NVIDIA Variable | AMD Equivalent | Purpose |
|----------------|----------------|---------|
| `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` | GPU selection (both work) |
| `PYTORCH_CUDA_ALLOC_CONF` | `PYTORCH_HIP_ALLOC_CONF` | Memory allocator config |
| `NCCL_*` | `NCCL_*` or `RCCL_*` | Collective communication (both work) |
| `CUDA_LAUNCH_BLOCKING` | `HIP_LAUNCH_BLOCKING` | Sync kernel launches for debugging |

#### C. Backend-Specific Code

| Pattern | Fix |
|---------|-----|
| `torch.backends.cudnn.*` settings | Wrap in `if torch.version.hip is None:` guard (no-ops on ROCm) |
| Hardcoded `torch.version.cuda` checks | Add `or torch.version.hip` alternative |
| `torch.cuda.get_device_capability()` | Works on ROCm but returns different tuples; avoid gating on specific SM versions |
| `dist.init_process_group(backend='nccl')` | No change needed (RCCL exposes as "nccl") |

#### D. Python Version Compatibility

AMD ROCm environments often use Python 3.10 (vs 3.11+ common on NVIDIA). Check for:

| Pattern | Fix |
|---------|-----|
| `datetime.UTC` | Use `datetime.timezone.utc` (3.10 compat) |
| `X \| Y` type unions | Add `from __future__ import annotations` |
| `match/case` statements | Rewrite as `if/elif` chains |
| `tomllib` (3.11+) | Use `tomli` package as fallback |

#### E. Create GPU Abstraction Utility

Create a utility module for cross-vendor GPU operations:

```python
# gpu_utils.py - template structure
import torch

def detect_gpu_vendor() -> str:
    """Returns 'nvidia', 'amd', or 'none'."""
    if not torch.cuda.is_available():
        return "none"
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return "amd"
    return "nvidia"

def configure_memory_optimizations():
    """Set vendor-specific memory/backend optimizations."""
    vendor = detect_gpu_vendor()
    if vendor == "amd":
        import os
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",
                              "expandable_segments:True")
    elif vendor == "nvidia":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
```

### 1.3 Validation

After porting, verify basic functionality:

```python
# Minimal validation script template
import torch

# 1. GPU detection
assert torch.cuda.is_available(), "No GPU detected"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"HIP version: {torch.version.hip}")

# 2. Basic ops
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
y = torch.matmul(x, x.T)
assert not torch.isnan(y).any(), "NaN in matmul output"

# 3. Model forward/backward
from torch import nn
model = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True).cuda().to(torch.bfloat16)
inp = torch.randn(4, 128, 512, dtype=torch.bfloat16, device="cuda")
out = model(inp)
out.sum().backward()
print("Basic validation PASSED")
```

---

## Phase 2: Benchmark Baseline (Measure It)

Goal: Establish performance metrics **before any optimization**, so later improvements can be quantified.

### 2.1 Precision Verification Script (Write First)

**Always verify correctness before benchmarking performance.** Compare outputs against a known baseline.

```python
#!/usr/bin/env python3
"""Template: verify optimized kernels match eager baseline."""

import torch

def verify_precision(model, create_input_fn, optimize_fn, device="cuda"):
    model.eval()

    # Run baseline (eager)
    torch.manual_seed(42)
    inputs = create_input_fn(device)
    with torch.no_grad():
        baseline_output = model(**inputs)

    # Enable optimizations
    optimize_fn(model)

    # Run optimized (same seed, same inputs)
    torch.manual_seed(42)
    inputs = create_input_fn(device)
    with torch.no_grad():
        optimized_output = model(**inputs)

    # Compare
    base_flat = baseline_output.flatten().float()
    opt_flat = optimized_output.flatten().float()

    cos_sim = torch.nn.functional.cosine_similarity(
        base_flat.unsqueeze(0), opt_flat.unsqueeze(0)
    ).item()
    max_diff = (base_flat - opt_flat).abs().max().item()
    has_nan = torch.isnan(optimized_output).any().item()

    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"Max Absolute Diff: {max_diff:.6f}")
    print(f"Has NaN:           {has_nan}")

    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
    assert max_diff < 1.0, f"Max diff too high: {max_diff}"
    assert not has_nan, "NaN detected in optimized output"
    print("PASSED")
```

### 2.2 Inference Benchmark Script

Measures end-to-end inference latency. Record these numbers as the **baseline**.

```python
#!/usr/bin/env python3
"""Template: inference latency benchmark."""

import time
import numpy as np
import torch

def benchmark_inference(model, create_input_fn, device="cuda",
                        warmup=10, iterations=30):
    model.eval()
    inputs = create_input_fn(device)

    # Warmup (critical for GPU clock stabilization)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "throughput_hz": 1000.0 / np.mean(latencies),
        "memory_gb": torch.cuda.memory_allocated(device) / 1e9,
    }
```

**Key practices:**
- Always `torch.cuda.synchronize()` before and after timing (GPU ops are async)
- Use `time.perf_counter()` (not `time.time()`)
- Warmup is mandatory: first iterations trigger kernel autotuning and GPU boost
- Report P50/P95, not just mean (captures variance)

### 2.3 Training Throughput Benchmark Script

```python
#!/usr/bin/env python3
"""Template: training throughput benchmark."""

import time
import torch

def benchmark_training(model, create_input_fn, device="cuda",
                       warmup=5, iterations=20):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = create_input_fn(device, for_training=True)
    batch_size = inputs["input"].shape[0]

    for _ in range(warmup):
        optimizer.zero_grad()
        loss = model(**inputs)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = model(**inputs)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "samples_per_sec": (batch_size * iterations) / elapsed,
        "ms_per_step": (elapsed / iterations) * 1000,
        "final_loss": loss.item(),
    }
```

### 2.4 Profiling with torch.profiler

Generate traces for kernel-level analysis:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for _ in range(5):
        model(**inputs)
    torch.cuda.synchronize()

prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

View traces at [Perfetto UI](https://ui.perfetto.dev/).

### 2.5 Record Baseline

After running benchmarks, document results:

```markdown
## Baseline (Eager PyTorch on AMD MI-XXX)

| Metric | Value |
|--------|-------|
| Inference latency (P50) | XX ms |
| Throughput | X.X Hz |
| GPU memory | X.X GB |
| Training samples/sec | X.X |

Date: YYYY-MM-DD
Hardware: AMD MI-XXX
PyTorch: X.X.X+rocmX.X
```

---

## Phase 3: Optimize (Make It Fast)

Goal: Apply optimizations incrementally, re-benchmarking after each change.

### Optimization Priority Order

Apply in this order (highest impact first). After each optimization, run the benchmark scripts from Phase 2 and record the improvement.

1. **torch.compile** - biggest single win, minimal code changes
2. **Attention kernel replacement** - replaces the most expensive operation
3. **Triton kernel fusion** - eliminates kernel launch overhead for elementwise ops
4. **Projection fusion** - reduces GEMM count in attention and MLP
5. **GEMM kernel tuning** - hardware-specific matrix multiply routing

### 3.1 torch.compile Configuration

**CRITICAL: ROCm-specific torch.compile settings differ significantly from NVIDIA.**

```python
import os
import torch

vendor = "amd" if (hasattr(torch.version, "hip") and torch.version.hip) else "nvidia"

if vendor == "amd":
    # Use "default" mode on ROCm. DO NOT use "reduce-overhead".
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
else:
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")

# Increase dynamo cache for models with dynamic shapes
import torch._dynamo.config as dynamo_config
dynamo_config.cache_size_limit = 64

model.inference_fn = torch.compile(model.inference_fn, mode=compile_mode)
```

#### ROCm Inductor Configuration

```python
import torch._inductor.config as inductor_config

def configure_inductor_for_amd():
    """Configure inductor defaults for AMD ROCm GPUs."""
    # CRITICAL: Disable CUDA/HIP graphs - broken or counterproductive on many ROCm versions
    inductor_config.triton.cudagraphs = False

    # Force ATEN backend for GEMMs (routes to rocBLAS, which is 35-55% faster than
    # Triton-generated GEMM kernels on AMD)
    inductor_config.max_autotune_gemm_backends = "ATEN"

    # Standard fusion optimizations (these work well on ROCm)
    inductor_config.epilogue_fusion = True
    inductor_config.pattern_matcher = True
    inductor_config.aggressive_fusion = True

    # Disable memory planning if you hit ROCm-specific bugs
    # inductor_config.memory_planning = False
```

> **Rule: `reduce-overhead` mode can be orders of magnitude SLOWER on ROCm.** It enables CUDA/HIP graphs which are unstable on ROCm. Always default to `"default"` mode on AMD and benchmark before trying other modes.

> **Rule: Triton GEMM kernels are significantly slower than rocBLAS on AMD.** Force the ATEN backend for GEMMs. Triton is still excellent for elementwise/fusion kernels (RMSNorm, activations).

### 3.2 Attention Kernel Optimization

Replace PyTorch's generic attention with AMD-optimized flash attention (e.g., from the `aiter` library):

```python
AITER_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    pass

USE_AITER_ATTENTION = os.environ.get("USE_AITER_ATTENTION", "0") == "1"

def aiter_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
    """Flash attention using AMD-optimized kernels."""
    q_len, k_len = query.shape[2], key.shape[2]

    # Classify mask type (flash attention only supports causal/full)
    use_causal, can_use_flash = _classify_mask(attention_mask, q_len, k_len)
    if not can_use_flash:
        return eager_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs)

    # Transpose to [batch, seq, heads, head_dim] for flash attention
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    result = aiter.flash_attn_func(q, k, v,
        dropout_p=0.0, softmax_scale=scaling, causal=use_causal)
    attn_output = result[0] if isinstance(result, tuple) else result
    return attn_output, None
```

> **Hint: Native GQA/MQA support.** If the flash attention kernel supports different Q/KV head counts natively, skip `repeat_kv` to avoid unnecessary memory copies. Check the kernel's documentation.

> **Hint: Avoid unnecessary `.contiguous()`.** After `transpose(1, 2)`, the last dimension usually remains contiguous (`stride(-1)==1`). Only call `.contiguous()` when the kernel explicitly rejects the tensor layout.

> **Hint: Direct op calls for torch.compile.** If using torch.compile, prefer calling the underlying registered op (e.g., `torch.ops.vendor.op_name`) instead of high-level Python wrappers. This avoids Python-side logic that creates graph breaks during tracing.

> **Hint: Multi-level fallback chain.** Implement graceful degradation: vendor flash attention -> SDPA -> eager. Each level catches errors and falls back to the next.

### 3.3 Triton Kernel Fusion

Write fused Triton kernels for elementwise operations. See [references/triton_kernel_patterns.md](references/triton_kernel_patterns.md) for complete implementations.

**High-value targets (ordered by impact):**

1. **RMSNorm** - replaces 5 ops with 1 kernel (~3.4x speedup)
2. **Fused GELU + Mul** - replaces 3 ops with 1 kernel (~1.6x speedup)
3. **Fused Add + RMSNorm** - combines residual addition with normalization (~2.8x speedup)
4. **Fused SiLU + Mul** - alternative activation fusion (~1.4x speedup)

**General Triton kernel pattern:**
```python
@triton.jit
def _fused_kernel(X_ptr, Y_ptr, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride + cols, mask=mask).to(tl.float32)
    # ... fused computation in float32 ...
    tl.store(Y_ptr + row * stride + cols, result.to(tl.bfloat16), mask=mask)
```

**Design rules:**
- Compute in float32, store in bfloat16 (numerical stability)
- BLOCK_SIZE must be a power of 2 and >= N
- Always provide an eager fallback when Triton is unavailable
- Use environment variables to toggle optimized vs eager paths

> **Rule: Clamp values before `exp()` in Triton on ROCm.** Some ROCm Triton builds lack a native tanh intrinsic, so tanh is implemented via `exp`. Large inputs cause overflow -> inf/inf -> NaN. Always clamp: `inner = tl.maximum(tl.minimum(inner, 10.0), -10.0)` before `exp`.

### 3.4 Projection Fusion

Reduce kernel launch count by fusing separate linear projections into combined GEMMs:

**QKV Fusion (3 GEMMs -> 1):**
```python
fused_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
qkv = F.linear(x, fused_weight)
q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
```

**Gate+Up Fusion (2 GEMMs -> 1) for MLP:**
```python
fused_weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0)
gate_up = F.linear(x, fused_weight)
out = gelu_and_mul(gate_up)  # Triton fused kernel
```

**Implementation pattern:**
```python
def fuse_projections(model, verbose=True):
    """Call AFTER loading weights, BEFORE torch.compile."""
    for layer in model.layers:
        attn = layer.self_attn
        fused = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
        attn.register_buffer("_fused_qkv_weight", fused)
        attn._use_fused_qkv = True
```

### 3.5 GEMM Kernel Routing

For vendor-specific GEMM libraries (e.g., aiter on AMD), route based on matrix shape:

```python
def vendor_linear(x, weight, bias=None):
    """Route GEMM to vendor-optimized kernel for small M, rocBLAS for large M."""
    M = x.numel() // x.shape[-1]
    M_THRESHOLD = int(os.environ.get("GEMM_M_THRESHOLD", "64"))

    if M <= M_THRESHOLD and VENDOR_GEMM_AVAILABLE:
        return vendor_gemm_fn(x, weight, bias)
    else:
        return F.linear(x, weight, bias)
```

> **Hint: Weight preshuffling.** Some vendor GEMM kernels expect weights in a specific layout. Pre-shuffle weights at model load time (after loading, before inference) to avoid per-call overhead. Keep the original weight intact for fallback paths.

> **Hint: M-threshold routing.** Vendor-tuned GEMM kernels often beat rocBLAS only for small M dimensions (e.g., batch=1 inference). For large M (training batches), rocBLAS/ATEN is usually faster. Profile to find the crossover point.

### 3.6 torch.compile Compatibility

When preparing code for torch.compile, watch out for these patterns:

> **Rule: Replace `while` loops with `for` loops.** `while condition:` loops cause torch.compile to recursively unroll or fail. Convert to `for step in range(num_steps):` with a precomputed iteration count.

> **Rule: Pre-allocate constant tensors as module buffers.** Tensors created inside `forward()` (e.g., attention masks, timestep schedules) trigger host-to-device transfers that break CUDA graph capture. Register them as buffers with `register_buffer()`.

> **Rule: Cache repeated tensor allocations.** If you create the same tensor every forward pass (e.g., a schedule, a padding mask), compute it once and cache it on the module.

### 3.7 Environment Variable Convention

All optimizations should be toggleable for A/B testing:

```python
USE_AITER_ATTENTION = os.environ.get("USE_AITER_ATTENTION", "0") == "1"
USE_FUSED_PROJECTIONS = os.environ.get("USE_FUSED_PROJECTIONS", "0") == "1"
USE_OPTIMIZED_OPS = os.environ.get("USE_OPTIMIZED_OPS", "0") == "1"
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "default")
```

Default to OFF ("0") so the codebase remains compatible with non-AMD systems.

---

## Phase 4: Re-benchmark & Iterate (Prove It)

Goal: Quantify each optimization's impact and iterate.

### 4.1 Incremental Benchmark Reporting

After each optimization from Phase 3, re-run the exact same benchmark scripts from Phase 2. Report results incrementally:

```markdown
## Results: Optimization Progression (AMD MI-XXX)

| Configuration | Latency (P50) | vs Baseline | Memory |
|---------------|---------------|-------------|--------|
| Eager baseline | XX ms | 1.00x | X.X GB |
| + torch.compile (default) | XX ms | X.Xx | X.X GB |
| + aiter attention | XX ms | X.Xx | X.X GB |
| + Triton kernels | XX ms | X.Xx | X.X GB |
| + fused projections | XX ms | X.Xx | X.X GB |
| + GEMM routing | XX ms | X.Xx | X.X GB |
| **All optimizations** | **XX ms** | **X.Xx** | X.X GB |
```

### 4.2 If an Optimization Regresses

- **Latency increased**: The optimization may not be suitable for this hardware/workload. Disable it and move on.
- **NaN appeared**: Check numerical stability (especially Triton kernels). See the clamping rule in 3.3.
- **torch.compile fails**: Check for graph breaks. Use `TORCH_LOGS="graph_breaks" python script.py` to diagnose.

### 4.3 Multi-GPU DDP Benchmark

After single-GPU optimizations are validated, test multi-GPU scaling:

```python
# Launch with: torchrun --nproc_per_node=NUM_GPUS script.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend="nccl")  # Works for both NVIDIA and AMD
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()
```

---

## ROCm-Specific Gotchas (Quick Reference)

These are hard-won lessons from real AMD GPU optimization work:

| Issue | Cause | Fix |
|-------|-------|-----|
| `reduce-overhead` mode extremely slow | CUDA/HIP graphs broken on many ROCm versions | Use `"default"` mode |
| Triton GEMM 35-55% slower than expected | Triton codegen suboptimal for AMD GEMM | Force `ATEN` backend in inductor |
| NaN in Triton GELU/tanh kernel | `exp()` overflow (no native tanh intrinsic) | Clamp input to `[-10, 10]` before `exp` |
| torch.compile infinite recursion | `while` loop in forward pass | Convert to `for` loop |
| CUDA graph capture fails | Host-to-device transfer during capture | Pre-allocate tensors as module buffers |
| Dynamo RNG state error during graph capture | ROCm bug: `get_state` unsupported in capture | Patch Dynamo to skip RNG state on ROCm |
| `repeat_kv` slow or breaks compile | Creates expanded/zero-stride views | Use native GQA support in attention kernel |
| `.contiguous()` adds overhead | Unnecessary copy after transpose | Only call when kernel rejects the layout |
| Sync overhead (30%+ of wall time) | Excessive `hipDeviceSynchronize` calls | Profile sync time, use async patterns |

---

## Common Pitfalls

1. **Don't replace `torch.cuda` with `torch.hip`** - The `torch.cuda` API is the correct abstraction on both platforms.

2. **Don't skip warmup in benchmarks** - First iterations include JIT compilation, autotuning, and GPU clock ramp-up.

3. **Don't forget `torch.cuda.synchronize()`** - GPU ops are asynchronous. Without sync, you measure CPU dispatch time, not GPU execution time.

4. **Don't assume NVIDIA compile modes work on AMD** - `reduce-overhead` and `max-autotune` behave very differently on ROCm. Always benchmark.

5. **Don't assume all attention masks work with flash attention** - Flash attention only supports pure causal and full bidirectional masks. Always implement fallback to eager for complex masks.

6. **Don't hardcode compilation paths** - Always make optimizations toggleable via environment variables.

7. **Always verify precision first** - Run the precision verification script before reporting any performance numbers.

8. **Always benchmark before AND after** - Never claim an optimization helps without measuring the delta against the established baseline.

---

## File Organization Convention

```
project/
  src/
    models/
      triton_ops.py          # Triton kernel implementations
      aiter_ops.py           # AMD optimization wrappers with fallbacks
  scripts/
    benchmark_inference.py    # Single-GPU inference latency
    benchmark_training.py     # Single-GPU training throughput
    benchmark_ddp.py          # Multi-GPU DDP training
    verify_precision.py       # Correctness validation
  shared/
    gpu_utils.py              # Cross-vendor GPU abstraction
  benchmark.md                # Results documentation (baseline + progression)
```
