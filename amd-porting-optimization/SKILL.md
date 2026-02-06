---
name: amd-porting-optimization
description: >
  Port NVIDIA-only PyTorch repositories to AMD ROCm GPUs, optimize performance with
  AMD-specific kernels, and write benchmarks to validate correctness and speedup.
  Use this skill when the user wants to: (1) make an NVIDIA-only codebase run on AMD GPUs,
  (2) optimize PyTorch code for AMD MI-series GPUs (MI100/MI200/MI300/MI350),
  (3) write benchmarks comparing eager vs optimized performance on AMD hardware.
  Covers the full workflow: compatibility porting, attention kernel optimization,
  Triton kernel fusion, projection fusion, torch.compile tuning, and benchmark scripting.
---

# AMD GPU Porting & Optimization for PyTorch Repositories

A three-phase workflow for porting NVIDIA-only PyTorch codebases to AMD ROCm GPUs and optimizing performance.

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

Search the codebase for each of these categories and apply fixes:

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

Search patterns:
```
grep -rn "PYTORCH_CUDA_ALLOC_CONF\|CUDA_LAUNCH_BLOCKING\|NCCL_" src/
```

#### C. Backend-Specific Code

| Pattern | Fix |
|---------|-----|
| `torch.backends.cudnn.*` settings | Wrap in `if torch.version.hip is None:` guard (no-ops on ROCm) |
| Hardcoded `torch.version.cuda` checks | Add `or torch.version.hip` alternative |
| `torch.cuda.get_device_capability()` | Works on ROCm but returns different tuples; avoid gating on specific SM versions |
| `dist.init_process_group(backend='nccl')` | No change needed (RCCL exposes as "nccl") |

Search patterns:
```
grep -rn "cudnn\|torch.version.cuda\|get_device_capability\|sm_[0-9]" src/
```

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

def get_ddp_backend() -> str:
    """Returns the appropriate DDP backend name."""
    # RCCL is exposed as 'nccl' in PyTorch
    if detect_gpu_vendor() != "none":
        return "nccl"
    return "gloo"
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
y = torch.matmul(x, x.T)  # Matrix multiply
assert not torch.isnan(y).any(), "NaN in matmul output"

# 3. Autograd
x.requires_grad_(True)
loss = torch.matmul(x, x.T).sum()
loss.backward()
assert x.grad is not None, "Gradient computation failed"

# 4. Model forward/backward
from torch import nn
model = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True).cuda().to(torch.bfloat16)
inp = torch.randn(4, 128, 512, dtype=torch.bfloat16, device="cuda")
out = model(inp)
out.sum().backward()
print("Basic validation PASSED")
```

---

## Phase 2: Performance Optimization (Make It Fast)

Goal: Achieve competitive or superior performance vs NVIDIA through AMD-specific kernel optimizations.

### Optimization Priority Order

Apply optimizations in this order (highest impact first):

1. **torch.compile** - biggest single win, minimal code changes
2. **Attention kernel replacement** - replaces the most expensive operation
3. **Triton kernel fusion** - eliminates kernel launch overhead for elementwise ops
4. **Projection fusion** - reduces GEMM count in attention and MLP
5. **GEMM kernel tuning** - hardware-specific matrix multiply

### 2.1 torch.compile Configuration

This is the single highest-impact optimization. Configure it properly for AMD:

```python
import os
import torch

# Use reduce-overhead mode for inference (enables CUDA graphs)
# Use max-autotune for training (longer compile, better kernels)
compile_mode = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")

# Increase dynamo cache for models with dynamic shapes (KV cache, variable seq len)
import torch._dynamo.config as dynamo_config
dynamo_config.cache_size_limit = 64  # default is 8

# Enable aggressive fusion in inductor
import torch._inductor.config as inductor_config
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.aggressive_fusion = True

# Apply to inference entry point
model.inference_fn = torch.compile(model.inference_fn, mode=compile_mode)
```

**Key modes:**
- `reduce-overhead`: Uses CUDA graphs, best for fixed-shape inference. 3-4x speedup typical.
- `max-autotune`: Tries many kernel variants, best for training. Longer compile time.
- `default`: Moderate optimization, fastest compile.

### 2.2 Attention Kernel Optimization

Replace PyTorch's generic attention with AMD-optimized flash attention from the `aiter` library:

```python
# Pattern: add aiter flash attention as a selectable backend

# In the attention module (e.g., modeling_gemma.py):
AITER_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    pass

USE_AITER_ATTENTION = os.environ.get("USE_AITER_ATTENTION", "0") == "1"

def aiter_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
    """
    Flash attention using aiter's AMD-optimized ASM kernels.

    Supports: pure causal masks, full bidirectional (no mask), GQA.
    Falls back to eager for: complex masks (padding, prefix-LM, cross-attention).
    """
    q_len, k_len = query.shape[2], key.shape[2]

    # Cross-attention or KV cache: fall back to eager
    if q_len != k_len:
        return eager_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs)

    # Classify mask type
    use_causal, can_use_flash = _classify_mask(attention_mask)
    if not can_use_flash:
        return eager_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs)

    # Expand KV heads for GQA
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Transpose to [batch, seq, heads, head_dim] for aiter
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    result = aiter.flash_attn_func(q, k, v,
        dropout_p=0.0,
        softmax_scale=scaling,
        causal=use_causal,
        return_lse=True,
    )
    attn_output = result[0] if isinstance(result, tuple) else result
    return attn_output, None
```

**Important constraints for aiter flash attention:**
- Requires contiguous tensors in `[batch, seq, heads, head_dim]` layout
- Only supports pure causal or full bidirectional masks
- Must fall back to eager for padding masks, prefix-LM masks, cross-attention
- `head_dim` must be supported by the hardware (commonly 64, 128, 256)

### 2.3 Triton Kernel Fusion

Write fused Triton kernels for frequently-used elementwise operations. These eliminate kernel launch overhead and reduce memory bandwidth by doing multiple operations in a single GPU kernel pass.

**High-value targets (ordered by impact):**

1. **RMSNorm** - replaces 5 ops (pow, mean, rsqrt, mul, mul) with 1 kernel
2. **Fused GELU + Mul** - replaces 3 ops (gelu, slice, mul) with 1 kernel
3. **Fused Add + RMSNorm** - combines residual addition with normalization
4. **Fused SiLU + Mul** - alternative activation fusion

See [references/triton_kernel_patterns.md](references/triton_kernel_patterns.md) for complete kernel implementations.

**General Triton kernel pattern:**
```python
import triton
import triton.language as tl

@triton.jit
def _fused_kernel(X_ptr, Y_ptr, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride + cols, mask=mask).to(tl.float32)
    # ... fused computation in float32 ...
    tl.store(Y_ptr + row * stride + cols, result.to(tl.bfloat16), mask=mask)

def fused_op(x: torch.Tensor) -> torch.Tensor:
    M, N = x.view(-1, x.shape[-1]).shape
    y = torch.empty_like(x)
    _fused_kernel[(M,)](x, y, x.stride(-2), N, BLOCK_SIZE=triton.next_power_of_2(N))
    return y
```

**Design rules for Triton kernels:**
- Compute in float32, store in bfloat16 (numerical stability)
- BLOCK_SIZE must be a power of 2 and >= N
- Always provide an eager fallback for when Triton is unavailable
- Use environment variables to toggle optimized vs eager paths

### 2.4 Projection Fusion

Reduce kernel launch count by fusing separate linear projections into combined GEMMs:

**QKV Fusion (3 GEMMs -> 1):**
```python
# Before: 3 separate matrix multiplies
q = self.q_proj(x)  # [B, S, H] @ [H, num_heads * head_dim]
k = self.k_proj(x)  # [B, S, H] @ [H, num_kv_heads * head_dim]
v = self.v_proj(x)  # [B, S, H] @ [H, num_kv_heads * head_dim]

# After: 1 fused matrix multiply + split
fused_weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
qkv = F.linear(x, fused_weight)
q = qkv[..., :q_size]
k = qkv[..., q_size:q_size+k_size]
v = qkv[..., q_size+k_size:]
```

**Gate+Up Fusion (2 GEMMs -> 1) for MLP:**
```python
# Before: 2 separate matrix multiplies
gate = self.gate_proj(x)
up = self.up_proj(x)
out = gelu(gate) * up

# After: 1 fused matrix multiply + fused activation
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

        mlp = layer.mlp
        fused = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        mlp.register_buffer("_fused_gate_up_weight", fused)
        mlp._use_fused = True
```

### 2.5 Environment Variable Convention

All optimizations should be toggleable via environment variables for A/B testing:

```python
# Standard pattern for optimization toggles
USE_AITER_ATTENTION = os.environ.get("USE_AITER_ATTENTION", "0") == "1"
USE_FUSED_PROJECTIONS = os.environ.get("USE_FUSED_PROJECTIONS", "0") == "1"
USE_OPTIMIZED_OPS = os.environ.get("USE_OPTIMIZED_OPS", "0") == "1"
USE_AITER_GEMM = os.environ.get("USE_AITER_GEMM", "0") == "1"
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
```

Default to OFF ("0") so the codebase remains compatible with non-AMD systems.

---

## Phase 3: Benchmarking & Validation (Prove It Works)

Goal: Quantify performance gains and verify numerical correctness.

### 3.1 Precision Verification Script (Write First)

**Always verify correctness before benchmarking performance.** Compare optimized outputs against eager baseline.

```python
#!/usr/bin/env python3
"""Template: verify optimized kernels match eager baseline."""

import torch

def verify_precision(model, create_input_fn, optimize_fn, device="cuda"):
    """
    Compare model outputs with and without optimizations.

    Args:
        model: The model to test
        create_input_fn: Callable that returns deterministic inputs
        optimize_fn: Callable that enables optimizations on the model
    """
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

    # Thresholds for BF16
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
    assert max_diff < 1.0, f"Max diff too high: {max_diff}"
    assert not has_nan, "NaN detected in optimized output"
    print("PASSED")
```

### 3.2 Inference Benchmark Script

Measures end-to-end inference latency. The baseline is the naive (eager) PyTorch running on AMD, and the comparison target is the optimized version.

```python
#!/usr/bin/env python3
"""Template: inference latency benchmark."""

import os
import time
import numpy as np
import torch

def benchmark_inference(model, create_input_fn, device="cuda",
                        warmup=10, iterations=30):
    """
    Benchmark inference latency with proper GPU synchronization.

    Returns dict with: mean_ms, std_ms, p50_ms, p95_ms, throughput_hz, memory_gb
    """
    model.eval()
    inputs = create_input_fn(device)

    # Warmup (critical for torch.compile and GPU clock stabilization)
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
- Warmup is mandatory: first iterations trigger compilation, kernel autotuning, GPU boost
- Report P50/P95, not just mean (captures variance)

### 3.3 Training Throughput Benchmark Script

Measures samples/second during training with full forward + backward + optimizer step.

```python
#!/usr/bin/env python3
"""Template: training throughput benchmark."""

import time
import torch

def benchmark_training(model, create_input_fn, device="cuda",
                       warmup=5, iterations=20):
    """
    Benchmark training throughput (samples/second).

    Includes: forward pass, loss computation, backward pass, optimizer step.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = create_input_fn(device, for_training=True)
    batch_size = inputs["input"].shape[0]

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = model(**inputs)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Benchmark
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

### 3.4 Multi-GPU DDP Benchmark Script

Measures scaling efficiency across multiple GPUs.

```python
#!/usr/bin/env python3
"""Template: multi-GPU DDP training benchmark."""
# Launch with: torchrun --nproc_per_node=NUM_GPUS script.py

import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend="nccl")  # Works for both NVIDIA and AMD
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def benchmark_ddp(model_fn, create_input_fn, warmup=5, iterations=20):
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    model = model_fn().to(device).to(torch.bfloat16)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    inputs = create_input_fn(device, for_training=True)
    batch_size = inputs["input"].shape[0]

    # Warmup
    model.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = model(**inputs)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = model(**inputs)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    total_samples = batch_size * world_size * iterations
    if rank == 0:
        print(f"Throughput: {total_samples / elapsed:.1f} samples/s")
        print(f"Step time:  {elapsed / iterations * 1000:.1f} ms")

    dist.destroy_process_group()
```

**Test configurations to sweep:**

| Batch/GPU | Total Batch | Seq Len | Purpose |
|-----------|-------------|---------|---------|
| 4 | 4 * N_GPU | 512 | Small batch baseline |
| 8 | 8 * N_GPU | 512 | Medium batch |
| 8 | 8 * N_GPU | 1024 | Long sequence |
| 16 | 16 * N_GPU | 512 | Large batch (peak throughput) |

### 3.5 Profiling with torch.profiler

Generate Perfetto-compatible traces for kernel-level analysis:

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    # Run a few iterations
    for _ in range(5):
        model(**inputs)
    torch.cuda.synchronize()

prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

View traces at [Perfetto UI](https://ui.perfetto.dev/).

### 3.6 Benchmark Reporting Template

Structure benchmark results as a comparison table:

```markdown
## Results: Eager vs Optimized (AMD MI-300X)

### Inference (batch=1)

| Configuration | Latency | Throughput | Memory |
|---------------|---------|------------|--------|
| Eager baseline | XX ms | X.X Hz | X.X GB |
| + torch.compile | XX ms | X.X Hz | X.X GB |
| + aiter attention | XX ms | X.X Hz | X.X GB |
| + Triton kernels | XX ms | X.X Hz | X.X GB |
| + fused projections | XX ms | X.X Hz | X.X GB |
| **All optimizations** | **XX ms** | **X.X Hz** | X.X GB |

### Training (8-GPU DDP)

| Batch/GPU | Eager | Optimized | Speedup |
|-----------|-------|-----------|---------|
| 4 | X samples/s | X samples/s | X.Xx |
| 8 | X samples/s | X samples/s | X.Xx |
| 16 | X samples/s | X samples/s | X.Xx |
```

---

## Common Pitfalls

1. **Don't replace `torch.cuda` with `torch.hip`** - The `torch.cuda` API is the correct abstraction on both platforms.

2. **Don't skip warmup in benchmarks** - First iterations include JIT compilation, autotuning, and GPU clock ramp-up. Always warm up 5-20 iterations.

3. **Don't forget `torch.cuda.synchronize()`** - GPU ops are asynchronous. Without sync, you measure CPU dispatch time, not GPU execution time.

4. **Don't use Triton custom ops inside DDP** - Custom Triton kernels with non-standard parameters may break DDP gradient synchronization. Use `nn.RMSNorm` etc. in DDP, save Triton ops for single-GPU or inference.

5. **Don't assume all attention masks work with flash attention** - Flash attention only supports pure causal and full bidirectional masks. Always implement fallback to eager for complex masks.

6. **Don't hardcode compilation paths** - Always make optimizations toggleable via environment variables so the same codebase runs on both NVIDIA and AMD.

7. **Always verify precision first** - Run the precision verification script before reporting any performance numbers. Optimized code that produces wrong results is useless.

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
  benchmark.md                # Results documentation
```
