# GEMM Optimization on AMD GPUs

## Table of Contents
- [Backend Hierarchy](#backend-hierarchy)
- [Inductor GEMM Configuration](#inductor-gemm-configuration)
- [aiter Tuned GEMM Dispatcher](#aiter-tuned-gemm-dispatcher)
- [Weight Preshuffling](#weight-preshuffling)
- [Bias Splitting](#bias-splitting)
- [Per-Shape Tuning](#per-shape-tuning)

## Backend Hierarchy

On MI300X/MI350X, GEMM backend performance (fastest to slowest):

1. **aiter asm bshuffle kernels** - 20-32% faster than rocBLAS for medium M (128-1024)
2. **hipBLASLt** - Best for large N or non-standard shapes
3. **rocBLAS (ATen)** - Reliable general-purpose (35-55% faster than Triton)
4. **Triton GEMM** - Slowest for GEMMs on AMD (but best for elementwise)

The optimal backend varies by shape. Use per-shape dispatch when possible.

## Inductor GEMM Configuration

```python
import torch._inductor.config as inductor_config

# Force rocBLAS for all GEMMs (simplest, most reliable)
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False

# Alternative: allow autotuning between ATEN and TRITON
# (rarely useful - ATEN almost always wins on AMD)
# inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"
# inductor_config.max_autotune = True
```

## aiter Tuned GEMM Dispatcher

The `aiter` library provides `tuned_gemm` which routes each GEMM to the best kernel based on (M, N, K) shape:

```python
from aiter.tuned_gemm import TunedGemm

# Initialize with tuning config (CSV of shape -> kernel mappings)
tuned = TunedGemm()
tuned.load_config("tuned_gemm_config.csv")

# BF16 GEMM with automatic kernel selection
output = tuned.gemm_a16w16(input_tensor, weight_tensor)
```

### Tuning Config Format (CSV)

```csv
M,N,K,lib,kernel_name,tflops,bandwidth_gb_s
532,2048,2048,asm,bf16gemm_fp32bf16_tn_96x64_pf3_splitk,266.78,761.98
532,2048,16384,hipblaslt,,540.44,1312.74
788,2048,2048,asm,bf16gemm_fp32bf16_tn_128x64_bshuffle,423.0,950.0
```

### Key Per-Shape Insights

- **M < 64**: rocBLAS or hipBLASLt (asm kernels struggle with very small M)
- **64 <= M <= 1024**: asm bshuffle kernels often 20-32% faster
- **Large N (e.g., N=32768)**: hipBLASLt consistently best
- **Non-divisible-by-256 shapes**: Cannot use asm kernels, fall back to hipBLASLt/rocBLAS

### Integration with F.linear

To route `F.linear` calls through aiter without modifying model code:

```python
def aiter_linear(x, weight, bias=None):
    """Drop-in replacement for F.linear using aiter tuned GEMM."""
    output = tuned_gemm.gemm_a16w16(x, weight.t())
    if bias is not None:
        output = output + bias
    return output

# Monkey-patch or use in model forward
```

## Weight Preshuffling

asm kernels support weight preshuffling (`shuffle_weight`) for better memory access patterns:

```python
from aiter.ops.shuffle import shuffle_weight

# Preshuffle weight for asm kernel path
shuffled_weight = shuffle_weight(weight, layout="tn")
```

**Requirements:**
- N and K must be divisible by 256
- Weight must be contiguous in the expected layout

**Critical caveat**: Preshuffled weights **regress small-M shapes** (e.g., M=1 decode). Use threshold-based dispatch:

```python
PRESHUFFLE_M_THRESHOLD = 128  # Only use preshuffled for M >= threshold

def smart_gemm(x, weight, shuffled_weight=None):
    M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
    if shuffled_weight is not None and M >= PRESHUFFLE_M_THRESHOLD:
        return tuned_gemm.gemm_a16w16(x, shuffled_weight)
    else:
        return tuned_gemm.gemm_a16w16(x, weight)  # Un-shuffled
```

**Best practice for B=1 inference**: Keep global preshuffling OFF. Small-M shapes dominate B=1 latency, and preshuffling hurts these.

## Bias Splitting

Some asm kernels run faster without bias. Split bias as a post-GEMM epilogue:

```python
# Instead of: output = F.linear(x, weight, bias)
# Do:
output = gemm_a16w16(x, weight)  # Bias-free GEMM (may use faster asm kernel)
output = output + bias            # Epilogue (can fuse with Inductor)
```

This allows Inductor's `epilogue_fusion` to fuse the bias add with subsequent operations.

## Per-Shape Tuning Workflow

1. **Profile** to identify all GEMM shapes in the model:
   ```python
   # Log shapes during a forward pass
   original_mm = torch.mm
   def logging_mm(a, b):
       print(f"GEMM: M={a.shape[0]}, N={b.shape[1]}, K={a.shape[1]}")
       return original_mm(a, b)
   torch.mm = logging_mm
   ```

2. **Benchmark** each shape across backends:
   ```python
   for M, N, K in shapes:
       a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
       b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
       # Benchmark: rocBLAS, hipBLASLt, asm, Triton
   ```

3. **Build tuning config** mapping each shape to its fastest backend

4. **Load config** into the aiter tuned GEMM dispatcher
