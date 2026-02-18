---
name: amd-inference-optimization
description: >
  Systematic kernel-level inference latency optimization for PyTorch models on AMD GPUs
  (MI300X, MI355X, ROCm series, etc..). Use when asked to optimize, profile, or reduce inference latency
  on AMD/ROCm hardware. Covers: GEMM backend selection (rocBLAS vs Triton), CUDAGraph/HIP
  graph capture, attention optimization (aiter flash attention, SDPA fast-path), Triton kernel
  authoring for AMD wavefronts, Inductor configuration for ROCm, kernel fusion strategies,
  synchronization overhead reduction, and profiling workflows. Focuses on single-batch (bsz=1)
  latency, not training or throughput. Triggered by: "optimize inference on AMD/MI300/MI355",
  "reduce latency on ROCm", "profile PyTorch on AMD GPU", "torch.compile on ROCm",
  "CUDAGraph on AMD", "Triton kernels for AMD".
---

# AMD GPU Inference Latency Optimization

Systematic workflow for reducing PyTorch inference latency on AMD GPUs (MI300X/MI355X/ROCm). Ordered by impact. Each phase builds on the previous.

## Optimization Ladder (Reference Impact)

On a representative transformer model, these optimizations reduced B=1 latency by 2.66x:

| Phase | Optimization | Typical Impact |
|-------|-------------|---------------|
| 0 | Profile baseline | - |
| 1 | CUDAGraph/HIP graph replay | -30-50% (largest single win) |
| 2 | SDPA attention fast-path | -5-10% |
| 3 | GEMM backend selection (rocBLAS/aiter) | -3-5% |
| 4 | Projection fusion (QKV, Gate+Up) | -2-3% |
| 5 | Inductor fusion + custom Triton kernels | -2-3% |
| 6 | Fine-tuning (coord descent, benchmarking) | -3-5% |

## Phase 0: Profile Baseline

Before optimizing, establish measurement infrastructure.

**GPU-side timing with CUDA events** (preferred - avoids device-wide sync overhead):
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
model_output = model(inputs)
end.record()
torch.cuda.synchronize()
latency_ms = start.elapsed_time(end)
```

**Kernel-level profiling** - see [references/profiling-workflow.md](references/profiling-workflow.md).

Collect: total latency, kernel count, GEMM time fraction, sync overhead, launch overhead.

## Phase 1: CUDAGraph / HIP Graph Capture

Largest single optimization. Eliminates CPU-side kernel launch overhead by capturing the entire GPU execution graph and replaying it. Typically -30-50% latency; on multi-step models (e.g., flow-matching denoise loops), can reach -60%.

**CRITICAL DECISION — Do NOT combine torch.compile with CUDAGraph on ROCm:**

| Approach | ROCm Status | Why |
|----------|-------------|-----|
| `torch.compile(mode="reduce-overhead")` | **BROKEN** | Stalls 20+ min on Triton autotune, 65x slowdown |
| `torch.compile(mode="default")` + manual graph | **CONFLICTS** | Double-graph semantics, tensor ownership errors |
| **Eager model + manual `torch.cuda.CUDAGraph()`** | **PROVEN** | Instantaneous capture, same or better latency |

The proven path: **do NOT use torch.compile at all for the CUDAGraph path**. Use the eager (uncompiled) model with manual `torch.cuda.CUDAGraph()` capture. This was validated on a 3.5B transformer where manual capture achieved the same latency as torch.compile but without the 20+ minute compilation stall.

**Incremental capture strategy** (do NOT try to capture everything at once):

- **Step A**: Capture ONLY the hot loop (e.g., denoise loop). Pre-compute prefix/constants outside the graph. Test correctness.
- **Step B**: Only after Step A works and passes correctness, expand to capture the full pipeline (prefix + loop).
- **Step C**: Step B requires DynamicCache bypass — see [references/dynamiccache-bypass.md](references/dynamiccache-bypass.md).

**Manual CUDAGraph capture pattern (eager model, no torch.compile):**
```python
# NO torch.compile — capture on the eager model directly
model.eval()

# 1. Pre-allocate static input buffers (fixed shapes, fixed memory addresses)
static_inputs = [inp.clone() for inp in example_inputs]

# 2. Warm up (run model a few times to stabilize memory layout)
with torch.no_grad():
    for _ in range(3):
        _ = model(*static_inputs)
torch.cuda.synchronize()

# 3. Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model(*static_inputs)
torch.cuda.synchronize()

# 4. Replay (in inference loop)
for inp in static_inputs:
    inp.copy_(new_data)  # Copy into static buffers (same memory addresses)
graph.replay()
result = static_output.clone()
```

**ROCm Dynamo RNG patch** (required if any torch.compile is used elsewhere):

The module path is `torch._dynamo.convert_frame`, **NOT** `torch._dynamo.utils`. Apply before any graph capture:

```python
import torch._dynamo.convert_frame as _convert_frame

# Guard torch.cuda.get_rng_state() calls with:
# if not torch.cuda.is_current_stream_capturing():
```

See [references/cudagraph-integration.md](references/cudagraph-integration.md) for the complete inline integration pattern.
See [references/cudagraph-strategy.md](references/cudagraph-strategy.md) for the Dynamo RNG patch and memory pool details.
See [references/dynamiccache-bypass.md](references/dynamiccache-bypass.md) for HuggingFace DynamicCache replacement.

## Phase 2: Attention Optimization

**SDPA fast-path for KV-cache attention** (q_len != k_len):

When query and key lengths differ (common in KV-cache inference), `F.scaled_dot_product_attention` uses an optimized backend. Without this, attention falls back to explicit `bmm + softmax + bmm`:

```python
# GOOD: Use F.scaled_dot_product_attention (dispatches to optimized backend)
attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

# BAD: Manual bmm path (much slower)
attn_weights = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
attn_weights = F.softmax(attn_weights, dim=-1)
attn_output = torch.bmm(attn_weights, v)
```

**aiter flash attention** (AMD-optimized):

If the `aiter` library is available, use its direct MHA op for torch.compile compatibility:

```python
try:
    # Direct MHA call - torch.compile friendly (no graph break)
    out = torch.ops.aiter.mha_fwd(q, k, v, dropout_p=0.0, softmax_scale=scale,
                                    is_causal=is_causal)
except:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
```

**Critical**: Compile *through* attention. Do not graph-break around attention ops. Graph breaks cause ~10ms regression. Ensure aiter attention is included in the compiled graph, not wrapped with `torch.compiler.disable`.

## Phase 3: GEMM Backend Selection

On AMD GPUs, **rocBLAS/hipBLASLt >> Triton for GEMMs** (35-55% faster). GEMMs typically account for 50-60% of transformer inference compute.

```python
# Force rocBLAS (ATEN backend) for all GEMMs
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False  # rocBLAS always wins, skip autotuning
```

**aiter tuned GEMM dispatcher** (per-shape kernel selection):

The `aiter` library provides a tuned GEMM dispatcher that selects the optimal kernel (asm/hipblaslt/skinny) per GEMM shape:

```python
from aiter.tuned_gemm import gemm_a16w16

# Routes to best kernel for each (M, N, K) shape:
# - asm bshuffle kernels: 20-32% faster for medium M (128-1024)
# - hipblaslt: best for large N or non-standard shapes
# - torch mm (rocBLAS): reliable fallback
output = gemm_a16w16(input, weight)
```

See [references/gemm-optimization.md](references/gemm-optimization.md) for weight preshuffling, bias splitting, and per-shape tuning.

## Phase 4: Projection Fusion

Reduce kernel count by fusing multiple GEMMs into one:

```python
# Fuse QKV projections: 3 GEMMs -> 1
# Before: q = F.linear(x, W_q); k = F.linear(x, W_k); v = F.linear(x, W_v)
fused_qkv_weight = torch.cat([W_q, W_k, W_v], dim=0)
qkv = F.linear(x, fused_qkv_weight)
q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)

# Fuse Gate+Up projections (for SwiGLU/GeGLU MLPs): 2 GEMMs -> 1
fused_gate_up_weight = torch.cat([W_gate, W_up], dim=0)
gate_up = F.linear(x, fused_gate_up_weight)
gate, up = gate_up.chunk(2, dim=-1)
output = activation(gate) * up
```

Also batch similar operations (e.g., batch multiple images into one forward pass for vision encoders).

## Phase 5: Inductor Fusion + Custom Triton Kernels

**Inductor configuration for ROCm** - apply all of these:

```python
inductor_config.epilogue_fusion = True       # Fuse post-GEMM pointwise ops
inductor_config.aggressive_fusion = True     # Aggressively fuse kernels
inductor_config.group_fusion = True          # Fuse groups of operations
inductor_config.max_fusion_size = 128        # Allow larger fused kernels
inductor_config.shape_padding = True         # Pad for better Triton perf
inductor_config.pattern_matcher = True       # Enable pattern-based fusion
inductor_config.freezing = True              # Constant-fold weights (inference)
inductor_config.memory_planning = False      # DISABLED on ROCm (deep recursion bug)
```

See [references/inductor-config.md](references/inductor-config.md) for the complete configuration.

**Custom Triton kernels** for elementwise/fused ops (where Triton excels on AMD):

| Kernel | Typical Speedup vs Eager |
|--------|------------------------|
| RMSNorm | 4x |
| Fused Add + RMSNorm | 2.8x |
| Fused GELU(tanh) + Mul | 2.5x |
| Fused SiLU + Mul | 1.4x |

See [references/triton-kernels.md](references/triton-kernels.md) for AMD-specific kernel patterns.

**aiter CK activation kernels** (if available, 3x faster than Triton for some ops):
```python
try:
    output = torch.ops.aiter.gelu_tanh_and_mul(gate_up)  # CK-optimized
except:
    output = triton_gelu_and_mul(gate_up)  # Triton fallback
```

## Phase 6: Fine-Tuning

```python
inductor_config.coordinate_descent_tuning = True  # Better Triton configs (longer compile)
inductor_config.benchmark_kernel = True            # Benchmark each kernel variant
inductor_config.triton.multi_kernel = 1            # Enable multi-kernel selection
```

Coordinate descent tuning explores block sizes and num_warps per Triton kernel. Increases first compile time but improves steady-state latency.

**Dynamo cache**:
```python
torch._dynamo.config.cache_size_limit = 256  # Increase for complex models
```

## Phase 7: Verify and Measure

After each optimization phase, re-measure:
1. End-to-end latency (CUDA events, not wall-clock)
2. Kernel count (fewer = better; target: reduce by 15-20%)
3. Numerical correctness (compare outputs to baseline within tolerance)

## Anti-Patterns on ROCm

| Do NOT | Why | Do Instead |
|--------|-----|-----------|
| `torch.compile` + CUDAGraph together | Double-graph conflicts, 20+ min Triton autotune stall | Eager model + manual `torch.cuda.CUDAGraph()` |
| `torch.compile(mode="reduce-overhead")` | Broken on ROCm (65x slowdown) | Eager + manual CUDAGraph, or `mode="default"` without graph |
| `inductor_config.triton.cudagraphs = True` | Inductor graph capture broken on ROCm | Manual CUDAGraph capture |
| `inductor_config.memory_planning = True` | Deep recursion on ROCm | Keep `False` |
| `torch.cuda.synchronize()` (device-wide) | hipDeviceSynchronize is 3.7x more expensive than CUDA | `torch.cuda.current_stream().synchronize()` |
| Triton GEMM kernels | 35-55% slower than rocBLAS on AMD | `max_autotune_gemm_backends = "ATEN"` |
| Global weight preshuffling | Regresses small-M decode shapes | Threshold-based: only preshuffle when M >= threshold |
| Graph-breaking around attention | ~10ms regression per break | Compile through attention ops |

## Environment Variables (HIP Runtime)

```bash
export HIP_LAUNCH_BLOCKING=0    # Non-blocking kernel launches
export AMD_LOG_LEVEL=0           # Reduce logging overhead
export HIP_CACHE_ENABLED=1       # Enable HIP code object caching
export HSA_ENABLE_SDMA=1         # Enable SDMA engine
```

## References

- [CUDAGraph Integration](references/cudagraph-integration.md) - Inline capture pattern for flow-matching models, correctness verification
- [CUDAGraph Strategy](references/cudagraph-strategy.md) - Manual capture, Dynamo RNG patch, memory pool
- [DynamicCache Bypass](references/dynamiccache-bypass.md) - Replace HuggingFace DynamicCache with static KV buffers for graph capture
- [BSNH Layout Unification](references/bsnh-layout.md) - Eliminate tensor transposes by keeping BSNH format end-to-end
- [AITER Integration](references/aiter-integration.md) - Flash attention, tensor layout, GQA support
- [GEMM Optimization](references/gemm-optimization.md) - rocBLAS, aiter tuned GEMM, weight preshuffling, per-shape tuning
- [Inductor Configuration](references/inductor-config.md) - Complete torch._inductor.config for ROCm
- [Triton Kernels for AMD](references/triton-kernels.md) - RMSNorm, fused activations, AMD wavefront tuning
- [Profiling Workflow](references/profiling-workflow.md) - rocprof, torch.profiler, Chrome tracing
