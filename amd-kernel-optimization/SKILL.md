---
name: amd-kernel-optimization
description: >
  Domain knowledge for kernel-level optimization on AMD GPUs (MI250/MI300/MI350) with PyTorch and ROCm.
  Use when optimizing latency or throughput of PyTorch models on AMD hardware: selecting GEMM backends,
  writing Triton kernels, configuring torch.compile, capturing CUDAGraphs, or tuning attention kernels.
  Teaches what AMD-specific alternatives exist to explore and how to use them.
---

# AMD Kernel Optimization (ROCm)

## AMD-Specific Alternatives to Explore

When optimizing PyTorch on AMD GPUs, these are the AMD-specific options available per operator category. **Always benchmark on your specific workload and shapes** — relative performance depends heavily on model architecture, sequence lengths, batch sizes, and GEMM shapes.

### GEMM / Linear Layers

| Alternative | What It Offers |
|-------------|---------------|
| **rocBLAS** (default ATen backend) | AMD's vendor-optimized BLAS; generally well-tuned for AMD hardware |
| **hipBLASLt** | Lightweight BLAS with fused epilogues; may outperform rocBLAS for some shapes |
| **aiter tuned GEMM** | AMD's auto-dispatcher; selects best kernel (asm/hipBLASLt/skinny/torch) per (M,N,K) shape from tuned configs |
| **Triton GEMM** | Cross-platform; may lag behind vendor BLAS on AMD but worth benchmarking |
| **CK (Composable Kernel)** | AMD's template-based kernel library; offers hand-tuned GEMM and attention kernels |
| **FP8 GEMM** (MI300+) | Quantized GEMM using E4M3/E5M2; available via aiter (`gemm_a8w8`) |

Additional GEMM strategies: projection fusion (QKV, Gate+Up), weight preshuffling for asm paths, bias splitting.
See [gemm-and-linear.md](references/gemm-and-linear.md).

### Elementwise / Reduction Ops

| Alternative | What It Offers |
|-------------|---------------|
| **Custom Triton kernels** | Fuse multiple ops into one kernel; reduces launch overhead and memory traffic |
| **CK kernels** | Pre-built fused kernels for common patterns |
| **Eager PyTorch** | Baseline; multiple separate kernel launches |

High-value Triton fusion targets: RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm, Add+LayerNorm.
See [triton-on-rocm.md](references/triton-on-rocm.md).

### Attention

| Alternative | What It Offers |
|-------------|---------------|
| **aiter flash attention** | AMD-optimized FA; supports GQA/MQA natively; `torch.ops` path for compile-friendliness |
| **SDPA** (`F.scaled_dot_product_attention`) | PyTorch built-in; dispatches to available backends; good for KV-cache decode |
| **CK flash attention** | AMD's Composable Kernel FA implementation |
| **Manual bmm+softmax+bmm** | Explicit implementation; typically the slowest but most flexible |

See attention section in [gemm-and-linear.md](references/gemm-and-linear.md).

### Compilation & Graph Capture

| Alternative | What It Offers | Known ROCm Issues |
|-------------|---------------|-------------------|
| `torch.compile(mode="default")` | **Recommended first.** Enables Triton fusion for elementwise ops — often the single largest speedup on ROCm. Requires inductor overrides from system prompt (the container defaults to `max_autotune=True` which causes hangs). | Stable on ROCm with correct inductor config |
| `torch.compile(mode="reduce-overhead")` | Adds CUDAGraph capture on top of `default` | **Depends on CUDAGraphs, which must be disabled on ROCm per system prompt rules.** Only attempt after `default` mode is confirmed working AND you intentionally re-enable inductor CUDAGraphs. Expect instability. |
| `torch.compile(mode="max-autotune")` | Benchmarks multiple GEMM backends per op | Triggers Triton GEMM autotuning that hangs on ROCm; avoid unless you have verified specific shapes work |
| **Manual CUDAGraph capture** | Capture full call as one graph | Needs Dynamo RNG patch on ROCm |
| **Eager (no compile)** | No compilation overhead | Misses fusion opportunities |

See [torch-compile-and-graphs.md](references/torch-compile-and-graphs.md).

## Workflow

Follow these steps in order. Each step builds on the previous one. Do NOT skip steps or defer them to "future work."

### Step 1: Profile and categorize

Use `torch.profiler` to generate a trace of the target workload:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    model_fn(inputs)
prof.export_chrome_trace("trace.json")
# Then use prof.key_averages() to print a table sorted by cuda_time_total
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

Categorize GPU time into: GEMM (%), attention (%), elementwise/normalization (%), kernel launch overhead (%), other (%). This breakdown determines your optimization priority.

### Step 2: Look for compute-reduction opportunities

Before optimizing kernels, check whether the workload does unnecessary compute:
- **Masked/padding inputs**: If some inputs are fully masked (e.g., padding tokens, dummy inputs, or fully-masked regions), skip processing them entirely rather than running compute on dummy data.
- **Redundant tensor ops**: Look for `repeat_kv` (GQA head expansion), unnecessary `.contiguous()` calls, or tensor copies that can be avoided.
- **Caching**: If the model has a sub-network or prefix whose output doesn't change across iterations, cache its output instead of recomputing.

### Step 3: Locate the inner model layers you will modify

Most compute in transformer models lives in attention and MLP modules inside third-party libraries (e.g., HuggingFace `transformers`). You MUST find and modify these files. To locate them:

```bash
python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))"
```

Then find the relevant `modeling_*.py` for the architecture your model uses. Read the attention `forward()` and MLP `forward()` methods. These are the functions you will modify in subsequent steps.

Alternatively, copy the file into the repo and adjust imports to use the local copy.

### Step 4: Write Triton kernels for elementwise ops

Create a file (e.g., `triton_ops.py`) with Triton kernels for the model's normalization and activation patterns. High-value targets:

- **RMSNorm**: Fuses variance computation, reciprocal sqrt, and weight scaling into one kernel. See `triton-on-rocm.md` for complete code.
- **Fused activation+mul** (SiLU+Mul or GELU+Mul, depending on the model's gated MLP activation): Fuses the activation function and element-wise multiply into one kernel. See `triton-on-rocm.md`.
- **Fused Add+RMSNorm**: Combines residual addition and normalization. See `triton-on-rocm.md`.

Then integrate by editing the inner model layer code:
```python
# In the MLP forward(), replace:
#   gate = self.activation(self.gate_proj(x))
#   out = gate * self.up_proj(x)
# With:
#   gate_up = F.linear(x, self._fused_gate_up_weight)
#   out = silu_and_mul_triton(gate_up)  # your Triton kernel
```

### Step 5: Fuse linear projections

Reduce GEMM count by fusing projections. Apply at model init time after weight loading:

**QKV fusion (3 GEMMs → 1):** In the attention module:
```python
# At init, after loading weights:
fused_qkv_weight = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
attn.register_buffer("_fused_qkv_weight", fused_qkv_weight)
attn._use_fused_qkv = True

# In attention forward():
if self._use_fused_qkv:
    qkv = F.linear(x, self._fused_qkv_weight)
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
```

**Gate+Up fusion (2 GEMMs → 1):** In the MLP module — same pattern, see `gemm-and-linear.md`.

### Step 6: Swap attention backend

Replace the attention computation in the inner attention forward with a faster backend:

**Option A — aiter flash attention** (compile-friendly path):
```python
output = torch.ops.aiter.mha_fwd.default(
    q, k, v, dropout_p=0.0, softmax_scale=scale,
    is_causal=is_causal, window_size_left=-1, window_size_right=-1,
    return_softmax_lse=False, return_dropout_randval=False,
)[0]
```

**Option B — SDPA** (good for KV-cache decode where q_len != k_len):
```python
output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
```

Benchmark both and pick the winner. For models with both prefill and decode phases, you may want different backends for each.

### Step 7: Configure torch.compile

Apply mandatory inductor overrides (from system prompt), then compile with `mode="default"`. This should be applied AFTER all the code-level optimizations above so that torch.compile can fuse the remaining elementwise ops around your optimized kernels.

### Step 8: Tune GEMM backend routing

Try monkey-patching `nn.Linear.forward` to route through aiter tuned GEMM:
```python
def _patched_forward(self, x):
    m = x.numel() // x.shape[-1]
    if m >= M_THRESHOLD and x.dtype == torch.bfloat16:
        return gemm_a16w16(x, self.weight, bias=self.bias, otype=x.dtype)
    return F.linear(x, self.weight, self.bias)
```

**Key: use M-threshold gating.** Small-M GEMMs (bsz=1 decode) are often faster with rocBLAS. Only route large-M GEMMs (prefill, fused projections) through aiter. Start with `M_THRESHOLD=64` and tune.

If preshuffling causes regression, disable it: `AITER_PRESHUFFLE_WEIGHTS=0`.

### Step 9: Attempt CUDAGraph capture

If profiling shows significant kernel launch overhead:

1. Apply the Dynamo RNG patch from `torch-compile-and-graphs.md` (the `patch_dynamo_for_rocm_capture()` function).
2. Refactor data-dependent loops: change `while condition:` to `for _ in range(fixed_count):`.
3. Pre-allocate tensors created inside the capture region as module buffers.
4. Attempt manual full-call capture:
```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_output = model_fn(static_input)
# In hot loop: graph.replay()
```

If capture fails, report the specific error and what you tried. Most failures have known workarounds documented in the skill reference.

### Step 10: Compose and re-measure

Apply all successful optimizations together. Measure E2E latency. Re-profile to see if the bottleneck distribution has shifted. If a new bottleneck emerges, address it.

## Common Pitfalls

- **Shallow profiling**: Generating a trace and looking at "top 10 kernels" is not enough. Classify kernels by category (GEMM / attention / elementwise / other) and compute time percentages to understand which optimization category will have the highest impact.
- **Testing only in isolation**: Testing one optimization at a time in a separate script and reverting each one that doesn't individually help misses that optimizations compose. A technique that shows 0% improvement alone may enable other techniques or reduce overhead that only matters in combination. Build optimizations incrementally — each new technique is applied ON TOP of all previously accepted ones.
- **Giving up on first failure**: When a technique causes regression, the right response is to diagnose and adjust, not revert and abandon. The technique may be correct but misconfigured (wrong shapes, missing tuned configs, applied too broadly). For example, aiter GEMM monkey-patching may regress because Python dispatch overhead dominates for small-M shapes — the fix is M-threshold gating, not abandoning the technique entirely.
- **Treating blockers as dead ends**: "CUDAGraph doesn't support control flow" means "refactor the control flow," not "CUDAGraph is impossible." "Requires HuggingFace modeling file rewrite" means "go edit the modeling file" — that IS the optimization work. Most reported blockers for CUDAGraph on ROCm have known workarounds (loop refactoring, pre-allocating tensors, Dynamo RNG patch).
- **Not modifying inner model layers**: If the optimization target is a transformer, the hottest code is in the attention and MLP modules (often in third-party libraries like `transformers`). You MUST find these files (use `python -c "import transformers; print(transformers.__file__)"` to locate them) and edit the attention `forward()`, MLP `forward()`, and normalization calls directly. Leaving these untouched and only changing the outer calling code will not produce meaningful speedups.
- **Confusing "code changes required" with "impossible"**: Every optimization technique in this skill requires code changes — that's the entire point. Projection fusion requires concatenating weights and splitting outputs in the forward method (~10 lines). Attention backend swaps require replacing one function call with another (~5 lines). Triton kernels require copying the template from this skill's references and adapting dimensions (~30 lines). None of these are "major rewrites." If you find yourself listing techniques as "not attempted because they require code changes," you are not doing the optimization work.

## Reference Files

Read these as needed for implementation details:

- **[gemm-and-linear.md](references/gemm-and-linear.md)** — GEMM backend APIs, aiter tuned GEMM usage, projection fusion patterns, weight preshuffling, bias splitting, nn.Linear monkey-patching, attention backends
- **[triton-on-rocm.md](references/triton-on-rocm.md)** — Writing Triton kernels for ROCm, platform gotchas (tanh, bf16), code examples for RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm
- **[torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)** — torch.compile modes and known ROCm issues, inductor config, compiling through vendor ops, manual CUDAGraph capture, Dynamo RNG patch, HIP env vars, profiling
