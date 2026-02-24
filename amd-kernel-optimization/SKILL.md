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

1. **Profile and categorize** — Use `torch.profiler` to generate a chrome trace. Categorize where time is spent: GEMM (linear layers, projections), attention, elementwise/normalization, kernel launch gaps, and other overhead. Compute the percentage each category contributes. This breakdown determines which optimizations will have impact.

2. **Look for compute-reduction opportunities first** — Before optimizing kernels, check whether the workload is doing unnecessary compute. Common opportunities: skipping processing for masked/padding inputs, avoiding redundant tensor copies or expansions (e.g., `repeat_kv` for GQA), caching values that don't change across iterations.

3. **Apply optimizations at the right layer** — Most compute in transformer models lives in the inner attention and MLP modules, not in the outer model wrapper. Optimizations (attention backend swaps, fused activations, projection fusion) must be applied inside those inner modules to have effect. Modifying only the outer wrapper or entry point will not meaningfully change latency.

4. **Benchmark alternatives for each hot category** — For each hot kernel category, benchmark the relevant alternatives listed above. When a technique regresses, diagnose *why* before giving up — for example, if aiter GEMM is slower, check whether tuned configs exist for your shapes (generate them if not); if preshuffling hurts, try gating it by input M-dimension rather than disabling entirely.

5. **Fuse operations** — Write Triton kernels for elementwise fusion targets. Fuse linear projections (QKV, Gate+Up) to reduce GEMM count. Apply fused weights after loading but before `torch.compile`.

6. **Configure torch.compile** — Apply the mandatory inductor overrides from the system prompt (the container sets `max_autotune=True` by default, which causes hangs). Then use `mode="default"` — it enables Triton elementwise fusion and is the most impactful single optimization on ROCm. Treat `default` mode as a real optimization to benchmark, not just a baseline. Do NOT use `reduce-overhead` (requires CUDAGraphs, which are disabled) or `max-autotune` (triggers autotuning hangs) unless `default` is already confirmed working and you have a specific reason. Compile through vendor ops to minimize graph breaks.

7. **Capture CUDAGraph** (optional) — If kernel launch overhead is significant, try manual full-call CUDAGraph capture. Common blockers are solvable: `while` loops with data-dependent conditions can be refactored into fixed `for` loops; tensors created inside the loop can be pre-allocated as module buffers; the Dynamo RNG bug on ROCm has a known patch.

8. **Compose and re-profile** — Apply optimizations incrementally and measure the cumulative effect. Different techniques target different bottlenecks (e.g., Triton for elementwise, rocBLAS for GEMM, aiter for attention) and are designed to compose — test them together, not only in isolation.

## Common Pitfalls

- **Shallow profiling**: Generating a trace and looking at "top 10 kernels" is not enough. Classify kernels by category (GEMM / attention / elementwise / other) and compute time percentages to understand which optimization category will have the highest impact.
- **Testing only in isolation**: Testing one optimization at a time in a separate script and reverting each one that doesn't individually help misses that optimizations compose. A technique that shows 0% improvement alone may enable other techniques or reduce overhead that only matters in combination.
- **Giving up on first failure**: When a technique causes regression, the right response is to diagnose and adjust, not revert and abandon. The technique may be correct but misconfigured (wrong shapes, missing tuned configs, applied too broadly).
- **Treating blockers as dead ends**: "CUDAGraph doesn't support control flow" means "refactor the control flow," not "CUDAGraph is impossible." Most reported blockers for CUDAGraph on ROCm have known workarounds (loop refactoring, pre-allocating tensors, Dynamo RNG patch).
- **Not modifying inner model layers**: If the optimization target is a transformer, the hottest code is in the attention and MLP modules (often in third-party libraries like `transformers`). Leaving these untouched and only changing the outer calling code will not produce meaningful speedups.

## Reference Files

Read these as needed for implementation details:

- **[gemm-and-linear.md](references/gemm-and-linear.md)** — GEMM backend APIs, aiter tuned GEMM usage, projection fusion patterns, weight preshuffling, bias splitting, nn.Linear monkey-patching, attention backends
- **[triton-on-rocm.md](references/triton-on-rocm.md)** — Writing Triton kernels for ROCm, platform gotchas (tanh, bf16), code examples for RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm
- **[torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)** — torch.compile modes and known ROCm issues, inductor config, compiling through vendor ops, manual CUDAGraph capture, Dynamo RNG patch, HIP env vars, profiling
