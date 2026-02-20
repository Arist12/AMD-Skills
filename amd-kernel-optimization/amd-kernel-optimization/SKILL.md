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
| `torch.compile(mode="default")` | Safe baseline; Triton fusion for elementwise | — |
| `torch.compile(mode="reduce-overhead")` | Adds CUDAGraph capture | CUDAGraph instability on ROCm; test carefully |
| `torch.compile(mode="max-autotune")` | Benchmarks multiple backends per op | Longer compile; Triton GEMM autotuning may not help |
| **Manual CUDAGraph capture** | Capture full call as one graph | Needs Dynamo RNG patch on ROCm |
| **Eager (no compile)** | No compilation overhead | Misses fusion opportunities |

See [torch-compile-and-graphs.md](references/torch-compile-and-graphs.md).

## Workflow

1. **Profile** — Use `torch.profiler` to generate a chrome trace (or `rocprofv3` for hardware counters). Identify the hottest kernels and classify them (GEMM / elementwise / attention / other).

2. **Benchmark alternatives** — For each hot kernel category, benchmark the relevant alternatives listed above on your actual workload. Pick what performs best for your shapes and batch sizes.

3. **Fuse operations** — Write Triton kernels for elementwise fusion targets. Fuse linear projections (QKV, Gate+Up) to reduce GEMM count.

4. **Configure torch.compile** — Start with ROCm-safe inductor defaults. Compile through vendor ops to minimize graph breaks. Benchmark compile modes.

5. **Capture CUDAGraph** (optional) — If kernel launch overhead is significant, try manual full-call CUDAGraph capture. Apply the Dynamo RNG patch first on ROCm.

6. **Re-profile and iterate** — Measure again, find the next bottleneck, repeat.

## Reference Files

Read these as needed for implementation details:

- **[gemm-and-linear.md](references/gemm-and-linear.md)** — GEMM backend APIs, aiter tuned GEMM usage, projection fusion patterns, weight preshuffling, bias splitting, nn.Linear monkey-patching, attention backends
- **[triton-on-rocm.md](references/triton-on-rocm.md)** — Writing Triton kernels for ROCm, platform gotchas (tanh, bf16), code examples for RMSNorm, SiLU+Mul, GELU+Mul, Add+RMSNorm
- **[torch-compile-and-graphs.md](references/torch-compile-and-graphs.md)** — torch.compile modes and known ROCm issues, inductor config, compiling through vendor ops, manual CUDAGraph capture, Dynamo RNG patch, HIP env vars, profiling
