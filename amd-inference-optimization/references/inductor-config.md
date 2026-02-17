# Complete Inductor Configuration for ROCm

## Table of Contents
- [Full Configuration Block](#full-configuration-block)
- [GEMM Settings](#gemm-settings)
- [Fusion Settings](#fusion-settings)
- [Triton Settings](#triton-settings)
- [Memory and Stability](#memory-and-stability)
- [Debugging](#debugging)

## Full Configuration Block

Apply before any `torch.compile()` call:

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# ── GEMM Backend ──
inductor_config.max_autotune_gemm_backends = "ATEN"  # rocBLAS >> Triton
inductor_config.max_autotune = False                  # Skip autotune (ATEN always wins)

# ── CUDAGraph (Inductor-level) ──
inductor_config.triton.cudagraphs = False  # MUST be False on ROCm

# ── Fusion ──
inductor_config.epilogue_fusion = True      # Fuse post-GEMM pointwise ops
inductor_config.pattern_matcher = True      # Pattern-based fusion
inductor_config.aggressive_fusion = True    # Aggressively fuse kernels
inductor_config.group_fusion = True         # Group related operations
inductor_config.max_fusion_size = 128       # Larger fused kernels (default: 64)

# ── Triton Kernel Tuning ──
inductor_config.coordinate_descent_tuning = True  # Better block sizes/warps
inductor_config.benchmark_kernel = True            # Benchmark each variant
inductor_config.triton.multi_kernel = 1            # Multi-kernel selection
inductor_config.shape_padding = True               # Pad for better Triton perf

# ── Weight Freezing (Inference) ──
inductor_config.freezing = True  # Constant-fold weights into compiled graph

# ── Memory ──
inductor_config.memory_planning = False  # DISABLED: deep recursion on ROCm

# ── Dynamo ──
dynamo_config.cache_size_limit = 256  # Increase for complex models
```

## GEMM Settings

### `max_autotune_gemm_backends = "ATEN"`

Forces all GEMMs through ATen (rocBLAS/hipBLASLt). This is the single most impactful Inductor setting on AMD.

- `"ATEN"`: rocBLAS only (35-55% faster than Triton on AMD)
- `"TRITON"`: Triton only (worse for GEMMs, better fusion potential)
- `"ATEN,TRITON"`: Autotune between both (slow compilation, marginal benefit)

### `max_autotune = False`

When `max_autotune_gemm_backends = "ATEN"`, autotuning adds compilation time with no benefit. Disable it.

## Fusion Settings

### `epilogue_fusion = True`
Fuses pointwise operations after GEMMs (bias add, activation, etc.) into the GEMM epilogue. Critical for reducing kernel count.

### `aggressive_fusion = True`
Enables more aggressive fusion heuristics. May increase individual kernel size but reduces total kernel count.

### `group_fusion = True`
Groups related operations and fuses them together. Helps with patterns like multiple parallel linear layers.

### `max_fusion_size = 128`
Maximum number of nodes in a fused kernel. Default is 64. Increasing to 128 allows larger fused kernels, reducing launch overhead. Values above 128 show diminishing returns.

### `pattern_matcher = True`
Enables pattern-based graph rewriting. Recognizes common patterns (like `x * sigmoid(x)` for SiLU) and replaces with optimized implementations.

## Triton Settings

### `coordinate_descent_tuning = True`
After initial autotuning, runs coordinate descent on Triton kernel configs (block sizes, num_warps, num_stages). Increases first compile time by ~2-3x but finds better steady-state configs.

### `benchmark_kernel = True`
Benchmarks each Triton kernel variant at compile time. More accurate than heuristic selection.

### `triton.multi_kernel = 1`
Generates multiple kernel variants and selects the best at runtime. Small compile-time cost, can find better solutions.

### `shape_padding = True`
Pads tensor dimensions to multiples that are friendly for Triton tile sizes. Small memory overhead, noticeable speedup for odd shapes.

## Memory and Stability

### `memory_planning = False` (CRITICAL on ROCm)
Inductor's memory planning triggers deep recursion on ROCm, causing crashes or hangs. **Always disable on ROCm.**

### `freezing = True`
Constant-folds model weights into the compiled graph. Enables additional optimizations (dead code elimination, constant propagation). Only safe for inference (weights don't change).

## Debugging

```python
# Enable Inductor logging for debugging
import logging
torch._inductor.config.trace.enabled = True
logging.getLogger("torch._inductor").setLevel(logging.DEBUG)

# Print generated Triton code
inductor_config.trace.debug_log = True

# Count kernels in compiled graph
# Look for "compiled X kernels" in debug output
```

## Environment Variable Overrides

All settings can be controlled via environment variables for A/B testing:

```bash
# Example: toggle settings without code changes
INDUCTOR_COORD_DESCENT=1        # coordinate_descent_tuning
INDUCTOR_BENCHMARK_KERNEL=1     # benchmark_kernel
INDUCTOR_GROUP_FUSION=1         # group_fusion
INDUCTOR_AGGRESSIVE_FUSION=1    # aggressive_fusion
INDUCTOR_MAX_FUSION_SIZE=128    # max_fusion_size
INDUCTOR_FREEZING=1             # freezing
```

Pattern for reading env vars:

```python
inductor_config.coordinate_descent_tuning = (
    os.environ.get("COORD_DESCENT", "1") == "1"
)
inductor_config.max_fusion_size = int(
    os.environ.get("MAX_FUSION_SIZE", "128")
)
```
