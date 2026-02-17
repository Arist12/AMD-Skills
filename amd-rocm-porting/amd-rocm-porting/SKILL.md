---
name: amd-rocm-porting
description: >
  Port NVIDIA CUDA codebases to AMD ROCm GPUs for functional equivalence. Covers two layers:
  (1) C/C++ CUDA kernel porting via HIPIFY tools, build system adaptation, warp-to-wavefront
  conversion, header sanitization, library mapping, and inline PTX handling;
  (2) PyTorch-level porting via ROCm detection, torch.compile mode adaptation, Inductor
  configuration, CUDAGraph/HIP Graph capture, library replacement (flash-attn to aiter),
  attention routing, GEMM dispatching, and transformers monkey-patching.
  Use when: porting CUDA code to HIP/ROCm, making PyTorch models run on AMD GPUs, replacing
  NVIDIA-specific libraries with AMD equivalents, adapting torch.compile or CUDAGraph for ROCm,
  fixing ROCm-specific build or runtime failures, or creating AMD/NVIDIA dual-target codebases.
---

# AMD ROCm Porting

8-phase checklist for porting NVIDIA CUDA codebases to AMD ROCm GPUs. Targets **functional
equivalence** (correctness), not performance optimization. Covers both C/C++ kernel-level and
PyTorch-level porting.

## Phase 1: Environment & Detection

### ROCm detection (Python/PyTorch)

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
```

### ROCm version gating

```python
rocm_version = torch.version.hip or "0.0"
major = int(rocm_version.split(".")[0]) if rocm_version else 0
```

### Docker base images

```dockerfile
# PyTorch on ROCm (recommended starting point)
FROM rocm/pytorch:rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0

# Install HIPIFY tools (for C/C++ porting)
RUN apt-get update && apt-get install -y hipify-clang hipify-perl
```

### Key HIP environment variables

```bash
export HIP_LAUNCH_BLOCKING=0        # async kernel launch (set 1 to debug)
export AMD_LOG_LEVEL=0               # reduce HIP runtime noise
export HIP_CACHE_ENABLED=1           # cache compiled kernels
export HSA_ENABLE_SDMA=1             # async DMA copies
export GPU_MAX_HW_QUEUES=8           # hardware queue count
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # memory allocator
```

### GPU info check

```bash
rocm-smi                    # GPU status, temperature, utilization
rocminfo | grep "gfx"      # architecture string (e.g., gfx942, gfx950)
hipcc --version             # HIP compiler version
```

## Phase 2: Source File Translation (C/C++)

Use HIPIFY to translate CUDA source to HIP. See [references/hipify-and-source-translation.md](references/hipify-and-source-translation.md) for full workflow.

### Quick translation

```bash
# Perl-based (fast, regex, good for initial pass)
hipify-perl --inplace src/kernels/*.cu

# Clang-based (AST-precise, handles complex templates)
hipify-clang --inplace --cuda-path=/usr/local/cuda src/kernels/*.cu
```

### Header mapping (automated by HIPIFY)

| CUDA Header | HIP Header |
|---|---|
| `cuda_runtime.h` | `hip/hip_runtime.h` |
| `cuda_fp16.h` | `hip/hip_fp16.h` |
| `cublas_v2.h` | `hipblas/hipblas.h` |
| `cudnn.h` | `miopen/miopen.h` |
| `nccl.h` | `rccl/rccl.h` |

### What HIPIFY does NOT handle

- Inline PTX assembly (flag for manual rewrite or remove)
- CUTLASS templates (replace with Composable Kernel)
- Closed-source `.cubin` / `.fatbin` binaries
- `__managed__` memory semantics (requires HSA_XNACK)

## Phase 3: Architecture Adaptation

AMD GPUs use **64-wide wavefronts** (not 32-wide warps). This affects masks, shuffles, and
occupancy. See [references/hipify-and-source-translation.md](references/hipify-and-source-translation.md) for detailed patterns.

### Critical changes

| CUDA (warp=32) | HIP (wavefront=64) |
|---|---|
| `__ballot_sync(0xFFFFFFFF, pred)` | `__ballot(pred)` returns 64-bit |
| `__shfl_sync(mask, val, lane)` | `__shfl(val, lane)` with 64-lane width |
| `__activemask()` | Returns `uint64_t` on AMD |
| Shared mem tile: 32 elements | Prefer 64-element tiles |

### Warp mask width fix

```cpp
// CUDA: 32-bit mask
unsigned mask = __ballot_sync(0xFFFFFFFF, pred);
int count = __popc(mask);

// HIP: 64-bit mask
uint64_t mask = __ballot(pred);
int count = __popcll(mask);
```

### Atomic operations

`atomicAdd` for `float` and `double` works on AMD. For `half`, wrap with CAS:

```cpp
__device__ half atomicAdd_half(half* addr, half val) {
    unsigned int* addr_as_uint = (unsigned int*)((size_t)addr & ~1);
    unsigned int old = *addr_as_uint, assumed;
    do {
        assumed = old;
        // Update the correct half of the 32-bit word
        half sum = __hadd(*reinterpret_cast<half*>((char*)&assumed + ((size_t)addr & 1)), val);
        unsigned int new_val = assumed;
        *reinterpret_cast<half*>((char*)&new_val + ((size_t)addr & 1)) = sum;
        old = atomicCAS(addr_as_uint, assumed, new_val);
    } while (assumed != old);
    return *reinterpret_cast<half*>((char*)&old + ((size_t)addr & 1));
}
```

## Phase 4: Build System Adaptation

CMake and setup.py changes for HIP/ROCm. See [references/build-system-adaptation.md](references/build-system-adaptation.md) for full patterns.

### CMake: FindCUDA to FindHIP

```cmake
if(USE_ROCM)
  find_package(HIP REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(MIOpen REQUIRED)
  set(CMAKE_HIP_ARCHITECTURES "gfx942;gfx950")
  # Compile .cu files with hipcc
  set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
else()
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES "80;90")
endif()
```

### Python setup.py

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

ext = CUDAExtension(
    name="my_kernels",
    sources=["src/kernels.cu"],  # hipcc compiles .cu files on ROCm
    extra_compile_args={
        "nvcc": [] if not is_rocm else [],
        "cxx": ["-O3"],
    },
)
```

## Phase 5: Library Replacement

Replace NVIDIA libraries with AMD equivalents. See [references/library-and-model-adaptation.md](references/library-and-model-adaptation.md) for integration patterns and fallback strategies.

### Library mapping table

| NVIDIA | AMD | Notes |
|---|---|---|
| cuBLAS | hipBLAS / rocBLAS | Drop-in via HIPIFY |
| cuBLASLt | hipBLASLt | Tuned GEMM backends |
| cuDNN | MIOpen | API differs; convolution/RNN/pooling |
| NCCL | RCCL | Drop-in via HIPIFY |
| cuSPARSE | hipSPARSE / rocSPARSE | Drop-in via HIPIFY |
| cuFFT | hipFFT / rocFFT | Drop-in via HIPIFY |
| cuRAND | hipRAND / rocRAND | Drop-in via HIPIFY |
| Thrust | hipCUB / rocThrust | Mostly compatible |
| CUTLASS | Composable Kernel (CK) | Manual rewrite required |
| flash-attn | aiter | Different API; see reference |
| TensorRT | MIGraphX | Different API |

### Three-tier fallback pattern

Always provide fallback paths for AMD-specific libraries:

```python
# Tier 1: AMD-optimized (aiter)
AITER_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    pass

# Tier 2: PyTorch built-in (SDPA)
# Tier 3: Pure PyTorch eager (always works)
def attention_forward(q, k, v, **kwargs):
    if AITER_AVAILABLE:
        return aiter_attention(q, k, v, **kwargs)
    elif hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    else:
        return eager_attention(q, k, v, **kwargs)
```

### Graceful import with fallback stubs

```python
try:
    from amd_optimized_ops import fast_gemm, fast_attention
except ImportError:
    def fast_gemm(x, weight, bias=None):
        return F.linear(x, weight, bias)
    def fast_attention(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
```

## Phase 6: PyTorch / torch.compile Adaptation

ROCm's Inductor backend requires different settings than CUDA. See [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md) for complete configuration.

### CRITICAL: Compile mode

```python
is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
compile_mode = "default" if is_rocm else "reduce-overhead"
model = torch.compile(model, mode=compile_mode)
```

**Never use `mode="reduce-overhead"` on ROCm** -- it triggers broken Inductor CUDAGraph
integration causing extreme slowdowns (up to 65x).

### Essential Inductor settings for ROCm

```python
import torch._inductor.config as inductor_config

# CRITICAL: Disable Inductor's internal cudagraphs (broken on ROCm)
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Use ATen/rocBLAS for GEMM (faster than Triton GEMM on ROCm)
inductor_config.max_autotune_gemm_backends = "ATEN"
inductor_config.max_autotune = False

# CRITICAL: Disable memory_planning (triggers deep recursion on ROCm)
inductor_config.memory_planning = False

# Coordinate descent for Triton kernel tuning
inductor_config.coordinate_descent_tuning = True

# Fusion (reduce kernel launch overhead)
inductor_config.epilogue_fusion = True
inductor_config.aggressive_fusion = True
```

### Monkey-patch to prevent reduce-overhead

```python
_original_compile = torch.compile

def _safe_compile(model, **kwargs):
    if kwargs.get("mode") in (None, "reduce-overhead"):
        kwargs["mode"] = "default"
    return _original_compile(model, **kwargs)

torch.compile = _safe_compile
```

## Phase 7: CUDAGraph / HIP Graph Adaptation

HIP Graph capture requires patching Dynamo's RNG handling. See [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md) for full details.

### Dynamo RNG patch (required before any graph capture)

ROCm forbids `torch.cuda.get_rng_state()` during stream capture. Patch Dynamo:

```python
import functools
import torch._dynamo.convert_frame

_orig_preserve = torch._dynamo.convert_frame.preserve_global_state

def _preserve_skip_rng(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rng_state = None
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            rng_state = torch.cuda.get_rng_state()
        try:
            return fn(*args, **kwargs)
        finally:
            if rng_state is not None:
                torch.cuda.set_rng_state(rng_state)
    return wrapper

torch._dynamo.convert_frame.preserve_global_state = _preserve_skip_rng
```

### Manual CUDAGraph capture pattern

```python
# 1. Apply Dynamo patch first
# 2. Warm up
with torch.no_grad():
    for _ in range(5):
        _ = model(inputs)
torch.cuda.current_stream().synchronize()

# 3. Capture
pool = torch.cuda.graphs.graph_pool_handle()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    static_output = model(inputs)

# 4. Replay
graph.replay()
result = static_output  # updated in-place
```

### Rules during graph capture

- **No `.item()` calls** -- device-to-host sync is forbidden during capture
- **No shape-dependent branching** -- all tensor shapes must be static
- **Stream sync only** -- use `torch.cuda.current_stream().synchronize()`, not `torch.cuda.synchronize()`
- **Pre-cache Python paths** -- warm up to resolve all conditional branches before capture

## Phase 8: Verification & Testing

4-level verification pyramid. See [references/verification-methodology.md](references/verification-methodology.md) for full methodology.

### Level 1: Static analysis

```bash
# Check for remaining CUDA references after HIPIFY
grep -rn "cuda_runtime\.h\|#include.*cuda" src/
# Check for inline PTX
grep -rn "asm\s*(" src/ --include="*.cu" --include="*.hip"
# Check for NVIDIA-specific types
grep -rn "cudaStream_t\|cudaError_t" src/
```

### Level 2: Build test

```bash
hipcc -c src/kernels.hip -o kernels.o --offload-arch=gfx942
# Or via CMake
cmake -DUSE_ROCM=ON -DCMAKE_HIP_ARCHITECTURES="gfx942" ..
make -j$(nproc)
```

### Level 3: Load test

```python
import torch
lib = torch.ops.load_library("my_kernels.so")
# Verify kernel is callable
x = torch.randn(32, 32, device="cuda")
y = torch.ops.my_kernels.forward(x)
assert y.shape == x.shape
```

### Level 4: Numerical correctness

```python
# Compare CUDA reference vs HIP output
ref = torch.load("golden_vectors.pt")  # generated on NVIDIA
got = model(test_input)

diff = (ref.float() - got.float()).abs()
print(f"Max abs diff: {diff.max().item():.6f}")
print(f"Mean abs diff: {diff.mean().item():.6f}")

torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=5e-2)
```

### Sync helper

```python
def _sync():
    """Stream-level sync (preferred on ROCm over device-wide sync)."""
    try:
        torch.cuda.current_stream().synchronize()
    except Exception:
        torch.cuda.synchronize()
```

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| `reduce-overhead` compile mode | 65x slowdown, hangs | Use `mode="default"` on ROCm |
| Inductor cudagraphs enabled | Slowdown, graph capture errors | Set `inductor_config.triton.cudagraphs = False` |
| Inductor memory_planning | Deep recursion crash | Set `inductor_config.memory_planning = False` |
| `torch.cuda.get_rng_state()` during capture | RuntimeError | Apply Dynamo RNG patch |
| `.item()` during graph capture | RuntimeError: sync during capture | Cache values before capture |
| 32-bit warp masks on AMD | Silent wrong results | Use `uint64_t` for ballot/active masks |
| Missing `__ballot` → `__ballot_sync` rename | Compile error | HIPIFY handles this; verify manually |
| `__managed__` memory | Runtime crash | Requires `HSA_XNACK=1` or rewrite |
| Inline PTX | Compile error | Manual rewrite or remove |
| Triton tanh overflow | NaN in activations | Clamp input to `[-10, 10]` before exp |

## Edge Cases & Graceful Failure

- **Inline PTX assembly**: Cannot be auto-ported. Flag with `grep -rn "asm\s*(" --include="*.cu"` and mark for manual rewrite using HIP intrinsics or GCN inline assembly.
- **CUTLASS templates**: No HIPIFY path. Replace with AMD Composable Kernel (CK) or use rocBLAS fallback. Mark for human review.
- **Closed-source CUDA binaries**: Cannot be ported. Identify and replace with open-source or AMD equivalents.
- **`__managed__` memory**: Requires `HSA_XNACK=1` on supported GPUs or refactor to explicit `hipMemcpy`.
- **Mixed CUDA/HIP builds**: Not supported. Port entire translation unit to HIP or isolate behind a clean interface.

## References

Detailed patterns and code for each phase:

- **[HIPIFY & Source Translation](references/hipify-and-source-translation.md)**: HIPIFY workflow, header mapping, warp/wavefront fixes, inline PTX handling, atomic shims
- **[Build System Adaptation](references/build-system-adaptation.md)**: CMake rewriting, setup.py/CUDAExtension, hipify_torch, compiler flags
- **[Library & Model Adaptation](references/library-and-model-adaptation.md)**: Library mapping, aiter integration, attention routing, GEMM dispatching, fallback patterns, transformers patching
- **[torch.compile & CUDAGraph](references/torch-compile-and-cudagraph.md)**: Inductor config, compile mode, Dynamo RNG patch, manual graph capture, stream capture rules
- **[Verification Methodology](references/verification-methodology.md)**: 4-level pyramid, golden vectors, tolerance management, profiling, stability testing
