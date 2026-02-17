# HIPIFY & Source Translation

Detailed patterns for translating CUDA C/C++ source to HIP for AMD ROCm GPUs.

## Table of Contents

- [HIPIFY Tool Chain](#hipify-tool-chain)
- [Header Mapping](#header-mapping)
- [API Name Translation](#api-name-translation)
- [Warp to Wavefront Adaptation](#warp-to-wavefront-adaptation)
- [Shuffle Operations](#shuffle-operations)
- [Atomic Operations](#atomic-operations)
- [Inline PTX Handling](#inline-ptx-handling)
- [NVTX to roctx](#nvtx-to-roctx)
- [Memory Management Differences](#memory-management-differences)
- [Post-HIPIFY Verification Checklist](#post-hipify-verification-checklist)

## HIPIFY Tool Chain

### hipify-perl (regex-based)

Fast, handles most API translations. Good for initial pass.

```bash
# Single file
hipify-perl --inplace kernel.cu

# Directory (preserving structure)
hipify-perl --inplace --print-stats src/kernels/*.cu

# Dry run (show what would change)
hipify-perl --no-output --print-stats kernel.cu
```

**Strengths**: Fast, no LLVM dependency, handles 95%+ of cases.
**Weaknesses**: Regex-based; can mangle string literals, comments, or complex macros.

### hipify-clang (AST-based)

Uses Clang AST for precise translation. Required for complex templates and overloaded operators.

```bash
# Requires CUDA toolkit path
hipify-clang --inplace --cuda-path=/usr/local/cuda kernel.cu

# With additional include paths
hipify-clang --inplace \
    --cuda-path=/usr/local/cuda \
    -I/path/to/includes \
    --extra-arg="-std=c++17" \
    kernel.cu
```

**Strengths**: AST-precise, handles templates, overloaded operators, complex macros.
**Weaknesses**: Slower, requires LLVM/Clang and CUDA toolkit installed.

### Choosing between them

| Scenario | Tool |
|---|---|
| Initial bulk translation | hipify-perl |
| Complex template libraries | hipify-clang |
| CI/CD automated porting | hipify-perl (faster) |
| CUTLASS-derived code | Neither (manual rewrite to CK) |

## Header Mapping

HIPIFY performs these header substitutions automatically. Verify post-translation.

```cpp
// CUDA                          // HIP
#include <cuda_runtime.h>     → #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h> → #include <hip/hip_runtime_api.h>
#include <cuda_fp16.h>        → #include <hip/hip_fp16.h>
#include <cuda_bf16.h>        → #include <hip/hip_bf16.h>
#include <cuda.h>             → #include <hip/hip_runtime.h>
#include <cublas_v2.h>        → #include <hipblas/hipblas.h>
#include <cudnn.h>            → <miopen/miopen.h>  // API differs
#include <nccl.h>             → #include <rccl/rccl.h>
#include <cusparse.h>         → #include <hipsparse/hipsparse.h>
#include <cufft.h>            → #include <hipfft/hipfft.h>
#include <curand.h>           → #include <hiprand/hiprand.h>
#include <curand_kernel.h>    → #include <hiprand/hiprand_kernel.h>
#include <cooperative_groups.h> → #include <hip/hip_cooperative_groups.h>
```

### Headers that need removal or manual replacement

```cpp
// NVTX: replace with roctx or remove
#include <nvToolsExt.h>       → #include <roctracer/roctx.h>  // or remove

// Thrust: use hipCUB or rocThrust
#include <thrust/sort.h>      → #include <thrust/sort.h>  // rocThrust is compatible

// CUTLASS: no HIP equivalent, must rewrite with CK
#include <cutlass/gemm/...>   → // MANUAL: replace with composable_kernel
```

## API Name Translation

HIPIFY handles these automatically. Reference for manual verification.

### Runtime API

```cpp
// Memory
cudaMalloc        → hipMalloc
cudaFree          → hipFree
cudaMemcpy        → hipMemcpy
cudaMemcpyAsync   → hipMemcpyAsync
cudaMemset        → hipMemset
cudaMallocManaged → hipMallocManaged  // requires HSA_XNACK=1

// Streams
cudaStreamCreate        → hipStreamCreate
cudaStreamSynchronize   → hipStreamSynchronize
cudaStreamDestroy       → hipStreamDestroy

// Events
cudaEventCreate       → hipEventCreate
cudaEventRecord       → hipEventRecord
cudaEventSynchronize  → hipEventSynchronize
cudaEventElapsedTime  → hipEventElapsedTime

// Device
cudaSetDevice       → hipSetDevice
cudaGetDeviceCount  → hipGetDeviceCount
cudaDeviceSynchronize → hipDeviceSynchronize

// Error handling
cudaGetLastError    → hipGetLastError
cudaGetErrorString  → hipGetErrorString
cudaError_t         → hipError_t
cudaSuccess         → hipSuccess
```

### Kernel launch syntax

```cpp
// CUDA
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);

// HIP (identical syntax, handled by hipcc)
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

### Type mappings

```cpp
cudaStream_t       → hipStream_t
cudaEvent_t        → hipEvent_t
cudaError_t        → hipError_t
cudaDeviceProp     → hipDeviceProp_t
cudaMemcpyKind     → hipMemcpyKind
cudaMemcpyHostToDevice   → hipMemcpyHostToDevice
cudaMemcpyDeviceToHost   → hipMemcpyDeviceToHost
cudaMemcpyDeviceToDevice → hipMemcpyDeviceToDevice
```

## Warp to Wavefront Adaptation

AMD GCN/CDNA architecture uses **64-wide wavefronts** (NVIDIA uses 32-wide warps). This is the
most common source of silent correctness bugs after HIPIFY.

### Ballot operations

```cpp
// CUDA: 32-bit result
unsigned int mask = __ballot_sync(0xFFFFFFFF, predicate);
int count = __popc(mask);

// HIP: 64-bit result (CRITICAL difference)
uint64_t mask = __ballot(predicate);
int count = __popcll(mask);
```

### Active mask

```cpp
// CUDA: 32-bit
unsigned int active = __activemask();

// HIP: 64-bit
uint64_t active = __activemask();
```

### Warp-level reductions

```cpp
// CUDA warp reduce (32 threads)
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

// HIP wavefront reduce (64 threads)
for (int offset = 32; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
```

### Shared memory tile sizing

```cpp
// CUDA: often use WARP_SIZE (32)
__shared__ float smem[WARP_SIZE];      // 32 elements

// HIP: use wavefront size (64)
#define WAVEFRONT_SIZE 64
__shared__ float smem[WAVEFRONT_SIZE]; // 64 elements
```

### Portable warp/wavefront size

```cpp
#ifdef __HIP_PLATFORM_AMD__
    #define WARP_SIZE 64
#else
    #define WARP_SIZE 32
#endif
```

### Workgroup size recommendations

Optimal workgroup (threadblock) sizes for 64-wide wavefronts:
- 64, 128, 256, 512 (multiples of 64)
- Avoid sizes that aren't multiples of 64 (wastes SIMD lanes)

## Shuffle Operations

### Basic shuffles

```cpp
// CUDA (sync variants)
__shfl_sync(mask, val, srcLane)        → __shfl(val, srcLane)
__shfl_up_sync(mask, val, delta)       → __shfl_up(val, delta)
__shfl_down_sync(mask, val, delta)     → __shfl_down(val, delta)
__shfl_xor_sync(mask, val, laneMask)   → __shfl_xor(val, laneMask)
```

### Lane indexing difference

```cpp
// CUDA: lanes 0-31 within a warp
int lane = threadIdx.x % 32;

// HIP: lanes 0-63 within a wavefront
int lane = threadIdx.x % 64;
```

## Atomic Operations

### Supported atomics (work identically)

```cpp
atomicAdd(addr, val)    // float, double, int, unsigned
atomicSub(addr, val)    // int, unsigned
atomicMin(addr, val)    // int, unsigned
atomicMax(addr, val)    // int, unsigned
atomicAnd(addr, val)    // int, unsigned
atomicOr(addr, val)     // int, unsigned
atomicXor(addr, val)    // int, unsigned
atomicCAS(addr, compare, val)  // int, unsigned, unsigned long long
atomicExch(addr, val)   // int, unsigned, float
```

### Half-precision atomicAdd (CAS shim)

AMD does not natively support `atomicAdd` for `half`. Implement via CAS loop:

```cpp
__device__ __forceinline__
half atomicAdd_half(half* address, half val) {
    unsigned int* base = (unsigned int*)((size_t)address & ~1);
    unsigned int old_val = *base, assumed;
    unsigned int byte_offset = ((size_t)address & 1);
    do {
        assumed = old_val;
        half stored = *reinterpret_cast<half*>((char*)&assumed + byte_offset);
        half updated = __hadd(stored, val);
        unsigned int new_val = assumed;
        *reinterpret_cast<half*>((char*)&new_val + byte_offset) = updated;
        old_val = atomicCAS(base, assumed, new_val);
    } while (assumed != old_val);
    return *reinterpret_cast<half*>((char*)&old_val + byte_offset);
}
```

## Inline PTX Handling

Inline PTX assembly **cannot** be automatically translated. It must be manually rewritten or removed.

### Detection

```bash
grep -rn "asm\s*(" --include="*.cu" --include="*.cuh" --include="*.h" src/
grep -rn "__asm" --include="*.cu" --include="*.cuh" --include="*.h" src/
```

### Common PTX patterns and HIP equivalents

```cpp
// CUDA PTX: warp vote
asm("vote.any.pred %0, %1;" : "=r"(result) : "r"(predicate));
// HIP: use __any() intrinsic instead
result = __any(predicate);

// CUDA PTX: memory fence
asm("membar.gl;");
// HIP: use __threadfence()
__threadfence();

// CUDA PTX: clock counter
asm("mov.u64 %0, %%clock64;" : "=l"(clock));
// HIP: use clock64() or wall_clock64()
clock = clock64();
```

### When PTX has no HIP equivalent

Mark for human review with a clear comment:

```cpp
#ifdef __HIP_PLATFORM_AMD__
    #error "MANUAL PORT REQUIRED: inline PTX at this location needs GCN assembly or C++ rewrite"
    // Original PTX: asm("...");
    // Options: (1) rewrite in GCN asm, (2) use HIP intrinsic, (3) C++ fallback
#else
    asm("original ptx here");
#endif
```

## NVTX to roctx

```cpp
// CUDA: NVTX markers
#include <nvToolsExt.h>
nvtxRangePushA("kernel_name");
nvtxRangePop();

// HIP: roctx markers
#include <roctracer/roctx.h>
roctxRangePushA("kernel_name");
roctxRangePop();

// Portable macro
#ifdef __HIP_PLATFORM_AMD__
    #include <roctracer/roctx.h>
    #define RANGE_PUSH(name) roctxRangePushA(name)
    #define RANGE_POP() roctxRangePop()
#else
    #include <nvToolsExt.h>
    #define RANGE_PUSH(name) nvtxRangePushA(name)
    #define RANGE_POP() nvtxRangePop()
#endif
```

## Memory Management Differences

### Managed memory

CUDA unified/managed memory requires `HSA_XNACK=1` on AMD:

```bash
# Required for hipMallocManaged to work
export HSA_XNACK=1
```

Not all AMD GPUs support XNACK. If unsupported, refactor to explicit copies:

```cpp
// Managed (may not work on all AMD GPUs)
hipMallocManaged(&ptr, size);

// Explicit (always works)
hipMalloc(&d_ptr, size);
hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
```

### Pinned memory

```cpp
// Works identically on CUDA and HIP
hipHostMalloc(&h_ptr, size, hipHostMallocDefault);
hipHostFree(h_ptr);
```

## Post-HIPIFY Verification Checklist

After running HIPIFY, verify these items manually:

1. **`grep -rn "cuda" src/`** -- no remaining CUDA references in non-comment code
2. **`grep -rn "0xFFFFFFFF" src/`** -- warp masks must be 64-bit on AMD
3. **`grep -rn "asm\s*(" src/`** -- all PTX flagged or rewritten
4. **`grep -rn "__managed__" src/`** -- managed memory flagged
5. **`grep -rn "warpSize\|WARP_SIZE" src/`** -- all warp size references updated
6. **`grep -rn "__ballot_sync\|__shfl_sync" src/`** -- sync variants removed (HIP uses unsync)
7. **`grep -rn "cutlass\|CUTLASS" src/`** -- CUTLASS references flagged
8. **Compile test**: `hipcc -c *.hip --offload-arch=gfx942`
9. **Link test**: verify all symbols resolve
10. **Runtime test**: run with `HIP_LAUNCH_BLOCKING=1` for synchronous error reporting
