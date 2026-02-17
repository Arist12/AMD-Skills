# Build System Adaptation

Patterns for adapting CMake, setup.py, and other build systems to support HIP/ROCm alongside or
instead of CUDA.

## Table of Contents

- [CMake Adaptation](#cmake-adaptation)
- [Python setup.py / CUDAExtension](#python-setuppy--cudaextension)
- [hipify_torch Integration](#hipify_torch-integration)
- [Compiler Flags](#compiler-flags)
- [Dual-Target Builds](#dual-target-builds)
- [Common Build Errors](#common-build-errors)

## CMake Adaptation

### Replacing FindCUDA with FindHIP

```cmake
option(USE_ROCM "Build for AMD ROCm" OFF)

if(USE_ROCM)
    # HIP is required
    find_package(HIP REQUIRED)

    # Set target architecture
    # gfx942 = MI300X, gfx950 = MI350X
    set(CMAKE_HIP_ARCHITECTURES "gfx942;gfx950" CACHE STRING "HIP architectures")

    # Find AMD libraries
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(MIOpen QUIET)  # optional, not all projects need it
    find_package(rccl QUIET)    # optional, for multi-GPU

    # HIP compiler
    set(CMAKE_HIP_COMPILER ${HIP_HIPCC_EXECUTABLE})

    # Mark .cu files as HIP language (hipcc compiles them)
    file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")
    set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)

    # Definitions
    add_definitions(-D__HIP_PLATFORM_AMD__ -DUSE_ROCM)

else()
    # Standard CUDA build
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;90" CACHE STRING "CUDA architectures")

    file(GLOB_RECURSE KERNEL_SOURCES "src/kernels/*.cu")
endif()

# Common library target
add_library(my_kernels SHARED ${KERNEL_SOURCES})

if(USE_ROCM)
    target_link_libraries(my_kernels
        hip::device
        roc::hipblas
        roc::rocblas
    )
    if(MIOpen_FOUND)
        target_link_libraries(my_kernels MIOpen)
    endif()
else()
    target_link_libraries(my_kernels
        ${CUDA_LIBRARIES}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_cudnn_LIBRARY}
    )
endif()
```

### CMake minimum version and HIP language support

```cmake
# CMake 3.21+ has native HIP language support
cmake_minimum_required(VERSION 3.21)
project(my_project LANGUAGES CXX HIP)

# For older CMake, use the HIP CMake module
# list(APPEND CMAKE_PREFIX_PATH "/opt/rocm/lib/cmake")
```

### Architecture strings reference

| GPU | Architecture | CMake value |
|---|---|---|
| MI250X | CDNA2 | `gfx90a` |
| MI300X | CDNA3 | `gfx942` |
| MI350X | CDNA4 | `gfx950` |
| RX 7900 XTX | RDNA3 | `gfx1100` |
| RX 9070 XT | RDNA4 | `gfx1201` |

## Python setup.py / CUDAExtension

PyTorch's `CUDAExtension` works on ROCm -- `hipcc` transparently compiles `.cu` files.

### Basic pattern

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

extra_cuda_args = []
extra_cxx_args = ["-O3", "-std=c++17"]

if is_rocm:
    # ROCm-specific flags
    extra_cuda_args = [
        "--offload-arch=gfx942",
        "-O3",
    ]
else:
    # NVIDIA-specific flags
    extra_cuda_args = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
    ]

setup(
    name="my_kernels",
    ext_modules=[
        CUDAExtension(
            name="my_kernels",
            sources=[
                "src/binding.cpp",
                "src/kernels.cu",
            ],
            extra_compile_args={
                "nvcc": extra_cuda_args,
                "cxx": extra_cxx_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### Conditional source selection

```python
sources = ["src/binding.cpp"]

if is_rocm:
    sources.append("src/kernels_hip.cu")  # HIP-specific implementation
else:
    sources.append("src/kernels_cuda.cu")  # CUDA-specific implementation

# Or use same files if already HIPIFYed
sources.append("src/kernels.cu")  # hipcc handles .cu on ROCm
```

### Include paths for ROCm

```python
include_dirs = ["src/include"]

if is_rocm:
    import os
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    include_dirs.extend([
        os.path.join(rocm_home, "include"),
        os.path.join(rocm_home, "include", "hip"),
        os.path.join(rocm_home, "include", "hipblas"),
    ])
```

## hipify_torch Integration

PyTorch provides `hipify_torch` for on-the-fly HIPIFY during build:

```python
from torch.utils.hipify import hipify_python

# HIPIFY all .cu files in a directory before building
hipify_python.hipify(
    project_directory="src/",
    output_directory="src_hip/",
    includes=["*.cu", "*.cuh"],
    is_pytorch_extension=True,
)
```

### Using hipify_torch in setup.py

```python
import torch

is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

if is_rocm:
    from torch.utils.hipify import hipify_python
    hipify_python.hipify(
        project_directory="src/",
        output_directory="src_hip/",
        includes=["*.cu", "*.cuh"],
        is_pytorch_extension=True,
    )
    sources_dir = "src_hip/"
else:
    sources_dir = "src/"
```

## Compiler Flags

### hipcc flags reference

```bash
# Target architecture (required)
--offload-arch=gfx942       # MI300X
--offload-arch=gfx950       # MI350X

# Optimization
-O3                          # Max optimization
-ffast-math                  # Fast math (less precise)
-munsafe-fp-atomics          # Faster atomics (may lose precision)

# Standards
-std=c++17                   # C++ standard
-fgpu-rdc                    # Relocatable device code (for separate compilation)

# Debug
-g                           # Debug symbols
-G                           # Device debug symbols
-DHIP_ENABLE_PRINTF          # Enable printf in device code

# Warnings
-Wall -Wextra                # Standard warnings
-Wno-unused-parameter        # Suppress unused param warnings (common after HIPIFY)
```

### Flags that differ from nvcc

| nvcc | hipcc | Notes |
|---|---|---|
| `-gencode arch=compute_80,code=sm_80` | `--offload-arch=gfx942` | Architecture specification |
| `--use_fast_math` | `-ffast-math` | Fast math mode |
| `-lineinfo` | `-gline-tables-only` | Line info for profiling |
| `--expt-relaxed-constexpr` | (default behavior) | Not needed on hipcc |
| `--expt-extended-lambda` | (default behavior) | Not needed on hipcc |
| `-rdc=true` | `-fgpu-rdc` | Relocatable device code |

## Dual-Target Builds

### Preprocessor-based code selection

```cpp
#ifdef __HIP_PLATFORM_AMD__
    // AMD-specific code path
    #include <hip/hip_runtime.h>
    #define WARP_SIZE 64
#else
    // NVIDIA code path
    #include <cuda_runtime.h>
    #define WARP_SIZE 32
#endif
```

### CMake dual-target

```cmake
# Single CMakeLists.txt supporting both
option(USE_ROCM "Build for AMD ROCm" OFF)

if(USE_ROCM)
    project(my_lib LANGUAGES CXX HIP)
    add_compile_definitions(__HIP_PLATFORM_AMD__ USE_ROCM)
else()
    project(my_lib LANGUAGES CXX CUDA)
    add_compile_definitions(USE_CUDA)
endif()
```

## Common Build Errors

### "error: unknown type name 'cudaStream_t'"

HIPIFY missed a translation. Fix:
```cpp
// Replace manually
cudaStream_t → hipStream_t
```

### "cannot find -lcuda"

ROCm doesn't link against libcuda. Fix:
```cmake
if(NOT USE_ROCM)
    target_link_libraries(my_lib ${CUDA_LIBRARIES})
endif()
```

### "error: unknown argument '--use_fast_math'"

hipcc uses different flag syntax. Fix:
```python
if is_rocm:
    extra_cuda_args = ["-ffast-math"]  # not --use_fast_math
```

### "fatal error: cuda_runtime.h: No such file or directory"

HIPIFY didn't translate a header. Fix:
```cpp
#include <hip/hip_runtime.h>  // not cuda_runtime.h
```

### "error: no matching function for call to 'atomicAdd' with half"

AMD lacks native half atomicAdd. Implement the CAS shim from
[hipify-and-source-translation.md](hipify-and-source-translation.md#half-precision-atomicadd-cas-shim).

### Linker errors for cuBLAS/cuDNN symbols

```cmake
if(USE_ROCM)
    target_link_libraries(my_lib roc::hipblas roc::rocblas)
    # MIOpen replaces cuDNN but has different API
    if(MIOpen_FOUND)
        target_link_libraries(my_lib MIOpen)
    endif()
else()
    target_link_libraries(my_lib ${CUDA_cublas_LIBRARY} ${CUDA_cudnn_LIBRARY})
endif()
```
