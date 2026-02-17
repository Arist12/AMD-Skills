# Verification Methodology

4-level verification pyramid for validating CUDA-to-ROCm ports, plus profiling and stability
testing patterns.

## Table of Contents

- [Verification Pyramid Overview](#verification-pyramid-overview)
- [Level 1: Static Analysis](#level-1-static-analysis)
- [Level 2: Build Test](#level-2-build-test)
- [Level 3: Load Test](#level-3-load-test)
- [Level 4: Numerical Correctness](#level-4-numerical-correctness)
- [Golden Vector Generation](#golden-vector-generation)
- [Tolerance Management](#tolerance-management)
- [CUDAGraph Correctness Testing](#cudagraph-correctness-testing)
- [Profiling](#profiling)
- [Stability Testing](#stability-testing)
- [Sync Helpers](#sync-helpers)

## Verification Pyramid Overview

```
           ┌─────────────────────┐
    L4     │ Numerical Correctness│  Most comprehensive
           ├─────────────────────┤
    L3     │    Load Test         │  Runtime validation
           ├─────────────────────┤
    L2     │    Build Test        │  Compilation check
           ├─────────────────────┤
    L1     │  Static Analysis     │  Fastest, catch obvious issues
           └─────────────────────┘
```

Each level catches different classes of bugs. Run all four levels in order.

## Level 1: Static Analysis

Catch untranslated CUDA references, wrong mask widths, and non-portable patterns.

### Remaining CUDA references

```bash
# Headers that should have been translated
grep -rn "cuda_runtime\.h\|cuda_fp16\.h\|cublas_v2\.h\|cudnn\.h" \
    --include="*.cu" --include="*.hip" --include="*.cpp" --include="*.h" src/

# CUDA types that should be HIP types
grep -rn "cudaStream_t\|cudaEvent_t\|cudaError_t\|cudaDeviceProp" \
    --include="*.cu" --include="*.hip" --include="*.cpp" src/

# CUDA API calls not translated
grep -rn "cudaMalloc\|cudaFree\|cudaMemcpy\|cudaSetDevice" \
    --include="*.cu" --include="*.hip" --include="*.cpp" src/
```

### Warp size issues

```bash
# 32-bit ballot masks (should be 64-bit on AMD)
grep -rn "0xFFFFFFFF" --include="*.cu" --include="*.hip" src/

# Hardcoded warp size
grep -rn "warpSize\b\|WARP_SIZE\b" --include="*.cu" --include="*.hip" src/

# 32-lane shuffle operations
grep -rn "__shfl_sync\|__ballot_sync\|__activemask" \
    --include="*.cu" --include="*.hip" src/
```

### Non-portable constructs

```bash
# Inline PTX (cannot be auto-ported)
grep -rn "asm\s*(" --include="*.cu" --include="*.hip" --include="*.cuh" src/

# Managed memory (requires HSA_XNACK)
grep -rn "__managed__\|cudaMallocManaged\|hipMallocManaged" src/

# CUTLASS (requires manual CK rewrite)
grep -rn "cutlass\|CUTLASS" --include="*.cu" --include="*.hip" --include="*.h" src/

# NVTX (replace with roctx or remove)
grep -rn "nvtx\|NVTX\|nvToolsExt" src/
```

### Python-level checks

```bash
# torch.compile with reduce-overhead
grep -rn "reduce-overhead\|reduce_overhead" --include="*.py" src/

# Direct CUDA API usage
grep -rn "torch\.cuda\.get_rng_state\|torch\.cuda\.synchronize" --include="*.py" src/

# flash_attn imports (should be aiter on AMD)
grep -rn "from flash_attn\|import flash_attn" --include="*.py" src/
```

## Level 2: Build Test

### C/C++ compilation

```bash
# Single file
hipcc -c src/kernel.hip -o kernel.o --offload-arch=gfx942

# With debug info for better error messages
hipcc -c -g src/kernel.hip -o kernel.o --offload-arch=gfx942

# Multi-architecture
hipcc -c src/kernel.hip -o kernel.o \
    --offload-arch=gfx942 --offload-arch=gfx950
```

### CMake build

```bash
mkdir build && cd build
cmake -DUSE_ROCM=ON \
    -DCMAKE_HIP_ARCHITECTURES="gfx942" \
    -DCMAKE_BUILD_TYPE=Release \
    ..
make -j$(nproc) 2>&1 | tee build.log

# Check for warnings (especially implicit type conversions)
grep -i "warning" build.log
```

### Python extension build

```bash
# Clean build
pip install -e . --no-build-isolation 2>&1 | tee build.log
# Or
python setup.py build_ext --inplace 2>&1 | tee build.log
```

### Common build failures and fixes

| Error message | Likely cause | Fix |
|---|---|---|
| `unknown type 'cudaStream_t'` | HIPIFY missed translation | Manual: `cudaStream_t` → `hipStream_t` |
| `cannot find -lcuda` | Linking against CUDA lib | Use `#if USE_ROCM` to skip `-lcuda` |
| `unknown argument '--use_fast_math'` | nvcc flag passed to hipcc | Use `-ffast-math` instead |
| `undefined reference to 'cublasCreate'` | cuBLAS not translated | Link `-lhipblas -lrocblas` |
| `no member named 'atomicAdd' for half` | Half atomicAdd unsupported | Implement CAS shim |

## Level 3: Load Test

Verify the compiled library loads and basic operations work.

### Shared library loading

```python
import torch

# Load the compiled extension
try:
    lib = torch.ops.load_library("my_kernels.so")
    print("Library loaded successfully")
except Exception as e:
    print(f"Load failed: {e}")
    # Check: ldd my_kernels.so (missing .so dependencies?)
```

### Basic operation test

```python
# Verify on correct device
assert torch.cuda.is_available(), "No GPU available"
device = torch.device("cuda")

# Simple forward pass
x = torch.randn(32, 64, device=device, dtype=torch.float32)
try:
    y = torch.ops.my_kernels.forward(x)
    print(f"Output shape: {y.shape}, dtype: {y.dtype}, device: {y.device}")
    assert y.device == x.device, "Output on wrong device"
    assert not torch.isnan(y).any(), "NaN in output"
    assert not torch.isinf(y).any(), "Inf in output"
    print("Load test passed")
except Exception as e:
    print(f"Runtime error: {e}")
```

### PyTorch model load test

```python
import torch

model = YourModel()
model = model.to("cuda").to(torch.bfloat16)
model.eval()

# Test with representative input
x = torch.randn(1, 128, 2048, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    try:
        y = model(x)
        print(f"Output: shape={y.shape}, has_nan={torch.isnan(y).any()}")
        print("Model load test passed")
    except Exception as e:
        print(f"Model forward failed: {e}")
```

## Level 4: Numerical Correctness

Compare ported output against NVIDIA reference (golden vectors).

### Basic comparison

```python
import torch

# Load reference (generated on NVIDIA GPU)
ref = torch.load("golden_vectors.pt", map_location="cpu")

# Run on AMD
model = load_model().to("cuda").eval()
with torch.no_grad():
    got = model(test_input.to("cuda")).cpu()

# Compare
diff = (ref.float() - got.float()).abs()
max_abs = float(diff.max().item())
mean_abs = float(diff.mean().item())
max_rel = float((diff / ref.float().abs().clamp_min(1e-6)).max().item())

print(f"Max absolute diff: {max_abs:.6f}")
print(f"Mean absolute diff: {mean_abs:.6f}")
print(f"Max relative diff: {max_rel:.6f}")

# Assert within tolerance
torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=5e-2)
print("Numerical correctness PASSED")
```

### Per-layer comparison

For debugging which layer diverges:

```python
import torch

hooks = {}
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach().cpu().float()
    return hook

# Register hooks on all layers
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # leaf modules only
        hooks[name] = module.register_forward_hook(make_hook(name))

# Run forward pass
with torch.no_grad():
    _ = model(test_input)

# Remove hooks
for h in hooks.values():
    h.remove()

# Compare each layer
ref_activations = torch.load("ref_activations.pt")
for name in sorted(activations.keys()):
    if name in ref_activations:
        diff = (activations[name] - ref_activations[name]).abs()
        print(f"{name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
```

## Golden Vector Generation

Generate reference outputs on NVIDIA GPU for cross-platform comparison.

### Generation script (run on NVIDIA)

```python
import torch

def generate_golden_vectors(model, test_inputs, output_path):
    """Generate reference outputs on NVIDIA GPU."""
    model.eval()
    golden = {}

    with torch.no_grad():
        for name, inp in test_inputs.items():
            out = model(inp.to("cuda"))
            golden[name] = {
                "input": inp.cpu(),
                "output": out.cpu().float(),
                "metadata": {
                    "dtype": str(inp.dtype),
                    "device": "nvidia",
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                },
            }

    torch.save(golden, output_path)
    print(f"Saved golden vectors to {output_path}")
```

### Test input recommendations

```python
test_inputs = {
    # Standard shapes
    "small": torch.randn(1, 32, 512, dtype=torch.bfloat16),
    "medium": torch.randn(4, 128, 2048, dtype=torch.bfloat16),
    "large": torch.randn(16, 512, 2048, dtype=torch.bfloat16),

    # Edge cases
    "single_token": torch.randn(1, 1, 2048, dtype=torch.bfloat16),
    "max_seq": torch.randn(1, 2048, 2048, dtype=torch.bfloat16),

    # Deterministic (fixed seed)
    "deterministic": _make_deterministic_input(seed=42),
}
```

### Deterministic input generation

```python
def _make_deterministic_input(seed, shape=(1, 128, 2048), dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=g)
```

## Tolerance Management

### Recommended tolerances by dtype

| Dtype | rtol | atol | Notes |
|---|---|---|---|
| float32 | 1e-4 | 1e-4 | Tight; expect near-exact match |
| float16 | 1e-2 | 1e-2 | Normal FP16 variance |
| bfloat16 | 5e-2 | 5e-2 | BF16 has less mantissa precision |
| bfloat16 (multi-layer) | 1e-1 | 1e-1 | Errors accumulate across layers |

### Adaptive tolerance

```python
def check_numerical_correctness(got, ref, dtype=torch.bfloat16):
    """Check correctness with dtype-appropriate tolerances."""
    tolerances = {
        torch.float32: (1e-4, 1e-4),
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (5e-2, 5e-2),
    }
    rtol, atol = tolerances.get(dtype, (1e-1, 1e-1))

    try:
        torch.testing.assert_close(
            got.float(), ref.float(), rtol=rtol, atol=atol
        )
        return True, "PASSED"
    except AssertionError as e:
        return False, str(e)
```

## CUDAGraph Correctness Testing

Verify that graph replay produces identical results to eager execution.

```python
def verify_cudagraph_correctness(model, inputs, graph, static_output, num_checks=5):
    """Compare eager vs graph-replayed output."""
    for i in range(num_checks):
        # Eager reference
        with torch.no_grad():
            ref = model(inputs).clone()

        # Graph replay
        graph.replay()
        got = static_output.clone()

        diff = (ref.float() - got.float()).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

        print(f"Check {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=5e-2)

    print("CUDAGraph correctness PASSED")
```

## Profiling

### torch.profiler

```python
from torch.profiler import profile, ProfilerActivity

def profile_model(model, inputs, trace_path="trace.json"):
    """Profile model execution on ROCm."""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(inputs)
    torch.cuda.current_stream().synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            _ = model(inputs)
        torch.cuda.current_stream().synchronize()

    # Export Perfetto-compatible trace
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")

    # Print summary
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

### CUDA event timing

```python
def time_with_events(fn, *args, warmup=5, iterations=50, **kwargs):
    """Time a function using CUDA events (avoids device-wide sync overhead)."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.current_stream().synchronize()

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iterations):
        start.record()
        fn(*args, **kwargs)
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))

    import statistics
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
    }
```

### rocm-smi monitoring

```bash
# Watch GPU utilization during test
watch -n 0.5 rocm-smi

# Log to CSV
rocm-smi --showuse --csv > gpu_usage.csv
```

## Stability Testing

### Repeated execution stability

```python
def test_stability(model, inputs, num_runs=100):
    """Check output stability across repeated runs."""
    outputs = []
    with torch.no_grad():
        for i in range(num_runs):
            out = model(inputs).cpu().float()
            outputs.append(out)

    # Check consistency
    ref = outputs[0]
    max_diffs = []
    for i, out in enumerate(outputs[1:], 1):
        diff = (ref - out).abs().max().item()
        max_diffs.append(diff)
        if diff > 1e-4:
            print(f"WARNING: Run {i} diverged by {diff:.6f}")

    print(f"Max cross-run diff: {max(max_diffs):.6f}")
    print(f"Mean cross-run diff: {sum(max_diffs)/len(max_diffs):.6f}")

    if max(max_diffs) < 1e-3:
        print("Stability test PASSED (deterministic)")
    elif max(max_diffs) < 1e-1:
        print("Stability test PASSED (non-deterministic but bounded)")
    else:
        print("Stability test FAILED (excessive variance)")
```

### Memory leak detection

```python
def test_memory_stability(model, inputs, num_runs=100):
    """Check for memory leaks across repeated runs."""
    torch.cuda.reset_peak_memory_stats()

    for i in range(num_runs):
        with torch.no_grad():
            _ = model(inputs)
        if i % 10 == 0:
            mem_mb = torch.cuda.memory_allocated() / 1024**2
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Run {i}: allocated={mem_mb:.1f}MB, peak={peak_mb:.1f}MB")

    print("Memory stability test complete (check for monotonic growth)")
```

## Sync Helpers

### Stream-level sync (preferred on ROCm)

```python
def _sync():
    """Stream-level sync (preferred on ROCm over device-wide sync)."""
    try:
        torch.cuda.current_stream().synchronize()
    except Exception:
        torch.cuda.synchronize()
```

### Why stream sync is preferred

- `torch.cuda.synchronize()` syncs ALL streams and ALL devices -- high overhead
- `torch.cuda.current_stream().synchronize()` syncs only the current stream
- On ROCm, device-wide sync can stall the HSA runtime unnecessarily
- For profiling and benchmarking, stream sync gives more accurate per-operation timing
