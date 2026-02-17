# Profiling Workflow for AMD GPU Inference

## Table of Contents
- [Quick Start: CUDA Events](#quick-start-cuda-events)
- [rocprof Kernel Tracing](#rocprof-kernel-tracing)
- [torch.profiler](#torchprofiler)
- [Chrome Tracing (Perfetto)](#chrome-tracing-perfetto)
- [What to Measure](#what-to-measure)
- [Measurement Methodology](#measurement-methodology)

## Quick Start: CUDA Events

Preferred method for latency measurement. GPU-side timing avoids device-wide sync overhead:

```python
import torch

def benchmark_latency(fn, *args, warmup=10, iterations=100):
    """Benchmark GPU latency using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(iterations):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    import statistics
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
```

**Why not wall-clock time?**
- `time.time()` or `time.perf_counter()` includes CPU overhead and sync wait
- CUDA events measure only GPU execution time
- On ROCm, `hipDeviceSynchronize` itself adds 3.7x more overhead than CUDA equivalent

## rocprof Kernel Tracing

AMD's profiling tool for kernel-level analysis.

### Basic Kernel Trace

```bash
# Trace all GPU kernels
rocprof --hip-trace --hsa-trace -o trace.csv python inference.py

# Output: trace.csv with per-kernel timing
```

### Kernel Statistics

```bash
# Aggregate kernel statistics
rocprof --stats -o stats.csv python inference.py
```

### Parse Results

```python
import pandas as pd

# Read kernel trace
df = pd.read_csv("trace.csv")

# Top kernels by total time
top = df.groupby("Name").agg(
    count=("DurationNs", "count"),
    total_us=("DurationNs", lambda x: x.sum() / 1000),
    mean_us=("DurationNs", lambda x: x.mean() / 1000),
).sort_values("total_us", ascending=False)

print(top.head(20))
```

### Key Metrics to Extract

- **Kernel count**: Total number of kernel launches per inference step
- **GEMM time fraction**: % of time in GEMM kernels (target: <60%)
- **Sync overhead**: Time in `hipDeviceSynchronize` / `hipStreamSynchronize`
- **Launch overhead**: Gap between kernel end and next kernel start
- **Top-10 hotspot kernels**: Focus optimization on these first

### Caveats

- rocprof adds ~5us overhead per kernel launch (do not use for absolute timing)
- Use for relative comparisons and identifying hotspots, not absolute latency
- Separate profiling runs from benchmark runs

## torch.profiler

PyTorch's built-in profiler for combined CPU+GPU view:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=3, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(10):
        output = model(inputs)
        prof.step()

# Print kernel summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
```

### Quick One-Shot Profile

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    with torch.no_grad():
        output = model(inputs)

# Sort by GPU time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

### Kernel Count

```python
events = prof.key_averages()
total_kernels = sum(e.count for e in events if e.device_type == DeviceType.CUDA)
print(f"Total kernel launches: {total_kernels}")
# Target: reduce by 15-20% with fusion optimizations
```

## Chrome Tracing (Perfetto)

Visualize traces in Chrome or Perfetto UI:

1. Export trace: `prof.export_chrome_trace("trace.json")`
2. Open `chrome://tracing` or https://ui.perfetto.dev
3. Load `trace.json`

### What to Look For

- **Gaps between kernels**: Launch overhead (target for CUDAGraph)
- **Wide sync bars**: `hipDeviceSynchronize` overhead
- **Thin kernel slivers**: Small kernels that should be fused
- **CPU-GPU overlap**: Ensure CPU isn't bottlenecking GPU
- **Memory copies**: Unexpected H2D/D2H transfers

## What to Measure

### Before Each Optimization Phase

| Metric | How to Measure | Target |
|--------|---------------|--------|
| E2E latency | CUDA events | Decreasing |
| Kernel count | torch.profiler / rocprof | Decreasing |
| GEMM time % | rocprof trace | <60% of total |
| Sync overhead | rocprof (hipSync* events) | <5% of total |
| Launch overhead | rocprof (inter-kernel gaps) | Near zero (with CUDAGraph) |
| Peak memory | `torch.cuda.max_memory_allocated()` | Stable |
| Numerical accuracy | `torch.allclose(baseline, optimized, atol=1e-2)` | True |

### A/B Testing Pattern

```python
def ab_test(baseline_fn, optimized_fn, inputs, tolerance=1e-2):
    """Compare baseline and optimized implementations."""
    # Correctness
    base_out = baseline_fn(*inputs)
    opt_out = optimized_fn(*inputs)
    correct = torch.allclose(base_out, opt_out, atol=tolerance)

    # Performance
    base_lat = benchmark_latency(baseline_fn, *inputs)
    opt_lat = benchmark_latency(optimized_fn, *inputs)

    speedup = base_lat["mean_ms"] / opt_lat["mean_ms"]
    print(f"Correct: {correct}")
    print(f"Baseline: {base_lat['mean_ms']:.2f} ms")
    print(f"Optimized: {opt_lat['mean_ms']:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    return correct, speedup
```

## Measurement Methodology

### Separate Compile Time from Steady-State

torch.compile JIT compilation happens on first call. Always separate:

```python
# First call: triggers compilation (can take minutes)
with torch.no_grad():
    _ = model(inputs)
torch.cuda.synchronize()

# Now measure steady-state (compilation already cached)
latency = benchmark_latency(model, inputs)
```

### Sufficient Warmup

After CUDAGraph capture, run a few replay iterations before measuring:

```python
# Warmup replays
for _ in range(5):
    graph.replay()
torch.cuda.synchronize()

# Now measure
latencies = []
for _ in range(100):
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    latencies.append(start.elapsed_time(end))
```

### Stream-Level Sync

On ROCm, always prefer stream-level sync over device-wide:

```python
# GOOD: Stream-level sync (fast)
torch.cuda.current_stream().synchronize()

# BAD: Device-wide sync (3.7x slower on ROCm vs CUDA)
torch.cuda.synchronize()  # Use only when absolutely necessary
```
