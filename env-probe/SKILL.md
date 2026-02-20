---
name: env-probe
description: >
  Inspect AMD/ROCm Docker runtime environment before writing any code. Use BEFORE torch.compile,
  CUDAGraph capture, or any kernel optimization. Detects hidden framework defaults (inductor
  max_autotune, triton.cudagraphs), known Docker-specific bugs (hipBLASLt solver crash, FP8
  flash attn), and missing packages. Outputs CRITICAL/WARNING/INFO report with recommended fixes.
  Triggered by: starting work in an AMD Docker, "check environment", "why is torch.compile hanging",
  "env probe", Phase 0 of any AMD optimization experiment.
---

# AMD/ROCm Docker Environment Probe

**Run this before writing any optimization code.** AMD Docker images silently set framework defaults
that differ from stock PyTorch. These hidden defaults cause stalls, crashes, and wrong results that
are impossible to diagnose by looking at code alone.

## Why This Exists

Problem: ROCm Docker images override PyTorch/Triton defaults at the system level. For example,
`max_autotune=True` as a global default means `torch.compile(mode="default")` benchmarks every
GEMM across ATEN+TRITON+CPP backends. With hundreds of matmuls in a compiled graph, autotuning
never finishes — the process hangs indefinitely with no error message.

These defaults are invisible to `pip list`, `rocm-smi`, or any surface-level inspection. You have
to introspect the framework config objects at runtime to see them.

## How to Use

### Step 1: Run the probe script

```bash
python /path/to/env_probe.py
```

Or if the skill is installed as a Claude Code command, copy the probe script from
[references/env_probe.py](references/env_probe.py) and run it inside your Docker container.

The probe script is self-contained — no dependencies beyond PyTorch (which your Docker already has).

### Step 2: Read the output

The probe outputs a structured report with three severity levels:

| Level | Meaning | Action |
|-------|---------|--------|
| **CRITICAL** | Will cause hangs, crashes, or silent wrong results | **Must fix before proceeding** |
| **WARNING** | Suboptimal default, will hurt performance | Fix before benchmarking |
| **INFO** | Informational, no action needed | Document for reproducibility |

### Step 3: Apply fixes

Each CRITICAL/WARNING item includes a recommended fix — either a Python config line or an
environment variable to set. Apply these fixes at the top of your script, before any
`torch.compile()` or `torch.cuda.CUDAGraph()` call.

## What the Probe Checks

### Category 1: Surface Facts (versions, hardware)
- Python version, PyTorch version, Triton version
- ROCm version, GPU architecture (gfx target)
- AITER, Composable Kernel, flash-attn availability and versions
- hipBLASLt availability

### Category 2: Runtime Behavior Defaults (the hidden landmines)
- `torch._inductor.config.max_autotune` — if True, causes indefinite stall with torch.compile
- `torch._inductor.config.max_autotune_gemm_backends` — which backends inductor will benchmark
- `torch._inductor.config.triton.cudagraphs` — unstable on ROCm
- `torch._inductor.config.triton.cudagraph_trees` — unstable on ROCm
- `torch._inductor.config.memory_planning` — causes deep recursion crash on ROCm
- `torch._dynamo.config.cache_size_limit` — too small causes recompilation loops
- `torch.backends.cudnn.benchmark` and `allow_tf32` defaults

### Category 3: Known Bug Markers
- hipBLASLt solver discovery (HIPBLAS_STATUS_NOT_INITIALIZED)
- FP8 flash attention availability
- gfx950/gfx942 ASM GEMM kernel availability
- AITER function signatures (argument combos that were broken in older versions)

### Category 4: Environment Variables
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`
- `HSA_ENABLE_SDMA`, `HIP_FORCE_DEV_KERNARG`
- `PYTORCH_TUNABLEOP_ENABLED`, `PYTORCH_TUNABLEOP_TUNING`
- `TORCH_COMPILE_DEBUG`, `TORCHINDUCTOR_*` overrides

## Recommended Inductor Configuration for ROCm

When the probe flags inductor defaults as CRITICAL, apply this configuration block before any
`torch.compile()` call:

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# Prevent indefinite GEMM autotuning stall
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm_backends = "ATEN"

# Disable unstable triton cudagraphs on ROCm
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Prevent deep recursion crash
inductor_config.memory_planning = False

# Prevent cache eviction / recompilation loops
dynamo_config.cache_size_limit = 128
```

See [references/inductor-rocm-defaults.md](references/inductor-rocm-defaults.md) for the full
explanation of each setting and when you might want to override them.

## Integration with Other Skills

- **amd-rocm-porting**: Run env-probe as Phase 0.5 (after Phase 0 environment setup, before Phase 1 porting)
- **amd-inference-optimization**: Run env-probe before Phase 0 profiling baseline
- **rocprofv3-profiler**: Probe checks that rocprofv3 is available and functional

## Adding New Checks

When you discover a new Docker-specific gotcha, add it to `references/env_probe.py`:
1. Add the check function
2. Add it to the appropriate category in `run_all_checks()`
3. Include the severity level (CRITICAL/WARNING/INFO) and recommended fix
4. Document the failure mode (what happens if the agent doesn't know about this)

This skill is meant to grow — every experiment that hits an environment issue should contribute
a new check back to the probe.
