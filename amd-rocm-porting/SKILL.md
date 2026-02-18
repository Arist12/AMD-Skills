---
name: amd-rocm-porting
description: >
  Port NVIDIA CUDA codebases to AMD ROCm GPUs. Use when making PyTorch models run on AMD GPUs,
  replacing NVIDIA-specific libraries with AMD equivalents, fixing ROCm build/runtime failures,
  or porting C/C++ CUDA kernels to HIP.
---

# AMD ROCm Porting

Port NVIDIA CUDA codebases to AMD ROCm GPUs for functional equivalence.

## 5 Critical Rules (read first)

1. **NVIDIA isolation**: Every ROCm change MUST be gated behind `is_rocm`. The NVIDIA code path
   must be byte-for-byte identical to the pre-porting state. Test NVIDIA behavior after every
   change.
   ```python
   is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
   ```

2. **Compile mode**: NEVER use `mode="reduce-overhead"` on ROCm — causes 65x slowdown.
   Use `mode="default"` on ROCm, keep original mode for NVIDIA.

3. **Inductor**: Disable `triton.cudagraphs`, `triton.cudagraph_trees`, and `memory_planning`
   on ROCm. Details: [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md)

4. **Warp width**: AMD wavefronts are 64-wide (not 32). All ballot/mask operations need
   `uint64_t`. (C/C++ repos only; pure Python repos skip this.)

5. **Three-tier fallback**: AMD-optimized lib → PyTorch SDPA → pure PyTorch eager.
   Details: [references/library-and-model-adaptation.md](references/library-and-model-adaptation.md)

## Decision Tree: Which Phases to Run

```
Does the repo have C/C++ CUDA kernels (.cu / .cuh files)?
├── NO  → Skip Phases 2, 3, 4. Run Phases 1, 5, 6, 7, 8 only.
│         (Pure Python/PyTorch repos: openpi, most HuggingFace models)
└── YES → Run all 8 phases.
          Does it use flash-attn, CUTLASS, or custom extensions?
          ├── flash-attn only → Phase 5 (replace with aiter)
          ├── CUTLASS         → Phase 3 + manual CK rewrite
          └── custom kernels  → Full Phase 2 + 3 HIPIFY workflow
```

## Context Management

This porting work spans many files. To keep context manageable:
- **Load reference files lazily** — only read a reference when actively working on that phase.
- **Summarize findings** — after each phase, record a brief summary rather than retaining raw
  grep output in context.

## Phase Checklist

### Phase 0: Environment Setup (do this before anything else)

**Step 1 — Try running before installing anything.**
AMD Docker images often have PyTorch ROCm, aiter, flash-attn, and other GPU packages
pre-installed. Add the repo `src/` to `sys.path` in scripts to make the package importable
without `pip install`:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
```
Then run the target script and note only the `ModuleNotFoundError`s that actually occur.
Install those packages individually (`pip install --no-deps <pkg>`).

**Step 2 — Never run `pip install -e .` on AMD without exclusions.**
The `pyproject.toml` was written for NVIDIA and contains `jax[cuda12]` and `torch==X.Y.Z`.
Running `pip install -e .` will overwrite your ROCm JAX/PyTorch with CUDA versions.
If you must use pip install: `pip install --no-deps --ignore-requires-python -e .`
then install only the missing packages one by one.

**Step 3 — Python version constraint.**
`requires-python = ">=3.11"` in `pyproject.toml` is often a conservative NVIDIA-era constraint.
Pure Python packages are version-agnostic; use `--ignore-requires-python` flag.
If the AMD environment uses Python 3.10, consider lowering the constraint to `>=3.10`.

**Step 4 — Repos with JAX + PyTorch: use PyTorch-only path.**
If the repo supports both frameworks, focus exclusively on the PyTorch code path on AMD.
Skip all JAX-dependent code; do not attempt to install or fix JAX for ROCm.

### Phase 1: ROCm Detection & Flags
- Detect ROCm: `is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None`
- Detect GPU arch (never hardcode): `rocminfo | grep -o 'gfx[0-9a-f]*' | head -1` → e.g. `gfx942` (MI300X) or `gfx950` (MI350X)
- Set ROCm-safe env vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (omit `max_split_size_mb`)
- Verify GPU: `rocm-smi`, `rocminfo | grep gfx`, `hipcc --version`

### Phase 2: Source Translation (C/C++ only)
- Run `hipify-perl --inplace` for initial pass, then `hipify-clang` for complex templates
- Key header mappings: `cuda_runtime.h`→`hip/hip_runtime.h`, `cublas_v2.h`→`hipblas/hipblas.h`
- Flag inline PTX (`grep -rn "asm\s*("`) — cannot be auto-ported; flag CUTLASS — needs manual CK rewrite

### Phase 3: Architecture Adaptation (C/C++ only)
- Replace 32-bit ballot masks with `uint64_t` for AMD 64-wide wavefronts
- Replace `__popc` with `__popcll` for 64-bit masks; prefer 64-element shared memory tiles

### Phase 4: Build System (C/C++ only)
- Detect GPU arch at runtime — never hardcode: `GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-f]*' | head -1)`
- CMake: `find_package(HIP)`, set `CMAKE_HIP_ARCHITECTURES` to the detected arch
- setup.py: detect `is_rocm`, use `CUDAExtension` (hipcc handles `.cu` on ROCm)

### Phase 5: Library Replacement
- flash-attn → aiter (different API; wrap with three-tier fallback)
- NCCL → RCCL, cuBLAS → hipBLAS (drop-in via HIPIFY)
- pynvml: guard with `try/except`, use `torch.cuda.is_available()` as primary GPU check
- `PYTORCH_CUDA_ALLOC_CONF`: remove `max_split_size_mb` on ROCm (rejected by HIP allocator)
- Details + fallback patterns: [references/library-and-model-adaptation.md](references/library-and-model-adaptation.md)

### Phase 6: torch.compile Adaptation
- Gate compile mode: `"default"` on ROCm, original mode on NVIDIA
- Apply Inductor config: disable cudagraphs, memory_planning; use `ATEN` GEMM backend
- Details + monkey-patch: [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md)

### Phase 7: CUDAGraph / HIP Graph
- Apply Dynamo RNG patch before any graph capture
- Rules: no `.item()`, no shape branches, stream-level sync only
- Details + capture pattern: [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md)

### Phase 8: Verification
- Static: grep for remaining `cuda_runtime.h`, inline PTX, NVIDIA-specific types
- Build (C/C++): `GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-f]*' | head -1); hipcc -c kernels.hip --offload-arch=$GPU_ARCH`
- Functional: forward + backward pass, compare loss to CPU reference
- Numerical: `torch.testing.assert_close(rocm_out, cuda_ref, rtol=5e-2, atol=5e-2)`
- Details + golden vector methodology: [references/verification-methodology.md](references/verification-methodology.md)

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| `pip install -e .` on AMD | Overwrites ROCm torch with CUDA version | Use `--no-deps --ignore-requires-python`; install missing pkgs individually |
| NVIDIA-specific deps in pyproject.toml | Conflicts with pre-installed ROCm packages | Install with `--no-deps`; check what's already available before installing |
| `requires-python = ">=3.11"` | Install fails on Python 3.10 | Use `--ignore-requires-python`; or lower constraint to `>=3.10` |
| `reduce-overhead` compile mode | 65x slowdown, hangs | `mode="default"` on ROCm |
| `max_split_size_mb` in `PYTORCH_CUDA_ALLOC_CONF` | RuntimeError at startup | Remove on ROCm |
| Top-level `import pynvml` | ImportError or tests silently run on CPU | Guard with `try/except`; use `torch.cuda.is_available()` first |
| Inductor cudagraphs enabled | Slowdown, capture errors | `inductor_config.triton.cudagraphs = False` |
| Inductor memory_planning | Deep recursion crash | `inductor_config.memory_planning = False` |
| `torch.cuda.get_rng_state()` during capture | RuntimeError | Apply Dynamo RNG patch |
| `torch.backends.cuda.matmul.allow_tf32` | AttributeError on ROCm | Gate behind `if not is_rocm` |
| 32-bit warp masks (C/C++) | Silent wrong results | Use `uint64_t` for ballot/active masks |

## References

Load only when actively working on that phase:

- **Phase 5**: [references/library-and-model-adaptation.md](references/library-and-model-adaptation.md) — Library mapping, aiter API, three-tier fallback, pynvml/alloc_conf fixes
- **Phases 6–7**: [references/torch-compile-and-cudagraph.md](references/torch-compile-and-cudagraph.md) — Inductor config, compile mode, Dynamo RNG patch, graph capture
- **Phase 8**: [references/verification-methodology.md](references/verification-methodology.md) — 4-level pyramid, tolerance table, static analysis greps
