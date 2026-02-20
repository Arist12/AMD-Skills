#!/usr/bin/env python3
"""
AMD/ROCm Docker Environment Probe

Run this BEFORE any torch.compile(), CUDAGraph capture, or kernel optimization.
Detects hidden framework defaults, known Docker bugs, and missing packages.

Usage:
    python env_probe.py              # Full probe
    python env_probe.py --json       # JSON output (for programmatic consumption)
    python env_probe.py --fix        # Print a Python snippet that applies all recommended fixes

No dependencies beyond PyTorch (which your AMD Docker already has).
"""

import argparse
import json
import os
import sys
import importlib
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str          # CRITICAL, WARNING, INFO
    category: str          # surface, runtime_defaults, known_bugs, env_vars
    title: str
    detail: str
    fix: Optional[str]     # Recommended fix (Python code or env var)

    def __str__(self):
        icon = {"CRITICAL": "!!!", "WARNING": " ! ", "INFO": " i "}[self.severity]
        lines = [f"[{icon}] {self.severity}: {self.title}"]
        lines.append(f"      {self.detail}")
        if self.fix:
            lines.append(f"      Fix: {self.fix}")
        return "\n".join(lines)


findings: list[Finding] = []


def add(severity, category, title, detail, fix=None):
    findings.append(Finding(severity, category, title, detail, fix))


# ─── Category 1: Surface Facts ─────────────────────────────────────────────

def check_python():
    v = sys.version.split()[0]
    add("INFO", "surface", "Python version", f"Python {v}")


def check_pytorch():
    try:
        import torch
        ver = torch.__version__
        cuda_ver = getattr(torch.version, "cuda", None)
        hip_ver = getattr(torch.version, "hip", None)
        is_rocm = hip_ver is not None

        if is_rocm:
            add("INFO", "surface", "PyTorch (ROCm)", f"torch {ver}, HIP {hip_ver}")
        elif cuda_ver:
            add("WARNING", "surface", "PyTorch (CUDA, not ROCm)",
                f"torch {ver}, CUDA {cuda_ver}. Expected ROCm build.",
                "Install PyTorch ROCm build or use an AMD Docker image")
        else:
            add("WARNING", "surface", "PyTorch (unknown backend)", f"torch {ver}")
    except ImportError:
        add("CRITICAL", "surface", "PyTorch not installed", "Cannot proceed without PyTorch",
            "pip install torch (ROCm build)")


def check_triton():
    try:
        import triton
        ver = triton.__version__
        add("INFO", "surface", "Triton version", f"triton {ver}")
    except ImportError:
        add("WARNING", "surface", "Triton not installed",
            "torch.compile Triton backend unavailable. Not critical if using eager + CUDAGraph.",
            "pip install triton")


def check_rocm_hardware():
    import subprocess
    # rocm-smi
    try:
        result = subprocess.run(["rocm-smi", "--showproductname"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if "GPU" in l or "Card" in l]
            if lines:
                add("INFO", "surface", "GPU (rocm-smi)", "; ".join(lines[:4]))
            else:
                add("INFO", "surface", "rocm-smi output", result.stdout.strip()[:200])
        else:
            add("WARNING", "surface", "rocm-smi failed", result.stderr.strip()[:200])
    except FileNotFoundError:
        add("WARNING", "surface", "rocm-smi not found", "Cannot detect GPU hardware")
    except subprocess.TimeoutExpired:
        add("WARNING", "surface", "rocm-smi timed out", "GPU detection timed out after 10s")

    # gfx target
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gfx_targets = set()
            for line in result.stdout.split("\n"):
                if "gfx" in line.lower():
                    for word in line.split():
                        if word.startswith("gfx"):
                            gfx_targets.add(word)
            if gfx_targets:
                add("INFO", "surface", "GFX targets", ", ".join(sorted(gfx_targets)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def check_aiter():
    try:
        import aiter
        ver = getattr(aiter, "__version__", "unknown")
        add("INFO", "surface", "AITER installed", f"version: {ver}")

        # Check key functions
        available = []
        missing = []
        for func_name in ["flash_attn_func", "fused_add_rms_norm",
                          "gelu_tanh_and_mul", "rope_cached_fwd_impl"]:
            try:
                getattr(aiter, func_name)
                available.append(func_name)
            except AttributeError:
                missing.append(func_name)

        if available:
            add("INFO", "surface", "AITER functions available", ", ".join(available))
        if missing:
            add("WARNING", "surface", "AITER functions missing", ", ".join(missing))
    except ImportError:
        add("WARNING", "surface", "AITER not installed",
            "AMD-optimized kernels unavailable. Flash attention, fused norms, etc. will use fallbacks.",
            "pip install aiter (or check if Docker has it at a non-standard path)")


def check_composable_kernel():
    try:
        import ck4inductor
        add("INFO", "surface", "Composable Kernel (ck4inductor)", "Available")
    except ImportError:
        try:
            # Some Docker images install CK differently
            import composable_kernel
            add("INFO", "surface", "Composable Kernel", "Available (composable_kernel)")
        except ImportError:
            add("INFO", "surface", "Composable Kernel", "Not found (optional)")


def check_hipblaslt():
    try:
        import torch
        # Check if hipBLASLt is available through torch
        has_hipblaslt = hasattr(torch.ops, "hipblaslt") or "hipblaslt" in str(getattr(torch, "_C", ""))
        # More reliable: try a small GEMM through the preferred backend
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            try:
                a = torch.randn(32, 32, device="cuda", dtype=torch.float16)
                b = torch.randn(32, 32, device="cuda", dtype=torch.float16)
                c = torch.mm(a, b)
                add("INFO", "surface", "hipBLASLt GEMM test", "Small GEMM succeeded on GPU")
            except Exception as e:
                err_str = str(e)
                if "HIPBLAS_STATUS_NOT_INITIALIZED" in err_str:
                    add("CRITICAL", "known_bugs", "hipBLASLt solver discovery broken",
                        f"HIPBLAS_STATUS_NOT_INITIALIZED on small GEMM. This Docker has the hipBLASLt bug.",
                        "Use rocBLAS fallback: set inductor_config.max_autotune_gemm_backends = 'ATEN'")
                else:
                    add("WARNING", "known_bugs", "GEMM test failed", err_str[:200])
        else:
            add("INFO", "surface", "hipBLASLt check", "Skipped (not ROCm)")
    except Exception as e:
        add("INFO", "surface", "hipBLASLt check", f"Could not check: {e}")


# ─── Category 2: Runtime Behavior Defaults ──────────────────────────────────

def check_inductor_defaults():
    try:
        import torch._inductor.config as cfg

        # max_autotune — THE big one
        max_autotune = getattr(cfg, "max_autotune", None)
        if max_autotune is True:
            add("CRITICAL", "runtime_defaults",
                "Inductor max_autotune is True (Docker default)",
                "torch.compile will benchmark every GEMM across ATEN+TRITON+CPP backends. "
                "With hundreds of matmuls in a compiled graph (e.g., 10 denoise steps), "
                "autotuning NEVER finishes. The process hangs indefinitely with no error.",
                "inductor_config.max_autotune = False")
        elif max_autotune is False:
            add("INFO", "runtime_defaults", "Inductor max_autotune", "False (safe)")
        else:
            add("INFO", "runtime_defaults", "Inductor max_autotune", f"{max_autotune}")

        # max_autotune_gemm_backends
        backends = getattr(cfg, "max_autotune_gemm_backends", None)
        if backends and "TRITON" in str(backends).upper() and "ATEN" in str(backends).upper():
            add("WARNING", "runtime_defaults",
                f"Inductor GEMM backends: {backends}",
                "Autotuning across multiple backends is slow. On ROCm, ATEN (rocBLAS) "
                "consistently wins for GEMMs. Skip Triton/CPP autotuning.",
                "inductor_config.max_autotune_gemm_backends = 'ATEN'")
        elif backends:
            add("INFO", "runtime_defaults", "Inductor GEMM backends", str(backends))

        # triton.cudagraphs
        triton_cfg = getattr(cfg, "triton", None)
        if triton_cfg:
            cudagraphs = getattr(triton_cfg, "cudagraphs", None)
            if cudagraphs is True:
                add("WARNING", "runtime_defaults",
                    "Inductor triton.cudagraphs is True",
                    "Triton-managed CUDAGraphs are unstable on ROCm. Can cause hangs or "
                    "incorrect results.",
                    "inductor_config.triton.cudagraphs = False")
            else:
                add("INFO", "runtime_defaults", "Inductor triton.cudagraphs", str(cudagraphs))

            cudagraph_trees = getattr(triton_cfg, "cudagraph_trees", None)
            if cudagraph_trees is True:
                add("WARNING", "runtime_defaults",
                    "Inductor triton.cudagraph_trees is True",
                    "CUDAGraph trees are unstable on ROCm.",
                    "inductor_config.triton.cudagraph_trees = False")
            else:
                add("INFO", "runtime_defaults", "Inductor triton.cudagraph_trees",
                    str(cudagraph_trees))

        # memory_planning
        memory_planning = getattr(cfg, "memory_planning", None)
        if memory_planning is True:
            add("WARNING", "runtime_defaults",
                "Inductor memory_planning is True",
                "Memory planning causes deep recursion crashes on ROCm with complex models.",
                "inductor_config.memory_planning = False")
        else:
            add("INFO", "runtime_defaults", "Inductor memory_planning", str(memory_planning))

        # coordinate_descent_tuning
        coord = getattr(cfg, "coordinate_descent_tuning", None)
        add("INFO", "runtime_defaults", "Inductor coordinate_descent_tuning", str(coord))

        # freezing
        freezing = getattr(cfg, "freezing", None)
        add("INFO", "runtime_defaults", "Inductor freezing", str(freezing))

    except ImportError:
        add("WARNING", "runtime_defaults", "Cannot import torch._inductor.config",
            "Inductor configuration not available")
    except Exception as e:
        add("WARNING", "runtime_defaults", "Inductor config check error", str(e)[:200])


def check_dynamo_defaults():
    try:
        import torch._dynamo.config as cfg

        cache_limit = getattr(cfg, "cache_size_limit", None)
        if cache_limit is not None and cache_limit < 64:
            add("WARNING", "runtime_defaults",
                f"Dynamo cache_size_limit is {cache_limit} (too small)",
                "Complex models with many graph breaks will trigger recompilation loops. "
                "Each recompilation re-traces and re-compiles the entire graph.",
                f"dynamo_config.cache_size_limit = 128  # currently {cache_limit}")
        else:
            add("INFO", "runtime_defaults", "Dynamo cache_size_limit", str(cache_limit))

        suppress = getattr(cfg, "suppress_errors", None)
        if suppress is True:
            add("WARNING", "runtime_defaults",
                "Dynamo suppress_errors is True",
                "Graph break errors are silently swallowed. You won't know when torch.compile "
                "falls back to eager mode.",
                "dynamo_config.suppress_errors = False")
        else:
            add("INFO", "runtime_defaults", "Dynamo suppress_errors", str(suppress))

    except ImportError:
        add("WARNING", "runtime_defaults", "Cannot import torch._dynamo.config",
            "Dynamo configuration not available")
    except Exception as e:
        add("WARNING", "runtime_defaults", "Dynamo config check error", str(e)[:200])


def check_torch_backends():
    try:
        import torch

        # cudnn benchmark
        benchmark = torch.backends.cudnn.benchmark
        add("INFO", "runtime_defaults", "cudnn.benchmark", str(benchmark))

        # allow_tf32
        cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        add("INFO", "runtime_defaults", "allow_tf32",
            f"cuda.matmul={cuda_tf32}, cudnn={cudnn_tf32}")

        # float32_matmul_precision
        precision = torch.get_float32_matmul_precision()
        add("INFO", "runtime_defaults", "float32_matmul_precision", str(precision))

    except Exception as e:
        add("INFO", "runtime_defaults", "torch.backends check", f"Error: {e}")


def check_tunableop():
    """PyTorch TunableOp — ROCm-specific GEMM tuning system."""
    try:
        import torch
        # TunableOp is controlled by env vars in ROCm PyTorch
        tunable_enabled = os.environ.get("PYTORCH_TUNABLEOP_ENABLED", None)
        tunable_tuning = os.environ.get("PYTORCH_TUNABLEOP_TUNING", None)

        if tunable_enabled == "1" and tunable_tuning == "1":
            add("WARNING", "runtime_defaults",
                "TunableOp tuning is ON",
                "PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=1. "
                "First-run GEMMs will be autotuned (slow first iteration). "
                "Results are cached in tunableop_results.csv. This is fine for benchmarking "
                "but will cause unpredictable first-run latency.",
                "Set PYTORCH_TUNABLEOP_TUNING=0 after initial tuning pass to use cached results")
        elif tunable_enabled == "1":
            add("INFO", "runtime_defaults", "TunableOp",
                "Enabled but not tuning (using cached results)")
        else:
            add("INFO", "runtime_defaults", "TunableOp",
                f"ENABLED={tunable_enabled}, TUNING={tunable_tuning}")
    except Exception as e:
        add("INFO", "runtime_defaults", "TunableOp check", f"Error: {e}")


# ─── Category 3: Known Bug Markers ─────────────────────────────────────────

def check_fp8_flash_attn():
    try:
        import torch
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            return

        try:
            from aiter import flash_attn_func
            # Try FP8 dtype
            if hasattr(torch, "float8_e4m3fnuz"):
                q = torch.randn(1, 1, 16, 64, device="cuda").to(torch.float8_e4m3fnuz)
                k = torch.randn(1, 1, 16, 64, device="cuda").to(torch.float8_e4m3fnuz)
                v = torch.randn(1, 1, 16, 64, device="cuda").to(torch.float8_e4m3fnuz)
                try:
                    out = flash_attn_func(q, k, v)
                    add("INFO", "known_bugs", "FP8 flash attention", "Works (float8_e4m3fnuz)")
                except Exception as e:
                    err = str(e)[:200]
                    add("WARNING", "known_bugs", "FP8 flash attention broken",
                        f"AITER flash_attn_func fails with FP8 inputs: {err}",
                        "Use bf16/fp16 for flash attention. FP8 attention may be fixed in newer Docker.")
            else:
                add("INFO", "known_bugs", "FP8 flash attention", "float8_e4m3fnuz dtype not available")
        except ImportError:
            pass  # AITER not installed, already flagged
    except Exception as e:
        add("INFO", "known_bugs", "FP8 flash attn check", f"Error: {e}")


def check_asm_gemm_kernels():
    """Check if gfx950/gfx942-tuned ASM GEMM kernels are present and don't hang."""
    try:
        import torch
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            return

        try:
            from aiter import gemm_a8w8_blockscaled
            add("INFO", "known_bugs", "AITER ASM GEMM (a8w8_blockscaled)", "Import succeeded")
        except (ImportError, AttributeError):
            add("INFO", "known_bugs", "AITER ASM GEMM", "a8w8_blockscaled not available (optional)")

        try:
            from aiter import gemm_ck
            add("INFO", "known_bugs", "AITER CK GEMM", "Import succeeded")
        except (ImportError, AttributeError):
            add("INFO", "known_bugs", "AITER CK GEMM", "Not available (optional)")

    except Exception as e:
        add("INFO", "known_bugs", "ASM GEMM check", f"Error: {e}")


# ─── Category 4: Environment Variables ──────────────────────────────────────

def check_env_vars():
    important_vars = {
        # GPU visibility
        "HIP_VISIBLE_DEVICES": ("Controls which GPUs are visible", None),
        "ROCR_VISIBLE_DEVICES": ("Controls which GPUs are visible (ROCr level)", None),
        "CUDA_VISIBLE_DEVICES": ("May also work on ROCm for GPU visibility", None),
        # Performance
        "HSA_ENABLE_SDMA": ("DMA engine for host-device transfers", None),
        "HIP_FORCE_DEV_KERNARG": ("Force device-side kernel arguments", None),
        "GPU_MAX_HW_QUEUES": ("Max hardware queues per GPU", None),
        "HIP_LAUNCH_BLOCKING": ("Synchronous kernel launches (debug)", "0 for production"),
        # PyTorch / Inductor
        "TORCH_COMPILE_DEBUG": ("Inductor debug output", None),
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES": ("Disable inductor caches", None),
        "TORCHINDUCTOR_MAX_AUTOTUNE": ("Override max_autotune from env", None),
        # Triton
        "TRITON_CACHE_DIR": ("Triton compilation cache directory", None),
        "TRITON_PRINT_AUTOTUNING": ("Print Triton autotuning results", None),
    }

    set_vars = []
    for var, (desc, _) in important_vars.items():
        val = os.environ.get(var, None)
        if val is not None:
            set_vars.append(f"{var}={val}")

    if set_vars:
        add("INFO", "env_vars", "Environment variables set", "; ".join(set_vars))
    else:
        add("INFO", "env_vars", "No AMD/ROCm env vars explicitly set",
            "Using Docker defaults for all HIP/ROCm/Inductor env vars")

    # Specific checks
    if os.environ.get("HIP_LAUNCH_BLOCKING") == "1":
        add("WARNING", "env_vars",
            "HIP_LAUNCH_BLOCKING=1 (synchronous mode)",
            "All kernel launches are synchronous. This kills performance but is useful for debugging.",
            "Unset HIP_LAUNCH_BLOCKING or set to 0 for production/benchmarking")

    if os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1":
        add("CRITICAL", "env_vars",
            "TORCHINDUCTOR_MAX_AUTOTUNE=1 set via env var",
            "This overrides any Python-level inductor_config.max_autotune setting. "
            "torch.compile will benchmark every GEMM and may hang indefinitely.",
            "unset TORCHINDUCTOR_MAX_AUTOTUNE")


# ─── Runner ─────────────────────────────────────────────────────────────────

def run_all_checks():
    # Category 1: Surface
    check_python()
    check_pytorch()
    check_triton()
    check_rocm_hardware()
    check_aiter()
    check_composable_kernel()
    check_hipblaslt()

    # Category 2: Runtime defaults
    check_inductor_defaults()
    check_dynamo_defaults()
    check_torch_backends()
    check_tunableop()

    # Category 3: Known bugs
    check_fp8_flash_attn()
    check_asm_gemm_kernels()

    # Category 4: Environment variables
    check_env_vars()


def print_report():
    criticals = [f for f in findings if f.severity == "CRITICAL"]
    warnings = [f for f in findings if f.severity == "WARNING"]
    infos = [f for f in findings if f.severity == "INFO"]

    print("=" * 72)
    print("  AMD/ROCm Docker Environment Probe Report")
    print("=" * 72)
    print()

    if criticals:
        print(f"  CRITICAL issues: {len(criticals)} (MUST fix before proceeding)")
    if warnings:
        print(f"  Warnings:        {len(warnings)} (fix before benchmarking)")
    print(f"  Info:            {len(infos)}")
    print()

    if criticals:
        print("-" * 72)
        print("  CRITICAL — Fix these before writing any code")
        print("-" * 72)
        for f in criticals:
            print()
            print(f)
        print()

    if warnings:
        print("-" * 72)
        print("  WARNINGS — Fix before benchmarking")
        print("-" * 72)
        for f in warnings:
            print()
            print(f)
        print()

    if infos:
        print("-" * 72)
        print("  INFO — Environment state")
        print("-" * 72)
        for f in infos:
            print()
            print(f)
        print()

    print("=" * 72)
    if criticals:
        print(f"  ACTION REQUIRED: {len(criticals)} critical issue(s) found.")
        print("  Apply fixes above before any torch.compile() or optimization work.")
    else:
        print("  No critical issues found. Environment looks safe to proceed.")
    print("=" * 72)


def print_fix_snippet():
    """Print a Python snippet that applies all recommended fixes."""
    fixes = [f for f in findings if f.fix and f.severity in ("CRITICAL", "WARNING")]
    if not fixes:
        print("# No fixes needed — environment looks good.")
        return

    print("# Auto-generated fix snippet from env_probe.py")
    print("# Apply this at the top of your script, before torch.compile() or CUDAGraph capture.")
    print()

    has_inductor = any("inductor_config" in (f.fix or "") for f in fixes)
    has_dynamo = any("dynamo_config" in (f.fix or "") for f in fixes)

    if has_inductor:
        print("import torch._inductor.config as inductor_config")
    if has_dynamo:
        print("import torch._dynamo.config as dynamo_config")
    if has_inductor or has_dynamo:
        print()

    for f in fixes:
        if f.fix:
            print(f"# {f.severity}: {f.title}")
            # Only print Python-assignable fixes (skip prose)
            if "=" in f.fix and not f.fix.startswith("Set ") and not f.fix.startswith("Use "):
                print(f"{f.fix}")
            else:
                print(f"# {f.fix}")
            print()


def print_json():
    print(json.dumps([asdict(f) for f in findings], indent=2))


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMD/ROCm Docker Environment Probe")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fix", action="store_true", help="Print Python fix snippet")
    args = parser.parse_args()

    run_all_checks()

    if args.json:
        print_json()
    elif args.fix:
        print_fix_snippet()
    else:
        print_report()

    # Exit code: 2 if criticals, 1 if warnings, 0 if clean
    criticals = sum(1 for f in findings if f.severity == "CRITICAL")
    warnings = sum(1 for f in findings if f.severity == "WARNING")
    if criticals:
        sys.exit(2)
    elif warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
