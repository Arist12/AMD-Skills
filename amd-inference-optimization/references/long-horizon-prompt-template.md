# Long-Horizon Optimization Prompt Template

Use this template when launching a Kimi-Agent session for overnight inference optimization.
Replace `{{placeholders}}` with actual values.

---

## Template

```
Optimize the inference latency of {{MODEL_NAME}} on AMD {{GPU_MODEL}} (ROCm).

Working directory: {{WORK_DIR}}

## Hard Target
- **p50 latency target: <{{TARGET_MS}}ms** at {{INFERENCE_PARAMS}}
- Current baseline: ~{{BASELINE_MS}}ms

## Mandatory Rules

1. **DO NOT choose STOP** until one of these conditions is true:
   - p50 < {{TARGET_MS}}ms (verified by running benchmark with CUDA events)
   - ALL optimization phases (0-7) have been attempted and measured
   - A blocking error prevents ANY further progress (document the specific error)

2. **Follow the optimization ladder in order**: Profile -> CUDAGraph -> Attention -> GEMM -> Fusion -> Triton -> Fine-tune -> Verify. After each phase, run the benchmark. If target not met, proceed to next phase.

3. **torch.compile + CUDAGraph strategy**: Use `torch.compile(mode="default")` for kernel fusion COMBINED WITH manual `torch.cuda.CUDAGraph()` capture. These are complementary. Do NOT disable torch.compile to use CUDAGraph. Set `inductor_config.triton.cudagraphs = False` to disable Inductor's broken internal capture.

4. **After each phase**, log a progress table:
   | Phase | p50 (ms) | vs Baseline | vs Target | Next Action |
   |-------|----------|-------------|-----------|-------------|

5. **Commit after each successful phase** with message format: `perf: Phase N - <description> (p50: XXms)`

6. Use `/opt/venv/bin/python` as the Python interpreter.
7. Do NOT reduce {{INFERENCE_PARAMS}} to hit the target artificially.
8. Measure with CUDA events (not wall-clock time).

## Phase-Specific Instructions

### Phase 1: CUDAGraph
- Apply Dynamo RNG patch (torch._dynamo.convert_frame, NOT torch._dynamo.utils)
- Convert while-loops to for-loops for static control flow
- torch.compile(mode="default") FIRST, then manual CUDAGraph capture
- If compiled capture fails, fall back to eager + CUDAGraph and document the error

### Phase 2: Attention
- Replace eager attention with SDPA (F.scaled_dot_product_attention)
- Try aiter flash attention if available
- Do NOT graph-break around attention ops

### Phase 3: GEMM
- Set inductor_config.max_autotune_gemm_backends = "ATEN" (rocBLAS >> Triton)
- Try aiter tuned GEMM dispatcher if available

### Phase 4-6: See skill references for details

## Verification
After EVERY phase: `python verification-gate.py --target-ms {{TARGET_MS}} --p50-ms <measured>`
```

---

## Example Usage

```
Optimize the inference latency of OpenPI pi0 (Gemma 2B + 300M expert) on AMD MI355X (ROCm).

Working directory: /sgl-workspace/openpi

## Hard Target
- **p50 latency target: <24ms** at num_steps=10, batch_size=1
- Current baseline: ~43ms

## Mandatory Rules
[... rest of template filled in ...]
```
