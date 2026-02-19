# CUDAGraph Integration (Not Just a Wrapper)

The most common agent failure mode: writing a `CUDAGraphWrapper` class in a utility file,
committing it, and never actually calling it. This reference shows how to integrate CUDAGraph
into an actual model so the production code path uses graph replay.

## The Key Principle

CUDAGraph must be integrated INTO the model's inference method, not wrapped around it. The
model's `sample_actions()` or `forward()` method itself must contain the capture/replay logic.

## CRITICAL: torch.compile + CUDAGraph Are Complementary

On ROCm, the **recommended** approach is:
1. `torch.compile(mode="default")` — provides kernel fusion (fusing pointwise ops, epilogues, etc.)
2. Manual `torch.cuda.CUDAGraph()` — eliminates CPU-side kernel launch overhead

These are **complementary optimizations**, not alternatives. Disabling torch.compile to use
CUDAGraph loses all kernel fusion benefits and typically regresses latency by 20-40%.

**Do NOT** confuse this with `mode="reduce-overhead"` (which uses Inductor's internal graph
capture and is broken on ROCm). The combination of `mode="default"` + manual graph capture
works because Inductor's internal capture is disabled while we capture the compiled output manually.

## Integration Pattern for Flow-Matching Models (e.g., OpenPI pi0)

Flow-matching models have a two-phase inference:
1. **Prefix computation** (once): image/language embedding + KV cache fill
2. **Denoise loop** (N steps): repeated expert forward passes

CUDAGraph should capture the ENTIRE `sample_actions` call (prefix + all denoise steps) as one
graph, since the benchmark measures the full call.

**Important**: The model should be compiled with `torch.compile(mode="default")` BEFORE graph
capture. This ensures kernel fusion is applied to both the prefix and denoise steps. Configure
Inductor to disable its internal graph capture (`triton.cudagraphs = False`) before compiling.

### Step 1: Convert while-loop to for-loop (prerequisite)

CUDAGraph requires static control flow. Replace:
```python
# BAD: dynamic loop condition — blocks CUDAGraph
while time >= -dt / 2:
    v_t = self.denoise_step(...)
    x_t = x_t + dt * v_t
    time += dt
```

With:
```python
# GOOD: fixed iteration count — CUDAGraph compatible
for step in range(num_steps):
    time = torch.tensor(1.0 + dt * step, dtype=torch.float32, device=device)
    v_t = self.denoise_step(...)
    x_t = x_t + dt * v_t
```

Do NOT add `@torch.compiler.disable` — the loop must remain inside the compiled region.

### Step 2: Add capture/replay directly in `sample_actions`

```python
class PI0Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing init ...
        self._graph = None
        self._static_observation = None
        self._static_output = None

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        if self._graph is not None:
            # REPLAY path: copy new inputs into static buffers, replay graph
            self._copy_observation(self._static_observation, observation)
            if noise is not None:
                self._static_noise.copy_(noise)
            self._graph.replay()
            return self._static_output.clone()

        # First call: run normally (triggers torch.compile JIT)
        output = self._sample_actions_impl(device, observation, noise, num_steps)

        # Second call: capture as CUDAGraph
        if self._graph is None:
            self._capture_graph(device, observation, noise, num_steps)

        return output

    def _capture_graph(self, device, observation, noise, num_steps):
        """Capture the full inference as a CUDAGraph."""
        # Store static observation buffers (same memory addresses for replay)
        self._static_observation = self._clone_observation(observation)
        if noise is None:
            noise_shape = (observation.state.shape[0], self.config.action_horizon,
                          self.config.action_dim)
            self._static_noise = torch.randn(noise_shape, device=device)
        else:
            self._static_noise = noise.clone()

        # Warmup (ensure all kernels are compiled)
        for _ in range(3):
            _ = self._sample_actions_impl(
                device, self._static_observation, self._static_noise, num_steps)
        torch.cuda.synchronize()

        # Capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._sample_actions_impl(
                device, self._static_observation, self._static_noise, num_steps)
        torch.cuda.synchronize()

    def _sample_actions_impl(self, device, observation, noise, num_steps):
        """The actual inference logic (prefix + denoise loop)."""
        # ... existing sample_actions body goes here ...

    @staticmethod
    def _clone_observation(obs):
        """Deep-clone observation tensors to create static buffers."""
        import dataclasses
        fields = {}
        for f in dataclasses.fields(obs):
            val = getattr(obs, f.name)
            if isinstance(val, torch.Tensor):
                fields[f.name] = val.clone()
            elif isinstance(val, dict):
                fields[f.name] = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                  for k, v in val.items()}
            else:
                fields[f.name] = val
        return type(obs)(**fields)

    @staticmethod
    def _copy_observation(dst, src):
        """Copy src observation data into dst's pre-allocated buffers."""
        import dataclasses
        for f in dataclasses.fields(src):
            src_val = getattr(src, f.name)
            dst_val = getattr(dst, f.name)
            if isinstance(src_val, torch.Tensor):
                dst_val.copy_(src_val)
            elif isinstance(src_val, dict):
                for k in src_val:
                    if isinstance(src_val[k], torch.Tensor):
                        dst_val[k].copy_(src_val[k])
```

### Step 3: Handle the Dynamo RNG patch (ROCm only)

ROCm forbids `torch.cuda.get_rng_state()` during graph capture. Apply BEFORE capture:

```python
import torch._dynamo.convert_frame as _convert_frame

def patch_dynamo_rng():
    if getattr(_convert_frame, "_rocm_patched", False):
        return
    _convert_frame._rocm_patched = True
    _orig = _convert_frame.preserve_global_state

    import functools
    def _skip_rng(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            rng = None
            if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
                try: rng = torch.cuda.get_rng_state()
                except Exception: pass
            try: return fn(*args, **kwargs)
            finally:
                if rng is not None:
                    try: torch.cuda.set_rng_state(rng)
                    except Exception: pass
        return wrapper
    _convert_frame.preserve_global_state = _skip_rng
```

**Note**: The module path is `torch._dynamo.convert_frame`, NOT `torch._dynamo.utils`. The
function is `preserve_global_state`. This was a common error in earlier runs.

## Correctness Verification After Capture

CUDAGraph output MUST match compiled output. If max diff > 0.01:

```python
# Run compiled (no graph)
ref_output = model._sample_actions_impl(device, observation, noise, num_steps)
# Run graph replay
model._graph.replay()
graph_output = model._static_output.clone()

diff = (ref_output - graph_output).abs()
print(f"Max diff: {diff.max().item():.6f}")
print(f"Mean diff: {diff.mean().item():.6f}")
assert diff.max().item() < 0.01, "CUDAGraph output diverges — investigate!"
```

Common causes of divergence:
- **RNG state**: Dynamo RNG patch not applied → different random noise
- **Uninitialized buffers**: Static output not cloned properly
- **In-place ops on external tensors**: Graph captures in-place ops on fixed memory addresses

## Rules During Graph Capture

| Forbidden | Why | Alternative |
|-----------|-----|-------------|
| `.item()` | Device-to-host sync | Use tensor directly |
| `torch.cuda.synchronize()` | Device-wide sync | `torch.cuda.current_stream().synchronize()` |
| `print(tensor)` | Forces sync | Remove or guard with `if not capturing` |
| Dynamic shapes | Graph assumes fixed shapes | Pad to max shape |
| `if tensor.shape[0] > N:` | Shape-dependent branch | Pre-compute or assert fixed |
| New allocations | Graph freezes memory layout | Pre-allocate all buffers |
