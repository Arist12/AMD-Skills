# Library & Model Adaptation

Patterns for replacing NVIDIA libraries with AMD equivalents, integrating aiter for attention and
GEMM, adapting model code, and monkey-patching transformers.

## Table of Contents

- [Library Mapping Reference](#library-mapping-reference)
- [aiter Integration](#aiter-integration)
- [Attention Routing](#attention-routing)
- [GEMM Dispatching](#gemm-dispatching)
- [Activation Kernels](#activation-kernels)
- [Weight Preshuffling](#weight-preshuffling)
- [Projection Fusion](#projection-fusion)
- [Transformers Monkey-Patching](#transformers-monkey-patching)
- [Three-Tier Fallback Architecture](#three-tier-fallback-architecture)

## Library Mapping Reference

### Drop-in replacements (HIPIFY handles automatically)

| NVIDIA | AMD | Header change | Link change |
|---|---|---|---|
| cuBLAS | hipBLAS + rocBLAS | `cublas_v2.h` → `hipblas/hipblas.h` | `-lcublas` → `-lhipblas -lrocblas` |
| cuBLASLt | hipBLASLt | `cublasLt.h` → `hipblaslt/hipblaslt.h` | `-lcublasLt` → `-lhipblaslt` |
| cuSPARSE | hipSPARSE + rocSPARSE | `cusparse.h` → `hipsparse/hipsparse.h` | `-lcusparse` → `-lhipsparse` |
| cuFFT | hipFFT + rocFFT | `cufft.h` → `hipfft/hipfft.h` | `-lcufft` → `-lhipfft` |
| cuRAND | hipRAND + rocRAND | `curand.h` → `hiprand/hiprand.h` | `-lcurand` → `-lhiprand` |
| NCCL | RCCL | `nccl.h` → `rccl/rccl.h` | `-lnccl` → `-lrccl` |
| Thrust | rocThrust | Same headers | Link rocThrust |

### API-different replacements (manual adaptation required)

| NVIDIA | AMD | Notes |
|---|---|---|
| cuDNN | MIOpen | Different API; convolution, pooling, RNN similar but not identical |
| CUTLASS | Composable Kernel (CK) | Completely different API; manual rewrite |
| flash-attn | aiter | Different Python API; see below |
| TensorRT | MIGraphX | Different optimization pipeline |
| cuDNN attention | aiter mha_fwd | torch.ops.aiter.mha_fwd |

### cuDNN to MIOpen key differences

```cpp
// cuDNN: descriptor-based API
cudnnConvolutionForward(handle, &alpha,
    inputDesc, input, filterDesc, filter,
    convDesc, algo, workspace, workspaceSize,
    &beta, outputDesc, output);

// MIOpen: similar descriptor pattern but different names
miopenConvolutionForward(handle,
    &alpha, inputDesc, input, filterDesc, filter,
    convDesc, algo, workspace, workspaceSize,
    &beta, outputDesc, output);
```

Key differences:
- `miopenFindConvolutionForwardAlgorithm` returns algorithms differently
- MIOpen lacks some cuDNN features (e.g., certain fused operations)
- Workspace size queries differ

## aiter Integration

aiter is AMD's optimized inference library providing flash-attention and tuned GEMM operations.

### Installation check

```python
AITER_AVAILABLE = False
AITER_GEMM_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
    try:
        from aiter.tuned_gemm import gemm_a16w16
        AITER_GEMM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass
```

### Feature flags pattern

```python
import os

def get_use_aiter_attention():
    if not AITER_AVAILABLE:
        return False
    return os.environ.get("USE_AITER_ATTENTION", "0") == "1"

def get_use_aiter_gemm():
    if not AITER_GEMM_AVAILABLE:
        return False
    return os.environ.get("USE_AITER_GEMM", "0") == "1"
```

## Attention Routing

### aiter flash attention (torch.ops.aiter.mha_fwd)

```python
def aiter_attention_forward(query, key, value, scaling, is_causal=False, dropout=0.0):
    """
    Dispatch to aiter's flash-attention kernel.

    Args:
        query: [batch, seq_q, num_heads, head_dim]
        key: [batch, seq_k, num_kv_heads, head_dim]
        value: [batch, seq_k, num_kv_heads, head_dim]
        scaling: float, typically 1/sqrt(head_dim)
        is_causal: whether to apply causal mask
        dropout: dropout probability (0.0 for inference)

    Returns:
        output: [batch, seq_q, num_heads, head_dim]
    """
    op = torch.ops.aiter.mha_fwd.default

    # aiter API evolves; check for sink_size parameter
    schema = str(getattr(op, "_schema", ""))
    has_sink_size = "sink_size" in schema

    if has_sink_size:
        outs = op(
            query, key, value,
            dropout,           # dropout_p
            float(scaling),    # softmax_scale
            bool(is_causal),   # is_causal
            -1,                # window_size_left (disabled)
            -1,                # window_size_right (disabled)
            0,                 # sink_size (disabled)
            False,             # return_softmax_lse
            False,             # return_dropout_randval
        )
    else:
        outs = op(
            query, key, value,
            dropout,
            float(scaling),
            bool(is_causal),
            -1, -1,
            False, False,
        )

    return outs[0]  # [out, lse, p, rng_state]
```

### MQA/GQA native support

aiter supports multi-query and grouped-query attention natively. K/V can have fewer heads than Q
without needing `repeat_kv` expansion:

```python
# DON'T expand K/V heads to match Q heads -- aiter handles it natively
# This avoids expand-based overlapping views that cause issues with torch.compile

# Bad: repeat_kv forces expand + potential .contiguous() copy
# key_states = repeat_kv(key_states, num_key_value_groups)

# Good: pass K/V with original head count
attn_output = aiter_attention_forward(
    query,       # [B, seq_q, num_heads, head_dim]
    key_states,  # [B, seq_k, num_kv_heads, head_dim]  (fewer heads)
    value_states,
    scaling=self.scaling,
    is_causal=is_causal,
)
```

### SDPA fast-path for KV-cache cross-attention

When q_len != k_len (KV-cache decode), PyTorch SDPA can be faster:

```python
def sdpa_attention_forward(query, key, value, is_causal=True, dropout=0.0):
    """Use PyTorch SDPA for KV-cache decode where q_len != k_len."""
    if query.shape[2] != key.shape[2] and dropout == 0.0:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    return None  # signal caller to use another path
```

### Complete routing function

```python
def attention_forward(query, key, value, scaling, is_causal=False, dropout=0.0):
    # Tier 1: aiter flash attention
    if get_use_aiter_attention():
        return aiter_attention_forward(query, key, value, scaling, is_causal, dropout)

    # Tier 2: PyTorch SDPA
    if hasattr(F, "scaled_dot_product_attention"):
        # Reshape for SDPA: [B, heads, seq, dim]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout)
        return out.transpose(1, 2)

    # Tier 3: Eager (manual matmul + softmax)
    return eager_attention(query, key, value, scaling, is_causal, dropout)
```

## GEMM Dispatching

### aiter tuned GEMM

```python
from aiter.tuned_gemm import gemm_a16w16

def aiter_linear(x, weight, bias=None):
    """Drop-in replacement for F.linear using aiter tuned GEMM."""
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        return F.linear(x, weight, bias)
    return gemm_a16w16(x, weight, bias=bias, otype=x.dtype)
```

### Global nn.Linear patching

```python
def patch_linear_forward():
    """Monkey-patch nn.Linear to use aiter GEMM globally."""
    original_forward = torch.nn.Linear.forward

    def patched_forward(self, x):
        if get_use_aiter_gemm():
            return aiter_linear(x, self.weight, self.bias)
        return original_forward(self, x)

    torch.nn.Linear.forward = patched_forward
```

### Per-shape GEMM tuning via CSV

aiter supports per-shape kernel selection via CSV configs:

```csv
cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw
256,532,2048,2048,False,torch.bfloat16,torch.bfloat16,False,False,asm,9,1,16.7,...,266.78,761.98
```

Load custom configs:

```python
import os
import pathlib

def extend_aiter_gemm_configs(custom_csv_path):
    """Add custom per-shape GEMM tuning to aiter."""
    import importlib
    spec = importlib.util.find_spec("aiter")
    aiter_pkg = pathlib.Path(list(spec.submodule_search_locations)[0])
    default_cfg = aiter_pkg / "configs" / "bf16_tuned_gemm.csv"

    paths = []
    if default_cfg.exists():
        paths.append(str(default_cfg))
    paths.append(str(custom_csv_path))  # custom shapes (highest priority)

    os.environ["AITER_CONFIG_GEMM_BF16"] = os.pathsep.join(paths)
```

## Activation Kernels

### CK-accelerated fused activations

aiter provides CK-based fused activation kernels (faster than Triton on AMD):

```python
def gelu_tanh_and_mul(x):
    """Fused GELU-tanh + mul activation using aiter CK kernel."""
    if AITER_AVAILABLE and os.environ.get("USE_OPTIMIZED_OPS", "0") == "1":
        orig_shape = x.shape
        hidden_size = orig_shape[-1] // 2
        x_2d = x.reshape(-1, orig_shape[-1])
        out = torch.empty(x_2d.shape[0], hidden_size, dtype=x.dtype, device=x.device)
        torch.ops.aiter.gelu_tanh_and_mul(out, x_2d)
        return out.reshape(*orig_shape[:-1], hidden_size)

    # Fallback: manual GELU-tanh + mul
    gate, up = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up
```

### SiLU and mul

```python
def silu_and_mul(x):
    """Fused SiLU + mul activation."""
    if AITER_AVAILABLE and os.environ.get("USE_OPTIMIZED_OPS", "0") == "1":
        orig_shape = x.shape
        hidden_size = orig_shape[-1] // 2
        x_2d = x.reshape(-1, orig_shape[-1])
        out = torch.empty(x_2d.shape[0], hidden_size, dtype=x.dtype, device=x.device)
        torch.ops.aiter.silu_and_mul(out, x_2d)
        return out.reshape(*orig_shape[:-1], hidden_size)

    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up
```

## Weight Preshuffling

aiter asm kernels benefit from pre-shuffled weight layouts:

```python
def preshuffle_linear_weights(model, layout=(16, 16), require_multiple=256):
    """Pre-shuffle nn.Linear weights for aiter asm kernel paths."""
    from aiter.ops.shuffle import shuffle_weight

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight
        if w.dtype != torch.bfloat16 or w.ndim != 2:
            continue
        n, k = w.shape
        if n % require_multiple != 0 or k % require_multiple != 0:
            continue
        w_shuf = shuffle_weight(w, layout=layout)
        module._preshuffled_weight = w_shuf
```

### M-threshold routing for preshuffled weights

Small-M GEMMs (decode phase) can regress with asm kernels. Gate by M-threshold:

```python
def linear_with_preshuffle(module, x):
    """Route to preshuffled weight if M exceeds threshold."""
    m_thresh = int(os.environ.get("PRESHUFFLE_M_THRESH", "128"))
    w = module.weight
    w_shuf = getattr(module, "_preshuffled_weight", None)

    if w_shuf is not None and m_thresh >= 0:
        m = int(x.numel() // x.shape[-1])
        if m_thresh == 0 or (m and m >= m_thresh):
            w = w_shuf

    return aiter_linear(x, w, module.bias)
```

## Projection Fusion

Fuse multiple linear projections into one GEMM call.

### Gate + Up fusion (MLP)

```python
def fuse_gate_up_projections(mlp_module):
    """Fuse gate_proj + up_proj into a single weight matrix."""
    if not hasattr(mlp_module, "gate_proj") or not hasattr(mlp_module, "up_proj"):
        return
    fused = torch.cat([mlp_module.gate_proj.weight, mlp_module.up_proj.weight], dim=0)
    mlp_module.register_buffer("_fused_gate_up_weight", fused)
    mlp_module._use_fused = True

# In forward:
def fused_mlp_forward(self, x):
    if getattr(self, "_use_fused", False):
        gate_up = F.linear(x, self._fused_gate_up_weight)
        return gelu_tanh_and_mul(gate_up)
    else:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return F.gelu(gate, approximate="tanh") * up
```

### QKV fusion (attention)

```python
def fuse_qkv_projections(attn_module):
    """Fuse q_proj + k_proj + v_proj into a single weight matrix."""
    fused_w = torch.cat([
        attn_module.q_proj.weight.data,
        attn_module.k_proj.weight.data,
        attn_module.v_proj.weight.data,
    ], dim=0)
    attn_module.register_buffer("_fused_qkv_weight", fused_w)

    if getattr(attn_module.q_proj, "bias", None) is not None:
        fused_b = torch.cat([
            attn_module.q_proj.bias.data,
            attn_module.k_proj.bias.data,
            attn_module.v_proj.bias.data,
        ], dim=0)
        attn_module.register_buffer("_fused_qkv_bias", fused_b)

    attn_module._use_fused_qkv = True
```

## Transformers Monkey-Patching

For HuggingFace transformers models, patch at runtime to inject AMD-optimized code.

### File overlay pattern

Copy modified modeling files into transformers site-packages:

```python
import pathlib
import shutil

def patch_transformers_models(replacements_dir):
    """
    Overlay modified model files into transformers site-packages.

    Args:
        replacements_dir: Path to directory containing modified model files,
            structured as models/{model_name}/modeling_{model_name}.py
    """
    import transformers
    dest = pathlib.Path(transformers.__file__).resolve().parent / "models"
    src = pathlib.Path(replacements_dir)

    for child in src.iterdir():
        if child.is_dir():
            shutil.copytree(child, dest / child.name, dirs_exist_ok=True)
    print(f"Patched transformers models from {src}")
```

### Version compatibility check

```python
def check_transformers_version(min_major=4, min_minor=53):
    """Check transformers version is compatible with patches."""
    import transformers
    parts = transformers.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    return major > min_major or (major == min_major and minor >= min_minor)
```

### Graceful import stubs

In modified model files, always provide fallback stubs:

```python
try:
    from my_amd_ops import optimized_gemm, optimized_attention
except ImportError:
    def optimized_gemm(x, weight, bias=None):
        return F.linear(x, weight, bias)
    def optimized_attention(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
```

## Three-Tier Fallback Architecture

Every AMD-optimized operation must have working fallbacks:

```
Tier 1: AMD-optimized (aiter CK kernels, asm GEMM)
  ↓ not available or disabled
Tier 2: PyTorch built-in (SDPA, F.linear with rocBLAS)
  ↓ not available
Tier 3: Pure eager PyTorch (manual matmul + softmax, F.linear)
```

### Pattern

```python
def compute_attention(q, k, v, **kwargs):
    # Tier 1
    if get_use_aiter_attention():
        try:
            return aiter_attention(q, k, v, **kwargs)
        except Exception:
            pass  # fall through

    # Tier 2
    try:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        ).transpose(1, 2)
    except Exception:
        pass  # fall through

    # Tier 3
    return eager_attention(q, k, v, **kwargs)
```

### Testing fallback paths

Test each tier independently by disabling higher tiers:

```bash
# Test Tier 1 (aiter)
USE_AITER_ATTENTION=1 python test.py

# Test Tier 2 (SDPA)
USE_AITER_ATTENTION=0 python test.py

# Test Tier 3 (eager)
USE_AITER_ATTENTION=0 FORCE_EAGER_ATTENTION=1 python test.py
```
