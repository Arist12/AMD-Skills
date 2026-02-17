# Triton Kernels for AMD GPUs

## Table of Contents
- [When to Use Triton on AMD](#when-to-use-triton-on-amd)
- [AMD-Specific Tuning](#amd-specific-tuning)
- [RMSNorm Kernel](#rmsnorm-kernel)
- [Fused GELU(tanh) + Mul](#fused-gelutanh--mul)
- [Fused SiLU + Mul](#fused-silu--mul)
- [Fused Add + RMSNorm](#fused-add--rmsnorm)
- [Numerical Stability on ROCm](#numerical-stability-on-rocm)

## When to Use Triton on AMD

**Use Triton for**: Elementwise/fused ops (normalization, activations, residuals)
**Do NOT use Triton for**: GEMMs (rocBLAS is 35-55% faster)

Check if `aiter` CK kernels are available first - they can be 3x faster than Triton for some activations:

```python
AITER_AVAILABLE = False
try:
    import aiter
    AITER_AVAILABLE = True
except ImportError:
    pass

def optimized_gelu_and_mul(x):
    if AITER_AVAILABLE:
        return torch.ops.aiter.gelu_tanh_and_mul(x)
    return triton_gelu_and_mul(x)  # Triton fallback
```

## AMD-Specific Tuning

AMD GPUs use **64-wide wavefronts** (vs NVIDIA's 32-wide warps). Key differences:

```python
@triton.jit
def kernel(...):
    # Block size must be power of 2 and >= feature dimension
    BLOCK_SIZE: tl.constexpr  # e.g., 2048, 4096

# AMD-friendly configs:
# - num_warps: 8 or 16 (more warps to fill CUs)
# - BLOCK_SIZE: >= N, power of 2
# - Use tl.float32 for accumulation (BF16 precision issues on ROCm)
```

### Coordinate Descent Tuning

Let Inductor find optimal configs automatically:

```python
inductor_config.coordinate_descent_tuning = True  # Explores block_size, num_warps
```

Or manually specify autotune configs:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    ...
```

## RMSNorm Kernel

~4x faster than eager `nn.LayerNorm` / manual RMSNorm on AMD.

```python
@triton.jit
def rms_norm_kernel(
    X,       # Input pointer [M, N]
    Y,       # Output pointer [M, N]
    W,       # Weight pointer [N]
    stride,  # Row stride
    N,       # Feature dimension
    eps,     # Epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    mean_sq = tl.sum(x * x, axis=0) / N
    rms = tl.rsqrt(mean_sq + eps)

    # Normalize and scale
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (x * rms * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride + cols, y, mask=mask)


def triton_rms_norm(x, weight, eps=1e-6):
    M, N = x.shape[-2], x.shape[-1]
    x_2d = x.reshape(-1, N)
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm_kernel[(x_2d.shape[0],)](
        x_2d, y, weight, x_2d.stride(0), N, eps, BLOCK_SIZE
    )
    return y.reshape(x.shape)
```

## Fused GELU(tanh) + Mul

~2.5x faster than separate `gelu(gate) * up`. Used in GeGLU MLPs.

```python
@triton.jit
def gelu_tanh_and_mul_kernel(
    X,       # Input pointer [M, 2*N] (gate concat up)
    Y,       # Output pointer [M, N]
    M,       # Batch dimension
    N,       # Feature dimension (half of input last dim)
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load gate and up from concatenated input
    gate = tl.load(X + row * stride_in + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_in + N + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU(tanh) approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Use stable tanh: clamp argument to [-10, 10] to avoid NaN on ROCm
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (gate + 0.044715 * gate * gate * gate)
    inner = tl.minimum(tl.maximum(inner, -10.0), 10.0)  # CLAMP for ROCm stability
    tanh_val = tl.math.tanh(inner)
    gelu = 0.5 * gate * (1.0 + tanh_val)

    # Fused multiply
    y = (gelu * up).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_out + cols, y, mask=mask)


def triton_gelu_and_mul(x):
    """x shape: [..., 2*N]. Returns [..., N]."""
    *batch, two_n = x.shape
    n = two_n // 2
    x_2d = x.reshape(-1, two_n)
    m = x_2d.shape[0]
    y = torch.empty(m, n, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(n)
    gelu_tanh_and_mul_kernel[(m,)](
        x_2d, y, m, n, x_2d.stride(0), y.stride(0), BLOCK_SIZE
    )
    return y.reshape(*batch, n)
```

## Fused SiLU + Mul

~1.4x faster. Used in SwiGLU MLPs.

```python
@triton.jit
def silu_and_mul_kernel(
    X, Y, M, N, stride_in, stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    gate = tl.load(X + row * stride_in + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * stride_in + N + cols, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu = gate * tl.sigmoid(gate)

    y = (silu * up).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_out + cols, y, mask=mask)
```

## Fused Add + RMSNorm

~2.8x faster. Combines residual addition with normalization. Returns both normalized output and the residual sum (for the next layer's residual connection).

```python
@triton.jit
def add_rms_norm_kernel(
    X,         # Input pointer
    Residual,  # Residual pointer (in/out)
    Y,         # Normalized output pointer
    W,         # Weight pointer
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # Fused add
    hidden = x + res

    # Store updated residual for next layer
    tl.store(Residual + row * stride + cols, hidden.to(Residual.dtype.element_ty), mask=mask)

    # RMSNorm
    mean_sq = tl.sum(hidden * hidden, axis=0) / N
    rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    y = (hidden * rms * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride + cols, y, mask=mask)
```

## Numerical Stability on ROCm

### Clamped Exponential for tanh

ROCm's Triton `tl.math.tanh` can produce NaN for large inputs. Always clamp:

```python
# BAD: Can produce NaN on ROCm Triton
tanh_val = tl.math.tanh(x)

# GOOD: Clamp to safe range
x_clamped = tl.minimum(tl.maximum(x, -10.0), 10.0)
tanh_val = tl.math.tanh(x_clamped)
```

### Float32 Accumulation

Always accumulate in float32, even with BF16 inputs:

```python
# GOOD: Cast to float32 for computation
x = tl.load(ptr, mask=mask).to(tl.float32)
result = compute(x)
output = result.to(tl.bfloat16)  # Cast back for store

# BAD: Accumulate in BF16
x = tl.load(ptr, mask=mask)  # BF16
result = tl.sum(x * x)  # BF16 accumulation - precision loss
```

### Epsilon Handling

Use eps in rsqrt, not sqrt, to avoid division by zero:

```python
# GOOD
rms = tl.rsqrt(mean_sq + eps)

# BAD (potential div-by-zero if mean_sq + eps underflows)
rms = 1.0 / tl.sqrt(mean_sq + eps)
```
