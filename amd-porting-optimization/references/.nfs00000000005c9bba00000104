# Triton Kernel Patterns for AMD GPU Optimization

Reference implementations for the most impactful Triton kernel fusions targeting AMD MI-series GPUs. These kernels eliminate kernel launch overhead and reduce memory bandwidth by performing multiple operations in a single GPU pass.

All kernels follow these conventions:
- Compute internally in `float32` for numerical stability
- Store results in `bfloat16` (the standard training/inference dtype)
- Use `BLOCK_SIZE = triton.next_power_of_2(N)` for efficient memory access
- Grid dimension = number of rows (each program instance handles one row)

---

## 1. RMSNorm (Highest Impact: ~3.4x speedup)

Replaces: `x.float().pow(2).mean(-1, keepdim=True)` -> `rsqrt` -> `mul(weight)` (5 kernel launches, 3 memory round-trips)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    y = x * rrms * w

    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_contiguous()
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rms_norm_fwd_kernel[(M,)](x_2d, weight, y, x_2d.stride(0), y.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(orig_shape)
```

**Eager fallback:**
```python
def rms_norm_eager(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight).to(x.dtype)
```

---

## 2. Fused GELU + Mul (~1.6x speedup)

Used in transformer MLP: `GELU(gate) * up` where input is `[*, 2*hidden]` (first half is gate, second half is up).

Replaces: slice + `F.gelu(approximate='tanh')` + slice + mul (4 ops, 3 memory passes)

```python
@triton.jit
def _gelu_tanh_and_mul_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,  # hidden_size (half of input last dim)
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    gate = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + N + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (gate + 0.044715 * gate * gate * gate)
    exp_2x = tl.math.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    gelu_gate = 0.5 * gate * (1.0 + tanh_val)

    y = gelu_gate * up
    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def gelu_tanh_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    orig_shape = x.shape
    N = x.shape[-1] // 2
    x_2d = x.view(-1, x.shape[-1])
    M = x_2d.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _gelu_tanh_and_mul_kernel[(M,)](x_2d, y, x_2d.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(orig_shape[:-1] + (N,))
```

---

## 3. Fused SiLU + Mul (~1.4x speedup)

Alternative activation fusion for models using SiLU instead of GELU.

```python
@triton.jit
def _silu_and_mul_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    gate = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + N + col_offsets, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    y = silu_gate * up

    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    orig_shape = x.shape
    N = x.shape[-1] // 2
    x_2d = x.view(-1, x.shape[-1])
    M = x_2d.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _silu_and_mul_kernel[(M,)](x_2d, y, x_2d.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(orig_shape[:-1] + (N,))
```

---

## 4. Fused Add + RMSNorm (~2.8x speedup)

Combines residual addition with RMSNorm in a single pass. Used at every transformer layer boundary.

Returns both the normalized output and the residual sum (needed by the next layer).

```python
@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, R_ptr, W_ptr, Y_ptr, RS_ptr,
    stride_x_row, stride_r_row, stride_y_row, stride_rs_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row_idx * stride_r_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    hidden = x + r
    tl.store(RS_ptr + row_idx * stride_rs_row + col_offsets, hidden.to(tl.bfloat16), mask=mask)

    variance = tl.sum(hidden * hidden, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    y = hidden * rrms * w
    tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, y.to(tl.bfloat16), mask=mask)


def fused_add_rms_norm_triton(x, residual, weight, eps=1e-6):
    assert x.is_contiguous() and residual.is_contiguous()
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    r_2d = residual.view(-1, residual.shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    rs = torch.empty_like(x_2d)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _fused_add_rms_norm_kernel[(M,)](
        x_2d, r_2d, weight, y, rs,
        x_2d.stride(0), r_2d.stride(0), y.stride(0), rs.stride(0),
        N, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.view(orig_shape), rs.view(orig_shape)
```

---

## Integration Pattern

Wrap all kernels with automatic backend selection:

```python
import os
import torch

TRITON_AVAILABLE = False
try:
    from .triton_ops import rms_norm_triton, gelu_tanh_and_mul_triton
    TRITON_AVAILABLE = True
except ImportError:
    pass

USE_OPTIMIZED_OPS = os.environ.get("USE_OPTIMIZED_OPS", "0") == "1"

def rms_norm(x, weight, eps=1e-6):
    if USE_OPTIMIZED_OPS and TRITON_AVAILABLE:
        return rms_norm_triton(x, weight, eps)
    # Eager fallback
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * weight).to(x.dtype)
```

This pattern ensures:
- Code works on any platform (NVIDIA, AMD, CPU)
- Optimizations are opt-in via environment variables
- No hard dependency on Triton or aiter
