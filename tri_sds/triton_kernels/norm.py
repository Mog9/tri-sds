"""fused layer norm kernel for gpt-2.

input:  [batch, hidden]  (e.g. [B, 768])
weight: [hidden]
bias:   [hidden]
eps:    1e-5

fuses: mean + var computation + normalize in one kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, o_ptr,
    stride_xb, stride_xh,
    stride_ob, stride_oh,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    off = pid * stride_xb + tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + off, mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_zm = x - mean
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0)

    y = x_zm * rstd * w + b

    off_o = pid * stride_ob + tl.arange(0, BLOCK_N)
    tl.store(o_ptr + off_o, y, mask=tl.arange(0, BLOCK_N) < N)


def layer_norm(x, weight, bias, eps=1e-5):
    B, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (B,)
    _layer_norm_kernel[grid](
        x, weight, bias, o,
        x.stride(0), x.stride(1),
        o.stride(0), o.stride(1),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return o
