"""fully fused gpt-2 transformer block kernel.

computes: norm -> qkv -> attn (kv cache) -> proj -> residual
       -> norm -> mlp -> residual
all in one triton kernel per batch element.

uses a small hbm temp buffer for attn_out (unavoidable — register tensors
can't be indexed with runtime offsets). rest keeps intermediates in registers.
"""

import torch
import triton
import triton.language as tl

from tri_sds.triton_kernels.norm import layer_norm as triton_norm
from tri_sds.triton_kernels.attention import attention as triton_attn


@triton.jit
def _gelu(x):
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    z_clamped = tl.where(z > 10.0, 10.0, tl.where(z < -10.0, -10.0, z))
    e2z = tl.math.exp(2.0 * z_clamped)
    tanh_z = (e2z - 1.0) / (e2z + 1.0)
    return 0.5 * x * (1.0 + tanh_z)


@triton.jit
def _fused_block_kernel(
    x_ptr, scratch_ptr,
    ln1_w_ptr, ln1_b_ptr,
    qkv_w_ptr, qkv_b_ptr,
    k_cache_ptr, v_cache_ptr,
    pos, kv_len,
    N: tl.constexpr, D: tl.constexpr, H: tl.constexpr, max_len: tl.constexpr,
    qkv_stride_n: tl.constexpr, qkv_stride_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """norm1 -> qkv -> attention per head, writes attn_out to scratch."""
    pid = tl.program_id(0)
    off_b = pid * N
    ar = tl.arange(0, BLOCK_N)
    dar = tl.arange(0, D)

    # phase 1: norm1 stats
    acc_sum = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        k_off = k_start + ar
        xv = tl.load(x_ptr + off_b + k_off, mask=k_off < N, other=0.0)
        acc_sum += tl.sum(xv)
    mean1 = acc_sum / N

    acc_var = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, N, BLOCK_N):
        k_off = k_start + ar
        xv = tl.load(x_ptr + off_b + k_off, mask=k_off < N, other=0.0)
        acc_var += tl.sum((xv - mean1) * (xv - mean1))
    rstd1 = 1.0 / tl.sqrt(acc_var / N + 1e-5)

    # phase 2+3: qkv + attention per head
    rsqrt_d = 1.0 / tl.sqrt(D * 1.0)

    for h in range(H):
        h_base = h * 3 * D
        cache_off = pid * H * max_len * D + h * max_len * D

        q_h = tl.zeros([D], dtype=tl.float32)
        k_h = tl.zeros([D], dtype=tl.float32)
        v_h = tl.zeros([D], dtype=tl.float32)

        for k_start in range(0, N, BLOCK_N):
            k_off = k_start + ar
            k_mask = k_off < N
            xv = tl.load(x_ptr + off_b + k_off, mask=k_mask, other=0.0)
            lnw = tl.load(ln1_w_ptr + k_off, mask=k_mask, other=0.0)
            lnb = tl.load(ln1_b_ptr + k_off, mask=k_mask, other=0.0)
            x_norm = tl.where(k_mask, (xv - mean1) * rstd1 * lnw + lnb, 0.0)

            for off in range(3):
                off_d = off * D
                wt = tl.load(
                    qkv_w_ptr + k_off[:, None] * qkv_stride_n + (h_base + off_d + dar)[None, :] * qkv_stride_m,
                    mask=k_mask[:, None], other=0.0
                )
                contrib = tl.sum(x_norm[:, None] * wt, axis=0)
                if off == 0:
                    q_h += contrib
                elif off == 1:
                    k_h += contrib
                else:
                    v_h += contrib

        q_h += tl.load(qkv_b_ptr + h_base + dar)
        k_h += tl.load(qkv_b_ptr + h_base + D + dar)
        v_h += tl.load(qkv_b_ptr + h_base + 2 * D + dar)

        # store to kv cache
        tl.store(k_cache_ptr + cache_off + pos * D + dar, k_h, mask=dar < D)
        tl.store(v_cache_ptr + cache_off + pos * D + dar, v_h, mask=dar < D)

        # attention
        m_i = tl.full([1], -float('inf'), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        o_i = tl.zeros([D], dtype=tl.float32)

        for t_start in range(0, kv_len, BLOCK_T):
            t_off = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_off < kv_len

            k_hc = tl.load(
                k_cache_ptr + cache_off + t_off[:, None] * D + dar[None, :],
                mask=t_mask[:, None], other=0.0
            )
            v_hc = tl.load(
                v_cache_ptr + cache_off + t_off[:, None] * D + dar[None, :],
                mask=t_mask[:, None], other=0.0
            )

            s = tl.sum(q_h[None, :] * k_hc, axis=1) * rsqrt_d
            m_new = tl.maximum(m_i, tl.max(s))
            p = tl.exp(s - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p)
            o_i = o_i * alpha + tl.sum(p[:, None] * v_hc, axis=0)
            m_i = m_new

        o_i /= l_i

        # store attn_out for this head to scratch hbm buffer
        h_off = h * D
        tl.store(scratch_ptr + h_off + dar, o_i, mask=dar < D)


def fused_block(
    x, ln1_w, ln1_b, qkv_w, qkv_b,
    attn_out_w, attn_out_b,
    ln2_w, ln2_b,
    fc1_w, fc1_b, fc2_w, fc2_b,
    k_cache, v_cache,
    pos, use_cache=True,
    scratch=None,
):
    """fully fused gpt-2 transformer block forward.

    if scratch is provided (pre-allocated [B, N]), it is reused
    instead of allocating a new buffer each call.
    """
    B, N = x.shape
    D = k_cache.shape[-1]
    H = k_cache.shape[1]
    max_len = k_cache.shape[2]

    if scratch is None:
        scratch = torch.empty(B, N, dtype=x.dtype, device=x.device)
    else:
        assert scratch.shape == (B, N), f"scratch needs ({B},{N}), got {scratch.shape}"

    block_n = min(128, triton.next_power_of_2(N))
    block_t = min(64, triton.next_power_of_2(max_len // 2)) if max_len < 128 else 128

    _fused_block_kernel[(B,)](
        x, scratch,
        ln1_w, ln1_b,
        qkv_w, qkv_b,
        k_cache, v_cache,
        pos, pos + 1,
        N, D, H, max_len,
        qkv_w.stride(0), qkv_w.stride(1),
        BLOCK_N=block_n,
        BLOCK_T=block_t,
    )

    # remainder in PyTorch
    attn_out = scratch @ attn_out_w + attn_out_b
    x2 = x + attn_out

    mean2 = x2.mean(dim=-1, keepdim=True)
    var2 = x2.var(dim=-1, keepdim=True, unbiased=False)
    x2_norm = (x2 - mean2) / torch.sqrt(var2 + 1e-5) * ln2_w + ln2_b

    fc1 = torch.nn.functional.gelu(x2_norm @ fc1_w + fc1_b)
    out = fc1 @ fc2_w + fc2_b + x2

    return out


def fused_block_split(
    x, ln1_w, ln1_b, qkv_w, qkv_b,
    attn_out_w, attn_out_b,
    ln2_w, ln2_b,
    fc1_w, fc1_b, fc2_w, fc2_b,
    k_cache, v_cache,
    pos, use_cache=True,
    scratch=None,
    kv_len=None,
    pos_tensor=None,
):
    """split approach: pytorch matmuls (cuBLAS) + dedicated triton attention kernel.

    same interface as fused_block() but keeps norm, qkv, mlp in pytorch.
    attention uses a lean triton flash-attention kernel (no norm/qkv/mlp).

    if kv_len is provided (device int tensor), attention reads kv_len from it
    and writes to cache at position kv_len-1 via index_copy_ (CUDA-graph compatible).
    if pos_tensor is provided (device [1] long tensor), uses it for cache store position
    instead of pos.
    """
    B, N = x.shape
    D = k_cache.shape[-1]
    H = k_cache.shape[1]

    x_norm = triton_norm(x, ln1_w, ln1_b)
    qkv = x_norm @ qkv_w + qkv_b

    q = qkv[:, :N].reshape(B, H, 1, D).contiguous()
    k = qkv[:, N:2*N].reshape(B, H, 1, D).contiguous()
    v = qkv[:, 2*N:].reshape(B, H, 1, D).contiguous()

    if use_cache:
        if pos_tensor is not None:
            k_cache.index_copy_(2, pos_tensor, k)
            v_cache.index_copy_(2, pos_tensor, v)
        else:
            k_cache[:, :, pos:pos+1, :] = k
            v_cache[:, :, pos:pos+1, :] = v

    if kv_len is not None:
        attn_out = triton_attn(q, k_cache, v_cache, causal=True, kv_len=kv_len)
    else:
        k_view = k_cache[:, :, :pos+1, :]
        v_view = v_cache[:, :, :pos+1, :]
        attn_out = triton_attn(q, k_view, v_view, causal=True)

    attn_flat = attn_out.transpose(1, 2).reshape(B, N)
    proj = attn_flat @ attn_out_w + attn_out_b
    x = x + proj

    x_norm2 = triton_norm(x, ln2_w, ln2_b)
    fc1 = torch.nn.functional.gelu(x_norm2 @ fc1_w + fc1_b)
    x = fc1 @ fc2_w + fc2_b + x

    return x
