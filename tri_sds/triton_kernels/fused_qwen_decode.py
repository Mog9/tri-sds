"""Fused Qwen3 decode kernels.

Replaces 10+ separate operations per layer with 3 fused kernel launches:
  1. fused_rmsnorm_qkv_rope:  RMSNorm -> QKV GEMV -> RoPE -> KV cache write
  2. fused_oproj_residual:    O-proj GEMV + residual add
  3. fused_gateup_silu_down:  GateUp GEMV + SiLU + Down GEMV + residual add

All matmuls use element-wise reduce over K to avoid poorly-optimized cuBLAS GEMV
and keep intermediates in registers.
"""

import torch
import triton
import triton.language as tl


# Kernel 1: RMSNorm + QKV GEMV + RoPE + KV cache write
# Grid (B, H_kv). Each program handles one KV head group (Q_PER_KV Q heads
# + 1 K head + 1 V head). Processes Q heads one at a time (not in a list)
# to avoid Triton list-of-tensor issues.

@triton.jit
def _rmsnorm_qkv_rope_kernel(
    x_ptr, norm1_w_ptr, qkv_w_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr,
    cos_ptr, sin_ptr,
    pos, N, q_dim, kv_dim, H_q, H_kv, D,
    stride_qkv_k, stride_qkv_out,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    eps,
    BLOCK_K: tl.constexpr, BLOCK_HALF: tl.constexpr,
    Q_PER_KV: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)

    ar_k = tl.arange(0, BLOCK_K)
    ar_h = tl.arange(0, BLOCK_HALF)

    # ---- Phase 1: RMSNorm statistics ----
    sq_sum = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + ar_k
        mask_k = offs_k < N
        x_tile = tl.load(x_ptr + pid_b * N + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        sq_sum += tl.sum(x_tile * x_tile)
    rstd = 1.0 / tl.sqrt(sq_sum / N + eps)

    # ---- Phase 2: QKV GEMV ----
    # Process each Q head individually (avoids list-of-tensors)
    k_acc1 = tl.zeros([BLOCK_HALF], dtype=tl.float32)
    k_acc2 = tl.zeros([BLOCK_HALF], dtype=tl.float32)
    v_acc1 = tl.zeros([BLOCK_HALF], dtype=tl.float32)
    v_acc2 = tl.zeros([BLOCK_HALF], dtype=tl.float32)

    q_base = pid_kv * Q_PER_KV

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + ar_k
        mask_k = offs_k < N
        x_tile = tl.load(x_ptr + pid_b * N + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w_tile = tl.load(norm1_w_ptr + offs_k, mask=mask_k, other=0.0)
        x_norm_tile = x_tile * rstd * w_tile

        k_col = q_dim + pid_kv * D
        w1 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (k_col + ar_h)[None, :] * stride_qkv_out,
                     mask=mask_k[:, None], other=0.0)
        w2 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (k_col + BLOCK_HALF + ar_h)[None, :] * stride_qkv_out,
                     mask=mask_k[:, None], other=0.0)
        k_acc1 += tl.sum(x_norm_tile[:, None] * w1, axis=0)
        k_acc2 += tl.sum(x_norm_tile[:, None] * w2, axis=0)

        v_col = q_dim + kv_dim + pid_kv * D
        w1 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (v_col + ar_h)[None, :] * stride_qkv_out,
                     mask=mask_k[:, None], other=0.0)
        w2 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (v_col + BLOCK_HALF + ar_h)[None, :] * stride_qkv_out,
                     mask=mask_k[:, None], other=0.0)
        v_acc1 += tl.sum(x_norm_tile[:, None] * w1, axis=0)
        v_acc2 += tl.sum(x_norm_tile[:, None] * w2, axis=0)

    # ---- Phase 3: RoPE + write K/V cache, Q output ----
    cos_val = tl.load(cos_ptr + pos * BLOCK_HALF + ar_h, mask=ar_h < BLOCK_HALF, other=0.0)
    sin_val = tl.load(sin_ptr + pos * BLOCK_HALF + ar_h, mask=ar_h < BLOCK_HALF, other=0.0)

    k_rope1 = k_acc1 * cos_val - k_acc2 * sin_val
    k_rope2 = k_acc2 * cos_val + k_acc1 * sin_val
    off_k = k_cache_ptr + pid_b * stride_kb + pid_kv * stride_kh + pos * stride_kt
    tl.store(off_k + ar_h * stride_kd, k_rope1, mask=ar_h < BLOCK_HALF)
    tl.store(off_k + (BLOCK_HALF + ar_h) * stride_kd, k_rope2, mask=ar_h < BLOCK_HALF)

    off_v = v_cache_ptr + pid_b * stride_vb + pid_kv * stride_vh + pos * stride_vt
    tl.store(off_v + ar_h * stride_vd, v_acc1, mask=ar_h < BLOCK_HALF)
    tl.store(off_v + (BLOCK_HALF + ar_h) * stride_vd, v_acc2, mask=ar_h < BLOCK_HALF)

    for qi in range(Q_PER_KV):
        q_head = q_base + qi
        q_acc1 = tl.zeros([BLOCK_HALF], dtype=tl.float32)
        q_acc2 = tl.zeros([BLOCK_HALF], dtype=tl.float32)

        for k_start in range(0, N, BLOCK_K):
            offs_k = k_start + ar_k
            mask_k = offs_k < N
            x_tile = tl.load(x_ptr + pid_b * N + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            w_tile = tl.load(norm1_w_ptr + offs_k, mask=mask_k, other=0.0)
            x_norm_tile = x_tile * rstd * w_tile

            col_base = q_head * D
            w1 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (col_base + ar_h)[None, :] * stride_qkv_out,
                         mask=mask_k[:, None], other=0.0)
            w2 = tl.load(qkv_w_ptr + offs_k[:, None] * stride_qkv_k + (col_base + BLOCK_HALF + ar_h)[None, :] * stride_qkv_out,
                         mask=mask_k[:, None], other=0.0)
            q_acc1 += tl.sum(x_norm_tile[:, None] * w1, axis=0)
            q_acc2 += tl.sum(x_norm_tile[:, None] * w2, axis=0)

        q1 = q_acc1 * cos_val - q_acc2 * sin_val
        q2 = q_acc2 * cos_val + q_acc1 * sin_val
        off_q = q_out_ptr + pid_b * stride_qb + q_head * stride_qh + 0 * stride_qt
        tl.store(off_q + ar_h * stride_qd, q1, mask=ar_h < BLOCK_HALF)
        tl.store(off_q + (BLOCK_HALF + ar_h) * stride_qd, q2, mask=ar_h < BLOCK_HALF)


def fused_rmsnorm_qkv_rope(x, norm1_w, qkv_w_t, q_out, k_cache, v_cache,
                            cos, sin, pos, N, q_dim, kv_dim, H_q, H_kv, D):
    B = x.shape[0]
    block_k = 128
    block_half = D // 2
    q_per_kv = H_q // H_kv

    grid = (B, H_kv)
    _rmsnorm_qkv_rope_kernel[grid](
        x, norm1_w, qkv_w_t,
        q_out, k_cache, v_cache,
        cos, sin,
        pos, N, q_dim, kv_dim, H_q, H_kv, D,
        qkv_w_t.stride(0), qkv_w_t.stride(1),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        1e-6,
        BLOCK_K=block_k, BLOCK_HALF=block_half,
        Q_PER_KV=q_per_kv,
        num_warps=4,
    )


# Kernel 2: O-proj GEMV + residual add
# Grid (N/BLOCK_N,). Each program computes one output tile [1xBLOCK_N].

@triton.jit
def _oproj_residual_kernel(
    attn_ptr, o_w_ptr, x_ptr, o_ptr,
    N, q_dim,
    stride_ow_k, stride_ow_out,
    stride_x, stride_o,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    ar_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_start in range(0, q_dim, BLOCK_K):
        offs_k = k_start + ar_k
        mask_k = offs_k < q_dim
        a_tile = tl.load(attn_ptr + pid_b * q_dim + offs_k, mask=mask_k, other=0.0)
        w_tile = tl.load(o_w_ptr + offs_k[:, None] * stride_ow_k + offs_n[None, :] * stride_ow_out,
                         mask=mask_k[:, None], other=0.0)
        acc += tl.sum(a_tile[:, None] * w_tile, axis=0)

    x_tile = tl.load(x_ptr + pid_b * stride_x + offs_n, mask=mask_n, other=0.0)
    y_tile = x_tile + acc
    tl.store(o_ptr + pid_b * stride_o + offs_n, y_tile, mask=mask_n)


def fused_oproj_residual(attn_flat, o_w_t, x):
    B, N = x.shape
    q_dim = attn_flat.shape[-1]
    block_n = min(128, triton.next_power_of_2(N))
    block_k = min(128, triton.next_power_of_2(q_dim))

    o = torch.empty_like(x)
    grid = (B, triton.cdiv(N, block_n))
    _oproj_residual_kernel[grid](
        attn_flat, o_w_t, x, o,
        N, q_dim,
        o_w_t.stride(0), o_w_t.stride(1),
        x.stride(0), o.stride(0),
        BLOCK_K=block_k, BLOCK_N=block_n,
        num_warps=4,
    )
    return o


# ---------------------------------------------------------------------------
# Kernel 3a: GateUp GEMV + SiLU -> hidden state
# ---------------------------------------------------------------------------
# Grid (INTERMEDIATE/BLOCK_GU,). Each program computes hidden[k] =
# silu(gate[k]) * up[k] where gate[k] = x_norm2 @ gate_up_w[:, k] and
# up[k] = x_norm2 @ gate_up_w[:, k + intermediate].

@triton.jit
def _gateup_silu_kernel(
    x_ptr, gate_up_w_ptr, hidden_ptr,
    N, intermediate_size,
    stride_gw_k, stride_gw_out,
    stride_x, stride_hidden,
    BLOCK_K: tl.constexpr, BLOCK_GU: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_GU + tl.arange(0, BLOCK_GU)
    mask_h = offs_h < intermediate_size
    ar_k = tl.arange(0, BLOCK_K)

    gate_cols = offs_h
    up_cols = intermediate_size + offs_h

    gate_acc = tl.zeros([BLOCK_GU], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_GU], dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + ar_k
        mask_k = offs_k < N
        x_tile = tl.load(x_ptr + pid_b * stride_x + offs_k, mask=mask_k, other=0.0)

        w_gate = tl.load(gate_up_w_ptr + offs_k[:, None] * stride_gw_k + gate_cols[None, :] * stride_gw_out,
                         mask=mask_k[:, None], other=0.0)
        gate_acc += tl.sum(x_tile[:, None] * w_gate, axis=0)

        w_up = tl.load(gate_up_w_ptr + offs_k[:, None] * stride_gw_k + up_cols[None, :] * stride_gw_out,
                       mask=mask_k[:, None], other=0.0)
        up_acc += tl.sum(x_tile[:, None] * w_up, axis=0)

    hidden = tl.sigmoid(gate_acc) * gate_acc * up_acc
    tl.store(hidden_ptr + pid_b * stride_hidden + offs_h, hidden, mask=mask_h)


def fused_gateup_silu(x_norm2, gate_up_w_t):
    B, N = x_norm2.shape
    intermediate = gate_up_w_t.shape[1] // 2
    block_gu = min(128, triton.next_power_of_2(intermediate))
    block_k = min(128, triton.next_power_of_2(N))

    hidden = torch.empty(B, intermediate, device=x_norm2.device, dtype=x_norm2.dtype)
    grid = (B, triton.cdiv(intermediate, block_gu))
    _gateup_silu_kernel[grid](
        x_norm2, gate_up_w_t, hidden,
        N, intermediate,
        gate_up_w_t.stride(0), gate_up_w_t.stride(1),
        x_norm2.stride(0), hidden.stride(0),
        BLOCK_K=block_k, BLOCK_GU=block_gu,
        num_warps=4,
    )
    return hidden


# Kernel 3b: Down GEMV + residual add
# Grid (N/BLOCK_N,). Each program computes one output tile:
# mlp_out = hidden @ down_w + x_residual.

@triton.jit
def _down_residual_kernel(
    hidden_ptr, down_w_ptr, x_ptr, o_ptr,
    N, intermediate_size,
    stride_dw_k, stride_dw_out,
    stride_hidden, stride_x, stride_o,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    ar_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_start in range(0, intermediate_size, BLOCK_K):
        offs_k = k_start + ar_k
        mask_k = offs_k < intermediate_size
        h_tile = tl.load(hidden_ptr + pid_b * stride_hidden + offs_k, mask=mask_k, other=0.0)
        w_tile = tl.load(down_w_ptr + offs_k[:, None] * stride_dw_k + offs_n[None, :] * stride_dw_out,
                         mask=mask_k[:, None], other=0.0)
        acc += tl.sum(h_tile[:, None] * w_tile, axis=0)

    x_tile = tl.load(x_ptr + pid_b * stride_x + offs_n, mask=mask_n, other=0.0)
    y_tile = x_tile + acc
    tl.store(o_ptr + pid_b * stride_o + offs_n, y_tile, mask=mask_n)


def fused_down_residual(hidden, down_w_t, x_residual):
    B, N = x_residual.shape
    intermediate = hidden.shape[-1]
    block_n = min(128, triton.next_power_of_2(N))
    block_k = min(128, triton.next_power_of_2(intermediate))

    o = torch.empty_like(x_residual)
    grid = (B, triton.cdiv(N, block_n))
    _down_residual_kernel[grid](
        hidden, down_w_t, x_residual, o,
        N, intermediate,
        down_w_t.stride(0), down_w_t.stride(1),
        hidden.stride(0), x_residual.stride(0), o.stride(0),
        BLOCK_K=block_k, BLOCK_N=block_n,
        num_warps=4,
    )
    return o
