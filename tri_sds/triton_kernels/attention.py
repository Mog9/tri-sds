"""triton attention kernel for multi-head self-attention.

supports prefill (T_q == T_kv) and decode (T_q < T_kv).
decode path uses grouped heads for fewer kernel launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    T_kv, D, softmax_scale,
    BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_gh = tl.program_id(1)
    h_base = pid_gh * GROUP_H

    for gi in range(GROUP_H):
        h = h_base + gi

        off_q = q_ptr + pid_b * stride_qb + h * stride_qh
        q = tl.load(off_q + tl.arange(0, BLOCK_D) * stride_qd, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        off_kb = k_ptr + pid_b * stride_kb + h * stride_kh
        off_vb = v_ptr + pid_b * stride_vb + h * stride_vh

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        m_i = tl.full([1], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)

        for start_kv in range(0, T_kv, BLOCK_KV):
            off_k = off_kb + start_kv * stride_kt
            off_v = off_vb + start_kv * stride_vt

            kv_range = start_kv + tl.arange(0, BLOCK_KV)
            kv_mask = kv_range < T_kv

            k = tl.load(off_k + tl.arange(0, BLOCK_D)[:, None] * stride_kd + tl.arange(0, BLOCK_KV)[None, :] * stride_kt,
                        mask=(tl.arange(0, BLOCK_D)[:, None] < D) & kv_mask[None, :], other=0.0)

            scores = tl.sum(q[:, None] * k, axis=0) * softmax_scale
            scores = tl.where(kv_mask, scores, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_ij)
            l_ij = tl.sum(p, axis=0)

            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha + tl.sum(p[:, None] * tl.load(
                off_v + tl.arange(0, BLOCK_D)[None, :] * stride_vd + tl.arange(0, BLOCK_KV)[:, None] * stride_vt,
                mask=(tl.arange(0, BLOCK_D)[None, :] < D) & kv_mask[:, None], other=0.0
            ), axis=0)

            m_i = m_ij
            l_i = l_i * alpha + l_ij

        acc = acc / l_i

        off_o = o_ptr + pid_b * stride_ob + h * stride_oh
        tl.store(off_o + tl.arange(0, BLOCK_D) * stride_od, acc, mask=tl.arange(0, BLOCK_D) < D)


@triton.jit
def _decode_attn_kernel_v2(
    q_ptr, k_ptr, v_ptr, o_ptr, kv_len_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    D, softmax_scale,
    BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr, GROUP_H: tl.constexpr,
):
    T_kv = tl.load(kv_len_ptr)
    pid_b = tl.program_id(0)
    pid_gh = tl.program_id(1)
    h_base = pid_gh * GROUP_H

    for gi in range(GROUP_H):
        h = h_base + gi

        off_q = q_ptr + pid_b * stride_qb + h * stride_qh
        q = tl.load(off_q + tl.arange(0, BLOCK_D) * stride_qd, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        off_kb = k_ptr + pid_b * stride_kb + h * stride_kh
        off_vb = v_ptr + pid_b * stride_vb + h * stride_vh

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        m_i = tl.full([1], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)

        for start_kv in range(0, T_kv, BLOCK_KV):
            off_k = off_kb + start_kv * stride_kt
            off_v = off_vb + start_kv * stride_vt

            kv_range = start_kv + tl.arange(0, BLOCK_KV)
            kv_mask = kv_range < T_kv

            k = tl.load(off_k + tl.arange(0, BLOCK_D)[:, None] * stride_kd + tl.arange(0, BLOCK_KV)[None, :] * stride_kt,
                        mask=(tl.arange(0, BLOCK_D)[:, None] < D) & kv_mask[None, :], other=0.0)

            scores = tl.sum(q[:, None] * k, axis=0) * softmax_scale
            scores = tl.where(kv_mask, scores, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_ij)
            l_ij = tl.sum(p, axis=0)

            alpha = tl.exp(m_i - m_ij)
            acc = acc * alpha + tl.sum(p[:, None] * tl.load(
                off_v + tl.arange(0, BLOCK_D)[None, :] * stride_vd + tl.arange(0, BLOCK_KV)[:, None] * stride_vt,
                mask=(tl.arange(0, BLOCK_D)[None, :] < D) & kv_mask[:, None], other=0.0
            ), axis=0)

            m_i = m_ij
            l_i = l_i * alpha + l_ij

        acc = acc / l_i

        off_o = o_ptr + pid_b * stride_ob + h * stride_oh
        tl.store(off_o + tl.arange(0, BLOCK_D) * stride_od, acc, mask=tl.arange(0, BLOCK_D) < D)


@triton.jit
def _prefill_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    T_kv, D, softmax_scale,
    BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    off_q = q_ptr + pid_b * stride_qb + pid_h * stride_qh + pid_t * stride_qt
    q = tl.load(off_q + tl.arange(0, BLOCK_D) * stride_qd, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

    off_kb = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    off_vb = v_ptr + pid_b * stride_vb + pid_h * stride_vh

    hi = pid_t + 1 if IS_CAUSAL else T_kv

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)

    for start_kv in range(0, hi, BLOCK_KV):
        off_k = off_kb + start_kv * stride_kt
        off_v = off_vb + start_kv * stride_vt

        kv_range = start_kv + tl.arange(0, BLOCK_KV)
        kv_mask = kv_range < hi

        k = tl.load(off_k + tl.arange(0, BLOCK_D)[:, None] * stride_kd + tl.arange(0, BLOCK_KV)[None, :] * stride_kt,
                    mask=(tl.arange(0, BLOCK_D)[:, None] < D) & kv_mask[None, :], other=0.0)

        scores = tl.sum(q[:, None] * k, axis=0) * softmax_scale
        scores = tl.where(kv_mask, scores, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_ij)
        l_ij = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha + tl.sum(p[:, None] * tl.load(
            off_v + tl.arange(0, BLOCK_D)[None, :] * stride_vd + tl.arange(0, BLOCK_KV)[:, None] * stride_vt,
            mask=(tl.arange(0, BLOCK_D)[None, :] < D) & kv_mask[:, None], other=0.0
        ), axis=0)

        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc = acc / l_i

    off_o = o_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_t * stride_ot
    tl.store(off_o + tl.arange(0, BLOCK_D) * stride_od, acc, mask=tl.arange(0, BLOCK_D) < D)


def attention(q, k, v, causal=True, kv_len=None):
    B, H, T_q, D = q.shape
    T_kv = k.shape[2]
    max_block_kv = 256 if T_q == 1 else 128
    block_kv = min(max_block_kv, triton.next_power_of_2(T_kv))
    block_d = triton.next_power_of_2(D)
    o = torch.empty_like(q)

    if T_q == 1:
        if kv_len is not None:
            group_h = min(H, 4)
            _decode_attn_kernel_v2[(B, H // group_h)](
                q, k, v, o, kv_len,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                D, D ** -0.5,
                BLOCK_KV=block_kv, BLOCK_D=block_d, GROUP_H=group_h,
                num_warps=8,
            )
        else:
            group_h = min(H, 4)
            _decode_attn_kernel[(B, H // group_h)](
                q, k, v, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                T_kv, D, D ** -0.5,
                BLOCK_KV=block_kv, BLOCK_D=block_d, GROUP_H=group_h,
                num_warps=8,
            )
    else:
        _prefill_attn_kernel[(B, H, T_q)](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            T_kv, D, D ** -0.5,
            BLOCK_KV=block_kv, BLOCK_D=block_d, IS_CAUSAL=causal,
        )

    return o


@triton.jit
def _decode_attn_gqa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, kv_len_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    D, softmax_scale,
    BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
    GROUP_KV: tl.constexpr, Q_PER_KV: tl.constexpr,
):
    T_kv = tl.load(kv_len_ptr)
    pid_b = tl.program_id(0)
    pid_gk = tl.program_id(1)
    kv_h_base = pid_gk * GROUP_KV

    for kgi in range(GROUP_KV):
        kv_h = kv_h_base + kgi

        off_kb = k_ptr + pid_b * stride_kb + kv_h * stride_kh
        off_vb = v_ptr + pid_b * stride_vb + kv_h * stride_vh

        for qgi in range(Q_PER_KV):
            q_h = kv_h * Q_PER_KV + qgi

            off_q = q_ptr + pid_b * stride_qb + q_h * stride_qh
            q = tl.load(off_q + tl.arange(0, BLOCK_D) * stride_qd, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            m_i = tl.full([1], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([1], dtype=tl.float32)

            for start_kv in range(0, T_kv, BLOCK_KV):
                off_k = off_kb + start_kv * stride_kt
                off_v = off_vb + start_kv * stride_vt
                kv_range = start_kv + tl.arange(0, BLOCK_KV)
                kv_mask = kv_range < T_kv

                k = tl.load(off_k + tl.arange(0, BLOCK_D)[:, None] * stride_kd + tl.arange(0, BLOCK_KV)[None, :] * stride_kt,
                            mask=(tl.arange(0, BLOCK_D)[:, None] < D) & kv_mask[None, :], other=0.0)

                scores = tl.sum(q[:, None] * k, axis=0) * softmax_scale
                scores = tl.where(kv_mask, scores, float('-inf'))

                m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
                p = tl.exp(scores - m_ij)
                l_ij = tl.sum(p, axis=0)

                alpha = tl.exp(m_i - m_ij)
                acc = acc * alpha + tl.sum(p[:, None] * tl.load(
                    off_v + tl.arange(0, BLOCK_D)[None, :] * stride_vd + tl.arange(0, BLOCK_KV)[:, None] * stride_vt,
                    mask=(tl.arange(0, BLOCK_D)[None, :] < D) & kv_mask[:, None], other=0.0
                ), axis=0)

                m_i = m_ij
                l_i = l_i * alpha + l_ij

            acc = acc / l_i

            off_o = o_ptr + pid_b * stride_ob + q_h * stride_oh
            tl.store(off_o + tl.arange(0, BLOCK_D) * stride_od, acc, mask=tl.arange(0, BLOCK_D) < D)


def attention_gqa(q, k, v, kv_len, q_per_kv):
    B, H_q, T_q, D = q.shape
    H_kv = k.shape[1]
    assert T_q == 1, "GQA decode only"
    block_kv = min(256, triton.next_power_of_2(D * 2))
    block_d = triton.next_power_of_2(D)
    o = torch.empty_like(q)
    group_kv = min(H_kv, 4)
    _decode_attn_gqa_kernel[(B, H_kv // group_kv)](
        q, k, v, o, kv_len,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0),         v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        D, D ** -0.5,
        BLOCK_KV=block_kv, BLOCK_D=block_d,
        GROUP_KV=group_kv, Q_PER_KV=q_per_kv,
        num_warps=8,
    )
    return o
