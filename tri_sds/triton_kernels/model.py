"""full model forward pass using custom triton kernels.

two paths:
  prefill (T > 1): all tokens processed simultaneously per layer.
  decode  (T = 1): single token with kv cache (fused block per layer).

kv cache is populated during prefill so decode can reuse it.
"""

import torch
import torch.nn.functional as F

from tri_sds.triton_kernels.fused_block import fused_block_split


def model_forward(
    input_ids,
    position_ids,
    wte_weight,
    wpe_weight,
    blocks_params,
    ln_f_weight, ln_f_bias,
    k_caches, v_caches,
    num_heads,
    use_cache=True,
    lm_head_weight=None,
    wte_weight_t=None,
    lm_head_weight_t=None,
    kv_len_tensor=None,
    pos_tensor=None,
):
    B, T = input_ids.shape
    N = wte_weight.shape[1]
    H = num_heads
    D = N // H

    x = wte_weight[input_ids] + wpe_weight[position_ids]

    if T == 1:
        for t in range(T):
            if kv_len_tensor is not None:
                xt = x[:, t, :]
                for i in range(len(blocks_params)):
                    params = blocks_params[i]
                    ln1_w, ln1_b, qkv_w, qkv_b, attn_out_w, attn_out_b, \
                        ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b = params
                    xt = fused_block_split(
                        xt, ln1_w, ln1_b, qkv_w, qkv_b,
                        attn_out_w, attn_out_b, ln2_w, ln2_b,
                        fc1_w, fc1_b, fc2_w, fc2_b,
                        k_caches[i], v_caches[i],
                        0, use_cache=True,
                        kv_len=kv_len_tensor, pos_tensor=pos_tensor,
                    )
                x[:, t, :] = xt
            else:
                p = position_ids[0, t].item()
                kv_len = torch.tensor([p + 1], device=x.device, dtype=torch.int32)
                xt = x[:, t, :]
                for i in range(len(blocks_params)):
                    params = blocks_params[i]
                    ln1_w, ln1_b, qkv_w, qkv_b, attn_out_w, attn_out_b, \
                        ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b = params
                    xt = fused_block_split(
                        xt, ln1_w, ln1_b, qkv_w, qkv_b,
                        attn_out_w, attn_out_b, ln2_w, ln2_b,
                        fc1_w, fc1_b, fc2_w, fc2_b,
                        k_caches[i], v_caches[i],
                        p, use_cache=True, kv_len=kv_len,
                    )
                x[:, t, :] = xt

        x_flat = x.reshape(B, N)
        x_flat = F.layer_norm(x_flat, (N,), ln_f_weight, ln_f_bias)
        if lm_head_weight_t is not None:
            logits = x_flat @ lm_head_weight_t
        elif wte_weight_t is not None:
            logits = x_flat @ wte_weight_t
        else:
            logits = x_flat @ wte_weight.T
        return logits.view(B, -1, logits.shape[-1])
    else:
        return _prefill_batched(x, position_ids, blocks_params,
                                ln_f_weight, ln_f_bias, k_caches, v_caches,
                                B, T, N, H, D, use_cache, lm_head_weight, wte_weight,
                                wte_weight_t=wte_weight_t, lm_head_weight_t=lm_head_weight_t)


def _prefill_batched(x, position_ids, blocks_params,
                     ln_f_weight, ln_f_bias, k_caches, v_caches,
                     B, T, N, H, D, use_cache, lm_head_weight, wte_weight,
                     wte_weight_t=None, lm_head_weight_t=None):
    """process all T tokens simultaneously per layer using batched QKV."""
    start_pos = position_ids[0, 0].item()

    for i, params in enumerate(blocks_params):
        ln1_w, ln1_b, qkv_w, qkv_b, attn_out_w, attn_out_b, \
            ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b = params

        x_flat = x.reshape(B * T, N)
        x_norm = F.layer_norm(x_flat, (N,), ln1_w, ln1_b).reshape(B, T, N)

        qkv = x_norm @ qkv_w + qkv_b
        q, k, v = qkv.split(N, dim=-1)
        q = q.reshape(B, T, H, D).transpose(1, 2)
        k = k.reshape(B, T, H, D).transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)

        if use_cache:
            k_caches[i][:, :, start_pos:start_pos + T, :] = k
            v_caches[i][:, :, start_pos:start_pos + T, :] = v

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_flat = attn.transpose(1, 2).reshape(B * T, N)
        proj = attn_flat @ attn_out_w + attn_out_b
        x = x + proj.reshape(B, T, N)

        x_flat = x.reshape(B * T, N)
        x_norm2 = F.layer_norm(x_flat, (N,), ln2_w, ln2_b)
        fc1 = torch.nn.functional.gelu(x_norm2 @ fc1_w + fc1_b)
        mlp_out = fc1 @ fc2_w + fc2_b
        x = x + mlp_out.reshape(B, T, N)

    x_flat = x.reshape(B * T, N)
    x_flat = F.layer_norm(x_flat, (N,), ln_f_weight, ln_f_bias)

    if lm_head_weight_t is not None:
        logits = x_flat @ lm_head_weight_t
    elif wte_weight_t is not None:
        logits = x_flat @ wte_weight_t
    else:
        head_w = lm_head_weight if lm_head_weight is not None else wte_weight
        logits = x_flat @ head_w.T
    return logits.view(B, T, logits.shape[-1])
