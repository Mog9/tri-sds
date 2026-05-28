import torch
import torch.nn.functional as F

from tri_sds.triton_kernels.norm import rms_norm
from tri_sds.triton_kernels.attention import attention_gqa


def precompute_rope(dim, max_len, base=1000000.0, device='cuda', dtype=torch.bfloat16):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x, cos, sin, pos):
    half = x.shape[-1] // 2
    T = pos.shape[0] if isinstance(pos, torch.Tensor) else 1
    cos_val = cos[pos].view(1, 1, T, half)
    sin_val = sin[pos].view(1, 1, T, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos_val - x2 * sin_val, x2 * cos_val + x1 * sin_val], dim=-1)


class QwenModelState:
    def __init__(self, hf_model, max_seq_len=None):
        cfg = hf_model.config
        self.N = cfg.hidden_size
        self.H_q = cfg.num_attention_heads
        self.H_kv = getattr(cfg, "num_key_value_heads", self.H_q)
        self.D = getattr(cfg, "head_dim", self.N // self.H_q)
        self.num_layers = cfg.num_hidden_layers
        self.intermediate_size = cfg.intermediate_size
        self.max_seq_len = max_seq_len or getattr(cfg, "max_position_embeddings", 4096)
        self.device = hf_model.device
        self.dtype = torch.bfloat16
        self.q_per_kv = self.H_q // self.H_kv
        self.q_dim = self.H_q * self.D
        self.kv_dim = self.H_kv * self.D

        rope_base = getattr(cfg, "rope_theta", 1000000.0)
        self.cos, self.sin = precompute_rope(self.D, self.max_seq_len, base=rope_base, device=self.device, dtype=self.dtype)

        pref = "model."
        self.norm_w = hf_model.get_parameter(pref + "norm.weight")

        self.embed_w = None
        self.lm_head_w = None
        try:
            self.embed_w = hf_model.get_parameter(pref + "embed_tokens.weight")
        except AttributeError:
            pass
        try:
            self.lm_head_w = hf_model.get_parameter("lm_head.weight")
        except AttributeError:
            pass

        if self.embed_w is None:
            self.embed_w = self.lm_head_w

        self.o_w_t = []
        self.down_w_t = []
        self.norm1_w = []
        self.norm2_w = []
        self.qkv_w_t = []
        self.gate_up_w_t = []

        for i in range(self.num_layers):
            lp = pref + f"layers.{i}."
            self.norm1_w.append(hf_model.get_parameter(lp + "input_layernorm.weight"))
            self.norm2_w.append(hf_model.get_parameter(lp + "post_attention_layernorm.weight"))

            q = hf_model.get_parameter(lp + "self_attn.q_proj.weight")
            k = hf_model.get_parameter(lp + "self_attn.k_proj.weight")
            v = hf_model.get_parameter(lp + "self_attn.v_proj.weight")
            self.qkv_w_t.append(torch.cat([q, k, v], dim=0).T.contiguous())
            self.o_w_t.append(hf_model.get_parameter(lp + "self_attn.o_proj.weight").T.contiguous())

            gate = hf_model.get_parameter(lp + "mlp.gate_proj.weight")
            up = hf_model.get_parameter(lp + "mlp.up_proj.weight")
            self.gate_up_w_t.append(torch.cat([gate, up], dim=0).T.contiguous())
            self.down_w_t.append(hf_model.get_parameter(lp + "mlp.down_proj.weight").T.contiguous())

    def reset_cache(self, batch_size=1):
        B = batch_size
        self.k_caches = [torch.zeros(B, self.H_kv, self.max_seq_len, self.D,
                                     device=self.device, dtype=self.dtype)
                         for _ in range(self.num_layers)]
        self.v_caches = [torch.zeros(B, self.H_kv, self.max_seq_len, self.D,
                                     device=self.device, dtype=self.dtype)
                         for _ in range(self.num_layers)]
        self._prefilled = 0

    def prefilled_len(self):
        return self._prefilled


def _qwen_prefill(x, state, start_pos):
    B, T = x.shape[:2]
    positions = torch.arange(start_pos, start_pos + T, device=x.device)

    for i in range(state.num_layers):
        x_norm = rms_norm(x.reshape(B * T, state.N), state.norm1_w[i])
        x_norm = x_norm.reshape(B, T, state.N)

        qkv = x_norm @ state.qkv_w_t[i]
        q = qkv[..., :state.q_dim].reshape(B, T, state.H_q, state.D).transpose(1, 2)
        k_start = state.q_dim
        k_end = state.q_dim + state.kv_dim
        k = qkv[..., k_start:k_end].reshape(B, T, state.H_kv, state.D).transpose(1, 2)
        v = qkv[..., k_end:].reshape(B, T, state.H_kv, state.D).transpose(1, 2)

        q = apply_rope(q, state.cos, state.sin, positions)
        k = apply_rope(k, state.cos, state.sin, positions)

        state.k_caches[i][:, :, start_pos:start_pos + T, :] = k
        state.v_caches[i][:, :, start_pos:start_pos + T, :] = v

        kv_len = start_pos + T
        k_full = state.k_caches[i][:, :, :kv_len, :].repeat_interleave(state.q_per_kv, dim=1)
        v_full = state.v_caches[i][:, :, :kv_len, :].repeat_interleave(state.q_per_kv, dim=1)
        if start_pos > 0:
            s = torch.arange(kv_len, device=q.device)
            t = torch.arange(T, device=q.device).unsqueeze(1)
            mask = (s.unsqueeze(0) <= start_pos + t).to(q.dtype)
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        else:
            attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        attn_flat = attn.transpose(1, 2).reshape(B * T, state.H_q * state.D)
        proj = attn_flat @ state.o_w_t[i]
        x = x + proj.reshape(B, T, state.N)

        x_norm2 = rms_norm(x.reshape(B * T, state.N), state.norm2_w[i])
        x_norm2 = x_norm2.reshape(B, T, state.N)

        gate_up = x_norm2 @ state.gate_up_w_t[i]
        gate = gate_up[..., :state.intermediate_size]
        up = gate_up[..., state.intermediate_size:]
        hidden = F.silu(gate) * up
        mlp_out = hidden @ state.down_w_t[i]
        x = x + mlp_out

    return x  # [B, T, N] hidden states before final RMSNorm + LM head


def _qwen_decode_step(x, state, i, pos, kv_len_tensor):
    B = x.shape[0]
    x_norm = rms_norm(x, state.norm1_w[i])

    qkv = x_norm @ state.qkv_w_t[i]
    q = qkv[..., :state.q_dim].reshape(B, state.H_q, 1, state.D)
    k_start = state.q_dim
    k_end = state.q_dim + state.kv_dim
    kh = qkv[..., k_start:k_end].reshape(B, state.H_kv, 1, state.D)
    v = qkv[..., k_end:].reshape(B, state.H_kv, 1, state.D)

    q = apply_rope(q, state.cos, state.sin, pos)
    kh = apply_rope(kh, state.cos, state.sin, pos)

    state.k_caches[i][:, :, pos:pos + 1, :] = kh
    state.v_caches[i][:, :, pos:pos + 1, :] = v

    attn_out = attention_gqa(q, state.k_caches[i], state.v_caches[i],
                             kv_len_tensor, state.q_per_kv)
    attn_flat = attn_out.reshape(B, state.H_q * state.D)
    proj = attn_flat @ state.o_w_t[i]
    x = x + proj

    x_norm2 = rms_norm(x, state.norm2_w[i])
    gate_up = x_norm2 @ state.gate_up_w_t[i]
    gate = gate_up[..., :state.intermediate_size]
    up = gate_up[..., state.intermediate_size:]
    hidden = F.silu(gate) * up
    mlp_out = hidden @ state.down_w_t[i]
    x = x + mlp_out

    return x


def qwen_model_forward(input_ids, state, return_hidden=False):
    B, T = input_ids.shape
    start_pos = state._prefilled

    if T > 1:
        x = state.embed_w[input_ids]
        hidden = _qwen_prefill(x, state, start_pos)
        state._prefilled += T
        x_flat = rms_norm(hidden.reshape(B * T, state.N), state.norm_w)
        logits = x_flat @ state.lm_head_w.T
        logits = logits.reshape(B, T, -1)
        if return_hidden:
            return logits, hidden
        return logits
    else:
        kv_len_tensor = torch.tensor([start_pos + 1], device=input_ids.device, dtype=torch.int32)
        x = state.embed_w[input_ids].squeeze(1)
        pos = start_pos
        for i in range(state.num_layers):
            x = _qwen_decode_step(x, state, i, pos, kv_len_tensor)
        hidden = x  # [B, N] before final RMSNorm + LM head
        x_flat = rms_norm(hidden, state.norm_w)
        logits = x_flat @ state.lm_head_w.T
        state._prefilled += 1
        if return_hidden:
            return logits.unsqueeze(1), hidden.unsqueeze(1)
        return logits.unsqueeze(1)
