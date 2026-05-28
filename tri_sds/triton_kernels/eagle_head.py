import torch
import torch.nn.functional as F

from tri_sds.triton_kernels.norm import rms_norm


def _eagle_attention(q, k, v):
    """Single-head causal self-attention via PyTorch flash attention.
    
    Avoids custom Triton kernel — the BLOCK_D=8192 needed for D=5120
    causes register pressure issues on MI300X/ROCm.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class EagleHead:
    """EAGLE draft head — single decoder layer predicting next features.

    Forward:  (features, token_embeds) → [B, t, D]  predicted features
        features:     [B, t, D]  target hidden states at positions 1..t
        token_embeds: [B, t, D]  embeddings of tokens at positions 2..t+1 (shifted)
    """

    def __init__(self, hidden_dim, intermediate_size, device='cuda', dtype=torch.bfloat16):
        self.D = hidden_dim
        self.ff_dim = intermediate_size
        self.device = device
        self.dtype = dtype

        self.fc_weight = None
        self.fc_bias = None

        self.attn_norm_w = None
        self.attn_q_w = None
        self.attn_k_w = None
        self.attn_v_w = None
        self.attn_o_w = None

        self.ffn_norm_w = None
        self.ffn_gate_w = None
        self.ffn_up_w = None
        self.ffn_down_w = None

    def to(self, device):
        self.device = device
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, torch.Tensor):
                setattr(self, attr, v.to(device))
        return self

    def load_state_dict(self, sd):
        def _t(k):
            return sd[k].to(self.dtype).to(self.device) if k in sd else None

        self.fc_weight = _t('fc.weight')
        self.fc_bias = _t('fc.bias')
        if self.fc_bias is None:
            self.fc_bias = torch.zeros(self.D, device=self.device, dtype=self.dtype)

        self.attn_norm_w = _t('attn.norm.weight')
        self.attn_q_w = _t('attn.q_proj.weight')
        self.attn_k_w = _t('attn.k_proj.weight')
        self.attn_v_w = _t('attn.v_proj.weight')
        o_w = _t('attn.o_proj.weight')
        self.attn_o_w = o_w.T.contiguous() if o_w is not None else None

        self.ffn_norm_w = _t('ffn.norm.weight')
        g_w = _t('ffn.gate_proj.weight')
        u_w = _t('ffn.up_proj.weight')
        d_w = _t('ffn.down_proj.weight')
        self.ffn_gate_w = g_w if g_w is None else g_w.T.contiguous()
        self.ffn_up_w = u_w if u_w is None else u_w.T.contiguous()
        self.ffn_down_w = d_w if d_w is None else d_w.T.contiguous()

    def forward(self, features, token_embeds):
        B, t = features.shape[:2]

        fused = torch.cat([features, token_embeds], dim=-1)
        h = fused @ self.fc_weight + self.fc_bias

        attn_in = rms_norm(h.reshape(B * t, self.D), self.attn_norm_w).reshape(B, t, self.D)
        q = (attn_in @ self.attn_q_w).unsqueeze(1)
        k = (attn_in @ self.attn_k_w).unsqueeze(1)
        v = (attn_in @ self.attn_v_w).unsqueeze(1)
        attn_out = _eagle_attention(q, k, v)
        attn_out = attn_out.squeeze(1) @ self.attn_o_w
        h = h + attn_out

        ffn_in = rms_norm(h.reshape(B * t, self.D), self.ffn_norm_w).reshape(B, t, self.D)
        gate = ffn_in @ self.ffn_gate_w
        up = ffn_in @ self.ffn_up_w
        hidden = F.silu(gate) * up
        ffn_out = hidden @ self.ffn_down_w
        h = h + ffn_out

        return h


def eagle_draft_decode(last_hidden, last_token, eagle_head, target_model_state, n_draft=4):
    B, D = last_hidden.shape[0], last_hidden.shape[-1]

    feat_list = [last_hidden]
    tok_list = [last_token]

    draft_tokens = []
    draft_logits = []

    for step in range(n_draft):
        feats = torch.cat(feat_list, dim=1)
        toks = torch.cat(tok_list, dim=1)

        tok_embed = target_model_state.embed_w[toks]

        pred_feats = eagle_head.forward(feats, tok_embed)

        next_feat = pred_feats[:, -1:, :]

        x_flat = rms_norm(next_feat.reshape(B, -1), target_model_state.norm_w)
        logits = x_flat @ target_model_state.lm_head_w.T
        token = logits.argmax(dim=-1, keepdim=True)

        draft_tokens.append(token)
        draft_logits.append(logits)

        feat_list.append(next_feat)
        tok_list.append(token)

    return draft_tokens, draft_logits
