"""draft model state — weight extraction from gpt-2 huggingface models.

usage:
    state = DraftModelState(hf_model)
    logits = draft_model_forward(state, input_ids, position_ids)
"""

import torch

from tri_sds.triton_kernels.model import model_forward


class DraftModelState:
    """holds model weights, kv caches, and config for triton inference."""

    def __init__(self, hf_model, max_seq_len=None):
        cfg = hf_model.config
        model_type = getattr(cfg, "model_type", "").lower()
        assert model_type in ("gpt2", "distilgpt2"), f"only gpt-2 family supported, got {model_type}"

        self.N = cfg.hidden_size
        self.H = cfg.num_attention_heads
        self.D = self.N // self.H
        self.num_layers = cfg.num_hidden_layers
        self.max_seq_len = max_seq_len or getattr(cfg, "max_position_embeddings", 2048)
        self.device = hf_model.device
        self.dtype = torch.float16
        self.model_type = model_type

        pref = "transformer."
        wte = hf_model.get_parameter(pref + "wte.weight")
        self.wte_w = wte
        self.wte_w_t = wte.T.contiguous()
        self.wpe_w = hf_model.get_parameter(pref + "wpe.weight")
        self.ln_f_w = hf_model.get_parameter(pref + "ln_f.weight")
        self.ln_f_b = hf_model.get_parameter(pref + "ln_f.bias")

        self.lm_head_w = None
        self.lm_head_w_t = None
        try:
            self.lm_head_w = hf_model.get_parameter("lm_head.weight")
            self.lm_head_w_t = self.lm_head_w.T.contiguous()
        except AttributeError:
            pass

        self.blocks_params = []
        for i in range(self.num_layers):
            blk = hf_model.get_submodule(pref + f"h.{i}")
            self.blocks_params.append((
                blk.ln_1.weight, blk.ln_1.bias,
                blk.attn.c_attn.weight, blk.attn.c_attn.bias,
                blk.attn.c_proj.weight, blk.attn.c_proj.bias,
                blk.ln_2.weight, blk.ln_2.bias,
                blk.mlp.c_fc.weight, blk.mlp.c_fc.bias,
                blk.mlp.c_proj.weight, blk.mlp.c_proj.bias,
            ))

    def reset_cache(self, batch_size=1):
        B = batch_size
        self.k_caches = [torch.zeros(B, self.H, self.max_seq_len, self.D,
                                     device=self.device, dtype=self.dtype)
                         for _ in range(self.num_layers)]
        self.v_caches = [torch.zeros(B, self.H, self.max_seq_len, self.D,
                                     device=self.device, dtype=self.dtype)
                         for _ in range(self.num_layers)]
        self.cache_len = 0


def draft_model_forward(state, input_ids, position_ids):
    B, T = input_ids.shape
    assert B == 1, "batch > 1 not yet supported"

    logits = model_forward(
        input_ids, position_ids,
        state.wte_w, state.wpe_w,
        state.blocks_params,
        state.ln_f_w, state.ln_f_b,
        state.k_caches, state.v_caches,
        state.H, use_cache=True,
        lm_head_weight=getattr(state, 'lm_head_w', None),
        wte_weight_t=getattr(state, 'wte_w_t', None),
        lm_head_weight_t=getattr(state, 'lm_head_w_t', None),
    )
    state.cache_len += T
    return logits
