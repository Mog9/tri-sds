#!/usr/bin/env python3
"""Train EAGLE draft head on generated data.

  python tri_sds/triton_kernels/train_eagle.py eagle_data/eagle_data_1000.pth

Loads saved hidden states + tokens, trains EagleHead with MSE loss,
saves weights to eagle_head_weights.pth.

Designed for CPU (fits 4GB VRAM). Use --device cuda if >8GB.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tri_sds.triton_kernels.eagle_head import EagleHead

DTYPE = torch.bfloat16


def init_head(hidden_dim, intermediate_size, device, dtype=DTYPE):
    head = EagleHead(hidden_dim, intermediate_size, device=device, dtype=dtype)
    specs = {
        'fc_weight': [2 * hidden_dim, hidden_dim],
        'fc_bias': [hidden_dim],
        'attn_norm_w': [hidden_dim],
        'attn_q_w': [hidden_dim, hidden_dim],
        'attn_k_w': [hidden_dim, hidden_dim],
        'attn_v_w': [hidden_dim, hidden_dim],
        'attn_o_w': [hidden_dim, hidden_dim],
        'ffn_norm_w': [hidden_dim],
        'ffn_gate_w': [hidden_dim, intermediate_size],
        'ffn_up_w': [hidden_dim, intermediate_size],
        'ffn_down_w': [intermediate_size, hidden_dim],
    }
    for name, shape in specs.items():
        t = torch.empty(*shape, device=device, dtype=dtype)
        t.requires_grad_(True)
        if 'bias' in name:
            t.data.zero_()
        elif 'norm' in name:
            t.data.fill_(1.0)
        else:
            torch.nn.init.normal_(t.data, std=0.02)
        setattr(head, name, t)
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to eagle_data_N.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save", type=str, default="eagle_head_weights.pth")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Loading {args.data_path} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    data = torch.load(args.data_path, map_location="cpu", weights_only=True)
    hs = data["hidden_states"]
    toks = data["tokens"]
    print(f"done ({time.perf_counter() - t0:.1f}s)")
    print(f"  hidden: {hs.shape}  tokens: {toks.shape}")

    N_seq, T, D = hs.shape
    from transformers import AutoModel
    from tri_sds.triton_kernels.qwen_model import QwenModelState
    hf_model = AutoModel.from_pretrained("Qwen/Qwen3-32B", torch_dtype=DTYPE, device_map="cpu")
    dummy_state = QwenModelState(hf_model)
    del hf_model
    embed_w = dummy_state.embed_w
    intermediate_size = dummy_state.intermediate_size
    del dummy_state
    print(f"  D={D}  intermediate={intermediate_size}")

    device = torch.device(args.device)
    head = init_head(D, intermediate_size, device)
    ps = [(k, v) for k, v in head.__dict__.items()
          if isinstance(v, torch.Tensor) and v.requires_grad]
    n_p = sum(v.numel() for _, v in ps)
    print(f"  params: {n_p:,} ({n_p * 4 / 1024**3:.1f} GB fp32)")

    embed_w = embed_w.to(device=device, dtype=DTYPE)

    m, v = {}, {}
    for k, p in ps:
        m[k] = torch.zeros_like(p)
        v[k] = torch.zeros_like(p)

    t_start = time.perf_counter()
    global_step = 0

    for epoch in range(args.epochs):
        torch.manual_seed(42 + epoch)
        perm = torch.randperm(N_seq)
        n_batches = (N_seq + args.batch_size - 1) // args.batch_size

        for bi in range(n_batches):
            idx = perm[bi * args.batch_size: (bi + 1) * args.batch_size]
            B = len(idx)
            if B == 0:
                continue

            h_b = hs[idx].to(device, dtype=DTYPE)
            t_b = toks[idx].to(device)

            feat = h_b[:, :-1]        # [B, T-1, D]
            tgt = h_b[:, 1:]          # [B, T-1, D]
            temb = embed_w[t_b[:, 1:]]  # [B, T-1, D]

            fused = torch.cat([feat, temb], dim=-1)  # [B, T-1, 2D]
            h = fused @ head.fc_weight + head.fc_bias  # [B, T-1, D]

            an = F.rms_norm(h.reshape(-1, D), (D,), weight=head.attn_norm_w).reshape(B, -1, D)
            q = (an @ head.attn_q_w).unsqueeze(1)
            k = (an @ head.attn_k_w).unsqueeze(1)
            v = (an @ head.attn_v_w).unsqueeze(1)
            ao = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ao = ao.squeeze(1) @ head.attn_o_w
            h = h + ao

            fn = F.rms_norm(h.reshape(-1, D), (D,), weight=head.ffn_norm_w).reshape(B, -1, D)
            gu = torch.cat([fn @ head.ffn_gate_w, fn @ head.ffn_up_w], dim=-1)
            gate, up = gu.chunk(2, dim=-1)
            ffn_h = F.silu(gate) * up
            h = h + ffn_h @ head.ffn_down_w

            loss = F.mse_loss(h.float(), tgt.float())
            lv = loss.item()

            loss.backward()

            for kk, pp in ps:
                if pp.grad is not None:
                    step = global_step + 1
                    m[kk].mul_(0.9).add_(pp.grad.float(), alpha=0.1)
                    v[kk].mul_(0.999).addcmul_(pp.grad.float(), pp.grad.float(), value=0.001)
                    mh = m[kk] / (1 - 0.9 ** step)
                    vh = v[kk] / (1 - 0.999 ** step)
                    pp.data.addcdiv_(mh, vh.sqrt().add_(1e-8), value=-args.lr)
                    pp.grad = None

            global_step += 1
            el = time.perf_counter() - t_start
            tps = (bi + 1) * B * (T - 1) / el if el > 0 else 0
            print(f"  e{epoch+1}/{args.epochs} b{bi+1}/{n_batches} "
                  f"loss={lv:.6f} tok/s={tps:.0f} el={el:.0f}s", flush=True)

    print(f"\nDone in {time.perf_counter() - t_start:.0f}s")

    sd = {
        'fc.weight': head.fc_weight.detach().cpu().to(DTYPE),
        'fc.bias': head.fc_bias.detach().cpu().to(DTYPE),
        'attn.norm.weight': head.attn_norm_w.detach().cpu().to(DTYPE),
        'attn.q_proj.weight': head.attn_q_w.detach().cpu().to(DTYPE),
        'attn.k_proj.weight': head.attn_k_w.detach().cpu().to(DTYPE),
        'attn.v_proj.weight': head.attn_v_w.detach().cpu().to(DTYPE),
        'attn.o_proj.weight': head.attn_o_w.detach().cpu().to(DTYPE).T.contiguous(),
        'ffn.norm.weight': head.ffn_norm_w.detach().cpu().to(DTYPE),
        'ffn.gate_proj.weight': head.ffn_gate_w.detach().cpu().to(DTYPE).T.contiguous(),
        'ffn.up_proj.weight': head.ffn_up_w.detach().cpu().to(DTYPE).T.contiguous(),
        'ffn.down_proj.weight': head.ffn_down_w.detach().cpu().to(DTYPE).T.contiguous(),
    }
    torch.save(sd, args.save)
    print(f"Saved: {args.save} ({os.path.getsize(args.save) / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
