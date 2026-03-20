"""
Extreme experimentation entry point derived from `train_gpt_competitive.py`.

Contract:
- `train_gpt_competitive.py` is the frozen incumbent / promotion target.
- `train_gpt_extreme.py` is the place for aggressive but still challenge-relevant work:
  - stronger static backbones
  - optimization-improving architectural changes
  - compression-aware training/export changes

Out of scope for this branch unless evidence changes:
- document-episodic meta training
- eval-time TTT variants
- persistent fast state across documents
- tokenizer-family changes

First divergence from the incumbent:
- NormFormer-lite block outputs via branch-local RMSNorm on attention and MLP outputs
  before the residual addition.

This file intentionally reuses the incumbent training loop and utilities, but swaps in an
alternative GPT/Block implementation so architectural experiments do not mutate the
validated competitive path.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor, nn

import train_gpt_competitive as base


class ExtremeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attn: base.CausalSelfAttention | None = None,
        mlp: base.MLP | None = None,
    ):
        super().__init__()
        self.use_normformer_lite = bool(int(os.environ.get("EXTREME_NORMFORMER_LITE", "1")))
        self.attn_norm = base.RMSNorm(dim)
        self.mlp_norm = base.RMSNorm(dim)
        self.attn = attn if attn is not None else base.CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = mlp if mlp is not None else base.MLP(dim, mlp_mult)
        self.attn_post_norm = base.RMSNorm(dim) if self.use_normformer_lite else nn.Identity()
        self.mlp_post_norm = base.RMSNorm(dim) if self.use_normformer_lite else nn.Identity()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, fast_state: dict[str, Tensor] | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), fast_state=fast_state)
        attn_out = self.attn_post_norm(attn_out)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x), fast_state=fast_state)
        mlp_out = self.mlp_post_norm(mlp_out)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x


class ExtremeGPT(base.GPT):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_factor_embed: bool,
        embed_dim: int,
        num_unique_attn: int,
        num_unique_mlp: int,
        use_fast_adapters: bool,
        fast_rank: int,
        fast_gate_init: float,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            tie_embeddings=tie_embeddings,
            tied_embed_init_std=tied_embed_init_std,
            logit_softcap=logit_softcap,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
            use_factor_embed=use_factor_embed,
            embed_dim=embed_dim,
            num_unique_attn=num_unique_attn,
            num_unique_mlp=num_unique_mlp,
            use_fast_adapters=use_fast_adapters,
            fast_rank=fast_rank,
            fast_gate_init=fast_gate_init,
        )
        self.blocks = nn.ModuleList(
            [
                ExtremeBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    attn=self.attn_bank[i % num_unique_attn],
                    mlp=self.mlp_bank[i % num_unique_mlp],
                )
                for i in range(num_layers)
            ]
        )


def main(argv: list[str] | None = None) -> None:
    # This branch is meant for architecture exploration, not artifact accounting yet.
    # We intentionally reuse the incumbent training loop and validation/export stack.
    base.GPT = ExtremeGPT
    base.Block = ExtremeBlock
    base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main(sys.argv[1:])
