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
- Optional recurrent reapplication of the shared stack with per-pass gates to buy
  effective depth without introducing a fully separate backbone.
- Explicit layer modulation on shared attention / MLP outputs.
- Depth-aware residual and skip-scale initialization for deeper effective stacks.

This file intentionally reuses the incumbent training loop and utilities, but swaps in an
alternative GPT/Block implementation so architectural experiments do not mutate the
validated competitive path.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import train_gpt_competitive as base


def depth_aware_branch_scale(num_layers: int, recurrent_passes: int) -> float:
    effective_depth = max(num_layers * recurrent_passes, 1)
    return (2.0 * effective_depth) ** -0.5

class LowRankDelta(nn.Module):
    def __init__(self, dim: int, rank: int, init_scale: float):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, dim, dtype=torch.float32))
        self.up = nn.Parameter(torch.empty(dim, rank, dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        nn.init.normal_(self.up, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.linear(x, self.down.to(dtype=x.dtype))
        return self.gain.to(dtype=x.dtype) * F.linear(hidden, self.up.to(dtype=x.dtype))

class LowRankLogitDelta(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, init_scale: float):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, d_in, dtype=torch.float32))
        self.up = nn.Parameter(torch.empty(d_out, rank, dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        nn.init.normal_(self.up, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.linear(x, self.down.to(dtype=x.dtype))
        return self.gain.to(dtype=x.dtype) * F.linear(hidden, self.up.to(dtype=x.dtype))

def extreme_attention_forward(
    attn: base.CausalSelfAttention,
    x: Tensor,
    fast_state: dict[str, Tensor] | None = None,
    q_gain_scale: Tensor | None = None,
) -> Tensor:
    bsz, seqlen, dim = x.shape
    q = attn.c_q(x).reshape(bsz, seqlen, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.c_k(x).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v_base = attn.c_v(x)
    if attn.v_fast is not None:
        v_base = v_base + attn.v_fast.delta(x, fast_state)
    v = v_base.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = attn.rotary(seqlen, x.device, q.dtype)
    q = base.apply_rotary_emb(q, cos, sin)
    k = base.apply_rotary_emb(k, cos, sin)
    q_gain = attn.q_gain
    if q_gain_scale is not None:
        q_gain = q_gain * q_gain_scale.to(dtype=q_gain.dtype)
    q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        enable_gqa=(attn.num_kv_heads != attn.num_heads),
    )
    y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
    return attn.proj(y)


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
        branch_scale_init: float = 1.0,
    ):
        super().__init__()
        self.use_normformer_lite = bool(int(os.environ.get("EXTREME_NORMFORMER_LITE", "1")))
        self.use_layer_modulation = bool(int(os.environ.get("EXTREME_LAYER_MODULATION", "1")))
        self.use_low_rank_deltas = bool(int(os.environ.get("EXTREME_LOW_RANK_DELTAS", "0")))
        self.delta_rank = int(os.environ.get("EXTREME_DELTA_RANK", "4"))
        self.delta_init_scale = float(os.environ.get("EXTREME_DELTA_INIT_SCALE", "0.05"))
        self.attn_norm = base.RMSNorm(dim)
        self.mlp_norm = base.RMSNorm(dim)
        self.attn = attn if attn is not None else base.CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = mlp if mlp is not None else base.MLP(dim, mlp_mult)
        self.attn_post_norm = base.RMSNorm(dim) if self.use_normformer_lite else nn.Identity()
        self.mlp_post_norm = base.RMSNorm(dim) if self.use_normformer_lite else nn.Identity()
        self.attn_delta = (
            LowRankDelta(dim, self.delta_rank, self.delta_init_scale)
            if self.use_low_rank_deltas and self.delta_rank > 0 else None
        )
        self.mlp_delta = (
            LowRankDelta(dim, self.delta_rank, self.delta_init_scale)
            if self.use_low_rank_deltas and self.delta_rank > 0 else None
        )
        self.attn_scale = nn.Parameter(torch.full((dim,), branch_scale_init, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), branch_scale_init, dtype=torch.float32))
        if self.use_layer_modulation:
            self.attn_mod_gain = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.attn_mod_bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.mlp_mod_gain = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.mlp_mod_bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_parameter("attn_mod_gain", None)
            self.register_parameter("attn_mod_bias", None)
            self.register_parameter("mlp_mod_gain", None)
            self.register_parameter("mlp_mod_bias", None)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        fast_state: dict[str, Tensor] | None = None,
        pass_attn_gain: Tensor | None = None,
        pass_attn_bias: Tensor | None = None,
        pass_mlp_gain: Tensor | None = None,
        pass_mlp_bias: Tensor | None = None,
        q_gain_scale: Tensor | None = None,
    ) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = extreme_attention_forward(self.attn, self.attn_norm(x), fast_state=fast_state, q_gain_scale=q_gain_scale)
        attn_out = self.attn_post_norm(attn_out)
        if self.use_layer_modulation:
            attn_out = (
                attn_out * self.attn_mod_gain.to(dtype=x.dtype)[None, None, :]
                + self.attn_mod_bias.to(dtype=x.dtype)[None, None, :]
            )
        if pass_attn_gain is not None:
            attn_out = attn_out * pass_attn_gain.to(dtype=x.dtype)[None, None, :]
        if pass_attn_bias is not None:
            attn_out = attn_out + pass_attn_bias.to(dtype=x.dtype)[None, None, :]
        if self.attn_delta is not None:
            attn_out = attn_out + self.attn_delta(attn_out)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x), fast_state=fast_state)
        mlp_out = self.mlp_post_norm(mlp_out)
        if self.use_layer_modulation:
            mlp_out = (
                mlp_out * self.mlp_mod_gain.to(dtype=x.dtype)[None, None, :]
                + self.mlp_mod_bias.to(dtype=x.dtype)[None, None, :]
            )
        if pass_mlp_gain is not None:
            mlp_out = mlp_out * pass_mlp_gain.to(dtype=x.dtype)[None, None, :]
        if pass_mlp_bias is not None:
            mlp_out = mlp_out + pass_mlp_bias.to(dtype=x.dtype)[None, None, :]
        if self.mlp_delta is not None:
            mlp_out = mlp_out + self.mlp_delta(mlp_out)
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
        self.recurrent_passes = int(os.environ.get("EXTREME_RECURRENT_PASSES", "1"))
        if self.recurrent_passes <= 0:
            raise ValueError(f"EXTREME_RECURRENT_PASSES must be positive, got {self.recurrent_passes}")
        self.use_normformer_lite = bool(int(os.environ.get("EXTREME_NORMFORMER_LITE", "1")))
        self.use_layer_modulation = bool(int(os.environ.get("EXTREME_LAYER_MODULATION", "1")))
        self.use_low_rank_deltas = bool(int(os.environ.get("EXTREME_LOW_RANK_DELTAS", "0")))
        self.use_recurrent_gates = bool(int(os.environ.get("EXTREME_RECURRENT_GATES", "1")))
        self.use_depth_aware_residuals = bool(int(os.environ.get("EXTREME_DEPTH_AWARE_RESIDUALS", "1")))
        self.use_pass_modulation = bool(int(os.environ.get("EXTREME_PASS_MODULATION", "0")))
        self.use_pass_q_gain = bool(int(os.environ.get("EXTREME_PASS_Q_GAIN", "0")))
        self.logit_delta_rank = int(os.environ.get("EXTREME_LOGIT_DELTA_RANK", "0"))
        self.logit_delta_init_scale = float(os.environ.get("EXTREME_LOGIT_DELTA_INIT_SCALE", "0.05"))
        pass_gate_init = float(os.environ.get("EXTREME_PASS_GATE_INIT", "1.0"))
        self.extreme_active = (
            self.use_normformer_lite
            or self.use_layer_modulation
            or self.use_low_rank_deltas
            or self.use_depth_aware_residuals
            or self.use_pass_modulation
            or self.use_pass_q_gain
            or self.logit_delta_rank > 0
            or self.recurrent_passes != 1
        )
        if not self.extreme_active:
            self.use_recurrent_gates = False
            self.pass_gates = None
            self.logit_delta = None
            return
        branch_scale_init = (
            depth_aware_branch_scale(num_layers, self.recurrent_passes)
            if self.use_depth_aware_residuals else 1.0
        )
        if self.use_recurrent_gates:
            self.pass_gates = nn.Parameter(
                torch.full((self.recurrent_passes, model_dim), pass_gate_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "pass_gates",
                torch.ones(self.recurrent_passes, model_dim, dtype=torch.float32),
                persistent=False,
            )
        if self.use_pass_modulation:
            self.pass_attn_gain = nn.Parameter(torch.ones(self.recurrent_passes, model_dim, dtype=torch.float32))
            self.pass_attn_bias = nn.Parameter(torch.zeros(self.recurrent_passes, model_dim, dtype=torch.float32))
            self.pass_mlp_gain = nn.Parameter(torch.ones(self.recurrent_passes, model_dim, dtype=torch.float32))
            self.pass_mlp_bias = nn.Parameter(torch.zeros(self.recurrent_passes, model_dim, dtype=torch.float32))
        else:
            self.register_parameter("pass_attn_gain", None)
            self.register_parameter("pass_attn_bias", None)
            self.register_parameter("pass_mlp_gain", None)
            self.register_parameter("pass_mlp_bias", None)
        if self.use_pass_q_gain:
            self.pass_q_gain = nn.Parameter(torch.ones(self.recurrent_passes, num_heads, dtype=torch.float32))
        else:
            self.register_parameter("pass_q_gain", None)
        self.logit_delta = (
            LowRankLogitDelta(model_dim, vocab_size, self.logit_delta_rank, self.logit_delta_init_scale)
            if self.logit_delta_rank > 0 else None
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
                    branch_scale_init=branch_scale_init,
                )
                for i in range(num_layers)
            ]
        )
        if self.use_depth_aware_residuals and self.num_skip_weights > 0:
            with torch.no_grad():
                self.skip_weights.fill_(branch_scale_init)

    def _run_stack(self, x: Tensor, x0: Tensor, fast_state: dict[str, Tensor] | None = None, pass_idx: int = 0) -> Tensor:
        pass_attn_gain = self.pass_attn_gain[pass_idx] if self.pass_attn_gain is not None else None
        pass_attn_bias = self.pass_attn_bias[pass_idx] if self.pass_attn_bias is not None else None
        pass_mlp_gain = self.pass_mlp_gain[pass_idx] if self.pass_mlp_gain is not None else None
        pass_mlp_bias = self.pass_mlp_bias[pass_idx] if self.pass_mlp_bias is not None else None
        q_gain_scale = self.pass_q_gain[pass_idx] if self.pass_q_gain is not None else None
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](
                x,
                x0,
                fast_state=fast_state,
                pass_attn_gain=pass_attn_gain,
                pass_attn_bias=pass_attn_bias,
                pass_mlp_gain=pass_mlp_gain,
                pass_mlp_bias=pass_mlp_bias,
                q_gain_scale=q_gain_scale,
            )
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](
                x,
                x0,
                fast_state=fast_state,
                pass_attn_gain=pass_attn_gain,
                pass_attn_bias=pass_attn_bias,
                pass_mlp_gain=pass_mlp_gain,
                pass_mlp_bias=pass_mlp_bias,
                q_gain_scale=q_gain_scale,
            )
        return x

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        fast_state: dict[str, Tensor] | None = None,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        if not self.extreme_active:
            return super().forward(input_ids, target_ids, fast_state=fast_state, loss_mask=loss_mask)

        x = self.tok_emb(input_ids)
        if getattr(self, "emb_proj", None) is not None:
            x = self.emb_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        if self.recurrent_passes == 1:
            x = self._run_stack(x, x0, fast_state=fast_state, pass_idx=0)
        else:
            for pass_idx in range(self.recurrent_passes):
                x_in = x
                x_out = self._run_stack(x, x0, fast_state=fast_state, pass_idx=pass_idx)
                gate = self.pass_gates[pass_idx].to(dtype=x.dtype)[None, None, :]
                x = x_in + gate * (x_out - x_in)

        x = self.final_norm(x)
        if self.out_fast is not None:
            x = x + self.out_fast.delta(x, fast_state)

        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            if getattr(self, "emb_proj", None) is not None:
                z = F.linear(x, self.emb_proj.weight.T.to(dtype=x.dtype))
                logits_proj = F.linear(z, self.tok_emb.weight.to(dtype=x.dtype))
            else:
                logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        if self.logit_delta is not None:
            logits_proj = logits_proj + self.logit_delta(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        losses = F.cross_entropy(logits.float(), targets, reduction="none")

        if loss_mask is None:
            return losses.mean()

        mask = loss_mask.reshape(-1).to(dtype=losses.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom


def main(argv: list[str] | None = None) -> None:
    # This branch is meant for architecture exploration, not artifact accounting yet.
    # We intentionally reuse the incumbent training loop and validation/export stack.
    base.GPT = ExtremeGPT
    base.Block = ExtremeBlock
    base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main(sys.argv[1:])
