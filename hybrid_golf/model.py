from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), weight=self.weight.to(dtype=x.dtype), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class LowRankWritableAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, key: str, gate_init: float = 0.05):
        super().__init__()
        self.key = key
        self.rank = rank
        self.u_basis = nn.Parameter(torch.empty(d_out, rank))
        self.v_basis = nn.Parameter(torch.empty(rank, d_in))
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.log_lr = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.logit_decay = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.u_basis, mean=0.0, std=0.02)
        nn.init.normal_(self.v_basis, mean=0.0, std=0.02)

    def lr(self) -> Tensor:
        return F.softplus(self.log_lr)

    def decay(self) -> Tensor:
        return torch.sigmoid(self.logit_decay)

    def delta(self, x: Tensor, writable_state: dict[str, Tensor] | None) -> Tensor:
        dummy = (
            0.0 * self.log_lr
            + 0.0 * self.logit_decay
            + 0.0 * self.gate
            + 0.0 * self.u_basis.sum()
            + 0.0 * self.v_basis.sum()
        ).to(dtype=x.dtype)
        if writable_state is None or self.key not in writable_state:
            return torch.zeros((*x.shape[:-1], self.u_basis.shape[0]), device=x.device, dtype=x.dtype) + dummy
        matrix = writable_state[self.key].to(dtype=x.dtype)
        hidden = F.linear(x, self.v_basis.to(dtype=x.dtype))
        hidden = F.linear(hidden, matrix)
        hidden = F.linear(hidden, self.u_basis.to(dtype=x.dtype))
        return self.gate.to(dtype=x.dtype) * hidden + dummy


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.relu(self.fc(x))
        return self.proj(hidden.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


@dataclass
class ForwardOutput:
    logits: Tensor
    hidden: Tensor


class HybridGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        use_factor_embed: bool,
        embed_dim: int,
        rope_base: float,
        logit_softcap: float,
        writable_rank: int,
        writable_blocks: int,
        recurrent_top_block: bool,
        max_extra_passes: int,
        tied_embed_init_std: float,
    ):
        super().__init__()
        if logit_softcap <= 0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.use_factor_embed = use_factor_embed
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.logit_softcap = logit_softcap
        self.max_extra_passes = max_extra_passes
        self.recurrent_top_block = recurrent_top_block
        self.writable_rank = writable_rank
        self.writable_blocks = min(max(writable_blocks, 0), num_layers)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim if use_factor_embed else model_dim)
        self.emb_proj = CastedLinear(embed_dim, model_dim, bias=False) if use_factor_embed else None
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(model_dim)
        self.fast_blocks = nn.ModuleDict()
        for idx in range(num_layers - self.writable_blocks, num_layers):
            if idx < 0:
                continue
            self.fast_blocks[f"block_{idx}"] = LowRankWritableAdapter(model_dim, model_dim, writable_rank, f"block_{idx}")
        self.out_fast = LowRankWritableAdapter(model_dim, model_dim, writable_rank, "out")
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        if self.emb_proj is not None:
            nn.init.normal_(self.emb_proj.weight, mean=0.0, std=tied_embed_init_std)

    def writable_state_keys(self) -> tuple[str, ...]:
        keys = [key for key in self.fast_blocks]
        keys.sort()
        return tuple(keys + ["out", "doc_bias"])

    def _apply_head(self, hidden: Tensor, writable_state: dict[str, Tensor] | None = None) -> Tensor:
        hidden_out = self.final_norm(hidden)
        hidden_out = hidden_out + self.out_fast.delta(hidden_out, writable_state)
        flat = hidden_out.reshape(-1, hidden_out.size(-1))
        if self.tie_embeddings:
            if self.emb_proj is not None:
                z = F.linear(flat, self.emb_proj.weight.T.to(dtype=flat.dtype))
                logits_proj = F.linear(z, self.tok_emb.weight.to(dtype=flat.dtype))
            else:
                logits_proj = F.linear(flat, self.tok_emb.weight.to(dtype=flat.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if writable_state is not None and "doc_bias" in writable_state:
            logits = logits + writable_state["doc_bias"].to(dtype=logits.dtype)[None, :]
        return logits.view(*hidden.shape[:2], -1)

    def forward_hidden(
        self,
        input_ids: Tensor,
        writable_state: dict[str, Tensor] | None = None,
        extra_passes: int = 0,
    ) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.emb_proj is not None:
            x = self.emb_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            fast_key = f"block_{idx}"
            if fast_key in self.fast_blocks:
                x = x + self.fast_blocks[fast_key].delta(x, writable_state)
        if self.recurrent_top_block and extra_passes > 0 and len(self.blocks) > 0:
            top_block = self.blocks[-1]
            fast_key = f"block_{len(self.blocks) - 1}"
            adapter = self.fast_blocks[fast_key] if fast_key in self.fast_blocks else None
            for _idx in range(min(extra_passes, self.max_extra_passes)):
                x = top_block(x)
                if adapter is not None:
                    x = x + adapter.delta(x, writable_state)
        return x

    def forward_logits(
        self,
        input_ids: Tensor,
        writable_state: dict[str, Tensor] | None = None,
        extra_passes: int = 0,
    ) -> Tensor:
        hidden = self.forward_hidden(input_ids, writable_state=writable_state, extra_passes=extra_passes)
        return self._apply_head(hidden, writable_state=writable_state)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        writable_state: dict[str, Tensor] | None = None,
        loss_mask: Tensor | None = None,
        extra_passes: int = 0,
    ) -> Tensor | ForwardOutput:
        logits = self.forward_logits(input_ids, writable_state=writable_state, extra_passes=extra_passes)
        if target_ids is None:
            return ForwardOutput(logits=logits, hidden=torch.empty(0, device=logits.device))
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="none",
        )
        if loss_mask is None:
            return losses.mean()
        mask = loss_mask.reshape(-1).to(dtype=losses.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom


def build_model(config: dict[str, Any]) -> HybridGPT:
    model_cfg = config["model"]
    return HybridGPT(
        vocab_size=int(model_cfg["vocab_size"]),
        num_layers=int(model_cfg["num_layers"]),
        model_dim=int(model_cfg["model_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_kv_heads=int(model_cfg["num_kv_heads"]),
        mlp_mult=int(model_cfg["mlp_mult"]),
        tie_embeddings=bool(model_cfg["tie_embeddings"]),
        use_factor_embed=bool(model_cfg["use_factor_embed"]),
        embed_dim=int(model_cfg["embed_dim"]),
        rope_base=float(model_cfg["rope_base"]),
        logit_softcap=float(model_cfg["logit_softcap"]),
        writable_rank=int(model_cfg["writable_rank"]),
        writable_blocks=int(model_cfg["writable_blocks"]),
        recurrent_top_block=bool(model_cfg["recurrent_top_block"]),
        max_extra_passes=int(model_cfg["max_extra_passes"]),
        tied_embed_init_std=float(model_cfg["tied_embed_init_std"]),
    )
