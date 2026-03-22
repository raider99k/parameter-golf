from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .export import apply_groupwise_row_scales, quantize_ternary_latent_tensor


def depth_aware_branch_scale(num_layers: int, recurrent_passes: int = 1) -> float:
    effective_depth = max(num_layers * max(recurrent_passes, 1), 1)
    return (2.0 * effective_depth) ** -0.5


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


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 0):
        super().__init__()
        self.group_size = max(int(group_size), 0)
        self.weight_latent = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        nn.init.normal_(self.weight_latent, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def _ternary_weight(self) -> Tensor:
        weight_detached = self.weight_latent.detach()
        ternary_detached, scale = quantize_ternary_latent_tensor(weight_detached, self.group_size)
        weight_ternary = ternary_detached.detach() - weight_detached + self.weight_latent
        return apply_groupwise_row_scales(
            weight_ternary,
            scale.to(dtype=self.weight_latent.dtype),
            self.group_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self._ternary_weight().to(dtype=x.dtype), bias)


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


class LowRankDelta(nn.Module):
    def __init__(self, dim: int, rank: int, init_scale: float):
        super().__init__()
        self.u_proj = nn.Parameter(torch.empty(dim, rank, dtype=torch.float32))
        self.v_proj = nn.Parameter(torch.empty(rank, dim, dtype=torch.float32))
        self.init_scale = float(init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.u_proj, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.v_proj, mean=0.0, std=self.init_scale)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.linear(x, self.v_proj.to(dtype=x.dtype))
        return F.linear(hidden, self.u_proj.to(dtype=x.dtype))


class LowRankLogitDelta(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, init_scale: float):
        super().__init__()
        self.u_proj = nn.Parameter(torch.empty(d_out, rank, dtype=torch.float32))
        self.v_proj = nn.Parameter(torch.empty(rank, d_in, dtype=torch.float32))
        self.init_scale = float(init_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.u_proj, mean=0.0, std=self.init_scale)
        nn.init.normal_(self.v_proj, mean=0.0, std=self.init_scale)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.linear(x, self.v_proj.to(dtype=x.dtype))
        return F.linear(hidden, self.u_proj.to(dtype=x.dtype))


def _build_linear(in_features: int, out_features: int, *, bias: bool, use_bitlinear: bool, group_size: int) -> nn.Module:
    if use_bitlinear:
        return BitLinear(in_features, out_features, bias=bias, group_size=group_size)
    return CastedLinear(in_features, out_features, bias=bias)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        use_bitlinear: bool = False,
        bitlinear_group_size: int = 0,
    ):
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
        self.c_q = _build_linear(dim, dim, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)
        self.c_k = _build_linear(dim, kv_dim, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)
        self.c_v = _build_linear(dim, kv_dim, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)
        self.proj = _build_linear(dim, dim, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_gain_scale: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q_gain = self.q_gain
        if q_gain_scale is not None:
            q_gain = q_gain * q_gain_scale.to(dtype=q_gain.dtype)
        q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]
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
    def __init__(self, dim: int, mlp_mult: int, use_bitlinear: bool = False, bitlinear_group_size: int = 0):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = _build_linear(dim, hidden, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)
        self.proj = _build_linear(hidden, dim, bias=False, use_bitlinear=use_bitlinear, group_size=bitlinear_group_size)

    def forward(self, x: Tensor) -> Tensor:
        hidden = torch.relu(self.fc(x))
        return self.proj(hidden.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        normformer_lite: bool,
        branch_scale_init: float,
        layer_modulation: bool,
        low_rank_deltas: bool,
        delta_rank: int,
        delta_init_scale: float,
    ):
        super().__init__()
        self.layer_modulation = layer_modulation
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn_post_norm = RMSNorm(dim) if normformer_lite else nn.Identity()
        self.mlp_post_norm = RMSNorm(dim) if normformer_lite else nn.Identity()
        self.attn_scale = nn.Parameter(torch.full((dim,), branch_scale_init, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), branch_scale_init, dtype=torch.float32))
        self.attn_delta = (
            LowRankDelta(dim, delta_rank, delta_init_scale)
            if low_rank_deltas and delta_rank > 0 else None
        )
        self.mlp_delta = (
            LowRankDelta(dim, delta_rank, delta_init_scale)
            if low_rank_deltas and delta_rank > 0 else None
        )
        if layer_modulation:
            self.attn_mod_gain = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.attn_mod_bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.mlp_mod_gain = nn.Parameter(torch.ones(dim, dtype=torch.float32))
            self.mlp_mod_bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.register_parameter("attn_mod_gain", None)
            self.register_parameter("attn_mod_bias", None)
            self.register_parameter("mlp_mod_gain", None)
            self.register_parameter("mlp_mod_bias", None)

    def forward(
        self,
        x: Tensor,
        attn: CausalSelfAttention,
        mlp: MLP,
        *,
        pass_attn_gain: Tensor | None = None,
        pass_attn_bias: Tensor | None = None,
        pass_mlp_gain: Tensor | None = None,
        pass_mlp_bias: Tensor | None = None,
        q_gain_scale: Tensor | None = None,
    ) -> Tensor:
        attn_out = attn(self.attn_norm(x), q_gain_scale=q_gain_scale)
        attn_out = self.attn_post_norm(attn_out)
        if self.layer_modulation:
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
        mlp_out = mlp(self.mlp_norm(x))
        mlp_out = self.mlp_post_norm(mlp_out)
        if self.layer_modulation:
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


@dataclass
class ForwardOutput:
    logits: Tensor
    hidden: Tensor


def restore_control_tensors_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            is_control = (
                param.ndim < 2
                or name.endswith(".bias")
                or name.endswith(".weight") and ".norm" in name
                or "pass_" in name
                or "_scale" in name
                or "_gain" in name
                or "_bias" in name
                or "gate" in name
                or "logit_delta" in name
            )
            if is_control and param.dtype != torch.float32:
                param.data = param.data.float()


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
        num_unique_attn: int,
        num_unique_mlp: int,
        normformer_lite: bool,
        depth_aware_init: bool,
        recurrent_passes: int,
        recurrent_gates: bool,
        layer_modulation: bool,
        depth_aware_residuals: bool,
        pass_modulation: bool,
        pass_q_gain: bool,
        low_rank_deltas: bool,
        delta_rank: int,
        delta_init_scale: float,
        logit_delta_rank: int,
        logit_delta_init_scale: float,
        pass_gate_init: float,
        linear_impl: str,
        bitlinear_targets: str,
        bitlinear_group_size: int,
        restore_control_tensors_fp32: bool,
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
        if recurrent_passes <= 0:
            raise ValueError(f"recurrent_passes must be positive, got {recurrent_passes}")
        linear_impl_normalized = str(linear_impl).strip().lower()
        bitlinear_targets_normalized = str(bitlinear_targets).strip().lower()
        if linear_impl_normalized not in {"dense", "bitlinear"}:
            raise ValueError(f"Unsupported linear_impl: {linear_impl!r}")
        if bitlinear_targets_normalized not in {"none", "mlp_only", "attn_and_mlp"}:
            raise ValueError(f"Unsupported bitlinear_targets: {bitlinear_targets!r}")
        if linear_impl_normalized == "dense" and bitlinear_targets_normalized != "none":
            raise ValueError("bitlinear_targets must be 'none' when linear_impl='dense'")

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
        self.num_layers = num_layers
        self.recurrent_passes = recurrent_passes
        self.num_unique_attn = num_layers if num_unique_attn <= 0 else num_unique_attn
        self.num_unique_mlp = num_layers if num_unique_mlp <= 0 else num_unique_mlp
        if not 1 <= self.num_unique_attn <= num_layers:
            raise ValueError(f"num_unique_attn must be in [1, {num_layers}], got {self.num_unique_attn}")
        if not 1 <= self.num_unique_mlp <= num_layers:
            raise ValueError(f"num_unique_mlp must be in [1, {num_layers}], got {self.num_unique_mlp}")

        self.linear_impl = linear_impl_normalized
        self.bitlinear_targets = bitlinear_targets_normalized
        self.bitlinear_group_size = max(int(bitlinear_group_size), 0)
        attn_bitlinear = self.linear_impl == "bitlinear" and self.bitlinear_targets == "attn_and_mlp"
        mlp_bitlinear = self.linear_impl == "bitlinear" and self.bitlinear_targets in {"mlp_only", "attn_and_mlp"}
        self.recurrent_bundle_active = (
            self.recurrent_passes != 1
            or bool(pass_modulation)
            or bool(pass_q_gain)
            or bool(low_rank_deltas)
            or int(logit_delta_rank) > 0
        )
        effective_recurrence = self.recurrent_passes if depth_aware_residuals and self.recurrent_bundle_active else 1
        branch_scale_init = depth_aware_branch_scale(num_layers, effective_recurrence) if depth_aware_init else 1.0

        self.tok_emb = nn.Embedding(vocab_size, embed_dim if use_factor_embed else model_dim)
        self.emb_proj = CastedLinear(embed_dim, model_dim, bias=False) if use_factor_embed else None
        self.attn_bank = nn.ModuleList(
            [
                CausalSelfAttention(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    rope_base,
                    use_bitlinear=attn_bitlinear,
                    bitlinear_group_size=self.bitlinear_group_size,
                )
                for _ in range(self.num_unique_attn)
            ]
        )
        self.mlp_bank = nn.ModuleList(
            [
                MLP(
                    model_dim,
                    mlp_mult,
                    use_bitlinear=mlp_bitlinear,
                    bitlinear_group_size=self.bitlinear_group_size,
                )
                for _ in range(self.num_unique_mlp)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    normformer_lite=normformer_lite,
                    branch_scale_init=branch_scale_init,
                    layer_modulation=bool(layer_modulation) and self.recurrent_bundle_active,
                    low_rank_deltas=bool(low_rank_deltas) and self.recurrent_bundle_active,
                    delta_rank=delta_rank,
                    delta_init_scale=delta_init_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(model_dim)
        self.fast_blocks = nn.ModuleDict()
        for idx in range(num_layers - self.writable_blocks, num_layers):
            if idx < 0:
                continue
            self.fast_blocks[f"block_{idx}"] = LowRankWritableAdapter(model_dim, model_dim, writable_rank, f"block_{idx}")
        self.out_fast = LowRankWritableAdapter(model_dim, model_dim, writable_rank, "out")
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self.use_pass_modulation = bool(pass_modulation) and self.recurrent_bundle_active
        self.use_pass_q_gain = bool(pass_q_gain) and self.recurrent_bundle_active
        self.use_recurrent_gates = bool(recurrent_gates) and self.recurrent_passes > 1
        if self.use_recurrent_gates:
            self.pass_gates = nn.Parameter(
                torch.full((self.recurrent_passes, model_dim), float(pass_gate_init), dtype=torch.float32)
            )
        else:
            self.register_buffer("pass_gates", torch.ones(self.recurrent_passes, model_dim, dtype=torch.float32), persistent=False)
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
            LowRankLogitDelta(model_dim, vocab_size, logit_delta_rank, logit_delta_init_scale)
            if self.recurrent_bundle_active and logit_delta_rank > 0 else None
        )

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        if self.emb_proj is not None:
            nn.init.normal_(self.emb_proj.weight, mean=0.0, std=tied_embed_init_std)
        if restore_control_tensors_fp32:
            restore_control_tensors_to_fp32(self)

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
        if self.logit_delta is not None:
            logits_proj = logits_proj + self.logit_delta(flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if writable_state is not None and "doc_bias" in writable_state:
            logits = logits + writable_state["doc_bias"].to(dtype=logits.dtype)[None, :]
        return logits.view(*hidden.shape[:2], -1)

    def _run_block(
        self,
        x: Tensor,
        block_idx: int,
        *,
        writable_state: dict[str, Tensor] | None = None,
        pass_attn_gain: Tensor | None = None,
        pass_attn_bias: Tensor | None = None,
        pass_mlp_gain: Tensor | None = None,
        pass_mlp_bias: Tensor | None = None,
        q_gain_scale: Tensor | None = None,
    ) -> Tensor:
        block = self.blocks[block_idx]
        attn = self.attn_bank[block_idx % self.num_unique_attn]
        mlp = self.mlp_bank[block_idx % self.num_unique_mlp]
        x = block(
            x,
            attn=attn,
            mlp=mlp,
            pass_attn_gain=pass_attn_gain,
            pass_attn_bias=pass_attn_bias,
            pass_mlp_gain=pass_mlp_gain,
            pass_mlp_bias=pass_mlp_bias,
            q_gain_scale=q_gain_scale,
        )
        fast_key = f"block_{block_idx}"
        if fast_key in self.fast_blocks:
            x = x + self.fast_blocks[fast_key].delta(x, writable_state)
        return x

    def _run_transformer_stack(
        self,
        x: Tensor,
        *,
        writable_state: dict[str, Tensor] | None = None,
        pass_idx: int = 0,
    ) -> Tensor:
        pass_attn_gain = self.pass_attn_gain[pass_idx] if self.pass_attn_gain is not None else None
        pass_attn_bias = self.pass_attn_bias[pass_idx] if self.pass_attn_bias is not None else None
        pass_mlp_gain = self.pass_mlp_gain[pass_idx] if self.pass_mlp_gain is not None else None
        pass_mlp_bias = self.pass_mlp_bias[pass_idx] if self.pass_mlp_bias is not None else None
        q_gain_scale = self.pass_q_gain[pass_idx] if self.pass_q_gain is not None else None
        for idx in range(self.num_layers):
            x = self._run_block(
                x,
                idx,
                writable_state=writable_state,
                pass_attn_gain=pass_attn_gain,
                pass_attn_bias=pass_attn_bias,
                pass_mlp_gain=pass_mlp_gain,
                pass_mlp_bias=pass_mlp_bias,
                q_gain_scale=q_gain_scale,
            )
        return x

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
        if self.recurrent_passes == 1:
            x = self._run_transformer_stack(x, writable_state=writable_state, pass_idx=0)
        else:
            for pass_idx in range(self.recurrent_passes):
                x_in = x
                x_out = self._run_transformer_stack(x, writable_state=writable_state, pass_idx=pass_idx)
                gate = self.pass_gates[pass_idx].to(dtype=x.dtype)[None, None, :]
                x = x_in + gate * (x_out - x_in)
        if self.recurrent_top_block and extra_passes > 0 and self.num_layers > 0:
            for _idx in range(min(extra_passes, self.max_extra_passes)):
                x = self._run_block(x, self.num_layers - 1, writable_state=writable_state)
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
        hidden = self.forward_hidden(input_ids, writable_state=writable_state, extra_passes=extra_passes)
        logits = self._apply_head(hidden, writable_state=writable_state)
        if target_ids is None:
            return ForwardOutput(logits=logits, hidden=hidden)
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
        num_unique_attn=int(model_cfg["num_unique_attn"]),
        num_unique_mlp=int(model_cfg["num_unique_mlp"]),
        normformer_lite=bool(model_cfg["normformer_lite"]),
        depth_aware_init=bool(model_cfg["depth_aware_init"]),
        recurrent_passes=int(model_cfg["recurrent_passes"]),
        recurrent_gates=bool(model_cfg["recurrent_gates"]),
        layer_modulation=bool(model_cfg["layer_modulation"]),
        depth_aware_residuals=bool(model_cfg["depth_aware_residuals"]),
        pass_modulation=bool(model_cfg["pass_modulation"]),
        pass_q_gain=bool(model_cfg["pass_q_gain"]),
        low_rank_deltas=bool(model_cfg["low_rank_deltas"]),
        delta_rank=int(model_cfg["delta_rank"]),
        delta_init_scale=float(model_cfg["delta_init_scale"]),
        logit_delta_rank=int(model_cfg["logit_delta_rank"]),
        logit_delta_init_scale=float(model_cfg["logit_delta_init_scale"]),
        pass_gate_init=float(model_cfg["pass_gate_init"]),
        linear_impl=str(model_cfg["linear_impl"]),
        bitlinear_targets=str(model_cfg["bitlinear_targets"]),
        bitlinear_group_size=int(model_cfg["bitlinear_group_size"]),
        restore_control_tensors_fp32=bool(model_cfg["restore_control_tensors_fp32"]),
        rope_base=float(model_cfg["rope_base"]),
        logit_softcap=float(model_cfg["logit_softcap"]),
        writable_rank=int(model_cfg["writable_rank"]),
        writable_blocks=int(model_cfg["writable_blocks"]),
        recurrent_top_block=bool(model_cfg["recurrent_top_block"]),
        max_extra_passes=int(model_cfg["max_extra_passes"]),
        tied_embed_init_std=float(model_cfg["tied_embed_init_std"]),
    )
