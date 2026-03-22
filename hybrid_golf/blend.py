from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor


@dataclass
class BlendDiagnostics:
    mean_entropy: float
    disagreement: float
    mean_weights: dict[str, float]


def _safe_log_softmax(logits: Tensor) -> Tensor:
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def build_gate_weights(
    base_logprobs: Tensor,
    expert_logprobs: dict[str, Tensor],
    current_inputs: Tensor,
    config: dict[str, Any],
) -> tuple[dict[str, Tensor], BlendDiagnostics]:
    gate_cfg = config["experts"]["gate"]
    probs = base_logprobs.exp()
    entropy = -(probs * base_logprobs).sum(dim=-1)
    entropy_norm = entropy / max(math.log(max(base_logprobs.size(-1), 2)), 1.0)
    current_inputs = current_inputs.to(dtype=torch.int64)
    repeat = torch.zeros_like(entropy)
    seen: set[int] = set()
    for idx in range(current_inputs.numel()):
        token = int(current_inputs[idx].item())
        repeat[idx] = 1.0 if token in seen else 0.0
        seen.add(token)
    weights: dict[str, Tensor] = {}
    mean_weights: dict[str, float] = {}
    disagreements = []
    base_top = base_logprobs.argmax(dim=-1)
    max_weight = float(gate_cfg.get("max_weight", 1.5))
    for name, values in expert_logprobs.items():
        expert_probs = _safe_log_softmax(values).exp()
        confidence = expert_probs.max(dim=-1).values
        expert_top = values.argmax(dim=-1)
        disagreements.append((expert_top != base_top).float())
        base = float(gate_cfg["base"].get(name, 0.0))
        entropy_scale = float(gate_cfg["entropy_scale"].get(name, 0.0))
        repeat_scale = float(gate_cfg["repeat_scale"].get(name, 0.0))
        confidence_scale = float(gate_cfg["confidence_scale"].get(name, 0.0))
        weight = base + entropy_scale * entropy_norm + repeat_scale * repeat + confidence_scale * confidence
        weight = weight.clamp(min=0.0, max=max_weight)
        weights[name] = weight
        mean_weights[name] = float(weight.mean().item())
    disagreement = float(torch.stack(disagreements).mean().item()) if disagreements else 0.0
    return weights, BlendDiagnostics(
        mean_entropy=float(entropy.mean().item()),
        disagreement=disagreement,
        mean_weights=mean_weights,
    )


def blend_logprobs(
    base_logprobs: Tensor,
    expert_logprobs: dict[str, Tensor],
    current_inputs: Tensor,
    config: dict[str, Any],
) -> tuple[Tensor, BlendDiagnostics]:
    if not expert_logprobs:
        entropy = -(base_logprobs.exp() * base_logprobs).sum(dim=-1).mean().item()
        return base_logprobs, BlendDiagnostics(mean_entropy=float(entropy), disagreement=0.0, mean_weights={})
    weights, diagnostics = build_gate_weights(base_logprobs, expert_logprobs, current_inputs, config)
    combined = base_logprobs.clone()
    for name, values in expert_logprobs.items():
        combined = combined + weights[name][:, None] * values.to(dtype=combined.dtype, device=combined.device)
    return _safe_log_softmax(combined), diagnostics
