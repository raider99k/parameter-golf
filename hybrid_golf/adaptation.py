from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import HybridGPT, LowRankWritableAdapter


def init_writable_state(model: HybridGPT, device: torch.device) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for module in model.modules():
        if isinstance(module, LowRankWritableAdapter):
            state[module.key] = torch.zeros(module.rank, module.rank, device=device, dtype=torch.float32)
    state["doc_bias"] = torch.zeros(model.vocab_size, device=device, dtype=torch.float32)
    return state


def clone_writable_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.clone() for key, value in state.items()}


def detach_writable_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach() for key, value in state.items()}


def iter_writable_adapters(module: nn.Module) -> list[LowRankWritableAdapter]:
    return [child for child in module.modules() if isinstance(child, LowRankWritableAdapter)]


def adapt_writable_state(
    model: HybridGPT,
    writable_state: dict[str, Tensor],
    x_prefix: Tensor,
    y_prefix: Tensor,
    steps: int,
    clip: float,
    doc_bias_lr: float,
    doc_bias_decay: float,
    create_graph: bool = False,
    detach_result: bool = True,
    loss_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    state = {key: value.clone().detach().requires_grad_(True) for key, value in writable_state.items()}
    if steps <= 0:
        return detach_writable_state(state) if detach_result else state
    adapters = iter_writable_adapters(model)
    adapter_by_key = {adapter.key: adapter for adapter in adapters}
    adapter_keys = [adapter.key for adapter in adapters if adapter.key in state]
    for _step in range(steps):
        loss = model(x_prefix, y_prefix, writable_state=state, loss_mask=loss_mask)
        grad_targets = [state[key] for key in adapter_keys]
        if "doc_bias" in state:
            grad_targets.append(state["doc_bias"])
        grads = torch.autograd.grad(loss, grad_targets, create_graph=create_graph, allow_unused=False)
        next_state: dict[str, Tensor] = {}
        adapter_grads = grads[: len(adapter_keys)]
        for key, grad in zip(adapter_keys, adapter_grads, strict=True):
            adapter = adapter_by_key[key]
            grad_norm = grad.norm()
            if clip > 0:
                grad = grad * (clip / grad_norm.clamp_min(clip))
            next_state[key] = adapter.decay() * state[key] - adapter.lr() * grad
        if "doc_bias" in state:
            doc_grad = grads[-1]
            grad_norm = doc_grad.norm()
            if clip > 0:
                doc_grad = doc_grad * (clip / grad_norm.clamp_min(clip))
            next_state["doc_bias"] = doc_bias_decay * state["doc_bias"] - doc_bias_lr * doc_grad
        state = next_state
    return detach_writable_state(state) if detach_result else state
