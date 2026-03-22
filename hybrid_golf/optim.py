from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def _muon_work_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=_muon_work_dtype(G.device))
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            backend_steps = int(group["backend_steps"])
            nesterov = bool(group["nesterov"])
            weight_decay = float(group.get("weight_decay", 0.0))
            update_dtype = _muon_work_dtype(params[0].device)
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=update_dtype)
            cursor = 0
            for index, param in enumerate(params):
                if index % world_size == rank:
                    grad = param.grad
                    if grad is None:
                        cursor += param.numel()
                        continue
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    grad = zeropower_via_newtonschulz5(grad, steps=backend_steps)
                    grad *= max(1.0, grad.size(0) / max(grad.size(1), 1)) ** 0.5
                    updates_flat[cursor : cursor + param.numel()] = grad.reshape(-1).to(dtype=update_dtype)
                cursor += param.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            cursor = 0
            for param in params:
                update = updates_flat[cursor : cursor + param.numel()].view_as(param).to(dtype=param.dtype)
                if weight_decay > 0.0:
                    param.data.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)
                cursor += param.numel()
        return loss
