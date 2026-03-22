from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .adaptation import adapt_writable_state, clone_writable_state, init_writable_state
from .blend import BlendDiagnostics, blend_logprobs
from .experts import OnlineExpert
from .model import HybridGPT


@dataclass
class BlockContext:
    x_full: Tensor
    y_full: Tensor
    score_offset: int
    score_len: int
    score_start: int
    score_end: int
    doc_offset: int
    extra_passes: int = 0


@dataclass
class BlockResult:
    logprobs: Tensor
    targets: Tensor
    current_inputs: Tensor
    score_tokens: Tensor
    diagnostics: BlendDiagnostics
    block_context: BlockContext
    writable_state_before_commit: dict[str, Tensor]


class EvalPolicy:
    name: str
    legal_for_submission: bool
    assumptions: list[str]

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.writable_state: dict[str, Tensor] | None = None

    def reset_document(self, model: HybridGPT) -> None:
        self.writable_state = init_writable_state(model, self.device)

    def prepare_window(self, doc_tokens: Tensor, doc_offset: int, score_start: int, score_end: int) -> BlockContext:
        eval_cfg = self.config["eval"]
        score_len = score_end - score_start
        adapt_tokens = int(eval_cfg["adapt_tokens"])
        if int(eval_cfg.get("window_tokens", 0)) > 0:
            window_start = max(0, score_end - int(eval_cfg["window_tokens"]))
        else:
            window_start = max(0, score_start - adapt_tokens)
        local = doc_tokens[window_start : score_end + 1].to(device=self.device, dtype=torch.int64, non_blocking=True)
        x_full = local[:-1].unsqueeze(0)
        y_full = local[1:].unsqueeze(0)
        score_offset = score_start - window_start
        return BlockContext(
            x_full=x_full,
            y_full=y_full,
            score_offset=score_offset,
            score_len=score_len,
            score_start=score_start,
            score_end=score_end,
            doc_offset=doc_offset,
        )

    def score_block(self, model: HybridGPT, context: BlockContext, experts: list[OnlineExpert]) -> BlockResult:
        raise NotImplementedError

    def commit_scored_tokens(self, model: HybridGPT, result: BlockResult, experts: list[OnlineExpert]) -> None:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "legal_for_submission": self.legal_for_submission,
            "assumptions": self.assumptions,
        }


class StrictCausalPolicy(EvalPolicy):
    name = "strict_causal"
    legal_for_submission = True
    assumptions = [
        "Experts and writable state commit only after the current score block is fully scored.",
        "Ephemeral state is reset at each document boundary.",
        "No prefix adaptation is performed on the current score block.",
    ]

    def score_block(self, model: HybridGPT, context: BlockContext, experts: list[OnlineExpert]) -> BlockResult:
        if self.writable_state is None:
            raise RuntimeError("Policy state is not initialized; call reset_document first")
        with torch.no_grad():
            logits = model.forward_logits(
                context.x_full,
                writable_state=self.writable_state,
                extra_passes=context.extra_passes,
            )
        base_logprobs = torch.log_softmax(logits[0, context.score_offset : context.score_offset + context.score_len], dim=-1)
        current_inputs = context.x_full[0, context.score_offset : context.score_offset + context.score_len]
        expert_logprobs = {
            expert.name: expert.logprob_delta(context.x_full[0], context.score_offset, context.score_len, model.vocab_size).to(self.device)
            for expert in experts
        }
        blended_logprobs, diagnostics = blend_logprobs(base_logprobs, expert_logprobs, current_inputs, self.config)
        targets = context.y_full[0, context.score_offset : context.score_offset + context.score_len]
        return BlockResult(
            logprobs=blended_logprobs,
            targets=targets,
            current_inputs=current_inputs,
            score_tokens=targets,
            diagnostics=diagnostics,
            block_context=context,
            writable_state_before_commit=clone_writable_state(self.writable_state),
        )

    def commit_scored_tokens(self, model: HybridGPT, result: BlockResult, experts: list[OnlineExpert]) -> None:
        if self.writable_state is None:
            raise RuntimeError("Policy state is not initialized; call reset_document first")
        for expert in experts:
            expert.observe(result.score_tokens)
        adapt_cfg = self.config["adaptation"]
        if int(adapt_cfg["strict_commit_steps"]) > 0:
            score_mask = torch.zeros_like(result.block_context.y_full, dtype=torch.float32)
            start = result.block_context.score_offset
            end = start + result.block_context.score_len
            score_mask[:, start:end] = 1.0
            self.writable_state = adapt_writable_state(
                model,
                self.writable_state,
                result.block_context.x_full,
                result.block_context.y_full,
                steps=int(adapt_cfg["strict_commit_steps"]),
                clip=float(adapt_cfg["inner_clip"]),
                doc_bias_lr=float(adapt_cfg["doc_bias_lr"]),
                doc_bias_decay=float(adapt_cfg["doc_bias_decay"]),
                detach_result=True,
                loss_mask=score_mask,
            )


class ExploratoryTTTPolicy(StrictCausalPolicy):
    name = "exploratory_ttt"
    legal_for_submission = False
    assumptions = [
        "Writable state may be adapted on the prefix of the current evaluation window before scoring.",
        "Optional persistence across documents is allowed when configured.",
        "This path is for research only and is not leaderboard-safe by default.",
    ]

    def reset_document(self, model: HybridGPT) -> None:
        persist = bool(self.config["adaptation"]["exploratory_persist_across_docs"])
        if self.writable_state is None or not persist:
            self.writable_state = init_writable_state(model, self.device)

    def score_block(self, model: HybridGPT, context: BlockContext, experts: list[OnlineExpert]) -> BlockResult:
        if self.writable_state is None:
            raise RuntimeError("Policy state is not initialized; call reset_document first")
        prefix_steps = int(self.config["adaptation"]["exploratory_prefix_steps"])
        if prefix_steps > 0 and context.score_offset > 0:
            prefix_mask = torch.zeros_like(context.y_full, dtype=torch.float32)
            prefix_mask[:, : context.score_offset] = 1.0
            self.writable_state = adapt_writable_state(
                model,
                self.writable_state,
                context.x_full,
                context.y_full,
                steps=prefix_steps,
                clip=float(self.config["adaptation"]["inner_clip"]),
                doc_bias_lr=float(self.config["adaptation"]["doc_bias_lr"]),
                doc_bias_decay=float(self.config["adaptation"]["doc_bias_decay"]),
                detach_result=True,
                loss_mask=prefix_mask,
            )
        return super().score_block(model, context, experts)


def build_policy(policy_name: str, config: dict[str, Any], device: torch.device) -> EvalPolicy:
    if policy_name in {"strict", "strict_causal"}:
        return StrictCausalPolicy(config, device)
    if policy_name in {"exploratory", "exploratory_ttt"}:
        return ExploratoryTTTPolicy(config, device)
    raise ValueError(f"Unknown evaluation policy: {policy_name!r}")
