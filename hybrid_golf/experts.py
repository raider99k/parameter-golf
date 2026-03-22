from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _hash_context(context: tuple[int, ...], num_buckets: int) -> int:
    return hash(context) % max(num_buckets, 1)


class OnlineExpert:
    name: str

    def reset(self) -> None:
        raise NotImplementedError

    def observe(self, tokens: Tensor) -> None:
        raise NotImplementedError

    def logprob_delta(self, window_tokens: Tensor, score_offset: int, score_len: int, vocab_size: int) -> Tensor:
        raise NotImplementedError


@dataclass
class HashedNGramExpert(OnlineExpert):
    order: int
    vocab_size: int
    num_buckets: int
    alpha: float
    name: str = "ngram"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.history: list[int] = []

    def observe(self, tokens: Tensor) -> None:
        ints = [int(x) for x in tokens.detach().cpu().tolist()]
        combined = self.history + ints
        n = self.order
        for idx in range(n - 1, len(combined)):
            ctx = tuple(combined[idx - (n - 1) : idx])
            bucket = _hash_context(ctx, self.num_buckets)
            self.counts[bucket][combined[idx]] += 1
        self.history = combined[-(n - 1) :] if n > 1 else []

    def logprob_delta(self, window_tokens: Tensor, score_offset: int, score_len: int, vocab_size: int) -> Tensor:
        result = torch.zeros(score_len, vocab_size, dtype=torch.float32, device=window_tokens.device)
        if self.order <= 1:
            return result
        prefix = self.history + [int(x) for x in window_tokens[:score_offset].detach().cpu().tolist()]
        if len(prefix) < self.order - 1:
            return result
        ctx = tuple(prefix[-(self.order - 1) :])
        bucket = _hash_context(ctx, self.num_buckets)
        counter = self.counts.get(bucket)
        if not counter:
            return result
        counts = torch.full((vocab_size,), self.alpha, dtype=torch.float32, device=window_tokens.device)
        for token_id, count in counter.items():
            if 0 <= token_id < vocab_size:
                counts[token_id] += float(count)
        probs = counts / counts.sum().clamp_min(1e-12)
        centered = torch.log(probs.clamp_min(1e-12)) + torch.log(torch.tensor(float(vocab_size), device=probs.device))
        result[:] = centered[None, :]
        return result


@dataclass
class RecentPointerExpert(OnlineExpert):
    vocab_size: int
    window: int
    alpha: float
    name: str = "pointer"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.history: list[int] = []

    def observe(self, tokens: Tensor) -> None:
        self.history.extend(int(x) for x in tokens.detach().cpu().tolist())
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

    def logprob_delta(self, window_tokens: Tensor, score_offset: int, score_len: int, vocab_size: int) -> Tensor:
        result = torch.zeros(score_len, vocab_size, dtype=torch.float32, device=window_tokens.device)
        context = self.history + [int(x) for x in window_tokens[:score_offset].detach().cpu().tolist()]
        if not context:
            return result
        recent = context[-self.window :]
        counts = torch.full((vocab_size,), self.alpha, dtype=torch.float32, device=window_tokens.device)
        for token in recent:
            if 0 <= token < vocab_size:
                counts[token] += 1.0
        probs = counts / counts.sum().clamp_min(1e-12)
        centered = torch.log(probs.clamp_min(1e-12)) + torch.log(torch.tensor(float(vocab_size), device=probs.device))
        result[:] = centered[None, :]
        return result


@dataclass
class DocumentBiasExpert(OnlineExpert):
    vocab_size: int
    alpha: float
    name: str = "doc_bias"

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts = torch.zeros(self.vocab_size, dtype=torch.float32)

    def observe(self, tokens: Tensor) -> None:
        values = tokens.detach().cpu().to(dtype=torch.int64)
        bincount = torch.bincount(values, minlength=self.vocab_size).to(dtype=torch.float32)
        self.counts += bincount

    def logprob_delta(self, window_tokens: Tensor, score_offset: int, score_len: int, vocab_size: int) -> Tensor:
        del window_tokens, score_offset
        if float(self.counts.sum().item()) <= 0:
            return torch.zeros(score_len, vocab_size, dtype=torch.float32)
        counts = self.counts.to(dtype=torch.float32) + self.alpha
        probs = counts / counts.sum().clamp_min(1e-12)
        centered = torch.log(probs.clamp_min(1e-12)) + torch.log(torch.tensor(float(vocab_size)))
        return centered[None, :].repeat(score_len, 1)


def build_experts(config: dict[str, Any], vocab_size: int) -> list[OnlineExpert]:
    expert_cfg = config["experts"]
    experts: list[OnlineExpert] = []
    if expert_cfg.get("enable_ngram", False):
        experts.append(
            HashedNGramExpert(
                order=int(expert_cfg["ngram_order"]),
                vocab_size=vocab_size,
                num_buckets=int(expert_cfg["ngram_buckets"]),
                alpha=float(expert_cfg["ngram_alpha"]),
            )
        )
    if expert_cfg.get("enable_pointer", False):
        experts.append(
            RecentPointerExpert(
                vocab_size=vocab_size,
                window=int(expert_cfg["pointer_window"]),
                alpha=float(expert_cfg["pointer_alpha"]),
            )
        )
    if expert_cfg.get("enable_doc_bias", False):
        experts.append(
            DocumentBiasExpert(
                vocab_size=vocab_size,
                alpha=float(expert_cfg["doc_bias_alpha"]),
            )
        )
    return experts
