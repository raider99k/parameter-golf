from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class TokenizerAdapter(Protocol):
    vocab_size: int
    bos_id: int
    eos_id: int

    def encode(self, text: str) -> list[int]:
        ...

    def build_byte_accounting(self, device: torch.device) -> "ByteAccounting":
        ...


@dataclass
class ByteAccounting:
    base_bytes: Tensor
    has_leading_space: Tensor
    is_boundary_token: Tensor

    def token_bytes(self, prev_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        token_bytes = self.base_bytes[tgt_ids].to(dtype=torch.int16)
        add_space = self.has_leading_space[tgt_ids] & ~self.is_boundary_token[prev_ids]
        token_bytes += add_space.to(dtype=torch.int16)
        return token_bytes


@dataclass
class SentencePieceTokenizer:
    path: Path

    def __post_init__(self) -> None:
        try:
            import sentencepiece as spm  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentencepiece is required to load SentencePiece tokenizers") from exc
        self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
        self.vocab_size = int(self.sp.vocab_size())
        self.bos_id = int(self.sp.bos_id())
        self.eos_id = int(self.sp.eos_id())

    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text, out_type=int))

    def build_byte_accounting(self, device: torch.device) -> ByteAccounting:
        table_size = self.vocab_size
        base_bytes_np = np.zeros((table_size,), dtype=np.int16)
        has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
        is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
        for token_id in range(table_size):
            if self.sp.is_control(token_id) or self.sp.is_unknown(token_id) or self.sp.is_unused(token_id):
                continue
            is_boundary_token_np[token_id] = False
            if self.sp.is_byte(token_id):
                base_bytes_np[token_id] = 1
                continue
            piece = self.sp.id_to_piece(token_id)
            if piece.startswith("\u2581"):
                has_leading_space_np[token_id] = True
                piece = piece[1:]
            base_bytes_np[token_id] = len(piece.encode("utf-8"))
        return ByteAccounting(
            base_bytes=torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            has_leading_space=torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            is_boundary_token=torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
        )


@dataclass
class PureByteTokenizer:
    path: Path
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    byte_offset: int
    byte_count: int

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + self.byte_count

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint16, copy=False) + self.byte_offset
        return arr.astype(np.int64, copy=False).tolist()

    def build_byte_accounting(self, device: torch.device) -> ByteAccounting:
        table_size = self.vocab_size
        base_bytes = torch.zeros(table_size, dtype=torch.int16, device=device)
        has_leading_space = torch.zeros(table_size, dtype=torch.bool, device=device)
        is_boundary_token = torch.ones(table_size, dtype=torch.bool, device=device)
        byte_ids = torch.arange(self.byte_offset, self.byte_offset + self.byte_count, device=device)
        base_bytes[byte_ids] = 1
        is_boundary_token[byte_ids] = False
        return ByteAccounting(
            base_bytes=base_bytes,
            has_leading_space=has_leading_space,
            is_boundary_token=is_boundary_token,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PureByteTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer_type = payload.get("tokenizer_type")
        if tokenizer_type != "pure_byte":
            raise ValueError(f"Unsupported pure-byte tokenizer payload: {path}")
        config = payload["config"]
        return cls(
            path=Path(path),
            pad_id=int(config["pad_id"]),
            bos_id=int(config["bos_id"]),
            eos_id=int(config["eos_id"]),
            unk_id=int(config["unk_id"]),
            byte_offset=int(config["byte_offset"]),
            byte_count=int(config["byte_count"]),
        )


def load_tokenizer(path: str | Path, kind: str = "auto") -> TokenizerAdapter:
    path = Path(path)
    resolved_kind = kind
    if resolved_kind == "auto":
        if path.suffix == ".model":
            resolved_kind = "sentencepiece"
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("tokenizer_type") == "pure_byte":
                resolved_kind = "pure_byte"
            else:
                raise ValueError(f"Unsupported tokenizer JSON payload: {path}")
        else:
            raise ValueError(f"Could not infer tokenizer kind from: {path}")
    if resolved_kind in {"sentencepiece", "sentencepiece_bpe"}:
        return SentencePieceTokenizer(path)
    if resolved_kind in {"byte", "pure_byte"}:
        return PureByteTokenizer.from_json(path)
    raise ValueError(f"Unsupported tokenizer kind: {kind!r}")
