from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def write_data_shard(path: Path, tokens: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    toks = tokens.detach().cpu().to(dtype=torch.int64).numpy()
    if toks.size >= 2**31:
        raise ValueError("Too many tokens for test shard")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(toks.size)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())


def find_documents_by_bos(tokens: Tensor, bos_id: int, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_positions = (tokens == bos_id).nonzero(as_tuple=True)[0].cpu().numpy()
    docs: list[tuple[int, int]] = []
    if len(bos_positions) == 0:
        if tokens.numel() >= 2:
            return [(0, int(tokens.numel()))]
        return []
    for idx in range(len(bos_positions)):
        start = int(bos_positions[idx])
        end = int(bos_positions[idx + 1]) if idx + 1 < len(bos_positions) else int(tokens.numel())
        if include_next_bos and idx + 1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def resolve_files(pattern: str) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files


def load_validation_tokens(pattern: str, limit: int = 0) -> Tensor:
    tokens = torch.cat([load_data_shard(file) for file in resolve_files(pattern)]).contiguous()
    if limit > 0 and tokens.numel() > limit:
        tokens = tokens[:limit]
    if tokens.numel() <= 1:
        raise ValueError("Validation split is too short")
    return tokens


class TokenStream:
    def __init__(self, pattern: str):
        self.files = resolve_files(pattern)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, int(avail))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def take_same_file(self, n: int) -> Tensor:
        while self.tokens.numel() < n or (self.tokens.numel() - self.pos) < n:
            self._advance_file()
        chunk = self.tokens[self.pos : self.pos + n]
        self.pos += n
        return chunk


@dataclass
class TokenBatcher:
    pattern: str
    device: torch.device
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        self.stream = TokenStream(self.pattern)

    def next_batch(self, global_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // self.world_size
        if local_tokens % seq_len != 0:
            raise ValueError(f"batch_tokens per rank must divide seq_len, got {local_tokens} and {seq_len}")
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def next_local_span(self, local_tokens_plus_one: int, same_file: bool = False) -> Tensor:
        take_fn = self.stream.take_same_file if same_file else self.stream.take
        chunk = take_fn(local_tokens_plus_one * self.world_size)
        start = self.rank * local_tokens_plus_one
        local = chunk[start : start + local_tokens_plus_one].to(dtype=torch.int64)
        return local.to(self.device, non_blocking=True)
