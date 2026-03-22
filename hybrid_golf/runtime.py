from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_autocast_dtype(device: torch.device, dtype_name: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if dtype_name == "auto":
        major, _minor = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return None
    raise ValueError(f"Unsupported dtype setting: {dtype_name!r}")


def build_run_dir(config: dict[str, Any]) -> Path:
    root = Path(config["run"]["output_root"])
    run_id = str(config["run"]["id"])
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RunLogger:
    run_dir: Path

    def __post_init__(self) -> None:
        self.text_log_path = self.run_dir / "train.log"
        self.events_path = self.run_dir / "events.jsonl"

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with self.text_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(self, kind: str, **payload: Any) -> None:
        item = {"kind": kind, "time": time.time(), **payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
