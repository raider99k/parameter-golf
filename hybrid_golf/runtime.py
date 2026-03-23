from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUBMISSION_CODE_PATTERNS = (
    "hybrid_golf/*.py",
    "scripts/hg_eval.py",
)


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


def configure_cuda_fast_math(device: torch.device) -> dict[str, bool]:
    if device.type != "cuda":
        return {
            "cudnn": False,
            "flash": False,
            "mem_efficient": False,
            "math": False,
            "tf32": False,
        }
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    return {
        "cudnn": False,
        "flash": True,
        "mem_efficient": False,
        "math": False,
        "tf32": True,
    }


def build_run_dir(config: dict[str, Any]) -> Path:
    root = Path(config["run"]["output_root"])
    run_id = str(config["run"]["id"])
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_submission_code_paths() -> list[Path]:
    root = repo_root()
    paths: set[Path] = set()
    for pattern in SUBMISSION_CODE_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths)


def count_submission_code_bytes() -> tuple[int, list[str]]:
    root = repo_root()
    paths = resolve_submission_code_paths()
    return (
        sum(int(path.stat().st_size) for path in paths),
        [str(path.relative_to(root)).replace("\\", "/") for path in paths],
    )


def build_submission_size_metrics(
    artifact_bytes: int,
    budget_bytes: int,
    budget_mode: str,
) -> dict[str, Any]:
    counted_code_bytes, counted_code_files = count_submission_code_bytes()
    submission_total_bytes = int(artifact_bytes) + counted_code_bytes
    budget_mode_normalized = str(budget_mode).strip().lower()
    if budget_mode_normalized == "artifact_only":
        budgeted_bytes = int(artifact_bytes)
    elif budget_mode_normalized == "submission_total":
        budgeted_bytes = submission_total_bytes
    else:
        raise ValueError(f"Unsupported budget mode: {budget_mode!r}")
    within_budget = budgeted_bytes <= int(budget_bytes)
    return {
        "model_artifact_bytes": int(artifact_bytes),
        "counted_code_bytes": counted_code_bytes,
        "counted_code_files": counted_code_files,
        "submission_total_bytes": submission_total_bytes,
        "budget_bytes": int(budget_bytes),
        "budget_mode": budget_mode_normalized,
        "budgeted_bytes": budgeted_bytes,
        "within_budget": within_budget,
        "over_budget": not within_budget,
    }


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
