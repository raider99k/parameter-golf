from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from hybrid_golf.data import write_data_shard


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_pure_byte_tokenizer(path: Path) -> None:
    payload = {
        "tokenizer_type": "pure_byte",
        "config": {
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "byte_offset": 4,
            "byte_count": 256,
        },
        "vocab_size": 260,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_config(path: Path, train_glob: str, val_glob: str, tokenizer_path: str, output_root: str) -> None:
    config = {
        "run": {"id": "cli_smoke", "output_root": output_root, "device": "cpu"},
        "data": {"train_glob": train_glob, "val_glob": val_glob, "val_tokens_limit": 0},
        "tokenizer": {"path": tokenizer_path, "kind": "pure_byte", "bos_id": 1, "eos_id": 2},
        "model": {
            "vocab_size": 260,
            "num_layers": 2,
            "model_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_mult": 2,
            "tie_embeddings": True,
            "use_factor_embed": False,
            "embed_dim": 32,
            "writable_rank": 2,
            "writable_blocks": 1,
            "recurrent_top_block": True
        },
        "train": {
            "iterations": 2,
            "batch_tokens": 64,
            "seq_len": 16,
            "lr": 0.001,
            "train_log_every": 1,
            "val_every": 0,
            "max_wallclock_seconds": 0
        },
        "experts": {"enable_ngram": True, "enable_pointer": True, "enable_doc_bias": True},
        "adaptation": {
            "meta_enabled": False,
            "strict_commit_steps": 1,
            "exploratory_prefix_steps": 1,
            "exploratory_persist_across_docs": False
        },
        "eval": {"policy": "strict_causal", "score_tokens": 8, "adapt_tokens": 16, "documentwise": True}
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def run_cmd(args: list[str]) -> dict:
    proc = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    stdout = proc.stdout.strip().splitlines()
    return json.loads(stdout[-1] if len(stdout) == 1 else "\n".join(stdout))


def test_cli_train_and_eval_smoke(tmp_path):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_tokens = torch.tensor([1, 10, 11, 12, 13, 14, 15, 16] * 16, dtype=torch.int64)
    val_tokens = torch.tensor([1, 10, 11, 12, 13, 14, 1, 10, 11, 12, 13, 14], dtype=torch.int64)
    write_data_shard(train_dir / "fineweb_train_000000.bin", train_tokens)
    write_data_shard(val_dir / "fineweb_val_000000.bin", val_tokens)
    tokenizer_path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(tokenizer_path)
    config_path = tmp_path / "config.json"
    output_root = tmp_path / "outputs"
    write_config(
        config_path,
        str(train_dir / "fineweb_train_*.bin"),
        str(val_dir / "fineweb_val_*.bin"),
        str(tokenizer_path),
        str(output_root),
    )
    train_result = run_cmd([sys.executable, "scripts/hg_train.py", "--config", str(config_path)])
    checkpoint = train_result["checkpoint"]
    assert Path(checkpoint).is_file()
    strict_eval = run_cmd([sys.executable, "scripts/hg_eval.py", "--config", str(config_path), "--checkpoint", checkpoint, "--policy", "strict"])
    exploratory_eval = run_cmd([sys.executable, "scripts/hg_eval.py", "--config", str(config_path), "--checkpoint", checkpoint, "--policy", "exploratory"])
    assert "val_bpb" in strict_eval
    assert exploratory_eval["policy"]["legal_for_submission"] is False
