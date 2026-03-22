from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from hybrid_golf.config import deep_merge
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


def write_config(
    path: Path,
    train_glob: str,
    val_glob: str,
    tokenizer_path: str,
    output_root: str,
    overrides: dict | None = None,
) -> None:
    config = {
        "run": {"id": "cli_smoke", "output_root": output_root, "device": "cpu", "stage": "proxy"},
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
            "use_factor_embed": True,
            "embed_dim": 16,
            "num_unique_attn": 1,
            "num_unique_mlp": 1,
            "normformer_lite": True,
            "depth_aware_init": True,
            "writable_rank": 2,
            "writable_blocks": 1,
            "recurrent_top_block": True,
        },
        "train": {
            "iterations": 2,
            "batch_tokens": 64,
            "seq_len": 16,
            "lr": 0.001,
            "optimizer": "muon_split",
            "embed_lr": 0.001,
            "head_lr": 0.001,
            "matrix_lr": 0.001,
            "scalar_lr": 0.001,
            "train_log_every": 1,
            "val_every": 0,
            "max_wallclock_seconds": 0,
            "swa_enabled": True,
            "swa_start_frac": 0.5,
            "swa_every": 1,
            "projection_qat_enabled": True,
            "qat_start_frac": 0.0,
            "qat_every_steps": 1,
            "compression_reg_weight": 0.01,
            "compression_reg_start_frac": 0.0,
            "compression_reg_every_steps": 1,
        },
        "experts": {"enable_ngram": True, "enable_pointer": True, "enable_doc_bias": True},
        "adaptation": {
            "meta_enabled": False,
            "strict_commit_steps": 1,
            "exploratory_prefix_steps": 1,
            "exploratory_persist_across_docs": False,
        },
        "eval": {"policy": "strict_causal", "score_tokens": 8, "adapt_tokens": 16, "documentwise": True},
        "export": {
            "write_after_train": True,
            "artifact_budget_bytes": 1_048_576,
            "budget_mode": "submission_total",
            "quant_scheme": "mixed_v1",
            "keep_float_policy": "small_and_control",
        },
    }
    if overrides:
        config = deep_merge(config, overrides)
    path.write_text(json.dumps(config), encoding="utf-8")


def write_eval_config_with_wrong_model_dims(
    path: Path,
    train_glob: str,
    val_glob: str,
    tokenizer_path: str,
    output_root: str,
    overrides: dict | None = None,
) -> None:
    config = {
        "run": {"id": "cli_smoke_eval_mismatch", "output_root": output_root, "device": "cpu"},
        "data": {"train_glob": train_glob, "val_glob": val_glob, "val_tokens_limit": 6},
        "tokenizer": {"path": tokenizer_path, "kind": "pure_byte", "bos_id": 1, "eos_id": 2},
        "model": {
            "vocab_size": 260,
            "num_layers": 4,
            "model_dim": 64,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_mult": 2,
            "tie_embeddings": True,
            "use_factor_embed": False,
            "embed_dim": 64,
            "writable_rank": 4,
            "writable_blocks": 2,
            "recurrent_top_block": False,
        },
        "train": {"enabled": False},
        "experts": {"enable_ngram": False, "enable_pointer": False, "enable_doc_bias": False},
        "adaptation": {"meta_enabled": False, "strict_commit_steps": 1},
        "eval": {"policy": "strict_causal", "score_tokens": 4, "adapt_tokens": 8, "documentwise": False},
        "export": {"quant_scheme": "mixed_v1"},
    }
    if overrides:
        config = deep_merge(config, overrides)
    path.write_text(json.dumps(config), encoding="utf-8")


def run_cmd(args: list[str]) -> dict:
    proc = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    stdout = proc.stdout.strip().splitlines()
    return json.loads(stdout[-1])


def build_tiny_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_tokens = torch.tensor([1, 10, 11, 12, 13, 14, 15, 16] * 16, dtype=torch.int64)
    val_tokens = torch.tensor([1, 10, 11, 12, 13, 14, 1, 10, 11, 12, 13, 14], dtype=torch.int64)
    write_data_shard(train_dir / "fineweb_train_000000.bin", train_tokens)
    write_data_shard(val_dir / "fineweb_val_000000.bin", val_tokens)
    tokenizer_path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(tokenizer_path)
    return train_dir, val_dir, tokenizer_path, tmp_path / "outputs"


def test_cli_train_and_eval_recurrent_smoke(tmp_path):
    train_dir, val_dir, tokenizer_path, output_root = build_tiny_dataset(tmp_path)
    config_path = tmp_path / "config_recurrent.json"
    write_config(
        config_path,
        str(train_dir / "fineweb_train_*.bin"),
        str(val_dir / "fineweb_val_*.bin"),
        str(tokenizer_path),
        str(output_root),
        overrides={
            "run": {"id": "cli_smoke_recurrent"},
            "model": {
                "recurrent_passes": 2,
                "recurrent_gates": True,
                "layer_modulation": True,
                "depth_aware_residuals": True,
            },
        },
    )
    eval_config_path = tmp_path / "eval_config_recurrent.json"
    write_eval_config_with_wrong_model_dims(
        eval_config_path,
        str(train_dir / "fineweb_train_*.bin"),
        str(val_dir / "fineweb_val_*.bin"),
        str(tokenizer_path),
        str(output_root),
    )
    train_result = run_cmd([sys.executable, "scripts/hg_train.py", "--config", str(config_path)])
    checkpoint = train_result["checkpoint"]
    strict_eval = run_cmd([
        sys.executable,
        "scripts/hg_eval.py",
        "--config",
        str(eval_config_path),
        "--checkpoint",
        checkpoint,
        "--policy",
        "strict",
        "--export-roundtrip",
    ])
    exploratory_eval = run_cmd([
        sys.executable,
        "scripts/hg_eval.py",
        "--config",
        str(config_path),
        "--checkpoint",
        checkpoint,
        "--policy",
        "exploratory",
    ])
    assert Path(checkpoint).is_file()
    assert train_result["export"]["artifact_bytes"] > 0
    assert strict_eval["base"]["gate_summary"] == {}
    assert strict_eval["base"]["token_count"] == 5
    assert strict_eval["roundtrip"]["artifact_bytes"] > 0
    assert strict_eval["roundtrip"]["submission_total_bytes"] >= strict_eval["roundtrip"]["artifact_bytes"]
    assert exploratory_eval["policy"]["legal_for_submission"] is False


def test_cli_train_and_eval_bitlinear_smoke(tmp_path):
    train_dir, val_dir, tokenizer_path, output_root = build_tiny_dataset(tmp_path)
    config_path = tmp_path / "config_bitlinear.json"
    write_config(
        config_path,
        str(train_dir / "fineweb_train_*.bin"),
        str(val_dir / "fineweb_val_*.bin"),
        str(tokenizer_path),
        str(output_root),
        overrides={
            "run": {"id": "cli_smoke_bitlinear"},
            "model": {
                "linear_impl": "bitlinear",
                "bitlinear_targets": "mlp_only",
                "bitlinear_group_size": 64,
            },
            "train": {
                "projection_qat_enabled": False,
                "compression_reg_weight": 0.0,
            },
            "export": {"quant_scheme": "mixed_v2"},
        },
    )
    eval_config_path = tmp_path / "eval_config_bitlinear.json"
    write_eval_config_with_wrong_model_dims(
        eval_config_path,
        str(train_dir / "fineweb_train_*.bin"),
        str(val_dir / "fineweb_val_*.bin"),
        str(tokenizer_path),
        str(output_root),
        overrides={"export": {"quant_scheme": "mixed_v1"}},
    )
    train_result = run_cmd([sys.executable, "scripts/hg_train.py", "--config", str(config_path)])
    checkpoint = train_result["checkpoint"]
    strict_eval = run_cmd([
        sys.executable,
        "scripts/hg_eval.py",
        "--config",
        str(eval_config_path),
        "--checkpoint",
        checkpoint,
        "--policy",
        "strict",
        "--export-roundtrip",
    ])
    assert Path(checkpoint).is_file()
    assert train_result["export"]["quant_scheme"] == "mixed_v2"
    assert strict_eval["roundtrip"]["artifact_bytes"] > 0
    assert strict_eval["roundtrip"]["quant_scheme"] == "mixed_v2"
    assert strict_eval["roundtrip"]["submission_total_bytes"] >= strict_eval["roundtrip"]["artifact_bytes"]
