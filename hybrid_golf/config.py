from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "id": "hybrid_smoke",
        "output_root": "outputs/hybrid",
        "seed": 1337,
        "device": "auto",
        "dtype": "auto",
        "stage": "smoke",
    },
    "data": {
        "train_glob": "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin",
        "val_glob": "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        "val_tokens_limit": 0,
    },
    "tokenizer": {
        "path": "./data/tokenizers/fineweb_1024_bpe.model",
        "kind": "auto",
        "bos_id": 1,
        "eos_id": 2,
    },
    "model": {
        "vocab_size": 1024,
        "num_layers": 4,
        "model_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 2,
        "mlp_mult": 2,
        "tie_embeddings": True,
        "use_factor_embed": False,
        "embed_dim": 96,
        "num_unique_attn": 0,
        "num_unique_mlp": 0,
        "normformer_lite": False,
        "depth_aware_init": False,
        "recurrent_passes": 1,
        "recurrent_gates": True,
        "layer_modulation": True,
        "depth_aware_residuals": True,
        "pass_modulation": False,
        "pass_q_gain": False,
        "low_rank_deltas": False,
        "delta_rank": 4,
        "delta_init_scale": 0.05,
        "logit_delta_rank": 0,
        "logit_delta_init_scale": 0.05,
        "pass_gate_init": 1.0,
        "linear_impl": "dense",
        "bitlinear_targets": "none",
        "bitlinear_group_size": 0,
        "restore_control_tensors_fp32": True,
        "rope_base": 10000.0,
        "logit_softcap": 30.0,
        "writable_rank": 4,
        "writable_blocks": 2,
        "recurrent_top_block": False,
        "max_extra_passes": 1,
        "tied_embed_init_std": 0.02,
    },
    "train": {
        "enabled": True,
        "iterations": 50,
        "batch_tokens": 1024,
        "grad_accum_steps": 1,
        "seq_len": 64,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "use_compile": True,
        "compile_fullgraph": False,
        "embed_lr": 3e-4,
        "head_lr": 3e-4,
        "matrix_lr": 3e-4,
        "scalar_lr": 3e-4,
        "muon_momentum": 0.95,
        "muon_backend_steps": 5,
        "muon_momentum_warmup_start": 0.85,
        "muon_momentum_warmup_steps": 500,
        "warmup_steps": 8,
        "grad_clip_norm": 1.0,
        "train_log_every": 10,
        "val_every": 0,
        "max_wallclock_seconds": 0.0,
        "projection_qat_enabled": False,
        "qat_start_frac": 0.8,
        "qat_every_steps": 4,
        "qat_lr_scale": 0.25,
        "compression_reg_weight": 0.0,
        "compression_reg_start_frac": 0.7,
        "compression_reg_every_steps": 1,
        "swa_enabled": False,
        "swa_start_frac": 0.8,
        "swa_every": 50,
        "restore_best_val_checkpoint": False,
    },
    "experts": {
        "enable_ngram": True,
        "enable_pointer": True,
        "enable_doc_bias": True,
        "ngram_order": 4,
        "ngram_buckets": 8192,
        "ngram_alpha": 0.25,
        "pointer_window": 256,
        "pointer_alpha": 0.1,
        "doc_bias_alpha": 0.1,
        "gate": {
            "base": {"ngram": 0.15, "pointer": 0.20, "doc_bias": 0.10},
            "entropy_scale": {"ngram": 0.10, "pointer": 0.25, "doc_bias": 0.05},
            "repeat_scale": {"ngram": 0.20, "pointer": 0.45, "doc_bias": 0.10},
            "confidence_scale": {"ngram": 0.35, "pointer": 0.35, "doc_bias": 0.20},
            "max_weight": 1.5,
        },
    },
    "adaptation": {
        "meta_enabled": False,
        "meta_steps": 1,
        "meta_adapt_tokens": 128,
        "meta_query_tokens": 128,
        "meta_loss_a_weight": 0.2,
        "meta_loss_b_weight": 0.8,
        "meta_every": 4,
        "inner_clip": 0.1,
        "doc_bias_lr": 0.05,
        "doc_bias_decay": 0.995,
        "strict_commit_steps": 1,
        "exploratory_prefix_steps": 1,
        "exploratory_persist_across_docs": False,
    },
    "eval": {
        "policy": "strict_causal",
        "score_tokens": 64,
        "adapt_tokens": 256,
        "window_tokens": 0,
        "documentwise": True,
        "adaptive_extra_pass": False,
        "adaptive_entropy_threshold": 4.5,
        "adaptive_disagreement_threshold": 0.35,
        "write_block_logs": True,
    },
    "export": {
        "artifact_name": "model.int8.ptz",
        "keep_float_max_numel": 4096,
        "keep_float_policy": "small_and_control",
        "quant_scheme": "mixed_v1",
        "budget_mode": "submission_total",
        "zlib_level": 9,
        "write_after_train": False,
        "artifact_budget_bytes": 16_000_000,
        "fail_if_over_budget": False,
    },
}


REQUIRED_TOP_LEVEL_KEYS = tuple(DEFAULT_CONFIG.keys())


def load_raw_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return loaded


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def coerce_override_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"Expected dotted.path=value override, got {text!r}")
    key, raw_value = text.split("=", 1)
    path = [part.strip() for part in key.split(".") if part.strip()]
    if not path:
        raise ValueError(f"Override is missing a key: {text!r}")
    return path, coerce_override_value(raw_value)


def set_deep(config: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for override in overrides:
        path, value = parse_override(override)
        set_deep(resolved, path, value)
    return resolved


def ensure_required_sections(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in resolved:
            resolved[key] = copy.deepcopy(DEFAULT_CONFIG[key])
        elif isinstance(DEFAULT_CONFIG[key], dict) and isinstance(resolved[key], dict):
            resolved[key] = deep_merge(DEFAULT_CONFIG[key], resolved[key])
    return resolved


def resolve_config_inheritance(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        cycle = " -> ".join(str(item) for item in (*seen, config_path))
        raise ValueError(f"Config extends cycle detected: {cycle}")
    seen.add(config_path)
    loaded = load_raw_config_file(config_path)
    extends_value = loaded.pop("extends", None)
    if extends_value is None:
        return loaded
    if not isinstance(extends_value, str) or not extends_value.strip():
        raise ValueError(f"Config 'extends' must be a non-empty string: {config_path}")
    parent_path = Path(extends_value)
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    parent = resolve_config_inheritance(parent_path, seen)
    return deep_merge(parent, loaded)


def load_config_file(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    loaded = resolve_config_inheritance(path)
    config = ensure_required_sections(deep_merge(DEFAULT_CONFIG, loaded))
    if overrides:
        config = ensure_required_sections(apply_overrides(config, overrides))
    return config
