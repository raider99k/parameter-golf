#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_golf.config import apply_overrides, deep_merge, ensure_required_sections, load_config_file, resolve_config_inheritance
from hybrid_golf.evaluate import evaluate_checkpoint


CHECKPOINT_LOCKED_EXPORT_KEYS = (
    "artifact_name",
    "keep_float_max_numel",
    "keep_float_policy",
    "quant_scheme",
    "zlib_level",
)


def resolve_eval_config(config_path: str, checkpoint_path: str, overrides: list[str]) -> dict[str, object]:
    raw_file_config = resolve_config_inheritance(config_path)
    file_config = load_config_file(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_config, dict):
        # Checkpoints own the train/model architecture. The passed config file is
        # allowed to override runtime and evaluation behavior only.
        resolved = ensure_required_sections(checkpoint_config)
        for section in ("run", "data", "tokenizer", "experts", "adaptation", "eval", "export"):
            if isinstance(raw_file_config.get(section), dict):
                merged = deep_merge(resolved.get(section, {}), raw_file_config[section])
                if section == "export":
                    export_base = resolved.get("export", {})
                    if isinstance(export_base, dict):
                        for key in CHECKPOINT_LOCKED_EXPORT_KEYS:
                            if key in export_base:
                                merged[key] = export_base[key]
                resolved[section] = merged
    else:
        resolved = file_config
    if overrides:
        resolved = ensure_required_sections(apply_overrides(resolved, overrides))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a hybrid-golf checkpoint with strict or exploratory policy.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--policy", default="strict", choices=("strict", "exploratory"), help="Evaluation policy.")
    parser.add_argument("--export-roundtrip", action="store_true", help="Validate int8+zlib roundtrip evaluation too.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Dotted override: section.key=value")
    args = parser.parse_args()
    config = resolve_eval_config(args.config, args.checkpoint, args.overrides)
    result = evaluate_checkpoint(
        args.checkpoint,
        config,
        policy_name="strict_causal" if args.policy == "strict" else "exploratory_ttt",
        export_roundtrip=args.export_roundtrip,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
