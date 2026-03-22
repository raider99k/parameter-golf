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

from hybrid_golf.config import apply_overrides, ensure_required_sections, load_config_file
from hybrid_golf.evaluate import evaluate_checkpoint


def resolve_eval_config(config_path: str, checkpoint_path: str, overrides: list[str]) -> dict[str, object]:
    file_config = load_config_file(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_config, dict):
        resolved = ensure_required_sections(checkpoint_config)
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
