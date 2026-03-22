#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_golf.config import load_config_file
from hybrid_golf.evaluate import evaluate_checkpoint
from hybrid_golf.train import run_training


def render_table(rows: list[dict[str, object]]) -> str:
    header = "| name | kind | checkpoint | val_bpb | policy |"
    sep = "|---|---|---|---:|---|"
    body = [
        f"| {row.get('name', '')} | {row.get('kind', '')} | {row.get('checkpoint', '')} | "
        f"{row.get('val_bpb', '')} | {row.get('policy', '')} |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential hybrid-golf sweep manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a JSON sweep manifest.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Sweep manifest must contain a non-empty jobs list")

    results: list[dict[str, object]] = []
    checkpoint_by_alias: dict[str, str] = {}
    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Job #{idx} must be an object")
        name = str(job.get("name", f"job_{idx}"))
        kind = str(job.get("kind", "train"))
        config = load_config_file(str(job["config"]), overrides=list(job.get("overrides", [])))
        if "run" not in config:
            config["run"] = {}
        config["run"]["id"] = str(job.get("run_id", config["run"]["id"]))
        row: dict[str, object] = {"name": name, "kind": kind}
        if kind == "train":
            train_result = run_training(config)
            checkpoint_path = str(train_result["checkpoint"])
            row["checkpoint"] = checkpoint_path
            checkpoint_by_alias[name] = checkpoint_path
        elif kind == "eval":
            checkpoint_value = str(job["checkpoint"])
            checkpoint_path = checkpoint_by_alias.get(checkpoint_value, checkpoint_value)
            eval_result = evaluate_checkpoint(
                checkpoint_path,
                config,
                policy_name="strict_causal" if str(job.get("policy", "strict")) == "strict" else "exploratory_ttt",
                export_roundtrip=bool(job.get("export_roundtrip", False)),
            )
            base_result = eval_result["base"] if isinstance(eval_result, dict) and "base" in eval_result else eval_result
            row["checkpoint"] = checkpoint_path
            row["val_bpb"] = base_result["val_bpb"]
            row["policy"] = base_result["policy"]["policy"]
        else:
            raise ValueError(f"Unsupported job kind: {kind!r}")
        results.append(row)

    output_path = manifest_path.with_suffix(".results.json")
    output_path.write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_path = manifest_path.with_suffix(".results.md")
    table_path.write_text(render_table(results), encoding="utf-8")
    print(render_table(results), end="")


if __name__ == "__main__":
    main()
