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
from hybrid_golf.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid-golf research model from JSON config.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Dotted override: section.key=value")
    args = parser.parse_args()
    config = load_config_file(args.config, overrides=args.overrides)
    result = run_training(config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
