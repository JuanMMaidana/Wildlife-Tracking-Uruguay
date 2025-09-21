#!/usr/bin/env python3
"""Evaluation stub for species classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate species classifier")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Crops manifest path")
    parser.add_argument("--splits", default="experiments/exp_003_autolabel/splits.json", help="Video-level splits JSON")
    parser.add_argument("--checkpoint", required=False, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="experiments/exp_003_species", help="Directory for metrics outputs")
    parser.add_argument("--device", help="Override inference device")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load pipeline configuration")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_splits(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    splits = load_splits(Path(args.splits))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluation configuration:")
    print(f"  Checkpoint: {args.checkpoint or 'TBD'}")
    print(f"  Manifest: {args.manifest}")
    print(f"  Splits: {args.splits}")
    print(f"  Device: {args.device or config.get('hardware', {}).get('device', 'cuda')}")
    print(f"  Output dir: {output_dir}")
    print(f"  Test videos: {len(splits.get('test', []))}")
    print("TODO: load trained model, run inference on test split, compute metrics (top-1, per-class F1, confusion matrix).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
