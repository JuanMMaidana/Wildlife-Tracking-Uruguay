#!/usr/bin/env python3
"""Training entry point for species classification (skeleton)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


@dataclass
class TrainingConfig:
    manifest_path: Path
    splits_path: Path
    output_dir: Path
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    device: str


DEFAULT_MODEL = "resnet50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline species classifier")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Crops manifest path")
    parser.add_argument("--splits", default="experiments/exp_003_autolabel/splits.json", help="Video-level splits JSON")
    parser.add_argument("--output-dir", default="models/classifier", help="Directory for checkpoints and logs")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v3_large"], help="Override backbone")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", help="Override device (cuda/cpu)")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load pipeline configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_training_config(args: argparse.Namespace, config: Dict) -> TrainingConfig:
    classification_cfg = config.get("classification", {})
    hardware_cfg = config.get("hardware", {})
    model_name = args.model or classification_cfg.get("model", DEFAULT_MODEL)
    epochs = args.epochs or classification_cfg.get("epochs", 10)
    batch_size = args.batch_size or hardware_cfg.get("batch_size", 32)
    learning_rate = args.lr or classification_cfg.get("learning_rate", 1e-4)
    num_workers = hardware_cfg.get("num_workers", 4)
    device = args.device or hardware_cfg.get("device", "cuda")

    return TrainingConfig(
        manifest_path=Path(args.manifest),
        splits_path=Path(args.splits),
        output_dir=Path(args.output_dir),
        model_name=model_name,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        num_workers=int(num_workers),
        device=str(device),
    )


def validate_inputs(cfg: TrainingConfig):
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {cfg.manifest_path}")
    if not cfg.splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {cfg.splits_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_splits(path: Path) -> Dict[str, list]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    training_cfg = build_training_config(args, config)
    validate_inputs(training_cfg)
    splits = load_splits(training_cfg.splits_path)

    split_counts = {split: len(videos) for split, videos in splits.items()}

    print("Training configuration:")
    print(f"  Model: {training_cfg.model_name}")
    print(f"  Epochs: {training_cfg.epochs}")
    print(f"  Batch size: {training_cfg.batch_size}")
    print(f"  Learning rate: {training_cfg.learning_rate}")
    print(f"  Device: {training_cfg.device}")
    print(f"  Manifest: {training_cfg.manifest_path}")
    print(f"  Splits: {training_cfg.splits_path}")
    print(f"  Output dir: {training_cfg.output_dir}")
    print(f"  Split counts: {split_counts}")
    print("TODO: implement dataset construction and training loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
