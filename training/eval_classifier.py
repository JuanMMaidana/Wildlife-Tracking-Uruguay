#!/usr/bin/env python3
"""Evaluate a trained species classifier on a chosen split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision import models

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run the evaluation script") from exc

import data_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate species classifier checkpoints")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Crops manifest path")
    parser.add_argument("--splits", default="experiments/exp_003_autolabel/splits.json", help="Dataset splits JSON")
    parser.add_argument("--checkpoint", default="experiments/exp_003_species/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--output-dir", default="experiments/exp_003_species", help="Directory for evaluation artefacts")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", help="Override inference device")
    parser.add_argument("--batch-size", type=int, help="Override batch size for evaluation")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_splits(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(model_name: str, num_classes: int, device: str) -> models.ResNet | models.MobileNetV3:
    model_name = model_name.lower()
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_name in {"mobilenet_v3_large", "mobilenetv3"}:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def evaluate_split(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    idx_to_species: Dict[int, str],
) -> Dict[str, object]:
    model.eval()
    total = 0
    correct = 0
    confusion = torch.zeros(len(idx_to_species), len(idx_to_species), dtype=torch.long)
    predictions: List[Dict[str, object]] = []

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            meta_dicts: List[Dict[str, str]] = []
            batch_size = labels.size(0)
            for i in range(batch_size):
                sample_meta = {key: metadata[key][i] for key in metadata}
                meta_dicts.append(sample_meta)

            for prob_vec, pred, label, meta in zip(probs, preds, labels, meta_dicts):
                top_prob = prob_vec[pred].item()
                predictions.append(
                    {
                        "crop_path": data_utils.normalize_rel_path(meta.get("crop_path", "")),
                        "video": meta.get("video", ""),
                        "video_stem": meta.get("video_stem", ""),
                        "track_id": meta.get("track_id", ""),
                        "species_true": idx_to_species[label.item()],
                        "species_pred": idx_to_species[pred.item()],
                        "confidence": top_prob,
                    }
                )
            for t, p in zip(labels, preds):
                confusion[t.long(), p.long()] += 1

    accuracy = correct / max(total, 1)
    per_class_f1: Dict[str, float] = {}
    for idx in range(len(idx_to_species)):
        tp = confusion[idx, idx].item()
        fp = confusion[:, idx].sum().item() - tp
        fn = confusion[idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class_f1[idx_to_species[idx]] = f1
    macro_f1 = sum(per_class_f1.values()) / max(len(per_class_f1), 1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion": confusion.tolist(),
        "predictions": predictions,
    }


def save_predictions(predictions: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    cfg_classification = config.get("classification", {})

    manifest_rows = data_utils.load_manifest_rows(Path(args.manifest))
    splits = load_splits(Path(args.splits))
    split_type = data_utils.determine_split_type(splits)
    if split_type == "unknown":
        raise ValueError("Unable to determine split type from splits.json")

    image_size = cfg_classification.get("image_size", 224)
    datasets, species_to_idx = data_utils.create_datasets(
        manifest_rows,
        splits,
        split_type,
        data_root=Path(args.manifest).parent,
        image_size=image_size,
    )
    eval_loader = torch.utils.data.DataLoader(
        datasets[args.split],
        batch_size=args.batch_size or cfg_classification.get("eval_batch_size", 64),
        shuffle=False,
        num_workers=cfg_classification.get("dataloader_workers", 4),
        pin_memory=True,
    )

    idx_to_species = {idx: species for species, idx in species_to_idx.items()}
    device = args.device or config.get("hardware", {}).get("device", "cuda")

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    meta = checkpoint.get("meta", {})
    species_to_idx = meta.get("species_to_idx", species_to_idx)
    idx_to_species = {idx: species for species, idx in species_to_idx.items()}
    model_name = meta.get("config", {}).get("model_name", cfg_classification.get("model", "resnet50"))

    model = build_model(model_name, len(species_to_idx), device)
    model.load_state_dict(checkpoint["model_state"])

    result = evaluate_split(model, eval_loader, device, idx_to_species)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"eval_{args.split}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "per_class_f1": result["per_class_f1"],
            },
            handle,
            indent=2,
        )

    predictions = result["predictions"]
    if predictions:
        preds_csv = output_dir / f"predictions_{args.split}.csv"
        save_predictions(predictions, preds_csv)
        print(f"Predictions saved to {preds_csv}")

    print(
        f"Evaluation complete → split: {args.split}, accuracy: {result['accuracy']:.4f}, "
        f"macro F1: {result['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
