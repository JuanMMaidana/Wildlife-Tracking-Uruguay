#!/usr/bin/env python3
"""Training entry point for wildlife species classification.

This script reads the auto-labeled crop manifest and the generated dataset
splits, builds PyTorch datasets/dataloaders, and trains a baseline classifier
(ResNet50 or MobileNetV3-Large). Metrics are logged per epoch and the best
checkpoint is persisted for later evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run the training script") from exc


@dataclass
class TrainingConfig:
    manifest_path: Path
    splits_path: Path
    output_dir: Path
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    image_size: int
    balance_classes: bool
    log_interval: int


DEFAULT_MODEL = "resnet50"
DEFAULT_OUTPUT_DIR = Path("experiments/exp_003_species")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline species classifier")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Crops manifest path")
    parser.add_argument("--splits", default="experiments/exp_003_autolabel/splits.json", help="Dataset splits JSON")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for metrics and checkpoints")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v3_large"], help="Override backbone")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", help="Override device (cuda/cpu)")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_training_config(args: argparse.Namespace, config: Dict) -> TrainingConfig:
    classification_cfg = config.get("classification", {})
    hardware_cfg = config.get("hardware", {})

    model_name = args.model or classification_cfg.get("model", DEFAULT_MODEL)
    epochs = args.epochs or classification_cfg.get("epochs", 15)
    batch_size = args.batch_size or hardware_cfg.get("batch_size", 32)
    learning_rate = args.lr or classification_cfg.get("learning_rate", 1e-4)
    weight_decay = classification_cfg.get("weight_decay", 1e-4)
    num_workers = int(classification_cfg.get("dataloader_workers", hardware_cfg.get("num_workers", 4)))
    device = args.device or hardware_cfg.get("device", "cuda")
    image_size = classification_cfg.get("image_size", 224)
    balance_classes = bool(classification_cfg.get("balance_classes", True))
    log_interval = int(classification_cfg.get("log_interval", 50))

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    return TrainingConfig(
        manifest_path=Path(args.manifest),
        splits_path=Path(args.splits),
        output_dir=output_dir,
        model_name=model_name,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        num_workers=num_workers,
        device=str(device),
        image_size=int(image_size),
        balance_classes=balance_classes,
        log_interval=log_interval,
    )


def validate_inputs(cfg: TrainingConfig) -> None:
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {cfg.manifest_path}")
    if not cfg.splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {cfg.splits_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)


def load_splits(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"crop_path", "species", "video_stem"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or set())
            raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def normalize_rel_path(path_str: str) -> str:
    return Path(path_str.replace("\\", "/")).as_posix()


def determine_split_type(splits: Dict[str, Sequence[str]]) -> str:
    sample_seq = next((seq for seq in splits.values() if seq), [])
    if not sample_seq:
        return "unknown"
    sample = sample_seq[0]
    return "crop" if sample.lower().endswith(".jpg") else "video"


def build_label_mapping(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    species = sorted({row["species"] for row in rows})
    return {label: idx for idx, label in enumerate(species)}


class CropDataset(Dataset):
    def __init__(
        self,
        manifest_rows: Iterable[Dict[str, str]],
        crop_paths: Optional[Sequence[str]],
        species_to_idx: Dict[str, int],
        data_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.transform = transform
        allowed = {normalize_rel_path(p) for p in crop_paths} if crop_paths is not None else None
        records: List[Tuple[Path, int, Dict[str, str]]] = []
        for row in manifest_rows:
            rel_path = normalize_rel_path(row["crop_path"])
            if allowed is not None and rel_path not in allowed:
                continue
            label = species_to_idx[row["species"]]
            full_path = data_root / Path(rel_path)
            records.append((full_path, label, row))
        if not records:
            raise ValueError("No records found for split. Check splits.json and manifest paths.")
        self.records = records
        self.targets = [label for _, label, _ in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label, metadata = self.records[idx]
        if not path.exists():
            raise FileNotFoundError(f"Crop not found: {path}")
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label, metadata


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def create_datasets(
    cfg: TrainingConfig,
    manifest_rows: List[Dict[str, str]],
    splits: Dict[str, Sequence[str]],
    split_type: str,
) -> Tuple[Dict[str, CropDataset], Dict[str, int]]:
    data_root = cfg.manifest_path.parent  # manifest lives inside data/
    species_to_idx = build_label_mapping(manifest_rows)
    train_tf, eval_tf = build_transforms(cfg.image_size)

    def crop_paths_for_split(split_name: str) -> Optional[Sequence[str]]:
        entries = splits.get(split_name, [])
        if split_type == "crop":
            return entries
        # video-level: include all crops whose video stem is in the split
        video_stems = {entry for entry in entries}
        return [row["crop_path"] for row in manifest_rows if row["video_stem"] in video_stems]

    datasets: Dict[str, CropDataset] = {}
    for split in ("train", "validation", "test"):
        crop_paths = crop_paths_for_split(split)
        transform = train_tf if split == "train" else eval_tf
        datasets[split] = CropDataset(
            manifest_rows=manifest_rows,
            crop_paths=crop_paths,
            species_to_idx=species_to_idx,
            data_root=data_root,
            transform=transform,
        )
    return datasets, species_to_idx


def create_dataloaders(datasets: Dict[str, CropDataset], cfg: TrainingConfig) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        if split == "train" and cfg.balance_classes:
            class_counts = Counter(dataset.targets)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in dataset.targets]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None

        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    return loaders


def build_model(model_name: str, num_classes: int, device: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name in {"mobilenet_v3_large", "mobilenetv3"}:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - guard for unsupported models
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, _) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if log_interval and batch_idx % log_interval == 0:
            print(f"    [train] batch {batch_idx:04d}/{len(dataloader)} - loss: {loss.item():.4f}")

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels, preds):
                confusion[t.long(), p.long()] += 1

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    per_class_f1: List[float] = []
    for cls_idx in range(num_classes):
        tp = confusion[cls_idx, cls_idx].item()
        fp = confusion[:, cls_idx].sum().item() - tp
        fn = confusion[cls_idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append(2 * precision * recall / (precision + recall))
    macro_f1 = sum(per_class_f1) / max(num_classes, 1)

    return {"loss": avg_loss, "accuracy": accuracy, "macro_f1": macro_f1}


def save_metrics_csv(metrics: List[Dict[str, float]], path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = metrics[0].keys()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(metrics)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "meta": metadata}, path)


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    cfg = build_training_config(args, config)
    validate_inputs(cfg)

    splits = load_splits(cfg.splits_path)
    manifest_rows = load_manifest_rows(cfg.manifest_path)
    split_type = determine_split_type(splits)
    if split_type == "unknown":
        raise ValueError("Unable to determine split type from splits.json")

    datasets, species_to_idx = create_datasets(cfg, manifest_rows, splits, split_type)
    dataloaders = create_dataloaders(datasets, cfg)
    num_classes = len(species_to_idx)
    idx_to_species = {idx: species for species, idx in species_to_idx.items()}

    print("Training configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Weight decay: {cfg.weight_decay}")
    print(f"  Device: {cfg.device}")
    print(f"  Image size: {cfg.image_size}")
    print(f"  Balance classes: {cfg.balance_classes}")
    print(f"  Manifest: {cfg.manifest_path}")
    print(f"  Splits: {cfg.splits_path} (type={split_type})")
    for split, dataset in datasets.items():
        counts = Counter(dataset.targets)
        readable = {idx_to_species[idx]: count for idx, count in sorted(counts.items())}
        print(f"  {split}: {len(dataset)} samples -> {readable}")

    model = build_model(cfg.model_name, num_classes, cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    metrics_per_epoch: List[Dict[str, float]] = []
    best_val_f1 = 0.0
    best_ckpt_path = cfg.output_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, cfg.device, cfg.log_interval)
        val_metrics = evaluate(model, dataloaders["validation"], criterion, cfg.device, num_classes)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        metrics_per_epoch.append(epoch_metrics)

        print(
            f"    Train loss: {train_metrics['loss']:.4f} | acc: {train_metrics['accuracy']:.4f}\n"
            f"    Val   loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                model,
                optimizer,
                best_ckpt_path,
                metadata={
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "species_to_idx": species_to_idx,
                    "config": cfg.__dict__,
                },
            )
            print(f"    ✅ Saved best checkpoint → {best_ckpt_path}")

    # Final evaluation using best checkpoint
    print("\nEvaluating best model on test split...")
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, dataloaders["test"], criterion, cfg.device, num_classes)
    print(
        f"    Test loss: {test_metrics['loss']:.4f} | acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['macro_f1']:.4f}"
    )

    # Persist metrics
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = cfg.output_dir / "metrics.csv"
    save_metrics_csv(metrics_per_epoch, metrics_csv)
    metrics_json = cfg.output_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump({
            "epochs": metrics_per_epoch,
            "best_val_macro_f1": best_val_f1,
            "test_metrics": test_metrics,
            "species_to_idx": species_to_idx,
        }, handle, indent=2)

    print(f"\nMetrics saved to {metrics_csv} and {metrics_json}")
    print(f"Best checkpoint: {best_ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
