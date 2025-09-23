"""Shared dataset utilities for wildlife species classification."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    """Load manifest rows and ensure required columns exist."""
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"crop_path", "species", "video_stem"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or set())
            raise ValueError(
                f"Manifest missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def normalize_rel_path(path_str: str) -> str:
    """Normalize manifest crop paths for cross-platform consistency."""
    return Path(path_str.replace("\\", "/")).as_posix()


def determine_split_type(splits: Dict[str, Sequence[str]]) -> str:
    """Infer whether splits.json contains crop or video identifiers."""
    sample_seq = next((seq for seq in splits.values() if seq), [])
    if not sample_seq:
        return "unknown"
    sample = sample_seq[0]
    return "crop" if sample.lower().endswith(".jpg") else "video"


def build_label_mapping(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    """Create deterministic species→index mapping."""
    species = sorted({row["species"] for row in rows})
    return {label: idx for idx, label in enumerate(species)}


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/eval torchvision transforms."""
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


class CropDataset(Dataset):
    """PyTorch dataset for crop images with metadata preserved."""

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


def _crop_paths_for_split(
    split_name: str,
    split_type: str,
    splits: Dict[str, Sequence[str]],
    manifest_rows: List[Dict[str, str]],
) -> Optional[Sequence[str]]:
    entries = splits.get(split_name, [])
    if split_type == "crop":
        return entries
    video_stems = set(entries)
    return [row["crop_path"] for row in manifest_rows if row["video_stem"] in video_stems]


def create_datasets(
    manifest_rows: List[Dict[str, str]],
    splits: Dict[str, Sequence[str]],
    split_type: str,
    data_root: Path,
    image_size: int,
) -> Tuple[Dict[str, CropDataset], Dict[str, int]]:
    """Create datasets for train/validation/test splits."""
    species_to_idx = build_label_mapping(manifest_rows)
    train_tf, eval_tf = build_transforms(image_size)

    datasets: Dict[str, CropDataset] = {}
    for split in ("train", "validation", "test"):
        crop_paths = _crop_paths_for_split(split, split_type, splits, manifest_rows)
        transform = train_tf if split == "train" else eval_tf
        datasets[split] = CropDataset(
            manifest_rows=manifest_rows,
            crop_paths=crop_paths,
            species_to_idx=species_to_idx,
            data_root=data_root,
            transform=transform,
        )
    return datasets, species_to_idx


def create_dataloaders(
    datasets: Dict[str, CropDataset],
    *,
    batch_size: int,
    num_workers: int,
    balance_classes: bool,
) -> Dict[str, DataLoader]:
    """Create dataloaders; optionally balance training classes."""
    loaders: Dict[str, DataLoader] = {}

    for split, dataset in datasets.items():
        shuffle = split == "train"
        sampler = None
        if split == "train" and balance_classes:
            class_counts = Counter(dataset.targets)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in dataset.targets]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


__all__ = [
    "load_manifest_rows",
    "normalize_rel_path",
    "determine_split_type",
    "build_label_mapping",
    "build_transforms",
    "CropDataset",
    "create_datasets",
    "create_dataloaders",
]
