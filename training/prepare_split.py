#!/usr/bin/env python3
"""Create train/val/test splits from the crops manifest on a per-video or per-crop basis."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

ManifestRow = Dict[str, str]
SplitAssignments = Dict[str, List[str]]
CropSplitAssignments = Dict[str, List[Dict[str, str]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stratified dataset splits")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Crops manifest path")
    parser.add_argument("--out-dir", default="experiments/exp_003_autolabel", help="Output directory for split files")
    parser.add_argument("--strategy", choices=["video", "crop"], default="video",
                       help="Split by video (prevents data leakage) or crop (balanced classes)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load pipeline configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_manifest(manifest_path: Path) -> List[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video", "video_stem", "species"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or set())
            raise ValueError(f"Manifest missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def infer_video_species(rows: Iterable[ManifestRow]) -> Dict[str, str]:
    video_species: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        video_stem = row["video_stem"]
        species = row.get("species", "unknown")
        video_species[video_stem][species] += 1
    resolved: Dict[str, str] = {}
    for video_stem, counts in video_species.items():
        species, _ = counts.most_common(1)[0]
        resolved[video_stem] = species
    return resolved


def assign_video_splits(
    video_species: Dict[str, str],
    *,
    ratios: Dict[str, float],
    seed: int,
) -> SplitAssignments:
    """Split by videos (prevents data leakage)."""
    rng = random.Random(seed)
    per_species: Dict[str, List[str]] = defaultdict(list)
    for video_stem, species in video_species.items():
        per_species[species].append(video_stem)

    splits = {split: [] for split in ratios.keys()}
    for species, videos in per_species.items():
        rng.shuffle(videos)
        total = len(videos)
        allocated = 0
        remainder = videos[:]
        for split, ratio in ratios.items():
            if split == "train":
                count = int(total * ratio)
            else:
                count = int(round(total * ratio))
            take = min(len(remainder), count)
            splits[split].extend(remainder[:take])
            remainder = remainder[take:]
            allocated += take
        if remainder:
            splits["train"].extend(remainder)
    return splits


def assign_crop_splits(
    manifest_rows: List[ManifestRow],
    *,
    ratios: Dict[str, float],
    seed: int,
) -> CropSplitAssignments:
    """Split by crops (balanced classes, risk of data leakage)."""
    rng = random.Random(seed)

    # Group crops by species
    per_species: Dict[str, List[ManifestRow]] = defaultdict(list)
    for row in manifest_rows:
        species = row.get("species", "unknown")
        per_species[species].append(row)

    splits = {split: [] for split in ratios.keys()}

    for species, crops in per_species.items():
        rng.shuffle(crops)
        total = len(crops)
        remainder = crops[:]

        for split, ratio in ratios.items():
            if split == "train":
                count = int(total * ratio)
            else:
                count = int(round(total * ratio))
            take = min(len(remainder), count)
            splits[split].extend(remainder[:take])
            remainder = remainder[take:]

        # Put any remainder in train
        if remainder:
            splits["train"].extend(remainder)

    return splits


def write_video_outputs(out_dir: Path, assignments: SplitAssignments):
    """Write video-level splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "splits.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(assignments, handle, indent=2, sort_keys=True)
    csv_path = out_dir / "splits.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "video_stem"])
        for split, videos in assignments.items():
            for video in videos:
                writer.writerow([split, video])
    return json_path, csv_path


def write_crop_outputs(out_dir: Path, assignments: CropSplitAssignments):
    """Write crop-level splits."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON format: just the crop paths for each split
    json_assignments = {}
    for split, crops in assignments.items():
        json_assignments[split] = [crop["crop_path"] for crop in crops]

    json_path = out_dir / "splits.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_assignments, handle, indent=2, sort_keys=True)

    # CSV format: full crop metadata
    csv_path = out_dir / "splits.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Write header with all columns from first crop (assuming all have same structure)
        if assignments and any(assignments.values()):
            first_crop = next(iter(next(iter(assignments.values()))))
            header = ["split"] + list(first_crop.keys())
            writer.writerow(header)

            for split, crops in assignments.items():
                for crop in crops:
                    row = [split] + [crop.get(col, "") for col in header[1:]]
                    writer.writerow(row)

    return json_path, csv_path


def summarise_video_splits(assignments: SplitAssignments) -> Dict[str, int]:
    return {split: len(videos) for split, videos in assignments.items()}


def summarise_crop_splits(assignments: CropSplitAssignments) -> Dict[str, int]:
    return {split: len(crops) for split, crops in assignments.items()}


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    manifest_rows = read_manifest(Path(args.manifest))

    ratios_cfg = config.get("split", {})
    defaults = {"train": 0.7, "validation": 0.15, "test": 0.15}
    ratios_numeric = {
        key: float(ratios_cfg.get(key, defaults[key]))
        for key in defaults
    }
    total_ratio = sum(ratios_numeric.values())
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    ratios_normalised = {key: value / total_ratio for key, value in ratios_numeric.items()}

    seed = args.seed if args.seed is not None else int(config.get("classification", {}).get("random_seed", 42))

    if args.strategy == "video":
        print("Using video-level splitting (prevents data leakage)")
        video_species = infer_video_species(manifest_rows)
        assignments = assign_video_splits(video_species, ratios=ratios_normalised, seed=seed)
        write_video_outputs(Path(args.out_dir), assignments)
        counts = summarise_video_splits(assignments)
        print("Split counts (videos):")
        for split, count in counts.items():
            print(f"  {split}: {count}")

    else:  # args.strategy == "crop"
        print("Using crop-level splitting (balanced classes, risk of data leakage)")
        assignments = assign_crop_splits(manifest_rows, ratios=ratios_normalised, seed=seed)
        write_crop_outputs(Path(args.out_dir), assignments)
        counts = summarise_crop_splits(assignments)
        print("Split counts (crops):")
        for split, count in counts.items():
            print(f"  {split}: {count}")

        # Show species distribution
        print("\nSpecies distribution:")
        for split, crops in assignments.items():
            species_counts = Counter(crop["species"] for crop in crops)
            print(f"  {split}: {dict(sorted(species_counts.items()))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
