#!/usr/bin/env python3
"""Aggregate per-track species counts using classifier predictions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate species counts by video")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Path to crops manifest")
    parser.add_argument("--predictions", default="experiments/exp_003_species/predictions_test.csv", help="Per-crop predictions CSV")
    parser.add_argument("--out-dir", default="experiments/exp_004_counts", help="Output directory")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum prediction confidence to consider")
    return parser.parse_args()


def normalize_path(path_str: str) -> str:
    return path_str.replace("\\", "/")


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def majority_vote(rows: List[Dict[str, str]]) -> Tuple[str, float]:
    counts = Counter(row["species_pred"] for row in rows)
    species, _ = counts.most_common(1)[0]
    confidences = [float(row.get("confidence", 0.0)) for row in rows if row["species_pred"] == species]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return species, mean_conf


def aggregate_counts(
    manifest_rows: List[Dict[str, str]],
    prediction_rows: List[Dict[str, str]],
    min_confidence: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    manifest_index: Dict[str, Dict[str, str]] = {}
    for row in manifest_rows:
        manifest_index[normalize_path(row["crop_path"])] = row

    merged_by_track: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for pred in prediction_rows:
        crop_path = normalize_path(pred["crop_path"])
        if crop_path not in manifest_index:
            continue
        if float(pred.get("confidence", 0.0)) < min_confidence:
            continue
        manifest_row = manifest_index[crop_path]
        merged = {
            **pred,
            **manifest_row,
        }
        key = (manifest_row["video"], manifest_row["track_id"])
        merged_by_track[key].append(merged)

    if not merged_by_track:
        raise ValueError("No overlapping predictions/manifest entries after filtering.")

    track_records: List[Dict[str, object]] = []
    for (video, track_id), rows in merged_by_track.items():
        species_pred, mean_conf = majority_vote(rows)
        dwell = float(rows[0].get("dwell_time_s", 0.0))
        species_true = rows[0].get("species", "")
        track_records.append(
            {
                "video": video,
                "track_id": track_id,
                "species_pred": species_pred,
                "species_true": species_true,
                "confidence_mean": mean_conf,
                "dwell_time_s": dwell,
            }
        )

    summary_records: Dict[Tuple[str, str], Dict[str, object]] = {}
    for record in track_records:
        key = (record["video"], record["species_pred"])
        summary = summary_records.setdefault(
            key,
            {
                "video": record["video"],
                "species": record["species_pred"],
                "n_tracks": 0,
                "avg_dwell_s": 0.0,
                "total_dwell_s": 0.0,
                "confidence_mean": 0.0,
            },
        )
        summary["n_tracks"] += 1
        summary["total_dwell_s"] += record["dwell_time_s"]
        summary["confidence_mean"] += record["confidence_mean"]

    for summary in summary_records.values():
        n = summary["n_tracks"]
        summary["avg_dwell_s"] = summary["total_dwell_s"] / n if n else 0.0
        summary["confidence_mean"] = summary["confidence_mean"] / n if n else 0.0

    track_records.sort(key=lambda r: (r["video"], int(r["track_id"])) if str(r["track_id"]).isdigit() else (r["video"], r["track_id"]))
    summary_list = sorted(summary_records.values(), key=lambda r: (r["video"], r["species"]))

    return track_records, summary_list


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = rows[0].keys()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    manifest_rows = load_csv(Path(args.manifest))
    predictions_rows = load_csv(Path(args.predictions))

    track_records, summary_records = aggregate_counts(manifest_rows, predictions_rows, args.min_confidence)

    out_dir = Path(args.out_dir)
    write_csv(track_records, out_dir / "track_predictions.csv")
    write_csv(summary_records, out_dir / "results.csv")

    print(f"Track-level predictions → {out_dir / 'track_predictions.csv'}")
    print(f"Video/species summary   → {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
