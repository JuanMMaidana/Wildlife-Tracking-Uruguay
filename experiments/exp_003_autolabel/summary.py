#!/usr/bin/env python3
"""Summarise auto-label manifest statistics and generate report scaffolding."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

ManifestRow = Dict[str, str]
SummaryRow = Dict[str, object]

SUMMARY_COLUMNS = [
    "species",
    "n_videos",
    "n_tracks",
    "n_crops",
    "median_track_len",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate autolabel summary from manifest")
    parser.add_argument("--manifest", default="data/crops_manifest.csv", help="Path to crops manifest")
    parser.add_argument(
        "--out-dir",
        default="experiments/exp_003_autolabel",
        help="Directory to write summary.csv and report.md",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> List[ManifestRow]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "video",
            "video_stem",
            "track_id",
            "species",
            "frame_index",
            "dwell_time_s",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or set())
            raise ValueError(
                f"Manifest missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            rows.append(row)
    return rows


def compute_summary(rows: Iterable[ManifestRow]) -> List[SummaryRow]:
    species_tracks: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = defaultdict(dict)
    species_crops: Dict[str, int] = defaultdict(int)
    species_videos: Dict[str, set] = defaultdict(set)

    for row in rows:
        species = row.get("species", "unknown")
        video_stem = row.get("video_stem", "")
        video = row.get("video", "")
        track_id = row.get("track_id", "")
        track_key = (video_stem, track_id)

        species_crops[species] += 1
        species_videos[species].add(video or video_stem)

        if track_key not in species_tracks[species]:
            dwell = float(row.get("dwell_time_s", 0.0) or 0.0)
            species_tracks[species][track_key] = {
                "dwell_time": dwell,
            }

    summary_rows: List[SummaryRow] = []
    for species in sorted(species_crops.keys()):
        tracks = species_tracks[species]
        dwell_values = [track_info["dwell_time"] for track_info in tracks.values() if track_info]
        median_dwell = median(dwell_values) if dwell_values else 0.0
        summary_rows.append(
            {
                "species": species,
                "n_videos": len(species_videos[species]),
                "n_tracks": len(tracks),
                "n_crops": species_crops[species],
                "median_track_len": round(median_dwell, 2),
            }
        )
    return summary_rows


def write_summary(out_dir: Path, rows: List[SummaryRow]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def write_report(out_dir: Path, rows: List[SummaryRow]) -> Path:
    report_path = out_dir / "report.md"
    generated = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = [
        "# Auto-label Summary",
        "",
        f"Generated: {generated}",
        "",
    ]

    if not rows:
        lines.append("No crops present in manifest.")
    else:
        lines.append("| Species | Videos | Tracks | Crops | Median Track Length (s) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['species']} | {row['n_videos']} | {row['n_tracks']} | {row['n_crops']} | {row['median_track_len']} |"
            )
        lines.extend(
            [
                "",
                "## Next Actions",
                "- [ ] Review crop quality for each species (attach thumbnails)",
                "- [ ] Confirm dwell time aligns with raw footage",
                "- [ ] Update parameter tweaks in validation_notes.md",
            ]
        )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return report_path


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    rows = read_manifest(manifest_path)
    summary = compute_summary(rows)
    write_summary(out_dir, summary)
    write_report(out_dir, summary)

    print(f"Summary written to {out_dir / 'summary.csv'}")
    print(f"Report scaffold written to {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
