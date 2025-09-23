#!/usr/bin/env python3
"""Run the full inference pipeline on a folder or single video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

DEFAULT_CONFIG = Path("config/pipeline.yaml")
DEFAULT_OUTPUT = Path("experiments/exp_005_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection → tracking → classification → counts → labeled video")
    parser.add_argument("--videos", default="data/video_inference", help="Video file or directory containing videos")
    parser.add_argument("--checkpoint", default="experiments/exp_003_species/best_model.pt", help="Trained classifier checkpoint")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Root directory for inference outputs")
    parser.add_argument("--device", default="auto", help="Device for classifier inference (cuda/cpu/auto)")
    return parser.parse_args()


def list_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        videos = sorted([p for p in path.glob("**/*.mp4")])
        return videos
    raise FileNotFoundError(f"Video path not found: {path}")


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_temp_config(config: Dict, temp_path: Path) -> None:
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def adjust_paths(config: Dict, output_root: Path) -> Dict:
    cfg = json.loads(json.dumps(config))  # deep copy via JSON

    paths = cfg.setdefault("paths", {})
    paths["md_json"] = str(output_root / "md_json")
    paths["tracks_json"] = str(output_root / "tracking_json")
    paths["crops"] = str(output_root / "crops")
    paths["crops_manifest"] = str(output_root / "crops_manifest.csv")
    paths["reports"] = str(output_root / "reports")
    paths["preview_out"] = str(output_root / "preview")

    classification = cfg.setdefault("classification", {})
    classification["skip_classes"] = []

    return cfg


def run(cmd: List[str]) -> None:
    full_cmd = [sys.executable] + cmd
    print("→", " ".join(full_cmd))
    subprocess.run(full_cmd, check=True)


def main() -> int:
    args = parse_args()
    input_path = Path(args.videos)
    videos = list_videos(input_path)
    if not videos:
        print("No videos found. Nothing to process.")
        return 0

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        staging_dir = output_root / "input_videos"
        staging_dir.mkdir(parents=True, exist_ok=True)
        for video in videos:
            shutil.copy2(video, staging_dir / video.name)
        videos_root = staging_dir
        videos = list_videos(staging_dir)
    else:
        videos_root = input_path

    base_config = load_config(DEFAULT_CONFIG)
    temp_config_path = output_root / "inference_config.yaml"
    adjusted_config = adjust_paths(base_config, output_root)
    write_temp_config(adjusted_config, temp_config_path)

    md_json_dir = output_root / "md_json"
    tracking_dir = output_root / "tracking_json"
    crops_dir = output_root / "crops"
    manifest_path = output_root / "crops_manifest.csv"
    predictions_path = output_root / "predictions.csv"
    counts_dir = output_root / "counts"
    labeled_dir = output_root / "videos_labeled"
    labeled_dir.mkdir(exist_ok=True, parents=True)

    videos_root = Path(args.videos)

    # 1. Detection
    run([
        "scripts/10_run_md_batch.py",
        "--config",
        str(temp_config_path),
        "--video-dir",
        str(videos_root),
    ])

    # 2. Tracking
    run([
        "scripts/20_run_tracking.py",
        "--config",
        str(temp_config_path),
        "--md-json",
        str(md_json_dir),
        "--video-root",
        str(videos_root),
        "--out-json",
        str(tracking_dir),
    ])

    # 3. Crop extraction
    run([
        "scripts/31_autolabel_from_filenames.py",
        "--config",
        str(temp_config_path),
        "--tracks-json",
        str(tracking_dir),
        "--video-root",
        str(videos_root),
        "--out-dir",
        str(crops_dir),
        "--manifest",
        str(manifest_path),
    ])

    # 4. Classifier inference
    run([
        "scripts/run_classifier_inference.py",
        "--checkpoint",
        str(Path(args.checkpoint)),
        "--manifest",
        str(manifest_path),
        "--output",
        str(predictions_path),
        "--device",
        args.device,
    ])

    # 5. Aggregate counts
    run([
        "scripts/40_counts_by_species.py",
        "--manifest",
        str(manifest_path),
        "--predictions",
        str(predictions_path),
        "--out-dir",
        str(counts_dir),
    ])

    track_predictions_path = counts_dir / "track_predictions.csv"

    # 6. Render labeled videos
    for video_path in videos:
        stem = video_path.stem
        tracking_json = tracking_dir / f"{stem}.json"
        if not tracking_json.exists():
            print(f"⚠️  Missing tracking JSON for {stem}, skipping video rendering")
            continue
        run([
            "scripts/render_labeled_video.py",
            "--video",
            str(video_path),
            "--tracking-json",
            str(tracking_json),
            "--track-predictions",
            str(track_predictions_path),
            "--output",
            str(labeled_dir / f"{stem}_labeled.mp4"),
        ])

    print(f"Inference complete. Outputs written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
