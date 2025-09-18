#!/usr/bin/env python3
"""
Auto-label tracks by filename regex and export crops + manifest.

PIPELINE OVERVIEW:
1. Load ByteTrack JSON files (output from scripts/20_run_tracking.py)
2. For each video, determine species from filename using regex patterns
3. For each track, extract high-quality crops around the representative frame
4. Save crops to data/crops/<species>/ and record metadata in CSV manifest

EXAMPLE FLOW:
Input:  data/tracking_json/margay_012.json + data/videos_raw/margay_012.mp4
Output: data/crops/margay/margay_012__tid1__f150.jpg + row in data/crops_manifest.csv

PARAMETER TUNING:
- neighbors: How many frames around rep_frame to sample (±1, ±2, etc.)
- min_track_len: Skip tracks shorter than this (quality filter)
- max_crops_per_track: Limit crops per track (prevents oversampling)
- crop_padding: Add padding around bbox (prevents edge artifacts)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from scripts.lib.species_map import SpeciesMap, SpeciesMapError, load_species_map


MANIFEST_COLUMNS = [
    "video",
    "video_stem",
    "track_id",
    "species",
    "frame_index",
    "crop_path",
    "confidence",
    "source",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "dwell_time_s",
]


@dataclass
class AutolabelConfig:
    crop_padding: float
    neighbors: int
    min_track_len: int
    max_crops_per_track: int
    skip_classes: Sequence[str]


@dataclass
class TrackDetection:
    frame: int
    bbox: Sequence[float]
    confidence: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label tracks from filenames")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Pipeline config path")
    parser.add_argument("--species-map", help="Species regex map (overrides config)")
    parser.add_argument("--tracks-json", help="Tracking JSON directory or file")
    parser.add_argument("--video-root", help="Directory with raw videos")
    parser.add_argument("--out-dir", help="Output directory for crops")
    parser.add_argument("--manifest", help="Manifest CSV path")
    parser.add_argument("--neighbors", type=int, help="Frames to sample on each side of representative frame")
    parser.add_argument("--crop-padding", type=float, help="Padding ratio around bbox")
    parser.add_argument("--min-track-len", type=int, help="Minimum detections required for track")
    parser.add_argument("--max-crops-per-track", type=int, help="Maximum crops per track")
    parser.add_argument("--skip-classes", nargs="*", help="Species labels to skip (space separated)")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_config(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, AutolabelConfig, Path]:
    config = load_config(Path(args.config))

    classific_cfg = config.get("classification", {})
    paths_cfg = config.get("paths", {})

    species_map_path = Path(args.species_map or classific_cfg.get("species_map", "config/species_map.yaml"))
    tracks_path = Path(args.tracks_json or paths_cfg.get("tracks_json", "data/tracking_json"))
    video_root = Path(args.video_root or paths_cfg.get("videos_raw", "data/videos_raw"))
    crops_dir = Path(args.out_dir or paths_cfg.get("crops", "data/crops"))
    manifest_path = Path(args.manifest or paths_cfg.get("crops_manifest", "data/crops_manifest.csv"))

    skip_classes = classific_cfg.get("skip_classes", ["no_animal", "unknown_animal"])
    if args.skip_classes:
        skip_classes = args.skip_classes

    autolabel_cfg = AutolabelConfig(
        crop_padding=float(args.crop_padding if args.crop_padding is not None else classific_cfg.get("crop_padding", 0.05)),
        neighbors=int(args.neighbors if args.neighbors is not None else classific_cfg.get("neighbors", 2)),
        min_track_len=int(args.min_track_len if args.min_track_len is not None else classific_cfg.get("min_track_len", 6)),
        max_crops_per_track=int(args.max_crops_per_track if args.max_crops_per_track is not None else classific_cfg.get("max_crops_per_track", 5)),
        skip_classes=skip_classes,
    )

    return species_map_path, tracks_path, video_root, manifest_path, autolabel_cfg, crops_dir


def list_tracking_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise FileNotFoundError(f"Tracking path not found: {path}")


def ensure_manifest(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(MANIFEST_COLUMNS)


def load_classes(config_path: Path) -> List[str]:
    classes_file = config_path.parent / "classes.yaml"
    species: List[str] = []
    with classes_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("- "):
                species.append(stripped[2:])
    return species


def resolve_species(video_stem: str, species_map: SpeciesMap, skip_classes: Sequence[str]) -> Tuple[str, bool]:
    """
    Species Resolution: Input filename → Output species label
    
    Example: "margay_012" → "margay" (matches regex pattern "^margay")
             "capybara_wild_003" → "capybara" (matches pattern "^capybara")
             "random_video" → "unknown_animal" (no pattern matches)
    """
    species = species_map.match(video_stem)
    if species is None:
        species = "unknown_animal"
    skip = species in skip_classes  # Skip if in ["no_animal", "unknown_animal"]
    return species, skip


def collect_detections(track: Dict) -> Dict[int, TrackDetection]:
    """
    Convert ByteTrack detection list into frame-indexed dictionary for fast lookup
    
    Input:  track.detections = [{"frame": 148, "bbox": [100,200,50,75], "confidence": 0.8}, ...]
    Output: {148: TrackDetection(frame=148, bbox=[100,200,50,75], confidence=0.8), ...}
    """
    mapping: Dict[int, TrackDetection] = {}
    for det in track.get("detections", []):
        mapping[det["frame"]] = TrackDetection(
            frame=det["frame"],
            bbox=det["bbox"],
            confidence=det.get("confidence", det.get("score", 0.0)),
            source=det.get("src", "unknown"),  # "high" or "low" from ByteTrack
        )
    return mapping


def select_frames(track: Dict, detections: Dict[int, TrackDetection], neighbors: int) -> List[int]:
    """
    Frame Selection: Pick best frames around representative frame
    
    Example with neighbors=2:
    - rep_frame = 150 (highest confidence detection from ByteTrack)
    - neighbors = 2
    - Candidates: [148, 149, 150, 151, 152]
    - Result: [148, 149, 150, 151, 152] (only if those frames have detections)
    """
    frames = set()
    rep_frame = track.get("rep_frame")  # Best frame chosen by ByteTrack
    if rep_frame is None and detections:
        # Fallback: find highest confidence detection manually
        rep_frame = max(detections.values(), key=lambda d: d.confidence).frame
    if rep_frame is None:
        return []
    
    frames.add(rep_frame)  # Always include the best frame
    
    # Add neighboring frames (±1, ±2, etc.) if they have detections
    for offset in range(1, neighbors + 1):
        for candidate in (rep_frame - offset, rep_frame + offset):
            if candidate in detections:  # Only if frame has actual detection
                frames.add(candidate)
    
    ordered = sorted(frames)
    return ordered


def apply_padding(
    bbox: Sequence[float],
    padding: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """
    Crop Bounds Calculation: Expand bbox with padding, clip to frame boundaries
    
    Example with padding=0.05 (5%):
    - bbox = [100, 200, 50, 75]  # x, y, w, h from ByteTrack
    - padding = 0.05 (5%)
    - Expands box: [95, 195, 60, 85] but clips to frame boundaries
    - Returns: (x1, y1, x2, y2) coordinates for cropping
    """
    x, y, w, h = bbox
    pad_w = w * padding  # 5% of width
    pad_h = h * padding  # 5% of height
    x1 = max(0, int(round(x - pad_w)))
    y1 = max(0, int(round(y - pad_h)))
    x2 = min(frame_width, int(round(x + w + pad_w)))
    y2 = min(frame_height, int(round(y + h + pad_h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop bounds computed")
    return x1, y1, x2, y2


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def save_crop(frame: np.ndarray, crop_bounds: Tuple[int, int, int, int], output_path: Path) -> bool:
    """
    Crop Extraction: Cut out animal from video frame and save as JPG
    
    Example:
    - Cuts out region from video frame: frame[195:280, 95:155] 
    - Saves as: data/crops/margay/margay_012__tid1__f150.jpg
    """
    x1, y1, x2, y2 = crop_bounds
    crop = frame[y1:y2, x1:x2]  # Extract the rectangle from the frame
    if crop.size == 0:
        return False  # Skip empty/invalid crops
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create species directory
    return bool(cv2.imwrite(str(output_path), crop))  # Save as JPG


def dwell_time_seconds(track: Dict, fps_effective: float) -> float:
    length = int(track.get("length", 0))
    if fps_effective <= 0 or length == 0:
        return 0.0
    return length / fps_effective


def append_manifest_rows(manifest_path: Path, rows: Iterable[List]):
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def process_track(
    track: Dict,
    detections: Dict[int, TrackDetection],
    frames: List[int],
    cap: cv2.VideoCapture,
    species: str,
    video_name: str,
    video_stem: str,
    crops_dir: Path,
    padding: float,
    frame_width: int,
    frame_height: int,
    dwell_seconds: float,
    max_crops: int,
) -> List[List]:
    manifest_rows: List[List] = []
    crops_written = 0

    for frame_idx in frames:
        detection = detections.get(frame_idx)
        if detection is None:
            continue
        crop_bounds = apply_padding(detection.bbox, padding, frame_width, frame_height)
        frame = read_frame(cap, frame_idx)
        if frame is None:
            continue
        crop_filename = f"{video_stem}__tid{track['track_id']}__f{frame_idx}.jpg"
        output_path = crops_dir / species / crop_filename
        success = save_crop(frame, crop_bounds, output_path)
        if not success:
            continue
        try:
            rel_crop = output_path.relative_to(crops_dir.parent)
        except ValueError:
            rel_crop = output_path

        manifest_rows.append([
            video_name,
            video_stem,
            track["track_id"],
            species,
            frame_idx,
            str(rel_crop),
            round(float(detection.confidence), 4),
            detection.source,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
            crop_bounds[0],
            crop_bounds[1],
            crop_bounds[2],
            crop_bounds[3],
            round(dwell_seconds, 2),
        ])
        crops_written += 1
        if crops_written >= max_crops:
            break
    return manifest_rows


def process_tracking_file(
    tracking_file: Path,
    video_root: Path,
    crops_dir: Path,
    manifest_path: Path,
    species_map: SpeciesMap,
    autolabel_cfg: AutolabelConfig,
) -> Dict[str, int]:
    """
    Main Processing Loop: Process one tracking JSON file
    
    Flow:
    1. Load ByteTrack results from JSON
    2. Figure out species from filename ("margay_012.json" → "margay") 
    3. For each track in this video:
       - Skip if track too short (< min_track_len)
       - Select frames around representative frame (±neighbors)
       - Extract crops and save to data/crops/{species}/
       - Record metadata in manifest CSV
    """
    with tracking_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    video_name = data.get("video") or tracking_file.stem + ".mp4"
    video_stem = Path(video_name).stem
    species, skip = resolve_species(video_stem, species_map, autolabel_cfg.skip_classes)

    stats = defaultdict(int)
    stats["tracks_total"] = len(data.get("tracks", []))

    if skip:
        stats["tracks_skipped_species"] = stats["tracks_total"]
        return stats

    video_path = video_root / video_name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frame_width = int(data.get("video_info", {}).get("width") or cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(data.get("video_info", {}).get("height") or cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_effective = float(data.get("video_info", {}).get("fps_effective") or cap.get(cv2.CAP_PROP_FPS))

        rows_to_append: List[List] = []

        for track in data.get("tracks", []):
            stats["tracks_considered"] += 1
            if int(track.get("length", 0)) < autolabel_cfg.min_track_len:
                stats["tracks_too_short"] += 1
                continue

            detections = collect_detections(track)
            frames = select_frames(track, detections, autolabel_cfg.neighbors)
            if not frames:
                stats["tracks_no_frames"] += 1
                continue

            dwell_seconds = dwell_time_seconds(track, fps_effective)
            rows = process_track(
                track,
                detections,
                frames,
                cap,
                species,
                video_name,
                video_stem,
                crops_dir,
                autolabel_cfg.crop_padding,
                frame_width,
                frame_height,
                dwell_seconds,
                autolabel_cfg.max_crops_per_track,
            )
            if not rows:
                stats["tracks_no_crops"] += 1
                continue

            rows_to_append.extend(rows)
            stats["tracks_labeled"] += 1
            stats["crops_written"] += len(rows)

        if rows_to_append:
            append_manifest_rows(manifest_path, rows_to_append)
    finally:
        cap.release()

    return stats


def main() -> int:
    args = parse_args()
    (
        species_map_path,
        tracks_path,
        video_root,
        manifest_path,
        autolabel_cfg,
        crops_dir,
    ) = resolve_config(args)

    ensure_manifest(manifest_path)

    classes = load_classes(species_map_path)
    try:
        species_map = load_species_map(species_map_path, valid_species=classes)
    except SpeciesMapError as exc:
        raise SystemExit(f"Species map validation error: {exc}")

    tracking_files = list_tracking_files(tracks_path)
    if not tracking_files:
        print(f"No tracking files found in {tracks_path}")
        return 1

    print("🦊 Autolabel from filenames")
    print(f"Species map: {species_map_path}")
    print(f"Tracks: {tracks_path} ({len(tracking_files)} files)")
    print(f"Videos root: {video_root}")
    print(f"Crops output: {crops_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Config → padding={autolabel_cfg.crop_padding}, neighbors={autolabel_cfg.neighbors}, min_track_len={autolabel_cfg.min_track_len}, max_crops={autolabel_cfg.max_crops_per_track}")

    aggregate = defaultdict(int)

    for idx, tracking_file in enumerate(tracking_files, 1):
        print(f"[{idx}/{len(tracking_files)}] Processing {tracking_file.name}")
        stats = process_tracking_file(
            tracking_file,
            video_root,
            crops_dir,
            manifest_path,
            species_map,
            autolabel_cfg,
        )
        for key, value in stats.items():
            aggregate[key] += value

    print("\nSummary:")
    for key in sorted(aggregate.keys()):
        print(f"  {key}: {aggregate[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
