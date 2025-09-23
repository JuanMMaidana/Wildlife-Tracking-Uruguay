#!/usr/bin/env python3
"""Render a video with track bounding boxes and predicted species labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

COLOR_PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (168, 218, 220),
    (255, 183, 3),
    (77, 144, 142),
    (113, 88, 226),
    (242, 132, 130),
    (38, 70, 83),
    (42, 157, 143),
    (233, 196, 106),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay species predictions on video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--tracking-json", required=True, help="Tracking JSON for the video")
    parser.add_argument("--track-predictions", required=True, help="CSV with track_id → species mapping")
    parser.add_argument("--output", help="Path for labeled MP4 (default: <video>_labeled.mp4)")
    parser.add_argument("--confidence-column", default="confidence_mean", help="Column used for confidence display")
    return parser.parse_args()


def load_track_predictions(csv_path: Path, video_name: str) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("video") == video_name:
                mapping[row["track_id"]] = row
    if not mapping:
        raise ValueError(f"No predictions for video {video_name} in {csv_path}")
    return mapping


def load_tracking(tracking_path: Path) -> Dict[str, List[Dict[str, float]]]:
    with tracking_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    tracks = data.get("tracks", [])
    per_frame: Dict[int, List[Tuple[str, List[float]]]] = {}
    for track in tracks:
        tid = str(track["track_id"])
        for det in track.get("detections", []):
            frame = det["frame"]
            per_frame.setdefault(frame, []).append((tid, det["bbox"]))
    return per_frame


def species_color(species: str) -> Tuple[int, int, int]:
    idx = abs(hash(species)) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def draw_tracks_on_video(
    video_path: Path,
    tracking_frames: Dict[int, List[Tuple[str, List[float]]]],
    track_predictions: Dict[str, Dict[str, str]],
    output_path: Path,
    confidence_column: str,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracking_frames.get(frame_idx, [])
        for track_id, bbox in detections:
            pred = track_predictions.get(track_id)
            if not pred:
                continue
            x, y, w, h = bbox
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            color = species_color(pred.get("species_pred", "unknown"))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            label = pred.get("species_pred", "unknown")
            conf = pred.get(confidence_column, pred.get("confidence", ""))
            text = f"{label}"
            if conf:
                text += f" ({float(conf):.2f})"
            cv2.putText(frame, text, (pt1[0], max(pt1[1] - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    tracking_path = Path(args.tracking_json)
    predictions_path = Path(args.track_predictions)

    video_name = video_path.name
    track_predictions = load_track_predictions(predictions_path, video_name)
    tracking_frames = load_tracking(tracking_path)

    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_labeled.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    draw_tracks_on_video(video_path, tracking_frames, track_predictions, output_path, args.confidence_column)
    print(f"Labeled video saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
