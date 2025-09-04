#!/usr/bin/env python3
"""
MegaDetector Parameter Sweep Script
Systematically tests parameter combinations and generates results CSV.

Usage:
  python scripts/99_sweep_md.py --params experiments/exp_001_md_calibration/params.yaml
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import torch
import yaml
from datetime import datetime
from tqdm import tqdm

# Use YOLOv5 for MegaDetector compatibility
try:
    import yolov5
    USE_YOLOV5 = True
except ImportError:
    from ultralytics import YOLO
    USE_YOLOV5 = False
    print("Warning: yolov5 not installed, using ultralytics (may have compatibility issues with MegaDetector)")

# Import your existing configuration loader
import sys
sys.path.append(str(Path(__file__).parent))


def load_base_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load base pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge_config(base: Dict, overrides: Dict) -> Dict:
    """Deep merge configuration dictionaries"""
    result = json.loads(json.dumps(base))  # Deep copy
    
    def merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                merge(target[key], value)
            else:
                target[key] = value
    
    merge(result, overrides)
    return result


def config_hash(cfg: Dict) -> str:
    """Generate short hash of configuration for tracking"""
    cfg_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(cfg_str.encode('utf-8')).hexdigest()[:8]


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_count,
            "frame_area": width * height
        }
    finally:
        cap.release()


def extract_frames_with_stride(video_path: Path, frame_stride: int = 5) -> List[Tuple[int, any]]:
    """Extract frames from video with configurable stride"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_stride == 0:
                frames.append((frame_idx, frame))
            
            frame_idx += 1
    finally:
        cap.release()
    
    return frames


def process_single_video(video_path: Path, config: Dict, model) -> Dict[str, Any]:
    """
    Process single video with MegaDetector using given configuration.
    Returns metrics for comparison.
    """
    start_time = time.time()
    
    # Get video metadata
    video_info = get_video_info(video_path)
    
    # Extract MegaDetector config
    md_config = config['megadetector']
    
    # Extract frames with stride
    frames = extract_frames_with_stride(video_path, md_config['frame_stride'])
    
    if not frames:
        return {
            "video": video_path.name,
            **video_info,
            "processed_frames": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "processing_time": time.time() - start_time,
            "error": "No frames extracted"
        }
    
    # Core MegaDetector processing with parameter sweep configuration
    detections_count = 0
    frames_with_detections = 0
    
    # Calculate minimum area threshold (relative to frame size)
    min_area_threshold = md_config.get('min_area_ratio', 0.005) * video_info['frame_area']
    
    for frame_idx, frame in tqdm(frames, desc=f"Processing {video_path.name}", leave=False):
        # Run MegaDetector inference
        if USE_YOLOV5:
            # YOLOv5 API
            results = model(frame)
        else:
            # Ultralytics YOLO API
            results = model(
                frame,
                conf=md_config['conf_threshold'],
                iou=md_config.get('iou_threshold', 0.55),
                classes=md_config.get('classes', [0]),  # animal + person if enabled
                max_det=md_config.get('max_detections', 5),
                verbose=False
            )
        
        # Process results for this frame
        frame_detections = 0
        
        if USE_YOLOV5:
            # YOLOv5 result format: tensor with [x1, y1, x2, y2, conf, class]
            if results is not None and len(results.xyxy[0]) > 0:
                detections = results.xyxy[0].cpu().numpy()
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Apply confidence threshold manually for YOLOv5
                    if conf >= md_config['conf_threshold']:
                        # Apply class filtering if specified
                        classes_to_keep = md_config.get('classes', [0, 1])
                        if int(cls) in classes_to_keep:
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            
                            # Apply area filtering
                            if area >= min_area_threshold:
                                frame_detections += 1
        else:
            # Ultralytics result format
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # Apply area filtering
                        if area >= min_area_threshold:
                            frame_detections += 1
        
        # Update counters
        detections_count += frame_detections
        if frame_detections > 0:
            frames_with_detections += 1
    
    processing_time = time.time() - start_time
    
    return {
        "video": video_path.name,
        **video_info,
        "processed_frames": len(frames),
        "frames_with_detections": frames_with_detections,
        "total_detections": detections_count,
        "processing_time": processing_time,
        "min_area_threshold": min_area_threshold,
    }


def setup_results_csv(csv_path: Path):
    """Initialize results CSV with headers"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "timestamp", "config_hash", "video", "width", "height", "fps", "frame_area",
        "conf_threshold", "frame_stride", "min_area_ratio", "min_area_threshold",
        "processed_frames", "frames_with_detections", "total_detections", 
        "processing_time", "detections_per_second", "frames_per_second"
    ]
    
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    
    return headers


def append_result_to_csv(csv_path: Path, cfg_hash: str, config: Dict, result: Dict, headers: List[str]):
    """Append single result to CSV"""
    md_config = config['megadetector']
    
    # Calculate derived metrics
    detections_per_sec = result['total_detections'] / max(0.001, result['processing_time'])
    frames_per_sec = result['processed_frames'] / max(0.001, result['processing_time'])
    
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "config_hash": cfg_hash,
        "video": result['video'],
        "width": result['width'],
        "height": result['height'], 
        "fps": result['fps'],
        "frame_area": result['frame_area'],
        "conf_threshold": md_config['conf_threshold'],
        "frame_stride": md_config['frame_stride'],
        "min_area_ratio": md_config.get('min_area_ratio', 0.005),
        "min_area_threshold": result.get('min_area_threshold', 0),
        "processed_frames": result['processed_frames'],
        "frames_with_detections": result['frames_with_detections'],
        "total_detections": result['total_detections'],
        "processing_time": result['processing_time'],
        "detections_per_second": detections_per_sec,
        "frames_per_second": frames_per_sec,
    }
    
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="MegaDetector parameter sweep")
    parser.add_argument("--params", required=True, help="Path to experiment parameters YAML")
    args = parser.parse_args()
    
    # Load experiment parameters
    params = yaml.safe_load(Path(args.params).read_text())
    
    # Find test videos
    videos_pattern = params['dataset']['videos_glob']
    videos = list(Path().glob(videos_pattern))
    videos = [v for v in videos if v.suffix.lower() == '.mp4']
    
    if not videos:
        print(f"No videos found matching pattern: {videos_pattern}")
        return
    
    # Sample subset if requested
    sample_size = params['dataset'].get('sample_size', 0)
    if sample_size > 0 and sample_size < len(videos):
        if params['dataset'].get('deterministic', True):
            import random
            rng = random.Random(params['dataset'].get('random_seed', 42))
            videos = sorted(rng.sample(videos, sample_size), key=lambda p: p.name)
        else:
            videos = videos[:sample_size]
    
    print(f"Found {len(videos)} test videos")
    
    # Load base configuration
    base_config = load_base_config()
    
    # Setup results CSV
    results_csv = Path(params['outputs']['results_csv'])
    if params['outputs'].get('overwrite', False) and results_csv.exists():
        results_csv.unlink()
    
    csv_headers = setup_results_csv(results_csv)
    
    # Initialize model once
    print("Loading MegaDetector model...")
    model_path = base_config['megadetector']['weights']
    if not Path(model_path).exists():
        print(f"Model weights not found: {model_path}")
        return
    
    # Load model based on available package
    if USE_YOLOV5:
        print("Using YOLOv5 for MegaDetector compatibility...")
        model = yolov5.load(model_path)
        model.eval()
    else:
        print("Using Ultralytics (may have compatibility issues)...")
        model = YOLO(model_path)
    
    # Generate parameter combinations
    sweep_params = params['sweep']
    param_combinations = list(itertools.product(
        sweep_params['conf_threshold'],
        sweep_params['frame_stride'], 
        sweep_params['min_area_ratio']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations on {len(videos)} videos")
    print(f"Total experiments: {len(param_combinations) * len(videos)}")
    
    # Run parameter sweep
    total_experiments = len(param_combinations) * len(videos)
    experiment_count = 0
    
    for conf, stride, area_ratio in param_combinations:
        # Create configuration for this parameter set
        overrides = {
            'megadetector': {
                'conf_threshold': conf,
                'frame_stride': stride,
                'min_area_ratio': area_ratio,
                **params.get('pipeline_overrides', {}).get('megadetector', {})
            }
        }
        
        test_config = deep_merge_config(base_config, overrides)
        cfg_hash = config_hash(test_config)
        
        print(f"\nTesting config: conf={conf}, stride={stride}, area_ratio={area_ratio} (hash: {cfg_hash})")
        
        # Process each video with this configuration
        for video_path in videos:
            experiment_count += 1
            
            try:
                result = process_single_video(video_path, test_config, model)
                append_result_to_csv(results_csv, cfg_hash, test_config, result, csv_headers)
                
                print(f"  [{experiment_count}/{total_experiments}] {video_path.name}: "
                      f"{result['frames_with_detections']}/{result['processed_frames']} frames, "
                      f"{result['total_detections']} detections, "
                      f"{result['processing_time']:.1f}s")
                      
            except Exception as e:
                print(f"  ERROR processing {video_path.name}: {str(e)}")
                continue
    
    print(f"\n✅ Sweep completed! Results saved to: {results_csv}")
    print(f"📊 Next steps:")
    print(f"   1. Analyze results: open {results_csv} in Excel/pandas")
    print(f"   2. Choose optimal parameters based on your criteria")  
    print(f"   3. Update config/pipeline.yaml with chosen defaults")
    print(f"   4. Complete the experiment report: {Path(args.params).parent / 'report.md'}")


if __name__ == "__main__":
    main()