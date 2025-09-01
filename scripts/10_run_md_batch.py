#!/usr/bin/env python3
"""
MegaDetector Batch Processing Script
Processes all videos in videos_raw/ folder using MegaDetector v5a
Saves detection results as JSON files in md_json/ folder
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(config: Dict) -> YOLO:
    """Initialize MegaDetector model"""
    model_path = config['megadetector']['weights']
    
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        print("Please download MegaDetector weights from:")
        print("https://github.com/microsoft/CameraTraps/releases")
        sys.exit(1)
    
    model = YOLO(model_path)
    return model

def get_video_files(videos_dir: str) -> List[Path]:
    """Get all video files from directory"""
    video_dir = Path(videos_dir)
    if not video_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    for file_path in video_dir.iterdir():
        if file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)

def extract_frames_with_stride(video_path: Path, frame_stride: int = 5) -> List[tuple]:
    """Extract frames from video with stride"""
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

def process_video(video_path: Path, model: YOLO, config: Dict) -> Dict:
    """Process single video with MegaDetector"""
    md_config = config['megadetector']
    
    print(f"Processing: {video_path.name}")
    
    # Extract frames with stride
    frames = extract_frames_with_stride(
        video_path, 
        frame_stride=md_config['frame_stride']
    )
    
    if not frames:
        print(f"Warning: No frames extracted from {video_path.name}")
        return {
            'video': str(video_path.name),
            'detections': [],
            'info': {'total_frames': 0, 'processed_frames': 0}
        }
    
    detections = []
    
    # Process frames in batches
    for frame_idx, frame in tqdm(frames, desc=f"Processing {video_path.name}"):
        
        # Run detection
        results = model(
            frame,
            conf=md_config['conf_threshold'],
            iou=md_config['iou_threshold'],
            classes=md_config['classes'],
            max_det=md_config['max_detections'],
            verbose=False
        )
        
        # Extract detections for this frame
        frame_detections = []
        
        if results and len(results) > 0:
            result = results[0]  # Single image result
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Filter by minimum area
                    if area >= md_config['min_area']:
                        frame_detections.append({
                            'bbox': [float(x1), float(y1), float(width), float(height)],  # x, y, w, h format
                            'confidence': float(conf),
                            'class': int(cls),
                            'area': float(area)
                        })
        
        # Only save frame if it has detections
        if frame_detections:
            detections.append({
                'frame': frame_idx,
                'detections': frame_detections
            })
    
    # Prepare output
    result = {
        'video': str(video_path.name),
        'detections': detections,
        'info': {
            'total_frames': len(frames) * md_config['frame_stride'],
            'processed_frames': len(frames),
            'frames_with_detections': len(detections),
            'total_detections': sum(len(f['detections']) for f in detections),
            'frame_stride': md_config['frame_stride'],
            'model_config': {
                'weights': md_config['weights'],
                'conf_threshold': md_config['conf_threshold'],
                'min_area': md_config['min_area']
            }
        }
    }
    
    return result

def save_detections(detections: Dict, output_dir: str, video_name: str):
    """Save detection results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename: video_name.json
    video_stem = Path(video_name).stem
    output_file = output_dir / f"{video_stem}.json"
    
    with open(output_file, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"Saved detections: {output_file}")

def main():
    """Main processing function"""
    # Load configuration
    config = load_config()
    
    # Setup model
    print("Loading MegaDetector model...")
    model = setup_model(config)
    
    # Get video files
    video_files = get_video_files(config['paths']['videos_raw'])
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No video files found!")
        return
    
    # Process each video
    start_time = time.time()
    
    for video_path in video_files:
        # Check if already processed
        video_stem = video_path.stem
        output_file = Path(config['paths']['md_json']) / f"{video_stem}.json"
        
        if output_file.exists():
            print(f"Skipping {video_path.name} (already processed)")
            continue
        
        try:
            # Process video
            detections = process_video(video_path, model, config)
            
            # Save results
            save_detections(detections, config['paths']['md_json'], video_path.name)
            
            # Print summary
            info = detections['info']
            print(f"  {info['frames_with_detections']}/{info['processed_frames']} frames with detections")
            print(f"  {info['total_detections']} total detections")
            print()
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    print(f"Processing complete! Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()