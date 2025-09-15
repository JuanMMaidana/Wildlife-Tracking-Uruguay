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

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model(config: Dict):
    """Initialize MegaDetector model using YOLOv5"""
    model_path = config['megadetector']['weights']
    
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        print("Please download MegaDetector weights from:")
        print("https://github.com/microsoft/CameraTraps/releases")
        sys.exit(1)
    
    # Load MegaDetector using torch.hub (YOLOv5 compatible)
    print(f"Loading MegaDetector from {model_path}...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def get_video_files(videos_dir: str) -> List[Path]:
    """Get all video files from directory"""
    video_dir = Path(videos_dir)
    if not video_dir.exists():
        print(f"Videos directory not found: {videos_dir}")
        sys.exit(1)
    
    video_extensions = {'.mp4'}
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

def process_video(video_path: Path, model, config: Dict) -> Dict:
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
    
    # Calculate minimum area in pixels from ratio
    sample_frame = frames[0][1]  # Get first frame
    frame_height, frame_width = sample_frame.shape[:2]
    min_area = md_config['min_area_ratio'] * frame_width * frame_height
    
    detections = []
    
    # Process frames in batches
    for frame_idx, frame in tqdm(frames, desc=f"Processing {video_path.name}"):
        
        # Run detection (YOLOv5 format)
        model.conf = md_config['md_export_threshold']  # Use md_export_threshold from config
        model.iou = md_config['iou_threshold']
        model.classes = md_config['classes']
        model.max_det = md_config['max_detections']
        
        results = model(frame, size=md_config['img_size'])
        
        # Extract detections for this frame (YOLOv5 format)
        frame_detections = []
        
        # YOLOv5 returns pandas DataFrame in results.pandas().xyxy[0]
        if results and len(results.pandas().xyxy) > 0:
            detections_df = results.pandas().xyxy[0]
            
            for _, detection in detections_df.iterrows():
                x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
                conf = detection['confidence']
                cls = int(detection['class'])
                
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Filter by minimum area
                if area >= min_area:
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
                'conf_threshold': md_config['md_export_threshold'],
                'min_area': min_area
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MegaDetector on video batch')
    parser.add_argument('--video-dir', help='Directory containing videos to process')
    parser.add_argument('--config', default='config/pipeline.yaml', help='Config file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup model
    print("Loading MegaDetector model...")
    model = setup_model(config)
    
    # Get video files - use provided directory or config default
    video_dir = args.video_dir or config['paths']['videos_raw']
    video_files = get_video_files(video_dir)
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