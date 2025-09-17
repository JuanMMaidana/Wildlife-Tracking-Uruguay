#!/usr/bin/env python3
"""
ByteTrack Visualization Script
Creates visual overlays of tracking results on video frames
Shows track IDs, H/L detection sources, and track continuity
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import colorsys

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_track_colors(num_tracks: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for tracks using HSV color space"""
    colors = []
    for i in range(num_tracks):
        hue = i / max(1, num_tracks)
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to BGR for OpenCV (0-255 range)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def load_tracking_results(tracking_json_path: Path) -> Dict:
    """Load tracking results from JSON"""
    with open(tracking_json_path, 'r') as f:
        return json.load(f)

def get_video_path(tracking_data: Dict, videos_root: Path) -> Path:
    """Extract video path from tracking data"""
    video_name = tracking_data.get('video', 'unknown.mp4')
    # Handle different video name formats
    if isinstance(video_name, str):
        video_name = Path(video_name).name
    
    video_path = videos_root / video_name
    return video_path

def draw_bbox_with_info(frame: np.ndarray, bbox: List[float], track_id: int, 
                       confidence: float, src: str, color: Tuple[int, int, int],
                       frame_idx: int) -> np.ndarray:
    """Draw bounding box with track information"""
    x, y, w, h = [int(coord) for coord in bbox]
    
    # Draw bounding box
    thickness = 3 if src == 'high' else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    conf_str = f"{confidence:.2f}"
    src_indicator = "H" if src == 'high' else "L"
    label = f"ID:{track_id} {src_indicator} {conf_str}"
    
    # Calculate label background size
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Draw label background
    label_y = max(y - 10, label_h + 10)
    cv2.rectangle(frame, (x, label_y - label_h - baseline), 
                  (x + label_w, label_y + baseline), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x, label_y - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw center point
    center_x, center_y = x + w // 2, y + h // 2
    cv2.circle(frame, (center_x, center_y), 4, color, -1)
    
    return frame

def create_frame_info_overlay(frame: np.ndarray, frame_idx: int, total_frames: int,
                             fps_effective: float, num_tracks: int) -> np.ndarray:
    """Add frame information overlay"""
    height, width = frame.shape[:2]
    
    # Frame info
    time_s = frame_idx / fps_effective if fps_effective > 0 else 0
    info_text = f"Frame: {frame_idx}/{total_frames} | Time: {time_s:.1f}s | Tracks: {num_tracks}"
    
    # Calculate text size and position
    (text_w, text_h), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - text_h - 20), 
                  (text_w + 20, height - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text
    cv2.putText(frame, info_text, (15, height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_legend(frame: np.ndarray, track_colors: Dict[int, Tuple[int, int, int]], 
                  track_info: Dict[int, Dict]) -> np.ndarray:
    """Create a legend showing track information"""
    height, width = frame.shape[:2]
    
    if not track_colors:
        return frame
    
    # Legend parameters
    legend_width = 300
    legend_height = 30 * len(track_colors) + 40
    legend_x = width - legend_width - 10
    legend_y = 10
    
    # Draw legend background
    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), 
                  (legend_x + legend_width, legend_y + legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Legend title
    cv2.putText(frame, "Tracks", (legend_x + 10, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Track entries
    y_offset = 50
    for track_id, color in track_colors.items():
        info = track_info.get(track_id, {})
        length = info.get('length', 0)
        conf = info.get('rep_confidence', 0)
        
        # Color box
        cv2.rectangle(frame, (legend_x + 10, legend_y + y_offset - 10), 
                      (legend_x + 30, legend_y + y_offset + 5), color, -1)
        
        # Track info text
        text = f"ID:{track_id} ({length} det, {conf:.2f})"
        cv2.putText(frame, text, (legend_x + 40, legend_y + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
    
    return frame

def visualize_tracking_results(tracking_json_path: Path, videos_root: Path, 
                              output_dir: Path, config: Dict, 
                              sample_frames: bool = True, max_frames: int = 100,
                              create_video: bool = False) -> bool:
    """
    Create visualization of tracking results
    
    Args:
        tracking_json_path: Path to tracking JSON file
        videos_root: Directory containing video files
        output_dir: Directory to save visualization frames
        config: Pipeline configuration
        sample_frames: If True, sample frames evenly across video
        max_frames: Maximum number of frames to visualize
    
    Returns:
        True if successful, False otherwise
    """
    print(f"📹 Processing tracking visualization: {tracking_json_path.name}")
    
    # Load tracking data
    try:
        tracking_data = load_tracking_results(tracking_json_path)
    except Exception as e:
        print(f"❌ Error loading tracking data: {e}")
        return False
    
    # Get video path
    video_path = get_video_path(tracking_data, videos_root)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    # Extract tracking information
    tracks = tracking_data.get('tracks', [])
    if not tracks:
        print(f"⚠️  No tracks found in {tracking_json_path.name}")
        return False
    
    video_info = tracking_data.get('video_info', {})
    fps_effective = video_info.get('fps_effective', 15.0)
    
    # Generate colors for tracks
    track_colors = {}
    colors = generate_track_colors(len(tracks))
    for i, track in enumerate(tracks):
        track_colors[track['track_id']] = colors[i]
    
    # Create track info dictionary
    track_info = {track['track_id']: track for track in tracks}
    
    # Build frame-to-detections mapping
    frame_detections = {}
    for track in tracks:
        for detection in track['detections']:
            frame_idx = detection['frame']
            if frame_idx not in frame_detections:
                frame_detections[frame_idx] = []
            
            # Add track ID and color to detection
            det_with_track = detection.copy()
            det_with_track['track_id'] = track['track_id']
            det_with_track['color'] = track_colors[track['track_id']]
            frame_detections[frame_idx].append(det_with_track)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frames to process
    detection_frames = sorted(frame_detections.keys())
    if sample_frames and len(detection_frames) > max_frames:
        # Sample frames evenly
        step = len(detection_frames) // max_frames
        sampled_frames = detection_frames[::step][:max_frames]
    else:
        sampled_frames = detection_frames[:max_frames]
    
    print(f"  📊 Visualizing {len(sampled_frames)} frames with detections")
    print(f"  🎨 {len(tracks)} tracks with distinct colors")
    
    # Create output directory
    video_output_dir = output_dir / tracking_json_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer if creating video
    video_writer = None
    if create_video:
        # Get frame dimensions
        cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frames[0])
        ret, sample_frame = cap.read()
        if ret:
            height, width = sample_frame.shape[:2]
            fps_out = min(fps_effective, 10.0)  # Reasonable output FPS
            
            video_filename = f"{tracking_json_path.stem}_tracking.mp4"
            video_path = video_output_dir / video_filename
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps_out, (width, height))
            print(f"  🎬 Creating video: {video_filename} at {fps_out:.1f}fps")
    
    frames_saved = 0
    
    for frame_idx in tqdm(sampled_frames, desc="Creating visualizations"):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Draw all detections for this frame
        detections = frame_detections[frame_idx]
        for detection in detections:
            frame = draw_bbox_with_info(
                frame, detection['bbox'], detection['track_id'],
                detection['confidence'], detection['src'], 
                detection['color'], frame_idx
            )
        
        # Add frame info overlay
        frame = create_frame_info_overlay(
            frame, frame_idx, total_frames, fps_effective, len(detections)
        )
        
        # Add legend
        frame = create_legend(frame, track_colors, track_info)
        
        # Save frame if not creating video only
        if not create_video:
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = video_output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
        
        # Add frame to video if creating video
        if video_writer is not None:
            video_writer.write(frame)
        
        frames_saved += 1
    
    cap.release()
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"  🎬 Video saved: {video_filename}")
    
    if create_video:
        print(f"  ✅ Created tracking video with {frames_saved} frames")
    else:
        print(f"  ✅ Saved {frames_saved} visualization frames to {video_output_dir}")
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize ByteTrack tracking results')
    parser.add_argument('--config', default='config/pipeline.yaml', 
                       help='Pipeline configuration file')
    parser.add_argument('--tracking-json', 
                       help='Tracking JSON file or directory (default: from config)')
    parser.add_argument('--video-root', 
                       help='Video directory (default: from config)')
    parser.add_argument('--output-dir', default='outputs/preview', 
                       help='Output directory for visualizations')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames to visualize per video')
    parser.add_argument('--sample-frames', action='store_true', default=True,
                       help='Sample frames evenly (vs all detection frames)')
    parser.add_argument('--create-video', action='store_true', 
                       help='Create MP4 videos instead of individual frames')
    return parser.parse_args()

def main():
    """Main visualization function"""
    print("🎬 ByteTrack Tracking Visualization")
    print("=" * 50)
    
    args = parse_args()
    config = load_config(args.config)
    
    # Resolve paths
    videos_root = Path(args.video_root or config['paths']['videos_raw'])
    tracking_json_path = args.tracking_json or config['paths']['tracks_json']
    output_dir = Path(args.output_dir)
    
    print(f"📂 Tracking JSONs: {tracking_json_path}")
    print(f"🎬 Videos: {videos_root}")
    print(f"📁 Output: {output_dir}")
    print()
    
    # Get tracking JSON files
    tracking_path = Path(tracking_json_path)
    if tracking_path.is_file():
        tracking_files = [tracking_path]
    elif tracking_path.is_dir():
        tracking_files = sorted(tracking_path.glob("*.json"))
    else:
        print(f"❌ Tracking path not found: {tracking_json_path}")
        return
    
    if not tracking_files:
        print("❌ No tracking JSON files found!")
        return
    
    print(f"📊 Found {len(tracking_files)} tracking files to visualize")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each tracking file
    successful = 0
    total = len(tracking_files)
    
    for i, tracking_file in enumerate(tracking_files, 1):
        print(f"\n🎥 [{i}/{total}] {tracking_file.name}")
        
        if visualize_tracking_results(
            tracking_file, videos_root, output_dir, config,
            args.sample_frames, args.max_frames, args.create_video
        ):
            successful += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Visualization Complete!")
    print(f"   ✅ Successful: {successful}/{total}")
    print(f"   📁 Output directory: {output_dir}")
    
    if successful > 0:
        print(f"\n🖼️  View results:")
        print(f"   Open frames in: {output_dir}")
        print(f"   Each video has its own subdirectory with frames")

if __name__ == "__main__":
    main()