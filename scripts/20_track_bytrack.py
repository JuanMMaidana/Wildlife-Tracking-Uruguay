#!/usr/bin/env python3
"""
ByteTrack-based Tracking Script
Takes MegaDetector JSON outputs and groups detections into tracks
Each track represents one animal moving through the video
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
from collections import defaultdict

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
    
    # Calculate intersection
    intersection_x1 = max(x1_min, x2_min)
    intersection_y1 = max(y1_min, y2_min)
    intersection_x2 = min(x1_max, x2_max)
    intersection_y2 = min(y1_max, y2_max)
    
    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0
    
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

class Track:
    """Represents a single animal track"""
    
    def __init__(self, track_id: int, frame: int, detection: Dict):
        self.track_id = track_id
        self.detections = [(frame, detection)]
        self.last_frame = frame
        self.lost_count = 0
        self.confidence_sum = detection['confidence']
        self.confidence_count = 1
    
    def add_detection(self, frame: int, detection: Dict):
        """Add a new detection to this track"""
        self.detections.append((frame, detection))
        self.last_frame = frame
        self.lost_count = 0
        self.confidence_sum += detection['confidence']
        self.confidence_count += 1
    
    def mark_lost(self):
        """Mark track as lost (no matching detection found)"""
        self.lost_count += 1
    
    def get_last_bbox(self) -> List[float]:
        """Get bounding box from most recent detection"""
        return self.detections[-1][1]['bbox']
    
    def get_mean_confidence(self) -> float:
        """Get average confidence across all detections"""
        return self.confidence_sum / self.confidence_count
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary format"""
        return {
            'track_id': self.track_id,
            'detections': [
                {
                    'frame': frame,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'area': det['area']
                }
                for frame, det in self.detections
            ],
            'length': len(self.detections),
            'frame_range': [self.detections[0][0], self.detections[-1][0]],
            'mean_confidence': self.get_mean_confidence()
        }

def process_detections_to_tracks(md_detections: Dict, config: Dict) -> Dict:
    """
    Convert MegaDetector detections to tracks using simple IoU-based tracking
    """
    tracking_config = config['tracking']
    match_thresh = tracking_config['match_thresh']
    track_buffer = tracking_config['track_buffer']
    min_box_area = tracking_config['min_box_area']
    
    tracks = []
    active_tracks = []
    lost_tracks = []
    next_track_id = 1
    
    # Process frames in order
    frame_detections = {}
    for frame_data in md_detections['detections']:
        frame_detections[frame_data['frame']] = frame_data['detections']
    
    frames = sorted(frame_detections.keys())
    
    for frame_idx in frames:
        detections = frame_detections[frame_idx]
        
        # Filter detections by minimum area
        valid_detections = [
            det for det in detections 
            if det['area'] >= min_box_area
        ]
        
        if not valid_detections:
            # No valid detections in this frame
            # Mark all active tracks as lost
            for track in active_tracks:
                track.mark_lost()
            continue
        
        # TODO(human): Implement the core tracking logic here
        # This function should:
        # 1. Match current frame detections with active tracks using IoU
        # 2. Update matched tracks with new detections
        # 3. Create new tracks for unmatched detections
        # 4. Handle lost tracks (move to lost_tracks or remove entirely)
        #
        # The matching algorithm should:
        # - Calculate IoU between each detection and each active track's last bbox
        # - Match detection to track with highest IoU if above match_thresh
        # - Handle one-to-one matching (each detection matches at most one track)
        # - Create new tracks for unmatched detections
        
        # After implementing the tracking logic, continue with cleanup:
        
        # Move lost tracks that exceeded buffer to finished tracks
        still_active = []
        for track in active_tracks:
            if track.lost_count > track_buffer:
                tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active
        
        # Also check lost_tracks for ones that can be removed
        still_lost = []
        for track in lost_tracks:
            if track.lost_count <= track_buffer:
                still_lost.append(track)
            else:
                tracks.append(track)
        lost_tracks = still_lost
    
    # Add remaining active and lost tracks to finished tracks
    tracks.extend(active_tracks)
    tracks.extend(lost_tracks)
    
    # Convert tracks to output format
    tracks_output = []
    for track in tracks:
        if len(track.detections) >= 1:  # Keep all tracks with at least 1 detection
            tracks_output.append(track.to_dict())
    
    # Sort tracks by track_id
    tracks_output.sort(key=lambda x: x['track_id'])
    
    result = {
        'video': md_detections['video'],
        'tracks': tracks_output,
        'info': {
            'total_tracks': len(tracks_output),
            'total_detections': sum(len(track['detections']) for track in tracks_output),
            'original_detections': md_detections['info']['total_detections'],
            'tracking_config': tracking_config
        }
    }
    
    return result

def load_md_detections(json_file: Path) -> Dict:
    """Load MegaDetector detection results"""
    with open(json_file, 'r') as f:
        return json.load(f)

def save_tracks(tracks_data: Dict, output_dir: str, video_name: str):
    """Save tracking results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_stem = Path(video_name).stem
    output_file = output_dir / f"{video_stem}.json"
    
    with open(output_file, 'w') as f:
        json.dump(tracks_data, f, indent=2)
    
    print(f"Saved tracks: {output_file}")

def main():
    """Main tracking function"""
    config = load_config()
    
    # Get MegaDetector JSON files
    md_json_dir = Path(config['paths']['md_json'])
    if not md_json_dir.exists():
        print(f"MegaDetector results directory not found: {md_json_dir}")
        print("Run 10_run_md_batch.py first")
        return
    
    json_files = list(md_json_dir.glob("*.json"))
    print(f"Found {len(json_files)} detection files to process")
    
    if not json_files:
        print("No detection files found!")
        return
    
    # Process each file
    for json_file in json_files:
        # Check if already processed
        video_stem = json_file.stem
        output_file = Path(config['paths']['tracks_json']) / f"{video_stem}.json"
        
        if output_file.exists():
            print(f"Skipping {json_file.name} (already processed)")
            continue
        
        try:
            print(f"Processing: {json_file.name}")
            
            # Load detections
            md_detections = load_md_detections(json_file)
            
            if not md_detections['detections']:
                print(f"  No detections found in {json_file.name}")
                # Still save empty result
                empty_result = {
                    'video': md_detections['video'],
                    'tracks': [],
                    'info': {
                        'total_tracks': 0,
                        'total_detections': 0,
                        'original_detections': 0,
                        'tracking_config': config['tracking']
                    }
                }
                save_tracks(empty_result, config['paths']['tracks_json'], md_detections['video'])
                continue
            
            # Process to tracks
            tracks_data = process_detections_to_tracks(md_detections, config)
            
            # Save results
            save_tracks(tracks_data, config['paths']['tracks_json'], md_detections['video'])
            
            # Print summary
            info = tracks_data['info']
            print(f"  {info['total_tracks']} tracks created")
            print(f"  {info['total_detections']} detections in tracks")
            print()
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
            continue
    
    print("Tracking complete!")

if __name__ == "__main__":
    main()