#!/usr/bin/env python3
"""
ByteTrack-based tracking for MegaDetector outputs
Converts MD JSON detections into coherent animal tracks
"""

import argparse
import json
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_thresholds(config: Dict):
    """Enhanced validation of ByteTrack threshold hierarchy and configuration"""
    md_export = config['megadetector']['md_export_threshold']
    det_thresh = config['tracking']['det_thresh']
    track_thresh = config['tracking']['track_thresh']
    
    print("🔍 Validating ByteTrack configuration...")
    
    # Validate threshold hierarchy
    if not (md_export <= det_thresh <= track_thresh):
        print(f"❌ THRESHOLD HIERARCHY ERROR:")
        print(f"   Current: md_export({md_export}) > det_thresh({det_thresh}) > track_thresh({track_thresh})")
        print(f"   Required: md_export_threshold ≤ det_thresh ≤ track_thresh")
        print(f"   Recommendation: Lower md_export_threshold to {det_thresh} or adjust thresholds")
        raise ValueError("Threshold hierarchy validation failed - see above for details")
    
    # Validate reasonable ranges
    warnings = []
    if track_thresh < 0.3:
        warnings.append(f"track_thresh ({track_thresh}) very low - may create many spurious tracks")
    if track_thresh > 0.8:
        warnings.append(f"track_thresh ({track_thresh}) very high - may miss legitimate tracks")
    if det_thresh < 0.1:
        warnings.append(f"det_thresh ({det_thresh}) very low - may recover too much noise")
    
    # Check for ultra-conservative mode
    match_thresh = config['tracking']['match_thresh']
    if match_thresh < 0.4:
        print(f"📊 Ultra-conservative tracking mode detected (match_thresh={match_thresh})")
        print("   This configuration optimized for stationary animals with pose variation")
    
    print(f"✅ Threshold hierarchy valid: {md_export} ≤ {det_thresh} ≤ {track_thresh}")
    
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print(f"🎯 Tracking mode: {config['tracking']['method']} with hungarian assignment")
    print(f"📐 Parameters: buffer={config['tracking']['track_buffer_s']}s, min_len={config['tracking']['min_track_len']}")
    print()

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
    """Enhanced Track class with ByteTrack lifecycle support"""
    
    def __init__(self, track_id: int, detection: dict, frame_idx: int):
        """Initialize track with first detection"""
        self.track_id = track_id
        self.detections = []           # List of detection dicts with metadata
        self.class_name = detection['class']  # 'animal' or 'human'
        self.start_frame = frame_idx
        self.last_frame = frame_idx
        self.misses = 0                # Consecutive frames without match
        
        # Add first detection with metadata
        self.add_detection(detection, frame_idx, src='high', iou_pred=1.0)
    
    def add_detection(self, detection: dict, frame_idx: int, src: str, iou_pred: float):
        """Add detection with ByteTrack metadata"""
        detection_entry = {
            'frame': frame_idx,
            'bbox': detection['bbox'],     # [x, y, w, h] in pixels
            'confidence': detection['score'],
            'src': src,                    # 'high' or 'low'
            'iou_pred': iou_pred          # IoU with predicted bbox
        }
        
        self.detections.append(detection_entry)
        self.last_frame = frame_idx
        self.misses = 0  # Reset miss counter
    
    def predict_next_bbox(self) -> List[float]:
        """Motion prediction (naive: return last bbox)"""
        if not self.detections:
            return [0, 0, 0, 0]
        return self.detections[-1]['bbox']
    
    def mark_miss(self):
        """Increment miss counter (called when no match found)"""
        self.misses += 1
    
    def should_remove(self, current_frame: int, track_buffer_frames: int) -> bool:
        """Check if track should be removed based on buffer"""
        return self.misses > track_buffer_frames
    
    def get_representative_frame(self) -> dict:
        """Get representative frame (highest confidence detection)"""
        if not self.detections:
            return None
        
        return max(self.detections, key=lambda d: d['confidence'])
    
    def to_dict(self) -> dict:
        """Export to JSON schema format for final output"""
        rep_frame = self.get_representative_frame()
        
        return {
            'track_id': self.track_id,
            'class': self.class_name,
            'start_frame': self.start_frame,
            'end_frame': self.last_frame,
            'length': len(self.detections),
            'rep_frame': rep_frame['frame'] if rep_frame else self.start_frame,
            'rep_confidence': rep_frame['confidence'] if rep_frame else 0.0,
            'rep_bbox': rep_frame['bbox'] if rep_frame else [0, 0, 0, 0],
            'detections': self.detections
        }

def get_video_info(video_path: Path) -> Optional[Dict]:
    """Read actual video FPS and dimensions using cv2"""
    if not video_path.exists():
        return None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return {
            'fps': fps,
            'width': width, 
            'height': height,
            'total_frames': total_frames
        }
    finally:
        cap.release()

def get_video_path_from_md_json(md_json_path: Path, videos_root: Path) -> Path:
    """Build video path from MD JSON filename: cow_0777.json -> cow_0777.mp4"""
    video_stem = md_json_path.stem  # Remove .json extension
    video_path = videos_root / f"{video_stem}.mp4"
    return video_path

def load_md_detections_adapter(md_json_path: Path, videos_root: Path, config: Dict) -> Tuple[Dict, Dict]:
    """
    Convert MegaDetector JSON to ByteTrack format
    
    Returns:
        frame_detections: dict[frame_idx] -> list[Detection]
        video_info: metadata including fps_effective
    """
    # 1. Load MD JSON (existing format)
    with open(md_json_path, 'r') as f:
        md_data = json.load(f)
    
    # 2. Get video path and info
    video_path = get_video_path_from_md_json(md_json_path, videos_root)
    video_info = get_video_info(video_path)
    
    # 3. Handle missing video with fallbacks
    if video_info is None:
        warnings.warn(f"Cannot read video info from {video_path}, using fallbacks")
        # Use fallbacks from config
        fallback_fps = 15  # Conservative estimate
        video_info = {
            'fps': fallback_fps,
            'width': None,   # Will keep bboxes as-is 
            'height': None,
            'total_frames': None,
            'video_path': str(video_path),
            'video_found': False
        }
    else:
        video_info['video_path'] = str(video_path)
        video_info['video_found'] = True
    
    # 4. Calculate effective FPS
    frame_stride = config['megadetector']['frame_stride']
    fps_effective = video_info['fps'] / frame_stride
    video_info['fps_effective'] = fps_effective
    
    # 5. Extract configuration
    md_export_thresh = config['megadetector']['md_export_threshold']  # 0.20
    class_map = config['tracking']['class_map']
    include_person = config['tracking']['include_person']
    
    print(f"Processing {md_json_path.name}: fps={video_info['fps']:.1f}, fps_effective={fps_effective:.1f}")
    
    # 6. Convert detections frame by frame  
    frame_detections = {}
    total_detections = 0
    exported_detections = 0
    
    for frame_data in md_data['detections']:
        frame_idx = frame_data['frame']
        detections = []
        
        for det in frame_data['detections']:
            total_detections += 1
            
            # Filter by md_export_threshold (0.20)
            if det['confidence'] < md_export_thresh:
                continue
                
            # Map MegaDetector class to generic species
            # det['class']: 0=animal, 1=person (from MD output)
            if det['class'] == 1:
                mapped_class = class_map.get('person', 'human')
            else:
                mapped_class = class_map.get('animal', 'animal')
            
            # Apply include_person filter
            if mapped_class == 'human' and not include_person:
                continue
            
            # Bbox is already in pixel format [x, y, w, h] from your MD script
            bbox = det['bbox']  # Already [x, y, w, h] in pixels
            
            detections.append({
                'bbox': bbox,           # [x, y, w, h] in pixels
                'score': det['confidence'],
                'class': mapped_class,  # 'animal' or 'human'
                'area': det.get('area', bbox[2] * bbox[3])  # w * h
            })
            
            exported_detections += 1
        
        if detections:
            frame_detections[frame_idx] = detections
    
    # 7. Preserve MD metadata in video_info
    if 'info' in md_data:
        md_info = md_data['info']
        video_info.update({
            'md_total_detections': md_info.get('total_detections', total_detections),
            'md_frames_with_detections': md_info.get('frames_with_detections', len(md_data['detections'])),
            'md_model_config': md_info.get('model_config', {}),
            'md_frame_stride': md_info.get('frame_stride', frame_stride)
        })
    
    # 8. Add adapter stats
    video_info.update({
        'adapter_total_detections': total_detections,
        'adapter_exported_detections': exported_detections,
        'adapter_md_export_threshold': md_export_thresh,
        'adapter_frames_with_detections': len(frame_detections)
    })
    
    print(f"  Exported {exported_detections}/{total_detections} detections "
          f"across {len(frame_detections)} frames")
    
    return frame_detections, video_info

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ByteTrack tracking for MegaDetector outputs')
    parser.add_argument('--config', default='config/pipeline.yaml', help='Config file')
    parser.add_argument('--md-json', help='MD JSON file or directory (default: from config)')
    parser.add_argument('--video-root', help='Video directory (default: from config)')
    parser.add_argument('--out-json', help='Output JSON directory (default: from config)')
    parser.add_argument('--viz-out', help='Visualization output directory (optional)')
    return parser.parse_args()

def get_md_json_files(md_json_path: str) -> List[Path]:
    """Get list of MD JSON files to process"""
    path = Path(md_json_path)
    
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.glob("*.json"))
    else:
        raise ValueError(f"MD JSON path not found: {md_json_path}")

def hungarian_assignment(cost_matrix: np.ndarray, match_thresh: float) -> List[Tuple[int, int]]:
    """Optimal Hungarian assignment for detection-track matching"""
    matches = []
    if cost_matrix.size == 0:
        return matches
    
    # Convert IoU to cost (Hungarian minimizes, but we want to maximize IoU)
    cost_matrix_inv = 1.0 - cost_matrix
    
    # Set high cost for matches below threshold (effectively infinite cost)
    cost_matrix_inv[cost_matrix < match_thresh] = 1e6
    
    # Hungarian assignment
    det_indices, track_indices = linear_sum_assignment(cost_matrix_inv)
    
    # Filter valid matches (only those above threshold)
    for det_idx, track_idx in zip(det_indices, track_indices):
        if cost_matrix[det_idx, track_idx] >= match_thresh:
            matches.append((det_idx, track_idx))
    
    return matches

def apply_nms_per_frame(detections: List[dict], nms_threshold: float) -> List[dict]:
    """Apply Non-Maximum Suppression to detections in a single frame"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []
    
    for detection in sorted_dets:
        should_keep = True
        for kept_det in keep:
            if calculate_iou(detection['bbox'], kept_det['bbox']) > nms_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(detection)
    
    return keep

def get_git_version() -> str:
    """Get git commit SHA for reproducibility"""
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                    cwd=Path(__file__).parent, text=True).strip()
        return f"git:{sha}"
    except:
        return "unknown"

def run_bytetrack(frame_detections: Dict, video_info: Dict, config: Dict) -> Dict:
    """
    Run ByteTrack algorithm on frame detections
    
    Args:
        frame_detections: dict[frame_idx] -> list[detection]
        video_info: video metadata including fps_effective
        config: pipeline configuration
    
    Returns:
        tracking results in JSON schema format
    """
    tracking_config = config['tracking']
    
    # Extract thresholds
    track_thresh = tracking_config['track_thresh']  # 0.70 - HIGH threshold
    det_thresh = tracking_config['det_thresh']      # 0.25 - LOW threshold  
    match_thresh = tracking_config['match_thresh']  # 0.35 - IoU threshold
    min_track_len = tracking_config['min_track_len'] # 8 - minimum track length
    nms_threshold = tracking_config['nms_iou']      # 0.85 - NMS threshold
    
    # Calculate track buffer in frames
    track_buffer_s = tracking_config['track_buffer_s']  # 0.8 seconds
    fps_effective = video_info['fps_effective']
    track_buffer_frames = round(track_buffer_s * fps_effective)
    
    print(f"  ByteTrack config: HIGH={track_thresh}, LOW={det_thresh}, "
          f"match_IoU={match_thresh}, NMS={nms_threshold}, buffer={track_buffer_frames}f")
    
    # Initialize tracking state
    active_tracks = []    # Currently active tracks
    lost_tracks = []      # Recently lost tracks (within buffer)
    finished_tracks = []  # Completed tracks
    next_track_id = 1
    
    # Process frames in order
    frames = sorted(frame_detections.keys())
    
    for frame_idx in frames:
        detections = frame_detections[frame_idx]
        
        if not detections:
            # No detections - mark all tracks as missed
            for track in active_tracks:
                track.mark_miss()
            for track in lost_tracks:
                track.mark_miss()
            continue
        
        # Apply NMS to remove duplicate detections
        nms_detections = apply_nms_per_frame(detections, nms_threshold)
        
        # Split NMS-filtered detections into HIGH and LOW confidence
        high_detections = [d for d in nms_detections if d['score'] >= track_thresh]
        low_detections = [d for d in nms_detections if det_thresh <= d['score'] < track_thresh]
        
        print(f"    Frame {frame_idx}: {len(high_detections)} HIGH, {len(low_detections)} LOW detections")
        
        # HIGH PASS: Match high-confidence detections with active tracks
        if active_tracks and high_detections:
            # Calculate IoU matrix between HIGH detections and active tracks
            iou_matrix = np.zeros((len(high_detections), len(active_tracks)))
            for det_idx, detection in enumerate(high_detections):
                for track_idx, track in enumerate(active_tracks):
                    predicted_bbox = track.predict_next_bbox()
                    iou_matrix[det_idx, track_idx] = calculate_iou(detection['bbox'], predicted_bbox)
            
            # Find matches using Hungarian assignment (optimal)
            matches = hungarian_assignment(iou_matrix, match_thresh)
            
            # Update matched tracks
            matched_det_indices = set()
            matched_track_indices = set()
            
            for det_idx, track_idx in matches:
                detection = high_detections[det_idx]
                track = active_tracks[track_idx]
                iou_pred = iou_matrix[det_idx, track_idx]
                
                # Only match if same class
                if detection['class'] == track.class_name:
                    track.add_detection(detection, frame_idx, src='high', iou_pred=iou_pred)
                    matched_det_indices.add(det_idx)
                    matched_track_indices.add(track_idx)
            
            # Move unmatched active tracks to lost
            new_active_tracks = []
            for track_idx, track in enumerate(active_tracks):
                if track_idx in matched_track_indices:
                    new_active_tracks.append(track)
                else:
                    track.mark_miss()
                    lost_tracks.append(track)
            active_tracks = new_active_tracks
            
            # Create new tracks for unmatched HIGH detections
            for det_idx, detection in enumerate(high_detections):
                if det_idx not in matched_det_indices:
                    new_track = Track(next_track_id, detection, frame_idx)
                    active_tracks.append(new_track)
                    next_track_id += 1
        
        elif high_detections:
            # No active tracks - create new tracks for all HIGH detections
            for detection in high_detections:
                new_track = Track(next_track_id, detection, frame_idx)
                active_tracks.append(new_track)
                next_track_id += 1
        
        else:
            # No HIGH detections - mark all active tracks as missed
            for track in active_tracks:
                track.mark_miss()
            # Move all to lost
            lost_tracks.extend(active_tracks)
            active_tracks = []
        
        # LOW PASS: Attempt to recover lost tracks with low-confidence detections
        if lost_tracks and low_detections:
            # Calculate IoU matrix between LOW detections and lost tracks
            iou_matrix = np.zeros((len(low_detections), len(lost_tracks)))
            for det_idx, detection in enumerate(low_detections):
                for track_idx, track in enumerate(lost_tracks):
                    predicted_bbox = track.predict_next_bbox()
                    iou_matrix[det_idx, track_idx] = calculate_iou(detection['bbox'], predicted_bbox)
            
            # Find recovery matches using Hungarian assignment (optimal)
            matches = hungarian_assignment(iou_matrix, match_thresh)
            
            # Recover matched tracks
            recovered_track_indices = set()
            used_low_det_indices = set()
            
            for det_idx, track_idx in matches:
                detection = low_detections[det_idx]
                track = lost_tracks[track_idx]
                iou_pred = iou_matrix[det_idx, track_idx]
                
                # Only recover if same class
                if detection['class'] == track.class_name:
                    track.add_detection(detection, frame_idx, src='low', iou_pred=iou_pred)
                    active_tracks.append(track)
                    recovered_track_indices.add(track_idx)
                    used_low_det_indices.add(det_idx)
            
            # Remove recovered tracks from lost_tracks
            new_lost_tracks = []
            for track_idx, track in enumerate(lost_tracks):
                if track_idx not in recovered_track_indices:
                    track.mark_miss()
                    new_lost_tracks.append(track)
            lost_tracks = new_lost_tracks
        
        else:
            # No LOW detections or lost tracks - mark all lost tracks as missed
            for track in lost_tracks:
                track.mark_miss()
        
        # Track cleanup - remove tracks that exceeded buffer
        still_active = []
        for track in active_tracks:
            if track.should_remove(frame_idx, track_buffer_frames):
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active
        
        still_lost = []
        for track in lost_tracks:
            if track.should_remove(frame_idx, track_buffer_frames):
                finished_tracks.append(track)
            else:
                still_lost.append(track)
        lost_tracks = still_lost
    
    # Add remaining tracks to finished
    finished_tracks.extend(active_tracks)
    finished_tracks.extend(lost_tracks)
    
    # Filter tracks by minimum length and convert to output format
    valid_tracks = []
    for track in finished_tracks:
        if len(track.detections) >= min_track_len:
            valid_tracks.append(track.to_dict())
    
    # Sort tracks by track_id
    valid_tracks.sort(key=lambda x: x['track_id'])
    
    # Build final output with enhanced metadata
    result = {
        'schema_version': '1.0',
        'video': Path(video_info.get('video_path', 'unknown.mp4')).name,
        'tracking_code_version': get_git_version(),
        'video_info': {
            'fps': video_info['fps'],
            'fps_effective': fps_effective,
            'width': video_info.get('width'),
            'height': video_info.get('height'),
            'total_frames': video_info.get('total_frames'),
            'frame_size': [video_info.get('width'), video_info.get('height')]
        },
        'tracking_config': {
            'method': 'bytetrack',
            'assignment': 'hungarian',
            'track_thresh': track_thresh,
            'det_thresh': det_thresh,
            'match_thresh': match_thresh,
            'nms_iou': nms_threshold,
            'track_buffer_s': track_buffer_s,
            'track_buffer_frames': track_buffer_frames,
            'min_track_len': min_track_len,
            'frame_stride': config['megadetector']['frame_stride']
        },
        'tracks': valid_tracks,
        'summary': {
            'total_tracks': len(valid_tracks),
            'total_detections': sum(len(track['detections']) for track in valid_tracks),
            'frames_processed': len(frames),
            'avg_track_length': round(sum(len(track['detections']) for track in valid_tracks) / max(1, len(valid_tracks)), 1),
            'validation_status': {
                'threshold_hierarchy_valid': True,
                'assignment_method': 'hungarian',
                'nms_applied': True
            }
        }
    }
    
    return result

def main():
    """Main processing function"""
    print("🦁 ByteTrack Wildlife Tracking Pipeline")
    print("=" * 50)
    
    args = parse_args()
    config = load_config(args.config)
    
    # Enhanced configuration validation
    validate_thresholds(config)
    
    # Resolve paths
    videos_root = Path(args.video_root or config['paths']['videos_raw'])
    md_json_path = args.md_json or config['paths']['md_json']
    out_json_dir = Path(args.out_json or config['paths']['tracks_json'])
    
    # Get files to process
    md_files = get_md_json_files(md_json_path)
    print(f"📁 Found {len(md_files)} MD JSON files to process")
    print(f"📂 Input: {md_json_path}")
    print(f"📂 Output: {out_json_dir}")
    print(f"🎬 Videos: {videos_root}")
    
    if not md_files:
        print("❌ No MD JSON files found!")
        print("   Make sure MegaDetector has been run first: python scripts/10_run_md_batch.py")
        return
    
    # Create output directory
    out_json_dir.mkdir(parents=True, exist_ok=True)
    print()
    
    # Process each file with progress tracking
    processed = 0
    total_tracks = 0
    
    for i, md_file in enumerate(md_files, 1):
        print(f"🎥 [{i}/{len(md_files)}] Processing {md_file.name}")
        
        try:
            # Load and adapt MD detections
            frame_detections, video_info = load_md_detections_adapter(
                md_file, videos_root, config
            )
            
            if not frame_detections:
                print(f"  No detections exported for {md_file.name}")
                continue
            
            # Run ByteTrack algorithm
            tracks_data = run_bytetrack(frame_detections, video_info, config)
            
            # Save tracking results
            output_file = out_json_dir / f"{md_file.stem}.json"
            
            with open(output_file, 'w') as f:
                json.dump(tracks_data, f, indent=2)
            
            num_tracks = len(tracks_data['tracks'])
            total_tracks += num_tracks
            processed += 1
            
            print(f"  ✅ Saved: {output_file}")
            print(f"  📊 Created {num_tracks} tracks, avg_length={tracks_data['summary']['avg_track_length']}")
            
        except Exception as e:
            print(f"  ❌ Error processing {md_file.name}: {str(e)}")
            continue
    
    # Final summary
    print("\n" + "=" * 50)
    print(f"🎯 ByteTrack Processing Complete!")
    print(f"   📁 Processed: {processed}/{len(md_files)} videos")
    print(f"   🏷️  Total tracks: {total_tracks}")
    print(f"   📂 Output directory: {out_json_dir}")
    print(f"   🔧 Tracking version: {get_git_version()}")
    
    if processed < len(md_files):
        print(f"   ⚠️  {len(md_files) - processed} files failed - check error messages above")

if __name__ == "__main__":
    main()