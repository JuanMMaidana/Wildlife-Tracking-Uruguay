#!/usr/bin/env python3
"""
ByteTrack-based tracking for MegaDetector outputs
Converts MD JSON detections into coherent animal tracks
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import cv2

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_thresholds(config: Dict):
    """Validate ByteTrack threshold hierarchy: md_export <= det_thresh <= track_thresh"""
    md_export = config['megadetector']['md_export_threshold']  # 0.20
    det_thresh = config['tracking']['det_thresh']              # 0.40  
    track_thresh = config['tracking']['track_thresh']          # 0.60
    
    if not (md_export <= det_thresh <= track_thresh):
        raise ValueError(
            f"Threshold hierarchy violated: "
            f"md_export_threshold({md_export}) <= det_thresh({det_thresh}) <= track_thresh({track_thresh})"
        )
    
    print(f"✅ Threshold hierarchy valid: {md_export} ≤ {det_thresh} ≤ {track_thresh}")

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

def main():
    """Main processing function"""
    args = parse_args()
    config = load_config(args.config)
    
    # Validate configuration
    validate_thresholds(config)
    
    # Resolve paths
    videos_root = Path(args.video_root or config['paths']['videos_raw'])
    md_json_path = args.md_json or config['paths']['md_json']
    out_json_dir = Path(args.out_json or config['paths']['tracks_json'])
    
    # Get files to process
    md_files = get_md_json_files(md_json_path)
    print(f"Found {len(md_files)} MD JSON files to process")
    
    if not md_files:
        print("No MD JSON files found!")
        return
    
    # Create output directory
    out_json_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for md_file in md_files:
        print(f"\n--- Processing {md_file.name} ---")
        
        try:
            # Load and adapt MD detections
            frame_detections, video_info = load_md_detections_adapter(
                md_file, videos_root, config
            )
            
            if not frame_detections:
                print(f"  No detections exported for {md_file.name}")
                continue
            
            # TODO: Implement ByteTrack algorithm here
            print(f"  TODO: Run ByteTrack on {len(frame_detections)} frames")
            
            # For now, just save adapter output for testing
            output_file = out_json_dir / f"{md_file.stem}_adapter_test.json"
            test_output = {
                'video': md_file.stem + '.mp4',
                'frame_detections': {str(k): v for k, v in frame_detections.items()},
                'video_info': video_info
            }
            
            with open(output_file, 'w') as f:
                json.dump(test_output, f, indent=2)
            
            print(f"  Saved test output: {output_file}")
            
        except Exception as e:
            print(f"Error processing {md_file.name}: {str(e)}")
            continue
    
    print(f"\nAdapter testing complete! Check {out_json_dir} for outputs")

if __name__ == "__main__":
    main()