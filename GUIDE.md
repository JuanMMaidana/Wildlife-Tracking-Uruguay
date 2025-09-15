# ByteTrack Implementation Guide

**Branch:** `feat/bytetrack-implementation`  
**Goal:** Implement ByteTrack tracking stage after MegaDetector  
**Status:** 🚧 In Progress

## Implementation Steps

### ✅ Step 0: Planning & Setup
- Created feature branch `feat/bytetrack-implementation`
- Defined 10-step implementation plan
- Created this guide for progress tracking

### ✅ Step 1: Update config/pipeline.yaml with ByteTrack parameters
**Status:** Completed  
**Commit:** 2ceb01c  
**Key Decisions:**
- **Threshold hierarchy:** `md_export (0.20) ≤ det_thresh (0.40) ≤ track_thresh (0.60)`
- **FPS handling:** Read real FPS from each video using cv2, calculate `fps_effective = fps / frame_stride`
- **Class mapping:** `person → human` in adapter, keep `include_person: true`
- **Directory naming:** Use consistent `paths.tracks_json = "data/tracking_json"`

**Parameters chosen:**
```yaml
megadetector:
  md_export_threshold: 0.20   # When MD starts detecting
  md_view_threshold: 0.25     # Visual debugging only
  # Keep existing calibrated values: frame_stride=2, min_area_ratio=0.005

tracking:
  method: bytetrack
  track_thresh: 0.60          # HIGH - creates new tracks
  det_thresh: 0.40            # LOW - recovery only  
  match_thresh: 0.60          # IoU for associations
  track_buffer_s: 0.8         # Buffer time in seconds
  min_track_len: 3            # Minimum detections to keep
  nms_iou: 0.7               # Per-frame NMS
  use_kalman: false          # Start simple
  class_map:
    person: human
    animal: animal
  include_person: true        # Track humans (in 13 species)
```

### ✅ Step 2: MD JSON Adapter and Validation
**Status:** Completed  
**Commit:** ed18a55  
**Implemented:**
- ✅ Load MD JSON outputs
- ✅ Apply class mapping (person→human)  
- ✅ Validate threshold hierarchy
- ✅ Handle pixel bbox format (confirmed from existing MD script)
- ✅ Read real video FPS with cv2
- ✅ Build video paths from JSON stems: cow_0777.json → cow_0777.mp4
- ✅ Graceful error handling with fallbacks
- ✅ Preserve MD metadata + adapter statistics

### ✅ Step 3: Enhanced Track Class with Parameter Tuning
**Status:** Completed  
**Commit:** [pending]  
**Implemented:**
- ✅ Enhanced Track class with ByteTrack lifecycle support
- ✅ HIGH/LOW detection source tracking with metadata
- ✅ IoU prediction tracking and motion prediction (naive)
- ✅ Representative frame selection (highest confidence)
- ✅ Greedy assignment algorithm for detection-track matching
- ✅ Two-pass ByteTrack algorithm (HIGH pass + LOW recovery)
- ✅ Track state management (active/lost/finished)
- ✅ JSON export schema with full metadata
- ✅ Real-world testing with 16 camera trap videos
- ✅ Ultra-conservative parameter tuning for camera trap scenarios

**Parameter Optimization Results:**
- Original config: 4+ tracks per video (high fragmentation)
- Conservative config: 2-3 tracks per video  
- Ultra-conservative config: 1-2 tracks per video (optimal)
- Successful handling of "tail-wagging paradox" and motion prediction challenges

**Final Tuned Parameters:**
```yaml
tracking:
  track_thresh: 0.70      # Very strict new track creation
  det_thresh: 0.25        # Permissive LOW recovery 
  match_thresh: 0.35      # Ultra-permissive association
  track_buffer_s: 2.5     # Long memory for gaps
  min_track_len: 8        # Filter spurious tracks
  nms_iou: 0.85          # Aggressive deduplication
```

**Key Insights:**
- Biological motion (tail wagging, pose changes) challenges tracking consistency
- LOW-confidence recovery essential for maintaining track continuity
- Ultra-permissive association critical for stationary animals with moving parts

### ⏳ Step 4: HIGH Pass Implementation
**Status:** Pending
- Hungarian matching for confident detections
- Create new tracks for unmatched HIGH detections

### ⏳ Step 5: LOW Pass Recovery
**Status:** Pending
- Recovery-only for missed tracks
- No new track creation from LOW detections

### ⏳ Step 6: NMS and Hungarian Integration
**Status:** Pending
- Per-frame NMS before HIGH/LOW split
- Hungarian assignment optimization

### ⏳ Step 7: CLI Interface
**Status:** Pending
- Full argument parsing
- Config validation

### ⏳ Step 8: JSON Export Schema
**Status:** Pending
- Exact schema with metadata
- fps_effective, tracking_code_version, git SHA

### ⏳ Step 9: Visualization Script
**Status:** Pending
- Track overlays with H/L labels
- Color coding by track ID

### ⏳ Step 10: Testing & Validation
**Status:** Pending
- End-to-end testing
- Create exp_002_tracking/ artifacts

## Key Technical Decisions

### Threshold Hierarchy Validation
```python
def validate_thresholds(config):
    md_export = config['megadetector']['md_export_threshold']
    det_thresh = config['tracking']['det_thresh'] 
    track_thresh = config['tracking']['track_thresh']
    
    if not (md_export <= det_thresh <= track_thresh):
        raise ValueError(f"Threshold hierarchy violated: {md_export} <= {det_thresh} <= {track_thresh}")
```

### FPS Calculation Strategy
```python
# Read actual FPS from video file
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
fps_effective = video_fps / frame_stride

# Convert buffer time to frames
track_buffer_frames = round(track_buffer_s * fps_effective)
```

### Schema Requirements
- `schema_version: "1.0"`
- `fps_effective` calculated per video
- `tracking_code_version: "git:<short_sha>"`
- Track detections with `src: "high"|"low"` and `iou_pred`

## Directory Structure
```
data/
├── videos_raw/         # Input videos
├── md_json/           # MegaDetector outputs  
└── tracking_json/     # ByteTrack outputs (NEW)

outputs/
└── preview/           # Visualization outputs (NEW)

experiments/
└── exp_002_tracking/  # Tracking validation (NEW)
```

## Validation Checklist
- [x] Threshold hierarchy: `md_export ≤ det_thresh ≤ track_thresh`
- [x] FPS read from video, fps_effective calculated correctly
- [ ] HIGH detections create tracks, LOW only recovers
- [x] Class mapping `person→human` applied
- [ ] JSON schema matches specification exactly
- [ ] Visualization shows H/L labels and track IDs
- [x] track_buffer_s converted to frames using fps_effective
- [ ] Git SHA logged in tracking_code_version

## Commits Log
- [x] Step 1: Config updates with ByteTrack parameters (2ceb01c)
- [x] Step 2: MD JSON adapter with validation (ed18a55)
- [ ] Step 3: Enhanced Track class  
- [ ] Step 4: HIGH pass implementation
- [ ] Step 5: LOW pass recovery
- [ ] Step 6: NMS and Hungarian matching
- [ ] Step 7: CLI interface
- [ ] Step 8: JSON export schema
- [ ] Step 9: Visualization script
- [ ] Step 10: Testing and validation

## Notes & Issues
- Conservative threshold set: reduces noise, allows ByteTrack recovery
- `match_thresh=0.6` - can lower to 0.5 if missing small/fast animals
- Single consistent directory naming: `tracking_json` throughout codebase