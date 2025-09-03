# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wildlife-Tracking-Uruguay is a computer vision pipeline for Uruguayan fauna recognition in camera trap videos. The pipeline follows a three-stage architecture: MegaDetector → ByteTrack → Species Classification, targeting 13 species classes (armadillo, bird, capybara, cow, dusky_legged_guan, gray_brocket, hare, human, margay, skunk, wild_boar, unknown_animal, no_animal).

## Key Commands

### Environment Setup
```bash
# Activate the conda environment (user typically uses 'md' environment)
conda activate md

# Or create new environment
conda env create -f environment.yml
conda activate megadetector-pipeline
```

### System Verification
```bash
# Test CUDA, PyTorch, Ultralytics setup
python scripts/00_probe_gpu.py
```

### Pipeline Execution (Sequential)
```bash
# Step 1: MegaDetector batch processing (COMPLETED)
python scripts/10_run_md_batch.py

# Step 2: ByteTrack tracking (70% COMPLETE - needs human input)
python scripts/20_track_bytrack.py

# Step 3: Frame sampling and crop extraction (NOT IMPLEMENTED)
python scripts/30_select_frames.py

# Step 4: Auto-labeling from filenames (NOT IMPLEMENTED) 
python scripts/40_autolabel_from_filename.py
```

### Model Preparation
MegaDetector v5a weights must be downloaded manually:
- Source: https://github.com/microsoft/CameraTraps/releases
- Location: `models/detectors/md_v5a.0.0.pt`

## Architecture and Data Flow

### Pipeline Stages
1. **Detection (10_run_md_batch.py)**: Processes videos with frame stride, runs MegaDetector inference, outputs JSON per video to `data/md_json/`
2. **Tracking (20_track_bytrack.py)**: Groups detections into tracks using IoU-based matching, outputs to `data/tracks_json/`
3. **Sampling (30_select_frames.py)**: Extracts diverse crops from tracks with quality filters
4. **Auto-labeling (40_autolabel_from_filename.py)**: Applies weak supervision using filename metadata with guardrails

### Configuration System
- `config/pipeline.yaml`: Main pipeline parameters (thresholds, paths, hardware settings)
- `config/classes.yaml`: Species definitions (13 Uruguayan fauna classes)

Configuration is centralized and loaded by all scripts using `yaml.safe_load()`. Key parameters:
- MegaDetector: conf_threshold=0.55, frame_stride=5, min_area=10000px
- Tracking: match_thresh=0.8, track_buffer=30 frames
- Hardware: CUDA preferred, batch processing optimized

### Data Structure
```
data/
├── videos_raw/      # Input videos (.mp4 only)
├── md_json/         # MegaDetector outputs (1 JSON per video)  
├── tracks_json/     # Tracking results (track_id assignments)
├── candidates/      # Extracted frame crops for labeling
└── datasets/        # Final training datasets
```

## Critical Implementation Details

### MegaDetector Integration (10_run_md_batch.py)
- Uses Ultralytics YOLO API with custom MegaDetector weights
- Frame extraction with configurable stride (default: every 5th frame)
- Detection filtering by minimum area and confidence
- Output format: `{'video': str, 'detections': [{'frame': int, 'detections': [bbox, conf, class]}]}`

### ByteTrack Implementation (20_track_bytrack.py)
**STATUS: 70% COMPLETE - NEEDS HUMAN INPUT**
- IoU calculation functions implemented
- Track class structure defined with state management
- **MISSING**: Core tracking algorithm at lines 134-139 (TODO(human))
- Required implementation: IoU-based detection-to-track matching, new track creation, lost track handling

### Auto-labeling Strategy
The pipeline uses filename metadata for weak supervision (e.g., "margay_012.mp4" → margay label) with quality guardrails:
- Dominance rules: track must cover ≥80% of detection frames
- Quality filters: minimum confidence, track length, area thresholds  
- Manual review required for confusable species pairs

### CVAT Integration
Designed for manual annotation workflow using crops extracted from tracks, not full videos. This reduces annotation workload while maintaining quality control.

## Current Implementation Status

**COMPLETED (Production Ready):**
- Project infrastructure and configuration system
- MegaDetector batch processing with frame stride optimization
- Video file handling (supports .mp4 format)
- JSON-based data serialization between pipeline stages

**PARTIALLY IMPLEMENTED:**
- ByteTrack tracking framework (missing core algorithm implementation)

**NOT IMPLEMENTED:**
- Frame sampling with diversity and quality filters
- Auto-labeling with guardrails and conflict detection
- Species classifier training and integration
- CVAT import/export utilities

## Development Context

This is an active research project for Uruguayan camera trap footage (~170 videos). The user collaborates with multiple AI systems (GPT-5, Claude Code) and maintains active development in conda environment 'md' with CUDA GPU acceleration.

The pipeline is designed for camera trap scenarios where most videos contain 0-1 animals, filenames contain species metadata, and manual annotation time is critical to minimize.