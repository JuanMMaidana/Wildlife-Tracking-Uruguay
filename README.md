# MegaDetector Pipeline for Camera Trap Analysis

A computer vision pipeline for Uruguayan fauna recognition in camera trap videos using MegaDetector, ByteTrack, and species classification.

## Overview

This pipeline processes camera trap videos to detect, track, and classify animals using a three-stage approach:

1. **Detection**: MegaDetector identifies animals in video frames
2. **Tracking**: ByteTrack groups detections into coherent animal tracks  
3. **Classification**: Species classifier identifies the animal type (13 Uruguayan species)

## Target Species (13 classes)
- armadillo, bird, capybara, cow, dusky_legged_guan, gray_brocket
- hare, human, margay, skunk, wild_boar, unknown_animal, no_animal

## Pipeline Architecture

```
Video Files → MegaDetector → ByteTrack → Frame Sampling → Auto-Labeling → CVAT → Species Classifier
```

### Key Features
- **Auto-labeling with guardrails**: Uses filename metadata for weak supervision
- **Smart sampling**: Extracts diverse, high-quality crops from each track
- **CVAT integration**: Manual annotation workflow for quality control
- **Balanced dataset creation**: Tools for managing class distribution

## Project Structure

```
megadetector-pipeline/
├── config/              # Configuration files
│   ├── pipeline.yaml    # Main pipeline settings
│   └── classes.yaml     # Species definitions
├── scripts/             # Processing scripts
│   ├── 00_probe_gpu.py  # System check
│   ├── 10_run_md_batch.py    # MegaDetector batch processing
│   ├── 20_track_bytrack.py   # ByteTrack tracking
│   ├── 30_select_frames.py   # Frame sampling (TODO)
│   └── 40_autolabel_from_filename.py  # Auto-labeling (TODO)
├── data/                # Data directories (gitignored)
│   ├── videos_raw/      # Original video files
│   ├── md_json/         # MegaDetector outputs
│   ├── tracks_json/     # Tracking results
│   ├── candidates/      # Extracted frame crops
│   └── datasets/        # Final training datasets
└── models/              # Model weights (gitignored)
    ├── detectors/       # MegaDetector weights
    └── classifiers/     # Species classifier weights
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate megadetector-pipeline

# Or use existing environment with required packages
conda activate md  # Your existing environment
```

### 2. Download MegaDetector Weights

```bash
# Create models directory
mkdir -p models/detectors

# Download MegaDetector v5a weights
# From: https://github.com/microsoft/CameraTraps/releases
# Save as: models/detectors/md_v5a.0.0.pt
```

### 3. Prepare Videos

```bash
# Copy your video files to data/videos_raw/
# Recommended naming: species_001.mp4, species_002.mp4, etc.
mkdir -p data/videos_raw
# cp your_videos/*.mp4 data/videos_raw/
```

### 4. System Check

```bash
python scripts/00_probe_gpu.py
```

### 5. Run Pipeline

```bash
# Step 1: Run MegaDetector on all videos
python scripts/10_run_md_batch.py

# Step 2: Generate tracks from detections  
python scripts/20_track_bytrack.py

# Step 3: Extract and sample frames (TODO - needs implementation)
python scripts/30_select_frames.py

# Step 4: Auto-label from filenames (TODO - needs implementation)
python scripts/40_autolabel_from_filename.py
```

## Configuration

Edit `config/pipeline.yaml` to adjust:
- Detection thresholds and frame sampling rates
- Tracking parameters and quality filters  
- Auto-labeling rules and guardrails
- Output paths and logging settings

## Current Status

✅ **Completed:**
- Project structure and configuration
- MegaDetector batch processing (10_run_md_batch.py)
- ByteTrack integration framework (20_track_bytrack.py)

🚧 **TODO (Needs Human Input):**
- Complete tracking algorithm in `20_track_bytrack.py` (IoU-based matching)
- Frame sampling and crop extraction (30_select_frames.py)
- Auto-labeling with guardrails (40_autolabel_from_filename.py)
- CVAT integration scripts
- Species classifier training

## Contributing

This is an active research project. Key areas needing development:

1. **Tracking Algorithm**: Complete the IoU-based tracking in `20_track_bytrack.py`
2. **Frame Sampling**: Implement diverse frame selection strategies
3. **Auto-labeling**: Build robust filename-based labeling with quality controls
4. **Species Classification**: Train and integrate species classifier

## Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (tested with RTX series)
- **Storage**: ~50GB+ for video processing (temp files can be large)
- **RAM**: 16GB+ recommended for video processing

## Background

This pipeline is designed for camera trap footage from Uruguay, targeting 13 species classes. The approach combines state-of-the-art detection (MegaDetector) with practical annotation workflows (CVAT) and intelligent automation (auto-labeling with guardrails).

The auto-labeling strategy is particularly useful for camera trap scenarios where:
- Most videos contain 0-1 animals
- Filenames contain species metadata  
- Manual annotation is time-consuming
- Quality control is critical

## License

[Add your license here]

## Related Projects

- [MegaDetector](https://github.com/microsoft/CameraTraps) - Animal detection in camera trap images
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool