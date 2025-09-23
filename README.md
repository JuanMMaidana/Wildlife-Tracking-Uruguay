# MegaDetector Pipeline for Camera Trap Analysis

A computer vision pipeline for Uruguayan fauna recognition in camera trap videos using MegaDetector, ByteTrack, and species classification.

## Overview

This pipeline processes camera trap videos to detect, track, and classify animals using a three-stage approach:

1. **Detection**: MegaDetector identifies animals in video frames
2. **Tracking**: ByteTrack groups detections into coherent animal tracks  
3. **Classification**: Species classifier identifies the animal type (13 Uruguayan species)

## Documentation & Guides

- `guides/README.md`: entry point for implementation progress
- `guides/GUIDE_BYTRACK.md`: ByteTrack feature branch checklist and tuning notes
- `guides/GUIDE_CLASSIFICATION.md`: Track → Crop → Classification → Counts roadmap with validation gates

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
wildlife-tracking-uruguay/
├── config/                  # Configuration files
│   ├── pipeline.yaml        # Main pipeline settings
│   └── classes.yaml         # Species definitions (13 classes)
├── guides/                  # Step-by-step implementation guides
│   ├── GUIDE_BYTRACK.md     # ByteTrack feature branch
│   └── GUIDE_CLASSIFICATION.md
├── scripts/                 # Processing scripts
│   ├── 00_probe_gpu.py      # System check
│   ├── 10_run_md_batch.py   # MegaDetector batch processing
│   ├── 20_run_tracking.py   # ByteTrack production runner (Hungarian + tuned thresholds)
│   ├── 20_track_bytrack.py  # Legacy prototype (kept for reference)
│   └── md_scripts/          # MegaDetector helpers & sweeps
├── data/                    # Data directories (gitignored)
│   ├── videos_raw/          # Original video files
│   ├── md_json/             # MegaDetector outputs
│   ├── tracking_json/       # ByteTrack outputs
│   ├── crops/               # Auto-labeled crops (planned)
│   └── datasets/            # Training datasets (planned)
├── experiments/             # Calibration + validation studies
└── models/                  # Model weights (gitignored)
    ├── detectors/           # MegaDetector weights
    └── classifier/          # Species classifier checkpoints (planned)
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
python scripts/20_run_tracking.py

# Step 3: Auto-label crops (planned)
# python scripts/31_autolabel_from_filenames.py --config config/pipeline.yaml ...

# Step 4: Train species classifier (planned)
# python training/train_classifier.py --config config/pipeline.yaml ...
# Example evaluation (after training)
# python training/eval_classifier.py --config config/pipeline.yaml --split test

# Aggregate counts per video (after predictions available)
# python scripts/40_counts_by_species.py --manifest data/crops_manifest.csv --predictions experiments/exp_003_species/predictions_test.csv
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
- MegaDetector batch processing (`scripts/10_run_md_batch.py`)
- ByteTrack production runner with Hungarian assignment (`scripts/20_run_tracking.py`)

🚧 **In Progress:**
- Tracking visualization + regression testing (`guides/GUIDE_BYTRACK.md` Steps 9-10)
- Classification pipeline design & validation (`guides/GUIDE_CLASSIFICATION.md` Phase 1)

🛠️ **Planned:**
- Auto-label crops from filename metadata (`scripts/31_autolabel_from_filenames.py`)
- Classifier training/evaluation scripts (`training/`)
- Counts by species analytics (`scripts/40_counts_by_species.py`)
- CVAT integration and manual review workflows

## Contributing

This is an active research project. Key areas needing development:

1. **Tracking QA**: Finish visualization + regression tests for ByteTrack outputs
2. **Autolabeling**: Build filename-based crop extraction with manual validation loop
3. **Classification**: Implement training/evaluation scripts and manage class balance
4. **Analytics & Docs**: Aggregate per-species counts, polish documentation, and integrate CVAT workflows

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

### Dataset Strategy

- **Tuning pool:** ~20 hand-curated videos processed on a Windows GPU workstation used to calibrate MegaDetector, ByteTrack, and upcoming classification steps
- **Scale-up plan:** once the full pipeline is validated, run a larger corpus with consistent species stems in filenames (e.g., `margay001.mp4`, `capybara002.mp4`) curated manually
- **Tracking:** keep tuning vs. production metrics separate to monitor generalization when the dataset expands

## License

[Add your license here]

## Related Projects

- [MegaDetector](https://github.com/microsoft/CameraTraps) - Animal detection in camera trap images
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool
