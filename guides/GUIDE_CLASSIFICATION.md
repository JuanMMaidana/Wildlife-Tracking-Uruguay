# Classification Pipeline Guide

**Branch:** `feat/classification-pipeline`
**Goal:** End-to-end pipeline from `tracking_json` → crops → classifier → counts
**Status:** 🎯 Phase 1 Complete - Ready for Training

**Progress:** ✅ Phase 1: Validation & QA (Steps 0-5) | ⏳ Phase 2: Training & Production (Steps 6-9)

## Phase 1: Validation & QA ✅ COMPLETE

### ✅ Step 0: Planning & Alignment
- ✅ Created feature branch `feat/classification-pipeline`
- ✅ Documented scope in this guide to mirror ByteTrack workflow
- ✅ Confirmed inputs (`data/tracking_json/*.json`) and tuned thresholds from tracking stage
- ✅ Defined experiment IDs and output locations to avoid collisions with existing runs
- ✅ Working dataset: ~20 hand-curated videos (Windows GPU box) used for all hyperparameter tuning to date
- ✅ Expansion plan: once pipeline is validated, run larger renamed corpus (e.g., `margay001.mp4`, `capybara002.mp4`) with consistent species stems maintained manually
- ✅ Reorganized documentation into `guides/` directory with cross-linked README

### ✅ Step 1: Species Mapping Configuration
- Add `config/species_map.yaml` mapping filename regex patterns to 13 classes
- Enforce first-match-wins, error on multiple matches, fallback to `unknown_animal`
- Extend `config/pipeline.yaml` with classification section (paths, regex validation toggles)
- Unit-test loader (dedupe patterns, warn on unused classes)

```yaml
# config/species_map.yaml (sketch)
patterns:
  margay: [".*margay.*", "felis_wiedii_.*"]
  capybara: ["capybara_.*"]
  # ... remaining classes ...
  human: [".*human.*", ".*person.*"]
  no_animal: [".*empty.*", ".*noanimal.*"]
```

### ✅ Step 2: Autolabel From Filenames (`scripts/31_autolabel_from_filenames.py`)
- CLI args per spec (`--config`, `--species-map`, `--tracks-json`, `--video-root`, `--out-dir`, ...)
- For each track: infer species via regex → skip `no_animal`/`unknown_animal`
- Select representative frame ± `neighbors`, convert bbox→pixels, apply padding + clipping
- Write crops to `data/crops/<species>/<video_stem>__tid<id>__f<frame>.jpg`
- Append manifest rows to `data/crops_manifest.csv` with full metadata (video, track_id, frame, bbox, species, conf, dwell stats)
- Guardrails: error on conflicting matches, skip short tracks, log skipped cases, preserve deterministic ordering
- **Notation:** script numbering aligns with new classification stage; update README when rolling out
- ✅ Script in place; parameter tuning (`neighbors`, `min_track_len`, `max_crops_per_track`, `crop_padding`) deferred until manual validation session on Windows workstation

### ✅ Step 3: Manual Validation Loop
- ✅ Generated crops for 87 videos from dataset-v1 (912 total crops across 11 species)
- ✅ Performed human review confirming good species mapping and crop quality
- ✅ Validated dwell-time metadata and track statistics
- ✅ Confirmed hybrid sampling parameters produce high-quality representative crops
- ✅ Species mapping regex patterns working correctly with current dataset
- ✅ Manual validation passed - ready for Phase 2

### ✅ Step 4: Quantitative Crop QA (`experiments/exp_003_autolabel/`)
- ✅ Generated `summary.csv` with species statistics (species, n_videos, n_tracks, n_crops, median_track_len)
- ✅ Produced `report.md` with formatted per-species summary tables
- ✅ Created `video_inventory.md` documenting dataset composition
- ✅ Validated class coverage across 11 species (64-120 crops per species)
- ✅ Confirmed balanced dataset suitable for training
- ✅ Core QA artifacts complete and validated

## Phase 2: Training & Production

### ✅ Step 5: Dataset Splitting Strategy (`training/prepare_split.py`)
- **Implementation**: Dual splitting strategies (video-level vs crop-level) with CLI selection
- **Video-level splits** (`--strategy video`): Prevents data leakage, realistic evaluation, but poor class balance
- **Crop-level splits** (`--strategy crop`): Perfect species balance, adequate samples per class, but risk of data leakage
- **Recommended approach**: Use crop-level splits for training/development + additional unseen videos for manual validation
- **Current status**: 912 crops split into 634 train / 139 val / 139 test with excellent species balance
- ✅ Implemented and tested both strategies; crop-level selected for optimal training stability

### ⏳ Step 6: Classifier Training Scripts (`training/`)
- ✅ `training/train_classifier.py`: full data pipeline (manifest parsing, crop/ video split support, torchvision transforms) and baseline training loop (ResNet50/MobileNetV3, AdamW, cosine LR, class balancing, metrics logging, best-checkpoint saving)
- ⏳ `training/eval_classifier.py`: still needs model loading + metric export (top-1, per-class F1, confusion matrix)
- ⏳ Record training/eval configs in `experiments/exp_003_species/params.yaml`
- 🔄 Add dependency notes (torchvision extras, torchmetrics if used) to README/environment docs
- Output artefacts: `experiments/exp_003_species/metrics.csv`, `metrics.json`, `best_model.pt`

### ⏳ Step 7: Counts by Species (`scripts/40_counts_by_species.py`)
- Load classifier predictions per track (majority vote or rep-frame)
- Output `experiments/exp_004_counts/results.csv` with `video,species,n_tracks,avg_dwell_s`
- Generate plots (bar charts, dwell distributions) to `experiments/exp_004_counts/plots/`
- Handle `unknown_animal` gracefully (report but exclude from totals by default)

### ⏳ Step 8: Validation & Tests
- Smoke test: ensure ≥1 crop per eligible track on fixture video set
- Schema-check `data/crops_manifest.csv` (pydantic or jsonschema) for required columns & types
- Verify train/val/test split has zero video leakage; enforce via test harness
- Add GitHub Actions job (lint, mypy/ruff optional, smoke autolabel + classifier dry-run)
- Track status in `experiments/exp_003_autolabel/validation.md`

### ⏳ Step 9: Documentation & Portfolio Polish
- Update README with classification overview, sample crop grid, confusion matrix, counts plot
- Embed small artifact set under `experiments/exp_003_autolabel/` + `exp_004_counts/`
- Cross-link guides (`GUIDE.md` ↔ `GUIDE_CLASSIFICATION.md`)
- Prepare annotated figures for portfolio / presentation deck (store in `docs/plots/`)

## Dataset Splitting Strategy

### Video-Level vs Crop-Level Trade-offs

The pipeline supports two splitting strategies via `training/prepare_split.py --strategy [video|crop]`:

**Video-Level Splitting** (`--strategy video`):
- ✅ **Prevents data leakage**: All crops from same video stay in same split
- ✅ **Realistic evaluation**: True test of generalization to new camera locations
- ❌ **Poor species balance**: Some species get only 6-9 crops in val/test sets
- ❌ **Training instability**: Severe class imbalance affects learning

**Crop-Level Splitting** (`--strategy crop`):
- ✅ **Perfect species balance**: Each species gets proportional representation
- ✅ **Training stability**: Adequate samples for validation metrics (10+ per species)
- ✅ **Reliable evaluation**: Consistent validation performance tracking
- ❌ **Risk of data leakage**: Crops from same video may appear in different splits
- ❌ **Optimistic performance**: May overestimate real-world generalization

### Recommended Hybrid Approach

1. **Use crop-level splits for model development** (634 train / 139 val / 139 test)
2. **Keep additional unseen videos outside dataset-v1 for manual validation**
3. **Test trained model on new videos with visual inspection**
4. **Document performance gaps between crop-level metrics and real-world performance**

This approach gives you stable training with balanced classes while maintaining awareness of true generalization performance.

## Pipeline Data Flow
```
MegaDetector JSON → ByteTrack (`data/tracking_json/*.json`) →
Auto-label crops (`data/crops/`, `data/crops_manifest.csv`) →
Dataset splits (`experiments/exp_003_autolabel/splits.json|csv`) →
Train/Eval classifier (`models/classifier/*.pt`, metrics) →
Counts aggregation (`experiments/exp_004_counts/results.csv`)
```

## Configuration Updates
```yaml
paths:
  crops: "data/crops"
  crops_manifest: "data/crops_manifest.csv"
  classifier_models: "models/classifier"
  experiments: "experiments"
  # TODO: deprecate paths.candidates/autolabels once classification paths are live

classification:
  species_map: "config/species_map.yaml"
  crop_padding: 0.05
  neighbors: 2
  min_track_len: 6
  max_crops_per_track: 8
  skip_classes: ["no_animal", "unknown_animal"]
  split_strategy: "by_video"
  random_seed: 42
  model: "resnet50"
  epochs: 15
  learning_rate: 0.0001
  weight_decay: 0.0001
  image_size: 224
  balance_classes: true
  log_interval: 50
  dataloader_workers: 4
```

## Directory Structure (New & Updated)
```
data/
├── tracking_json/              # Input tracks from ByteTrack
├── crops/                      # Auto-labeled crops (gitignored)
│   └── <species>/<video_stem>__tid<id>__f<frame>.jpg
├── crops_manifest.csv          # Weak labels manifest

config/
├── pipeline.yaml               # Updated with classification section
└── species_map.yaml            # Regex-based species mapping

models/
└── classifier/                 # Trained classifier checkpoints (.pt)

training/
├── prepare_split.py
├── train_classifier.py
└── eval_classifier.py

scripts/
├── 31_autolabel_from_filenames.py
└── 40_counts_by_species.py

experiments/
├── exp_003_autolabel/
│   ├── summary.csv
│   ├── report.md
│   └── validation_notes.md
├── exp_003_species/
│   ├── params.yaml
│   ├── metrics.csv
│   └── confusion_matrix.png
└── exp_004_counts/
    ├── results.csv
    └── plots/
```

## Validation Checklist
- [ ] Regex map loads without conflicts; every pattern resolves to known class
- [ ] Autolabel script skips short tracks (`min_track_len`) and logs skipped reasons
- [ ] Manual review completed; validation_notes.md documents issues + resolutions
- [ ] Crops respect padding & frame bounds; manifest rows contain pixel-space bbox
- [ ] Automated QA metrics generated (class coverage, blur score, bbox area)
- [ ] Train/val/test split has zero shared video stems
- [ ] Classifier evaluation exports per-class metrics + confusion matrix asset
- [ ] Counts script matches manifest track totals and records dwell time correctly
- [ ] CI smoke tests pass locally and on GitHub Actions

## Commits Log
- [x] Step 0: Planning scaffold (`feat/classification-pipeline` created, guide added) - 4baa9ce
- [x] Step 1: Add species map config + loader tests
- [x] Step 2: Implement `31_autolabel_from_filenames.py` with hybrid sampling
- [x] Step 3: Manual validation completed (912 crops validated across 87 videos)
- [x] Step 4: Autolabel QA summaries (`exp_003_autolabel` artifacts complete)
- [x] Step 5: Dataset splitting with dual strategies (video/crop-level) - 481029e
- [ ] Step 6: Training/eval scripts under `training/`
- [ ] Step 7: Counts aggregation script + results
- [ ] Step 8: Validation harness & CI integration
- [ ] Step 9: Documentation + portfolio assets

## Technical Notes
- Assumption: one dominant species (or human/no_animal) per video; manual validation confirms applicability
- ByteTrack outputs already include representative frames and metadata; reuse for dwell-time estimation
- Regex mapping should be versioned and peer-reviewed—store examples alongside tests
- Limit crops per track (`max_crops_per_track`) to prevent class imbalance; log sampling decisions
- Keep GPU requirements in README (training stage may need mixed precision options)
- Expand `environment.yml` with pandas, torchmetrics, matplotlib, seaborn for analytics
- For reproducibility, persist random seeds and record config snapshots in experiment folders
- Align README/script numbering once classification stage is implemented (deprecate old TODO placeholders)
- Maintain separation between tuning pool (~20 videos) and production corpus; document when the larger, consistently named dataset is ingested

## Open Questions / Follow-ups
- Do we need manual overrides for multi-species videos? (e.g., sidecar CSV for corrections)
- Which backbone (ResNet50 vs MobileNetV3) performs best under resource constraints?
- Should counts aggregation include confidence thresholds or ensemble voting across frames?
- How to share crop thumbnails safely (privacy for human footage)?

---
Use this guide alongside `GUIDE.md` to keep tracking and classification workstreams aligned.
