# Experiment 001 — MegaDetector Parameter Calibration

**Date:** TBD  
**Dataset:** 15 videos from @vids_test_MD (mixed: animals + empty scenes)  
**Goal:** Find balanced defaults for `conf_threshold`, `frame_stride`, `min_area_ratio`

## Setup
- **Pipeline:** `10_run_md_batch.py` (modified to use relative area filtering)
- **Params file:** `experiments/exp_001_md_calibration/params.yaml`
- **Command:** 
  ```bash
  python scripts/99_sweep_md.py --params experiments/exp_001_md_calibration/params.yaml
  ```

## Parameter Grid
- **conf_threshold:** [0.15, 0.20, 0.25] (vs. current 0.55)
- **frame_stride:** [1, 2, 5] (vs. current 5)
- **min_area_ratio:** [0.0025, 0.005, 0.01] (vs. current 10000px absolute)

## Results (to be completed after experiment)

### Proposed Defaults
- `conf_threshold = TBD`
- `frame_stride = TBD`  
- `min_area_ratio = TBD`

### Rationale
- TBD: Analysis of recall vs runtime trade-offs
- TBD: Performance on small animals (birds, hares) vs large animals (cattle, capybara)
- TBD: Impact of stride on fast-moving vs stationary animals

## Key Observations
- TBD: Detection quality improvements with lower confidence thresholds
- TBD: Efficiency gains from frame striding vs missed detections
- TBD: Relative area filtering performance across 1080p videos

## Data Files
- **Results CSV:** `experiments/exp_001_md_calibration/results.csv`
- **Sample frames:** `experiments/exp_001_md_calibration/samples/`
- **Configuration used:** Logged in results CSV and individual JSON outputs

## Next Steps
- [ ] Update `config/pipeline.yaml` with optimal defaults
- [ ] Validate on larger dataset (50+ videos)
- [ ] Consider species-specific parameter tuning
- [ ] Document parameter selection methodology for portfolio

## Technical Notes
- Videos tested: 1920x1080 @ 30fps
- MegaDetector version: v5a
- Processing environment: RTX 3060 Ti, 8GB VRAM