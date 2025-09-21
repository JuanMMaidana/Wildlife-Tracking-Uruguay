# Validation Notes — Auto-label Pipeline

**Review Sessions:** _(add date + reviewer + dataset subset)_

## 1. Crop Quality Checks
- Observed species:
- Issues found (blurry, off-center, duplicates):
- Parameter tweaks to try next run:

## 2. Species Mapping Accuracy
- Regex hits reviewed:
- False positives / negatives:
- Filenames requiring manual overrides:

## 3. Track Statistics
- Tracks reviewed (IDs):
- Dwell time anomalies:
- Short track behaviour (< min_track_len):

## 4. Action Items
- [ ] Update `config/species_map.yaml` patterns
- [ ] Adjust autolabel thresholds (`neighbors`, `crop_padding`, `max_crops_per_track`)
- [ ] Re-run `scripts/31_autolabel_from_filenames.py` and regenerate summary
- [ ] Sync findings with README / guides

---
_Add screenshots or sample crops references here once available._
