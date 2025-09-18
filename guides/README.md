# Implementation Guides

This directory contains step-by-step implementation guides for the Wildlife-Tracking-Uruguay pipeline.

## Available Guides

### 📋 [GUIDE.md](./GUIDE.md) - ByteTrack Implementation
**Branch:** `feat/bytetrack-implementation`  
**Status:** ✅ Steps 1-8 Complete, ⏳ Steps 9-10 Pending  
**Goal:** Implement ByteTrack tracking stage after MegaDetector

**Key Achievements:**
- ✅ Ultra-conservative parameter tuning for camera trap scenarios
- ✅ Hungarian assignment with explicit NMS
- ✅ Enhanced CLI validation and JSON schema
- ✅ Real-world testing with 16+ camera trap videos

**Next Steps:**
- Step 9: Visualization script (scripts/21_viz_tracks.py)
- Step 10: Testing and validation (exp_002_tracking/)

---

### 🎯 [GUIDE_CLASSIFICATION.md](./GUIDE_CLASSIFICATION.md) - Classification Pipeline  
**Branch:** `feat/classification-pipeline`  
**Status:** 🚧 In Progress (Planning Phase)  
**Goal:** End-to-end pipeline from tracking_json → crops → classifier → counts

**Phase 1: Validation & QA**
- ⏳ Species mapping configuration 
- ⏳ Autolabel from filenames
- ⏳ Manual validation loop (≥95% accuracy gate)
- ⏳ Quantitative crop QA

**Phase 2: Training & Production**
- ⏳ Classifier training scripts
- ⏳ Counts by species aggregation
- ⏳ Validation & tests
- ⏳ Documentation & portfolio polish

---

## Guide Conventions

All guides follow consistent formatting:
- **Branch name** and implementation status
- **Step-by-step checklist** with ✅/⏳/🚧 status indicators
- **Configuration snippets** and directory structures
- **Validation checklist** and commit log
- **Technical notes** and open questions

## Cross-References

- Main project documentation: [../README.md](../README.md)
- Configuration files: [../config/](../config/)
- Implementation scripts: [../scripts/](../scripts/)
- Experimental results: [../experiments/](../experiments/)