# MegaDetector Parameter Sweep – Report V2

## What Changed From V1
- Fixed the stats: now we aggregate per video → no bias from long clips
- Added composite scoring (detections + speed + coverage)
- Visualizations highlight the trade-offs directly (see Figs 1–3)

## Key Findings
- **Frame stride dominates**: Stride=1 keeps all detections but is ~4x slower. Stride=2 halves runtime with small losses. Stride=5 is fast but drops too much (Fig 3).
- **Confidence threshold barely matters**: 0.15 vs 0.25 gave the same total detections in this dataset (needs testing with more species).
- **Min area ratio filters noise**: going from 0.0025 → 0.01 only cut detections by ~2% (Fig 2), so it’s safe to push this up a bit.

## Visual Insights
- **Fig 1:** Efficiency frontier → trade-off between detections and runtime.  
- **Fig 2:** Area ratio filtering has minimal impact.  
- **Fig 3:** Boxplots clearly show stride is the main lever.

## Practical Configurations
-  **Balanced (recommended):** conf=0.20, stride=2, area=0.005  
-  **Sensitive (rare species):** conf=0.15, stride=1, area=0.0025  
-  **High throughput:** conf=0.25, stride=5, area=0.01  

## Why This Matters
Instead of hardcoding arbitrary numbers, we now have evidence-backed defaults and flexibility for different use cases. This also sets up a framework for future experiments as more videos come in.
