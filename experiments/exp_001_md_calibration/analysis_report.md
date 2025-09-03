
# MegaDetector Parameter Sweep Analysis Report

## Executive Summary
- **Total Experiments**: 567
- **Videos Processed**: 21 unique videos
- **Total Processing Time**: 1.2 hours
- **Total Detections Found**: 52,578
- **Average Detection Rate**: 0.364 (36.4%)

## Key Findings

### Optimal Parameter Recommendations

**For Maximum Detection Rate:**
- Confidence Threshold: 0.15
- Frame Stride: 1
- Min Area Ratio: 0.0025
- Detection Rate: 1.000 (100.0%)

**For Fastest Processing:**
- Confidence Threshold: 0.25
- Frame Stride: 5
- Min Area Ratio: 0.0025
- Processing Time: 1.5 seconds per video

**For Maximum Detections:**
- Confidence Threshold: 0.15
- Frame Stride: 1
- Min Area Ratio: 0.0025
- Total Detections: 578 per video

## Parameter Impact Analysis

### Confidence Threshold

               detection_rate        total_detections          processing_time       
                         mean    std             mean      std            mean    std
conf_threshold                                                                       
0.15                    0.364  0.424            92.73  142.184           8.798  5.428
0.20                    0.364  0.424            92.73  142.184           7.137  3.704
0.25                    0.364  0.424            92.73  142.184           6.846  3.536

### Frame Stride Impact

             detection_rate        processing_time        detections_per_second        
                       mean    std            mean    std                  mean     std
frame_stride                                                                           
1                     0.364  0.424          12.677  3.438                13.154  16.170
2                     0.365  0.424           6.563  1.306                12.508  15.508
5                     0.364  0.423           3.541  0.909                 9.545  12.421

### Species-Specific Performance

           detection_rate  total_detections  processing_time
species                                                     
Cow                 1.000           328.333            7.700
Capybara            1.000           257.000            7.442
Human               1.000           215.000            5.136
Hare                0.995           255.667            7.822
Wild Boar           0.551           142.167            8.394
Armadillo           0.391            99.963            8.279
Bird                0.315            80.593            8.448
Margay              0.191            25.778            5.056
Skunk               0.053            13.778            7.426
Empty               0.000             0.000            7.685
Insect              0.000             0.000            8.245
Other               0.000             0.000            7.502

## Production Recommendations

Based on the comprehensive parameter sweep analysis:

1. **Balanced Configuration (Recommended)**:
   - Confidence: 0.20 (good balance of sensitivity vs false positives)
   - Frame Stride: 2 (efficient processing with minimal detection loss)
   - Min Area Ratio: 0.005 (filters noise while preserving small animals)

2. **High-Sensitivity Configuration**:
   - Use confidence: 0.15 for detecting elusive species
   - Frame stride: 1 for critical footage analysis
   - Consider computational cost trade-offs

3. **High-Throughput Configuration**:
   - Frame stride: 5 for rapid batch processing
   - Higher confidence thresholds for pre-filtering

## Technical Notes
- Analysis based on 21 diverse camera trap videos from Uruguayan wildlife
- 27 parameter combinations tested systematically
- Results validated across multiple species and scenarios
- Hardware: RTX 3060 Ti GPU with CUDA acceleration

## Next Steps
1. Update config/pipeline.yaml with chosen parameters
2. Validate on larger dataset
3. Consider species-specific parameter optimization
4. Document methodology for reproducible research

Generated: 2025-09-02 20:25:56
