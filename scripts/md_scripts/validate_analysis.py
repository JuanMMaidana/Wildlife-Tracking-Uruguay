#!/usr/bin/env python3
"""
Sanity Check Script for MegaDetector Parameter Sweep Analysis
Validates that confidence thresholds and frame stride are properly applied
Based on GPT-5 feedback on suspicious constant detection rates
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_results(csv_path="experiments/exp_001_md_calibration/results.csv"):
    """Load results and add derived columns"""
    df = pd.read_csv(csv_path)
    df['detected_frame_rate'] = df['frames_with_detections'] / df['processed_frames']
    return df

def sanity_check_1_single_video_conf(df):
    """
    Check A: Single video, fixed min_area_ratio, compare conf=0.15 vs 0.25
    Expected: Lower frames_with_detections and total_detections at higher confidence
    """
    print("🔍 SANITY CHECK 1: Confidence threshold effect on single video")
    print("=" * 60)
    
    # Pick a video with good detection coverage
    video_detection_counts = df.groupby('video')['total_detections'].sum().sort_values(ascending=False)
    test_video = video_detection_counts.index[0]  # Video with most detections
    
    print(f"Test video: {test_video}")
    
    # Filter to single video, fixed area ratio
    g = df[(df['video'] == test_video) & (df['min_area_ratio'] == 0.005)]
    
    if len(g) == 0:
        print("❌ No data for test video with area ratio 0.005")
        return False
    
    # Pivot table to compare confidence thresholds
    pivot = g.pivot_table(
        index='frame_stride', 
        columns='conf_threshold', 
        values=['frames_with_detections', 'total_detections'], 
        aggfunc='sum'
    )
    
    print("\nFrames with detections by confidence:")
    print(pivot['frames_with_detections'])
    print("\nTotal detections by confidence:")
    print(pivot['total_detections'])
    
    # Check if values decrease with higher confidence
    frames_data = pivot['frames_with_detections']
    detections_data = pivot['total_detections']
    
    issues = []
    for stride in frames_data.index:
        conf_values = [0.15, 0.20, 0.25]
        available_confs = [c for c in conf_values if c in frames_data.columns]
        
        if len(available_confs) < 2:
            continue
            
        frames_series = [frames_data.loc[stride, conf] for conf in available_confs]
        detections_series = [detections_data.loc[stride, conf] for conf in available_confs]
        
        # Check if generally decreasing
        frames_decreasing = all(frames_series[i] >= frames_series[i+1] for i in range(len(frames_series)-1))
        detections_decreasing = all(detections_series[i] >= detections_series[i+1] for i in range(len(detections_series)-1))
        
        print(f"\nStride {stride}:")
        print(f"  Frames decreasing with conf: {frames_decreasing} {frames_series}")
        print(f"  Detections decreasing with conf: {detections_decreasing} {detections_series}")
        
        if not frames_decreasing:
            issues.append(f"Frames with detections not decreasing with confidence (stride {stride})")
        if not detections_decreasing:
            issues.append(f"Total detections not decreasing with confidence (stride {stride})")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ Confidence thresholds working correctly")
        return True

def sanity_check_2_single_video_stride(df):
    """
    Check B: Single video, fixed conf and area ratio, compare stride=1 vs 2 vs 5
    Expected: processed_frames decreases, processing_time decreases significantly
    """
    print("\n🔍 SANITY CHECK 2: Frame stride effect on single video")
    print("=" * 60)
    
    # Pick same test video
    video_detection_counts = df.groupby('video')['total_detections'].sum().sort_values(ascending=False)
    test_video = video_detection_counts.index[0]
    
    # Filter to single video, fixed parameters
    g = df[(df['video'] == test_video) & 
           (df['min_area_ratio'] == 0.005) & 
           (df['conf_threshold'] == 0.20)]
    
    if len(g) == 0:
        print("❌ No data for test video with conf=0.20, area=0.005")
        return False
    
    result = g[['frame_stride', 'processed_frames', 'frames_with_detections', 
                'detected_frame_rate', 'total_detections', 'processing_time']].sort_values('frame_stride')
    
    print(f"Test video: {test_video}")
    print("\nStride effects:")
    print(result.to_string(index=False))
    
    # Check expectations
    issues = []
    
    # Processed frames should decrease
    processed_frames = result['processed_frames'].values
    if not all(processed_frames[i] >= processed_frames[i+1] for i in range(len(processed_frames)-1)):
        issues.append("Processed frames not decreasing with stride")
    
    # Processing time should decrease significantly
    processing_times = result['processing_time'].values
    if not all(processing_times[i] >= processing_times[i+1] for i in range(len(processing_times)-1)):
        issues.append("Processing time not decreasing with stride")
    
    # Speedup should be meaningful
    if len(processing_times) >= 3:
        speedup_2 = processing_times[0] / processing_times[1] if processing_times[1] > 0 else 0
        speedup_5 = processing_times[0] / processing_times[2] if processing_times[2] > 0 else 0
        
        print(f"\nSpeedup factors:")
        print(f"  Stride 2 vs 1: {speedup_2:.1f}x")
        print(f"  Stride 5 vs 1: {speedup_5:.1f}x")
        
        if speedup_2 < 1.5:
            issues.append(f"Stride 2 speedup too low: {speedup_2:.1f}x")
        if speedup_5 < 2.5:
            issues.append(f"Stride 5 speedup too low: {speedup_5:.1f}x")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ Frame stride effects working correctly")
        return True

def sanity_check_3_global_detection_rate(df):
    """
    Check C: Recalculate detection rate from sums (not averages of averages)
    Should show variation across conf/stride combinations
    """
    print("\n🔍 SANITY CHECK 3: Global detection rates from sums")
    print("=" * 60)
    
    # Calculate global rates properly (sums, not averages of averages)
    check = (df.groupby(['conf_threshold', 'frame_stride', 'min_area_ratio'])
               .agg(F=('frames_with_detections', 'sum'), 
                    P=('processed_frames', 'sum'))
               .reset_index())
    check['global_rate'] = check['F'] / check['P']
    
    print("Global detection rates (from sums):")
    display = check.pivot_table(
        index=['conf_threshold'], 
        columns=['frame_stride', 'min_area_ratio'],
        values='global_rate'
    )
    print(display.round(4))
    
    # Check for variation
    conf_variation = check.groupby('conf_threshold')['global_rate'].agg(['min', 'max', 'std'])
    stride_variation = check.groupby('frame_stride')['global_rate'].agg(['min', 'max', 'std'])
    
    print(f"\nVariation by confidence threshold:")
    print(conf_variation.round(4))
    print(f"\nVariation by frame stride:")
    print(stride_variation.round(4))
    
    issues = []
    
    # Check if confidence has any effect
    if conf_variation['std'].max() < 0.001:
        issues.append("Confidence threshold has no effect on detection rate (std < 0.001)")
    
    # Check if stride has any effect
    if stride_variation['std'].max() < 0.001:
        issues.append("Frame stride has no effect on detection rate (std < 0.001)")
    
    # Check if rates are suspiciously constant
    all_rates = check['global_rate'].values
    if np.std(all_rates) < 0.001:
        issues.append(f"All detection rates suspiciously constant (std={np.std(all_rates):.6f})")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ Detection rates show expected variation")
        return True

def sanity_check_4_raw_data_inspection(df):
    """
    Check D: Inspect raw data patterns for obvious issues
    """
    print("\n🔍 SANITY CHECK 4: Raw data inspection")
    print("=" * 60)
    
    print("Summary statistics by parameter:")
    
    # Group by parameters and show key metrics
    summary = df.groupby(['conf_threshold', 'frame_stride', 'min_area_ratio']).agg({
        'frames_with_detections': ['count', 'mean', 'std'],
        'total_detections': ['mean', 'std'],
        'processing_time': ['mean', 'std'],
        'detected_frame_rate': ['mean', 'std']
    }).round(3)
    
    print("\nFrames with detections stats:")
    print(summary['frames_with_detections'])
    
    print("\nDetection rate stats:")
    print(summary['detected_frame_rate'])
    
    # Look for suspicious patterns
    issues = []
    
    # Check if detection rates are identical across conditions
    unique_rates = df['detected_frame_rate'].round(3).unique()
    if len(unique_rates) <= 3:
        issues.append(f"Only {len(unique_rates)} unique detection rates found: {unique_rates}")
    
    # Check for impossible values
    if (df['frames_with_detections'] > df['processed_frames']).any():
        issues.append("Frames with detections > processed frames found")
    
    if (df['detected_frame_rate'] > 1.0).any():
        issues.append("Detection rate > 1.0 found")
    
    if (df['detected_frame_rate'] < 0.0).any():
        issues.append("Negative detection rate found")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ Raw data patterns look reasonable")
        return True

def diagnose_confidence_issue(df):
    """
    Diagnose why confidence thresholds might not be working
    """
    print("\n🔧 DIAGNOSTIC: Confidence threshold application")
    print("=" * 60)
    
    # Look at the raw CSV structure
    print("Columns in CSV:")
    print(df.columns.tolist())
    
    # Check if we have the original detection data
    if 'detections_per_second' in df.columns:
        print(f"\nDetections per second range: {df['detections_per_second'].min():.3f} - {df['detections_per_second'].max():.3f}")
    
    # Look for patterns that suggest confidence wasn't applied properly
    print(f"\nUnique confidence values: {sorted(df['conf_threshold'].unique())}")
    print(f"Unique frame strides: {sorted(df['frame_stride'].unique())}")
    print(f"Unique area ratios: {sorted(df['min_area_ratio'].unique())}")
    
    # Check if total_detections vs frames_with_detections relationship makes sense
    df['detections_per_frame_with_detection'] = df['total_detections'] / df['frames_with_detections'].replace(0, np.nan)
    
    print(f"\nDetections per frame with detection:")
    print(f"  Mean: {df['detections_per_frame_with_detection'].mean():.2f}")
    print(f"  Range: {df['detections_per_frame_with_detection'].min():.2f} - {df['detections_per_frame_with_detection'].max():.2f}")
    
    # This ratio should be fairly consistent if confidence is working
    ratio_by_conf = df.groupby('conf_threshold')['detections_per_frame_with_detection'].mean()
    print(f"\nDetections per frame by confidence:")
    print(ratio_by_conf.round(2))
    
    if ratio_by_conf.std() < 0.01:
        print("⚠️  Ratio is very constant - confidence might not be affecting frame-level counts")
    else:
        print("✅ Ratio varies with confidence - looks normal")

def main():
    """Run all sanity checks"""
    print("🧪 MegaDetector Analysis Validation - Sanity Checks")
    print("Based on GPT-5 feedback identifying suspicious constant detection rates")
    print("=" * 80)
    
    # Load data
    df = load_results()
    print(f"Loaded {len(df)} experiments from results CSV")
    
    # Run all checks
    checks_passed = []
    
    checks_passed.append(sanity_check_1_single_video_conf(df))
    checks_passed.append(sanity_check_2_single_video_stride(df))
    checks_passed.append(sanity_check_3_global_detection_rate(df))
    checks_passed.append(sanity_check_4_raw_data_inspection(df))
    
    # Diagnostic if issues found
    if not all(checks_passed):
        diagnose_confidence_issue(df)
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 80)
    
    passed_count = sum(checks_passed)
    total_checks = len(checks_passed)
    
    if passed_count == total_checks:
        print("✅ ALL SANITY CHECKS PASSED")
        print("Your analysis results are trustworthy!")
        print("\nRecommended actions:")
        print("- Proceed with parameter recommendations")
        print("- Update config/pipeline.yaml with chosen defaults")
        print("- Close issue #1 with confidence")
    else:
        print(f"❌ {total_checks - passed_count}/{total_checks} SANITY CHECKS FAILED")
        print("Your analysis has methodological issues that need fixing!")
        print("\nRecommended actions:")
        print("- Review confidence threshold application in sweep script")
        print("- Ensure frames_with_detections is recalculated after filtering")
        print("- Re-run parameter sweep with corrected methodology")
        print("- Do not use current results for production decisions")
    
    print(f"\nCheck results: {['✅' if passed else '❌' for passed in checks_passed]}")

if __name__ == "__main__":
    main()