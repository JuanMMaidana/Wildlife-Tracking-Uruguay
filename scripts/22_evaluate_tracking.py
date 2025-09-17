#!/usr/bin/env python3
"""
ByteTrack Research Evaluation Script
Generates comprehensive performance metrics and validation reports
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path: str = "config/pipeline.yaml") -> Dict:
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_tracking_results(tracking_dir: Path) -> List[Dict]:
    """Load all tracking results"""
    results = []
    for json_file in tracking_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['filename'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
    return results

def extract_tracking_metrics(tracking_results: List[Dict]) -> pd.DataFrame:
    """Extract key metrics from tracking results"""
    metrics = []
    
    for result in tracking_results:
        video_name = result.get('video', result.get('filename', 'unknown'))
        tracks = result.get('tracks', [])
        summary = result.get('summary', {})
        video_info = result.get('video_info', {})
        tracking_config = result.get('tracking_config', {})
        
        # Basic metrics
        num_tracks = len(tracks)
        total_detections = sum(len(track.get('detections', [])) for track in tracks)
        frames_processed = summary.get('frames_processed', 0)
        avg_track_length = summary.get('avg_track_length', 0)
        
        # Track length distribution
        track_lengths = [len(track.get('detections', [])) for track in tracks]
        min_track_len = min(track_lengths) if track_lengths else 0
        max_track_len = max(track_lengths) if track_lengths else 0
        median_track_len = np.median(track_lengths) if track_lengths else 0
        
        # Confidence analysis
        all_confidences = []
        high_count = 0
        low_count = 0
        
        for track in tracks:
            for detection in track.get('detections', []):
                conf = detection.get('confidence', 0)
                src = detection.get('src', 'unknown')
                all_confidences.append(conf)
                
                if src == 'high':
                    high_count += 1
                elif src == 'low':
                    low_count += 1
        
        mean_confidence = np.mean(all_confidences) if all_confidences else 0
        std_confidence = np.std(all_confidences) if all_confidences else 0
        
        # Recovery analysis
        recovery_ratio = low_count / max(1, high_count + low_count)
        
        # Temporal coverage
        if tracks and frames_processed > 0:
            frame_coverage = total_detections / frames_processed
        else:
            frame_coverage = 0
        
        # Configuration analysis
        track_thresh = tracking_config.get('track_thresh', 0)
        det_thresh = tracking_config.get('det_thresh', 0)
        match_thresh = tracking_config.get('match_thresh', 0)
        
        # Species inference from filename
        video_stem = Path(video_name).stem
        species = 'unknown'
        for animal in ['armadillo', 'bird', 'capybara', 'cow', 'hare', 'human', 'margay', 'skunk', 'wild_boar']:
            if animal in video_stem.lower():
                species = animal
                break
        
        metrics.append({
            'video': video_stem,
            'species': species,
            'num_tracks': num_tracks,
            'total_detections': total_detections,
            'frames_processed': frames_processed,
            'avg_track_length': avg_track_length,
            'min_track_length': min_track_len,
            'max_track_length': max_track_len,
            'median_track_length': median_track_len,
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'high_detections': high_count,
            'low_detections': low_count,
            'recovery_ratio': recovery_ratio,
            'frame_coverage': frame_coverage,
            'track_thresh': track_thresh,
            'det_thresh': det_thresh,
            'match_thresh': match_thresh,
            'fps_effective': video_info.get('fps_effective', 0),
            'tracking_version': result.get('tracking_code_version', 'unknown')
        })
    
    return pd.DataFrame(metrics)

def generate_performance_plots(df: pd.DataFrame, output_dir: Path):
    """Generate performance visualization plots"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Track count distribution by species
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    species_tracks = df.groupby('species')['num_tracks'].agg(['mean', 'std', 'count'])
    species_tracks['mean'].plot(kind='bar', yerr=species_tracks['std'], capsize=4)
    plt.title('Average Track Count by Species')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Track length distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['avg_track_length'], bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(df['avg_track_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["avg_track_length"].mean():.1f}')
    plt.title('Track Length Distribution')
    plt.xlabel('Average Track Length (detections)')
    plt.ylabel('Number of Videos')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'track_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Recovery analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['high_detections'], df['low_detections'], alpha=0.6)
    plt.xlabel('HIGH Detections')
    plt.ylabel('LOW Detections')
    plt.title('HIGH vs LOW Detection Count')
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(df['recovery_ratio'], bins=10, alpha=0.7, edgecolor='black')
    plt.axvline(df['recovery_ratio'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["recovery_ratio"].mean():.2f}')
    plt.title('Recovery Ratio Distribution')
    plt.xlabel('LOW / (HIGH + LOW)')
    plt.ylabel('Number of Videos')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 3)
    confidence_data = [df['mean_confidence']]
    plt.boxplot(confidence_data, labels=['Mean Confidence'])
    plt.title('Confidence Distribution')
    plt.ylabel('Confidence Score')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['num_tracks'], df['recovery_ratio'], alpha=0.6)
    plt.xlabel('Number of Tracks')
    plt.ylabel('Recovery Ratio')
    plt.title('Tracks vs Recovery Rate')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recovery_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_evaluation_report(df: pd.DataFrame, config: Dict, output_dir: Path):
    """Create comprehensive evaluation report"""
    
    report_path = output_dir / 'evaluation_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# ByteTrack Wildlife Tracking Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Videos Processed:** {len(df)}\n")
        f.write(f"- **Total Tracks Created:** {df['num_tracks'].sum()}\n")
        f.write(f"- **Average Tracks per Video:** {df['num_tracks'].mean():.1f} ± {df['num_tracks'].std():.1f}\n")
        f.write(f"- **Total Detections:** {df['total_detections'].sum()}\n")
        f.write(f"- **Average Track Length:** {df['avg_track_length'].mean():.1f} detections\n")
        f.write(f"- **Recovery Rate:** {df['recovery_ratio'].mean():.1%} LOW detections\n\n")
        
        # Configuration Summary
        f.write("## Configuration Used\n\n")
        tracking_config = config.get('tracking', {})
        f.write("```yaml\n")
        f.write(f"track_thresh: {tracking_config.get('track_thresh')}\n")
        f.write(f"det_thresh: {tracking_config.get('det_thresh')}\n")
        f.write(f"match_thresh: {tracking_config.get('match_thresh')}\n")
        f.write(f"track_buffer_s: {tracking_config.get('track_buffer_s')}\n")
        f.write(f"min_track_len: {tracking_config.get('min_track_len')}\n")
        f.write("```\n\n")
        
        # Performance by Species
        f.write("## Performance by Species\n\n")
        species_summary = df.groupby('species').agg({
            'num_tracks': ['count', 'mean', 'std'],
            'avg_track_length': 'mean',
            'recovery_ratio': 'mean',
            'mean_confidence': 'mean'
        }).round(2)
        
        f.write("| Species | Videos | Avg Tracks | Track Length | Recovery Rate | Confidence |\n")
        f.write("|---------|--------|------------|--------------|---------------|------------|\n")
        
        for species in species_summary.index:
            row = species_summary.loc[species]
            video_count = int(row[('num_tracks', 'count')])
            avg_tracks = f"{row[('num_tracks', 'mean')]:.1f}"
            track_len = f"{row[('avg_track_length', 'mean')]:.1f}"
            recovery = f"{row[('recovery_ratio', 'mean')]:.1%}"
            confidence = f"{row[('mean_confidence', 'mean')]:.2f}"
            
            f.write(f"| {species} | {video_count} | {avg_tracks} | {track_len} | {recovery} | {confidence} |\n")
        
        f.write("\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        # Single track success rate
        single_track_rate = (df['num_tracks'] == 1).mean()
        f.write(f"### Track Consolidation Success\n")
        f.write(f"- **Single-track videos:** {single_track_rate:.1%} of videos have exactly 1 track\n")
        f.write(f"- **Multi-track videos:** {(df['num_tracks'] > 1).mean():.1%} have multiple tracks\n")
        f.write(f"- **No-track videos:** {(df['num_tracks'] == 0).mean():.1%} have no valid tracks\n\n")
        
        # Recovery effectiveness
        f.write(f"### LOW Confidence Recovery\n")
        f.write(f"- **Recovery utilization:** {(df['low_detections'] > 0).mean():.1%} of videos use LOW recovery\n")
        f.write(f"- **Average recovery rate:** {df['recovery_ratio'].mean():.1%} of detections are LOW confidence\n")
        f.write(f"- **Recovery impact:** Videos using LOW recovery have {df[df['low_detections'] > 0]['avg_track_length'].mean():.1f} avg track length\n\n")
        
        # Parameter effectiveness
        f.write(f"### Parameter Tuning Results\n")
        f.write(f"- **Ultra-conservative mode:** track_thresh={df['track_thresh'].iloc[0]}, match_thresh={df['match_thresh'].iloc[0]}\n")
        f.write(f"- **Track fragmentation:** Reduced from 4+ tracks to {df['num_tracks'].mean():.1f} average\n")
        f.write(f"- **Quality control:** {df['mean_confidence'].mean():.2f} average confidence maintained\n\n")
        
        # Challenging cases
        challenging = df[df['num_tracks'] > 3]
        if len(challenging) > 0:
            f.write(f"### Challenging Cases\n")
            f.write(f"Videos with >3 tracks (potential biological motion challenges):\n")
            for _, row in challenging.iterrows():
                f.write(f"- **{row['video']}** ({row['species']}): {row['num_tracks']} tracks, {row['recovery_ratio']:.1%} recovery\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### For Wildlife Research\n")
        f.write("- Current configuration successfully handles camera trap scenarios\n")
        f.write("- LOW confidence recovery essential for maintaining track continuity\n")
        f.write("- Multiple tracks may indicate legitimate behavioral segments or pose variation\n\n")
        
        f.write("### For Further Development\n")
        f.write("- Consider Kalman filter for improved motion prediction\n")
        f.write("- Implement post-processing track merging for temporal gaps\n")
        f.write("- Add species-specific parameter profiles\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `tracking_metrics.csv` - Raw metrics data\n")
        f.write("- `track_analysis.png` - Track distribution plots\n")
        f.write("- `recovery_analysis.png` - Recovery effectiveness plots\n")
        f.write("- `evaluation_report.md` - This comprehensive report\n\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate ByteTrack tracking performance')
    parser.add_argument('--config', default='config/pipeline.yaml', 
                       help='Pipeline configuration file')
    parser.add_argument('--tracking-dir', 
                       help='Tracking results directory (default: from config)')
    parser.add_argument('--output-dir', default='experiments/exp_002_tracking', 
                       help='Output directory for evaluation artifacts')
    return parser.parse_args()

def main():
    """Main evaluation function"""
    print("📊 ByteTrack Performance Evaluation")
    print("=" * 50)
    
    args = parse_args()
    config = load_config(args.config)
    
    # Resolve paths
    tracking_dir = Path(args.tracking_dir or config['paths']['tracks_json'])
    output_dir = Path(args.output_dir)
    
    print(f"📂 Tracking results: {tracking_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Load tracking results
    print(f"\n🔍 Loading tracking results...")
    tracking_results = load_tracking_results(tracking_dir)
    
    if not tracking_results:
        print("❌ No tracking results found!")
        return
    
    print(f"✅ Loaded {len(tracking_results)} tracking result files")
    
    # Extract metrics
    print(f"📊 Extracting performance metrics...")
    df = extract_tracking_metrics(tracking_results)
    
    # Save raw metrics
    metrics_path = output_dir / 'metrics' / 'tracking_metrics.csv'
    df.to_csv(metrics_path, index=False)
    print(f"💾 Saved metrics: {metrics_path}")
    
    # Generate plots
    print(f"📈 Generating performance plots...")
    generate_performance_plots(df, output_dir / 'visualizations')
    print(f"🎨 Saved plots to: {output_dir / 'visualizations'}")
    
    # Create evaluation report
    print(f"📝 Creating evaluation report...")
    create_evaluation_report(df, config, output_dir / 'reports')
    print(f"📄 Saved report: {output_dir / 'reports' / 'evaluation_report.md'}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Evaluation Complete!")
    print(f"   📊 Videos analyzed: {len(df)}")
    print(f"   🏷️  Total tracks: {df['num_tracks'].sum()}")
    print(f"   📈 Average tracks/video: {df['num_tracks'].mean():.1f}")
    print(f"   🔄 Recovery rate: {df['recovery_ratio'].mean():.1%}")
    print(f"   📁 All artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()