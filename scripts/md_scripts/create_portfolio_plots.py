#!/usr/bin/env python3
"""
Portfolio-Quality Visualization Generator
Creates individual high-quality plots for MegaDetector parameter calibration analysis
Based on GPT-5 recommendations for clear, interpretable visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_results(csv_path="experiments/exp_001_md_calibration/results_ranked.csv"):
    """Load calibration results"""
    return pd.read_csv(csv_path)

def create_pareto_frontier(df, outdir):
    """
    1. Pareto Frontier - The hero chart
    Shows fundamental trade-off: processing time vs total detections
    """
    plt.figure(figsize=(12, 8))
    
    # Pareto optimal points
    pareto_points = df[df['is_pareto'] == True]
    non_pareto = df[df['is_pareto'] == False]
    
    # Scatter plot
    plt.scatter(non_pareto['sum_seconds'], non_pareto['sum_total_detections'], 
               alpha=0.6, color='lightgray', s=60, label='Sub-optimal combinations')
    plt.scatter(pareto_points['sum_seconds'], pareto_points['sum_total_detections'], 
               alpha=0.9, color='#2E86C1', s=100, label='Pareto optimal combinations')
    
    # Connect Pareto points
    pareto_sorted = pareto_points.sort_values('sum_seconds')
    plt.plot(pareto_sorted['sum_seconds'], pareto_sorted['sum_total_detections'], 
             '--', color='#2E86C1', alpha=0.7, linewidth=2)
    
    # Annotations for key points
    best_overall = df.loc[df['score'].idxmax()]
    fastest = df.loc[df['sum_seconds'].idxmin()]
    most_detections = df.loc[df['sum_total_detections'].idxmax()]
    
    plt.annotate('Best Overall\n(Highest Score)', 
                xy=(best_overall['sum_seconds'], best_overall['sum_total_detections']),
                xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate('Fastest Processing', 
                xy=(fastest['sum_seconds'], fastest['sum_total_detections']),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Total Detections')
    plt.title('MegaDetector Parameter Optimization: Efficiency Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{outdir}/01_pareto_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 01_pareto_frontier.png")

def create_stride_speed_chart(df, outdir):
    """
    2. Frame Stride vs Processing Time
    Shows clear acceleration benefits
    """
    plt.figure(figsize=(10, 7))
    
    stride_speed = df.groupby('frame_stride')['sum_seconds'].mean().reset_index()
    
    colors = ['#E74C3C', '#F39C12', '#27AE60']
    bars = plt.bar(stride_speed['frame_stride'], stride_speed['sum_seconds'], 
                   color=colors, alpha=0.8, width=0.6)
    
    # Add speedup annotations
    base_time = stride_speed.loc[stride_speed['frame_stride']==1, 'sum_seconds'].iloc[0]
    for i, (stride, time) in enumerate(zip(stride_speed['frame_stride'], stride_speed['sum_seconds'])):
        speedup = base_time / time
        plt.text(stride, time + 5, f'{speedup:.1f}x faster', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        plt.text(stride, time/2, f'{time:.0f}s', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    plt.xlabel('Frame Stride')
    plt.ylabel('Average Processing Time (seconds)')
    plt.title('Processing Speed vs Frame Stride\nHigher stride = Skip more frames = Faster processing')
    plt.xticks([1, 2, 5])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{outdir}/02_stride_speed.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 02_stride_speed.png")

def create_stride_detections_chart(df, outdir):
    """
    3. Frame Stride vs Total Detections
    Shows cost of acceleration (lost detections)
    """
    plt.figure(figsize=(10, 7))
    
    stride_detections = df.groupby('frame_stride')['sum_total_detections'].mean().reset_index()
    
    colors = ['#27AE60', '#F39C12', '#E74C3C']
    bars = plt.bar(stride_detections['frame_stride'], stride_detections['sum_total_detections'], 
                   color=colors, alpha=0.8, width=0.6)
    
    # Add detection loss annotations
    max_detections = stride_detections['sum_total_detections'].max()
    for i, (stride, detections) in enumerate(zip(stride_detections['frame_stride'], stride_detections['sum_total_detections'])):
        loss_pct = (max_detections - detections) / max_detections * 100
        if loss_pct > 0:
            plt.text(stride, detections + 50, f'-{loss_pct:.0f}%', 
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=11)
        plt.text(stride, detections/2, f'{detections:.0f}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    plt.xlabel('Frame Stride')
    plt.ylabel('Average Total Detections')
    plt.title('Detection Count vs Frame Stride\nTrade-off: Speed vs Sensitivity')
    plt.xticks([1, 2, 5])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{outdir}/03_stride_detections.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 03_stride_detections.png")

def create_confidence_detections_chart(df, outdir):
    """
    4. Confidence Threshold vs Total Detections
    Shows filtering effect of confidence
    """
    plt.figure(figsize=(10, 7))
    
    conf_detections = df.groupby('conf_threshold')['sum_total_detections'].mean().reset_index()
    
    colors = ['#3498DB', '#9B59B6', '#E67E22']
    bars = plt.bar(conf_detections['conf_threshold'], conf_detections['sum_total_detections'], 
                   color=colors, alpha=0.8, width=0.03)
    
    # Add filtering annotations
    max_detections = conf_detections['sum_total_detections'].max()
    for i, (conf, detections) in enumerate(zip(conf_detections['conf_threshold'], conf_detections['sum_total_detections'])):
        filter_pct = (max_detections - detections) / max_detections * 100
        if filter_pct > 0:
            plt.text(conf, detections + 50, f'-{filter_pct:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=11)
        plt.text(conf, detections/2, f'{detections:.0f}', 
                ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Average Total Detections')
    plt.title('Detection Filtering vs Confidence Threshold\nHigher confidence = More selective = Fewer detections')
    plt.xticks([0.15, 0.20, 0.25])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f"{outdir}/04_confidence_detections.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 04_confidence_detections.png")

def create_interaction_heatmap(df, outdir):
    """
    5. Confidence × Stride Interaction Heatmap
    Shows parameter interaction effects on detection rate
    """
    plt.figure(figsize=(10, 8))
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(
        index='conf_threshold', 
        columns='frame_stride', 
        values='mean_detected_frame_rate',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'Detection Rate'}, 
                square=True, linewidths=0.5)
    
    plt.title('Parameter Interaction: Detection Rate Sensitivity\nConf × Stride Effects')
    plt.xlabel('Frame Stride')
    plt.ylabel('Confidence Threshold')
    
    plt.savefig(f"{outdir}/05_interaction_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 05_interaction_heatmap.png")

def create_area_ratio_line_chart(df, outdir):
    """
    6. Min Area Ratio vs Detections
    Shows how area filtering affects small animal detection
    """
    plt.figure(figsize=(10, 7))
    
    # Group by area ratio
    area_detections = df.groupby('min_area_ratio').agg({
        'sum_total_detections': ['mean', 'std']
    }).reset_index()
    area_detections.columns = ['min_area_ratio', 'mean_detections', 'std_detections']
    
    # Line plot with error bars
    plt.errorbar(area_detections['min_area_ratio'], area_detections['mean_detections'], 
                yerr=area_detections['std_detections'], 
                marker='o', linewidth=3, markersize=8, capsize=5, capthick=2,
                color='#E74C3C', alpha=0.8)
    
    # Add annotations for filtering effect
    for i, (area, detections) in enumerate(zip(area_detections['min_area_ratio'], area_detections['mean_detections'])):
        if i > 0:
            prev_detections = area_detections['mean_detections'].iloc[i-1]
            loss = prev_detections - detections
            loss_pct = loss / prev_detections * 100
            plt.annotate(f'-{loss:.0f}\n(-{loss_pct:.1f}%)', 
                        xy=(area, detections), xytext=(10, 20), 
                        textcoords='offset points', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('Minimum Area Ratio (fraction of frame)')
    plt.ylabel('Average Total Detections')
    plt.title('Small Animal Filtering vs Minimum Area Ratio\nHigher ratio = Filters out smaller detections')
    plt.grid(True, alpha=0.3)
    plt.xticks([0.0025, 0.005, 0.01])
    
    plt.savefig(f"{outdir}/06_area_ratio_line.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 06_area_ratio_line.png")

def create_paired_comparison_boxplots(df, outdir):
    """
    7. Paired Comparison: Stride Effects on Same Videos
    Shows robustness across different videos
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Prepare data for paired comparison
    stride_data = []
    for stride in [1, 2, 5]:
        stride_subset = df[df['frame_stride'] == stride]
        for _, row in stride_subset.iterrows():
            stride_data.append({
                'stride': f'Stride {stride}',
                'detections': row['sum_total_detections'],
                'time': row['sum_seconds'],
                'combo': f"conf={row['conf_threshold']}, area={row['min_area_ratio']}"
            })
    
    stride_df = pd.DataFrame(stride_data)
    
    # Detections boxplot
    sns.boxplot(data=stride_df, x='stride', y='detections', ax=ax1, palette=['#27AE60', '#F39C12', '#E74C3C'])
    ax1.set_title('Detection Count Distribution by Stride')
    ax1.set_ylabel('Total Detections')
    ax1.grid(True, alpha=0.3)
    
    # Processing time boxplot
    sns.boxplot(data=stride_df, x='stride', y='time', ax=ax2, palette=['#E74C3C', '#F39C12', '#27AE60'])
    ax2.set_title('Processing Time Distribution by Stride')
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Robustness Analysis: Stride Effects Across Parameter Combinations', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{outdir}/07_paired_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 07_paired_comparison.png")

def main():
    """Generate all portfolio-quality plots"""
    print("Creating Portfolio-Quality Visualizations")
    print("=" * 60)
    
    # Load data
    df = load_results()
    print(f"Loaded {len(df)} parameter combinations")
    
    # Create output directory
    outdir = Path("docs/plots/exp_001_md_calibration")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")
    print()
    
    # Generate each plot
    create_pareto_frontier(df, outdir)
    create_stride_speed_chart(df, outdir)
    create_stride_detections_chart(df, outdir)
    create_confidence_detections_chart(df, outdir)
    create_interaction_heatmap(df, outdir)
    create_area_ratio_line_chart(df, outdir)
    create_paired_comparison_boxplots(df, outdir)
    
    print()
    print("All plots generated successfully!")
    print(f"Location: {outdir.absolute()}")
    print()
    print("Portfolio-ready visualizations:")
    print("1. Pareto Frontier (hero chart) - trade-off visualization")
    print("2. Stride Speed - acceleration benefits")  
    print("3. Stride Detections - sensitivity cost")
    print("4. Confidence Filtering - selectivity effects")
    print("5. Interaction Heatmap - parameter combinations")
    print("6. Area Ratio Filtering - small animal detection")
    print("7. Robustness Analysis - consistency across parameters")

if __name__ == "__main__":
    main()