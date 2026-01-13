#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" updated on 2025-11-14
analyze_multi_seed_1.py - Multi-seed Experiment Analysis Tool part 1
RPS tournament results summary
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Professional color palette
COLORS = {
    'positive': '#2E7D32',  # Dark green
    'negative': '#C62828',  # Dark red
    'neutral': '#757575',   # Gray
    'accent': '#1565C0',    # Dark blue
    'light': '#E0E0E0',     # Light gray
}


def set_professional_style():
    """Set publication-quality plot style"""
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.2',
        'axes.linewidth': 1.2
    })
    sns.set_palette("husl")


def load_experiment_data(result_dir: str = "RPS_train_summary"):
    """Load experiment data from directory"""
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"Error: Directory {result_dir} does not exist")
        return None, None, None, None, None
    
    # Load metadata
    meta_path = result_path / "experiment_metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"Experiment info: {metadata['n_seeds']} seeds")
        print(f"Seeds: {metadata['seeds']}")
        # 从 train_kwargs 中取 rounds（默认为 500）
        train_kwargs = metadata.get("train_kwargs", {})
        n_rounds = int(train_kwargs.get("rounds", 500))
    else:
        metadata = None
        n_rounds = 500  # 默认值
        print("Warning: Metadata not found, using default rounds=500")
    
    # Load statistics
    stats_path = result_path / "RPS_train_statistics.csv"
    stats = pd.read_csv(stats_path, index_col=0) if stats_path.exists() else None
    
    # Load all seeds data
    all_seeds_path = result_path / "RPS_train_all_seeds.csv"
    all_seeds = pd.read_csv(all_seeds_path) if all_seeds_path.exists() else None
    
    # Try to load round-wise data if available
    round_data = None
    for seed in (metadata['seeds'] if metadata else []):
        record_path = result_path / f"RPS_record_seed{seed}.csv"
        if record_path.exists():
            df = pd.read_csv(record_path)
            if round_data is None:
                round_data = df
            else:
                round_data = pd.concat([round_data, df])
            break  # Just load one for structure
    
    return metadata, stats, all_seeds, round_data, n_rounds


def extract_agent_type(agent_name) -> str:
    """
    Extract agent type from agent name 
    Examples:
      - '35_LSTM_v1' -> 'LSTM_v1'
      - '6_A3C_v2' -> 'A3C_v2'
      - '23_MSA' -> 'MSA'
      - '16_XGB' -> 'XGB'
    """
    # 确保是字符串类型
    agent_name = str(agent_name)
    parts = agent_name.split('_')
    if len(parts) > 1:
        # Join all parts after the first underscore
        # This handles cases like LSTM_v1, A3C_v2, Tr_v1, etc.
        return '_'.join(parts[1:])
    else:
        return agent_name


def plot_score_distribution(all_seeds: pd.DataFrame, stats: pd.DataFrame, 
                          output_dir: str) -> None:
    """Create professional boxplot of score distributions"""
    set_professional_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort agents by mean score
    agents_sorted = stats.sort_values('mean', ascending=False).index.tolist()
    all_seeds_sorted = all_seeds.set_index('agent').loc[agents_sorted].reset_index()
    
    # Extract agent types for coloring
    agent_types = all_seeds_sorted['agent'].apply(extract_agent_type)
    type_colors = {t: sns.color_palette("husl", len(agent_types.unique()))[i] 
                   for i, t in enumerate(agent_types.unique())}
    colors = [type_colors[t] for t in agent_types.unique() for _ in range(len(all_seeds_sorted['agent'].unique())//len(agent_types.unique()) + 1)][:len(agents_sorted)]
    
    # Create boxplot
    bp = ax.boxplot([all_seeds_sorted[all_seeds_sorted['agent'] == agent]['score'].values 
                     for agent in agents_sorted],
                    labels=agents_sorted,
                    patch_artist=True,
                    notch=False,
                    showfliers=True,
                    boxprops=dict(linewidth=1.2, alpha=0.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
    
    # Color boxes by performance
    for i, (patch, agent) in enumerate(zip(bp['boxes'], agents_sorted)):
        mean_score = stats.loc[agent, 'mean']
        if mean_score > 50:
            patch.set_facecolor('#4CAF50')  # Green for top performers
        elif mean_score > 0:
            patch.set_facecolor('#FFC107')  # Amber for positive
        elif mean_score > -30:
            patch.set_facecolor('#FF9800')  # Orange for slightly negative
        else:
            patch.set_facecolor('#F44336')  # Red for poor performers
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Agent', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Score Distribution across Multiple Seeds', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    fig_path = Path(output_dir) / "figure_1_score_distribution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_confidence_intervals(stats: pd.DataFrame, output_dir: str) -> None:
    """Create confidence interval plot"""
    set_professional_style()
    
    n_agents = len(stats)
    fig, ax = plt.subplots(figsize=(8, max(4, n_agents * 0.35)))
    
    # Sort by mean score
    stats_sorted = stats.sort_values('mean')
    
    # Calculate positions
    y_pos = np.arange(len(stats_sorted))
    
    # Plot confidence intervals
    for i, (agent, row) in enumerate(stats_sorted.iterrows()):
        color = COLORS['positive'] if row['mean'] > 0 else COLORS['negative']
        
        # Error bar
        ax.errorbar(row['mean'], y_pos[i], 
                   xerr=row['ci95'], 
                   fmt='o', 
                   color=color,
                   capsize=5, 
                   capthick=2,
                   markersize=6,
                   alpha=0.8)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats_sorted.index)
    ax.set_xlabel('Mean Score (with 95% CI)', fontweight='bold')
    ax.set_title('Agent Performance with Confidence Intervals', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    fig_path = Path(output_dir) / "figure_2_confidence_intervals.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_correlation_matrix(all_seeds: pd.DataFrame, output_dir: str) -> None:
    """Create correlation matrix between agents across seeds"""
    set_professional_style()
    
    # Pivot data for correlation
    pivot = all_seeds.pivot_table(index='seed', columns='agent', values='score')
    
    # Calculate correlation
    corr = pivot.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Hierarchical clustering for better visualization
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    # Calculate distance matrix and perform clustering
    distances = 1 - corr.abs()
    linkage = hierarchy.linkage(squareform(distances), method='average')
    dendro = hierarchy.dendrogram(linkage, labels=corr.columns, no_plot=True)
    cluster_order = dendro['leaves']
    
    # Reorder correlation matrix
    corr_ordered = corr.iloc[cluster_order, cluster_order]
    
    # Create heatmap
    im = ax.imshow(corr_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_ordered.columns)))
    ax.set_yticks(np.arange(len(corr_ordered.columns)))
    #ax.set_xticklabels(corr_ordered.columns, rotation=90, ha='right')
    #ax.set_yticklabels(corr_ordered.columns)
    ax.set_xticklabels(corr_ordered.columns, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(corr_ordered.columns, fontsize=14)
    
    # Add grid
    #ax.set_xticks(np.arange(len(corr_ordered.columns) + 1) - 0.5, minor=True)
    #ax.set_yticks(np.arange(len(corr_ordered.columns) + 1) - 0.5, minor=True)
    #ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # Remove grid lines (关闭网格线)
    ax.grid(False)

    # Title
    ax.set_title('Agent Performance Correlation Matrix (Clustered)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    fig_path = Path(output_dir) / "figure_3_correlation_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_stability_analysis(all_seeds: pd.DataFrame, stats: pd.DataFrame, 
                           output_dir: str) -> None:
    """Create stability analysis plot (CV vs Mean)"""
    set_professional_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate coefficient of variation
    cv = (stats['std'] / stats['mean'].abs() * 100).replace([np.inf, -np.inf], np.nan)
    
    # Create scatter plot
    scatter = ax.scatter(stats['mean'], cv, 
                        c=stats['mean'], 
                        cmap='RdYlGn', 
                        s=100, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Score', fontweight='bold')
    
    # Add agent labels for outliers
    for agent, row in stats.iterrows():
        cv_value = cv.get(agent, np.nan)
        if pd.notna(cv_value):
            # Label extreme points
            if abs(row['mean']) > stats['mean'].quantile(0.9) or cv_value > cv.quantile(0.9):
                ax.annotate(agent, (row['mean'], cv_value), 
                           fontsize=8, alpha=0.7,
                           xytext=(5, 5), textcoords='offset points')
    
    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.3, label='CV=50%')
    
    # Labels and title
    ax.set_xlabel('Mean Score', fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    ax.set_title('Agent Stability Analysis (Mean vs Variability)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    fig_path = Path(output_dir) / "figure_4_stability_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


def plot_performance_over_rounds(result_dir: str, output_dir: str, all_seeds: pd.DataFrame, 
                                n_rounds: int = 500) -> None:
    """Create professional plot showing score evolution over rounds using REAL data from game records"""
    set_professional_style()
    
    # Extract agent method names (smart parsing to handle both "35_LSTM_v1" and "LSTM_v1")
    def extract_method(agent_name):
        """
        Extract method name from agent name, handling both formats:
        - '35_LSTM_v1' -> 'LSTM_v1' (with seat number prefix)
        - 'LSTM_v1' -> 'LSTM_v1' (without seat number prefix)
        - 'MSA_v2' -> 'MSA_v2' (keep as is)
        """
        agent_name = str(agent_name)
        if '_' not in agent_name:
            return agent_name
        
        # Split only at the first underscore
        parts = agent_name.split('_', 1)
        if len(parts) == 2:
            first_part, rest = parts
            # Check if first part is a number (seat index)
            try:
                int(first_part)
                # It's a seat number, return the rest
                return rest
            except ValueError:
                # Not a number, return full name
                return agent_name
        else:
            return agent_name
    
    result_path = Path(result_dir)
    
    # Scan for available seeds
    seeds = []
    for f in result_path.glob("RPS_record_seed*.csv"):
        match = re.search(r'seed(\d+)', f.name)
        if match:
            seeds.append(int(match.group(1)))
    seeds = sorted(seeds)
    
    if not seeds:
        print("Warning: No RPS_record_seed*.csv files found, cannot plot real performance evolution")
        return
    
    print(f"  Found {len(seeds)} seeds for real performance evolution")
    
    # Load and process real game data
    method_curves = defaultdict(list)
    
    for seed in seeds:
        record_path = result_path / f"RPS_record_seed{seed}.csv"
        if not record_path.exists():
            continue
            
        try:
            df = pd.read_csv(record_path)
            
            # Check required columns
            if not all(col in df.columns for col in ["who_agent", "whom_agent", "score_delta_who"]):
                print(f"  Warning: Missing required columns in {record_path.name}")
                continue
            
            # Sort by round if available
            if "round" in df.columns and "pair_index" in df.columns:
                df = df.sort_values(["round", "pair_index"]).reset_index(drop=True)
            elif "round" in df.columns:
                df = df.sort_values("round").reset_index(drop=True)
            
            # Extract data
            who = df["who_agent"].astype(str).values
            whom = df["whom_agent"].astype(str).values
            score_delta = df["score_delta_who"].astype(int).values
            
            # Calculate contribution for each method
            who_methods = np.array([extract_method(x) for x in who])
            whom_methods = np.array([extract_method(x) for x in whom])
            
            # Get unique methods
            all_methods = set(who_methods) | set(whom_methods)
            
            for method in all_methods:
                # Calculate cumulative score for this method
                # Method gets +score_delta when it's who_agent and wins
                # Method gets -score_delta when it's whom_agent and loses
                contrib = (who_methods == method) * score_delta + (whom_methods == method) * (-score_delta)
                cumulative = np.cumsum(contrib)
                # Prepend 0 for round 0 (initial state)
                cumulative_with_init = np.concatenate([[0], cumulative])
                method_curves[method].append(cumulative_with_init)
        except Exception as e:
            print(f"  Warning: Failed to process {record_path.name}: {e}")
            continue
    
    if not method_curves:
        print("  Warning: No method data found for performance evolution")
        return
    
    # Calculate mean curves for each method across seeds
    method_means = {}
    for method, curves in method_curves.items():
        # Align to minimum length across seeds
        min_len = min(len(c) for c in curves)
        aligned = np.vstack([c[:min_len] for c in curves])
        method_means[method] = np.mean(aligned, axis=0)
    
    # Sort methods by final score
    method_final_scores = {m: curve[-1] for m, curve in method_means.items()}
    sorted_methods = sorted(method_final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Limit number of methods for readability
    #n_methods = len(sorted_methods)
    #if n_methods > 15:
    #    print(f"  Note: Showing top 10 and bottom 5 methods out of {n_methods} total")
    #    methods_to_plot = [m for m, _ in sorted_methods[:10]] + [m for m, _ in sorted_methods[-5:]]
    #else:
    #    methods_to_plot = [m for m, _ in sorted_methods]

    # Use all methods; rely on smart label placement to keep the plot readable
    methods_to_plot = [m for m, _ in sorted_methods]
    n_methods = len(methods_to_plot)
    print(f"  Plotting {n_methods} methods in performance evolution plot")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Generate color palette
    colors = sns.color_palette("husl", len(methods_to_plot))
    
    final_positions = []
    
    for idx, method in enumerate(methods_to_plot):
        curve = method_means[method]
        game_indices = np.arange(len(curve))
        
        # Plot the line
        ax.plot(game_indices, curve, label=method, linewidth=1.5, 
               color=colors[idx], alpha=0.8)
        
        # Store final position for label placement
        final_positions.append((method, curve[-1], colors[idx]))
    
    # Sort final positions for better label placement
    final_positions.sort(key=lambda x: x[1], reverse=True)
    
    # Get the actual max game index from curves
    max_game_idx = max(len(method_means[m]) - 1 for m in methods_to_plot)
    
    # Calculate appropriate label position
    label_x_position = max_game_idx + max(50, max_game_idx * 0.02)
    
    # Set x-axis limits with appropriate margin for labels
    x_max = max_game_idx + max(200, max_game_idx * 0.15)
    ax.set_xlim(-0.5, x_max)
    
    # Add method names at the end of lines
    if len(methods_to_plot) > 0:
        # Base positions are the final scores (so labels stay near their own curves)
        base_scores = np.array([final_score for _, final_score, _ in final_positions], dtype=float)
        n_labels = len(base_scores)
        
        if n_labels == 1:
            adjusted_scores = base_scores
        else:
            # Compute overall vertical range; avoid zero-range (all same score)
            y_min = float(base_scores.min())
            y_max = float(base_scores.max())
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1.0
            
            # Minimum vertical gap between labels.
            # Larger number of methods -> smaller gap, but keep it at least a few points.
            min_gap = y_range / (n_labels * 1.3)
            min_gap = max(min_gap, 5.0)
            
            # Start from the true final scores and iteratively "relax" them
            # so that neighbouring labels are at least min_gap apart.
            adjusted_scores = base_scores.copy()
            for _ in range(12):  # a few relaxation passes are enough
                moved = False
                for i in range(n_labels - 1):
                    gap = adjusted_scores[i] - adjusted_scores[i + 1]
                    if gap < min_gap:
                        # Push the upper label up and the lower label down symmetrically
                        shift = (min_gap - gap) / 2.0
                        adjusted_scores[i] += shift
                        adjusted_scores[i + 1] -= shift
                        moved = True
                if not moved:
                    break
        
        # Finally draw labels at the adjusted positions
        for (method, _final_score, color), y_pos in zip(final_positions, adjusted_scores):
            ax.text(
                label_x_position, y_pos, method,
                fontsize=10, va='center', color=color, fontweight='bold'
            )
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Game Index', fontweight='bold')
    ax.set_ylabel('Cumulative Score', fontweight='bold')
    ax.set_title(f'Score Evolution over {max_game_idx} Games (by Method)', 
                 fontsize=14, fontweight='bold', pad=20)

    # Set adaptive x-axis ticks similar to analyze_multi_seed_2.py:
    # Let Matplotlib choose a reasonable number of ticks based on the data range.
    from matplotlib.ticker import MaxNLocator

    # Use at most around 8 major ticks and keep them as integers (game indices)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    # Avoid scientific notation on the x-axis
    ax.ticklabel_format(style='plain', axis='x')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Don't use legend if too many methods, labels on the right are sufficient
    if len(methods_to_plot) <= 10:
        ax.legend(loc='best', framealpha=0.9, ncol=2 if len(methods_to_plot) > 6 else 1)
    
    plt.tight_layout()
    
    # Save
    fig_path = Path(output_dir) / "figure_5_performance_evolution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")


def save_summary_csv(stats: pd.DataFrame, all_seeds: pd.DataFrame, output_dir: str) -> None:
    """Save comprehensive summary statistics to CSV (without agent_type_summary.csv)"""
    # Add ranking
    stats['rank'] = stats['mean'].rank(ascending=False).astype(int)
    
    # Reorder columns
    cols = ['rank', 'mean', 'std', 'min', 'max', 'median', 'ci95', 'lower_bound', 'upper_bound']
    stats = stats[cols]
    
    # Save summary statistics only
    summary_path = Path(output_dir) / "summary_statistics.csv"
    stats.to_csv(summary_path, float_format='%.2f')
    print(f"Saved summary: {summary_path}")
    
    # Note: agent_type_summary.csv has been removed as requested


def main(args):
    """Main analysis pipeline"""
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Load data
    print("\n📊 Loading experiment data...")
    metadata, stats, all_seeds, round_data, n_rounds = load_experiment_data(args.input_dir)
    
    if stats is None or all_seeds is None:
        print("Error: Required data files not found")
        return
    
    print(f"Loaded {len(all_seeds)} total records")
    print(f"Analyzing {len(stats)} agents")
    print(f"Rounds: {n_rounds}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # 1. Score Distribution
    print("Creating score distribution plot...")
    plot_score_distribution(all_seeds, stats, args.output_dir)
    
    # 2. Confidence Intervals
    print("Creating confidence intervals plot...")
    plot_confidence_intervals(stats, args.output_dir)
    
    # 3. Correlation Matrix
    print("Creating correlation matrix...")
    plot_correlation_matrix(all_seeds, args.output_dir)
    
    # 4. Stability Analysis
    print("Creating stability analysis plot...")
    plot_stability_analysis(all_seeds, stats, args.output_dir)
    
    # 5. Performance Evolution
    print("Creating performance evolution plot...")
    plot_performance_over_rounds(args.input_dir, args.output_dir, all_seeds, n_rounds)
    
    # Save summary statistics
    print("\n💾 Saving summary statistics...")
    save_summary_csv(stats, all_seeds, args.output_dir)
    
    # Print summary
    print("\n✅ Analysis complete!")
    print(f"All results saved to: {output_path}")
    
    # Display top performers
    print("\n🏆 Top 5 Performers (by mean score):")
    top_5 = stats.nlargest(5, 'mean')[['mean', 'std', 'median']]
    for agent, row in top_5.iterrows():
        print(f"  {agent}: {row['mean']:.1f} ± {row['std']:.1f}")
    
    print("\n📉 Bottom 5 Performers (by mean score):")
    bottom_5 = stats.nsmallest(5, 'mean')[['mean', 'std', 'median']]
    for agent, row in bottom_5.iterrows():
        print(f"  {agent}: {row['mean']:.1f} ± {row['std']:.1f}")
    
    # Stability champions
    cv = (stats['std'] / stats['mean'].abs() * 100)
    cv_valid = cv[cv.notna() & ~np.isinf(cv)]
    
    if len(cv_valid) > 0:
        print("\n🎯 Most Stable Agents (lowest CV):")
        stable_5 = cv_valid.nsmallest(5)
        for agent, cv_value in stable_5.items():
            print(f"  {agent}: CV={cv_value:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze multi-seed RPS experiment results')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory containing experiment results')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    main(args)