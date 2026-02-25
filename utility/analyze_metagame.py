#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_metagame.py - Non-transitive Meta-game Analysis Tool

Provides:
  1. Pairwise payoff matrix construction and heatmap visualization
  2. Cyclic dominance (3-cycle) detection and enumeration
  3. α-Rank stationary distribution computation
  4. Cross-pool rank correlation analysis

Usage:
# 1. Core-19 基础分析
OUT=outputs/val_core19_r500_s10
python utility/analyze_metagame.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/metagame_analysis \
  --alpha 0.1 \
  --palette nature

# 2. 带跨池比较的分析
python utility/analyze_metagame.py \
  --input-dir outputs/val_core19_r500_s10 \
  --output-dir outputs/val_core19_r500_s10/metagame_analysis \
  --alpha 0.1 \
  --compare-dirs "Core54:outputs/paper_full_54_r500_s10,TopR:outputs/overTopR_r500_s10"

# 3. 复制图表到论文目录
mkdir -p Fig/RPS_metagame
cp outputs/val_core19_r500_s10/metagame_analysis/figures/pairwise_heatmap.png \
   Fig/RPS_metagame/pairwise_heatmap_core19.png
cp outputs/val_core19_r500_s10/metagame_analysis/figures/alpha_rank_stationary.png \
   Fig/RPS_metagame/alpharank_stationary_core19.png
"""

import argparse
import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from itertools import permutations, combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Statistical imports
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# Set publication-quality defaults
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['axes.linewidth'] = 1.2

# Professional color palettes
PALETTES = {
    'nature': ['#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97', '#6A6599', '#80796B'],
    'science': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9', '#F0E442'],
    'cell': ['#0073B7', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#CC79A7', '#999999'],
}


def get_palette_colors(palette_name='nature', n_colors=None):
    """Get actual color list from palette name"""
    if palette_name in PALETTES:
        colors = PALETTES[palette_name]
        if n_colors and n_colors > len(colors):
            colors = colors * (n_colors // len(colors) + 1)
        return colors[:n_colors] if n_colors else colors
    else:
        return sns.color_palette("husl", n_colors) if n_colors else sns.color_palette("husl")


def set_publication_style(palette='nature'):
    """Set publication-quality plot style"""
    sns.set_context("paper", rc={"lines.linewidth": 2})
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.25
    })
    colors = get_palette_colors(palette)
    sns.set_palette(colors)


def _parse_idxname(agent_str: str) -> Tuple[Optional[int], str]:
    """Parse agent string to extract seat number and method name"""
    if not isinstance(agent_str, str):
        return (None, str(agent_str))
    if "_" in agent_str:
        parts = agent_str.split("_", 1)
        if len(parts) == 2:
            seat_str, rest = parts
            try:
                seat = int(seat_str)
                return (seat, rest)
            except ValueError:
                return (None, agent_str)
    return (None, agent_str)


def _agent_method(agent_str: str) -> str:
    """Extract method name from agent string"""
    return _parse_idxname(agent_str)[1]


def _scan_seeds(input_dir: str) -> List[int]:
    """Scan directory for available seed numbers"""
    seeds = []
    pat = re.compile(r"RPS_record_seed(\d+)\.csv$")
    for fn in os.listdir(input_dir):
        m = pat.match(fn)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(list(set(seeds)))


# =============================================================================
# 1. Pairwise Payoff Matrix Construction
# =============================================================================

def compute_pairwise_payoff_matrix(input_dir: str, seeds: List[int] = None,
                                    use_methods: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute pairwise payoff matrix G(i,j) = mean score of agent i against agent j.
    
    Args:
        input_dir: Directory containing RPS_record_seed*.csv files
        seeds: List of seeds to use (None = all available)
        use_methods: If True, aggregate by method name; if False, use full agent names
    
    Returns:
        payoff_matrix: DataFrame with G(i,j) values
        agents: List of agent/method names
    """
    if seeds is None:
        seeds = _scan_seeds(input_dir)
    
    if not seeds:
        raise ValueError(f"No seed files found in {input_dir}")
    
    print(f"[info] Computing pairwise payoffs from {len(seeds)} seeds...")
    
    # Accumulate pairwise scores across seeds
    pairwise_scores = defaultdict(list)  # (i, j) -> list of scores
    
    for sd in tqdm(seeds, desc="Loading records"):
        path = os.path.join(input_dir, f"RPS_record_seed{sd}.csv")
        if not os.path.exists(path):
            continue
        
        df = pd.read_csv(path)
        
        # Determine agent key function
        if use_methods:
            df['who_key'] = df['who_agent'].apply(_agent_method)
            df['whom_key'] = df['whom_agent'].apply(_agent_method)
        else:
            df['who_key'] = df['who_agent'].astype(str)
            df['whom_key'] = df['whom_agent'].astype(str)
        
        # Compute per-matchup scores
        # winner == 'who' means who wins (+1 for who, -1 for whom)
        # winner == 'whom' means whom wins (-1 for who, +1 for whom)
        # winner == 'tie' means draw (0 for both)
        
        df['who_score'] = df['winner'].map({'who': 1, 'whom': -1, 'tie': 0})
        df['whom_score'] = df['winner'].map({'who': -1, 'whom': 1, 'tie': 0})
        
        # Aggregate by matchup direction
        who_agg = df.groupby(['who_key', 'whom_key'])['who_score'].sum().reset_index()
        for _, row in who_agg.iterrows():
            pairwise_scores[(row['who_key'], row['whom_key'])].append(row['who_score'])
    
    # Get unique agents
    agents = sorted(set([k[0] for k in pairwise_scores.keys()] + 
                        [k[1] for k in pairwise_scores.keys()]))
    n = len(agents)
    
    # Build payoff matrix
    payoff_matrix = pd.DataFrame(np.zeros((n, n)), index=agents, columns=agents)
    
    for (i, j), scores in pairwise_scores.items():
        if i in agents and j in agents:
            payoff_matrix.loc[i, j] = np.mean(scores)
    
    print(f"[info] Payoff matrix: {n} agents, Frobenius norm = {np.linalg.norm(payoff_matrix.values):.2f}")
    
    return payoff_matrix, agents


def plot_pairwise_heatmap(payoff_matrix: pd.DataFrame, output_path: str,
                          highlight_cycle: List[str] = None,
                          title: str = "Pairwise Payoff Matrix",
                          figsize: Tuple[int, int] = (12, 10),
                          palette: str = 'nature') -> None:
    """
    Plot pairwise payoff matrix as heatmap with optional cycle highlighting.
    
    Args:
        payoff_matrix: DataFrame with G(i,j) values
        output_path: Path to save figure
        highlight_cycle: List of agents forming a cycle to highlight (e.g., ['A', 'B', 'C'])
        title: Plot title
        figsize: Figure size
        palette: Color palette name
    """
    set_publication_style(palette)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get matrix values
    matrix = payoff_matrix.values
    agents = list(payoff_matrix.index)
    n = len(agents)
    
    # Determine color limits (symmetric around 0)
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    vmin = -vmax
    
    # Create heatmap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Blue to Red
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Payoff G(i,j)', fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(agents, fontsize=12)
    
    # Labels
    ax.set_xlabel('Opponent (j)', fontsize=11)
    ax.set_ylabel('Agent (i)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Highlight cycle if provided
    if highlight_cycle and len(highlight_cycle) >= 3:
        # Draw arrows for cycle
        cycle_indices = [agents.index(a) for a in highlight_cycle if a in agents]
        if len(cycle_indices) >= 3:
            colors = get_palette_colors(palette)
            arrow_color = colors[1]  # Orange from nature palette
            
            for k in range(len(cycle_indices)):
                i_idx = cycle_indices[k]
                j_idx = cycle_indices[(k + 1) % len(cycle_indices)]
                
                # Draw arrow from (i, j) cell
                # Arrow points from row i to indicate i beats j
                start = (j_idx, i_idx)  # (x, y) in image coordinates
                end = (cycle_indices[(k + 1) % len(cycle_indices)], 
                       cycle_indices[(k + 2) % len(cycle_indices)])
                
                # Add rectangle highlight around the cell
                rect = plt.Rectangle((j_idx - 0.5, i_idx - 0.5), 1, 1, 
                                     fill=False, edgecolor=arrow_color, linewidth=3)
                ax.add_patch(rect)
            
            # Add cycle annotation
            cycle_str = ' → '.join(highlight_cycle) + ' → ' + highlight_cycle[0]
            ax.annotate(f'Cycle: {cycle_str}', xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=10, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[save] Pairwise heatmap: {output_path}")


# =============================================================================
# 2. Cyclic Dominance Detection
# =============================================================================

def detect_3_cycles(payoff_matrix: pd.DataFrame, 
                    min_edge_strength: float = 0.0) -> List[Tuple[str, str, str, float]]:
    """
    Detect all 3-cycles (A ⊃ B ⊃ C ⊃ A) in the payoff matrix.
    
    A cycle exists if G(A,B) > 0, G(B,C) > 0, G(C,A) > 0.
    
    Args:
        payoff_matrix: DataFrame with G(i,j) values
        min_edge_strength: Minimum G(i,j) value to count as dominance
    
    Returns:
        List of (A, B, C, min_strength) tuples sorted by minimum edge strength
    """
    agents = list(payoff_matrix.index)
    n = len(agents)
    cycles = []
    
    # Check all 3-combinations
    for combo in combinations(agents, 3):
        # Check all 6 orderings (2 cycle directions × 3 starting points)
        for perm in [(0, 1, 2), (0, 2, 1)]:  # Only need 2 directions
            a, b, c = combo[perm[0]], combo[perm[1]], combo[perm[2]]
            
            g_ab = payoff_matrix.loc[a, b]
            g_bc = payoff_matrix.loc[b, c]
            g_ca = payoff_matrix.loc[c, a]
            
            if g_ab > min_edge_strength and g_bc > min_edge_strength and g_ca > min_edge_strength:
                min_strength = min(g_ab, g_bc, g_ca)
                avg_strength = (g_ab + g_bc + g_ca) / 3
                # Canonical form: start with alphabetically smallest
                cycle_agents = [a, b, c]
                min_idx = cycle_agents.index(min(cycle_agents))
                canonical = tuple(cycle_agents[min_idx:] + cycle_agents[:min_idx])
                cycles.append((canonical[0], canonical[1], canonical[2], 
                              min_strength, avg_strength, g_ab, g_bc, g_ca))
    
    # Remove duplicates and sort by min_strength descending
    seen = set()
    unique_cycles = []
    for cycle in cycles:
        key = (cycle[0], cycle[1], cycle[2])
        if key not in seen:
            seen.add(key)
            unique_cycles.append(cycle)
    
    unique_cycles.sort(key=lambda x: x[3], reverse=True)
    
    print(f"[info] Detected {len(unique_cycles)} 3-cycles with min_edge_strength > {min_edge_strength}")
    
    return unique_cycles


def save_cycles_table(cycles: List[Tuple], output_path: str) -> pd.DataFrame:
    """Save cycles to CSV and return DataFrame"""
    df = pd.DataFrame(cycles, columns=['Agent_A', 'Agent_B', 'Agent_C', 
                                        'Min_Strength', 'Avg_Strength',
                                        'G(A,B)', 'G(B,C)', 'G(C,A)'])
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"[save] Cycles table: {output_path}")
    return df


# =============================================================================
# 3. α-Rank Computation
# =============================================================================

def compute_alpha_rank(payoff_matrix: pd.DataFrame, alpha: float = 0.1,
                       max_iterations: int = 10000, tol: float = 1e-8) -> pd.DataFrame:
    """
    Compute α-Rank stationary distribution over strategies.
    
    α-Rank models evolutionary selection dynamics using a Markov chain
    where transition probabilities depend on fitness differences scaled by α.
    
    Args:
        payoff_matrix: Symmetric payoff matrix G(i,j)
        alpha: Selection intensity parameter (larger = more deterministic selection)
        max_iterations: Maximum power iteration steps
        tol: Convergence tolerance
    
    Returns:
        DataFrame with agents and their stationary masses
    """
    agents = list(payoff_matrix.index)
    n = len(agents)
    G = payoff_matrix.values
    
    # Build transition matrix P
    # P[i,j] = probability of transitioning from population i to population j
    # Using Fermi selection: P(i→j) ∝ 1 / (1 + exp(-α * (fitness_j - fitness_i)))
    
    # For pure strategy populations, fitness of strategy i when facing j is G(i,j)
    # Transition i→j happens when a j-mutant invades i-population
    
    P = np.zeros((n, n))
    
    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                # Fitness advantage of j over i when j is rare in i-population
                # Approximation: use G(j,i) as invasion fitness
                fitness_diff = G[j, i]  # j's payoff against i
                
                # Fermi probability
                prob = 1.0 / (1.0 + np.exp(-alpha * fitness_diff))
                P[i, j] = prob
                row_sum += prob
        
        # Normalize row
        if row_sum > 0:
            P[i, :] /= row_sum
            # Self-transition for remaining probability
            P[i, i] = 0  # Will be set after normalization
        else:
            P[i, i] = 1.0  # Absorbing state
    
    # Ensure rows sum to 1
    for i in range(n):
        P[i, i] = max(0, 1.0 - np.sum(P[i, :]) + P[i, i])
    
    # Compute stationary distribution via power iteration
    pi = np.ones(n) / n
    
    for iteration in range(max_iterations):
        pi_new = pi @ P
        pi_new = pi_new / pi_new.sum()  # Normalize
        
        if np.max(np.abs(pi_new - pi)) < tol:
            print(f"[info] α-Rank converged after {iteration+1} iterations")
            break
        pi = pi_new
    else:
        print(f"[warn] α-Rank did not converge within {max_iterations} iterations")
    
    # Create result DataFrame
    result = pd.DataFrame({
        'agent': agents,
        'mass': pi,
        'rank': scipy_stats.rankdata(-pi, method='min')
    }).sort_values('mass', ascending=False)
    
    print(f"[info] α-Rank (α={alpha}): top strategy = {result.iloc[0]['agent']} "
          f"with mass = {result.iloc[0]['mass']:.4f}")
    
    return result


def plot_alpha_rank(alpha_rank_df: pd.DataFrame, output_path: str,
                    alpha: float = 0.1, title: str = None,
                    figsize: Tuple[int, int] = (10, 6),
                    palette: str = 'nature') -> None:
    """
    Plot α-Rank stationary distribution as bar chart.
    """
    set_publication_style(palette)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by mass
    df = alpha_rank_df.sort_values('mass', ascending=True)
    
    colors = get_palette_colors(palette, len(df))
    
    bars = ax.barh(range(len(df)), df['mass'], color=colors[0], edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['agent'], fontsize=9)
    ax.set_xlabel('Stationary Mass', fontsize=11)
    ax.set_ylabel('Strategy', fontsize=11)
    
    if title is None:
        title = f'α-Rank Stationary Distribution (α={alpha})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['mass'] > 0.01:
            ax.text(row['mass'] + 0.005, i, f"{row['mass']:.3f}", 
                   va='center', fontsize=8)
    
    ax.set_xlim(0, df['mass'].max() * 1.15)
    ax.axvline(x=1/len(df), color='red', linestyle='--', alpha=0.7, 
               label=f'Uniform ({1/len(df):.3f})')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[save] α-Rank plot: {output_path}")


# =============================================================================
# 4. Cross-Pool Rank Correlation
# =============================================================================

def compute_rank_correlation(scores_a: pd.DataFrame, scores_b: pd.DataFrame,
                              name_a: str, name_b: str) -> Dict:
    """
    Compute Spearman and Kendall rank correlations between two evaluation pools.
    
    Args:
        scores_a: DataFrame with columns ['agent', 'score'] from pool A
        scores_b: DataFrame with columns ['agent', 'score'] from pool B
        name_a: Name of pool A
        name_b: Name of pool B
    
    Returns:
        Dictionary with correlation statistics
    """
    # Get common agents (by method name)
    scores_a = scores_a.copy()
    scores_b = scores_b.copy()
    
    if 'method' not in scores_a.columns:
        scores_a['method'] = scores_a['agent'].apply(_agent_method)
    if 'method' not in scores_b.columns:
        scores_b['method'] = scores_b['agent'].apply(_agent_method)
    
    # Aggregate by method if multiple instances
    agg_a = scores_a.groupby('method')['score'].mean().reset_index()
    agg_b = scores_b.groupby('method')['score'].mean().reset_index()
    
    # Find common methods
    common = set(agg_a['method']) & set(agg_b['method'])
    
    if len(common) < 3:
        print(f"[warn] Only {len(common)} common methods between {name_a} and {name_b}")
        return {
            'pool_a': name_a,
            'pool_b': name_b,
            'n_common': len(common),
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'kendall_tau': np.nan,
            'kendall_p': np.nan
        }
    
    # Filter to common methods
    agg_a = agg_a[agg_a['method'].isin(common)].set_index('method')
    agg_b = agg_b[agg_b['method'].isin(common)].set_index('method')
    
    # Align indices
    common_methods = sorted(common)
    scores_vec_a = agg_a.loc[common_methods, 'score'].values
    scores_vec_b = agg_b.loc[common_methods, 'score'].values
    
    # Compute correlations
    spearman_rho, spearman_p = scipy_stats.spearmanr(scores_vec_a, scores_vec_b)
    kendall_tau, kendall_p = scipy_stats.kendalltau(scores_vec_a, scores_vec_b)
    
    return {
        'pool_a': name_a,
        'pool_b': name_b,
        'n_common': len(common),
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def load_pool_scores(input_dir: str) -> pd.DataFrame:
    """Load aggregated scores from an experiment directory"""
    # Try different possible paths
    paths_to_try = [
        os.path.join(input_dir, "RPS_train_statistics.csv"),
        os.path.join(input_dir, "RPS_train_all_seeds.csv"),
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'agent' in df.columns and 'score' in df.columns:
                return df
            elif 'agent' in df.columns and 'mean' in df.columns:
                df['score'] = df['mean']
                return df
    
    # Try to aggregate from per-seed summaries
    seeds = []
    pat = re.compile(r"RPS_train_summary_seed(\d+)\.csv$")
    for fn in os.listdir(input_dir):
        m = pat.match(fn)
        if m:
            seeds.append(int(m.group(1)))
    
    if seeds:
        rows = []
        for sd in seeds:
            path = os.path.join(input_dir, f"RPS_train_summary_seed{sd}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    rows.append({'agent': row['agent'], 'score': row['score'], 'seed': sd})
        
        if rows:
            df = pd.DataFrame(rows)
            # Aggregate by agent
            agg = df.groupby('agent')['score'].mean().reset_index()
            return agg
    
    raise ValueError(f"Could not find score data in {input_dir}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_metagame_analysis(input_dir: str, output_dir: str,
                          alpha: float = 0.1,
                          min_cycle_strength: float = 0.0,
                          palette: str = 'nature',
                          comparison_dirs: Dict[str, str] = None) -> Dict:
    """
    Run complete metagame analysis pipeline.
    
    Args:
        input_dir: Directory with RPS experiment results
        output_dir: Directory to save analysis outputs
        alpha: α parameter for α-Rank
        min_cycle_strength: Minimum edge strength for cycle detection
        palette: Color palette for plots
        comparison_dirs: Dict of {name: dir_path} for cross-pool correlation
    
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    
    results = {}
    
    # 1. Compute pairwise payoff matrix
    print("\n" + "="*60)
    print("1. Computing Pairwise Payoff Matrix")
    print("="*60)
    
    payoff_matrix, agents = compute_pairwise_payoff_matrix(input_dir, use_methods=True)
    
    # Save matrix
    matrix_path = os.path.join(output_dir, 'tables', 'pairwise_payoff_matrix.csv')
    payoff_matrix.to_csv(matrix_path, float_format='%.2f')
    print(f"[save] Payoff matrix: {matrix_path}")
    
    results['payoff_matrix'] = payoff_matrix
    results['n_agents'] = len(agents)
    results['frobenius_norm'] = np.linalg.norm(payoff_matrix.values)
    results['matrix_rank'] = np.linalg.matrix_rank(payoff_matrix.values)
    
    # 2. Detect cycles
    print("\n" + "="*60)
    print("2. Detecting 3-Cycles")
    print("="*60)
    
    cycles = detect_3_cycles(payoff_matrix, min_edge_strength=min_cycle_strength)
    
    cycles_path = os.path.join(output_dir, 'tables', 'detected_3_cycles.csv')
    cycles_df = save_cycles_table(cycles, cycles_path)
    
    results['n_cycles'] = len(cycles)
    results['cycles'] = cycles
    
    # Get strongest cycle for highlighting
    strongest_cycle = None
    if cycles:
        strongest_cycle = [cycles[0][0], cycles[0][1], cycles[0][2]]
        results['strongest_cycle'] = strongest_cycle
        results['strongest_cycle_strength'] = cycles[0][3]
        print(f"[info] Strongest cycle: {' → '.join(strongest_cycle)} → {strongest_cycle[0]} "
              f"(min strength = {cycles[0][3]:.2f})")
    
    # 3. Plot heatmap
    print("\n" + "="*60)
    print("3. Generating Pairwise Heatmap")
    print("="*60)
    
    heatmap_path = os.path.join(output_dir, 'figures', 'pairwise_heatmap.png')
    plot_pairwise_heatmap(payoff_matrix, heatmap_path, 
                          highlight_cycle=strongest_cycle,
                          palette=palette)
    
    # 4. Compute α-Rank
    print("\n" + "="*60)
    print(f"4. Computing α-Rank (α={alpha})")
    print("="*60)
    
    alpha_rank_df = compute_alpha_rank(payoff_matrix, alpha=alpha)
    
    alpha_rank_path = os.path.join(output_dir, 'tables', 'alpha_rank_distribution.csv')
    alpha_rank_df.to_csv(alpha_rank_path, index=False, float_format='%.6f')
    print(f"[save] α-Rank table: {alpha_rank_path}")
    
    results['alpha_rank'] = alpha_rank_df
    
    # Plot α-Rank
    alpha_rank_fig_path = os.path.join(output_dir, 'figures', 'alpha_rank_stationary.png')
    plot_alpha_rank(alpha_rank_df, alpha_rank_fig_path, alpha=alpha, palette=palette)
    
    # 5. Cross-pool correlation (if comparison dirs provided)
    if comparison_dirs:
        print("\n" + "="*60)
        print("5. Computing Cross-Pool Rank Correlations")
        print("="*60)
        
        # Load scores from current pool
        try:
            current_scores = load_pool_scores(input_dir)
            current_name = os.path.basename(input_dir.rstrip('/'))
            
            correlations = []
            for other_name, other_dir in comparison_dirs.items():
                if not os.path.exists(other_dir):
                    print(f"[warn] Comparison directory not found: {other_dir}")
                    continue
                
                try:
                    other_scores = load_pool_scores(other_dir)
                    corr = compute_rank_correlation(current_scores, other_scores,
                                                    current_name, other_name)
                    correlations.append(corr)
                    print(f"  {current_name} vs {other_name}: "
                          f"ρ={corr['spearman_rho']:.3f}, τ={corr['kendall_tau']:.3f} "
                          f"(n={corr['n_common']})")
                except Exception as e:
                    print(f"[warn] Error loading {other_dir}: {e}")
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                corr_path = os.path.join(output_dir, 'tables', 'rank_correlation.csv')
                corr_df.to_csv(corr_path, index=False, float_format='%.3f')
                print(f"[save] Rank correlation table: {corr_path}")
                results['rank_correlations'] = corr_df
        
        except Exception as e:
            print(f"[warn] Could not compute cross-pool correlations: {e}")
    
    # 6. Summary statistics
    print("\n" + "="*60)
    print("6. Summary Statistics")
    print("="*60)
    
    summary = {
        'n_agents': results['n_agents'],
        'frobenius_norm': results['frobenius_norm'],
        'matrix_rank': results['matrix_rank'],
        'n_3_cycles': results['n_cycles'],
        'alpha': alpha,
        'top_alpha_rank_agent': alpha_rank_df.iloc[0]['agent'],
        'top_alpha_rank_mass': alpha_rank_df.iloc[0]['mass'],
    }
    
    if strongest_cycle:
        summary['strongest_cycle'] = ' → '.join(strongest_cycle)
        summary['strongest_cycle_min_strength'] = results['strongest_cycle_strength']
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, 'tables', 'metagame_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"[save] Summary: {summary_path}")
    
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    results['summary'] = summary
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Non-transitive Meta-game Analysis for RPS Tournament',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python utility/analyze_metagame.py \\
    --input-dir outputs/val_core19_r500_s10 \\
    --output-dir outputs/val_core19_r500_s10/metagame_analysis

  # With cross-pool comparison
  python utility/analyze_metagame.py \\
    --input-dir outputs/val_core19_r500_s10 \\
    --output-dir outputs/val_core19_r500_s10/metagame_analysis \\
    --compare-dirs "Core54:outputs/paper_full_54_r500_s10,TopR:outputs/overTopR_r500_s10"
        """
    )
    
    parser.add_argument('--input-dir', '-i', required=True,
                        help='Input directory containing RPS experiment results')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory for analysis (default: {input-dir}/metagame_analysis)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter for α-Rank (default: 0.1)')
    parser.add_argument('--min-cycle-strength', type=float, default=0.0,
                        help='Minimum edge strength for cycle detection (default: 0.0)')
    parser.add_argument('--palette', default='nature',
                        choices=['nature', 'science', 'cell'],
                        help='Color palette (default: nature)')
    parser.add_argument('--compare-dirs', default=None,
                        help='Comma-separated list of "name:path" pairs for cross-pool correlation')
    
    args = parser.parse_args()
    
    # Set default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'metagame_analysis')
    
    # Parse comparison directories
    comparison_dirs = None
    if args.compare_dirs:
        comparison_dirs = {}
        for pair in args.compare_dirs.split(','):
            if ':' in pair:
                name, path = pair.split(':', 1)
                comparison_dirs[name.strip()] = path.strip()
    
    # Run analysis
    run_metagame_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        min_cycle_strength=args.min_cycle_strength,
        palette=args.palette,
        comparison_dirs=comparison_dirs
    )


if __name__ == '__main__':
    main()