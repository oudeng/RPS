#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_lipschitz_tightness.py - Compute tightness ratio and certificate vs realized regret

This script computes additional Lipschitz diagnostics for the paper revision:
1. Tightness ratio: tau_t = Delta_cert / (2 * ||p - p_hat||_1)
2. Realized regret: Delta_play = U(a*, p) - U(a_play, p)
3. Comparison between certificate and realized regret
4. Exceedance rate for realized regret

Usage:
    python analyze_lipschitz_tightness.py --input-dir <lipschitz_data_dir> --output-dir <output_dir>

The input directory should contain CSV files with Lipschitz analysis data, 
specifically with columns: l1_distance, regret (certificate regret), regret_played, br_match

Output:
    - tightness_summary.csv: Summary statistics for each matchup direction
    - tightness_distributions.png: Visualization of tightness ratio distributions
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_tightness_ratio(l1_distance: np.ndarray, regret_cert: np.ndarray, 
                            epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute tightness ratio tau = Delta_cert / (2 * ||p - p_hat||_1)
    
    Returns values in [0, 1] by construction (assuming bound holds).
    For l1_distance near 0, we return 0 to avoid division issues.
    """
    bound = 2 * l1_distance
    # Avoid division by zero
    valid_mask = bound > epsilon
    tau = np.zeros_like(regret_cert)
    tau[valid_mask] = regret_cert[valid_mask] / bound[valid_mask]
    # Clip to [0, 1] for numerical stability
    tau = np.clip(tau, 0, 1)
    return tau


def check_exceedances(l1_distance: np.ndarray, regret: np.ndarray, 
                      tolerance: float = 1e-6) -> Tuple[int, float]:
    """
    Check how many points exceed the theoretical bound Delta <= 2 * ||p - p_hat||_1
    
    Returns:
        n_exceed: Number of exceedances
        exceed_rate: Fraction of exceedances
    """
    bound = 2 * l1_distance
    exceedances = regret > (bound + tolerance)
    n_exceed = np.sum(exceedances)
    exceed_rate = n_exceed / len(regret) if len(regret) > 0 else 0
    return n_exceed, exceed_rate


def analyze_matchup_data(data: pd.DataFrame, matchup_name: str) -> Dict:
    """
    Analyze data for a single matchup direction and compute all tightness/regret metrics.
    
    Expected columns:
        - l1_distance: ||p_t - p_hat_t||_1
        - regret: certificate regret Delta_cert
        - regret_played: realized regret Delta_play
        - br_match: whether played action = BR(p_hat)
    """
    results = {"matchup": matchup_name}
    
    l1 = data["l1_distance"].values
    regret_cert = data["regret"].values
    
    # Basic statistics
    results["n_samples"] = len(data)
    results["l1_mean"] = np.mean(l1)
    results["l1_std"] = np.std(l1)
    results["regret_cert_mean"] = np.mean(regret_cert)
    results["regret_cert_std"] = np.std(regret_cert)
    
    # Tightness ratio
    tau = compute_tightness_ratio(l1, regret_cert)
    # Only compute stats where l1 > epsilon (avoid 0/0)
    valid_tau = tau[l1 > 1e-6]
    if len(valid_tau) > 0:
        results["tau_mean"] = np.mean(valid_tau)
        results["tau_median"] = np.median(valid_tau)
        results["tau_p75"] = np.percentile(valid_tau, 75)
        results["tau_p90"] = np.percentile(valid_tau, 90)
    else:
        results["tau_mean"] = np.nan
        results["tau_median"] = np.nan
        results["tau_p75"] = np.nan
        results["tau_p90"] = np.nan
    
    # Certificate regret exceedances (should be 0 by construction)
    n_exceed_cert, rate_exceed_cert = check_exceedances(l1, regret_cert)
    results["cert_exceedances"] = n_exceed_cert
    results["cert_exceed_rate"] = rate_exceed_cert
    
    # Realized regret
    if "regret_played" in data.columns:
        regret_play = data["regret_played"].values
        results["regret_play_mean"] = np.mean(regret_play)
        results["regret_play_std"] = np.std(regret_play)
        
        # Realized regret exceedances
        n_exceed_play, rate_exceed_play = check_exceedances(l1, regret_play)
        results["play_exceedances"] = n_exceed_play
        results["play_exceed_rate"] = rate_exceed_play
        
        # Difference between realized and certificate regret
        regret_diff = regret_play - regret_cert
        results["regret_diff_mean"] = np.mean(regret_diff)
        results["regret_diff_std"] = np.std(regret_diff)
    
    # BR-match rate
    if "br_match" in data.columns:
        results["br_match_rate"] = np.mean(data["br_match"].values)
    
    # Spearman correlation (excluding zero-regret points)
    nonzero_mask = regret_cert > 1e-6
    if np.sum(nonzero_mask) > 10:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(l1[nonzero_mask], regret_cert[nonzero_mask])
        results["spearman_rho"] = rho
        results["spearman_pval"] = pval
    else:
        results["spearman_rho"] = np.nan
        results["spearman_pval"] = np.nan
    
    return results


def load_and_aggregate_by_matchup(input_dir: str, warmup: int = 50) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files and aggregate by matchup direction (who→whom).
    
    Returns:
        Dictionary mapping matchup name (e.g., "A3C_v2→RNN_v2") to aggregated DataFrame
    """
    # Find all lipschitz CSV files
    patterns = [
        os.path.join(input_dir, "lipschitz_seed*.csv"),
        os.path.join(input_dir, "lipschitz*.csv"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = list(set(files))
    
    if not files:
        print(f"No lipschitz CSV files found in {input_dir}")
        return {}
    
    print(f"Found {len(files)} data files")
    
    # Load and concatenate all data
    all_data = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not all_data:
        return {}
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter warmup rounds
    if "round" in combined.columns:
        combined = combined[combined["round"] > warmup]
    
    # Group by matchup direction
    matchup_data = {}
    if "who_agent" in combined.columns and "whom_agent" in combined.columns:
        for (who, whom), group in combined.groupby(["who_agent", "whom_agent"]):
            matchup_name = f"{who}→{whom}"
            matchup_data[matchup_name] = group.copy()
            print(f"  {matchup_name}: {len(group)} samples")
    else:
        # Fall back to treating each file as a matchup
        matchup_data["combined"] = combined
        print(f"  combined: {len(combined)} samples")
    
    return matchup_data


def plot_tightness_distributions(matchup_data: Dict[str, pd.DataFrame], 
                                 output_path: str) -> None:
    """Create visualization of tightness ratio distributions."""
    n_matchups = len(matchup_data)
    if n_matchups == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, min(n_matchups, 2), figsize=(6*min(n_matchups,2), 5))
    if n_matchups == 1:
        axes = [axes]
    
    colors = ['#0173B2', '#DE8F05']  # Blue and orange
    
    for idx, (matchup, data) in enumerate(matchup_data.items()):
        if idx >= 2:  # Only plot up to 2 matchups
            break
        
        ax = axes[idx]
        
        l1 = data["l1_distance"].values
        regret = data["regret"].values
        
        # Compute tightness ratio
        tau = compute_tightness_ratio(l1, regret)
        valid_tau = tau[l1 > 1e-6]
        
        # Plot histogram
        ax.hist(valid_tau, bins=50, density=True, alpha=0.7, color=colors[idx], 
                edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_tau = np.mean(valid_tau)
        median_tau = np.median(valid_tau)
        ax.axvline(mean_tau, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_tau:.3f}')
        ax.axvline(median_tau, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_tau:.3f}')
        
        ax.set_xlabel(r'Tightness ratio $\tau = \Delta^{cert} / (2\|p-\hat{p}\|_1)$')
        ax.set_ylabel('Density')
        ax.set_title(matchup.replace('→', r'$\rightarrow$'))
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tightness distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute Lipschitz tightness metrics")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing Lipschitz analysis CSV files")
    parser.add_argument("--output-dir", type=str, default="./tightness_output",
                        help="Output directory for results")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup rounds to exclude")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and aggregate data by matchup direction
    matchup_data = load_and_aggregate_by_matchup(args.input_dir, args.warmup)
    
    if not matchup_data:
        print("No data to analyze")
        return
    
    # Analyze each matchup direction
    all_results = []
    
    for matchup_name, data in matchup_data.items():
        print(f"\nAnalyzing: {matchup_name}")
        results = analyze_matchup_data(data, matchup_name)
        all_results.append(results)
    
    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, "tightness_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")
        
        # Print formatted table for paper
        print("\n" + "="*80)
        print("RESULTS FOR PAPER TABLE (copy these values to RPS_v5_5.tex)")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\nMatchup: {row['matchup']}")
            print(f"  n = {row.get('n_samples', 0)}")
            print(f"  ||p-p̂||₁: {row.get('l1_mean', np.nan):.3f}±{row.get('l1_std', np.nan):.3f}")
            print(f"  Δ^cert: {row.get('regret_cert_mean', np.nan):.3f}±{row.get('regret_cert_std', np.nan):.3f}")
            print(f"  Δ^play: {row.get('regret_play_mean', np.nan):.3f}±{row.get('regret_play_std', np.nan):.3f}")
            print(f"  τ̄ (mean): {row.get('tau_mean', np.nan):.3f}")
            print(f"  τ̃ (median): {row.get('tau_median', np.nan):.3f}")
            print(f"  ρ (Spearman): {row.get('spearman_rho', np.nan):.3f}")
            print(f"  BR-match: {row.get('br_match_rate', np.nan):.3f}")
            print(f"  Cert exceed rate: {row.get('cert_exceed_rate', np.nan):.3%}")
            print(f"  Play exceed rate: {row.get('play_exceed_rate', np.nan):.3%}")
        
        # Print LaTeX table rows
        print("\n" + "="*80)
        print("LATEX TABLE ROWS (paste into Table)")
        print("="*80)
        for _, row in summary_df.iterrows():
            matchup_latex = row['matchup'].replace('→', r'$\rightarrow$').replace('_', r'\_')
            l1_str = f"{row.get('l1_mean', np.nan):.3f}$\\pm${row.get('l1_std', np.nan):.3f}"
            cert_str = f"{row.get('regret_cert_mean', np.nan):.3f}$\\pm${row.get('regret_cert_std', np.nan):.3f}"
            play_str = f"{row.get('regret_play_mean', np.nan):.3f}$\\pm${row.get('regret_play_std', np.nan):.3f}"
            tau_mean = f"{row.get('tau_mean', np.nan):.3f}"
            tau_median = f"{row.get('tau_median', np.nan):.3f}"
            rho = f"{row.get('spearman_rho', np.nan):.3f}"
            br_match = f"{row.get('br_match_rate', np.nan):.3f}"
            print(f"{matchup_latex} & {l1_str} & {cert_str} & {play_str} & {tau_mean} & {tau_median} & {rho} & {br_match} \\\\")
    
    # Plot distributions
    if matchup_data:
        plot_path = os.path.join(args.output_dir, "tightness_distributions.png")
        plot_tightness_distributions(matchup_data, plot_path)


if __name__ == "__main__":
    main()