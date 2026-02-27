#!/usr/bin/env python3
"""
Cumulative Score Evolution Analysis

Generates cumulative score trajectories over tournament rounds, aggregated by method family.
This produces figure_si_score_evolution.png for the Supplementary Information.

Usage:
OUT=outputs/paper_full_54_r500_s10
python utility/analyze_score_evolution.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_evolution \
  --palette nature
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------
PALETTES = {
    "nature": {
        "colors": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
                   "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                   "#DDA0DD", "#98D8C8", "#F7DC6F"],
        "bg": "#FFFFFF",
        "grid": "#E5E5E5",
        "text": "#333333",
    },
    "science": {
        "colors": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280",
                   "#BB0021", "#5F559B", "#A20056", "#808180", "#1B1919"],
        "bg": "#FFFFFF",
        "grid": "#E5E5E5",
        "text": "#333333",
    },
    "cell": {
        "colors": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                   "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
        "bg": "#FFFFFF",
        "grid": "#E5E5E5",
        "text": "#333333",
    },
}


def load_perf_evolution_files(input_dir: str) -> dict:
    """
    Load pre-computed performance evolution files from analysis_2/tables/.
    
    Returns dict mapping method name to DataFrame with columns:
    step, median, q25, q75, mean
    """
    input_path = Path(input_dir)
    
    # Try analysis_2/tables/ directory
    tables_dir = input_path / "analysis_2" / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Could not find {tables_dir}")
    
    method_data = {}
    for csv_file in sorted(tables_dir.glob("perf_evolution_*.csv")):
        # Extract method name from filename: perf_evolution_RNN_v2.csv -> RNN_v2
        method_name = csv_file.stem.replace("perf_evolution_", "")
        try:
            df = pd.read_csv(csv_file)
            method_data[method_name] = df
            print(f"[info] Loaded {method_name}: {len(df)} steps")
        except Exception as e:
            print(f"[warn] Could not load {csv_file}: {e}")
    
    if not method_data:
        raise FileNotFoundError(
            f"No perf_evolution_*.csv files found in {tables_dir}"
        )
    
    return method_data


def load_raw_records(input_dir: str) -> pd.DataFrame:
    """
    Load raw tournament records from RPS_record_seed*.csv files.
    
    Returns DataFrame with all round-level data.
    """
    input_path = Path(input_dir)
    
    all_records = []
    for csv_file in sorted(input_path.glob("RPS_record_seed*.csv")):
        try:
            # Extract seed from filename: RPS_record_seed1.csv -> 1
            seed_str = csv_file.stem.replace("RPS_record_seed", "")
            seed = int(seed_str)
            
            df = pd.read_csv(csv_file)
            df["seed"] = seed
            all_records.append(df)
            print(f"[info] Loaded seed {seed}: {len(df)} records")
        except Exception as e:
            print(f"[warn] Could not load {csv_file}: {e}")
    
    if not all_records:
        raise FileNotFoundError(
            f"No RPS_record_seed*.csv files found in {input_dir}"
        )
    
    return pd.concat(all_records, ignore_index=True)


def get_method_family(agent_name: str) -> str:
    """Extract method family from agent name (e.g., '01_RNN_v2' -> 'RNN_v2')."""
    parts = agent_name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return agent_name


def compute_cumulative_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative scores from raw tournament records.
    
    Input columns: round, who_agent, whom_agent, score_delta_who, seed
    """
    records = []
    
    # Process each row: who_agent gets score_delta_who, whom_agent gets -score_delta_who
    for _, row in df.iterrows():
        seed = row["seed"]
        round_num = row["round"]
        
        # who's perspective
        records.append({
            "seed": seed,
            "round": round_num,
            "agent": row["who_agent"],
            "score": row["score_delta_who"],
        })
        
        # whom's perspective (opponent gets negative of who's delta)
        records.append({
            "seed": seed,
            "round": round_num,
            "agent": row["whom_agent"],
            "score": -row["score_delta_who"],
        })
    
    expanded = pd.DataFrame(records)
    expanded["method"] = expanded["agent"].apply(get_method_family)
    
    # Aggregate scores per agent per round per seed (sum across all matchups in that round)
    round_scores = expanded.groupby(["seed", "method", "round"]).agg({
        "score": "sum"
    }).reset_index()
    
    # Sort and compute cumulative
    round_scores = round_scores.sort_values(["seed", "method", "round"])
    round_scores["cumulative"] = round_scores.groupby(["seed", "method"])["score"].cumsum()
    
    # Aggregate across seeds: mean, std, quartiles
    method_round = round_scores.groupby(["method", "round"]).agg({
        "cumulative": ["mean", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).reset_index()
    
    method_round.columns = ["method", "round", "mean", "std", "q25", "q75"]
    
    return method_round


def plot_evolution_from_precomputed(
    method_data: dict,
    output_path: str,
    palette_name: str = "nature",
    title: str = "Cumulative Score Evolution by Method",
    figsize: tuple = (12, 8),
):
    """
    Plot cumulative score evolution using pre-computed perf_evolution files.
    """
    palette = PALETTES.get(palette_name, PALETTES["nature"])
    colors = palette["colors"]
    
    # Sort methods by final mean score
    final_scores = {}
    for method, df in method_data.items():
        if "mean" in df.columns and len(df) > 0:
            final_scores[method] = df["mean"].iloc[-1]
    
    methods_sorted = sorted(final_scores.keys(), key=lambda m: final_scores[m], reverse=True)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["bg"])
    ax.set_facecolor(palette["bg"])
    
    # Ensure enough colors
    n_methods = len(methods_sorted)
    if n_methods > len(colors):
        colors = (colors * ((n_methods // len(colors)) + 1))[:n_methods]
    
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods_sorted)}
    
    # Plot each method
    for method in methods_sorted:
        df = method_data[method]
        if "step" in df.columns:
            x = df["step"]
        else:
            x = df.index
        
        ax.plot(
            x,
            df["mean"],
            label=method,
            color=color_map[method],
            linewidth=1.5,
            alpha=0.85,
        )
    
    # Add zero line
    ax.axhline(y=0, color=palette["grid"], linestyle="--", linewidth=1.0, alpha=0.7)
    
    # Formatting
    ax.set_xlabel("Tournament Progress (Round)", fontsize=12, color=palette["text"])
    ax.set_ylabel("Mean Cumulative Score", fontsize=12, color=palette["text"])
    ax.set_title(title, fontsize=14, fontweight="bold", color=palette["text"])
    
    ax.tick_params(colors=palette["text"], labelsize=10)
    ax.grid(True, alpha=0.3, color=palette["grid"])
    
    # Legend outside plot
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        frameon=True,
        facecolor=palette["bg"],
        edgecolor=palette["grid"],
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=palette["bg"])
    plt.close()
    
    print(f"[save] Score evolution plot: {output_path}")


def plot_evolution_with_bands(
    method_data: dict,
    output_path: str,
    selected_methods: list = None,
    palette_name: str = "nature",
    title: str = "Cumulative Score Evolution (Selected Methods)",
    figsize: tuple = (10, 7),
):
    """
    Plot cumulative score evolution with IQR bands for selected methods.
    """
    palette = PALETTES.get(palette_name, PALETTES["nature"])
    colors = palette["colors"]
    
    if selected_methods is None:
        # Default: top 5 and bottom 3 by final score
        final_scores = {}
        for method, df in method_data.items():
            if "mean" in df.columns and len(df) > 0:
                final_scores[method] = df["mean"].iloc[-1]
        
        sorted_methods = sorted(final_scores.keys(), key=lambda m: final_scores[m], reverse=True)
        selected_methods = sorted_methods[:5] + sorted_methods[-3:]
    
    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["bg"])
    ax.set_facecolor(palette["bg"])
    
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(selected_methods)}
    
    for method in selected_methods:
        if method not in method_data:
            continue
        
        df = method_data[method]
        if "step" in df.columns:
            x = df["step"]
        else:
            x = df.index
        
        color = color_map[method]
        
        # Plot mean line
        ax.plot(
            x,
            df["mean"],
            label=method,
            color=color,
            linewidth=2.0,
            alpha=0.9,
        )
        
        # Plot IQR band if available
        if "q25" in df.columns and "q75" in df.columns:
            ax.fill_between(
                x,
                df["q25"],
                df["q75"],
                color=color,
                alpha=0.15,
            )
    
    ax.axhline(y=0, color=palette["grid"], linestyle="--", linewidth=1.0, alpha=0.7)
    
    ax.set_xlabel("Tournament Progress (Round)", fontsize=12, color=palette["text"])
    ax.set_ylabel("Mean Cumulative Score", fontsize=12, color=palette["text"])
    ax.set_title(title, fontsize=14, fontweight="bold", color=palette["text"])
    
    ax.tick_params(colors=palette["text"], labelsize=10)
    ax.grid(True, alpha=0.3, color=palette["grid"])
    ax.legend(loc="best", fontsize=10, frameon=True, facecolor=palette["bg"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=palette["bg"])
    plt.close()
    
    print(f"[save] Selected evolution plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cumulative score evolution plots by method family."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing tournament output (with analysis_2/tables/ or RPS_record_seed*.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures (default: {input-dir}/analysis_evolution).",
    )
    parser.add_argument(
        "--palette",
        choices=["nature", "science", "cell"],
        default="nature",
        help="Color palette for plots.",
    )
    parser.add_argument(
        "--selected-methods",
        default=None,
        help="Comma-separated list of methods to highlight (default: auto-select top/bottom).",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Force using raw RPS_record_seed*.csv files instead of pre-computed evolution data.",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "analysis_evolution"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Cumulative Score Evolution Analysis")
    print("=" * 60)
    
    method_data = None
    
    # Try loading pre-computed evolution data first (unless --use-raw specified)
    if not args.use_raw:
        try:
            print(f"[info] Looking for pre-computed evolution data...")
            method_data = load_perf_evolution_files(input_dir)
            print(f"[info] Loaded {len(method_data)} methods from perf_evolution files.")
        except FileNotFoundError as e:
            print(f"[info] Pre-computed data not found: {e}")
    
    # Fall back to raw records if needed
    if method_data is None:
        print(f"[info] Loading raw tournament records...")
        try:
            raw_df = load_raw_records(input_dir)
            print(f"[info] Computing cumulative scores from {len(raw_df)} records...")
            
            # Convert to method_data format
            cumulative_df = compute_cumulative_from_raw(raw_df)
            
            method_data = {}
            for method in cumulative_df["method"].unique():
                method_df = cumulative_df[cumulative_df["method"] == method].copy()
                method_df = method_df.rename(columns={"round": "step"})
                method_data[method] = method_df
            
            print(f"[info] Computed evolution for {len(method_data)} methods.")
        except FileNotFoundError as e:
            print(f"[error] {e}")
            sys.exit(1)
    
    # Save combined data
    combined_rows = []
    for method, df in method_data.items():
        df_copy = df.copy()
        df_copy["method"] = method
        combined_rows.append(df_copy)
    
    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_df.to_csv(output_dir / "method_cumulative_scores.csv", index=False)
    print(f"[save] Combined data: {output_dir / 'method_cumulative_scores.csv'}")
    
    # Plot all methods
    plot_evolution_from_precomputed(
        method_data,
        output_dir / "figure_si_score_evolution.png",
        palette_name=args.palette,
        title="Cumulative Score Evolution by Method Family",
    )
    
    # Plot selected methods with IQR bands
    selected = None
    if args.selected_methods:
        selected = [m.strip() for m in args.selected_methods.split(",")]
    
    plot_evolution_with_bands(
        method_data,
        output_dir / "figure_si_score_evolution_selected.png",
        selected_methods=selected,
        palette_name=args.palette,
        title="Cumulative Score Evolution (Selected Methods)",
    )
    
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - figure_si_score_evolution.png")
    print(f"  - figure_si_score_evolution_selected.png")
    print(f"  - method_cumulative_scores.csv")


if __name__ == "__main__":
    main()