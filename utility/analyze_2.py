#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" updated on 2025-11-18
analyze_multi_seed_2.py - Multi-seed Experiment Analysis Tool part 2
Emphsizing on Mean, not Median, i.e. changeing legend of mean/median from analyze_multi_seed_2_median.py
RPS tournament results summary in details
"""
import argparse
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# 统计检验
try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0

# Professional color palettes (fixed to use actual color lists)
PALETTES = {
    'nature': ['#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97', '#6A6599', '#80796B'],
    'science': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9', '#F0E442'],
    'cell': ['#0073B7', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#CC79A7', '#999999'],
    'deep': sns.color_palette("deep"),
    'muted': sns.color_palette("muted"),
    'bright': sns.color_palette("bright"),
    'colorblind': sns.color_palette("colorblind"),
    'husl': sns.color_palette("husl", 12)
}

# RPS baseline (changed from 0.5 to 0.33)
RPS_BASELINE = 0.33  # Theoretical random baseline for Rock-Paper-Scissors

def get_palette_colors(palette_name='nature', n_colors=None):
    """Get actual color list from palette name"""
    if palette_name in PALETTES:
        colors = PALETTES[palette_name]
        if n_colors and n_colors > len(colors):
            # Cycle colors if needed
            colors = colors * (n_colors // len(colors) + 1)
        return colors[:n_colors] if n_colors else colors
    else:
        # Fallback to seaborn palette
        return sns.color_palette("husl", n_colors) if n_colors else sns.color_palette("husl")

def set_publication_style(palette='nature'):
    """Set publication-quality plot style with selected palette"""
    sns.set_context("paper", rc={"lines.linewidth": 2})
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.15',
        'axes.linewidth': 1.25
    })
    
    # Set color palette - use actual colors, not name
    colors = get_palette_colors(palette)
    sns.set_palette(colors)

def _ensure_dirs(out_dir: str) -> Dict[str, str]:
    tables = os.path.join(out_dir, "tables")
    figs = os.path.join(out_dir, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    return {"tables": tables, "figures": figs}

def _parse_idxname(agent_str: str) -> Tuple[Optional[int], str]:
    if not isinstance(agent_str, str):
        return (None, str(agent_str))
    if "_" in agent_str:
        parts = agent_str.split("_", 1)
        if len(parts) == 2:
            seat_str, rest = parts
            try:
                seat = int(seat_str)
                # Successfully parsed seat number
                return (seat, rest)
            except ValueError:
                # First part is not a number, so no seat prefix
                # Return the full agent_str as the name
                return (None, agent_str)
    return (None, agent_str)

def _scan_seeds(input_dir: str) -> List[int]:
    seeds = []
    pat = re.compile(r"RPS_train_summary_seed(\d+)\.csv$")
    for fn in os.listdir(input_dir):
        m = pat.match(fn)
        if m:
            seeds.append(int(m.group(1)))
    seeds = sorted(list(set(seeds)))
    return seeds

def load_scores_by_seed(input_dir: str) -> pd.DataFrame:
    seeds = _scan_seeds(input_dir)
    rows = []
    for sd in tqdm(seeds, desc="Load seed summaries"):
        path = os.path.join(input_dir, f"RPS_train_summary_seed{sd}.csv")
        tqdm.write(f"[read] {path}")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "agent" not in df.columns or "score" not in df.columns:
            # 兼容其它命名
            cols = [c for c in df.columns if c.lower().strip() in {"agent", "who_agent"}]
            score_col = [c for c in df.columns if "score" in c.lower()]
            if cols:
                a_col = cols[0]
            else:
                raise ValueError(f"{path} 缺少 agent 列")
            if score_col:
                s_col = score_col[0]
            else:
                raise ValueError(f"{path} 缺少 score 列")
            df = df.rename(columns={a_col: "agent", s_col: "score"})

        for _, r in df.iterrows():
            seat, method = _parse_idxname(str(r["agent"]))
            rows.append({"seed": sd, "agent": r["agent"], "seat": seat, "method": method, "score": int(r["score"])})

    out = pd.DataFrame(rows)
    return out

def method_stats_across_seeds(scores_by_seed: pd.DataFrame) -> pd.DataFrame:
    grp = scores_by_seed.groupby("method")["score"]
    stats = grp.agg(["count", "mean", "std", "min", "max", "median"]).reset_index()
    stats["ci95"] = stats["std"] * 1.96 / np.sqrt(stats["count"].clip(lower=1))
    stats["lower_bound"] = stats["mean"] - stats["ci95"]
    stats["upper_bound"] = stats["mean"] + stats["ci95"]
    q05 = grp.quantile(0.05)
    stats = stats.merge(q05.rename("q05"), on="method", how="left")
    return stats.sort_values(by="mean", ascending=False)

def seat_method_stats(scores_by_seed: pd.DataFrame) -> pd.DataFrame:
    df = scores_by_seed.dropna(subset=["seat"])
    grp = df.groupby(["method", "seat"])["score"]
    stats = grp.agg(["count", "mean", "std", "min", "max", "median"]).reset_index()
    stats["ci95"] = stats["std"] * 1.96 / np.sqrt(stats["count"].clip(lower=1))
    stats["lower_bound"] = stats["mean"] - stats["ci95"]
    stats["upper_bound"] = stats["mean"] + stats["ci95"]
    return stats.sort_values(["method", "seat"])

def _holm_bonferroni(pairs: List[Tuple[Tuple[str, str], float]]) -> Dict[Tuple[str, str], float]:
    m = len(pairs)
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    adj = {}
    prev = 0.0
    for i, ((a,b), p) in enumerate(sorted_pairs, start=1):
        p_holm = (m - i + 1) * p
        p_holm = max(prev, p_holm)
        p_holm = min(1.0, p_holm)
        adj[(a,b)] = p_holm
        prev = p_holm
    return {k: adj[k] for (k, _) in pairs}

def nonparam_wilcoxon_holm(scores_by_seed: pd.DataFrame, alpha: float=0.05) -> pd.DataFrame:
    methods = sorted(scores_by_seed["method"].unique().tolist())
    pivot = scores_by_seed.pivot_table(index="seed", columns="method", values="score", aggfunc="mean")
    pairs = []
    raw_results = []
    it = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            it.append((methods[i], methods[j]))
    for (a,b) in tqdm(it, desc="Wilcoxon pairwise"):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        df = pivot[[a,b]].dropna()
        if len(df) < 5:
            continue
        x = df[a].values
        y = df[b].values
        if _HAVE_SCIPY:
            try:
                stat, p = wilcoxon(x, y, zero_method="pratt", alternative="two-sided", correction=False)
            except Exception:
                stat, p = np.nan, 1.0
        else:
            # 简单符号检验近似
            diff = x - y
            pos = np.sum(diff > 0); neg = np.sum(diff < 0)
            n_eff = pos + neg
            if n_eff == 0:
                p = 1.0; stat = 0
            else:
                from math import comb
                p_one = sum(comb(n_eff, k) * (0.5**n_eff) for k in range(pos, n_eff+1))
                p = min(1.0, 2 * min(p_one, 1 - p_one)); stat = pos - neg
        raw_results.append(((a,b), (len(df), stat, p)))
        pairs.append(((a,b), p))

    if not pairs:
        return pd.DataFrame(columns=["A","B","n","stat","p_raw","p_adj_holm","significant"])

    adj_map = _holm_bonferroni(pairs)
    rows = []
    for (a,b), (n, stat, p) in raw_results:
        p_adj = adj_map[(a,b)]
        rows.append({"A": a, "B": b, "n": n, "stat": stat, "p_raw": p, "p_adj_holm": p_adj, "significant": bool(p_adj < alpha)})
    out = pd.DataFrame(rows).sort_values(["p_adj_holm","p_raw"])
    return out

def _agent_method_from_idxname(idxname: str) -> str:
    return _parse_idxname(idxname)[1]

def winrate_distribution_fast(input_dir: str, seeds: List[int]) -> pd.DataFrame:
    """
    向量化：读取 RPS_record_seed*.csv，按方法×种子统计胜平负与占比。
    """
    rows_all = []
    for sd in tqdm(seeds, desc="Winrate seeds"):
        path = os.path.join(input_dir, f"RPS_record_seed{sd}.csv")
        tqdm.write(f"[read] {path}")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["who_agent","whom_agent","winner"])
        if df.empty:
            continue
        # who 视角
        who_ct = df.groupby(["who_agent","winner"]).size().unstack(fill_value=0)
        # whom 视角
        whom_ct = df.groupby(["whom_agent","winner"]).size().unstack(fill_value=0)
        # 标准化列
        for tab in (who_ct, whom_ct):
            for col in ("who","whom","tie"):
                if col not in tab.columns:
                    tab[col] = 0
        who_stats = pd.DataFrame({
            "agent": who_ct.index,
            "wins": who_ct["who"].values,
            "ties": who_ct["tie"].values,
            "losses": who_ct["whom"].values,
        })
        whom_stats = pd.DataFrame({
            "agent": whom_ct.index,
            "wins": whom_ct["whom"].values,
            "ties": whom_ct["tie"].values,
            "losses": whom_ct["who"].values,
        })
        stats = pd.concat([who_stats, whom_stats], ignore_index=True)
        if stats.empty:
            continue
        stats["games"] = stats[["wins","ties","losses"]].sum(axis=1)
        stats["method"] = [ _agent_method_from_idxname(str(a)) for a in stats["agent"] ]
        # 聚合到方法
        agg = stats.groupby("method")[["games","wins","ties","losses"]].sum().reset_index()
        agg["seed"] = sd
        agg["win_rate"] = agg["wins"] / agg["games"].replace(0, np.nan)
        agg["tie_rate"] = agg["ties"] / agg["games"].replace(0, np.nan)
        rows_all.append(agg)
    if not rows_all:
        return pd.DataFrame(columns=["seed","method","games","wins","ties","losses","win_rate","tie_rate"])
    return pd.concat(rows_all, ignore_index=True)

def perf_evolution_seaborn(input_dir: str, seeds: List[int], methods: Optional[List[str]], 
                          out_dir: str, max_steps: Optional[int]=None, palette='nature'):
    """
    Enhanced performance evolution curves with Seaborn styling (FIXED)
    修复版：确保正确读取数据并生成性能演化曲线
    Modified: Mean is now the main thick line, median is the dashed line
    """
    # 先扫描所有方法
    methods_used = set()
    seed_files = [os.path.join(input_dir, f"RPS_record_seed{sd}.csv") for sd in seeds]
    
    # 扫描第一个存在的文件以拉取方法集合
    for fp in seed_files:
        if os.path.exists(fp):
            try:
                # 只读取agent列来获取方法列表
                df = pd.read_csv(fp, nrows=1000)  # 读取前1000行作为样本
                if 'who_agent' in df.columns and 'whom_agent' in df.columns:
                    agents = pd.concat([df["who_agent"], df["whom_agent"]]).dropna().unique()
                    methods_used |= set(_agent_method_from_idxname(str(x)) for x in agents)
                    if methods_used:  # 如果找到了方法，就停止扫描
                        break
            except Exception as e:
                tqdm.write(f"[warn] Error scanning {fp}: {e}")
                continue
    
    # 如果没有找到方法，尝试从scores数据推断
    if not methods_used:
        tqdm.write("[info] No methods found in RPS_record files, trying to infer from summary files...")
        summary_files = [os.path.join(input_dir, f"RPS_train_summary_seed{sd}.csv") for sd in seeds]
        for sf in summary_files:
            if os.path.exists(sf):
                try:
                    df = pd.read_csv(sf)
                    if 'agent' in df.columns:
                        methods_used |= set(_agent_method_from_idxname(str(x)) for x in df['agent'].unique())
                except:
                    pass
    
    methods_list = sorted(list(methods_used if not methods else set(methods) & methods_used))
    
    if not methods_list:
        tqdm.write("[warn] No methods found for performance evolution curves")
        return
    
    tqdm.write(f"[info] Found {len(methods_list)} methods for performance curves: {methods_list}")
    
    # Get color palette for methods
    colors = get_palette_colors(palette, len(methods_list))
    
    for idx, method in enumerate(tqdm(methods_list, desc="Perf evolution (methods)")):
        tqdm.write(f"[perf] method={method}")
        curves = []
        
        for sd in tqdm(seeds, desc=f"  seeds for {method}", leave=False):
            path = os.path.join(input_dir, f"RPS_record_seed{sd}.csv")
            if not os.path.exists(path):
                continue
                
            tqdm.write(f"  [read] {path}")
            
            try:
                # 尝试读取所有需要的列
                required_cols = ["who_agent", "whom_agent", "score_delta_who"]
                optional_cols = ["round", "pair_index"]
                
                # 首先检查哪些列存在
                df_sample = pd.read_csv(path, nrows=1)
                available_cols = list(df_sample.columns)
                
                # 构建要读取的列列表
                cols_to_read = [col for col in required_cols if col in available_cols]
                cols_to_read.extend([col for col in optional_cols if col in available_cols])
                
                if len(cols_to_read) < len(required_cols):
                    tqdm.write(f"    [skip] Missing required columns in {path}")
                    continue
                
                # 读取数据
                df = pd.read_csv(path, usecols=cols_to_read)
                
                # 尝试排序（如果有排序列）
                if "round" in df.columns and "pair_index" in df.columns:
                    df = df.sort_values(["round", "pair_index"]).reset_index(drop=True)
                elif "round" in df.columns:
                    df = df.sort_values("round").reset_index(drop=True)
                else:
                    df = df.reset_index(drop=True)
                
                if df.empty:
                    continue
                
                who = df["who_agent"].astype(str).values
                whom = df["whom_agent"].astype(str).values
                d = df["score_delta_who"].astype(int).values
                
                # 限制长度（若 max_steps 设定）
                if max_steps is not None and len(d) > max_steps:
                    who = who[:max_steps]
                    whom = whom[:max_steps]
                    d = d[:max_steps]
                
                # 计算每个方法的贡献
                who_m = np.array([_agent_method_from_idxname(x) for x in who], dtype=object)
                whom_m = np.array([_agent_method_from_idxname(x) for x in whom], dtype=object)
                
                contrib = (who_m == method) * d + (whom_m == method) * (-d)
                curves.append(np.cumsum(contrib))
                
            except Exception as e:
                tqdm.write(f"    [error] Failed to process {path}: {e}")
                continue
        
        if not curves:
            tqdm.write(f"  [skip] No data for method {method}")
            continue
        
        # 对齐长度
        min_len = min(len(c) for c in curves)
        mat = np.vstack([c[:min_len] for c in curves])
        median = np.median(mat, axis=0)
        q25 = np.quantile(mat, 0.25, axis=0)
        q75 = np.quantile(mat, 0.75, axis=0)
        mean = np.mean(mat, axis=0)
        
        # 保存 CSV
        steps = np.arange(min_len)
        df_curve = pd.DataFrame({"step": steps, "median": median, "q25": q25, "q75": q75, "mean": mean})
        csv_path = os.path.join(out_dir, "tables", f"perf_evolution_{method}.csv")
        df_curve.to_csv(csv_path, index=False, encoding="utf-8")
        tqdm.write(f"[write] {csv_path}")
        
        # Create publication-quality plot with Seaborn
        png_path = os.path.join(out_dir, "figures", f"perf_curve_{method}.png")
        
        # Set style for this plot
        set_publication_style(palette)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Main line - MODIFIED: Mean is now the main thick solid line
        color = colors[idx % len(colors)]
        ax.plot(steps, mean, label=f"{method} (mean)", 
                linewidth=2.5, color=color)
        
        # Confidence interval
        ax.fill_between(steps, q25, q75, alpha=0.25, color=color)
        
        # Median line - MODIFIED: Median is now the dashed line
        ax.plot(steps, median, '--', alpha=0.6, linewidth=1.5, 
                color=color, label=f"{method} (median)")
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # Styling
        ax.set_xlabel("Game Index", fontweight='bold', fontsize=12)
        ax.set_ylabel("Cumulative Score", fontweight='bold', fontsize=12)
        ax.set_title(f"Performance Evolution - {method}", fontweight='bold', fontsize=14)
        
        # Legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add annotation for final score - MODIFIED: Use mean instead of median
        if len(mean) > 0:
            final_mean = mean[-1]
            ax.annotate(f'Final: {final_mean:.0f}',
                       xy=(steps[-1], final_mean),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=1.0),
                       color='white',  # 字体颜色改为白色
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        tqdm.write(f"[write] {png_path}")

def winrate_visualizations(winrate_df: pd.DataFrame, out_dir: str, palette='nature'):
    """
    Enhanced winrate visualizations with baseline=0.33 (instead of 0.5)
    """
    if winrate_df.empty:
        return
    
    set_publication_style(palette)
    methods = sorted(winrate_df["method"].unique().tolist())
    
    # Get actual colors for this palette
    palette_colors = get_palette_colors(palette, len(methods))
    
    # 1. Individual histograms (enhanced)
    for i, m in enumerate(tqdm(methods, desc="Win-rate histograms")):
        d = winrate_df[winrate_df["method"] == m]["win_rate"].dropna().values
        if len(d) == 0:
            continue
            
        png_path = os.path.join(out_dir, "figures", f"winrate_hist_{m}.png")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use seaborn for beautiful histogram
        hist_color = palette_colors[i % len(palette_colors)]
        sns.histplot(d, bins=10, kde=True, stat='density', 
                    color=hist_color, alpha=0.7, ax=ax)
        
        # Add vertical lines for mean and median
        ax.axvline(np.mean(d), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(d):.3f}')
        ax.axvline(np.median(d), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(d):.3f}')
        
        # Add baseline at 0.33
        ax.axvline(RPS_BASELINE, color='red', linestyle=':', linewidth=1.5,
                  label=f'Baseline: {RPS_BASELINE:.2f}')
        
        ax.set_xlabel("Win Rate", fontweight='bold', fontsize=12)
        ax.set_ylabel("Density", fontweight='bold', fontsize=12)
        ax.set_title(f"Win Rate Distribution - {m}", fontweight='bold', fontsize=14)
        ax.legend(loc='best', frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        tqdm.write(f"[write] {png_path}")
    
    # 2. Combined violin plot for all methods (with baseline=0.33)
    if len(methods) > 1:
        png_path = os.path.join(out_dir, "figures", "winrate_violin_all.png")
        
        fig, ax = plt.subplots(figsize=(max(10, len(methods)*0.8), 8))
        
        # Prepare data for violin plot
        plot_data = []
        for m in methods:
            rates = winrate_df[winrate_df["method"] == m]["win_rate"].dropna().values
            for r in rates:
                plot_data.append({'Method': m, 'Win Rate': r})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot - pass actual colors, not palette name
        sns.violinplot(data=plot_df, x='Method', y='Win Rate', 
                      palette=palette_colors,  # Fixed: use actual colors
                      inner='box', ax=ax)
        
        # Add horizontal line at 0.33 (changed from 0.5)
        ax.axhline(y=RPS_BASELINE, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                  label=f'Baseline ({RPS_BASELINE:.2f})')
        
        ax.set_xlabel("Method", fontweight='bold', fontsize=12)
        ax.set_ylabel("Win Rate", fontweight='bold', fontsize=12)
        ax.set_title("Win Rate Distribution Comparison", fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        
        # Rotate x labels if many methods
        if len(methods) > 8:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Auto-adaptive y-axis (matplotlib will handle this automatically)
        # No need to set ylim explicitly
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        tqdm.write(f"[write] {png_path}")
    
    # 3. Box plot comparison (with baseline=0.33)
    if len(methods) > 1:
        png_path = os.path.join(out_dir, "figures", "winrate_boxplot_all.png")
        
        fig, ax = plt.subplots(figsize=(max(10, len(methods)*0.8), 8))
        
        # Prepare data
        plot_data = []
        for m in methods:
            rates = winrate_df[winrate_df["method"] == m]["win_rate"].dropna().values
            for r in rates:
                plot_data.append({'Method': m, 'Win Rate': r})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot with swarm overlay - use actual colors
        sns.boxplot(data=plot_df, x='Method', y='Win Rate',
                   palette=palette_colors,  # Fixed: use actual colors
                   ax=ax)
        sns.swarmplot(data=plot_df, x='Method', y='Win Rate',
                     color='black', alpha=0.3, size=3, ax=ax)
        
        # Add horizontal line at 0.33 (changed from 0.5)
        ax.axhline(y=RPS_BASELINE, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                  label=f'Baseline ({RPS_BASELINE:.2f})')
        
        ax.set_xlabel("Method", fontweight='bold', fontsize=12)
        ax.set_ylabel("Win Rate", fontweight='bold', fontsize=12)
        ax.set_title("Win Rate Box Plot Comparison", fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        
        if len(methods) > 8:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Auto-adaptive y-axis
        # Matplotlib/Seaborn will automatically adjust the y-axis range
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        tqdm.write(f"[write] {png_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default="RPS_train_summary", help="训练输出目录（包含 *_seed*.csv）")
    ap.add_argument("--out-dir", type=str, default="RPS_analysis_v3_seaborn", help="分析输出目录")
    ap.add_argument("--methods", type=str, default="", help="仅分析这些方法，逗号分隔（留空=全部）")
    ap.add_argument("--max-steps", type=int, default=None, help="性能演化曲线的最大步数（可加速绘图）")
    ap.add_argument("--palette", type=str, default="nature", 
                   choices=['nature', 'science', 'cell', 'deep', 'muted', 'bright', 'colorblind', 'husl'],
                   help="Color palette for visualizations")
    args = ap.parse_args()

    t0 = time.time()
    out_dirs = _ensure_dirs(args.out_dir)
    os.makedirs(args.input_dir, exist_ok=True)
    methods_filter = [s.strip() for s in args.methods.split(",") if s.strip()]

    # 1) 每种子得分
    scores_by_seed = load_scores_by_seed(args.input_dir)
    if scores_by_seed.empty:
        print("[warn] 未在 input-dir 中找到 RPS_train_summary_seed*.csv，无法完成分析。")
        return
    scores_by_seed.to_csv(os.path.join(out_dirs["tables"], "scores_by_seed.csv"), index=False, encoding="utf-8")
    tqdm.write(f"[write] {os.path.join(out_dirs['tables'], 'scores_by_seed.csv')}")

    # 2) 每方法统计
    mstats = method_stats_across_seeds(scores_by_seed)
    mstats.to_csv(os.path.join(out_dirs["tables"], "method_stats.csv"), index=False, encoding="utf-8")
    tqdm.write(f"[write] {os.path.join(out_dirs['tables'], 'method_stats.csv')}")

    # 3) 每席位×每方法统计
    smstats = seat_method_stats(scores_by_seed)
    smstats.to_csv(os.path.join(out_dirs["tables"], "seat_method_stats.csv"), index=False, encoding="utf-8")
    tqdm.write(f"[write] {os.path.join(out_dirs['tables'], 'seat_method_stats.csv')}")

    # 4) 非参数检验 + Holm
    npt = nonparam_wilcoxon_holm(scores_by_seed)
    npt.to_csv(os.path.join(out_dirs["tables"], "nonparam_wilcoxon_holm.csv"), index=False, encoding="utf-8")
    tqdm.write(f"[write] {os.path.join(out_dirs['tables'], 'nonparam_wilcoxon_holm.csv')}")

    # 5) 胜率分布（向量化）
    seeds = sorted(scores_by_seed["seed"].unique().tolist())
    winrate_df = winrate_distribution_fast(args.input_dir, seeds)
    if not winrate_df.empty:
        win_csv = os.path.join(out_dirs["tables"], "winrate_distribution.csv")
        winrate_df.to_csv(win_csv, index=False, encoding="utf-8")
        tqdm.write(f"[write] {win_csv}")
        winrate_visualizations(winrate_df, args.out_dir, args.palette)

    # 6) 下行风险
    downside = mstats[["method","q05","lower_bound"]].copy()
    downside = downside.rename(columns={"q05":"p05_score", "lower_bound":"ci95_lower"})
    down_csv = os.path.join(out_dirs["tables"], "method_downside.csv")
    downside.to_csv(down_csv, index=False, encoding="utf-8")
    tqdm.write(f"[write] {down_csv}")

    # 7) 性能演化曲线（Seaborn增强版 - 修复版）
    tqdm.write("\n[info] Starting performance evolution analysis...")
    perf_evolution_seaborn(args.input_dir, seeds, methods_filter or None, 
                          args.out_dir, max_steps=args.max_steps, palette=args.palette)
    
    # 8) 不再创建summary_dashboard（已删除）
    # create_summary_dashboard已被移除
    
    t1 = time.time()
    print(f"\n✅ 完成分析。输出目录：{args.out_dir}  （总耗时 {t1 - t0:.1f}s）")
    print("  - 统计表：tables/*.csv")
    print("  - 图像：figures/*.png (Publication-ready Seaborn visualizations)")
    print(f"  - 配色方案：{args.palette}")
    print(f"  - RPS基准线：{RPS_BASELINE:.2f} (理论随机胜率)")
    
    # 列出实际生成的文件
    print("\n生成的文件:")
    import glob
    png_files = glob.glob(os.path.join(args.out_dir, "figures", "*.png"))
    csv_files = glob.glob(os.path.join(args.out_dir, "tables", "*.csv"))
    
    print(f"  PNG文件 ({len(png_files)}个):")
    for f in sorted(png_files)[:10]:  # 显示前10个
        print(f"    - {os.path.basename(f)}")
    if len(png_files) > 10:
        print(f"    ... 还有 {len(png_files)-10} 个文件")
    
    print(f"  CSV文件 ({len(csv_files)}个):")
    for f in sorted(csv_files)[:10]:  # 显示前10个
        print(f"    - {os.path.basename(f)}")

if __name__ == "__main__":
    main()