#!/usr/bin/env python3
"""Merge method scores, runtime, and param counts into a unified table
and produce a publication-quality compute–performance trade-off scatter.

Outputs:
  - outputs/sophistication/compute_tradeoff_table.csv
  - paper_RPS/Fig/RPS_sophistication/score_vs_compute.png
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure code_RPS is on sys.path
_code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

# ---------- paths ----------
CORE54_STATS = "outputs/paper_full_54_r500_s10/analysis_2/tables/method_stats.csv"
RUNTIME_CSV  = "outputs/runtime_full18/runtime_benchmark_summary.csv"
PARAMS_CSV   = "outputs/param_counts/torch_param_counts.csv"
OUT_TABLE    = "outputs/sophistication/compute_tradeoff_table.csv"
OUT_FIG      = "/home/dengou/RPS/paper_RPS/Fig/RPS_sophistication/score_vs_compute.png"

# ---------- tier definitions ----------
TIER_MAP = {
    "R": 0,
    "CG": 1, "WL": 1,
    "B_v1": 2, "B_v2": 2, "M_v1": 2, "M_v2": 2,
    "SVM": 3, "RF": 3, "XGB": 3,
    "RNN_v2": 4, "LSTM_v1": 4, "LSTM_v2": 4,
    "Tr_v1": 4, "Tr_v2": 4, "MSA_v2": 4,
    "A3C_v1": 5, "A3C_v2": 5,
}

TIER_LABELS = {
    0: "Random baseline",
    1: "Rule-based heuristic",
    2: "Probabilistic/Markov",
    3: "Classical ML",
    4: "Neural predictor",
    5: "Actor\u2013critic RL",
}

# Nature-inspired palette
TIER_COLORS = {
    0: "#8491B4",  # slate
    1: "#4DBBD5",  # cyan
    2: "#00A087",  # green
    3: "#F39B7F",  # salmon
    4: "#E64B35",  # red
    5: "#3C5488",  # dark blue
}

TIER_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    4: "v",
    5: "P",
}


def main():
    os.chdir(_code_dir)

    # ---- load data ----
    ms = pd.read_csv(CORE54_STATS)
    rt = pd.read_csv(RUNTIME_CSV)
    pc = pd.read_csv(PARAMS_CSV)

    # normalize column names
    rt = rt.rename(columns={"agent": "method"})

    # merge
    df = ms[["method", "mean", "std", "lower_bound", "upper_bound"]].merge(
        rt[["method", "ms_per_decision"]], on="method", how="left"
    ).merge(
        pc.rename(columns={"torch_params": "n_params"}), on="method", how="left"
    )

    # add tier
    df["tier"] = df["method"].map(TIER_MAP)
    df["tier_label"] = df["tier"].map(TIER_LABELS)

    # filter to 18 methods
    df = df[df["method"].isin(TIER_MAP.keys())].copy()

    # sort by tier then mean descending
    df = df.sort_values(["tier", "mean"], ascending=[True, False])

    # ---- save table ----
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    df.to_csv(OUT_TABLE, index=False)
    print(f"[OK] Table saved: {OUT_TABLE}")
    print(df[["method", "tier", "n_params", "ms_per_decision", "mean"]].to_string(index=False))

    # ---- plot ----
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    # plot each tier as a group (for legend)
    for tier in sorted(df["tier"].unique()):
        sub = df[df["tier"] == tier]
        yerr_low = sub["mean"] - sub["lower_bound"]
        yerr_high = sub["upper_bound"] - sub["mean"]
        ax.errorbar(
            sub["ms_per_decision"], sub["mean"],
            yerr=[yerr_low.values, yerr_high.values],
            fmt=TIER_MARKERS[tier],
            color=TIER_COLORS[tier],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=9,
            elinewidth=0.8,
            capsize=2,
            capthick=0.6,
            label=f"Tier {tier}: {TIER_LABELS[tier]}",
            zorder=5,
        )

    # label each point
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in df.iterrows():
            t = ax.text(
                row["ms_per_decision"], row["mean"],
                row["method"],
                fontsize=7, ha="center", va="bottom",
                color="#333333",
            )
            texts.append(t)
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4),
            expand=(1.4, 1.6),
            force_text=(0.5, 0.8),
        )
        print("[OK] adjustText applied")
    except ImportError:
        print("[WARN] adjustText not available, using manual offset")
        for _, row in df.iterrows():
            ax.annotate(
                row["method"],
                (row["ms_per_decision"], row["mean"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=6.5, color="#333333",
            )

    ax.set_xscale("log")
    ax.set_xlabel("End-to-end runtime (ms/decision, CPU)", fontsize=10)
    ax.set_ylabel("Mean tournament score (Core-54)", fontsize=10)

    # add zero line
    ax.axhline(y=0, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # legend
    ax.legend(
        fontsize=7.5, loc="lower right",
        frameon=True, fancybox=False, edgecolor="#cccccc",
        borderpad=0.6, labelspacing=0.5,
    )

    # style: Nature-clean
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"[OK] Figure saved: {OUT_FIG}")

    # also save a copy in code_RPS for reference
    local_copy = "outputs/sophistication/score_vs_compute.png"
    fig.savefig(local_copy, dpi=300, bbox_inches="tight")
    print(f"[OK] Local copy: {local_copy}")


if __name__ == "__main__":
    main()
