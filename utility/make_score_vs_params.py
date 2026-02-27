#!/usr/bin/env python3
"""
Generate publication-quality score vs. parameter count scatter plot
for neural agent families (Fig S8 in SI).

Only includes neural/RL agents (8 agents with trainable parameters).
Uses tiered colour coding, Nature-style formatting, and error bars.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- data ----------
# Values from Table S13 (which matches Table 2 / method_stats.csv)
DATA = {
    # method: (tier, params, mean_score)
    "MSA_v2":  (4, 537,     2410.7),
    "RNN_v2":  (4, 13443,   7955.5),
    "LSTM_v1": (4, 17859,   5186.6),
    "A3C_v1":  (5, 20996,   3380.9),
    "A3C_v2":  (5, 25604,   3837.5),
    "LSTM_v2": (4, 79283,   3204.0),
    "Tr_v1":   (4, 417667,  1386.9),
    "Tr_v2":   (4, 564291,  -98.4),
}

# 95% CI from method_stats.csv (will be loaded if available)
STATS_CSV = "outputs/paper_full_54_r500_s10/analysis_2/tables/method_stats.csv"

# ---------- style ----------
TIER_COLORS = {
    4: "#E64B35",  # red — Neural predictor
    5: "#3C5488",  # dark blue — Actor-critic RL
}
TIER_MARKERS = {
    4: "o",  # circle
    5: "D",  # diamond
}
TIER_LABELS = {
    4: "Neural predictor (Tier 4)",
    5: "Actor\u2013critic RL (Tier 5)",
}

# ---------- load CI data ----------
ci_dict = {}
if os.path.exists(STATS_CSV):
    df_stats = pd.read_csv(STATS_CSV)
    for _, row in df_stats.iterrows():
        m = row["method"]
        if m in DATA:
            # Use ci95 as symmetric half-width (robust to mean differences
            # between Table S13 and method_stats.csv aggregation)
            ci_dict[m] = float(row.get("ci95", 0))
    print(f"[OK] Loaded CI data for {len(ci_dict)} methods from {STATS_CSV}")
else:
    print(f"[WARN] {STATS_CSV} not found, no error bars")

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7, 5))

for tier in sorted(TIER_COLORS.keys()):
    methods = [(m, d) for m, d in DATA.items() if d[0] == tier]
    if not methods:
        continue

    xs = [d[1] for _, d in methods]
    ys = [d[2] for _, d in methods]

    # Error bars (symmetric ci95)
    yerr_vals = []
    has_ci = True
    for m, d in methods:
        if m in ci_dict:
            yerr_vals.append(ci_dict[m])
        else:
            yerr_vals.append(0)
            has_ci = False

    ax.errorbar(
        xs, ys,
        yerr=yerr_vals if has_ci else None,
        fmt=TIER_MARKERS[tier],
        color=TIER_COLORS[tier],
        markeredgecolor="white",
        markeredgewidth=0.6,
        markersize=10,
        elinewidth=0.8,
        capsize=3,
        capthick=0.6,
        label=TIER_LABELS[tier],
        zorder=5,
    )

# Labels with adjustText
try:
    from adjustText import adjust_text
    texts = []
    for m, (tier, params, score) in DATA.items():
        t = ax.text(
            params, score,
            m.replace("_", " "),
            fontsize=8,
            ha="center", va="bottom",
            color="#333333",
            fontweight="medium",
        )
        texts.append(t)
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4),
        expand=(1.5, 1.8),
        force_text=(0.6, 1.0),
    )
    print("[OK] adjustText applied")
except ImportError:
    print("[WARN] adjustText not available, using manual labels")
    offsets = {
        "MSA_v2": (-15, 10), "RNN_v2": (15, 10), "LSTM_v1": (15, -15),
        "A3C_v1": (-15, -15), "A3C_v2": (15, 10), "LSTM_v2": (15, -15),
        "Tr_v1": (-15, 10), "Tr_v2": (15, -15),
    }
    for m, (tier, params, score) in DATA.items():
        ox, oy = offsets.get(m, (5, 5))
        ax.annotate(
            m.replace("_", " "),
            (params, score),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=7.5, color="#333333",
        )

# Zero line
ax.axhline(y=0, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

# Axis
ax.set_xscale("log")
ax.set_xlabel("Trainable parameters (log scale)", fontsize=11)
ax.set_ylabel("Mean tournament score (Core-54)", fontsize=11)

# Legend
ax.legend(
    fontsize=9, loc="upper right",
    frameon=True, fancybox=False, edgecolor="#cccccc",
    borderpad=0.6, labelspacing=0.4,
)

# Nature-clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

fig.tight_layout(pad=1.0)

# Save
OUT_PAPER = os.path.join(os.path.dirname(__file__), "..", "..", "paper_RPS", "Fig", "RPS_sophistication", "score_vs_params.png")
OUT_LOCAL = "outputs/sophistication/score_vs_params.png"
os.makedirs(os.path.dirname(OUT_LOCAL), exist_ok=True)
os.makedirs(os.path.dirname(OUT_PAPER), exist_ok=True)

fig.savefig(OUT_PAPER, dpi=300)
fig.savefig(OUT_LOCAL, dpi=300)
print(f"[OK] Saved: {OUT_PAPER}")
print(f"[OK] Saved: {OUT_LOCAL}")

# Report dimensions
try:
    from PIL import Image
    img = Image.open(OUT_PAPER)
    print(f"[OK] Dimensions: {img.size[0]}x{img.size[1]} px")
except ImportError:
    print("[INFO] PIL not available, skipping dimension check")
