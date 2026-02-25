#!/usr/bin/env python3
"""
Analyze sensitivity of Lipschitz diagnostics to the sliding-window length K used to estimate p_t.

This script recomputes p_t (empirical opponent action distribution) from the raw match records
(RPS_record_seed*.csv) for multiple window lengths K, while keeping the agent's predicted distribution
\hat p_t from lipschitz_analysis/lipschitz_seed*.csv fixed. It then recomputes:
  - L1 distance: ||p_t - \hat p_t||_1
  - instantaneous regret: Δ_t = max_a u(a, p_t) - u(â_t, p_t), where â_t is the best response to \hat p_t
  - correlation metrics and a linear regression slope on non-zero pairs
  - bound violation rate for Δ_t <= 2 ||p_t - \hat p_t||_1 (up to numerical tolerance)

Expected input directory structure (from RPS_main.py --lipschitz-analysis):
  <input_dir>/
    RPS_record_seed1.csv
    RPS_record_seed2.csv
    ...
    lipschitz_analysis/
      lipschitz_seed1.csv
      lipschitz_seed2.csv
      ...

Outputs:
  <out_dir>/
    lipschitz_K_sensitivity_summary.csv
    lipschitz_K_sensitivity_spearman.png
"""

from __future__ import annotations

import argparse
import os
import glob
import io
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

CHOICE_TO_INT = {
    "Rock": 0, "Paper": 1, "Scissors": 2,
    "R": 0, "P": 1, "S": 2,
    0: 0, 1: 1, 2: 2,
}

def payoff(a: int, b: int) -> int:
    """RPS payoff in {-1,0,1} for action a vs opponent action b."""
    if a == b:
        return 0
    if (a == 0 and b == 2) or (a == 1 and b == 0) or (a == 2 and b == 1):
        return 1
    return -1

def expected_payoff(a: int, p_true: np.ndarray) -> float:
    return float(sum(payoff(a, b) * p_true[b] for b in range(3)))

def best_response_payoff(p_true: np.ndarray) -> float:
    return max(expected_payoff(a, p_true) for a in range(3))

def regret_for_action(a_hat: int, p_true: np.ndarray) -> float:
    return best_response_payoff(p_true) - expected_payoff(a_hat, p_true)

def l1_distance(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.abs(p_true - p_pred).sum())

def compute_p_true(history: List[int], K: int) -> np.ndarray:
    """Empirical distribution over last K opponent actions."""
    window = history[-K:]
    counts = np.bincount(window, minlength=3).astype(float)
    return counts / float(K)

def parse_ks(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _seed_from_filename(path: str) -> int:
    import re
    m = re.search(r"seed(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse seed from filename: {path}")
    return int(m.group(1))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Experiment output dir containing RPS_record_seed*.csv and lipschitz_analysis/")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: <input-dir>/lipschitz_K_sensitivity)")
    ap.add_argument("--Ks", default="5,10,20,50", help="Comma-separated window lengths K (default: 5,10,20,50)")
    ap.add_argument("--tol", type=float, default=1e-12, help="Numerical tolerance for bound violations (default: 1e-12)")
    args = ap.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir or os.path.join(input_dir, "lipschitz_K_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    Ks = parse_ks(args.Ks)
    Ks = sorted(set(Ks))

    record_files = sorted(glob.glob(os.path.join(input_dir, "RPS_record_seed*.csv")))
    lip_files = sorted(glob.glob(os.path.join(input_dir, "lipschitz_analysis", "lipschitz_seed*.csv")))

    if not record_files:
        raise FileNotFoundError(f"No record files found in {input_dir} (expected RPS_record_seed*.csv)")
    if not lip_files:
        raise FileNotFoundError(f"No lipschitz files found in {os.path.join(input_dir,'lipschitz_analysis')}")

    seed_to_record = { _seed_from_filename(p): p for p in record_files }
    seed_to_lip = { _seed_from_filename(p): p for p in lip_files }
    seeds = sorted(set(seed_to_record).intersection(seed_to_lip))
    if not seeds:
        raise RuntimeError("No overlapping seeds between record and lipschitz files.")

    all_rows: List[Dict] = []

    t0 = time.time()
    for seed in seeds:
        df_rec = _read_csv(seed_to_record[seed])
        df_lip = _read_csv(seed_to_lip[seed])

        # Identify matchups present in the lipschitz file
        matchups = df_lip[["who_agent", "whom_agent"]].drop_duplicates().values.tolist()

        # Build a lookup for directed records per matchup: opponent action sequence by round
        # For a directed game row, opponent action is 'whom_choice'
        for who_agent, whom_agent in matchups:
            df_dir = df_rec[(df_rec["who_agent"] == who_agent) & (df_rec["whom_agent"] == whom_agent)].copy()
            if df_dir.empty:
                continue
            df_dir = df_dir.sort_values("round")

            opp_actions = [CHOICE_TO_INT.get(x, None) for x in df_dir["whom_choice"].tolist()]
            if any(v is None for v in opp_actions):
                bad = [x for x in df_dir["whom_choice"].unique().tolist() if CHOICE_TO_INT.get(x, None) is None]
                raise ValueError(f"Unknown action tokens in record file: {bad}")

            df_lm = df_lip[(df_lip["who_agent"] == who_agent) & (df_lip["whom_agent"] == whom_agent)].copy()
            if df_lm.empty:
                continue
            df_lm = df_lm.sort_values("round")

            rounds = df_lm["round"].astype(int).to_numpy()
            p_pred = df_lm[["p_pred_rock", "p_pred_paper", "p_pred_scissors"]].to_numpy(dtype=float)
            a_br = df_lm["action_br_pred"].astype(int).to_numpy()

            for i, t_round in enumerate(rounds):
                hist = opp_actions[: t_round - 1]  # rounds 1..t-1
                for K in Ks:
                    if len(hist) < K:
                        continue
                    p_true = compute_p_true(hist, K)
                    l1 = l1_distance(p_true, p_pred[i])
                    reg = regret_for_action(int(a_br[i]), p_true)
                    all_rows.append({
                        "seed": seed,
                        "who_agent": who_agent,
                        "whom_agent": whom_agent,
                        "round": int(t_round),
                        "K": int(K),
                        "l1": float(l1),
                        "regret": float(reg),
                    })

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No rows were produced. Check that record files and lipschitz files contain matching matchups.")

    # Aggregate summary per matchup and K
    summary_rows = []
    for (who, whom, K), g in df.groupby(["who_agent", "whom_agent", "K"]):
        l1 = g["l1"].to_numpy()
        reg = g["regret"].to_numpy()
        mask = (l1 > 0) & (reg > 0)

        if mask.sum() >= 3:
            rho = float(spearmanr(l1[mask], reg[mask]).correlation)
            r = float(pearsonr(l1[mask], reg[mask]).statistic)
            X = l1[mask].reshape(-1, 1)
            y = reg[mask]
            lr = LinearRegression()
            lr.fit(X, y)
            slope = float(lr.coef_[0])
            intercept = float(lr.intercept_)
            r2 = float(lr.score(X, y))
        else:
            rho = float("nan")
            r = float("nan")
            slope = float("nan")
            intercept = float("nan")
            r2 = float("nan")

        viol = float((reg > 2.0 * l1 + args.tol).mean())

        summary_rows.append({
            "who_agent": who,
            "whom_agent": whom,
            "K": int(K),
            "n_samples": int(len(g)),
            "n_nonzero": int(mask.sum()),
            "l1_mean": float(np.mean(l1)),
            "l1_std": float(np.std(l1, ddof=1)),
            "regret_mean": float(np.mean(reg)),
            "regret_std": float(np.std(reg, ddof=1)),
            "spearman_rho": rho,
            "pearson_r": r,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "violation_rate": viol,
        })

    summary = pd.DataFrame(summary_rows).sort_values(["who_agent", "whom_agent", "K"])
    summary_csv = os.path.join(out_dir, "lipschitz_K_sensitivity_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot: Spearman rho vs K for each matchup
    import matplotlib.pyplot as plt

    summary_plot = summary.copy()
    summary_plot["matchup"] = summary_plot["who_agent"] + "→" + summary_plot["whom_agent"]

    plt.figure(figsize=(7, 4))
    for matchup, gg in summary_plot.groupby("matchup"):
        gg = gg.sort_values("K")
        plt.plot(gg["K"], gg["spearman_rho"], marker="o", label=matchup)
    plt.xlabel("Sliding window length K (for $p_t$)")
    plt.ylabel("Spearman correlation between $\\|p_t-\\hat p_t\\|_1$ and $\\Delta_t$")
    plt.xticks(sorted(summary_plot["K"].unique().tolist()))
    plt.ylim(-0.05, 1.0)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "lipschitz_K_sensitivity_spearman.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    dt = time.time() - t0
    print(f"[OK] Wrote: {summary_csv}")
    print(f"[OK] Wrote: {fig_path}")
    print(f"[Done] Processed {len(seeds)} seeds, {len(df)} samples in {dt:.1f}s.")

if __name__ == "__main__":
    main()
