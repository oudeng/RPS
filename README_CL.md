# Command-Line Reference for Experiment Reproduction

This document provides exact commands to reproduce all experiments reported in the paper.

## Prerequisites

```bash
# Create and activate conda environment
conda create -n rps310 python=3.10
conda activate rps310
pip install torch numpy pandas scipy scikit-learn xgboost matplotlib seaborn adjustText
```

---

## 1. Core-54 Main Benchmark (Table 1, Table 2)

```bash
# Tournament
python RPS_main.py \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --seats Agent_seats/paper_full_54.csv \
    --output-dir outputs/paper_full_54_r500_s10

# Analysis: score evolution + stability
python utility/analyze_1.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --output-dir outputs/paper_full_54_r500_s10/Analysis_1

# Analysis: method-level statistics + Wilcoxon tests
# NOTE: analyze_2.py uses --out-dir (not --output-dir)
python utility/analyze_2.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --out-dir outputs/paper_full_54_r500_s10/Analysis_2

# Analysis: metagame (heatmap + cycle detection + α-Rank)
python utility/analyze_metagame.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --output-dir outputs/paper_full_54_r500_s10/Analysis_metagame
```

**Key outputs**:
- `Analysis_2/tables/method_stats.csv` → Table 2 source data
- `Analysis_metagame/tables/detected_3_cycles.csv` → Three-cycle enumeration
- `Analysis_metagame/figures/pairwise_heatmap.png` → Pairwise payoff heatmap

---

## 2. Core-18 Validation (Table S2, Table S3)

```bash
# Tournament
python RPS_main.py \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --seats Agent_seats/val_core18.csv \
    --output-dir outputs/val_core18_r500_s10

# Analysis
python utility/analyze_1.py \
    --input-dir outputs/val_core18_r500_s10 \
    --output-dir outputs/val_core18_r500_s10/Analysis_1

python utility/analyze_2.py \
    --input-dir outputs/val_core18_r500_s10 \
    --out-dir outputs/val_core18_r500_s10/Analysis_2

python utility/analyze_metagame.py \
    --input-dir outputs/val_core18_r500_s10 \
    --output-dir outputs/val_core18_r500_s10/Analysis_metagame
```

**Key outputs**:
- `Analysis_2/tables/method_stats.csv` → Table S3 source data
- `Analysis_metagame/tables/metagame_summary.csv` → 134 cycles, Frobenius norm
- `Analysis_metagame/tables/alpha_rank_distribution.csv` → α-Rank stationary distribution

---

## 3. Simple7 Non-Neural Control (Section S13, Table S14, Figure S9)

```bash
# Tournament
python RPS_main.py \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --seats Agent_seats/reviewer_simple7.csv \
    --output-dir outputs/reviewer_simple7_r500_s10

# Analysis: metagame
python utility/analyze_metagame.py \
    --input-dir outputs/reviewer_simple7_r500_s10 \
    --output-dir outputs/reviewer_simple7_r500_s10/Analysis_metagame
```

**Key outputs**:
- `Analysis_metagame/figures/pairwise_heatmap.png` → Figure S9
- `Analysis_metagame/tables/detected_3_cycles.csv` → 7 three-cycles (Table S14)

---

## 4. Runtime Benchmark (Table S12)

```bash
# Full 18-method benchmark (run in groups for efficiency)
# Group A: fast agents
python utility/benchmark_runtime_end2end.py \
    --agents R,CG,WL,B_v2,M_v2 \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --output-base outputs/runtime_supp_groupA

# Group B: classical ML
python utility/benchmark_runtime_end2end.py \
    --agents SVM,RF,XGB \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --output-base outputs/runtime_supp_groupB

# Group C: remaining neural + probabilistic
python utility/benchmark_runtime_end2end.py \
    --agents B_v1,M_v1,MSA_v2,RNN_v2,LSTM_v1,LSTM_v2,Tr_v1,Tr_v2,A3C_v1,A3C_v2 \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --output-base outputs/runtime_supp_groupC
```

Merge group outputs into `outputs/runtime_full18/runtime_benchmark_summary.csv`.

---

## 5. Neural Parameter Counts (Table S13)

```bash
# No CLI arguments needed — runs with hardcoded agent list
python utility/count_torch_params.py
# Output: outputs/param_counts/torch_param_counts.csv
```

---

## 6. Score vs Parameters Plot (Figure S8)

```bash
# No CLI arguments needed — uses hardcoded paths
python utility/make_score_vs_params.py
# Output: paper_RPS/Fig/RPS_sophistication/score_vs_params.png
```

---

## Notes

- **`analyze_2.py` uses `--out-dir`** (not `--output-dir`); all other analysis scripts use `--output-dir`
- `benchmark_runtime_end2end.py` uses `--output-base` (not `--output-dir`)
- `count_torch_params.py` and `make_score_vs_params.py` have no CLI arguments
- All tournaments use 10 seeds (Fibonacci-inspired: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89) and 500 rounds
- Neural agents run in CPU mode for reproducible runtime benchmarks
- Experiment outputs are large and not tracked in git

---

## Seed Convention

All experiments use the same 10 seeds: `1, 2, 3, 5, 8, 13, 21, 34, 55, 89` (first 10 Fibonacci numbers).
