# RPS: Multi-Agent Rock–Paper–Scissors Benchmark

A reproducible benchmark for evaluating learning algorithms in iterated Rock–Paper–Scissors, a canonical non-transitive game. Accompanies the paper:

> **Population-dependent agent performance in non-transitive games: a multi-agent Rock–Paper–Scissors benchmark**
> Ou Deng, Jianting Xu, Shoji Nishimura, Atsushi Ogihara, Qun Jin
> *Scientific Reports* (under review)

## Features

- **18 method families** spanning random baselines, probabilistic predictors, classical ML, deep sequence models, and reinforcement learning
- **Multi-seed tournament protocol** with double round-robin pairwise matchups
- **Three roster configurations**: Core-54 (main benchmark, 54 agents), Core-18 (validation, 18 agents), Simple7 (non-neural control, 7 agents)
- **Analysis suite**: score evolution, method-level statistics, Wilcoxon tests, pairwise heatmaps, cycle detection, α-Rank
- **Auditable regret certificates** via per-round Lipschitz-bounded prediction error

## Quick Start

### Prerequisites

```bash
# Python 3.10+ with PyTorch (CPU-only is sufficient)
conda create -n rps310 python=3.10
conda activate rps310
pip install torch numpy pandas scipy scikit-learn xgboost matplotlib seaborn adjustText
```

### Run a Tournament

```bash
# Core-54 main benchmark (10 seeds × 500 rounds)
python RPS_main.py \
    --rounds 500 \
    --seeds 1,2,3,5,8,13,21,34,55,89 \
    --seats Agent_seats/paper_full_54.csv \
    --output-dir outputs/paper_full_54_r500_s10
```

### Run Analysis

```bash
# Score evolution + stability
python utility/analyze_1.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --output-dir outputs/paper_full_54_r500_s10/Analysis_1

# Method-level statistics + Wilcoxon tests
# NOTE: analyze_2.py uses --out-dir (not --output-dir)
python utility/analyze_2.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --out-dir outputs/paper_full_54_r500_s10/Analysis_2

# Metagame: heatmap + cycle detection + α-Rank
python utility/analyze_metagame.py \
    --input-dir outputs/paper_full_54_r500_s10 \
    --output-dir outputs/paper_full_54_r500_s10/Analysis_metagame
```

## Project Structure

```
code_RPS/
├── RPS_main.py             # Tournament runner (entry point)
├── AI_RPS/                 # Agent implementations
│   ├── R.py, CG.py, WL.py           # Baselines
│   ├── B_v1.py, B_v2.py, ...        # Probabilistic predictors
│   ├── SVM.py, RF.py, XGB.py        # Classical ML
│   ├── RNN_v2.py, LSTM_v1.py, ...   # Deep sequence models
│   └── A3C_v1.py, A3C_v2.py         # Reinforcement learning
├── Agent_seats/            # Roster CSV files
│   ├── paper_full_54.csv   # Core-54 main benchmark
│   ├── val_core18.csv      # Core-18 validation
│   └── reviewer_simple7.csv # Simple7 non-neural control
├── utility/                # Analysis scripts
├── outputs/                # Experiment outputs (not tracked in git)
├── README.md               # This file
├── README_CL.md            # Full command-line reference
└── CHANGELOG.md            # Version history
```

## Agent Families

| Category | Methods | Count |
|----------|---------|-------|
| Baseline | R (random), CG (counter-guesser), WL (win-stay-lose-shift) | 3 |
| Probabilistic | B_v1, B_v2 (Dirichlet), M_v1, M_v2 (Markov), MSA_v2 (multi-scale) | 5 |
| Classical ML | SVM, RF (random forest), XGB (gradient boosting) | 3 |
| Deep neural network | RNN_v2, LSTM_v1, LSTM_v2, Tr_v1, Tr_v2 | 5 |
| Reinforcement learning | A3C_v1, A3C_v2 | 2 |

## Roster Configurations

| Roster | File | Agents | Purpose |
|--------|------|--------|---------|
| Core-54 | `paper_full_54.csv` | 18 families × 3 seats = 54 | Main benchmark |
| Core-18 | `val_core18.csv` | 18 families × 1 seat = 18 | Validation / hyperparameter tuning |
| Simple7 | `reviewer_simple7.csv` | 7 non-neural agents | Control experiment |

## Key CLI Arguments (RPS_main.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--seats` | required | Path to agent roster CSV |
| `--rounds` | 500 | Rounds per matchup |
| `--seeds` | `1,2,3,...` | Comma-separated seed list |
| `--batch-size` | 64 | Mini-batch size for batch-capable agents |
| `--batch-freq` | 64 | Update frequency for batch-capable agents |
| `--warmup` | 50 | Warmup rounds before logging |
| `--output-dir` | required | Output directory |

## Analysis Scripts

| Script | Key arguments | Key Outputs |
|--------|---------------|-------------|
| `analyze_1.py` | `--input-dir`, `--output-dir` | Score evolution, stability scatter |
| `analyze_2.py` | `--input-dir`, **`--out-dir`** | `method_stats.csv`, `nonparam_wilcoxon_holm.csv` |
| `analyze_metagame.py` | `--input-dir`, `--output-dir` | `pairwise_heatmap.png`, `detected_3_cycles.csv`, `alpha_rank_distribution.csv` |
| `benchmark_runtime_end2end.py` | `--agents`, `--output-base` | `runtime_benchmark_summary.csv` |
| `count_torch_params.py` | (no args) | `torch_param_counts.csv` |
| `make_score_vs_params.py` | (no args) | `score_vs_params.png` |

## Citation

```bibtex
@article{Deng2026RPS,
	title = {Population-dependent agent performance in non-transitive games: a multi-agent rock--paper--scissors benchmark},
	author = {Deng, Ou and Xu, Jianting and Nishimura, Shoji and Ogihara, Atsushi and Jin, Qun},
	journal = {Scientific Reports},
	doi = {10.1038/s41598-026-55417-9},
	url = {https://doi.org/10.1038/s41598-026-55417-9},
	year = {2026}
}
```

## License

MIT License.

