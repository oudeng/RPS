# RPS Multi-Agent Benchmark

A reproducible benchmark for evaluating diverse learning algorithms in iterated Rock–Paper–Scissors, measuring population-dependent performance, non-transitive structure, and auditable regret certificates.

**Paper**: "Population-dependent agent performance in non-transitive games: a multi-agent Rock–Paper–Scissors benchmark" (under review)

## Features

- **18 agent archetypes** across 5 categories: rule-based baselines, probabilistic predictors, classical ML, neural sequence models (RNN/LSTM/Transformer), and actor–critic RL (A3C)
- **54-agent double round-robin** tournament with 500 rounds × 10 random seeds
- **Meta-game analysis**: pairwise payoff heatmaps, three-cycle enumeration, α-Rank evolutionary dynamics
- **Regret certificate diagnostics**: Lipschitz-type bounds linking prediction error to exploitability
- **Reproducible**: fixed seeds, CSV logging, command-line interface for all experiments

## Quick Start

### Requirements

```bash
conda create -n rps310 python=3.10
conda activate rps310
pip install torch numpy pandas scipy scikit-learn xgboost matplotlib seaborn adjustText
```

### Run the Core-54 benchmark

```bash
python RPS_main.py \
  --seats Agent_seats/paper_full_54.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --output-dir outputs/paper_full_54_r500_s10
```

### Run analysis

```bash
# Method-level statistics + Wilcoxon tests
python utility/analyze_2.py \
  --input-dir outputs/paper_full_54_r500_s10 \
  --output-dir outputs/paper_full_54_r500_s10/analysis_2

# Meta-game: heatmap + cycle detection + α-Rank
python utility/analyze_metagame.py \
  --input-dir outputs/paper_full_54_r500_s10 \
  --output-dir outputs/paper_full_54_r500_s10/metagame_analysis \
  --palette nature
```

## Agent Categories

| Tier | Category | Agents | Description |
|------|----------|--------|-------------|
| 1 | Rule-based | R, CG, WL | Memoryless / simple heuristics |
| 2 | Probabilistic | B_v1/v2, M_v1/v2 | Frequency tracking / Markov predictors |
| 3 | Classical ML | SVM, RF, XGB | Feature-engineered + online refit |
| 4 | Neural predictor | RNN_v2, LSTM_v1/v2, Tr_v1/v2, MSA_v2 | Sequence models with online SGD |
| 5 | Actor–critic RL | A3C_v1/v2 | Policy-gradient with online updates |

## Project Structure

```
code_RPS/
├── RPS_main.py                 # Tournament runner (entry point)
├── AI_RPS/                     # Agent implementations
├── Agent_seats/                # Roster definitions (CSV)
│   ├── paper_full_54.csv       # Core-54 main benchmark
│   ├── val_core19.csv          # Core-19 validation set
│   └── reviewer_simple7.csv    # 7-agent non-neural control
├── utility/
│   ├── analyze_1.py            # Stability + score evolution
│   ├── analyze_2.py            # Method-level stats + Wilcoxon tests
│   ├── analyze_metagame.py     # Heatmap + cycles + α-Rank
│   ├── benchmark_runtime_end2end.py  # CPU runtime benchmarking
│   ├── count_torch_params.py   # Trainable parameter counting
│   └── make_score_vs_params.py # Capacity–performance scatter plot
├── outputs/                    # Experiment outputs (not tracked in git)
├── README.md                   # This file
├── README_CL.md                # Full command-line manual
└── CHANGELOG.md                # Version history
```

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

| Script | Input | Key Outputs |
|--------|-------|-------------|
| `analyze_2.py` | Tournament output dir | `method_stats.csv`, `nonparam_wilcoxon_holm.csv` |
| `analyze_metagame.py` | Tournament output dir | `pairwise_heatmap.png`, `detected_3_cycles.csv`, `alpha_rank_distribution.csv` |
| `benchmark_runtime_end2end.py` | Agent names | `runtime_benchmark_summary.csv` |
| `count_torch_params.py` | — | `torch_param_counts.csv` |
| `make_score_vs_params.py` | method_stats + params | `score_vs_params.png` |

## Citation

Coming soon...

## License
MIT license.
