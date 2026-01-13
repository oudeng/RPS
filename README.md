# RPS-Benchmark: Multi-Agent Rock–Paper–Scissors Tournament Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive **multi-seed, multi-agent Rock–Paper–Scissors (RPS) tournament runner** with **Lipschitz-style audit metrics** for evaluating prediction–outcome alignment in adaptive agents.

## Overview

This framework enables rigorous empirical evaluation of learning agents in repeated games through:

- **Multi-seed tournaments**: Directed round-robin competitions across multiple random seeds for statistical robustness
- **Seat-file protocol**: Clean separation between validation (hyperparameter selection) and final benchmarking
- **Hyperparameter injection**: Agent-specific parameters via `hp_*` columns in seat files
- **Audit metrics**: Lipschitz-style diagnostics quantifying the relationship between prediction error and regret

---

## Features

### Tournament System
- **Directed round-robin**: For each ordered pair (A, B), agent A plays against B for a configurable number of rounds
- **Payoff convention**: Standard RPS payoff with win (+1), loss (−1), tie (0)
- **Total games per seed**: N × (N−1) × rounds (directed pairing)

### Agent Library
The framework includes implementations of diverse agent families:

| Category | Agents |
|----------|--------|
| Neural | A3C, RNN, LSTM, Transformer, MSA |
| Machine Learning | Random Forest, SVM, XGBoost |
| Rule-based | Win-Lose, Markov, Bayes, Copy-Guess |
| Baseline | Random |

### Audit Metrics
For eligible agents, the system records:
- Empirical opponent distribution `p_true` (windowed or one-hot)
- Predicted distribution `p_pred` (from policy logits / counts / ML probabilities)
- L1 distance `‖p_true − p_pred‖₁`
- RPS regret proxy against `p_true`

---

## Repository Structure

```
.
├── RPS_main.py              # Main tournament runner
├── AI_RPS/                  # Agent implementations
│   ├── A3C_v2.py           # Actor-Critic agent
│   ├── LSTM_v2.py          # LSTM-based agent
│   ├── RNN_v2.py           # RNN-based agent
│   ├── Tr_v2.py            # Transformer agent
│   ├── RF.py               # Random Forest agent
│   └── ...                 # Additional agents
├── Agent_seats/             # Experiment configurations (CSV)
│   ├── paper_full_54.csv   # Main benchmark (54 agents)
│   ├── val_core19.csv      # Validation set (19 agents)
│   ├── val_transformer_sweep.csv  # Hyperparameter sweep
│   └── ...                 # Additional configurations
├── utility/                 # Analysis and visualization scripts
│   ├── analyze_1.py        # Primary analysis
│   ├── analyze_2.py        # Extended analysis
│   ├── analyze_Lipschitz_v2.py
│   ├── analyze_Lipschitz_violations.py
│   ├── analyze_Lipschitz_K_sensitivity.py
│   ├── analyze_lipschitz_tightness.py
│   ├── analyze_metagame.py
│   └── benchmark_runtime_end2end.py
├── env_setup/               # Environment setup files
│   ├── environment.yml     # Conda environment
│   ├── requirements.txt    # pip requirements
│   └── README_SETUP.md     # Detailed setup guide
├── README.md                # This file
└── README_CL.md             # Command reference
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional, recommended for neural agents)

### Option A: Conda (Recommended)

```bash
conda env create -f env_setup/environment.yml
conda activate rps310
```

### Option B: pip + venv

```bash
bash env_setup/setup_pip.sh
source rps310/bin/activate
```

### Verification

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import numpy, pandas, scipy, sklearn, xgboost; print('deps ok')"
```

See `env_setup/README_SETUP.md` for detailed setup instructions.

---

## Quick Start

Run a smoke test with 8 agents to verify installation:

```bash
# Run tournament
python RPS_main.py \
  --seats Agent_seats/smoke_reviewer_8.csv \
  --rounds 100 \
  --seeds 1,2 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 100 \
  --output-dir outputs/smoke_test

# Generate analysis
python utility/analyze_1.py \
  --input-dir outputs/smoke_test \
  --output-dir outputs/smoke_test/analysis_1

python utility/analyze_2.py \
  --input-dir outputs/smoke_test \
  --out-dir outputs/smoke_test/analysis_2 \
  --palette nature
```

---

## Usage

### Command-Line Interface

```bash
python RPS_main.py [OPTIONS]
```

**Core Arguments:**

| Argument | Description |
|----------|-------------|
| `--seats <csv>` | Roster definition file (required) |
| `--rounds <int>` | Interactions per ordered pair |
| `--seeds <str>` | Comma-separated random seeds (e.g., `1,2,3,5,8`) |
| `--output-dir <path>` | Output directory for logs and models |
| `--input-dir <path>` | Directory for loading pretrained models |

**Audit (Lipschitz) Arguments:**

| Argument | Description |
|----------|-------------|
| `--history-k <int>` | Window size for empirical distribution (default: 20) |
| `--use-onehot` | Use one-hot instead of windowed distribution |
| `--warmup <int>` | Rounds excluded from Lipschitz logging |
| `--debug` | Enable verbose diagnostics |

**Neural Training Arguments:**

| Argument | Description |
|----------|-------------|
| `--batch-size <int>` | Batch size for neural agents |
| `--batch-freq <int>` | Update frequency metadata |

### Seat File Format

Seat files (CSV) define tournament configurations:

| Column | Required | Description |
|--------|:--------:|-------------|
| `seat` | ✓ | Unique integer seat ID |
| `agent` | ✓ | Agent key (must exist in `AGENT_MODULES`) |
| `idxname` | ○ | Unique name for logs and model files |
| `seed` |   | Load pretrained model from specified seed |
| `hp_*` |   | **Hyperparameters forwarded to agent constructor** |

**Important:** The `hp_*` columns allow per-agent hyperparameter injection. Any column prefixed with `hp_` will have its value parsed and passed to the agent's `__init__()` method. This enables:
- Architecture sweeps (e.g., different `d_model`, `num_layers` for Transformers)
- Learning rate tuning per agent
- Reproducible configuration without code changes

**Example (Transformer with custom hyperparameters):**

```csv
seat,agent,idxname,hp_ctx_len,hp_d_model,hp_num_layers,hp_nhead,hp_dropout,hp_lr
11,Tr_v2,11_Tr_v2_ctx32_d64_L3,32,64,3,4,0.05,0.001
12,Tr_v2,12_Tr_v2_ctx64_d128_L4,64,128,4,8,0.05,0.0005
```

In this example:
- Agent at seat 11 uses `ctx_len=32, d_model=64, num_layers=3, nhead=4, dropout=0.05, lr=0.001`
- Agent at seat 12 uses `ctx_len=64, d_model=128, num_layers=4, nhead=8, dropout=0.05, lr=0.0005`

---

## Experiments

### Recommended Workflow

1. **Validation**: Tune hyperparameters on `val_core19.csv` (19 agents)
2. **Sweep**: Compare configurations using `val_transformer_sweep.csv`
3. **Benchmark**: Evaluate on `paper_full_54.csv` (54 agents) with frozen hyperparameters

### Experiment Catalog

| Experiment | Seat File | Purpose |
|------------|-----------|---------|
| Smoke Test | `smoke_reviewer_8.csv` | Quick sanity check |
| Validation | `val_core19.csv` | Hyperparameter tuning |
| Transformer Sweep | `val_transformer_sweep.csv` | Architecture comparison |
| Main Benchmark | `paper_full_54.csv` | Paper results |
| Pairwise Lipschitz | `test_3_*.csv` | Detailed diagnostics |
| Training Effect | `test_4_*.csv`, `test_5_*.csv` | Trained vs untrained |

### Lipschitz-K Sensitivity Analysis

Evaluate how the sliding-window length K affects diagnostic metrics:

```bash
python utility/analyze_Lipschitz_K_sensitivity.py \
  --input-dir outputs/lip_A3CvsRNN_r500_s10 \
  --Ks 5,10,20,50
```

**Outputs:**
- `lipschitz_K_sensitivity_summary.csv`: Per-matchup statistics for each K
- `lipschitz_K_sensitivity_spearman.png`: Correlation visualization

### Runtime Benchmark

Measure per-decision latency for each agent type:

```bash
python utility/benchmark_runtime_end2end.py \
  --agents RNN_v2,LSTM_v2,A3C_v2,Tr_v2,B_v1,M_v1 \
  --rounds 500 \
  --seeds 1,2,3 \
  --output-base outputs/runtime_benchmark
```

**Output:** `runtime_benchmark_summary.csv` with ms/decision metrics.

---

## Outputs

Running `RPS_main.py` produces:

```
<output-dir>/
├── RPS_record_seed<k>.csv              # Per-interaction records
├── RPS_train_summary_seed<k>.csv       # Per-agent summaries
├── models_seed<k>/                     # Saved model states
├── experiment_metadata.json            # Run configuration
└── lipschitz_analysis/                 # (if enabled)
    └── lipschitz_seed<k>.csv
```

Analysis scripts generate figures and tables under their respective output directories.

---

## Performance Notes

- **Games per seed**: N × (N−1) × rounds
  - Example: 54 agents × 500 rounds = 1,431,000 games/seed
- **Lipschitz logging**: Set `--warmup >= --rounds` to disable for large rosters
- **GPU acceleration**: Recommended for neural agents (A3C, RNN, LSTM, Transformer)

---

## Full Command Reference

See **[README_CL.md](README_CL.md)** for complete, copy-paste ready commands covering all experiments.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rps_benchmark,
  title = {RPS-Benchmark: Multi-Agent Rock-Paper-Scissors Tournament Framework},
  year = {2025},
  url = {https://github.com/your-repo/rps-benchmark}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
