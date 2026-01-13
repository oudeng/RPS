# Reproduction Commands

This document provides a consolidated, copy-paste ready command reference for reproducing all experiments.

All commands assume execution from the repository root directory.

> **Important Notes**
> - `--seeds` requires a **comma-separated** string (e.g., `1,2,3,5`).
> - The tournament is **directed**: ordered pairs (A, B) and (B, A) are both played.
> - Hyperparameters in seat files (`hp_*` columns) are automatically injected into agent constructors.

---

## 0. Environment Setup

### Option A: Conda

```bash
conda env create -f env_setup/environment.yml
conda activate rps310
```

### Option B: pip + venv

```bash
bash env_setup/setup_pip.sh
source rps310/bin/activate
```

**Verification:**

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import numpy, pandas, scipy, sklearn, xgboost; print('deps ok')"
```

---

## 1. Smoke Test (Reviewer Quick Check)

A lightweight experiment (8 agents) for rapid sanity checking.

```bash
OUT=outputs/smoke_reviewer_8_r100_s2

python RPS_main.py \
  --seats Agent_seats/smoke_reviewer_8.csv \
  --rounds 100 \
  --seeds 1,2 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 100 \
  --output-dir ${OUT}

python utility/analyze_1.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_1

python utility/analyze_2.py \
  --input-dir ${OUT} \
  --out-dir ${OUT}/analysis_2 \
  --palette nature
```

**Key outputs:**
- `${OUT}/analysis_1/figure_1_score_distribution.png`
- `${OUT}/analysis_2/figures/` and `${OUT}/analysis_2/tables/`

---

## 2. Validation Set (Core-19)

Use for hyperparameter tuning, ablations, and sensitivity analyses.

```bash
OUT=outputs/val_core19_r500_s10

python RPS_main.py \
  --seats Agent_seats/val_core19.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 500 \
  --output-dir ${OUT}

python utility/analyze_1.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_1

python utility/analyze_2.py \
  --input-dir ${OUT} \
  --out-dir ${OUT}/analysis_2 \
  --palette nature
```

---

## 3. Transformer Hyperparameter Sweep

`Agent_seats/val_transformer_sweep.csv` includes multiple `Tr_v2` configurations with different `hp_*` hyperparameters. Each Transformer variant has distinct values for `hp_ctx_len`, `hp_d_model`, `hp_num_layers`, `hp_nhead`, `hp_dropout`, and `hp_lr`.

```bash
OUT=outputs/val_transformer_sweep_r500_s10

python RPS_main.py \
  --seats Agent_seats/val_transformer_sweep.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 500 \
  --output-dir ${OUT}

python utility/analyze_1.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_1

python utility/analyze_2.py \
  --input-dir ${OUT} \
  --out-dir ${OUT}/analysis_2 \
  --palette nature
```

**Selecting the optimal Transformer configuration:**

1. Open `${OUT}/analysis_2/tables/method_stats.csv` (or score tables in `analysis_1`).
2. Compare `Tr_v2` variants by their `idxname` (each row represents a distinct hyperparameter configuration).
3. Freeze the best-performing configuration and carry it forward to the final 54-agent benchmark.

---

## 4. Final Benchmark (Paper Main Result)

54-agent roster for the primary experimental results.

```bash
OUT=outputs/paper_full_54_r500_s10

python RPS_main.py \
  --seats Agent_seats/paper_full_54.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 500 \
  --output-dir ${OUT}

python utility/analyze_1.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_1

python utility/analyze_2.py \
  --input-dir ${OUT} \
  --out-dir ${OUT}/analysis_2 \
  --palette nature
```

---

## 5. Pairwise Lipschitz Analyses

These experiments use small rosters to avoid generating excessively large Lipschitz CSV files.

### 5.1 A3C_v2 vs RNN_v2

```bash
OUT=outputs/lip_A3CvsRNN_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_3_1_A3CvsRNN.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --history-k 20 \
  --warmup 50 \
  --debug \
  --output-dir ${OUT}

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature

python utility/analyze_Lipschitz_violations.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/lipschitz_violation_analysis
```

### 5.2 A3C_v2 vs LSTM_v2

```bash
OUT=outputs/lip_A3CvsLSTM_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_3_2_A3CvsLSTM.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --history-k 20 \
  --warmup 50 \
  --debug \
  --output-dir ${OUT}

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature

python utility/analyze_Lipschitz_violations.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/lipschitz_violation_analysis
```

### 5.3 A3C_v2 vs Random Baseline (R)

```bash
OUT=outputs/lip_A3CvsR_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_3_3_A3CvsR.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --history-k 20 \
  --warmup 50 \
  --debug \
  --output-dir ${OUT}

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature
```

### 5.4 RNN_v2 vs LSTM_v2 (Pretrained)

```bash
OUT=outputs/lip_RNNvsLSTM_pretrained_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_3_4_RNNvsLSTM.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --input-dir ${OUT_TRAIN:-outputs/train_core20_r500_s10} \
  --batch-size 64 \
  --batch-freq 64 \
  --history-k 20 \
  --warmup 50 \
  --debug \
  --output-dir ${OUT}

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature
```

---

## 6. Training Effect Experiments (Trained vs Untrained)

These experiments require pretrained models from a prior training run.

### 6.1 Generate Pretrained Models

```bash
OUT_TRAIN=outputs/train_core20_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_1_20agents.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 500 \
  --output-dir ${OUT_TRAIN}
```

### 6.2 A3C_v2 Trained vs A3C_v2un (Untrained)

```bash
OUT_TRAIN=outputs/train_core20_r500_s10
OUT=outputs/trained_vs_un_A3C

python RPS_main.py \
  --seats Agent_seats/test_4_1_A3C_v2_TrainedVsUn.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --input-dir "${OUT_TRAIN}" \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir "${OUT}"

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature

python utility/analyze_Lipschitz_violations.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/lipschitz_violation_analysis
```

### 6.3 RNN_v2 Trained vs RNN_v2un (Untrained)

```bash
OUT_TRAIN=outputs/train_core20_r500_s10
OUT=outputs/trained_vs_un_RNN

python RPS_main.py \
  --seats Agent_seats/test_4_2_RNN_v2_TrainedVsUn.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --input-dir ${OUT_TRAIN} \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir ${OUT}

python utility/analyze_Lipschitz_v2.py \
  --input-dir ${OUT} \
  --separate-matchups \
  --palette nature
```

### 6.4 A3C_v2un vs RNN_v2un (Both Untrained)

```bash
OUT_TRAIN=outputs/train_core20_r500_s10
OUT=outputs/untrained_A3CvsRNN

python RPS_main.py \
  --seats Agent_seats/test_4_3_A3Cv2unVsRNNv2un.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir ${OUT}
```

### 6.5 Four-Family Trained vs Untrained Comparison

Compares A3C, RNN, LSTM, and Transformer variants.

```bash
OUT_TRAIN=outputs/train_core20_r500_s10
OUT=outputs/trained_vs_un_pack4

python RPS_main.py \
  --seats Agent_seats/test_5_1_TrainedVsUn.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --input-dir ${OUT_TRAIN} \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir ${OUT}

python utility/analyze_1.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/analysis_1
```

---

## 7. Top Agents Over Simple Baselines

Evaluates pretrained agents against simple baseline strategies.

```bash
OUT_TRAIN=outputs/train_core20_r500_s10
OUT=outputs/overTopR_r500_s10

python RPS_main.py \
  --seats Agent_seats/test_2_overTopR.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --input-dir ${OUT_TRAIN} \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 500 \
  --output-dir ${OUT}
```

---

## 8. Lipschitz Tightness Analyses

**Script:** `utility/analyze_lipschitz_tightness.py`

```bash
# 1. A3C vs Random
lip_dir=outputs/lip_A3CvsR_r500_s10
python utility/analyze_lipschitz_tightness.py \
  --input-dir ${lip_dir}/lipschitz_analysis \
  --output-dir ${lip_dir}/lipschitz_tightness

# 2. RNN vs LSTM (pretrained)
lip_dir=outputs/lip_RNNvsLSTM_pretrained_r500_s10
python utility/analyze_lipschitz_tightness.py \
  --input-dir ${lip_dir}/lipschitz_analysis \
  --output-dir ${lip_dir}/lipschitz_tightness

# 3. A3C vs RNN (pretrained)
lip_dir=outputs/lip_A3CvsRNN_r500_s10
python utility/analyze_lipschitz_tightness.py \
  --input-dir ${lip_dir}/lipschitz_analysis \
  --output-dir ${lip_dir}/lipschitz_tightness
```

---

## 9. Supplementary Analyses

### 9.1 Lipschitz-K Sensitivity Analysis

Evaluates how the sliding-window length K affects Lipschitz diagnostic metrics.

**Script:** `utility/analyze_Lipschitz_K_sensitivity.py`

```bash
python utility/analyze_Lipschitz_K_sensitivity.py \
  --input-dir outputs/lip_A3CvsRNN_r500_s10 \
  --Ks 5,10,20,50
```

**Outputs:**
- `outputs/lip_A3CvsRNN_r500_s10/lipschitz_K_sensitivity/lipschitz_K_sensitivity_summary.csv`
- `outputs/lip_A3CvsRNN_r500_s10/lipschitz_K_sensitivity/lipschitz_K_sensitivity_spearman.png`

### 9.2 End-to-End Runtime Benchmark

Measures per-decision latency (ms/decision) for each agent type. Used to generate Table S8.

**Script:** `utility/benchmark_runtime_end2end.py`

```bash
python utility/benchmark_runtime_end2end.py \
  --agents RNN_v2,LSTM_v2,A3C_v2,Tr_v2,B_v1,M_v1 \
  --rounds 500 \
  --seeds 1,2,3 \
  --output-base outputs/runtime_benchmark
```

**Outputs:**
- `outputs/runtime_benchmark/runtime_benchmark_summary.csv`

> **Note:** For publication-quality timing results, run this benchmark on the same hardware used for the main experiments.

---

## 10. Non-transitive Meta-game Analysis

**Script:** `utility/analyze_metagame.py`

Generates:
- Pairwise payoff heatmap (Figure: `pairwise_heatmap.png`)
- 3-cycle detection and enumeration (Table: `detected_3_cycles.csv`)
- α-Rank stationary distribution (Figure: `alpha_rank_stationary.png`)
- Cross-pool rank correlation (Table: `rank_correlation.csv`)

### 10.1 Basic Analysis (Core-19)

```bash
OUT=outputs/val_core19_r500_s10

python utility/analyze_metagame.py \
  --input-dir ${OUT} \
  --output-dir ${OUT}/metagame_analysis \
  --alpha 0.1 \
  --palette nature
```

**Outputs:**
- `${OUT}/metagame_analysis/figures/pairwise_heatmap.png`
- `${OUT}/metagame_analysis/figures/alpha_rank_stationary.png`
- `${OUT}/metagame_analysis/tables/pairwise_payoff_matrix.csv`
- `${OUT}/metagame_analysis/tables/detected_3_cycles.csv`
- `${OUT}/metagame_analysis/tables/alpha_rank_distribution.csv`
- `${OUT}/metagame_analysis/tables/metagame_summary.csv`

### 10.2 Analysis with Cross-Pool Comparison

Compare rankings across multiple evaluation pools (Core-54, Core-19, Top-R, Pack4):

```bash
OUT_CORE54=outputs/paper_full_54_r500_s10
OUT_CORE19=outputs/val_core19_r500_s10
OUT_TOPR=outputs/overTopR_r500_s10
OUT_PACK4=outputs/trained_vs_un_pack4

# Analyze Core-19 with comparisons to other pools
python utility/analyze_metagame.py \
  --input-dir ${OUT_CORE19} \
  --output-dir ${OUT_CORE19}/metagame_analysis \
  --alpha 0.1 \
  --compare-dirs "Core54:${OUT_CORE54},TopR:${OUT_TOPR},Pack4:${OUT_PACK4}"

# Analyze Core-54 (full benchmark)
python utility/analyze_metagame.py \
  --input-dir ${OUT_CORE54} \
  --output-dir ${OUT_CORE54}/metagame_analysis \
  --alpha 0.1 \
  --compare-dirs "Core19:${OUT_CORE19},TopR:${OUT_TOPR}"
```

### 10.3 Copy Figures for Paper

After running the analysis, copy the generated figures to the paper's `Fig/` directory:

```bash
# For Core-19 analysis (main paper figures)
mkdir -p Fig/RPS_metagame
cp outputs/val_core19_r500_s10/metagame_analysis/figures/pairwise_heatmap.png \
   Fig/RPS_metagame/pairwise_heatmap_core19.png
cp outputs/val_core19_r500_s10/metagame_analysis/figures/alpha_rank_stationary.png \
   Fig/RPS_metagame/alpharank_stationary_core19.png
```

### 10.4 Parameter Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.1 | Selection intensity for α-Rank (larger = more deterministic) |
| `--min-cycle-strength` | 0.0 | Minimum G(i,j) value to count as dominance edge |
| `--palette` | nature | Color palette: `nature`, `science`, or `cell` |
| `--compare-dirs` | None | Cross-pool comparison directories (format: `name1:path1,name2:path2`) |

### 10.5 Interpreting Outputs

**Pairwise Heatmap:**
- Entry (i,j) shows mean payoff G(i,j) of agent i against agent j
- Red = positive (i dominates j), Blue = negative (j dominates i)
- Highlighted rectangles mark edges of the strongest detected 3-cycle

**3-Cycle Table:**
- Lists all cycles A → B → C → A where G(A,B) > 0, G(B,C) > 0, G(C,A) > 0
- Sorted by minimum edge strength (strongest cycles first)
- Use top cycle for paper placeholder: `[PLACEHOLDER: INSERT IDENTIFIED CYCLE]`

**α-Rank Distribution:**
- Shows evolutionary stable state under selection pressure α
- Strategies with mass > 1/n survive under evolutionary dynamics
- Multiple strategies with positive mass = mixed ESS (non-transitive structure)

**Rank Correlation:**
- Spearman ρ and Kendall τ between evaluation pools
- Low correlation confirms population-dependent evaluation
