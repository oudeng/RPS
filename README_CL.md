# Experiment Command-Line Manual

This document contains the exact commands used to generate all experimental results in the paper. All commands assume:

```bash
cd /home/dengou/RPS/code_RPS
source /home/dengou/anaconda3/etc/profile.d/conda.sh
conda activate rps310
```

---

## 1. Core-54 Main Benchmark (Table 1–2, Figures 1–3)

```bash
python RPS_main.py \
  --seats Agent_seats/paper_full_54.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir outputs/paper_full_54_r500_s10
```

### Analysis pipeline

```bash
# Method-level statistics
python utility/analyze_2.py \
  --input-dir outputs/paper_full_54_r500_s10 \
  --output-dir outputs/paper_full_54_r500_s10/analysis_2

# Meta-game structure (heatmap, cycles, α-Rank)
python utility/analyze_metagame.py \
  --input-dir outputs/paper_full_54_r500_s10 \
  --output-dir outputs/paper_full_54_r500_s10/metagame_analysis \
  --palette nature
```

**Key outputs**:
- `analysis_2/tables/method_stats.csv` → Table 2 source data
- `metagame_analysis/tables/detected_3_cycles.csv` → 177 three-cycles
- `metagame_analysis/figures/pairwise_heatmap.png` → Core-19 heatmap

---

## 2. Core-19 Validation Tournament (Table S3, S4)

```bash
python RPS_main.py \
  --seats Agent_seats/val_core19.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --output-dir outputs/val_core19_r500_s10
```

---

## 3. Runtime Benchmark (Table S12)

Representative 6 agents:
```bash
python utility/benchmark_runtime_end2end.py \
  --agents RNN_v2,LSTM_v2,A3C_v2,Tr_v2,B_v1,M_v1 \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --output-base outputs/runtime_benchmark_s10
```

Full 18-method benchmark (supplementary, added in v5.10):
```bash
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

# Group C: remaining neural
python utility/benchmark_runtime_end2end.py \
  --agents MSA_v2,LSTM_v1,Tr_v1,A3C_v1 \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --output-base outputs/runtime_supp_groupC
```

Merge into unified CSV:
```python
# See utility/make_compute_tradeoff.py for merge logic
# Output: outputs/runtime_full18/runtime_benchmark_summary.csv
```

---

## 4. Neural Parameter Counting (Table S13)

```bash
python utility/count_torch_params.py
# Output: outputs/param_counts/torch_param_counts.csv
```

---

## 5. Capacity–Performance Scatter Plot (Figure S8)

```bash
python utility/make_score_vs_params.py
# Output: paper_RPS/Fig/RPS_sophistication/score_vs_params.png
```

---

## 6. Simple7 Non-Neural Control (Section S13, Figure S9, Table S15)

```bash
# Run tournament
python RPS_main.py \
  --seats Agent_seats/reviewer_simple7.csv \
  --rounds 500 \
  --seeds 1,2,3,5,8,13,21,34,55,89 \
  --batch-size 64 \
  --batch-freq 64 \
  --warmup 50 \
  --output-dir outputs/reviewer_simple7_r500_s10

# Analyze meta-game
python utility/analyze_metagame.py \
  --input-dir outputs/reviewer_simple7_r500_s10 \
  --output-dir outputs/reviewer_simple7_r500_s10/metagame_analysis \
  --palette nature
```

**Key outputs**:
- `metagame_analysis/figures/pairwise_heatmap.png` → Figure S9
- `metagame_analysis/tables/detected_3_cycles.csv` → 7 three-cycles (Table S15)

---

## 7. LaTeX Compilation

```bash
cd /home/dengou/RPS/paper_RPS
latexmk -pdf RPS_v5_10.tex
latexmk -pdf RPS_SI_v5_10.tex
```

---

## Seed Convention

All experiments use the same 10 seeds: `1, 2, 3, 5, 8, 13, 21, 34, 55, 89` (first 10 Fibonacci numbers).
