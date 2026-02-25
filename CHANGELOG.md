# Changelog

## v5.10 (2025-02 — current revision)

### New experiments
- **Full 18-method runtime benchmark**: Extended `benchmark_runtime_end2end.py` to cover all 18 method families (previously only 6 representative agents). Results merged in `outputs/runtime_full18/`.
- **Neural parameter counting**: New script `utility/count_torch_params.py` counts trainable PyTorch parameters for all 8 neural/RL agents.
- **Simple7 non-neural control tournament**: New roster `Agent_seats/reviewer_simple7.csv` (7 agents: R, CG, WL, B_v1, B_v2, M_v1, M_v2). Detected 7 directed three-cycles; strongest: B_v1 ≻ B_v2 ≻ M_v1 ≻ B_v1 (min edge = 197.5).

### New scripts
- `utility/count_torch_params.py` — Counts trainable torch parameters for all 18 method families
- `utility/make_score_vs_params.py` — Generates capacity–performance scatter plot (Fig S8)

### New figures
- `paper_RPS/Fig/RPS_sophistication/score_vs_params.png` — Score vs. parameter count for neural families
- `paper_RPS/Fig/RPS_metagame/pairwise_heatmap_simple7.png` — Pairwise payoff heatmap for Simple7 roster

### Paper changes (v5.8 → v5.10)
- **SI Table S13**: Tiered taxonomy + capacity proxies for all 18 method families
- **SI Figure S8**: Capacity–performance scatter (neural families only)
- **SI Section S13**: Low-sophistication roster control (Simple7)
- **SI Figure S9**: Simple7 pairwise heatmap
- **SI Table S15**: Simple7 three-cycle enumeration
- **Main text Discussion**: Two new substantive paragraphs ("Sophistication and tournament performance", "Non-transitivity without neural agents") with specific numbers
- **Main text Related work**: 3 generic sentences removed, transition paragraph added
- **Main text Theoretical framework**: Population I/II formal definition added
- **Main text Methods**: "Computational environment" paragraph added
- **Main text Introduction**: "minimal" → "widely studied" for RPS description

## v5.8 (2025-01 — first revision)

### Paper changes
- Streamlined Related work, made contributions explicit
- Clarified meta-game definition (population II payoff convention)
- Expanded tournament structure description (schedule, state handling, logging)
- Added sophistication spectrum discussion in Limitations

### Experiments
- Core-54 tournament (unchanged from initial submission)
- Core-19 validation tournament
- Runtime benchmark for 6 representative agents
