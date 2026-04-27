# Self-Assessment Supplemental Evidence

Companion document to [self_assessment_alex_free.docx](self_assessment_alex_free.docx). The .docx evidence-pointer cells are size-constrained; this file provides extended pointers, regeneration instructions for gitignored outputs, and historical artifact recovery commands so that every ML rubric claim is verifiable end-to-end.

## How to use this document

For each claim in the .docx, the cell points to the **primary** evidence. This file adds:
1. **Secondary evidence** — additional file/line pointers that further support the claim
2. **Regeneration commands** — how to produce outputs that are gitignored locally
3. **Historical recovery** — `git show <sha>:<path>` commands for archived files

A grader who wants to spot-check any claim can use the verification block at the end of each section.

---

## Per-claim supplemental evidence

### Claim #5 — Error analysis with failure cases (7 pts)

**Primary evidence (in .docx):** `QB/diagnose_qb_outliers.py`, `RB/analyze_rb_errors.py`, `docs/expert_comparison.md:99-141`.

**Secondary evidence:**
- The QB diagnostic script's output schema is declared at [QB/diagnose_qb_outliers.py:16-17,47-48](../QB/diagnose_qb_outliers.py) — it produces `analysis_output/qb_outlier_diagnostic.{md,json}` with per-row attributions, contamination stats, and 8-game player history.
- The RB analyzer's figure-output is declared at [RB/analyze_rb_errors.py:138](../RB/analyze_rb_errors.py) — produces `analysis_output/rb_error_mae_by_{stratum}.png` for snap-bucket / opp-tier / TD-bucket / week-phase / volatility-quintile slices.
- Both output paths land under `analysis_output/`, which is in [.gitignore](../.gitignore) — these files are not committed.

**Regeneration:**
```bash
# Requires data/splits/*.parquet and trained models in {POS}/outputs/models/
python QB/diagnose_qb_outliers.py     # writes analysis_output/qb_outlier_diagnostic.{md,json}
python RB/analyze_rb_errors.py        # writes analysis_output/rb_error_*.png
```

**Verification:** Open [docs/expert_comparison.md:103-127](../docs/expert_comparison.md) — the per-position narrative analysis is the discussion component and is committed to the repo. The two diagnostic scripts are the analysis machinery.

---

### Claim #6 — Ablation study (7 pts) ⭐ EVIDENCE OF IMPACT

**Primary evidence (in .docx):** README NN-vs-Attn columns; `.github/workflows/ablate-rb-gate.yml`; `tune_rb_gate_results.json`; `analysis_output/rb_feature_signal_ablation.png`.

**Secondary evidence — full ablation table from [tune_rb_gate_results.json](../tune_rb_gate_results.json):**

| Configuration | attn_mae |
|---|---|
| baseline (w=1.0, h=16, d=3.0) | 4.2271 |
| gate_weight=1.5 | 4.2453 |
| gate_weight=2.0 | 4.2355 |
| gate_weight=3.0 | 4.2418 |
| **gate_hidden=24 (w=3.0)** | **4.2217** ← best |
| gate_hidden=32 (w=3.0) | 4.2658 |
| huber_delta=2.0 (w=3.0, h=24) | 4.2534 |
| huber_delta=1.5 (w=3.0, h=24) | 4.2844 |
| td_loss_w=2.5 (gw=3.0, h=24, d=1.5) | 4.2832 |
| td_loss_w=3.0 (gw=3.0, h=24, d=1.5) | 4.2976 |
| td_lw=2.5, d=2.0 (gw=3.0, h=24) | 4.2739 |
| d=2.0, td_lw=2.0 (gw=3.0, h=24) | 4.2534 |

12 configurations swept across 4 hyperparameter dimensions (gate_weight, gate_hidden, huber_delta, td_loss_w). Spread: 0.0759 MAE between best and worst. Best variant (`gate_hidden=24`) became the production config.

**Verification:** `python -c "import json; print(json.dumps(json.load(open('tune_rb_gate_results.json')), indent=2))"` reproduces the table.

---

### Claim #7 — Preprocessing with quantitative impact (7 pts) ⭐ EVIDENCE OF IMPACT

**Primary evidence (in .docx):** `TODO.md:73-77` (feature clipping with measured frequency 0.3-0.4% values clipped); `shared/pipeline.py:115-125` `_scale_xs`; per-position feature files for trade safety.

**Secondary evidence:**
- The `scale_and_clip` helper in `shared/feature_build.py` is the single source of truth — both training ([shared/pipeline.py:115](../shared/pipeline.py)) and serving paths use it. This deliberately closes the training-vs-inference skew the [TODO.md "Fixed" archive](../TODO.md) repeatedly warned about.
- Stint-aware grouping for trade safety also appears in [TE/te_features.py:51](../TE/te_features.py) (not just RB/WR cited in .docx).
- The "Open" section of [TODO.md:21-23](../TODO.md) acknowledges `drop_last=True` discards 121 of 32,521 WR training rows (~0.4%) — same order of magnitude as the clipping rate, and explicitly accepted as a known trade.

---

### Claim #11 — Hyperparameter tuning with documented impact (5 pts) ⭐ EVIDENCE OF IMPACT

**Primary evidence (in .docx):** `tune_lgbm_results.json` (WR LightGBM Optuna 50 trials with old/new metric comparison); `tune_rb_gate_results.json` (12 configs); `TODO.md:146-150` (loss-weight rebalance).

**Secondary evidence — WR LightGBM Optuna comparison from [tune_lgbm_results.json](../tune_lgbm_results.json):**

| Metric | Old | New | Δ |
|---|---:|---:|---:|
| total MAE | 4.2112 | 4.2358 | +0.025 |
| total RMSE | 6.0370 | **5.9476** | **−0.089** |
| total R² | 0.3212 | **0.3411** | **+0.020** |
| receiving_floor R² | 0.3967 | **0.4032** | **+0.007** |

Best trial #26 of 50, best CV MAE 4.7319. The retuning trades a hair of MAE for a meaningful R² + RMSE gain (less variance in errors), captured in the explicit `comparison.old_metrics` vs `comparison.new_metrics` block.

**Per-position config diversity** (from [instructions/DESIGN_DOC.md:279-286](../instructions/DESIGN_DOC.md)):

| Pos | Backbone | Head | Dropout | LR | Epochs | Batch | Scheduler |
|---|---|---|---|---|---|---|---|
| QB | [128] | 32 | 0.20 | 5e-4 | 300 | 128 | CosineWarmRestarts |
| RB | [128, 64] | 48 (td: 64) | 0.15 | 1e-3 | 300 | 256 | CosineWarmRestarts |
| WR | [128] | 32 | 0.20 | 1e-3 | 250 | 512 | CosineWarmRestarts |
| TE | [96, 48] | 24 (td: 32) | 0.30 | 5e-4 | 300 | 128 | OneCycleLR |
| K | [64, 32] | 16 | 0.25 | 3e-4 | 250 | 128 | OneCycleLR |
| DST | [128, 64] | 32/16/48 | 0.30 | 3e-4 | 300 | 128 | CosineWarmRestarts |

6 positions × 7 distinct hyperparameter axes — well above the rubric's "≥3 configurations" minimum.

---

### Claim #12 — ≥2 documented iterations of improvement (5 pts) ⭐ EVIDENCE OF IMPACT

**Primary evidence (in .docx):** `TODO.md:43-150` Fixed archive — 17 entries with What → Files → Fix → Lesson structure.

**Full inventory of "iterations" with quantitative impact** (from [TODO.md](../TODO.md)):

| TODO.md lines | Issue | Quantitative before/after |
|---|---|---|
| 47-51 | ECS deploy waiter timeout under AZ rebalancing | Deploy hung 40 min × 2 attempts → reliable 10-min completion after disabling AZ rebalance + grace-period restore |
| 53-57 | NN aux total-loss redundancy | Removed `w_total · Huber(preds["total"], target)` term; per-target Huber suffices |
| 59-63 | Total-loss / adjustment double-count | ~1.9 pts/game systematic QB bias eliminated |
| 65-71 | Softplus floor (twice — landed, regressed, re-fixed) | softplus(0)≈0.693 per head → ~2.08 pt floor on 3-head positions; switched to clamp(min=0); allows exact zeros, restores Ridge-NN scale parity |
| 73-77 | No feature clipping after StandardScaler | Test z-scores up to 19.5 → ±4σ clip → catches 0.3-0.4% of values, prevents catastrophic extrapolation |
| 79-83 | `kicker_week_split` import crash | App crashed on startup → renamed to `kicker_season_split` |
| 85-89 | DST `pts_allowed_bonus` clamped to ≥0 (range −4 to +10) | Per-position `non_negative_targets` parameter added |
| 91-95 | Eval added adjustment to preds but not target | Inflated MAE → removed `+ adj_test.values` from totals |
| 97-101 | `run_cv_pipeline` missed `non_negative_targets` kwarg | DST CV path silently broken → mirrored from `_train_nn` |
| 103-106 | Dead `adj_val`/`adj_test` post-removal | Removed unused variables |
| 108-112 | Weather/Vegas features missing at inference | Models trained on 12 features got zeros at serving time → added `merge_schedule_features()` to `_apply_position_models` |
| 114-117 | ReDoS in `/api/predictions` search + `int(week)` crash | `regex=False` + try/except 400 wrap |
| 119-123 | API serves multiple scoring formats with PPR-only models | Added `scoring_note` field to API response |
| 125-129 | RB test fixture missing `receiving_epa`/`receiving_air_yards` | 11 tests failed → fixture + RB_FEATURE_COLS updated |
| 131-134 | DST prior-season feature alignment via `.values` (silent reorder bug) | Switched to index-preserving merge |
| 136-139 | Feature column filtering could silently drop features at inference | Added count comparison + warning log |
| 141-144 | No API error handling | Added Flask `@app.errorhandler(Exception)` returning JSON |
| 146-150 | Huber-δ asymmetry starved count heads (~20-2500× gradient imbalance) | Rebalanced to w ≈ 2.0/δ across QB/RB/WR/TE; count heads viable; TD signal recovered |

**18 documented iterations**, well above the rubric's "≥2" minimum. Each entry includes a Lesson distillation.

---

## Outputs that require local regeneration

These files are produced by scripts in the repo but live under `analysis_output/` which is in [.gitignore](../.gitignore). A grader cloning the repo will not see them. They can be regenerated locally:

| Output | Generator | Command |
|---|---|---|
| `analysis_output/qb_outlier_diagnostic.{md,json}` | [QB/diagnose_qb_outliers.py](../QB/diagnose_qb_outliers.py) | `python QB/diagnose_qb_outliers.py` |
| `analysis_output/rb_error_mae_by_*.png` | [RB/analyze_rb_errors.py](../RB/analyze_rb_errors.py) | `python RB/analyze_rb_errors.py` |
| `benchmark_results.json` (raw-stat target naming) | [benchmark.py](../benchmark.py) | `python benchmark.py` |

The 6 PNGs already in [analysis_output/](../analysis_output/) (weather/Vegas correlations × 4 positions, cross-position summary, RB feature-signal ablation) **are committed** — they were force-added past `.gitignore`.

---

## Historical artifacts recoverable from git history

Files removed in [PR #134](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/134) (commit `6f993a0`). Their content is preserved in git history and can be retrieved at any time:

### `benchmark_BEFORE.json`

Single-entry RB snapshot under the pre-raw-stat-migration target naming (`rushing_floor`, `receiving_floor`, `td_points`).

```bash
git show f400a5c:benchmark_BEFORE.json
```

Snapshot content: RB Ridge MAE 3.784 (R² 0.551, top-12 0.491) vs NN MAE 3.852 (R² 0.505, top-12 0.477).

### `benchmark_BEFORE_defense.json`

Single-entry QB snapshot, same naming convention.

```bash
git show f400a5c:benchmark_BEFORE_defense.json
```

Snapshot content: QB Ridge MAE 6.333 (R² 0.178, top-12 0.505) vs NN MAE 6.388 (R² 0.127, top-12 0.523).

These were ad-hoc development snapshots that never ended up referenced from any code, doc, or .docx evidence pointer. They were removed for repo cleanliness (Cohesion: "Clean codebase, no extraneous/stale files").

---

## Grader verification checklist

For each claim, the fastest path to verify:

| Claim | Quickest verification path |
|---|---|
| #1 Solo | [README.md:5,156-158](../README.md) |
| #2 Web app | Visit [alexfree.me](https://alexfree.me); read [app.py](../app.py), [Dockerfile](../Dockerfile), [infra/aws/](../infra/aws/) |
| #3 Production deploy | `grep -n "_cache\|_cache_lock\|errorhandler" app.py` finds caching + error handling; [infra/ec2/cloudwatch-agent.json](../infra/ec2/cloudwatch-agent.json) for logging |
| #4 Architectures | [README.md:90-99](../README.md) eval table |
| #5 Error analysis | [docs/expert_comparison.md:99-141](../docs/expert_comparison.md) (committed discussion) + script files exist |
| #6 Ablation ⭐ | Table above (this doc) reproduced from [tune_rb_gate_results.json](../tune_rb_gate_results.json) |
| #7 Preprocessing ⭐ | [TODO.md:73-77](../TODO.md) (z-score 19.5 → ±4σ → 0.3-0.4% clipped) |
| #8 Time-series | [src/data/split.py:6-37](../src/data/split.py) (temporal split) + [shared/neural_net.py:202+](../shared/neural_net.py) (history attention) |
| #9 Simulation | [src/evaluation/backtest.py](../src/evaluation/backtest.py) (219 lines, week-by-week replay) |
| #10 Custom NN | `grep -nE "^class " shared/neural_net.py` lists 5 modules |
| #11 Hyperparam ⭐ | Tables above (this doc) reproduced from `tune_lgbm_results.json` and per-position configs |
| #12 Iterations ⭐ | 18-entry table above (this doc) reproduced from [TODO.md "Fixed" archive](../TODO.md) |
| #13 Both qual+quant | [README.md:90-99](../README.md) (quantitative table) + [docs/expert_comparison.md](../docs/expert_comparison.md) (narrative) + 6 figures in [analysis_output/](../analysis_output/) |
| #14 Regularization | `grep -nE "Dropout\|BatchNorm\|patience\|weight_decay" shared/neural_net.py shared/training.py` |
| #15 Feature selection | [docs/ARCHITECTURE.md:225-245](../docs/ARCHITECTURE.md) (D6) + [tests/test_attn_static_columns.py](../tests/test_attn_static_columns.py) enforces it |
