# Scoring-tier (top-drafted) tail accuracy — findings & plan

Goal: improve accuracy for the **highest-scoring / commonly-drafted players** without
regressing overall accuracy. Tracked because the tail defect is bias, not MAE, so it is
invisible in the headline benchmark and needs its own metric (the lesson behind the
rejected draft-capital features).

## Phase 1 diagnostic (shipped: `scoring_tier` cohort, PR #870)

`src/analysis/cohort_analysis.py` gained a read-only `scoring_tier` cohort. The elite slice
is **a-priori**: players ranked by prior-season mean fantasy points within `(season,
position)`, top-N (`--tier-topn`, default 24) = `elite_top_drafted`, lower veterans =
`field`, no-prior-season rows = `unknown`. No model feature, no retrain.

```
python -m src.analysis.cohort_analysis scoring_tier --positions QB RB WR TE \
    --tier-topn 24 --with-model-error --deep-dive
```

## Baseline result (TEST_SEASONS, PPR, seed 42)

`dMAE` = bucket MAE − model's overall MAE. `bias = mean(pred − actual)`; **negative = under-prediction**.

Elite-slice **bias is the real signal** (overall best model shown; direction is the same
across *all four* models, so it is robust at one seed):

| Pos | Best model (overall MAE) | Elite bias | Field bias | Read |
|-----|--------------------------|-----------|-----------|------|
| **RB** | LightGBM (3.96) | **−2.12** | −1.12 | Clear tail under-prediction (~2× field) |
| **TE** | LightGBM (3.35) | **−1.52** | −0.59 | Clear (~3× field) |
| WR  | Attention NN (3.97) | −1.19 | −1.55 | NOT a bias problem — field is *more* negative; variance, not bias |
| QB  | LightGBM (5.80) | +0.47 | −0.91 | Calibrated; slight over-prediction |

`dMAE` is large and positive on elite for **every** position (+0.26 QB → +2.5 WR) purely
because elites score more — confirming MAE alone would mis-target WR. Bias is the lens.

Deep-dive (which head drives it, best model):
- **RB elite:** `rushing_yards` −10.5, `receiving_yards` −5.3.
- **TE elite:** `receiving_yards` −5.6, `receptions` −0.43.
- QB `passing_yards` is internally over-predicted (+13.7 on elite, LightGBM) but offsets in total FP.

## ALL MODELS MATTER (standing guidance)

Do **not** scope this to the served best model only (the draft-capital trap was about a gain
that reached only a non-best model — a different failure). For this work the requirement is
the inverse: **every model's elite accuracy matters**, and an intervention should be judged
on whether it lifts the tail across Ridge / NN / Attention NN / LightGBM, not just the
current best. Two consequences:
1. The elite under-prediction is present in **all four** models for RB and TE — a good fix
   should reduce it broadly, and per-model elite bias is the acceptance metric.
2. The expert head-to-head (`analysis_expert_comparison.py`) compares **only** the attention
   NN (`pred_attn_nn_total`). The new `tier_expert_comparison` (below) compares **all** our
   models and each expert per tier; future expert tooling should keep all-model coverage.

## Phase 2 candidates (revised by the evidence)

Target = **elite RB + TE**, yardage heads. Because the best model for RB/TE is **LightGBM**
(and all models under-predict), the priority is reordered vs the original plan:

1. **Per-sample loss weighting first** — threads through LightGBM's native `sample_weight`
   *and* the NN loss, so it can lift the tail for every model at once (incl. the served best
   model for RB/TE). Weight = monotone fn of the a-priori prior-season-FP percentile.
2. **Asymmetric / quantile-tilted loss** on yardage heads — NN-side; secondary given best
   model is LightGBM, but still relevant under "all models matter."
3. **Eligible non-temporal role/volume static features** — largely already wired; last.

Each default-off, A/B'd via an `src/tuning/ablate_*.py` harness, accepted only if elite bias
shrinks across models/seeds **and** overall per-target MAE stays flat (≥8 seeds, mean±std,
Ridge data-identity sentinel).

## Expert comparison (elite vs field, our models vs NFL.com / Sleeper)

`python -m src.analysis.tier_expert_comparison --positions QB RB WR TE --tier-topn 24`
(read-only; runs each pipeline once, joins expert projections on matched player-weeks,
slices by the same a-priori tier). Run on TEST_SEASONS (2025), PPR, CPU/FP32 (CI-identical), seed 42.

**Elite tier — our best model vs NFL.com** (Sleeper tells the same story; bias = mean(pred−actual), negative = under-prediction):

| Pos | Our best (MAE / bias) | NFL.com (MAE / bias) | Takeaway |
|-----|-----------------------|----------------------|----------|
| QB | LightGBM 6.05 / **+0.47** | 6.23 / +0.96 | We win MAE **and** are better calibrated — QB needs nothing |
| RB | Attn NN 5.95 / −0.63 (LGBM 6.05 / **−2.12**) | 6.20 / **+0.28** | We win MAE, but expert is ~unbiased while we under-predict — gap is recoverable |
| WR | Attn NN 6.40 / **−1.19** | 6.48 / +0.61 | We edge MAE; expert is +biased, our NN-family under-predicts elite |
| TE | LightGBM 5.00 / **−1.52** | 5.15 / **−0.30** | We win MAE; expert much better calibrated on elite |

**The decisive finding:** on the elite slice our models **match or beat both experts on MAE at
every position**, but the experts are **near-unbiased on elite RB/TE/WR (+0.3 / −0.3 / +0.6)
while we carry a consistent −0.6…−2.1 under-prediction.** Since the experts forecast the *same*
player-weeks without that bias, the elite under-prediction is a **removable training artifact
(regression-to-the-mean), not an inherent property of the problem** — and the expert's ~0 bias
is the concrete **target** for Phase 2's acceptance metric. QB is the exception (we already beat
experts on both MAE and calibration). Caveats: one season, one seed, CPU/FP32 — fine for bias
*direction*; Phase-2 8-seed A/Bs belong on the GPU production path.

All-models note: the under-prediction spans the whole family; on this matched elite slice the
**Attention NN** is both lowest-MAE and least-biased for RB (−0.63), while **LightGBM** — best
*overall* on RB — is the most under-biased on elite RB (−2.12). Best-model identity shifts on the
tail, reinforcing that interventions must be judged per-model, not on the served best alone.

## Root cause of the 1–2 pt elite under-projection

Ranked by contribution:

1. **Robust regression objective (Huber / Fair) discounts the tail.** Beyond its delta, Huber
   penalizes a miss *linearly*, not quadratically, so a big miss on a monster elite game produces
   far less gradient than under squared error — the optimizer is told "don't chase the big weeks,"
   which are exactly the elite weeks. Confirmed by **Ridge** (the only model on squared error)
   being the least-biased on elites everywhere (even over-projects elite WR, +0.46), matching the
   MSE-oriented experts. The objective is **per-position** (`lgbm_objective`, read at
   `src/shared/pipeline.py:1161`), and the NN yardage heads use `"huber"` via `head_losses`:

   | Position | LightGBM `lgbm_objective` | NN yardage heads (`head_losses`) |
   |----------|---------------------------|----------------------------------|
   | QB | `fair` | `huber` (passing_yards, rushing_yards) |
   | RB | `huber` | `huber` (rushing_yards, receiving_yards) |
   | WR | `huber` | `huber` (receiving_yards) |
   | TE | `huber` | `huber` (receiving_yards) |
   | K  | `huber` | — |
   | DST| `huber` | — |

   The resulting under-projection scales with **objective × target skew**: severe on RB/TE
   (huber + right-skewed yardage), absent on QB (fair + a higher-floor, less-skewed target).

2. **Regression to the mean × extrapolation ability.** Trees (**LightGBM**) cannot predict above
   the training max — a leaf emits the average of its targets — so elite weeks get capped/averaged
   down (worst bias). Linear (**Ridge**) extrapolates → least biased. NN sits between. This
   compounds with (1) in LightGBM and is absent in Ridge.

3. **Raw-stat decomposition stacks the shrinkage.** Each head shrinks toward its own mean and the
   heads are summed to FP, so an elite RB's `rushing_yards` (−10.5) + `receiving_yards` (−5.3)
   under-projections **add up** in the total.

4. **Count heads (Poisson NLL) mean-revert on TDs** — expected-rate predictions sit below what
   actual TD-scorers do; at 6 pts/TD even −0.1 TD/gm = −0.6 FP. Secondary, additive.

5. **Sample imbalance** — elites are ~13–26% of rows, so the average loss is dominated by the
   mid/low bulk and the tail is under-served in fitting.

Mirror image (sanity check): the same shrinkage + the ≥0 non-negativity clamp pushes the *bottom*
up, so bias flows out of the tail and into the middle.

## Decision: switch the regression objective Huber/Fair → MSE (in progress)

Directly attacks cause (1): squared error chases the tail like the experts. **Trade-off to
measure:** FP is heavy-tailed (errors ~Laplacian), so MSE is theoretically mismatched for the
*bulk* and may **raise overall MAE** even as it shrinks elite bias — exactly the tension this work
must adjudicate. The NN switch additionally breaks the documented `LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS`
coupling (dropping it once regressed QB FP MAE 6.33→6.63), so NN loss weights must be re-derived,
not left at the Huber-tuned values. Acceptance: report per-position overall MAE **and** elite bias
(toward the expert ~0) before/after; keep if tail improves without an unacceptable overall hit.
