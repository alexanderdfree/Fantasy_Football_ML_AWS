# Rookie calibration — implementation plan (next priority)

A pick-up plan for a future session. The diagnosis is done and the evaluation
harness exists; this is the *fix*. Read [rookie_cohort_findings_priority.md](rookie_cohort_findings_priority.md)
first — it has the numbers this plan refers to.

## Context (what we know)

- The tracked rookie-subgroup metric now exists in [cohort_analysis.py](../src/analysis/cohort_analysis.py) (`rookie`; legacy wrapper [rookie_cohort_metrics.py](../src/analysis/rookie_cohort_metrics.py))
  (PR #620). This satisfies the precondition the #519 draft-capital rejection set
  (`TODO.md` `[TESTED, REJECTED] Draft-capital / combine rookie cold-start features`).
- **Rookies are NOT a high-MAE cohort** — they score fewer points, so smaller
  absolute errors (rookie MAE ≤ veteran MAE at 3/4 skill positions). The defect is
  **directional bias, invisible to overall MAE**:
  - **QB / WR / TE over-predict rookies' first ~3 games** (`rookie_early` bias: QB
    **+2.9 to +4.4 FP/g across every model**, WR +0.4 to +2.7, TE +0.6 to +1.8),
    then flip to *under*-prediction on `rookie_rest`.
  - **RB under-predicts throughout** (the ascension lag; see `rb_ascension.py`).
- This is exactly why #519 read benchmark-flat: a 14–21% subgroup whose error is a
  *sign-flipping bias* cannot move overall MAE.

## Goal

Reduce the rookie **early-game over-prediction bias** for the **best model per
position**, judged on the subgroup metric (**bias / bias-corrected MAE / top-12
ranking**) — **NOT** overall benchmark MAE.

> ⚠️ The overall benchmark WILL stay ~flat by construction (~1 FP/g recoverable on
> 36 QB `rookie_early` rows ≈ 0.05 overall MAE). **Do not read a flat benchmark as
> failure** — that is the #519 trap. Judge on the subgroup metric, which now exists.

## Where the win is (from the bias-corrected-MAE decomposition)

| Pos | best model | recoverable on `rookie_early` | verdict |
|-----|-----------|------------------------------:|---------|
| QB  | Ridge   | **+1.05** (up to +1.6 any model) | **strongest, reaches the best model** |
| WR  | NN      | +0.15 (but +1.25 in Ridge)      | best model already handles rookies |
| TE  | Attn NN | +0.29 (but +0.80 in Ridge)      | best model already handles rookies |
| RB  | Attn NN | −0.17 (none)                    | **no bias to remove — do NOT target** |

**Scope the first cut to QB.** Its best model (Ridge, linear) carries the largest
recoverable bias, and every model over-predicts rookie QBs early. WR/TE's best
NN/attn models already absorb the rookie signal; RB's error is irreducible spread.

## Recommended approach: a game-phase-aware rookie indicator (the model learns the correction)

Add two **non-temporal** static features (roster metadata, known pre-kickoff — no
leakage; `years_exp`/`draft_number` already flow through `src/data/loader.py` per #519):

1. `is_rookie` — 1 if the player's first NFL season is the current season
   (`years_exp == 0` / first-season-in-data). Reuse the labeling logic in
   [cohort_analysis.py](../src/analysis/cohort_analysis.py) `label_rookie_rows` for parity.
2. `rookie_early` (or a decaying `rookie_game_phase`) — 1 for the first ~3 games of
   the rookie season, else 0.

> 🔑 **A flat `is_rookie` offset is WRONG.** `rookie_early` is over-predicted (+) and
> `rookie_rest` under-predicted (−); a single rookie indicator averages to ~0 and
> does nothing (the pooled-bias cancellation in the findings). The signal **must be
> game-phase-aware** — `is_rookie × game_index`, or separate `rookie_early` /
> `rookie_rest` indicators.

Wire-up (QB first; this is CLAUDE.md "eligible reach #1: non-temporal features into
`ATTN_STATIC_FEATURES`"):

- `src/qb/features.py` — compute `is_rookie` + `rookie_early` (game index within the
  rookie season).
- `src/qb/config.py` — add to `_INCLUDE_FEATURES` (so Ridge/LightGBM see it) **and**
  `ATTN_STATIC_FEATURES` (non-temporal → eligible). Update `CONFIG_TINY` + the test
  fixture (`tests/qb/conftest.py`). Ridge will learn a negative coefficient on
  `rookie_early` ≈ −bias, directly cancelling the over-prediction.
- Rebuild splits (`build_features`), retrain QB (`python -m src.qb.run_pipeline`),
  then **evaluate with `python -m src.analysis.cohort_analysis rookie --positions QB --with-model-error`**.

**Success criteria** (vs the baselines below): `rookie_early` bias shrinks toward 0,
`MAEbc ≈ MAE` for `rookie_early` (the systematic part is gone), and top-12 hit rate
flat-or-up. Overall MAE flat is expected and fine.

## Hard constraints / stop-rules (read before starting)

- **Judge on the subgroup metric, not overall benchmark MAE** (the #519 trap).
- **Do NOT re-introduce draft CAPITAL** (`log(pick)`) — that's the specifically
  rejected pedigree prior (benchmark-flat, plus a serving-time `draft_number`
  dependency). `is_rookie` + game-phase is roster metadata only.
- **Game-phase-aware, not a flat rookie offset** — opposite signs cancel otherwise.
- `is_rookie`/`rookie_early` are **static indicators** — fine for `ATTN_STATIC_FEATURES`.
  Do NOT add any rolling/windowed rookie feature there (the no-rolling-in-static rule);
  temporal signal stays in `ATTN_HISTORY_STATS`.
- **Multi-seed**: judge the bias *direction* across ≥2 seeds, not a single-seed
  magnitude (NN/attn rookie deltas are ±0.08 single-seed noise).
- Check **training vs serving** feature parity: confirm `years_exp`/first-season is
  available for 2025 inference in `src/serving/app.py`, not just training.

## Baselines to beat (seed 42, from rookie_cohort_findings_priority.md)

- **QB Ridge `rookie_early`**: bias **+3.619**, MAE 6.222, MAEbc 5.172 (→ ~1.05 recoverable).
- Cross-model QB `rookie_early` bias all strongly positive (+2.9 to +4.4).

## Alternative (fallback): post-hoc residual calibration

If the feature underperforms, fit a per-position `rookie_early` mean-residual
correction on **train+val** (never test) and subtract it at prediction time (in
`aggregate_targets` or a thin calibration layer). Surgical, no retrain of the
model, but another moving part and must be re-fit per scoring format. The feature
approach is preferred (the model learns the magnitude and interactions).

## Open questions

- Feature vs post-hoc calibration (above) — start with the feature.
- Is a second cut at WR/TE worth it? The metric says their best models already
  handle rookies; likely low ROI — gate on whether the QB cut validates first.
