# Late-season (weeks 17–18) effect — findings

**Question.** Should the NFL season's final week(s) be excluded because teams rest
starters, making those games unrepresentative for fantasy?

**Decision.**
- **Training: keep all weeks** (ablation-confirmed — dropping the final week does not
  help and slightly hurts the best model).
- **Eval/reporting: exclude the season's *final* week** (season-aware) — it is not a
  fantasy week and it is high-variance rest noise. **Keep week 17** (the championship).
  **Status: recommendation, NOT yet wired into production eval.** `src/shared/evaluation.py`
  and `src/shared/backtest.py` still iterate every `test_df['week'].unique()` (incl. 2025
  wk18), so the dashboard / `benchmark_history/` numbers currently grade against the final
  week. Implementing the filter rebaselines every benchmark metric, so it is deferred to the
  next retraining cycle and tracked in TODO.md rather than shipped silently here. (#615)
- **Never** a flat `week == 18` cut — it is era-wrong (see below).

Run with `python -m src.analysis.cohort_analysis late_week --deep-dive`
(stage 1), `--with-model-error` (stage 2), or `--ablation`.

## Why "the final week", not "weeks 17–18"

The NFL went from 17 regular-season weeks (≤2020) to 18 (2021+). The rest week is the
season's *last* week — **wk17 in 2012–2020, wk18 in 2021+** — so the right unit is
`max(week)` per season. A flat `week == 18` is a **no-op for 9 of the 12 training
seasons** (2012–2020 have no wk18); a flat `week <= 16` would wrongly drop the
post-2021 championship (wk17). Week 17 in the modern era is the fantasy championship —
the most decision-relevant week — and its label drop is mild.

## Stage 1 — label anomaly (no models; skill positions only)

The raw split's `fantasy_points` is skill-scoring only (≈0 for K; DST absent from that
split), so Stage 1 covers QB/RB/WR/TE. The final week is depressed, concentrated in
wk18, and high-variance:

| Pos | wk17 (championship) Δmean | wk18 (final/dead) Δmean | 2025 test |
|-----|------|------|-----------|
| QB  | −2.2% | **−24.0%** (−3.19 pts) | most rested |
| WR  | −8.8% | **−16.9%** (−1.14) | |
| RB  | −3.9% | **−10.0%** (−0.74) | |
| TE  | −2.7% | −2.1% | ~flat |

Player-relative (established players vs their own early-season baseline): the *median*
established starter declines every position in the final week, but the *mean* is
near-zero/positive — a few players explode in garbage time. The final week is both
**lower and fat-tailed** (variance, not just level). The drop is **era-stable**
(QB −1.92 in 2012–2020 vs −1.77 in 2021+), validating the season-aware bucketing.

## Ablation — is the final week worth keeping in *training*?

KEEP (train all weeks) vs CUT (drop each season's final week from train+val), same
seed=42, both scored on the identical 2025 test set restricted to **weeks 1–17** (the
deployment-relevant slice). Dropping the final week removed 3,759 train rows (6.2%).
`dMAE = CUT − KEEP` > 0 means cutting hurts.

| Pos | best model | MAE keep | MAE cut | dMAE | verdict |
|-----|-----------|---------|--------|------|---------|
| QB  | Ridge      | **6.571** | 6.618 | **+0.047** | KEEP better |
| WR  | Neural Net | **4.223** | 4.285 | **+0.061** | KEEP better |

- Neither best model benefits from cutting; **WR is unanimous** (every model
  neutral-or-worse when cutting). The QB non-best models (Attention −0.045, LightGBM
  −0.029) are within NN seed noise.
- The **deterministic Ridge** result (QB +0.047, WR +0.004) is the clean read — a real
  data effect, not seed variance.
- **`Season Avg` dMAE = +0.000 exactly** sanity-checks the harness: that baseline does
  not depend on final-week training rows, so identical-across-conditions is expected.

Dropping ~6% of training data buys nothing and costs the best model a little — exactly
what "Huber already absorbs the outliers + more data helps" predicts.

## Caveats / follow-ups
- **Magnitude is small** (~0.05 MAE). The eval-side exclusion is primarily a
  metric-*alignment* fix (report on weeks you actually play), not a big number move
  (wk18 ≈ 5% of the test set). A multi-seed ablation would tighten the NN deltas.
- **K/DST** were not in Stage 1 (raw-split scoring limitation). They get correct totals
  via the per-position pipeline in `--with-model-error` / `--ablation` if coverage is wanted; the
  effect there is second-order (kickers/defenses play regardless of offensive rest).
- **Latent bug surfaced (independent of this) — investigated 2026-05-30, benign:**
  `snap_pct` imputation groups by raw `(position, week)`
  (`src/data/preprocessing.py::impute_snap_pct`), conflating 2012–2020's *final* wk17
  with 2021+'s *penultimate* wk17 — an era-contaminated median in principle. **No
  production impact: the median branch is dead code.** `src/features/engineer.py`
  zero-fills every `snap_pct` NaN (`groupby(["player_id","season"]).shift(1).fillna(0)`,
  L290) *before* any split function runs, so by the time `impute_snap_pct` executes — in
  `temporal_split` (on that in-memory post-`build_features` frame) and in the CV /
  rolling-origin folds (`expanding_window_folds` / `rolling_origin_folds`, which re-read
  the post-`build_features` on-disk splits) — there are **0 NaN** to fill — confirmed
  empirically (train/val/test all 0.00% NaN `snap_pct`; the 11–21% exact-zero mass is the
  lag signature). The era-conflated median is never computed against a real row, and would
  be small anyway (week-17 era-Δ ≤ 0.06 snap-share across positions). Left unchanged —
  a `src/data/` edit to dead code would force a guaranteed Δ=0 six-position retrain. Full
  write-up in TODO.md → `[INVESTIGATED, BENIGN]`.
