### [FIXED] Pipeline reports scored full fantasy points and merged seasons' weekly slates

Extracted from PR #1565 (`codex/audit-runtime-correctness` @ 7af61481, commits
028aa896 and 80c9db4d) and re-derived on the certified comparison truth that
#1574 added to `main` (`src/shared/comparison_truth.py`,
`src/shared/comparison_scoring.py`). Reported numbers change; raw predictions
and fitted weights do not.

#### Pipeline ranking and backtests charged unprojected actual components

**File(s)**: `src/shared/pipeline.py`, `src/tuning/tune_lgbm.py`,
`tests/shared/test_pipeline_reporting_truth.py`; reproduced against `c9f7dc4c`.

**What**: Holdout, CV, split-branch ranking, the weekly backtest, the
season-average comparator and the tuned-LightGBM comparison aggregated model
forecasts from the configured targets but scored them against the full
`fantasy_points` column. Perfect RB component forecasts therefore incurred
errors on the three 2025 player-weeks with passing points, which the RB models
do not project (mean absolute truth gap 4.52 on those rows; 0.0077 spread over
the 1,759-row test set), and the true weekly top 12 was selected on a broader
score than the forecasts covered.

**Fix**: Every report scores `actual_projected_total`. Canonical target
builders already certify that column from the raw observation before their
training fills (#1574), so `_reporting_frame` reuses it whenever the
configured heads cover the shared components under the canonical scoring;
custom or reduced configurations derive it from their own `y` dict with the
configured aggregation, masked by finiteness and by the preprocessing
availability mask when present. Nothing falls back to full fantasy points, and
`fantasy_points` itself stays untouched for history consumers. Ranking and the
weekly simulation drop rows whose truth is unavailable instead of scoring a
fill; the certified DST truth omits the non-shared points-allowed tier
(ADR-0024), so its model totals are rescored from their raw heads with
`comparison_model_totals` before ranking, matching the cohort, serving and
analysis comparisons. The season-average baseline is built from the same
truth. Cohort reports carry `evaluation_rows_total` and
`actual_rows_unavailable` on every record so split Batch branches expose the
same coverage before merging. `tune_lgbm._run_comparison` ranks the old and
tuned models on the same basis.

**Validation**: `tests/shared/test_pipeline_reporting_truth.py` (76 checks:
all six positions in holdout, CV and both split-branch modes, with certified
and uncertified frames; optional model families; reduced and custom
configurations; missing raw components; split-cohort merges; regular-season
coverage counts) plus the evaluation, backtest, cohort, tuning, e2e and
rolling-origin suites (741 further checks) passed. RB CPU eager seed-42
benchmark against the `origin/main` baseline: all 38 fitted artifacts (NN and
attention weights, scalers, per-target Ridge and LightGBM files) are
byte-identical; the eight manifest JSONs differ only in `saved_at`,
`bundle_id`, `provenance.code_id`, `provenance.data_id` and digests of those.
Per-family MAE/RMSE/R² and every cohort block are identical; the top-12 hit
rates moved by +0.004 to +0.005 (Ridge 0.463 → 0.468, NN 0.454 → 0.458,
Attention NN 0.463 → 0.468, LightGBM 0.491 → 0.495) and the backtest MAE now
equals the per-target total MAE by construction. No RB test row lost its
actual (0 of 1,759 unavailable).

**Lesson**: Carry an explicit component basis through each report instead of
assuming a broad fantasy-score column matches the model's configured heads;
when a certified truth already exists, reuse it rather than re-deriving a
second copy of its availability rules.

#### Weekly rankings combined the same week across different seasons

**File(s)**: `src/shared/evaluation.py`, `src/shared/backtest.py`,
`tests/shared/test_multiseason_weekly_ranking.py`; reproduced against `c9f7dc4c`.

**What**: The multi-season expert comparison and the weekly simulation
combined week 1 from different years into one player slate. Repeated player
IDs then made perfect forecasts score a 50% top-12 hit rate, while
single-season inputs scored 100%.

**Fix**: Group by season and week when a season column is present, retain
week-only input support, and carry the season identity into result rows and
chart labels.

**Validation**: The focused suite covers single- and multi-season ranking,
week-only input, the weekly simulation and the expert-comparison caller; the
existing six-position evaluation/backtest suites are unchanged. Model
predictions and overall point-error calculations are unaffected.

**Lesson**: Week numbers are not unique game-period identifiers across seasons.
