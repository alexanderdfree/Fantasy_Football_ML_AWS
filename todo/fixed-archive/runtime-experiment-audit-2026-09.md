### [FIXED] Experiments lost seeds, native frames, matched controls and scoring identity

**File(s)**: `src/analysis/`, `src/tuning/feature_groups.py`,
`src/tuning/ablate_backbone_norm.py`, `src/tuning/ablate_injury_features.py`,
`src/tuning/ablate_ridge_pca.py`, `src/tuning/ablate_rb_gate.py`,
`src/tuning/ablate_scheduler_type.py`, `src/shared/error_analysis.py`,
`src/shared/registry.py`, `src/dst/data.py`, and offensive-position `targets.py` files.
Defects reproduced against `92be2873` during the 2026-09-10 audit.

**What**:

- TabPFN/significance entrypoints discarded requested seeds; cached TabPFN
  results also conflated seeds. Native K/DST diagnostics used generic player
  splits, and ablations called incompatible self-loading runner signatures.
- Leave-one-group-out effects used other dropped groups instead of baseline.
  Multi-position normalization reports lost positions; filtered seed lists
  were paired by order; several reports discarded measured standard deviation.
- Scheduler subsets crashed without required comparisons, or used the wrong
  comparator variance. A malformed Markdown separator broke its report table.
- Missing chart weeks erased actual starter transitions from alignment checks.
  A global shift leaked one kicker's future expanding mean into another player.
- Non-PPR artifact/fresh/reliability paths mixed scoring formats or component
  bases. Disjoint valid selections produced NaN F1, and external benchmark
  paths crashed final metadata serialization.
- Fixed quartile labels conflicted with duplicate quantile edges. K nested
  inference omitted configured head widths. Two-point conversion adjustments
  falsely warned about correct canonical targets and could hide corruption.
- Native D/ST data omitted its proven REG season type. The subsequent pass
  reproduced public `run_cv()` failing the shared fold contract at `0f0fec55`.
- QB air-yards reports selected rare receiving touchdowns instead of passing
  touchdowns. RB diagnostics charged unprojected passing points as errors.
- Ridge validation scoring supplied the same rows as validation and test,
  corrupting history features when feature construction concatenated them.
- Warm-start reports hid failed requested arms and compared unpaired seeds;
  a missing arm could reverse the reported effect or leave an all-failed run
  with a successful exit status.
- Sleeper and FFToday cached partial responses after transient source failures.
  FFToday's joined cache could also retain the partial result or overwrite the
  default identity cache with caller-supplied rosters.
- TabPFN, weekly expert and tier reports counted unprojected actual stats as
  errors. Their older slim caches did not retain enough scoring information.
- Local expert loading discarded populated DST stat lines with zero or negative
  totals. Top-N exports mislabeled a retrospective actual-ranked slice as
  `elite_top24`, which denotes prior-season importance.

**Fix**: Preserve requested seeds and cache them explicitly; share native frame
preparation and K history closure semantics inside analysis-only helpers.
Keep supplied KEEP/CUT/validation/test frames and all six supported positions.
Use same-position, same-seed controls and measured uncertainty. Handle missing
comparison arms explicitly. Calculate starter adjacency before chart joining,
shift expanding statistics within player groups, and use the requested format
with canonical shared projected components. Preserve missing-data status,
return zero F1 for valid zero overlap, support external result paths, and keep
tied quantile values together. Mirror K head widths and validate target
decomposition against the actual upstream scoring contract. Preserve REG
metadata from the native D/ST schedule population so shared CV folds can run.
Select the position's intended TD cohort and score RB actuals from matching
components. Supply validation rows only once to Ridge-only scoring. Retain
warm-start failures, fail incomplete requests, and compare matched seed/origin
pairs with explicit unavailable status when no pair exists.
Only complete expert fetches are persisted, at both raw and joined layers;
unmarked legacy caches rebuild and supplied rosters never replace shared IDs.
Expert reports consume the pipeline's explicit projected-total truth or rescore
raw components. Format changes preserve the producer's pre-imputation missing
data mask. Full fantasy labels remain available to standalone diagnostics;
versioned slim caches retain the required truth metadata and raw components.
Populated nonpositive local projections remain available while empty placeholders
stay excluded. The retrospective Top-N slice is named `seasonal_actual_top24`.

**Validation**: Original-code controls reproduce each defect. Four real
CONFIG_TINY KEEP/CUT cells ran for K/DST with attention enabled and finite
predictions. Analysis and legacy tuning suites passed 323 and 307 unit tests
respectively before combined delivery checks. Six-position checkpoint tests
verify training/inference shapes; source-table controls retain healthy PPR,
complete coverage, correctly paired seeds and real-corruption detection.
The D/ST CV regression reaches all four real fold partitions before a mocked
training boundary. Its paired normal-pipeline comparison is recorded in
`benchmark_history/audits/2026-09-10-runtime-dst-comparison.json`.
Actual QB/RB preparation verified that corrected Ridge scoring matches the
canonical validation matrix, with unchanged training matrices; an isolated
tiny pipeline exercised real alpha tuning and PCA. Warm-start regression
controls reproduce a reversed unpaired conclusion and retain complete-pair
results without changing training or initialization functions.
The follow-up gate passed 424 tests with one skip; 22 frontend Node tests passed.
Source recovery controls include HTTP/connection/timeout failures, valid 404s,
legacy partial caches, successful cache reuse and supplied-roster isolation.
Four WR stat-line controls reproduce 0.825 spurious MAE for otherwise perfect
component forecasts. PPR, Half-PPR and Standard report tests preserve full labels,
verify actual pipeline metadata and reject missing components. Populated DST
zero/negative forecasts and empty placeholders have separate controls.

**Lesson**: A correctly shaped report can still describe a different seed,
dataset, scoring basis or comparison. Follow the real caller and preserve
identity through preparation, training, caching and aggregation.

Model-impact corrections to hurdle means, trade histories, signed-stat
activity filtering and validation-loss weighting are tracked separately.

### [FIXED] Weekly rankings combined the same week across different seasons

**File(s)**: `src/shared/evaluation.py`, `src/shared/backtest.py`, and
`tests/shared/test_multiseason_weekly_ranking.py`; reproduced against `13fe89d6`.

**What**: The supported multi-season expert comparison combined week 1 from
different years into one player slate. Repeated player IDs then made perfect
forecasts score a 50% top-12 hit rate, while single-season inputs scored 100%.

**Fix**: Group by season and week when season is present, retain week-only
input support, and carry season identity into result rows and chart labels.

**Validation**: The actual expert-comparison caller and weekly simulation now
retain two distinct slates with perfect rankings. The focused suite and existing
six-position backtest/evaluation tests passed 105 checks. Model predictions and
overall point-error calculations are unchanged.

**Lesson**: Week numbers are not unique game-period identifiers across seasons.

### [FIXED] Pipeline ranking and backtests charged unprojected actual components

**File(s)**: `src/shared/pipeline.py`, `src/tuning/tune_lgbm.py`, and focused
reporting tests; reproduced against `13fe89d6`.

**What**: Holdout, CV, split-branch ranking and tuned-LightGBM comparisons
aggregated model forecasts from configured targets but used full fantasy
actuals. Perfect RB component forecasts therefore incurred errors on three
player-weeks with passing points, which the RB models do not project.

**Fix**: Add `actual_projected_total` and scoring-basis metadata on evaluation
copies, using the configured target aggregation. Preserve full fantasy scores,
training/history frames and model predictions. Use the explicit truth column
for ranking and weekly simulation, and compute the season-average comparator
on matching components. Missing/nonfinite configured components remain
unavailable, including original raw observations filled during target
preparation. Cohort reports retain regular-season row totals and unavailable
actual counts; both Batch branches emit matching metadata before merging.

**Validation**: All six positions pass holdout/CV/partial reporting controls,
including optional model families, reduced configurations, missing components,
and split-cohort merges. The focused pipeline, tuning, cohort and Batch subset
passed 190 tests. Actual cached RB preparation and pipeline reporting with
exact raw-head predictors changed weekly MAE from `0.007717700626` to zero
across 1,757 test rows. The 1,754 nonpassing rows remain a zero-error control.
Source frames, prepared arrays, model predictions and full fantasy totals have
identical before/after hashes. Model fits were mocked for this reporting proof;
it makes no numerical-training or GPU claim.

**Lesson**: Carry an explicit component basis through each report instead of
assuming a broad fantasy-score column matches the model's configured heads.
