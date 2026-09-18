### [FIXED] K and D/ST cross-validation folds imputed game context before slicing

**File(s)**: `src/k/data.py`, `src/k/run_pipeline.py`, `src/dst/data.py`,
`src/dst/run_pipeline.py`, `src/benchmarking/benchmark.py`
(`_self_load_full_frame_and_cfg`), `src/tuning/tune_lgbm.py`
(`_prepare_cv_folds`). Extracted from #1565 (`codex/audit-runtime-correctness`
@ `7af61481`, commits `8974c382`, `00316644`, `f21d4f4c`); the defect was
reproduced against `0f0fec55` in the 2026-09-10 runtime audit.

**What**:

- D/ST native CV and rolling-origin preparation filled missing Vegas lines,
  the `opp_scoring_L{3,5}` / `opp_turnovers_L5` / `opp_sacks_allowed_L5`
  windows and the opposing-QB rate features with statistics computed over the
  fixed `TRAIN_SEASONS` (2013–2023) inside `build_data()`, *before* the
  expanding-window fold or rolling origin was sliced. Perturbing only 2023
  held-out scores therefore changed 2013–2021 training rows and 2022
  validation season openers of an earlier origin.
- K native CV and rolling-origin preparation likewise filled missing
  `total_line` / `implied_team_total` with the ≤ 2023 median inside
  `load_data()`. In a real 3,573 × 19 prepared K matrix with one deliberately
  missing historical line, a later-season change altered two training rows.
  Current 2015–2025 REG schedules carry no missing Vegas lines, so this is a
  supported missing-data boundary rather than observed default-cohort
  incidence.
- The D/ST loader did not stamp `season_type`, which the shared
  cross-validation fold contract reads; every D/ST row comes from
  `schedules_reg`.
- K PBP caches were keyed by `seasons[0]_seasons[-1]`, so a sparse season
  selection could reuse a cache built from a different selection sharing the
  same endpoints, and a per-kick PBP fetch that failed for *every* season
  returned the empty schema frame instead of raising like a partial failure.

**Fix**: `load_data(impute_context=True)` and `build_data(impute_context=True)`
keep their defaults, so production `run()` is unchanged; `impute_context=False`
leaves the context columns missing and the new
`impute_context_from_train(df, fit_on=...)` helpers apply the same fills from
an explicit training population (missing training evidence fills `0.0`, never
a holdout statistic; the ordinary loader keeps its full-frame fallback).
`with_fold_imputation(cfg)` wraps the position's `fill_nans_fn`, the existing
post-split boundary that `_prepare_position_data` runs for every CV fold,
rolling origin and the final refit, so each preparation fits the context on
its own train frame. `run_cv`, the benchmark's `_self_load_full_frame_and_cfg`
and `tune_lgbm._prepare_cv_folds` load with `impute_context=False` and use the
wrapped config. D/ST rows carry `season_type="REG"`. K cache filenames use
`_seasons_cache_signature` (contiguous ranges keep their legacy names) and the
per-kick builder raises when every season failed.

**Validation**: `pytest tests/k tests/dst tests/test_benchmark_rolling_origin.py`
413 passed (16 in the new `tests/k/test_fold_imputation.py`,
`tests/dst/test_fold_imputation.py`, `tests/dst/test_native_cv_metadata.py`),
including invariance of a 2023-origin train/validation matrix to held-out
values and a positive control in which perturbing training-season values moves
the imputed training rows. No-fit identity: `provide_dataset(CONFIG)` and
`_prepare_position_data` under `run()`'s config hash identically on this branch
and `origin/main` `12da0f92` for K (frames and matrices) and for D/ST matrices
and targets; the D/ST frames differ only by the added `season_type` column.
The K and DST production benchmark (CPU eager, seed 42) reproduces main's saved
model artifacts and benchmark `results` bit-identically. CV / rolling-origin
numbers for K and DST move by design and are not comparable with earlier
`benchmark_history` CV entries.

**Lesson**: A "train-only" statistic computed in the loader is still leakage
once a fold slices earlier than the loader's notion of train. Fit
preprocessing at the boundary that sees the population actually used for
training, and prove production neutrality without fitting (frame hashes)
before spending a benchmark on it.
