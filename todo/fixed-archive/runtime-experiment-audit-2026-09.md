### [FIXED] Experiments lost seeds, native frames, matched controls and scoring identity

**File(s)**: `src/tuning/` experiment entrypoints (`ab_air_yards.py`,
`ab_boom_signals_{wr,te}.py`, `ab_rolling_origin_rotowire.py`, `ab_opp_def.py`,
`ablate_backbone_norm.py`, `ablate_injury_features.py`, `ablate_rb_gate.py`,
`ablate_ridge_pca.py`, `ablate_scheduler_type.py`, `aggregate_results.py`,
`aggregate_scheduler.py`, `attn_knob_experiments.py`, `feature_groups.py`,
`feature_selection.py`, `warmstart_walkforward.py`, `launch_ab.py`,
`launch_tune.py`, `launch_ablate_scheduler.py`, `resource_probe.py`,
`tune_lgbm.py`), the stacked-seed path (`ab_ensemble_seeds.py`,
`ab_harness.py`, `tune_nn.py`, `tune_nn_storage.py`),
`src/benchmarking/parallel_train.py`, `src/artifacts/model_sync.py` +
`src/batch/launch.py`, `src/shared/utils.py`, `src/shared/training.py`,
`src/shared/error_analysis.py`, `src/shared/position_data.py`,
`src/rb/analyze_errors.py` and the offensive `targets.py` decomposition
diagnostics. Defects reproduced against `92be2873` and `0f0fec55` during the
2026-09-10 audit; this slice was extracted from #1565.

**What**:

- Leave-one-group-out effects used other dropped groups instead of the
  same-seed baseline. Multi-position normalization reports kept only the first
  position's targets; filtered seed lists were paired by order rather than by
  seed; several tables collapsed a measured seed std to a single value.
- Scheduler-type ablations crashed on subsets without the production
  comparator, pooled the wrong comparator variance, and reported a partial
  fleet run as success. The aggregate Markdown separator had one column too few.
- QB air-yards cohorts selected the rare receiving-TD games instead of passing
  TDs. RB error analysis charged unprojected passing points as errors. Fixed
  volatility quartile labels conflicted with duplicate quantile edges.
- Two-point conversions were backed out of the target-decomposition check even
  though the canonical `fantasy_points` (recomputed with `SCORING`) never
  contains them, so a genuinely corrupt row could hide behind the adjustment.
- Ridge-PCA validation scoring supplied the validation rows as both val and
  test, duplicating them inside cross-split history features; K/DST called a
  self-loading runner signature that cannot accept frames.
- Warm-start walk-forward reports hid failed arms and compared unpaired seeds.
  The Stage-1 plan printed a seed count that did not match the stacked seed
  list; `--only` could not take variant names starting with `-`.
- Concurrent stacked tuner trials installed process-global capture stubs and
  could steal each other's trainers or retain hooks after failure. Four stacked
  entrypoints discarded the captured trainer's device, and read-only best-study
  lookups resolved a graph namespace stacked training never wrote. Explicit
  Apple MPS runs did not seed the accelerator RNG; stacked MPS training failed
  deep inside vmap instead of at entry.
- A training loader with fewer rows than one dropped-tail batch reported zero
  training loss and saved an untouched model; captured stacked/sequential loops
  could report a finite Optuna objective from untouched weights.
- Rolling-origin benchmark workers for one position shared the production
  `{pos}/outputs` model/scaler paths; the affinity hook crashed on platforms
  without `sched_setaffinity`. Tarball extraction could leave members from a
  rejected archive in the model directory that a later fallback then served.
- `launch_tune` printed one storage namespace for positions that resolve to
  different ones (eager K/DST vs stacked skill positions) and omitted
  `FF_TUNE_STACKED_SEEDS=0`, re-enabling the container's stacked default. The
  tuned-LightGBM holdout comparison rebuilt the tuned model without its fixed
  objective.

**Fix**: Compare each drop against the same-seed baseline; report every
position; pair deltas by seed and print observed std. Require the production
comparator and an alternative before a scheduler verdict; fail incomplete fleet
runs. Select the position's TD cohort, score RB actuals from projected
components, and bucket tied volatility values together. Validate decomposition
against the canonical scoring contract. Supply validation rows once to
Ridge-only scoring and route K/DST through `prepare_native_ablation` (native
frames plus the K kick-history closure, in `src/shared/position_data.py`) into
`run_pipeline`. Retain warm-start failures and compare matched seed/origin
pairs. Dispatch capture hooks by thread and nesting depth, keep the captured
device, resolve stacked lookup namespaces identically to training under
`scheduler_v3`/`history_v3`, seed the selected MPS generator and reject MPS
stacking at entry. Reject empty or exhausted training epochs in both the
ordinary trainer and the captured loops. Isolate rolling-origin origins in a
private output root and skip affinity where unsupported. Stage tarball
extraction and replace the destination only after a complete, validated
extract. Resolve one namespace per position, pass `FF_TUNE_STACKED_SEEDS`
explicitly, and keep the tuned comparison's objective.

**Validation**: Unit suites for the touched tuning, shared, batch and
diagnostic paths pass locally. The RB production run (CPU eager, seed 42) on
this branch is bit-identical to `origin/main` (models, scalers and benchmark
results). MPS seeding only executes under opt-in `FF_DEVICE=mps`; the
empty-training guard only fires on loaders that yield no batch.

**Lesson**: A correctly shaped report can still describe a different seed,
dataset, device or comparison. Follow the real caller and preserve identity
through preparation, capture, caching and aggregation.

Sample-weighted validation loss, count-likelihood precision, LightGBM bagging,
WR/TE stint resets, K/DST fold imputation and reporting truth are tracked in
separate PRs.
