# Model investigation and validation

<a id="ml-modeling-investigation-method"></a>
<a id="ml-modeling--investigation-method"></a>

## Production path

- **Use the actual affected-position pipeline for NN/feature/loss/target changes,
  including investigations.** Compare its `benchmark_history/` result with the
  baseline; unit tests and CI do not establish metric neutrality. The K refactor
  and QB metric-label regressions shipped on green tests without that comparison.
  Reduced/unregularized proxies (low-tree LightGBM, bare Ridge) can give the wrong
  sign. Use `result["test_df"]` (`pred_{model}_total`) for subgroup predictions.
- Match the production loader's fetch/normalization, selected features,
  `POSITION_CONFIG`, splits and NaN handling. Feature diagnostics must impute
  NaN→0 as production does; `dropna` retained only ~52–59% of rows in #594 and
  biased the sample toward veterans. A raw `nfl_source.*` shim can lag schemas:
  “0 for season Y” was the warning in #588→#592/#593.
- **Equal deterministic Ridge MAE is a clue, not proof of data identity.** Inspect
  actual frames, selected features, split/configuration and fingerprints before
  claiming a no-op. Verify data effects with an A/B: even a test-constant or
  float-noise feature can map to ~−4σ after scaling, and offline split-build
  changes are invisible to a runtime-only code A/B.
- **Check activation preconditions before benchmarking or calling a finding a
  metric mover.** RB/WR/TE `min_games=1` made the filter finding #574/#531 inert.
  Likewise nflverse PBP/weekly already used modern team codes (#808); schedules,
  injuries and rosters were the real legacy-code paths (#728/#971/#1269).
  Rosters 2012–2015 also have ARZ/BLT/CLV/HST/SL gamebook codes beyond the shared
  relocation map. Verify the data/configuration the branch actually receives.
- Check imputation reachability at the function input: upstream fillna/lag/dropna
  can remove every NaN before a proposed fix (#608→#609). **Run a positive control
  on the known-bad case** before claiming a guard works. Choose the right stage:
  built splits drop no-snap rows, while the `rosters` cache retains listed/benched
  players (#611).
- Structural-equivalence can bound extra testing: a default-guarded path inert
  on the non-prevailing device plus one byte-identical position makes repeat runs
  of that same inert path redundant. This does not replace checking all relevant
  callers/configurations or executing changed GPU paths.

## Metrics and subgroups

- Single-seed NN overall MAE is noise. A targeted fix needs the subgroup metric's
  direction across ≥2 seeds (#596's seed-42 win was flat at seed 123).
- Default FP-MAE A/Bs use **3 seeds, mean±std**; increase to 5–8 when the conclusion
  hinges on a delta inside the seed band. Backbone-norm went from
  −0.022±0.019 to +0.007±0.034 at 8 seeds: noise, not a win.
- Compare bias when judging errors across low-scoring subgroups; their lower MAE
  may just reflect skewed targets. MAE deltas within a fixed ablation slice are
  valid. Filter unprojected roster placeholders (no stat line), whose 0.0 scores
  falsely improve expert MAE. Cross-source cohorts follow the section below.

## A/B harness

Use the [existing parallel entrypoints](experiments.md#running-code), not bespoke
sequential loops. The shared [ab_harness](../src/tuning/ab_harness.py) accepts
`run(train,val,test,seed,config)` frame/config injection. Frame injection supports
QB/RB/WR/TE only; K/DST `run(seed,config)` build their own splits, so use config
mutators there (config injection supports all six). Inject onto train/validation,
compare the same seeds on `result["test_df"]`, and keep frame injectors pre-kickoff
and leakage-safe: the Ridge-invariance sentinel cannot detect feature-side leaks
or prove data identity. Post-run result slicing is read-only and needs no retrain.

Both `ab_harness` and the older [ablation_runner](../src/tuning/ablation_runner.py)
share `resolve_jobs` platform autodetection/`FF_AB_JOBS` and per-cell
chdir+symlink-`data/` isolation, so hardcoded `{pos}/outputs` saves cannot overwrite
served artifacts. Reuse `parallel_train`/`core_pool`; cap BLAS and use physical
cores rather than SMT. Results aggregate mean±std and deltas against baseline.
See [harness design](../todo/ab_harness_priority.md) and
[isolation helpers](../tests/_pipeline_e2e_utils.py).

Use `--max-workers 1` on legacy runners for timing-clean ablations. Choose eager
versus stacked specs using [GPU execution stop rules](stop-rules.md#gpu-execution),
which preserve K/DST fallbacks and the deliberately eager ablations. Never compare
stacked and eager arms seed-by-seed. Production device/dtype/graph defaults come
from [platform policy](platform.md#device-and-dtype-policy), not an old run label.

## GPU and Batch validation

- GPU-guarded code is untested by CPU unit tests. Exercise it on Batch before
  merge. Before fanning out a new A/B or feature-screen spec, dry-run then execute
  **one real 1-seed/1-position cell** and confirm its JSON has `ok:true`.
  `--list` and synthetic unit checks only validate grid construction/mutators.
  The #1187→#1212 fan-out failed all six jobs: an all-drop arm left zero features
  for `StandardScaler`, while other arms fell below `ridge_pca_components`.
  On the affected macOS torch/LightGBM/sklearn stack, libomp SIGSEGV prevented a
  local pipeline substitute; a listed grid was not live-pipeline evidence.
- The [Batch execution contract](../docs/adr/0020-batch-gpu-execution-path-for-ab-harness.md#decision)
  covers `launch_ab`, `ab-batch.yml`, `FF_TUNE_AB_SPEC` dispatch and per-cell S3
  checkpoint/resume. For unmerged code, build `batch-image.yml` on the branch
  first; non-main images get only the SHA tag and `launch_ab` clones `ff-ab-job`
  so production job-definition names never point to branch images.
- **`launch_ab --env KEY=VAL` reaches the container, not the local submitter.**
  Env-parametrized specs (`ab_feature_subscreen` / `ab_feature_confirm`) are
  imported locally to size `--max-cells`, write manifests and collect `--wait`
  results. Set the values in **both** places:
  `FF_…=… python -m src.tuning.launch_ab … --env FF_…=…`.
  Otherwise default `pb*` arms may silently replace intended `drop_*` arms and
  corrupt submission/collection. `feature_selection substage` and `confirm`
  already emit the prefix; verify hand-edited commands with `--dry-run` and the
  intended variant list.

## Investigation details

- `[timing] phase=X` followed by silence does not locate a hang in X+1; CPU/GPU
  branches run concurrently, so inspect matching CPU-branch logs first.
- The attention NN uses learned-query pooling (`AttentionPool`); inspect
  `attn_self_layers` before describing it as a transformer. The dormant
  `SelfAttentionBlock` is not enabled by any production position configuration.
- Do not close over pipeline functions in a factory: tests monkeypatch
  `src.{pos}.run_pipeline.run_pipeline`; define `run()` locally per position.
- An abstraction's foundation PR can add LOC; only its migration PR saves them.
  Do not promise the foundation alone is a net reduction.

## Evaluation cohorts

**Evaluation cohorts (ADR-0024):** compare sources on identical regular-season player-weeks and the same projected scoring components in forecasts AND actuals (owner clarification, 2026-09-10). Use `src/shared/comparison_scoring.py`; missing component data is unavailable, never a full-fantasy fallback. NFL.com K is excluded from matched comparisons; the K reference uses ESPN. `weekly_reference_top24` uses the versioned archived pregame reference; `elite_top24` retains prior-season importance. Actual weekly leaders are for ranking, and actual seasonal leaders are retrospective. Never use a model's own top-N pool for cross-source MAE comparison, restore the static expert summary as live accuracy, or silently omit missing cohort data from serialized Batch/local results.
