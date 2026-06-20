# Whole-codebase `/simplify` sweep — findings report

**Date:** 2026-06-20 · **Scope:** entire repo (`src/` 65.3K LOC/187 files, `tests/`
68.6K LOC/271 files, 22 CI workflows, hooks/shell). **Method:** ~42 parallel
read-only Opus/max reviewers over file-disjoint bundles, each applying the
`/simplify` rubric (reuse · simplification · efficiency · altitude) and
self-classifying every finding against the AGENTS.md stop-rules. **Quality only**
— no correctness-bug hunt (that is `/code-review`).

This is the report half of the sweep. It captures **all 193 findings**, tiered by
apply-risk. The companion PR applies the **verified safe subset** (Tier 0 + the
self-contained Tier A); everything retrain-path (Tier B) or behavioral /
stop-rule-adjacent / large-architecture / coverage-shifting (Tier C) is documented
here for sign-off rather than auto-applied.

## Tier model

| Tier | Definition | Disposition |
|---|---|---|
| **0** | dead code / mechanical, provably inert, non-retrain | apply |
| **A** | reuse/dedup **off the metric path, non-retrain** (`serving`, `analysis`, `tuning`, `benchmarking`, `scripts`, `tests`) | apply, gated by `pytest -m unit` + ruff |
| **B** | retrain-firing (`src/shared`, `src/features`, `src/data`) but **numerically inert** | **needs off-box Δ=0 Ridge benchmark before merge** — not applied here |
| **C** | behavioral / metric-moving / stop-rule-adjacent / large-architecture / CI-behavior / per-position-coverage-shift | report-only, sign-off required |

Tally: **3 Tier 0 · 95 Tier A · 34 Tier B · 61 Tier C** (193 total).

### Why Tier B is not auto-applied
Editing `src/shared/**`, `src/features/engineer.py`, or split-affecting
`src/data/*` fires a 6-position GPU retrain. The reviewers verified each Tier B
finding is *intended* numerically inert, but the data-identity tell (identical
deterministic Ridge MAE) **cannot be checked in this environment** — there is no
GPU, and the full pipeline needs `data/splits` + torch/lightgbm. Several Tier B
items also change **module-construction order** (`neural_net.py` `_build_head`,
`_resolve_non_negative`) which must be confirmed to preserve the RNG init-draw
order. So Tier B ships as a **draft + a 1-seed Batch/GPU benchmark issue**, never
on green unit tests alone.

---

## Tier 0 — apply (dead/mechanical, inert, non-retrain)

- **`.claude`/`.codex` hooks re-inline existing `lib.sh` helpers** —
  `ruff-format.sh`/`pre-pr.sh` re-type the jq-resolution loop though
  `claude_find_jq()` exists, and the main-worktree `awk` one-liner is inlined 3×
  though `claude_main_worktree()` exists (`scripts/agent-memory-sync.sh` has a 4th
  copy). Call the helpers. *NB: `agent-memory-sync.sh` + the gh-pr tokenizer are
  regression-pinned (`tests/scripts`, `tests/hooks`) — re-run those.*
- **`src/scripts/__init__.py`** and similar empty package markers — confirmed
  intentional, no-op.
- **`src/tuning/ab_harness.py:368`** — `run_cell` sets `base_cfg` identically in
  both `if`/`else` arms; import the module once (matches `run_group_stacked`).

## Tier A — apply (reuse/dedup off the metric path)

95 findings; the high-value clusters:

### serving (non-retrain; live app maintainability) — ~18
- **`serving/core.py`**: 6-key splits dict literal repeated at `:923`/`:1042`
  (`_build_splits_dict`); 3-line splits-load+scoring-format dup at `:786`/`:1005`
  (`_load_base_splits`); per-format total-MAE 4× if-blocks at `:739` → loop;
  K/DST/`_load_splits` reindex-onto-rows logic 3× at `:939`
  (`_reindex_pos_test_onto_results` — **must preserve the `.copy()` aliasing
  guard**); fingerprint field-update 3× at `:1455` (keep byte order).
- **`serving/routes.py`**: sort-key whitelist + `api_player` weekly dict hardcode
  the 6 pred-prefixes that `serialization._ROW_PRED_PREFIXES` owns
  (`:119`,`:228`); `_series_to_list` dup of `_safe_num` (`:376`);
  `api_position_details` hardcodes the pos list vs imported `_ALL_POSITIONS`
  (`:398`).
- **`serving/serialization.py:75`**: `_PLAYER_ROW_COLS` hand-enumerates 33 cols
  derivable from `_ROW_PRED_PREFIXES × _VALID_SCORING`.
- **`serving/comparison.py:81`**: `_model_blocks_from_results` /
  `_model_reliabilities_from_results` share scaffold → `_per_model_from_results(stat_fn)`.
- **`serving/upcoming_week.py`**: map-then-coalesce 5× (`:224`), sorted-join 3×
  (`:398`), atomic tmp-write+`os.replace` 3× (`:420`).
- **`serving/espn_live.py`**: `fetch_injuries_df`/`fetch_injury_status_map` both
  fetch+parse `/injuries` (`:469`); NaN-sentinel set dup (`:104`).
- **`serving/live_sources.py:85`** worst-per-player min-coalesce dup;
  **`benchmark_history.py:140`** `_extract_per_target` called 2×; **`wiki.py:260`**
  repo-root path arith 3×.

### analysis (non-retrain, operator/diagnostic) — ~22
- **`_feature_stats.py` consolidation**: `_present_numeric`, `_pre_registered_table`,
  `_condition_number` are byte-identical/near-identical across
  `analysis_rb_feature_audit.py` + `analysis_k_feature_audit.py` (+ a third in
  `analysis_feature_audit.py`) and belong in the existing
  `src/analysis/_feature_stats.py`.
- **REG-split loader** `_load_splits`/`_load_split_subset`/`covariate_shift._load_split`
  dup the parquet+`season_type==REG` read → `_read_reg_split`.
- **`cohort_analysis.py`**: `bucket_model_table` reimplements
  `error_analysis.compute_stratum_metrics` (`:233`); stable-cumcount idiom dup
  (`:670`); `best_model` recomputes MAE inline (`:272`).
- **`rmse_gap_decomposition.py`**: `_decompose`/`_culprits` rebuild the per-head
  contribution matrix 2× (`:201`).
- **dead `sys.path.insert`** in `analysis_shap_lgbm.py`, `analysis_dst_rare_dispersion.py`,
  `attn_weekly_accuracy.py` (modules run via `python -m`).
- **expert family**: verbatim `_normalize_keys` (`tier_expert_comparison.py:59`),
  per-call `compute_metrics` import (`expert_uncertainty.py:99`), `_join_actuals`
  dup 3× (`:117`); inline MAE ×10 in `analysis_k_signal_floor.py` → `compute_metrics`.
- **`build_comparison_summary` / `analysis_tabpfn_benchmark`**: `top_n_ids`,
  `rounded_metrics`, `_normalize_join_keys` shared; hardcoded tier set `{all,top12,top30}`.

### tuning (non-retrain) — ~13
- **`tune_lgbm.py:131`** `_physical_core_ids` byte-identical to
  `parallel_train.physical_cores`; **`tune_nn.py:140`** `_ensure_data_from_s3`
  near-verbatim dup of `tune_lgbm`'s → shared `data_bootstrap` helper.
- **`tune_nn.py`**: `_apply_attention_scheduler_overrides` re-hardcodes
  `_SCHEDULER_PARAM_KEYS` (`:1023`); `_run_mps_optimize` 6× `if KEY in environ`
  (`:1402`); two consecutive stacked guards + double import (`:1689`).
- **`feature_selection.py`/`_stage2.py`**: launch-command builders ×4
  (`:304`), `build_*_spec` ×2 (`:433`), per-model MAE|RMSE markdown table ×3
  (`:331`), boto3-client+`collect_results` dup (`:127`).
- **`ablate_attn_arch.py:130`** `KNOWN_FLAG_KEYS` declared as a typo-guard but
  **never referenced** — wire the `assert` or delete; **`ablate_rb_gate.py:284`**
  `fmt_mean_std([single])` prints a misleading `±0.000`; function-body
  `import defaultdict` (`:225`).

### benchmarking / scripts / batch(launch) — ~14
- **`benchmark.py`↔`parallel_train.py`** keep two copies of the history-entry build
  + summary-augmentation block (drifted on `total_wall_sec`) →
  `_finalize_run`/`build_history_entry`/`_summarize_one`; argparse spec dup;
  `_split_cores` re-implements the `core_pool` lease math (`parallel_train.py:121`).
- **`scripts/`**: dead `sys.path.insert` (4 files), `_isnan`→`math.isnan`
  (`batch_size_sweep.py:186`), `promote.py:266` re-fetches the manifest.
- **`batch/launch.py`** (retrain-exempt basename): `_pin` revision-suffix helper
  (`:188`), `is_split` computed once (`:314`), merge-result filter once (`:751`).

### tests (non-retrain, test-only) — ~28
Overwhelmingly **in-position / in-shard helper & fixture consolidation** (safe
because it stays on the same Codecov flag):
- **`tests/_pipeline_e2e_utils.py`** is the canonical E2E home — many files re-roll
  `_build_tiny_cfg`, the chdir+symlink-`data/` dance, and top-N-player split
  slicers that `_top_n_players`/`load_tiny_splits` already provide (qb/rb/wr/te).
- **`tests/shared/`**: `_make_tarball`/`_FakeBody`/`_nosuchkey` byte-dup across the
  two `model_sync` files; the 3-target `TARGETS` constant in 6 files; the CUDA-mock
  idiom that `test_platform_detect._force_cuda` already solves; 4 artifact-dir
  builders → `_write_base_artifacts`. Landing spot: a new `tests/shared/_helpers.py`.
- **`tests/tuning/`** (no `conftest.py` exists): the Batch env-dict extractor (14×),
  `_fake_s3`, `batch_env`, cell-row + `AblationResult` builders, platform-mock →
  new `tests/tuning/conftest.py`.
- **`tests/` app suite**: ~80-LOC model-loader stub block dup, `POSITION_REGISTRY`
  stub 6×, **dead `import app_mod`** in ~12 functions, and ~10 manual Flask-client
  blocks that the existing `client_with_data` fixture already covers.
- **`tests/batch` + `tests/scripts`**: `_FakeS3Producer`/`_write_fake_model_dir`
  ~80 LOC dup; the bash/jq/git hook scaffolding + **PR-matcher case corpora**
  shared by `test_claude_hooks`/`test_codex_hooks`/`test_pr_tokenizer` — consolidate
  the *union* of cases, never drop any.
- **`tests/analysis`**: `pred_{model}_total` test_df builder 5×, expert-loader
  stubs 3× → `make_pred_test_df` + shared expert fixtures.
- **`tests/{k,dst}/test_data_loaders`**: `_no_network` guard 8×, the 2025-branch
  setUp + parquet-write boilerplate.

---

## Tier B — numerically-inert but retrain-firing (DRAFT + off-box Δ=0 required)

34 findings. **Do not merge on green unit tests** — each needs a 1-seed Batch/GPU
run confirming Δ=0.0000 Ridge MAE first. Bundle into ONE PR to amortize the retrain.

### `src/features/engineer.py` → `rolling_agg` (strongest, reviewer-verified value-identical)
Five inline `groupby().transform(x.shift(1).rolling(w,min_periods=1).<agg>())`
blocks are byte-equivalent to the existing `src.shared.feature_build.rolling_agg`:
the rolling mean/std/max/min (`:85`), trend short/long (`:273`), share sums
(`:300`), matchup opp-pts (`:1089`), and opp-defense L5 + schedule (`:1192`/`:1231`).
The reviewer confirmed `rolling_agg`'s body is the identical transform for the same
group keys/shift/min_periods. **Verify:** Δ=0 Ridge after swap. (EWMA at `:262`
has *no* existing helper → Tier C.)

### `src/shared/neural_net.py` — internal dedup (6; **verify construction order**)
`predict_numpy` boilerplate 3× (`:900`), `non_negative` default-resolve 3×
(`:297` → `_resolve_non_negative`), positional-encoding add 3× (`:789`), plain
2-layer head literal 3× (`:845` → `_build_head`), `attention_entropy_loss` 2×
(`:875`), `predict_numpy` opp-guard dup of `forward()` (`:914`). **`_build_head`
and `_resolve_non_negative` change module-construction order** — confirm the RNG
init-draw order is preserved (or the trained weights shift).

### `src/shared/pipeline.py` — internal dedup (5)
`_total` closure verbatim 2× (`:1863`), optional-model pred-attach block ~9×
(`:1892` → `_attach_model_preds`), env-truthy predicate 3× (`:139`), ridge/lgbm
feature-importance plot 4× (`:1999` → figures only), `attn_static_cols` 2×
(`:1804`), ridge-tune-grid build 2× (`:1521`). All inert; the CV per-fold NN block
and `_eval_alpha/enet_cv` merges are higher-blast-radius → Tier C.

### `src/shared/models.py` + `model_sync.py` — multi-target wrapper dedup (6)
`non_negative` resolve 3× (`:521`), per-target clamp 3× (`:838`), JSON round-trip
3× (`:612`), ElasticNet `l1_ratio` validation dup of `_init_common` (`:705`);
`model_sync` S3 preamble 6× (`:514` → `_s3_context`), ThreadPool download fan-out
3× (`:637`).

### `src/data/*` + `src/shared/{eval,utils,rest}` (the remainder)
`redzone_pbp._cached_rz_pbp_is_current` delegate to
`external_sources._cached_parquet_has_columns`; `loader.py` 4× merge+backfill →
`_merge_and_backfill` (preserve per-call keys/fill); `evaluation.py:124`
`compute_target_metrics` calls `predictions_to_fantasy_points` 4× where 2 suffice;
`backtest.py:55` per-week ranking dup of `compute_ranking_metrics`;
`utils.py:182` `cuda_graph_enabled`/`_full` force-off literal set →
`_env_force_off`; `team_box_score`/`weather_features` open-code
`.replace(TEAM_CODE_NORMALIZATION)` 4× → `normalize_schedule_team_codes`;
`feature_cache.py:128` hand-rolled O(n) LRU → `OrderedDict`.

**Rejected as Tier-B churn (correctly):** moving `_env_truthy`/`_env_float` from
`training.py` to `utils.py` — they are **single-definition** (verified), so the
move adds no caller while firing a retrain. This was the draft plan's weak Tier-B
example; the reviewer downgraded it to "leave."

---

## Tier C — report-only (sign-off required)

61 findings. Grouped by why they are *not* auto-applied:

### Stop-rule / behavioral / metric-moving
- **`run_pipeline.py` `run()`/`run_cv()` closures** — the no-closure monkeypatch
  contract (`run_pipeline_factory.py` documents it). Skip.
- **The 16 defensive `.copy()` in `serving/core.py`** (COW aliasing; one mutates
  `.index`) and **`_branch_config` `deepcopy`** in `src/batch/train.py:578`
  (guards the module-cached `get_config()`) — behavioral, keep.
- **`registry.py` attn-kwarg mapping enumerated 3×** — contract-tested served-kwargs
  boundary; consolidating risks the condq `cond_proj` serving-NaN. Human-reviewed
  refactor + regenerated contract test only.
- **`src/shared/training.py`** — every real dedup (loss-dispatch ×3, capturable
  hurdle losses, autocast-context, BN-snapshot, `_forward_batch`/`_graph_inputs`)
  sits on the **CUDA-graph capture / metric path** — fenced.
- **`weather_features` implied-total** + **`feature_build` EWMA/sentinel-impute** —
  not provably inert (row-survivorship / window edge semantics).

### Latent issues surfaced (recommend fixing, but verify)
- **`src/tuning/aggregate_scheduler.py`**: non-atomic JSON write (`:117`, sibling
  uses tmp+`os.replace`) **and** a bare `except Exception` (`:45`) that swallows
  real S3 errors (auth/throttle) where the sibling narrows to missing-key
  `ClientError`. Latent but real.
- **`src/shared/platform_detect.py:105`**: `recommended_cuda_wheel` returns
  `cu128` for Blackwell sm_120 vs the documented `cu130` (AGENTS.md /
  requirements-gpu.txt). Reporting-only field; **verify the wheel index** before
  changing (it is a correctness nit, not a simplify — fires a retrain).
- **`launch_ab.py` exit-code contract drift**: the scheduler launcher raises on
  no-success while the others `sys.exit(1)` on any-failure, and drops the
  `submit_failures` path. Worth unifying (Tier C because it changes operator UX).

### Large-architecture / cross-file (worth doing, but design-level)
- **tuning `launch_common.py`**: the 4 Batch launchers duplicate the ThreadPool
  fan-out + status-bucket+exit + base-container-env (~hundreds of LOC). A shared
  tuning-local module would consolidate (`launch_ablate` already imports
  `launch_ab` helpers, proving the pattern).
- **legacy `ablate_*` → `ablation_runner`**: `ablate_scheduler_type` is the lone
  script bypassing the runner (~150-LOC hand-rolled); `_parse_positions`,
  `_make_cfg`, `_extract_run_payload`, the Ridge-sentinel loop, and the
  argparse+`main()` skeleton are triplicated → shared runner helpers (keep the
  bespoke decision rules; add a unit test pinning the sentinel).
- **ab-signal specs**: the boom-subgroup `metric_fn` (4×), inheritor `metric_fn`
  (3×), the role-inheritance injector (3×, converge on `ab_qb_inheritance._make_injector`),
  and `_inject_wr_signals`/`_inject_te_signals` (byte-identical bar the position)
  → a shared `ab_*_common.py` (leakage-safe semantics preserved).

### CI-behavior (composite-action proposals — never blind-apply)
- EC2-SSM lifecycle dup across `train-ec2`/`ablate-rb-gate`/`retune-lgbm` (~400
  LOC); AWS-creds step (~15 sites, one SHA pin); setup-python+uv (5+); buildx
  dual-cache (2); and the byte-identical splits-race-gate / artifact-freshness /
  benchmark-history-append blocks shared by `train-batch`↔`train-ec2`. Each is a
  composite-action / reusable-workflow extraction → user judges. **`_detect-positions.yml`
  position list is contract-coupled to `scope_positions.py` — excluded.**

### Cross-provider lib.sh (stop-ruled)
- The gh-pr tokenizer + `*_refresh_parent_main`/`*_promote_worktree_splits` parity
  twins and the venv `bin/`+`Scripts/` probe are **deliberately per-provider /
  regression-pinned** — AGENTS.md says do not merge. Report-only.

### Per-position test-coverage trade-offs
- The 5-position `simple_data`/`test_models`/`test_neural_net`/`test_evaluation`
  mirrors and `_patch_shared_pipeline` harness could collapse into `tests/shared`,
  **but that shifts coverage off the per-position Codecov flags (80%/flag)**. Offer
  a single pilot (e.g. one shared-module test) so the owner can judge the trade
  before a wholesale collapse.

---

## Landmines confirmed during the sweep (do NOT touch)
`run()` closures · `serving/core.py` `.copy()` · `batch/train.py` `deepcopy` ·
cross-provider `lib.sh` · `registry` served-kwargs · `ab_feature_screen.py`
(frozen for the #1172 regression pins + PCA-ON vs helper-default-PCA-OFF) ·
`_detect-positions.yml` position list · the default-OFF attention investigation
knobs (`attn_learn_temperature`/`history_dropout`/`swiglu`/`entropy`/`alibi`/`self_layers`)
· `predict_total` (already TODO.md-tracked).
