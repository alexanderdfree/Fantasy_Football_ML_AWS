# Code Review Findings

**Original audit:** 2026-05-20 · `claude/admiring-rubin-87b3cd` · `bd80ba1`
**Re-verified at:** 2026-05-20 HEAD of `claude/kind-kare-bd547f` (19 commits ahead, PRs #259, #262, #263, #264, #267, #268, #270, #271 merged since)
**Wave 1 shipped:** 2026-05-21 — H1 ([#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287)) and H2 ([#290](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/290)) merged.

## Summary

- **133 findings originally reported** · **9 false positives / already-fixed pruned on re-verification** · **2 HIGH shipped 2026-05-21** · **~122 persisting**
- **0 HIGH severity persisting** — H1 fixed ([#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287)), H2 fixed ([#290](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/290)), H3 was already fixed at re-verification
- **17 MEDIUM severity** — was 18, M14 pruned
- **~105 LOW severity** — was 112, L-WR9, L-TE1, L-TE5, L-DST5, L-S2, L-S4, L-E2 pruned

### Re-verification deltas
- **H3 (gunicorn `--preload` S3 syncs)** → ALREADY FIXED: syncs moved to `gunicorn.conf.py::on_starting`.
- **M14 (`LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS` invariant test missing)** → FALSE POSITIVE: `test_loss_weights_match_huber_deltas` exists at [tests/test_invariants.py:46](tests/test_invariants.py:46).
- **L-WR9, L-TE1, L-TE5, L-DST5** → FALSE POSITIVE on close reading (comments scope-correct or not stale as claimed).
- **L-S2 (LastWeekBaseline dead)** → ALREADY FIXED: `src/models/` deleted by PR #263.
- **L-S4 (legacy Trainer dead)** → ALREADY FIXED: `src/training/` deleted by PR #264.
- **L-E2 (`print_comparison_table` no callers)** → FALSE POSITIVE: has production callers at [src/shared/pipeline.py:1408](src/shared/pipeline.py:1408), [:1861](src/shared/pipeline.py:1861).

### Wave 1 shipped (2026-05-21)
1. **H1** → FIXED in [#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287) (`2eb4f43`). Removed the `if pos != "K"` guard so K's `aggregate_fn` is now plumbed through `build_pipeline_config`. Local benchmark: K ranking metrics improved across the board — `ridge_top12` 0.468→0.505, `nn_top12` 0.477→0.523, `attn_nn_top12` 0.403→0.514, `lgbm_top12` 0.449→0.468. Pre-fix the ranking was sorting kickers by "more misses = more points". TODO.md archive entry added.
2. **H2** → FIXED in [#290](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/290) (`6618243`). Chose Option B (architectural fix in `src/shared/weather_features.py`): added `spread_line`/`div_game` to the upfront drop list and removed them from the cleanup-drop loop. New regression test in `tests/shared/test_weather_features.py` asserts both columns survive with `std() > 0` and `unique() ⊇ {0, 1}`. Local benchmark: DST Attn NN total MAE 5.288→5.113 (-0.175), `points_allowed` Attn NN MAE now 7.342 (beats NN 7.657). TODO.md archive entry added.

---

## High severity

> **Both H1 and H2 shipped on 2026-05-21.** Original descriptions preserved below for historical record; see "Wave 1 shipped" above for fix details and metric deltas.

### H1. K aggregate_fn skipped → training-time totals add miss penalties as positive [FIXED — [#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287) `2eb4f43`]
- **File:** [src/shared/position_pipeline.py:262-265](src/shared/position_pipeline.py:262)
- **Type:** bug
- **Description:** `build_pipeline_config` only sets `cfg["aggregate_fn"]` when `pos != "K"`. In [src/shared/pipeline.py:1337-1338](src/shared/pipeline.py:1337) the `_total(preds)` helper falls back to `sum(preds[t] for t in targets)` when no aggregator is registered. K's four targets are `[fg_yard_points, pat_points, fg_misses, xp_misses]` with `target_signs` `[+1, +1, -1, -1]` ([src/k/config.py:222-227](src/k/config.py:222)). The naive sum **adds** `fg_misses` and `xp_misses` instead of subtracting them, inflating every K `pred_*_total`. Corrupts `pred_baseline`, all `pred_*_total` columns (pipeline.py:1341-1347, 1365, 1374, 1383), `compute_ranking_metrics` ranking output, the weekly backtest summary, and the `benchmark_history/{run_id}.json` K row. **Serving is unaffected** — `src/serving/app.py:599-600` already uses `target_signs`. The shared `_k_predictions_to_fantasy_points` aggregator in `src/shared/aggregate_targets.py:205` already exists and `aggregate_fn_for("K")` would route through it — only the `if pos != "K"` skip prevents it.
- **Same root cause in CV path:** [src/shared/pipeline.py:1789-1792](src/shared/pipeline.py:1789) mirrors the bug. K has `has_cv_runner=False` so currently dormant, but fixing the factory fixes both for free.
- **Recommended fix:** remove `if pos != "K"` at position_pipeline.py:264 (or delete that line so the conditional sets unconditionally). Add a regression test in `tests/k/test_pipeline_e2e.py` that asserts `pred_nn_total[i] ≈ fg_yard_points + pat_points - fg_misses - xp_misses`.

### H2. DST `spread_line` and `div_game` silently zeroed by `merge_schedule_features` [FIXED — [#290](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/290) `6618243`]
- **File:** [src/shared/weather_features.py:140-227](src/shared/weather_features.py:140) interacting with [src/dst/data.py:340-360](src/dst/data.py:340) and [src/dst/config.py:65,77,266,269](src/dst/config.py:65) (DST's `_ALL_FEATURES` and `attn_static_features` both include these).
- **Type:** bug (training/inference drift class; both paths affected)
- **Description:**
  1. `src/dst/data.py` materializes `spread_line` and `div_game` directly from the schedule on every DST row.
  2. `merge_schedule_features` drops `WEATHER_FEATURES_ALL + ["implied_total_x_dome"]` first — but `WEATHER_FEATURES_ALL` (defined at `weather_features.py:17-29`) does NOT include `spread_line` or `div_game`. So both columns survive the drop and remain on `df`.
  3. The merge at line 149 (`df.merge(lookup, ...)`) has both sides carrying `spread_line` and `div_game`. Pandas suffix-renames to `spread_line_x`/`_y` and `div_game_x`/`_y`.
  4. The merge-back loop at lines 169-171 reads bare-name columns only; `if col in df_merged.columns: df[col] = df_merged[col].values` silently skips them.
  5. The cleanup loop at lines 215-227 unconditionally drops bare `spread_line` and `div_game` from `df`.
  6. End state: both columns are gone. Downstream, `build_position_features` catch-all backfill fills them with 0 and prints `WARNING: 2 feature columns missing, filling with 0`.
- **Impact:** DST trains and serves with `spread_line = 0` and `div_game = 0` for every row. Two genuinely-predictive features (point spread, divisional-game flag) are dead. `is_divisional` is recomputed correctly from the original `div_game` BEFORE the cleanup (line 186), so that one survives.
- **Why no test catches it:** [tests/test_invariants.py:115-143](tests/test_invariants.py:115) asserts `attn_static_features ⊆ all_features` (config consistency), not that values survive the pipeline (see Finding M3).
- **Recommended fix:** Either drop `spread_line` and `div_game` from DST's `data.py` so they get populated cleanly from the schedule merge, or add them to the `merge_cols` AND keep them out of the cleanup-drop loop. Add a runtime invariant test that runs DST through `build_position_features` and asserts `spread_line.std() > 0` and `div_game.unique() ⊇ {0, 1}`.

---

## Medium severity

### M1. `nn_non_negative_targets` doc-vs-code drift, with three inconsistent registry call sites [VERIFIED — corroborated by 3 workers]
- **Files:** [CLAUDE.md](CLAUDE.md) ("non_negative_targets is per-head, not global" section) · [src/shared/position_config.py:139](src/shared/position_config.py:139) · [src/shared/registry.py:104-105, 124, 155](src/shared/registry.py:104) · all 6 of [src/{qb,rb,wr,te,k,dst}/config.py](src/qb/config.py)
- **Type:** doc-rot + latent footgun + dead conditional
- **Description:** CLAUDE.md states *"K and DST set `NN_NON_NEGATIVE_TARGETS = set(TARGETS)` explicitly; QB/RB/WR/TE rely on the default `None`."* This is no longer true — all six positions now set `nn_non_negative_targets=set(_TARGETS)` explicitly (qb:141, rb:215, wr:163, te:146, k:168, dst:173). Functionally identical to the default-None semantics in `MultiHeadNet` (which clamps every head when arg is `None`), but the doc lies about the code shape. Additionally:
  - `PositionConfig.nn_non_negative_targets: set[str] = field(default_factory=set)` — the dataclass default is `set()` (empty), NOT `None`. A new position that forgets to set this field would silently disable non-negativity clamping for every head (registry pass-through interprets non-`None` as "use what was provided").
  - `src/shared/registry.py:104-105` has `if pc.nn_non_negative_targets is not None: kwargs["non_negative_targets"] = ...` — UNREACHABLE because the field can never be `None` under the current dataclass default.
  - `registry.py:124` and `:155` pass it unconditionally. Three call sites disagree on guard style for the same value.
- **Recommended fix:** Either (a) change `field(default_factory=set)` to `default=None` with type `set[str] | None`, drop the explicit setters in all six configs, restore the CLAUDE.md claim — OR (b) keep explicit-everywhere, delete the dead `is not None` check in registry.py:104, and update CLAUDE.md to acknowledge "all six positions set this explicitly."

### M2. K schedule-merge work partially wasted + silently overwritten by shared merge [VERIFIED]
- **File:** [src/k/data.py:330-394](src/k/data.py:330) vs [src/shared/feature_build.py:30-60](src/shared/feature_build.py:30) + [src/shared/weather_features.py:137-229](src/shared/weather_features.py:137)
- **Type:** drift / dead code
- **Description:** K's `load_data` performs a schedule merge populating `is_home`, `total_line`, `implied_team_total`, `roof`, `surface`, `is_dome`. Later, both training (`src/shared/pipeline.py:504`) and serving (`src/serving/app.py:573`) call `build_position_features` → `merge_schedule_features`, which (a) skips iff `_schedule_merged` column is present, and (b) drops + re-populates `is_dome`, `implied_team_total`, `total_line`. K never sets `_schedule_merged=True`, so the shared merge runs and overwrites. Mostly wasted work; whatever survives is the small set of PBP-derived columns (`game_wind`, `game_temp` for XP-only fallback).
- **Recommended fix:** Set `df["_schedule_merged"] = True` at the end of K's schedule merge (the shared call becomes a no-op), or remove the redundant K-side logic.

### M3. `test_attn_static_features_subset_of_include` doesn't exercise the actual feature-build pipeline [VERIFIED]
- **File:** [tests/test_invariants.py:115-143](tests/test_invariants.py:115)
- **Type:** test-gap (would have caught H2)
- **Description:** The invariant verifies `attn_static_features ⊆ include_features/all_features` (config consistency). It never runs `build_position_features` and never asserts that listed columns actually arrive at the model with non-zero variance. H2 (DST spread_line/div_game zeroed) passes this test because both columns ARE in DST's config — the bug is at pipeline runtime.
- **Recommended fix:** Add a runtime invariant test per position that (a) runs `build_position_features` against tiny real data, (b) asserts every `attn_static_feature` column exists, (c) asserts `.std() > 0` for non-constant ones.

### M4. PR #258 CodeQL injection hardening missed two workflows [VERIFIED]
- **Files:** [.github/workflows/retune-lgbm.yml:95-97, 123](.github/workflows/retune-lgbm.yml:95) · [.github/workflows/ablate-rb-gate.yml:99](.github/workflows/ablate-rb-gate.yml:99)
- **Type:** unfinished-PR-artifact
- **Description:** PR #258 moved `${{ github.event.inputs.* }}` interpolation into `env:` indirection across `_detect-positions.yml`, `train-batch.yml`, and `train-ec2.yml`, and added top-level `permissions: contents: read`. The same pattern is missed in two workflows that interpolate `${{ github.event.inputs.* }}` directly into shell:
  - `retune-lgbm.yml:95-97`: `POSITIONS="${{ github.event.inputs.positions || 'QB RB WR TE' }}"` etc., then `$POSITIONS` flows into a `jq -n --arg cmd "...$POSITIONS..."` block.
  - `ablate-rb-gate.yml:99`: `SEED="${{ github.event.inputs.seed || '42' }}"`.
- **Recommended fix:** Apply the same `env:`-indirection pattern PR #258 used for the other three workflows. Add top-level `permissions: contents: read`.

### M5. `_run_rb_gate_ablation` in `src/batch/train.py` is a stale fork of `src/tuning/ablate_rb_gate.py` [VERIFIED]
- **File:** [src/batch/train.py:479-598](src/batch/train.py:479) vs [src/tuning/ablate_rb_gate.py:44-148](src/tuning/ablate_rb_gate.py:44)
- **Type:** drift / dead-code with footgun shape
- **Description:** The in-container ablation in `train.py` knows only 3 variants (A/B/C) with the old "≥ 0.05 pt/game keep-gate" rule. `ablate_rb_gate.py` has since grown to six variants (A/B/C/D/E/Bf) with a different decision logic ("lowest sum of per-target MAEs"). `.github/workflows/ablate-rb-gate.yml:125` triggers the train.py (stale) path; running `python -m src.tuning.ablate_rb_gate` locally runs the current path. Two sources of truth for the same ablation.
- **Recommended fix:** Have `src/batch/train.py::_run_rb_gate_ablation` call into `src/tuning/ablate_rb_gate.py`'s VARIANTS + decision logic, deleting the local copy. Or delete the train.py path and route the workflow at `ablate-rb-gate.yml:125` to `python -m src.tuning.ablate_rb_gate`.

### M6. `_tune_lgbm._format_config_lines` paste hint targets a non-existent file format [VERIFIED]
- **File:** [src/tuning/tune_lgbm.py:329-356, 487](src/tuning/tune_lgbm.py:329) + [.github/workflows/retune-lgbm.yml:204](.github/workflows/retune-lgbm.yml:204)
- **Type:** doc-rot / unfinished-PR-artifact
- **Description:** `_format_config_lines` emits uppercase constants `QB_LGBM_N_ESTIMATORS = 1500` and tells operators to paste them into `{pos}/{pos.lower()}_config.py`. Both are wrong: configs live at `src/{pos.lower()}/config.py`, not `{POS}/`, and they use lowercase dict-style kwargs `lgbm_n_estimators=1500` inside `CONFIG = build_position_config(...)`. There are zero `*_LGBM_*` constants in the codebase. The workflow comment at retune-lgbm.yml:204 still references a "PR 3b" that would consume the structured output — apparently never shipped, or replaced by manual paste.
- **Recommended fix:** Update `_format_config_lines` to emit the lowercase kwarg form matching `build_position_config()`. Drop the "PR 3b" references or rewrite to describe the current manual-paste flow.

### M7. K per-kick PBP cache has no schema/freshness check [VERIFIED]
- **File:** [src/k/data.py:564-566](src/k/data.py:564)
- **Type:** bug (latent)
- **Description:** `reconstruct_kicker_kicks_from_pbp` returns the cached parquet unconditionally if the file exists. The weekly cache has a `_cached_pbp_is_current` guard at line 57-70, but the per-kick cache does not, and there's no `play_id`-column presence check. A stale cache from before `play_id` was added would silently propagate, and `build_nested_kick_history` would fall back to insertion-order sorting — defeating the deterministic `play_id` ordering that `test_inner_truncation_uses_play_id_when_present` validates. Mid-season 2025 data also can't invalidate the cache (key is `seasons[0]_seasons[-1]`).
- **Recommended fix:** Mirror the weekly-cache pattern: add a `_cached_kick_pbp_is_current` (or reuse) that validates schema (incl. `play_id`) and modification time vs. upstream parquets.

### M8. `build_and_push.sh` references pre-migration Dockerfile path [VERIFIED]
- **File:** [src/batch/build_and_push.sh:53](src/batch/build_and_push.sh:53)
- **Type:** bug
- **Description:** `docker build … -f batch/Dockerfile.train ...` was written before the src/ migration. The actual file lives at `src/batch/Dockerfile.train`. Running the script from repo root fails immediately with `unable to prepare context: ... batch/Dockerfile.train: no such file or directory`. The prod path (`batch-image.yml:96`) uses the correct `--file src/batch/Dockerfile.train`, so this only burns operators running `build_and_push.sh` locally.
- **Recommended fix:** Update line 53 to `-f src/batch/Dockerfile.train \`.

### M9. `REQUIRED_PIPELINE_CFG_KEYS` validator misses three keys `feature_build.py` reads unconditionally [VERIFIED]
- **File:** [src/shared/position_pipeline.py:41-67](src/shared/position_pipeline.py:41) vs [src/shared/feature_build.py:61-69](src/shared/feature_build.py:61)
- **Type:** test-gap
- **Description:** `REQUIRED_PIPELINE_CFG_KEYS` doesn't include `specific_features`, `add_features_fn`, or `fill_nans_fn`, but `build_position_features` reads all three with `cfg["..."]` (not `.get`). The factory always sets them, so the validator never fires in normal use — but the validator's docstring claims to "catch missing required keys at construction time" and a hand-built cfg would crash inside `feature_build.py` with the trace the validator was designed to prevent.
- **Recommended fix:** Add the three missing keys to `REQUIRED_PIPELINE_CFG_KEYS`, plus a test that constructs a minimal cfg omitting each.

### M10. K smoke test never exercises nested attention with `attn_history_stats` populated [VERIFIED]
- **File:** [tests/shared/test_smoke_test.py:328-394](tests/shared/test_smoke_test.py:328)
- **Type:** test-gap
- **Description:** `_fake_nested_attn_reg` doesn't populate `attn_history_stats`. So the nested branch in `_smoke_attention` ([src/shared/smoke_test.py:138-141](src/shared/smoke_test.py:138)) — the `if game_history_stats:` block — never executes under test, even though K's real config has `attn_history_stats=_ATTN_HISTORY_STATS` populated. A bug in the per-game-aggregates tensor construction would slip through CI and fire at promotion time in `src/batch/train.py:358`.
- **Recommended fix:** Add a nested-attn smoke test that uses K's real `POSITION_CONFIG` and a tiny synthetic dataset including kick PBP, asserting `_smoke_attention` returns without exception.

### M11. Smoke test suite never uses a real `POSITION_CONFIG` from the registry [VERIFIED]
- **File:** [tests/shared/test_smoke_test.py:36-394](tests/shared/test_smoke_test.py:36)
- **Type:** test-gap
- **Description:** Every test uses a synthetic `"TST"` position and hand-built `_fake_reg(...)` registry. None wire a real `POSITION_CONFIG` through `src/shared/registry.get_inference_spec(pos)` for any of QB/RB/WR/TE/K/DST and run `run_smoke_test` against it. A change that breaks the contract between `get_inference_spec` and `run_smoke_test` (renaming `attn_nn_kwargs_static`, adding a key only K provides, etc.) would land green here and blow up at promotion time.
- **Recommended fix:** Add `@pytest.mark.parametrize("pos", ["QB","RB","WR","TE","K","DST"])` that loads each real `POSITION_CONFIG`, materializes a tiny dataset, and runs `run_smoke_test(pos)`.

### M12. DST opp-OFFENSE attention branch never exercised end-to-end [VERIFIED]
- **File:** [tests/dst/test_pipeline_e2e.py:265-296](tests/dst/test_pipeline_e2e.py:265) + [tests/dst/conftest.py:194-242](tests/dst/conftest.py:194)
- **Type:** test-gap
- **Description:** DST is the only position whose parallel attention branch attends over the opposing OFFENSE (`opp_attn_kind="offense"`). The E2E attention smoke flips on `train_attention_nn=True` but doesn't set `opp_attn_history_stats`/`opp_attn_max_seq_len`/`opp_attn_kind` on the cfg, and the synthetic tiny dataset doesn't materialize `off_*` columns. A contract-only test (`test_opp_attn_kind_is_offense` at feature_contract.py:77-85) checks the config says `"offense"` but no test wires the offense-side branch through `run_pipeline`.
- **Recommended fix:** Extend the synthetic DST fixture to materialize `off_pass_yards`, `off_rush_yards`, `off_pts_scored`, etc., and add an opp-offense-branch E2E case.

### M13. RB e2e test bypasses `build_pipeline_config` dispatch [VERIFIED]
- **File:** [tests/rb/test_pipeline_e2e.py:30-79](tests/rb/test_pipeline_e2e.py:30)
- **Type:** drift risk
- **Description:** `_build_tiny_config` reads fields directly off `POSITION_CONFIG` and hardcodes `"two_stage_targets": {}` and `"classification_targets": pc.gated_ordinal_targets`, bypassing `_rb_classification_targets`/`_rb_two_stage_targets`. Any new field added to `build_pipeline_config` is silently absent from this e2e until manually added. The factory was introduced specifically to centralize this assembly.
- **Recommended fix:** Replace `_build_tiny_config` with a call to `build_pipeline_config("RB", POSITION_CONFIG, overrides=_TINY_OVERRIDES)`, matching the QB pattern.

### M15. Predictions cache fingerprint misses every raw data file except kicker PBP [VERIFIED]
- **File:** [src/serving/app.py:1145-1161](src/serving/app.py:1145)
- **Type:** bug (latent — depends on operational refresh cadence)
- **Description:** `_iter_fingerprint_paths` walks each position's model dir + `data/splits/*.parquet`, then only emits `data/raw/kicker_kicks_pbp_*.parquet`. Serving inference reads many other raw caches at request time: `weekly_*.parquet`, `schedules_*.parquet`, and (indirectly via `build_position_features`) `team_stats_*`, `injuries_*`, `depth_charts_*`, `snap_counts_*`, `rosters_*`. `sync_data_from_s3` pulls all `data/raw/*.parquet` matching the prefix, so any of these can be updated at container boot. With the current fingerprint, a refreshed `weekly_*.parquet` or `schedules_*.parquet` does NOT invalidate the predictions cache — next container hydrates stale predictions.
- **Recommended fix:** Either widen `_iter_fingerprint_paths` to all `data/raw/*.parquet`, or — if the operational story is "raw refreshes always coincide with retraining" — document that explicitly in the fingerprint function's docstring (currently misleading: "Any change to a trained model or to the test split invalidates the cache automatically.").

### M16. Disk-hydrate path doesn't restore `position_details` or `position_load_errors` [VERIFIED]
- **File:** [src/serving/app.py:1221-1227](src/serving/app.py:1221) + [:1241-1259](src/serving/app.py:1241) + [:891-895](src/serving/app.py:891)
- **Type:** bug
- **Description:** `_try_hydrate_from_disk` only restores `results`, `metrics_by_format`, `metrics`, `positions_loaded`, `base_loaded`. `position_details` (used by `/api/position_details`) and `position_load_errors` (used by `/api/predictions::degraded_positions` and `/health`) are not restored. The endpoints handle missing keys defensively, so a hydrated container reports empty `target_metrics` per position and reports `degraded_positions: []` + `/health: ok` even if the original run had per-model failures. The disk cache is the steady-state hot path (every ECS task replacement uses it), so the per-target MAE breakdown is effectively never shown in prod.
- **Recommended fix:** Persist `position_details` + `position_load_errors` alongside `metrics.json` (or fold them into the same file), and restore them in `_try_hydrate_from_disk`.

### M17. `sync_data_from_s3` has no per-file failure isolation (asymmetric with PR #236 model-sync fix) [VERIFIED]
- **File:** [src/shared/model_sync.py:411-415](src/shared/model_sync.py:411) vs [:270-285](src/shared/model_sync.py:270)
- **Type:** inconsistency
- **Description:** PR #236 added per-position failure isolation to `sync_models_from_s3` (catches per-future exceptions, returns `failed_positions`). Sibling `sync_data_from_s3` and `sync_benchmark_history_from_s3:369-370` still use bare `f.result()` in a list comprehension — a single broken raw parquet kills the whole sync and the container won't start. May be intentional ("data is shared infra, one bad split = inference broken anyway"), but worth a one-line docstring note explaining the divergence so a future maintainer doesn't blindly mirror #236 down here.
- **Recommended fix:** Either extend per-file isolation to both functions (and surface failed files in the return tuple), or document the divergence in the docstrings.

### M18. WR `benchmark_ridge_variants.py` table prints stale column headers vs data [VERIFIED]
- **File:** [src/wr/benchmark_ridge_variants.py:235-255](src/wr/benchmark_ridge_variants.py:235)
- **Type:** bug (diagnostic output only — not on training path)
- **Description:** Header columns are `recv_fl`, `rush_fl`, `td_pts`; data row prints `m['receiving_tds']['mae']`, `m['receiving_yards']['mae']`, `m['receptions']['mae']`. Labels and data completely mismatched — a relic from the pre-raw-stats era when targets were receiving/rushing fantasy losses + TD points. `fumbles_lost` also silently omitted (4 targets, 3 columns).
- **Recommended fix:** Replace header literals with `TARGETS` iteration, and print all four MAE columns.

---

## Low severity

### Per-position findings

#### QB
- **L-QB1.** [src/qb/diagnose_outliers.py:128-132](src/qb/diagnose_outliers.py:128) — `MultiTargetLoss` constructed without `head_losses=cfg["head_losses"]`. Defaults unspecified targets to `"huber"`, so QB's Poisson NLL heads (`passing_tds`, `rushing_tds`, `interceptions`, `fumbles_lost`) are silently re-cast under Huber with the production weight (1.0) — diagnostic doesn't characterize the production model. [VERIFIED]
- **L-QB2.** [tests/qb/test_pipeline_e2e.py:101-130](tests/qb/test_pipeline_e2e.py:101) + [tests/qb/test_run_cv_pipeline.py:34](tests/qb/test_run_cv_pipeline.py:34) — local `_tiny_config` diverges from canonical `tests/_pipeline_e2e_utils.py::build_tiny_config` (`nn_backbone_layers=[8]` vs `[8, 8]`). Two definitions of "tiny QB config" in use. [VERIFIED]
- **L-QB3.** [src/qb/diagnose_outliers.py:36-43, 81-149](src/qb/diagnose_outliers.py:36) — reimplements the NN training loop instead of calling `src.shared.pipeline._train_nn`. Drift risk on every future training-side change. [VERIFIED]
- **L-QB4.** [tests/qb/test_features.py:117-128](tests/qb/test_features.py:117) — `zero_input_features["attempts"]` omits `sack_rate_L3`, `passing_epa_per_dropback_L3`, `sack_damage_per_dropback_L3` from the 0/0 division contract. [VERIFIED]
- **L-QB5.** [tests/qb/test_run_pipeline_main.py:42-49](tests/qb/test_run_pipeline_main.py:42) — `test_main_default_invokes_run_pipeline` doesn't assert `args[-1] == seed`; sibling tests do. Regression on the wire format would slip through this higher-coverage path. [VERIFIED]
- **L-QB6.** [src/qb/data.py:1-11](src/qb/data.py:1) — no comment explaining QB intentionally has no team-totals aggregator (RB/WR/TE do); future contributor copy-pasting from `src/rb/data.py` may add one. [VERIFIED]
- **L-QB7.** [src/qb/diagnose_outliers.py:46-48, 467](src/qb/diagnose_outliers.py:46) — `OUTPUT_DIR = "analysis_output"` is cwd-relative + unversioned, so two runs overwrite each other. [VERIFIED]
- **L-QB8.** [tests/qb/test_run_pipeline_main.py:79-83](tests/qb/test_run_pipeline_main.py:79) — docstring references `run.py`, file is `run_pipeline.py`. [VERIFIED]

#### RB
- **L-RB1.** [src/rb/config.py:174-184](src/rb/config.py:174) — `CONFIG_TINY` duplicates 5 of 8 keys with `tests/_pipeline_e2e_utils.py::_TINY_OVERRIDES`. [VERIFIED]
- **L-RB2.** [src/rb/features.py:24, 49](src/rb/features.py:24) — docstring claims "14 RB-specific features" but `_compute_features` emits 19 columns (13 of 14 specific + 4 game-level + 2 prior-season rate). [VERIFIED]
- **L-RB3.** [src/rb/targets.py:47](src/rb/targets.py:47) — `compute_targets` writes a phantom `fantasy_points_check` column to the DataFrame (debug only); QB equivalent uses a local variable. [VERIFIED]
- **L-RB4.** [src/rb/run_pipeline.py:3-5](src/rb/run_pipeline.py:3) + [src/shared/position_pipeline.py:111-122](src/shared/position_pipeline.py:111) — docstring lists `td_model_type` variants `(ridge / two_stage / ordinal / gated_ordinal)`, but `"ridge"` is silently the fall-through default with no validation. A typo like `"gated_oridinal"` degrades to plain ridge with no warning. [VERIFIED]
- **L-RB5.** [src/rb/config.py:354-379](src/rb/config.py:354) — `two_stage_targets` and `ordinal_targets` config blocks are configured but dormant under current `td_model_type="gated_ordinal"`. Drift risk if only one is maintained. [VERIFIED]
- **L-RB6.** [src/rb/config.py:79-89](src/rb/config.py:79) — L5 rolling means/stds/maxes excluded from `rolling` category with no nearby comment explaining why (other positions document the multicollinearity rationale). [VERIFIED]
- **L-RB7.** No test asserts RB red-zone PBP stats stay in `attn_history_stats` and NOT in `attn_static_features`. Future contributor could mistakenly move them and regress PR #235's architecture. [VERIFIED — test-gap]
- **L-RB8.** [src/rb/features.py:164-175](src/rb/features.py:164) — guards `prior_season_mean_receptions`/`_rushing_yards` presence but reads `prior_season_mean_targets`/`_carries` unconditionally. Latent `KeyError` if upstream `ROLL_STATS` ever loses those columns. [VERIFIED]
- **L-RB9.** [src/rb/analyze_errors.py](src/rb/analyze_errors.py) — operator-only CLI, no CI hook, no tests, runs `run()` unconditionally (full pipeline). [VERIFIED]
- **L-RB10.** [src/rb/features.py:50](src/rb/features.py:50) — `add_specific_features` mutates input via `sort_values(..., inplace=True)`. Docstring claims per-split isolation; caller's row ordering silently changes. [VERIFIED]

#### WR
- **L-WR1.** [src/te/config.py:75](src/te/config.py:75) referenced `is_home` "zero-variance for WR/RB", but WR/RB both still include it. Fixed inline (te-review removed the comment); the inconsistency between WR's production config keeping `is_home` and `src/wr/benchmark_ridge_variants.py:39-40` dropping it as "Zero variance" remains. [VERIFIED]
- **L-WR2.** [tests/wr/conftest.py:47-80](tests/wr/conftest.py:47) + [tests/wr/test_features.py:23-51](tests/wr/test_features.py:23) — duplicated player-games factory (WR is the only position with this duplication). [VERIFIED]
- **L-WR3.** No `tests/wr/test_run_cv_pipeline.py` despite `has_cv_runner=True` (RB has the same gap). [VERIFIED — test-gap]
- **L-WR4.** [src/wr/benchmark_ridge_variants.py:214-223](src/wr/benchmark_ridge_variants.py:214) — `pos_train` passed twice in `_run_variant` call (3rd and 7th positional). Functionally OK but obscures intent. [VERIFIED]
- **L-WR5.** [tests/wr/test_pipeline_e2e.py:53-54](tests/wr/test_pipeline_e2e.py:53) — hardcoded `train_seasons=(2022, 2023)` means smoke never exercises the prior-season aggregate path. [VERIFIED]
- **L-WR6.** [src/shared/position_pipeline.py:211-215](src/shared/position_pipeline.py:211) — `attn_lr`/`attn_weight_decay`/`attn_batch_size`/`attn_patience` only plumbed for `pos in ("QB", "RB", "K", "DST")` — WR/TE silently fall back to `nn_*` defaults. PositionConfig `attn_patience=35` change would have no effect for WR. [VERIFIED]
- **L-WR7.** [src/wr/targets.py:42-47](src/wr/targets.py:42) — `compute_targets` decomposition check uses hardcoded scoring literals (0.1, 6, 0.04, 4, -2) instead of `src.config` constants. QB has the same pattern. Will silently disagree if scoring constants change. [VERIFIED]
- **L-WR8.** [tests/wr/test_feature_contract.py:48-49](tests/wr/test_feature_contract.py:48) — `air_yards_per_target_L3`/`yac_per_reception_L3` contract floor `-5.0` is looser than the actual constraint (ratio of non-negatives). [VERIFIED]

#### TE
- **L-TE2.** [src/te/config.py:53-60](src/te/config.py:53) — TE keeps both `target_share_L3` and `target_share_L5` (and `carry_share_L3`/`_L5`) while RB dropped these as collinear. Either TE has been audited separately (needs comment) or just inherited WR's list. [VERIFIED]
- **L-TE3.** [tests/te/test_pipeline_e2e.py:38-65](tests/te/test_pipeline_e2e.py:38) + [tests/te/conftest.py:51-152](tests/te/conftest.py:51) — TE's E2E uses fully-synthetic splits while WR/QB/RB slice real engineered parquets. Drift in `build_features` for TE-specific upstream merges won't be caught. [VERIFIED]
- **L-TE4.** [src/te/data.py:13-15](src/te/data.py:13) — docstring lacks the "must be called AFTER build_features()/temporal_split()" ordering note that RB's has. [VERIFIED]
- **L-TE6.** [tests/te/test_features.py:133-150](tests/te/test_features.py:133) — `test_team_target_share_single_player` weakened from WR's `len(shares) == 3` to TE's `len(shares) > 0`. [VERIFIED]
- **L-TE7.** [tests/te/test_features.py:21-50](tests/te/test_features.py:21) — defines `_make_player_games` as module-level helper while WR/QB exposes it as a session-scoped fixture. [VERIFIED]
- **L-TE8.** [src/te/run_pipeline.py:8-11](src/te/run_pipeline.py:8) — imports `src.shared.*` before `src.te.config` (WR same); QB/RB import `src.{pos}.config` first. Cosmetic. [VERIFIED]

#### K
- **L-K1.** [src/k/config.py:69](src/k/config.py:69) — comment claims `is_home`, `is_dome`, `implied_team_total`, `game_wind` are "not in the static whitelist" but all four ARE in `_CONTEXTUAL_FEATURES` which feeds the static branch. Dual-feed may be intentional but the comment misleads. [VERIFIED]
- **L-K2 (overlaps M7).** Per-kick PBP cache freshness gap.
- **L-K3.** [src/k/data.py:124-125, 152-155, 470-475](src/k/data.py:124) — `clutch_fg_att`, `clutch_fg_made`, `max_fg_distance` computed and persisted to cache but never consumed. Wastes groupby + parquet bytes + 2025 backfill round-trip. [VERIFIED]
- **L-K4.** [src/k/config.py:174-175](src/k/config.py:174) + [src/shared/position_pipeline.py:229-230](src/shared/position_pipeline.py:229) — K's `head_losses={t: "huber"}` is declared but `build_pipeline_config` excludes K from `cfg["head_losses"]`. Pure documentation; not plumbed. [VERIFIED]
- **L-K5.** [src/k/data.py:363](src/k/data.py:363) — `_team_remap = {"OAK": "LV", "SD": "LAC", "STL": "LA"}` duplicates `src/shared/weather_features.py:46::_TEAM_CODE_NORMALIZATION`. K should import the shared constant. [VERIFIED]
- **L-K6.** [tests/k/test_data.py](tests/k/test_data.py) is fully subsumed by [tests/k/test_data_loaders.py:593-619](tests/k/test_data_loaders.py:593). [VERIFIED]
- **L-K7 (overlaps M2).** K schedule-merge redundancy.
- **L-K8.** [src/k/data.py:327-328](src/k/data.py:327) — `MIN_GAMES = 4` filter applied to ALL splits (other positions: train-only). Effective filter is `>=6 train, >=4 val/test`, asymmetric with other positions. [VERIFIED]
- **L-K9.** [src/k/config.py:124-142](src/k/config.py:124) — `CONFIG_TINY_ATTN` omits `attn_max_games` + `attn_max_kicks_per_game`. Works today because the builder closure provides them, but fragile if `MultiHeadNetWithNestedHistory` ever required them from cfg. [VERIFIED]
- **L-K10.** [src/k/data.py:15-16, 272-277](src/k/data.py:15) — section header "(2015-2024)" but code uses `s <= 2024` with no 2015 lower bound (relies on `SEASONS` starting at 2015). [VERIFIED]
- **L-K11.** [src/k/config.py:146-147](src/k/config.py:146) — `attn_max_seq_len=None` rationale is in K's config but the consuming gate at [src/shared/position_pipeline.py:206-207](src/shared/position_pipeline.py:206) is silent about why it exists. [VERIFIED]

#### DST
- **L-DST1.** [src/serving/app.py:577-583](src/serving/app.py:577) — stale comment claims "DST still applies a post-hoc adjustment (defensive TDs + safeties)" but DST's `compute_adjustment_fn` is `None`; aggregation is fully handled via `aggregate_fn → _dst_predictions_to_fantasy_points`. [VERIFIED]
- **L-DST2 (overlaps M12).** Opp-OFFENSE attention not exercised E2E.
- **L-DST3.** [tests/dst/test_feature_contract.py:124-127](tests/dst/test_feature_contract.py:124) — `_PRIOR_SEASON_COLS` exemption is unreachable; the columns live in `_CONTEXTUAL_FEATURES` but the test loops only `SPECIFIC_FEATURES`. [VERIFIED]
- **L-DST4.** [tests/dst/conftest.py:118-127](tests/dst/conftest.py:118) — `make_team_games` fixture hand-codes the linear scoring; will silently desync from `compute_targets` if linear coefficients change. [VERIFIED]
- **L-DST6.** [src/dst/targets.py:26-41](src/dst/targets.py:26) — `_pts_allowed_to_bonus` raises `ValueError` on NaN input. Safe today because callers `fillna` first, but the helper is exported. [VERIFIED]
- **L-DST7.** [src/dst/data.py:100-105](src/dst/data.py:100) — `weekly_copy = weekly.copy()` is a full DataFrame copy when only two scratch columns are needed. Minor perf nit. [VERIFIED]
- **L-DST8.** [src/dst/config.py:156-157](src/dst/config.py:156) — `ridge_pca_components=20` comment "5 features → 20 components" is truncated and unclear; actual input is 38 features. [VERIFIED]
- **L-DST9.** [tests/dst/test_data.py:55-69](tests/dst/test_data.py:55) — `**overrides` factory accepts arbitrary keys with no schema validation; typos silently create new columns. [VERIFIED]

### Cross-cutting + shared findings

#### Shared infrastructure
- **L-S1.** [src/shared/models.py:350-352, 464-466](src/shared/models.py:350) — `RidgeMultiTarget.predict_total` / `ElasticNetMultiTarget.predict_total` return unweighted raw-stat sums (yards + TDs + receptions, all summed as if commensurate). No production callers; only `tests/*/test_models.py` exercise it as "the total". A future caller using this for ranking would regress the `~1.9 pt/game double-count fix` documented in TODO.md. [VERIFIED]
- **L-S3.** [src/shared/aggregate_targets.py:165-167](src/shared/aggregate_targets.py:165) — `_tier_bonuses` docstring references an "aux-total loss" / "aux-total gate" architectural concept that does not exist in code (`grep -rn aux_total src/` returns only this docstring). [VERIFIED]
- **L-S5.** [src/shared/models.py:263-285](src/shared/models.py:263) — `GatedOrdinalTDClassifier.save` rewrites the meta file that `ordinal_.save` already wrote; `load` doesn't restore the gate's `_class_values_cfg`/`_ordinal_alpha`. Works today (not consulted at predict-time) but fragile. [VERIFIED]
- **L-S6.** [tests/test_aggregate_targets_branches.py:24-40](tests/test_aggregate_targets_branches.py:24) — `_tier_bonuses` torch-vs-numpy parity test only covers the PA path, not the YA path. [VERIFIED]
- **L-S7.** [src/shared/neural_net.py:715-729](src/shared/neural_net.py:715) — `MultiHeadNetWithHistory.gated_targets` is stored regardless of `gated`, but only consulted when `gated=True`. Misconfigured `cfg` with `attn_gated=False, gated_targets=["rushing_tds"]` silently builds plain heads. [VERIFIED]
- **L-S8.** [src/shared/models.py:564-575](src/shared/models.py:564) — `LightGBMMultiTarget.fit` allocates callbacks inside the loop but only the validation branch uses them. Plus unconditional `print(...)` for `best_iteration_`. [VERIFIED]
- **L-S9.** [src/shared/models.py:30](src/shared/models.py:30) — `_LGBM_N_JOBS` reads env at module-import time. Subtle landmine for ad-hoc Jupyter/benchmark sessions. [VERIFIED]
- **L-S10.** [src/shared/pipeline.py:594-668, 786-809](src/shared/pipeline.py:594) — `_train_nn`/`_train_attention_nn`/`_train_nested_attention_nn` take an unused `position` parameter; callers thread it through pointlessly. [VERIFIED]
- **L-S11.** [src/shared/pipeline.py:1106, 1187, 1270, 1308, 1746](src/shared/pipeline.py:1106) — `*_val_preds` captured but never used downstream. Wasted forward pass. [VERIFIED]
- **L-S12.** [src/shared/position_pipeline.py:268-269](src/shared/position_pipeline.py:268) — `compute_adjustment_fn = None` is a serving-side concept; `pipeline.py` never reads it. Dead cfg key. [VERIFIED]
- **L-S13.** [src/shared/pipeline.py:940, 966](src/shared/pipeline.py:940) — `run_pipeline` docstring says "QB/RB/WR/TE" but K and DST also call it. Scheduler list at :966 omits `"plateau"`. [VERIFIED]
- **L-S14.** [src/shared/position_pipeline.py](src/shared/position_pipeline.py) (multiple sites) — branches on string literals (`pos == "K"`, etc.) despite `Position` StrEnum existing from PR #247. [VERIFIED]
- **L-S15.** [src/shared/run_pipeline_factory.py:60](src/shared/run_pipeline_factory.py:60) — only `src/` → `tests/` import in the repo (`from tests._pipeline_e2e_utils import build_tiny_config`). [VERIFIED]
- **L-S16.** [src/shared/pipeline.py:1066-1068](src/shared/pipeline.py:1066) — silent ridge-fallback for `enet_alpha_grids`; the field doesn't exist on `PositionConfig`, so no position can declare ElasticNet-specific alphas without bypassing the factory. [VERIFIED]
- **L-S17.** All `src/{pos}/run_pipeline.py` build `CONFIG = build_pipeline_config(...)` at module top level, which eagerly imports `data.py`/`features.py`/`targets.py`. Import-time failures surface in confusing places. [VERIFIED]
- **L-S18.** [src/shared/pipeline.py:200-207](src/shared/pipeline.py:200) — `_run_nn_training` docstring says "three bodies" then lists four. [VERIFIED]
- **L-S19.** [tests/shared/conftest.py:62](tests/shared/conftest.py:62) — `tiny_synthetic_games` fixture's `_POSITIONS = ("QB", "RB", "WR", "TE")` omits K/DST; docstring doesn't flag the carve-out. [VERIFIED]
- **L-S20.** [src/shared/smoke_test.py:121-149](src/shared/smoke_test.py:121) — nested attention branch unconditionally entered regardless of `attn_static_from_df`. K is the only nested position and sets it True; future position with nested + !attn_static_from_df has no contract test. [VERIFIED]

#### Data / features
- **L-D1.** [src/features/engineer.py:1017-1042](src/features/engineer.py:1017) — `fill_nans_safe` has zero callers in `src/`. Production uses `src/shared/feature_build.py::fill_nans_with_train_means`. Dead code. [VERIFIED]
- **L-D2.** [src/shared/weather_features.py:140](src/shared/weather_features.py:140) — `implied_total_x_dome` is in the drop list but never created anywhere in the repo. Leftover placeholder. [VERIFIED]
- **L-D3.** [src/shared/feature_cache.py:57-61](src/shared/feature_cache.py:57) — `_df_fingerprint` uses sum-of-hashes (commutative). Order-insensitive: two frames with same rows in different order produce the same fingerprint. Footgun for future shuffle/non-deterministic groupby. [VERIFIED]
- **L-D4.** [src/shared/feature_cache.py:87](src/shared/feature_cache.py:87) — `_config_fingerprint` lists `attn_static_features` but `_prepare_position_data_uncached` doesn't consume it. Unnecessary cache misses on Optuna sweeps that vary only attention static cols. [VERIFIED]
- **L-D5.** [src/data/loader.py:17-23](src/data/loader.py:17) vs [src/data/redzone_pbp.py:32-44](src/data/redzone_pbp.py:32) — red-zone column lists duplicated. Adding a 6th column in `redzone_pbp.py` won't be backfilled with 0 by `loader.py`. [VERIFIED]
- **L-D6.** [src/data/loader.py:26, 67, 291](src/data/loader.py:26) + [src/data/split.py:10-12, 50](src/data/split.py:10) — `list[int] = None` defaults typed as `list[int]` rather than `list[int] | None`. Inconsistent with rest of codebase. [VERIFIED]
- **L-D7.** [src/shared/team_box_score.py:208-220](src/shared/team_box_score.py:208) — `merge_team_box_score_features` warning only counts unmatched rows for `TEAM_BOX_SCORE_FEATURES[0]`, not `OPP_BOX_SCORE_FEATURES[0]`. Opponent-side unmatched rows silently zero-fill. [VERIFIED]
- **L-D8.** [src/shared/weather_features.py:40, 55-64](src/shared/weather_features.py:40) — `_schedule_cache` is module-level mutable without a `threading.Lock`. `feature_cache._lru_lock` has one; the precedent exists. [VERIFIED]
- **L-D9.** [src/features/engineer.py:830-853](src/features/engineer.py:830) + [src/shared/weather_features.py:138-201](src/shared/weather_features.py:138) — `implied_team_total` computed twice in different files; survives only because of drop-and-rebuild ordering. [VERIFIED]
- **L-D10.** [src/data/redzone_pbp.py:62-138](src/data/redzone_pbp.py:62) — `_aggregate_one_season` doesn't enforce `season_type == 'REG'` (caller does). Playoff plays would silently flow in if `_aggregate_one_season` is ever called directly. [VERIFIED]

#### Evaluation / registry
- **L-E1.** [tests/shared/test_model_sync.py:1205](tests/shared/test_model_sync.py:1205) — `test_predcache_respects_custom_prefix` ends with `assert not (tmp_path / "benchmark_history" / "y.json").exists()` — copy-paste from another test, vacuous. [VERIFIED]
- **L-E3.** [tests/shared/test_benchmark_utils.py:474](tests/shared/test_benchmark_utils.py:474) — `ts[:19].replace("-", "T", 0)` is a no-op (third arg `0` means "replace 0 occurrences"). [VERIFIED]
- **L-E4.** [src/shared/model_sync.py:60](src/shared/model_sync.py:60) — `_repo_root` docstring ambiguous ("three parents up"). [VERIFIED]
- **L-E5.** [src/shared/model_sync.py:476-498](src/shared/model_sync.py:476) — `sync_predictions_cache_from_s3` cleanup only runs when `missing` (NoSuchKey/404) is non-empty. Different ClientError class → no cleanup, no log. [VERIFIED]
- **L-E6.** [src/shared/evaluation.py](src/shared/evaluation.py) (after PR #262 merge) — `np.sqrt(mean_squared_error(...))` could use sklearn 1.4+'s `root_mean_squared_error`. Pure style. [VERIFIED]

#### Serving / scripts
- **L-SS1.** [src/scripts/promote.py:172-179](src/scripts/promote.py:172) — manifest written before legacy mirror copy; partial failure leaves inconsistent intermediate state with no clean error logging (raw `ClientError` propagates uncaught). [VERIFIED]
- **L-SS2.** [src/scripts/promote.py:73-89](src/scripts/promote.py:73) + [tests/scripts/test_promote.py:182-185](tests/scripts/test_promote.py:182) — `_parse_version_from_key` returns garbage on malformed keys; test pins the broken-by-design contract. [VERIFIED]
- **L-SS3.** [src/scripts/audit_features.py](src/scripts/audit_features.py) — operator-only utility, no CI hook, no tests, K/DST coverage omitted. [VERIFIED]
- **L-SS4.** [src/serving/app.py:95](src/serving/app.py:95), [:1801](src/serving/app.py:1801) — three cache disciplines in one file (RLock `_cache_lock`, separate `_BENCHMARK_HISTORY_LOCK`, module-global state). Wiki + predictions share the RLock; a hot wiki hit can serialize behind a slow `_ensure_metrics`. [VERIFIED]
- **L-SS5.** [src/serving/app.py:1010-1013](src/serving/app.py:1010) — `_ensure_position_loaded` silent early-return when `"splits" not in _cache`. Should never happen post-`_ensure_base_data` but masks if it does. [VERIFIED]

#### Batch / CI / analysis
- **L-B1.** [src/tuning/tune_lgbm.py:127-138](src/tuning/tune_lgbm.py:127) — `_prepare_cv_folds` raises `UnboundLocalError` on empty `folds` (`feature_cols` referenced outside the loop where it's assigned). [VERIFIED]
- **L-B2.** [src/batch/launch.py:386-393](src/batch/launch.py:386) — `TIMED_OUT` positions silently absent from main()'s success/failure summary. [VERIFIED]
- **L-B3.** [src/batch/benchmark.py:152](src/batch/benchmark.py:152) — no `--wait-timeout` CLI; silently inherits 3-hour default. Asymmetric with `src/batch/launch.py`. [VERIFIED]
- **L-B4.** [src/analysis/analysis_k_signal_floor.py:74](src/analysis/analysis_k_signal_floor.py:74) — `BENCHMARK_PATH` is hardcoded to one snapshot file; falls back to constants if rotated. [VERIFIED]
- **L-B5.** [src/analysis/analysis_k_signal_floor.py:51](src/analysis/analysis_k_signal_floor.py:51), [analysis_k_feature_audit.py:81](src/analysis/analysis_k_feature_audit.py:81), [analysis_rb_feature_audit.py:40](src/analysis/analysis_rb_feature_audit.py:40) — pop GUI windows on Mac (no `matplotlib.use("Agg")`). Only `analysis_shap_lgbm.py:32` sets backend. [VERIFIED]
- **L-B6.** No tests under `tests/tuning/` or `tests/analysis/`. Two analysis scripts have tests (`test_nflcom_baseline.py`, `test_shap_analysis.py`); four don't. Both tuning modules have zero tests. [VERIFIED — test-gap]
- **L-B7.** [src/analysis/analysis_dst_rare_dispersion.py](src/analysis/analysis_dst_rare_dispersion.py) — zero references anywhere in the repo. Dead-code-or-operator-CLI status unclear. [VERIFIED]
- **L-B8.** [src/analysis/analysis_k_feature_audit.py:61-70](src/analysis/analysis_k_feature_audit.py:61) — docstring bakes in 2026-05-20 audit numbers ("condition number 8.33", VIF 8.49, etc.). Drift-prone. [VERIFIED]
- **L-B9.** [pyproject.toml](pyproject.toml) vs [src/batch/requirements.txt](src/batch/requirements.txt) — three sources of Python pins (root, Batch, and base-image torch). No automated parity check. [VERIFIED]
- **L-B10.** [.github/workflows/_detect-positions.yml:31-66](.github/workflows/_detect-positions.yml:31) — `fetch-depth: 2` works for `workflow_dispatch` only because the empty-input branch short-circuits to all-positions. Working correctly but brittle to read. [VERIFIED]

---

## Cross-position consistency matrix

| Concern | QB | RB | WR | TE | K | DST |
|---------|----|----|----|----|----|----|
| INCLUDE_FEATURES opt-in (no select-all) | ✅ | ✅ | ✅ | ✅ | ⚠️ (`all_features` enum) | ⚠️ (`all_features` enum) |
| ATTN_STATIC_FEATURES has NO windowed features (PR #260) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ATTN_HISTORY_STATS present, consistent shape | ✅ | ✅ | ✅ | ✅ | ✅ nested + flat | ✅ |
| `LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS[t]` (or doc'd exception) | ⚠️ Poisson w=1.0 doc'd | ⚠️ Poisson/hurdle w=1.0 doc'd | ⚠️ Poisson/hurdle w=1.0 doc'd | ⚠️ Poisson/hurdle w=1.0 doc'd | ✅ all Huber 1.0 | ⚠️ Poisson w=5.0 doc'd |
| `nn_non_negative_targets` set explicitly (see M1) | ⚠️ explicit | ⚠️ explicit | ⚠️ explicit | ⚠️ explicit | ⚠️ explicit | ⚠️ explicit |
| Every TARGET is a raw NFL stat | ✅ | ✅ | ✅ | ✅ | ⚠️ `fg_yard_points`/`pat_points` (derived raw — not fantasy_points) | ✅ |
| `run_pipeline.py` uses `cli_main` (PR #257) | ✅ | ✅ | ✅ | ✅ | ⚠️ custom (doc'd) | ⚠️ custom (doc'd) |
| `data.py` uses `position_data` (PR #254) | ✅ | ✅ | ✅ | ✅ | ⚠️ custom PBP (doc'd) | ⚠️ custom team-level (doc'd) |
| `conftest.py` uses `register_standard_fixtures` (PR #245) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `test_run_pipeline_main.py` monkeypatches `src.{pos}.run_pipeline.run_pipeline` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| In `scope_positions.ALL_POSITIONS` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| In `src/batch/train.py` orchestration list | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| All `attn_*` hyperparams plumbed through factory | ✅ | ✅ | ❌ L-WR6 | ❌ L-WR6 | ✅ | ✅ |
| `test_run_cv_pipeline.py` exists for positions with `has_cv_runner=True` | ✅ | ❌ L-WR3 | ❌ L-WR3 | n/a | n/a | n/a |
| E2E test uses real engineered parquets | ✅ | ✅ | ✅ | ❌ L-TE3 (synthetic only) | n/a | ❌ (synthetic only — same shape) |
| `head_losses` correctly plumbed to cfg | ✅ | ✅ | ✅ | ✅ | ❌ L-K4 (declared but not plumbed) | ✅ |
| `aggregate_fn` plumbed to cfg | ✅ | ✅ | ✅ | ✅ | ✅ (H1 fixed [#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287)) | ✅ |

Legend: ✅ uniform · ⚠️ documented divergence · ❌ undocumented gap or bug

---

## Inline style fixes applied (already in this worktree)

| File:line | Fix | Worker |
|-----------|-----|--------|
| [src/qb/features.py:18, :25](src/qb/features.py:18) | Replaced `SPECIFIC_FEATURES` (removed module-level constant) with `POSITION_CONFIG.specific_features` in docstrings | qb-review |
| [src/rb/run_pipeline.py:4](src/rb/run_pipeline.py:4) | Replaced `TD_MODEL_TYPE` (retired UPPER_CASE constant) with `td_model_type` in docstring | rb-review |
| [src/te/config.py:75](src/te/config.py:75) | Removed stale comment claiming `is_home` is zero-variance for WR/RB (both still include it) | te-review |
| [src/dst/targets.py:32, :41](src/dst/targets.py:32) | Fixed misleading fallback-branch comments (the fallback only fires for impossible >999/>99999 inputs) | dst-review |
| [src/shared/pipeline.py:606-610](src/shared/pipeline.py:606) | Removed stale "Weather NN" sentence from `_train_nn` docstring | shared-pipeline |
| [src/data/preprocessing.py:1](src/data/preprocessing.py:1), [src/data/split.py:1-5](src/data/split.py:1) | Sorted/grouped imports, added blank lines (ruff I001 + cascading line-length fixes) | shared-data-features |
| [src/scripts/audit_features.py:290-294](src/scripts/audit_features.py:290) | Updated stale `run.py` references to `python -m src.k.run_pipeline` / `src.dst.run_pipeline` | serving-scripts |
| [src/tuning/ablate_rb_gate.py:268](src/tuning/ablate_rb_gate.py:268) | Updated `--only` help text from "default: run all three" to "run all variants" (`VARIANTS` has 6 entries) | batch-ci-bench-analysis |

Audit with: `git diff main..HEAD --stat` from this worktree.

---

## Methodology

- 13 Opus 4.7 1M agents launched in parallel:
  1-6. one per football position (`src/{qb,rb,wr,te,k,dst}/` + `tests/{pos}/`)
  7. shared pipeline + training + factory + config + Position enum
  8. shared models + neural net + aggregation + `src/models/` + `src/training/`
  9. shared data + features + weather + `src/data/` + `src/features/`
  10. shared evaluation + registry + sync + artifact + `src/evaluation/`
  11. serving + scripts + tests/test_app*
  12. batch + CI workflows + benchmarking + tuning + analysis
  13. cross-position consistency comparator
- Each worker was given a self-contained brief, the CLAUDE.md stop-rules verbatim, and authority to apply 100%-certain trivial style fixes inline.
- Each finding was verified by the orchestrator reading the cited code at the cited line and confirming the worker's claim. High-severity findings (H1-H3) were verified end-to-end with reproducer traces. The `nn_non_negative_targets` finding (M1) was cross-confirmed by three independent workers.
- Findings cross-link to overlapping ones via short refs (e.g. `(overlaps M7)`).

## False positives (from re-verification 2026-05-20)

The original audit reported zero false positives. Re-verification at HEAD of `claude/kind-kare-bd547f` (19 commits ahead) by 8 parallel agents pruned 9 entries:

| ID | Original claim | Re-verification |
|----|----------------|-----------------|
| H3 | Module-level S3 syncs run before `bind()` under `--preload` | ALREADY FIXED — syncs now live in `gunicorn.conf.py::on_starting` (master-level, pre-import). `app.py` lines 79-82 explicitly document this. |
| M14 | No invariant test for `LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS[t]` | FALSE POSITIVE — `test_loss_weights_match_huber_deltas` exists at [tests/test_invariants.py:46-91](tests/test_invariants.py:46). |
| L-WR9 | `head_losses` comments mix "current state" and "what changed" | FALSE POSITIVE — comment is a clear, accurate statement of current state. |
| L-TE1 | `nn_head_hidden_overrides` comment has stale WR `receiving_tds` clause | FALSE POSITIVE — comment is about TE; the `receiving_tds` note correctly describes TE's own head. |
| L-TE5 | L5 collinearity filter is uncommented (inconsistent with QB/WR) | FALSE POSITIVE — pattern + silence is consistent with QB/RB; not a TE-specific inconsistency. |
| L-DST5 | `is_dome` quirk comment doesn't match actual behavior | FALSE POSITIVE — comment is scope-correct (describes the test fixture, not DST's broader handling). |
| L-S2 | `LastWeekBaseline` has zero callers | ALREADY FIXED — `src/models/` directory deleted by [#263](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/263). |
| L-S4 | Legacy `Trainer` + `make_dataloaders` only used by tests | ALREADY FIXED — `src/training/` directory deleted by [#264](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/264). |
| L-E2 | `print_comparison_table` has zero production callers | FALSE POSITIVE — has production callers at [src/shared/pipeline.py:1408](src/shared/pipeline.py:1408), [:1861](src/shared/pipeline.py:1861) (PR #262 merged eval modules but kept the function in use). |

## Out of scope

- No fixes opened as PRs.
- No changes to `.github/workflows/`, `Dockerfile*`, `requirements.txt`, `pyproject.toml`, `CLAUDE.md`, `TODO.md`, `docs/ARCHITECTURE.md` — those are documented as separate-PR territory.
- No benchmark or training run was triggered.
- No new tests written.

---

## Remediation plan

Grouped into 4 waves. Each PR is **bounded**, **independent of siblings in its wave**, and writeable in one focused sub-agent session. PRs across waves are sequenced where they touch the same file or share infra.

Touch-point conflict map (sequence these):
- **H1, M5, M9** all touch `src/shared/position_pipeline.py` — H1 first (small targeted line), then M5/M9 (orthogonal edits).
- **H2, M2** both touch `src/shared/weather_features.py` — H2 first, M2 layers on top.
- **M16, M17** both touch `src/serving/app.py` + `src/shared/model_sync.py` — keep in same wave; rebase the second.
- **F-WR, B5** both touch `src/wr/` files but different files (`config.py` vs `benchmark_ridge_variants.py`) — safe in parallel.
- **L-K2 (overlaps M2)** and **L-K7 (overlaps M2)** — fixed by M2.
- **L-DST2 (overlaps M12)** — fixed by M12.

### Wave 1 — HIGH severity (2 PRs, sequential within wave) — ✅ SHIPPED 2026-05-21

| PR | Findings | Status | Touches |
|----|----------|--------|---------|
| **W1.1** | H1 | ✅ Shipped in [#287](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/287) (`2eb4f43`). Removed `if pos != "K"` guard. Local K benchmark: all four ranking metrics improved (`attn_nn_top12` 0.403→0.514 was the largest). | src/shared/position_pipeline.py, tests/k/test_pipeline_e2e.py, TODO.md |
| **W1.2** | H2 | ✅ Shipped in [#290](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/290) (`6618243`). Picked Option B — added `spread_line`/`div_game` to upfront drop and removed from cleanup-drop in `merge_schedule_features`. Local DST benchmark: Attn NN total MAE 5.288→5.113 (-0.175). | src/shared/weather_features.py, tests/shared/test_weather_features.py, TODO.md |

### Wave 2 — MEDIUM bugs + drift (10 PRs, parallel)

| PR | Findings | Scope | Touches |
|----|----------|-------|---------|
| **W2.1** | M1 | Pick (a) or (b) from the M1 entry. Recommended: (a) `field(default_factory=set)` → `default=None`, type `set[str] \| None`, drop the explicit `set(_TARGETS)` in all 6 configs, restore CLAUDE.md claim. | src/shared/position_config.py, src/shared/registry.py, src/{6 positions}/config.py, CLAUDE.md |
| **W2.2** | M2 | Set `df["_schedule_merged"] = True` at end of K's `load_data` schedule merge. Verify shared merge becomes a no-op for K. Resolves L-K7 and L-K2 by extension. | src/k/data.py |
| **W2.3** | M5 | Route `src/batch/train.py::_run_rb_gate_ablation` to call `src/tuning/ablate_rb_gate.py`'s VARIANTS + decision logic; delete the in-train.py copy. | src/batch/train.py |
| **W2.4** | M6 | Update `_format_config_lines` in `src/tuning/tune_lgbm.py` to emit lowercase kwarg form matching `build_position_config()`. Drop or rewrite `PR 3b` reference in `retune-lgbm.yml:204`. | src/tuning/tune_lgbm.py, .github/workflows/retune-lgbm.yml |
| **W2.5** | M7 | Mirror weekly-cache pattern: add `_cached_kick_pbp_is_current` validating schema (incl. `play_id`) and mtime vs upstream parquets. | src/k/data.py |
| **W2.6** | M8 | One-line fix: `-f batch/Dockerfile.train` → `-f src/batch/Dockerfile.train`. | src/batch/build_and_push.sh |
| **W2.7** | M9 | Add `specific_features`, `add_features_fn`, `fill_nans_fn` to `REQUIRED_PIPELINE_CFG_KEYS`. Add tests omitting each from a hand-built cfg. | src/shared/position_pipeline.py, tests/test_invariants.py (or new) |
| **W2.8** | M13 | Replace `tests/rb/test_pipeline_e2e.py::_build_tiny_config` with `build_pipeline_config("RB", POSITION_CONFIG, overrides=_TINY_OVERRIDES)`. | tests/rb/test_pipeline_e2e.py |
| **W2.9** | M16 | Persist `position_details` + `position_load_errors` alongside `metrics.json` (or fold into same file). Restore in `_try_hydrate_from_disk`. | src/serving/app.py |
| **W2.10** | M18 | Replace `recv_fl`/`rush_fl`/`td_pts` header literals with `TARGETS` iteration; print all 4 MAE columns (include `fumbles_lost`). | src/wr/benchmark_ridge_variants.py |

### Wave 3 — MEDIUM CI/test gaps + serving (4 PRs, parallel)

| PR | Findings | Scope | Touches |
|----|----------|-------|---------|
| **W3.1** | M3 + M11 | Add runtime invariant test: parametrize over all 6 positions, run `build_position_features` against tiny real data, assert every `attn_static_feature` column exists with `.std() > 0` for non-constant. Plus: add `@pytest.mark.parametrize("pos", ["QB","RB","WR","TE","K","DST"])` smoke test that loads each real `POSITION_CONFIG`, materializes a tiny dataset, runs `run_smoke_test(pos)`. | tests/test_invariants.py, tests/shared/test_smoke_test.py |
| **W3.2** | M4 | Apply `env:` indirection pattern + top-level `permissions: contents: read` to `retune-lgbm.yml` and `ablate-rb-gate.yml` (same pattern PR #258 used). | .github/workflows/retune-lgbm.yml, ablate-rb-gate.yml |
| **W3.3** | M10 + M12 | (a) Add nested-attn smoke test that uses K's real `POSITION_CONFIG` and tiny synthetic dataset including kick PBP, asserting `_smoke_attention` returns without exception. (b) Extend synthetic DST fixture to materialize `off_pass_yards`, `off_rush_yards`, `off_pts_scored`, etc., and add an opp-offense-branch E2E case. | tests/shared/test_smoke_test.py, tests/dst/conftest.py, tests/dst/test_pipeline_e2e.py |
| **W3.4** | M15 + M17 | Either widen `_iter_fingerprint_paths` to all `data/raw/*.parquet`, or document raw-refresh-with-retrain coupling in docstring. Extend per-file isolation to `sync_data_from_s3` and `sync_benchmark_history_from_s3` (or document the divergence). | src/serving/app.py, src/shared/model_sync.py |

### Wave 4 — LOW cleanups by area (11 PRs, parallel)

These are independent file groups. Each worker takes one row, makes mechanical edits, and opens one bundled PR.

| PR | Findings (count) | Topic | Touches |
|----|-----------------|-------|---------|
| **W4.QB** | L-QB1–L-QB8 (8) | Diagnostic loop, contract weakness, docstring drift | src/qb/{data.py, diagnose_outliers.py}, tests/qb/* |
| **W4.RB** | L-RB1–L-RB10 (10) | Phantom debug column, defaulting, mutation, dead config | src/rb/{config.py, features.py, run_pipeline.py, targets.py, analyze_errors.py}, tests/rb/* |
| **W4.WR** | L-WR1–L-WR8 (8) | Benchmark/config drift, duplicated factories, attn plumbing gap | src/wr/{config.py, benchmark_ridge_variants.py, targets.py}, tests/wr/*, src/shared/position_pipeline.py (for L-WR6 — coordinate with W4.TE) |
| **W4.TE** | L-TE2, L-TE3, L-TE4, L-TE6, L-TE7, L-TE8 (6) | Collinearity audit comment, weak test, fixture style, import order | src/te/{config.py, data.py, run_pipeline.py}, tests/te/* |
| **W4.K** | L-K1, L-K3, L-K4, L-K5, L-K6, L-K8, L-K9, L-K10, L-K11 (9) | Mislabeled comment, dead computed cols, head_losses plumbing, dup constant, asymmetric filter | src/k/{config.py, data.py}, tests/k/*, src/shared/position_pipeline.py (for L-K4 — coordinate with W4.WR) |
| **W4.DST** | L-DST1, L-DST3, L-DST4, L-DST6, L-DST7, L-DST8, L-DST9 (7) | Stale comments, unreachable exemption, NaN guard, perf nit | src/dst/{data.py, targets.py, config.py}, src/serving/app.py (for L-DST1), tests/dst/* |
| **W4.SHARED-A** | L-S1, L-S3, L-S5, L-S6, L-S7, L-S8, L-S9, L-S10 (8) | Models cleanup: unused params, env-at-import, dead docstrings, save/load asymmetry | src/shared/models.py, src/shared/aggregate_targets.py, src/shared/neural_net.py, src/shared/pipeline.py, tests/test_aggregate_targets_branches.py |
| **W4.SHARED-B** | L-S11, L-S12, L-S13, L-S14, L-S15, L-S16, L-S17, L-S18, L-S19, L-S20 (10) | Pipeline cleanup: unused vars, Position StrEnum adoption, dead cfg keys, src→tests import | src/shared/{pipeline.py, position_pipeline.py, run_pipeline_factory.py, smoke_test.py}, tests/shared/conftest.py |
| **W4.DATA** | L-D1–L-D10 (10) | Dead code, duplicated lists, fingerprint footguns, threading | src/features/engineer.py, src/shared/{weather_features.py, feature_cache.py, team_box_score.py}, src/data/{loader.py, split.py, redzone_pbp.py} |
| **W4.EVAL-SERV** | L-E1, L-E3, L-E4, L-E5, L-E6, L-SS1, L-SS2, L-SS3, L-SS4, L-SS5 (10) | Vacuous tests, ambiguous docstring, error-class breadth, cache discipline | tests/shared/{test_model_sync.py, test_benchmark_utils.py}, src/shared/{model_sync.py, evaluation.py}, src/scripts/{promote.py, audit_features.py}, src/serving/app.py |
| **W4.BATCH-CI** | L-B1–L-B10 (10) | UnboundLocalError, TIMED_OUT summary gap, GUI backend, dead analysis, version pin sources | src/tuning/tune_lgbm.py, src/batch/{launch.py, benchmark.py}, src/analysis/*, pyproject.toml + src/batch/requirements.txt, .github/workflows/_detect-positions.yml |

### Parallelization recipe

```
Wave 1 (SHIPPED 2026-05-21 — sequential, both touch metric-impacting code):
  W1.1 (#287) → W1.2 (#290)  [K-only and DST-only benchmark re-runs done before each merge]

Wave 2 (10 parallel sub-agents, no shared files within wave):
  W2.1 || W2.2 || W2.3 || W2.4 || W2.5 || W2.6 || W2.7 || W2.8 || W2.9 || W2.10

Wave 3 (4 parallel sub-agents, watch shared touches):
  W3.1 || W3.2 || W3.3 || W3.4

Wave 4 (11 parallel sub-agents, one per area):
  W4.QB || W4.RB || W4.WR || W4.TE || W4.K || W4.DST
  || W4.SHARED-A || W4.SHARED-B || W4.DATA || W4.EVAL-SERV || W4.BATCH-CI
```

### Per-agent contract (from CLAUDE.md "Sub-agent contract")

Each worker must:
1. Read the cited finding + surrounding code.
2. Make the change.
3. Run `pytest -m unit` + ruff locally.
4. For NN/feature/loss/target changes (W1.1, W1.2, anything in `src/{pos}/features.py` / `src/shared/pipeline.py`), run `python -m src.{pos}.run_pipeline` and diff `benchmark_history/` against prior run.
5. Commit, push the branch, open the PR with `gh pr create`.
6. Verify `gh pr list` shows the PR before reporting completion.

### Suggested skip / defer

- **W4.WR L-WR1** (benchmark vs. production `is_home` divergence) and **W4.K L-K1** (mislabeled comment) — single-line comment fixes, OK to bundle inline with the more substantive PRs in their area.
- **W4.BATCH-CI L-B9** (3 pin sources) — needs a design decision (consolidate via pip-compile? sync via Makefile?). Defer until owner picks an approach.
- **W4.SHARED-B L-S17** (eager module-top-level `CONFIG = build_pipeline_config(...)`) — non-trivial refactor; defer behind W1 + W2 to stabilize the factory first.
