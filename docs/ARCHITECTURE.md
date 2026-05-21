# ADR-001: Fantasy Football Predictor — Consolidated Architecture

**Status:** Accepted · **Date:** 2026-04-16 · **Author:** Alex Free

### Update history

- **2026-05-21** — D13 hardened (BLAS thread oversubscription cap): pinned `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `NUMEXPR_NUM_THREADS=1` at the Batch job-definition level via [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml)'s existing jq filter, alongside the existing `NN_DATALOADER_NUM_WORKERS=3`. After PR [#301](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/301) unblocked the FP16 path on T4, the parallel CPU branch (`_tune_ridge_alphas_cv` via `joblib.Parallel(n_jobs=-1, prefer="threads")` per [src/shared/pipeline.py](../src/shared/pipeline.py):339) intermittently deadlocked OpenBLAS on 4-vCPU g4dn.xlarge: 4 joblib threads × default BLAS threads (`nproc=4`) + 3 DataLoader worker subprocesses + 1 main GPU thread ≈ 19 thread slots on 4 cores (~5× oversubscription). PR [#293](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/293) (BF16-on-T4 hang) had been masking the issue by hanging the GPU branch first, so the CPU branch had the cores to itself; PR #301 fixed BF16 → the PR [#267](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/267) ThreadPoolExecutor design finally exercised both branches in parallel and the BLAS oversubscription surfaced as a ridge_tune wedge (5 of 6 jobs in the 26218950001 + 26219279092 validation runs hit the 30-min timeout, ~25 min/job wasted). Single-threaded BLAS + 4 joblib threads = exact 4-core fan-out, no overcommit. Verified pre-merge by `ff-dst-verify-omp1-1779359690-A34E55` (manual `aws batch submit-job` with the caps as `containerOverrides`): `ridge_tune=6.1s`, `attn_nn_train=87.8s` (beat #293's 110s target), `total=225s`. Companion archive entry in [TODO.md](../TODO.md) captures the debugging-by-process-of-elimination and the "unblocking a regression can unmask a second hang that was hiding behind the first" lesson.
- **2026-05-21** — D14 extended (in-flight model refresh): [gunicorn.conf.py](../gunicorn.conf.py)'s `on_starting` now spawns a daemon thread via [src/shared/model_sync.py::start_refresh_poller](../src/shared/model_sync.py) that polls `s3://{bucket}/models/{POS}/manifest.json` etags every `FF_MODEL_REFRESH_INTERVAL_S` seconds (default 30; set 0 to disable). When an etag advances, `refresh_position` re-downloads via the same stable→current→previous chain `_sync_one` uses, extracts to `src/{pos.lower()}/outputs/models.new/`, swaps via `models→models.bak→rmtree, models.new→models`, then touches `src/{pos.lower()}/outputs/.refreshed_at`. [src/serving/app.py::_ensure_position_loaded](../src/serving/app.py) stats that sentinel on every call and re-runs `_apply_position_models` when its mtime advances past the value recorded at the last successful load. Decouples per-position deploy-to-prod from the all-positions ECS `update-service` wait — the first-finished position in a parallel Batch fan-out (e.g. WR at t≈2 min) goes live within one poll interval instead of waiting for the slowest position + ECS rolling redeploy. ECS `update-service` step in [train-batch.yml](../.github/workflows/train-batch.yml) kept as belt-and-suspenders; remove in a follow-up after the in-flight path runs cleanly for ~2 retrain cycles. The pre-refresh boot-time `sync_models_from_s3` contract is preserved — the poller's first observation per position records the etag without re-downloading (`last_etag=None` bootstrap branch).
- **2026-05-21** — D13 hardened (Option B race fix): closed the post-`systemctl restart containerd` race in [infra/batch/userdata.sh](../infra/batch/userdata.sh) where `Type=notify` READY fires when containerd's daemon socket opens — before proxy plugin discovery completes. If userdata exits in that window, ecs.service starts, claims a task, and the first image pull silently falls back to overlayfs for the rest of the boot (SOCI [#190](https://github.com/awslabs/soci-snapshotter/issues/190): fallback is transparent). Live evidence from the post-PR #295 migration: **4/5 fresh Spot hosts measured ~1–2 s SOCI pulls; 1/5 (instance i-095d8bf38eba1641c) measured 131 s** — same launch template, same userdata, only the plugin-discovery-vs-first-pull timing differed. New "Step 4" block at the end of userdata polls `ctr plugin ls` (exact-match `STATUS=ok` on the `io.containerd.snapshotter.v1 soci` row), 60s timeout, exit 1 on failure → instance marked unhealthy in the CE rather than silently serving overlayfs pulls. Existing socket-wait (step 3, after `systemctl enable --now soci-snapshotter`) is preserved as a second gate — together they guard the windows before and after the containerd restart.
- **2026-05-21** — D13 hardened (Option B SOCI activation, infra-landing): added `ff-batch-lt` EC2 launch template + [infra/batch/userdata.sh](../infra/batch/userdata.sh) to the `ff-gpu-spot` Compute Environment. Userdata installs `soci-snapshotter-grpc` v0.13.0 on the AL2 host, registers it as a containerd proxy plugin, and starts it as a systemd unit ordered `Before=ecs.service` — so the first image pull on a fresh Spot host streams from ECR via SOCI instead of doing the full ~122s pull. [batch-image.yml](../.github/workflows/batch-image.yml)'s `Publish SOCI index` step is now load-bearing: `continue-on-error: false` and version-pinned to `0.13.0` (matches the host snapshotter). [infra/batch/setup.sh](../infra/batch/setup.sh) reconciles the launch template onto an existing CE via `DISABLE → update-compute-environment → ENABLE` polling; [infra/batch/teardown.sh](../infra/batch/teardown.sh) deletes the launch template after CE delete. Operator runs `bash infra/batch/setup.sh` post-merge to attach the launch template to the live CE; expected pull-window drop from ~122 s to ~5–10 s (cold-start total ~258 s → ~135 s, ~115 s saved per job). Post-merge verification: `gh workflow run train-batch.yml -f positions=K -f seed=42` + `aws ecs describe-tasks` pullStartedAt/pullStoppedAt. Bottlerocket (Option A) explicitly rejected in this round for OS-family migration risk. GPU + nvidia-container-runtime interaction with SOCI is untested upstream — if first job fails CUDA, Tier-1 rollback (detach launch template from CE) per [infra/batch/README.md](../infra/batch/README.md).
- **2026-05-21** — D13 race-fix follow-ups: (1) operator tools migrated off the legacy key — [src/scripts/promote.py](../src/scripts/promote.py) no longer mirrors the freshly-promoted artifact into `models/{POS}/model.tar.gz`, and [src/scripts/analyze_gpu_profile.py](../src/scripts/analyze_gpu_profile.py) now resolves the GPU-profile tarball via `load_manifest()` instead of the legacy key. (2) `legacy_model_key()` removed from [src/shared/model_sync.py](../src/shared/model_sync.py) — no callers remain. (3) Per-position SHA coherency surfacing added: [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml) exports `FF_TRAIN_GIT_SHA=github.event.workflow_run.head_sha` to the launch step → [src/batch/launch.py](../src/batch/launch.py) forwards it via `containerOverrides.environment` → [src/batch/train.py::_extract_metrics](../src/batch/train.py) stamps it into `benchmark_metrics.json` → [src/batch/benchmark.py::find_git_sha_divergence](../src/batch/benchmark.py) warns when any position's recorded SHA diverges from the run's expected SHA. Defense-in-depth against the lingering "two parallel train-batch runs overlap on the same position's manifest write" case that Layer A pins at submit time but can still surface if Batch jobs interleave.
- **2026-05-21** — D13 hardened (layer C of three): the legacy `s3://{bucket}/models/{POS}/model.tar.gz` mirror is gone. Producer ([src/batch/train.py::upload_artifacts](../src/batch/train.py)) used to write it last as a pre-manifest-consumer compat key; two parallel train-batch runs writing the same legacy key were last-write-wins and a downstream reader (`benchmark.py --download-only`) could pull a different image's artifact than its own train job produced. PR #279 (Layer B) migrated benchmark.py off the legacy key; this PR deletes the producer write and converts the consumer-side manifest-absent branch in [src/shared/model_sync.py::_sync_one](../src/shared/model_sync.py) from "fall back to legacy" to a loud `RuntimeError`. Operator tools ([src/scripts/promote.py](../src/scripts/promote.py), [src/scripts/analyze_gpu_profile.py](../src/scripts/analyze_gpu_profile.py)) still reference `legacy_model_key()` for their own purposes — separate migration. Net effect: production has exactly one S3 pointer per position (`models/{POS}/manifest.json`) and one atomic-PUT path to update it.
- **2026-05-21** — D13 hardened (layer A of three): train-batch submissions now pin to the Batch job-definition *revision* that the triggering image build registered, not the bare definition name. Without pinning, AWS Batch resolves bare names to the latest active revision at submit time, so two parallel `batch-image.yml` runs registering revisions in the same window cause the downstream `train-batch.yml` for image A to silently submit jobs against image B's revision. [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml) now stashes the registered revision at `s3://ff-predictor-training/job-def-revisions/${sha}.txt`; [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml) reads it back and exports `FF_JOB_DEFINITION_REVISION`; [src/batch/launch.py](../src/batch/launch.py)'s `_job_definition_for()` appends `:N` when the env var is set. `workflow_dispatch` (smoke-test break-glass) skips the resolution and falls back to bare-name (current behavior, intentional — no upstream image build to pin to). Observed motivating incident: the `00c6d10` (CPU/GPU overlap) production run produced mixed-image artifacts on S3 because two `train-batch.yml` runs raced against the same job-definition revision.
- **2026-05-21** — D14 extended: gunicorn `on_starting` hook in [gunicorn.conf.py](../gunicorn.conf.py) now owns the four boot-time S3 syncs (`sync_data_from_s3`, `sync_models_from_s3`, `sync_benchmark_history_from_s3`, `sync_predictions_cache_from_s3`); previously they ran as module-level side effects in [src/serving/app.py](../src/serving/app.py). Wall-clock unchanged (still master, still before `bind()`), but `src.serving.app` is now side-effect-free at import — the docstring's "no module-level pre-warm under `--preload`" rule from the PR #148/#149 lesson now extends from prediction-cache warm to the S3 sync itself. Operationally paired with ALB target-group tuning in [.github/workflows/deploy.yml](../.github/workflows/deploy.yml): `deregistration_delay.timeout_seconds=30` (was AWS default 300) drops the old-task drain by ~270 s, `HealthyThresholdCount=2 / HealthCheckIntervalSeconds=10` (was 5/30) marks the new task healthy ~130 s sooner. "Refresh ECS service" wall-clock measured before tuning: ~8 min; expected after: ~3 min. Wall-clock claims in CLAUDE.md / [docs/batch_design.md](batch_design.md) / [infra/batch/README.md](../infra/batch/README.md) updated from the original ~25–30 min design estimate to the measured ~15 min full train job.
- **2026-05-21** — D13 hardened: Spot-reclaim & ECR-pull retry now defense-in-depth across all submitter paths. [src/batch/launch.py](../src/batch/launch.py)'s per-submission `RETRY_STRATEGY` (the active production path via `train-batch.yml`) gains a `CannotPullContainerError*` rule alongside the existing `Host EC2*` retry; the job-definition fallback now mirrors the same rules (was `attempts: 1`) so any non-launch.py submitter (manual `aws batch submit-job`, future tools) gets the same protection. [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml) becomes the source of truth — it hardcodes the canonical `retryStrategy` + `timeout` rather than carrying forward the existing definition, and `infra/batch/**` is added to its paths trigger so retry/timeout edits propagate on merge. [infra/batch/setup.sh](../infra/batch/setup.sh)'s seed (used only for brand-new accounts; idempotent on existing ones) updated in parallel. Doc fixes: [docs/batch_design.md](batch_design.md) corrected `SPOT_CAPACITY_OPTIMIZED` → `SPOT_PRICE_CAPACITY_OPTIMIZED` (the production value all along; the newer AWS strategy is a strict superset weighing capacity AND price) and now documents the two-layer retry contract.
- **2026-05-20** — D13 correction: the SOCI v2 cold-start optimization is **not active** on the live Batch path. Measured cold-start is ~258 s (≈120 s Spot fulfillment + EC2 boot + ECS register, ≈122 s full image pull, ≈10 s container start) vs the original ~60–90 s design target. Root cause: the default ECS-optimized AL2 AMI that Batch picks for the `ff-gpu-spot` CE (`ami-03dd7084ddd63d5d0`, ECS agent 1.102.2, Docker 25.0.14) does **not** run `soci-snapshotter-grpc`, so the ~18 SOCI indexes published to the `ff-training` ECR repo by [batch-image.yml](../.github/workflows/batch-image.yml) are ignored. Realistic floor with the snapshotter active is ~135 s (120 s prov + ~5 s lazy-start + 10 s container start). Two activation paths captured in [docs/batch_design.md §2a](batch_design.md): pin a Bottlerocket GPU AMI via launch template (snapshotter default on recent versions), or pin AL2 via launch template with userdata that installs `soci-snapshotter-grpc` and restarts containerd before the ECS agent claims the instance. Docs-only correction; no code or infra changed in this entry.
- **2026-05-20** — D15 added: attention-NN hyperparameter tuning via Optuna + Batch Spot fan-out (one g4dn.xlarge per position, mirrors D13). New `--mode={train,tune}` dispatch flag on [src/batch/train.py](../src/batch/train.py) so the existing training container handles both jobs without a second image or job definition. `_S3Checkpoint` round-trips the SQLite study DB so Spot interruptions resume on Batch's retry. Trial objective reads `min(result["attn_history"]["val_loss"])` — val-only, no test contamination. K/DST's `run()` gained a `config=` kwarg (preserves existing callers via default `None`); K still injects its runtime-only `attn_history_builder_fn` closure on top so the nested attention path keeps working under the tuner. Pre-PR hook gained a content-inspector (`is_additive_and_safe`) so additive-only diffs to `src/shared/{pipeline,training}.py` skip the benchmark-freshness gate. PR arc: #269 (MVP tuner + hook), #272 (Batch fan-out), this PR (K/DST extension + ADR).
- **2026-05-20** — D12 extended: per-position `run_pipeline()` now overlaps its CPU branch (Ridge tune+fit, ElasticNet tune+fit, LightGBM) with its GPU branch (base NN + Attention NN) via a 1-worker `ThreadPoolExecutor` in [src/shared/pipeline.py](../src/shared/pipeline.py). The two branches share read-only `X_train`/`y_train_dict`/`pos_*` and produce independent prediction dicts; the first place they meet is the `comparison` block, so the original sequential flow had no correctness reason to serialize. Wall-clock collapses from `cpu + gpu` to `max(cpu, gpu)` per position. Threading (not processes) because the CPU branch already nests `joblib.Parallel(prefer="threads")` for CV alpha grids and the GPU branch's PyTorch + CUDA both release the GIL — no pickling or process-spawn overhead. New `train_models_total` phase recorded in `phase_seconds` makes the overlap win visible alongside the existing per-phase timings.
- **2026-05-20** — D2/D6 consolidation follow-up (deliver the LOC reduction): removed the back-compat dual-write in `src/{pos}/config.py` (module-level `TARGETS` / `INCLUDE_FEATURES` / `NN_*` / `ATTN_*` / `LGBM_*` constants deleted; values inlined into `POSITION_CONFIG(...)` constructor calls), migrated ~90 importers to read from `POSITION_CONFIG.<field>`, deleted three deprecated analysis/tuning scripts (`analysis_rb_feature_signal.py`, `analysis_weather_vegas_correlation.py`, `tune_rb_gate.py` — flagged in `docs/rb_feature_history.md` as referencing pre-migration targets), genericized `tests/_pipeline_e2e_utils.py` (six near-identical position-specific tiny-config builders → one `build_tiny_config(pos)` reusing `build_pipeline_config` + `build_position_callables`), and unified `tests/wr/conftest.py` / `tests/te/conftest.py` under `register_standard_fixtures` (~270 test-site renames `wr_sim_df_factory` → `make_sim_df`, etc.). Net LOC delta -3,011 across 86 files (PRs [#242–#247](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/242) introduced the abstractions; this PR delivers the deletions). Parity guard `tests/shared/test_position_config_parity.py` deleted — the two views it pinned no longer exist. `CONFIG_TINY` / `CONFIG_TINY_ATTN` (K) and `ATTN_STATIC_CATEGORIES` (QB/RB/WR/TE) remain at module level as test contracts; `src/k/data.py` keeps local `SEASONS` / `MIN_GAMES` aliases (read 5+ times within the file).
- **2026-05-20** — D2 consolidation: introduced `src/shared/position.py::Position` (StrEnum) as the canonical position-code source of truth and wired it into `src/shared/registry.py::ALL_POSITIONS`, the aggregator's `POSITION_TARGET_MAP` keys, and `PositionConfig.name` (now validated at construction time). Drift-guard test in [tests/scripts/test_scope_positions.py](../tests/scripts/test_scope_positions.py) pins `Position` to `scope_positions.ALL_POSITIONS` (which stays string-only due to its CI-detect-job zero-dependency contract). Bulk of the codebase keeps using bare strings; `Position(str, Enum)` makes those interoperable (`Position.QB == "QB"`, `f"{Position.QB}" == "QB"`).
- **2026-05-20** — D2 consolidation: `src/shared/registry.py::get_inference_spec()` collapsed from six `if pos == "QB":` / `elif ...` branches (~330 LOC of dispatch) into a generic spec builder that reads from each position's `POSITION_CONFIG` and resolves the per-position callables via `importlib`. The legacy `_POSITION_META` dict went away with it — `accepts_dataframes` / `cpu_only` / `has_cv_runner` now live on `PositionConfig`. K's `target_signs` (sign vector used by the serving aggregator to combine the four raw-count heads into fantasy points) moved into `KickerConfig`. Registry shrank 482 LOC → 229 LOC; pipeline runners (PR 2) and registry (this PR) now share `POSITION_CONFIG` as the single source-of-truth.
- **2026-05-20** — D2/D6 consolidation: per-position pipeline runners (`src/{pos}/run_pipeline.py`) collapsed via `src/shared/position_pipeline.py::build_pipeline_config(pos, POSITION_CONFIG)`. The six ~150-line CONFIG-dict bundlers shrank to ~30-75-line thin shims that call the factory; LOC delta -436 across the runners. K and DST keep their per-position `run()` orchestration (data loading happens at the runner level, not in the shared pipeline) but consume the factory for the base CONFIG.
- **2026-05-20** — D2/D6 foundation: introduced `src/shared/position_config.py::PositionConfig` dataclass and a `POSITION_CONFIG` instance per `src/{pos}/config.py` that bundles every per-position hyperparameter. Module-level constants (`TARGETS`, `NN_BACKBONE_LAYERS`, ...) stay unchanged for existing importers; the dataclass is the new structural source of truth so upcoming registry / runner / pipeline consolidation can read positions generically instead of branching by name. Parity guard at [tests/shared/test_position_config_parity.py](../tests/shared/test_position_config_parity.py) pins the two views to identical values. Shared helpers (`alpha_grid`, `derive_attn_static_features`, `DEFAULT_OPP_DEF_HISTORY_STATS`, `DEFAULT_ENET_L1_RATIOS`) eliminate cross-position duplication of the alpha-grid construction, attention static-feature derivation, default opposing-defense stat list, and ElasticNet L1-ratio grid. No behaviour change — every existing test passes; loss/Huber-delta values and the `2.0/δ` coupling are untouched.
- **2026-05-20** — K diagnostic finding documented (PRs #231, #234): K's MAE is at the noise floor on existing data sources (condition number 8.33, max VIF 8.49, max |corr| with `fg_yards_made` only 0.157; `vegas_only` 2-feature Ridge matches all production models within 0.05 test MAE). Captured as a Fixed-archive entry in [TODO.md](../TODO.md); no D-entry change since the finding constrains future K feature work rather than altering an architectural decision. Future K MAE improvement requires a new input modality, not more features on the current inputs.
- **2026-05-20** — D6 extended: RB attention history + static branches now carry red-zone & goal-line touch signal sourced from PBP — five per-game cols (`redzone_carries`, `redzone_targets`, `inside10_carries`, `inside5_carries`, `redzone_target_share`) added to `ATTN_HISTORY_STATS`, plus two prior-season aggregates (`prior_season_total_redzone_touches`, `prior_season_mean_redzone_touches_per_game`) added to `INCLUDE_FEATURES["prior_season"]`. Motivation: attn-NN was losing to Ridge by +0.080 MAE on `rushing_tds` and +0.035 on `receiving_tds`; the `hurdle_poisson` architectural fix was rejected for regressing aggregate FP MAE (`[TESTED, REJECTED]` archive entry), so the upstream-signal route was taken instead. New loader `src/data/redzone_pbp.py::reconstruct_redzone_from_pbp` follows the same schema-gated cache pattern as `src/k/data.py` to avoid re-creating the stale-PBP-cache bug class. Columns are merged for all positions in `src/data/loader.py` (one PBP pass, position-agnostic) so QB/WR/TE can wire the same per-game stats into their own `ATTN_HISTORY_STATS` as one-line follow-ups.
- **2026-05-20** — D13 augmented further: the `tests.yml` `detect` job (~95 lines of inline bash for docs-strip / global / per-position / shared / fallback scoping) collapsed into the same [src/scripts/scope_positions.py](../src/scripts/scope_positions.py) helper via a new `--mode test` flag (`compute_test_shards`). All three CI `detect` jobs (`tests.yml`, `train-batch.yml`, `train-ec2.yml`) now share one tested source of truth; new shard rules are added under pytest in [tests/scripts/test_scope_positions.py](../tests/scripts/test_scope_positions.py).
- **2026-05-20** — D13 augmented: the train-batch / train-ec2 `detect` job's path → positions mapping moved out of inline bash into [src/scripts/scope_positions.py](../src/scripts/scope_positions.py), pinned by [tests/scripts/test_scope_positions.py](../tests/scripts/test_scope_positions.py). The contract (tests/ stripping, global-trigger fan-out, per-position scoping) is unchanged; the change collapses a verbatim duplicate across two workflows and locks the rules under pytest.
- **2026-05-20** — `BATCH_ACTIVE=true` flipped on as the push-driven default. D7/D9/§2 framings reconciled — Batch + Spot fan-out is now the active training path; the warm-EC2 implementation in D9 becomes the rollback path. `gh variable set BATCH_ACTIVE --body "false"` restores D7's warm-OD trainer on the next push.
- **2026-05-20** — D14 (serving prediction-cache + `post_fork` pre-warm) added. Resolves the first-request cold-start window left over from the predecessor attempt in PRs #148/#149; pairs background pre-warm with a fingerprint-keyed disk + S3 cache so the *first* container after a deploy is also fast.
- **2026-05-20** — D5 extended with `hurdle_poisson` loss family (zero-truncated Poisson on positives + BCE gate) as an available primitive alongside `hurdle_negbin`. RB sparse-count ablation (Variants D/E/Bf added to `src/tuning/ablate_rb_gate.py`) showed Variant E (hurdle_poisson on rushing_tds, receiving_tds, fumbles_lost) wins per-target MAE — count_sum 0.353 vs Ridge 0.369 — but regresses aggregate FP MAE +0.163 vs current Variant C. **Rejected for shipping**; primitive kept available for future use, current RB config unchanged.
- **2026-05-20** — D13 (Spot fan-out via AWS Batch) added; overrides D7's warm-EC2 choice when `BATCH_ACTIVE=true`. The warm-EC2 path remains as a one-flag fallback. PRs #215 (infra), #216 (cold-start opts), #217 (train-batch workflow).
- **2026-05-19** — D6 extended: RB attention history tokens now carry per-historical-game *game-script context* (`implied_team_total`, `implied_opp_total`, `is_home`, `days_rest`) and a *team box score* (7 team_* columns + `opp_team_points_scored`) in addition to the player's own stats. Per-game raw context is what the history branch is designed to consume — distinct from the rolling/EWMA/trend signals D6 keeps out of the static channel. New helper `src/shared/team_box_score.py::merge_team_box_score_features` materialises the box-score columns once per split inside `build_position_features`, so both training and inference paths see them.
- **2026-05-19** — D11 (smoke-test gate + always-stable manifest v2) and D12 (training-step perf composition; `torch.compile` measured and rejected on T4) added. D2 extended to note the user-facing PPR/Half-PPR/Standard scoring switch (PR #153). D4/D6 reconciled with K's `ATTN_L1_FEATURES` removal (PR #199) — K's attention static branch now matches the documented "no rolling features in the attention static channel" convention across all six positions.
- **2026-04-21** — D2/D4/D6 reconciled with attention now running on all six positions (K nested per-kick + outer per-game; DST standard attention), DST target migration from 5 mixed-bucket heads to 10 raw stats, and the per-position `{POS}_ATTN_STATIC_FEATURES` allowlist that blocks rolling features from leaking into the attention branch.
- **2026-04-19** — D7/D9 and §2 diagram reconciled with the EC2 training switch; the Batch path is preserved as standby (see [docs/batch_design.md](batch_design.md)).

## Table of Contents

1. [Context](#1-context)
2. [System Overview](#2-system-overview)
3. [Decision Log](#3-decision-log)
   - [D1: Temporal split](#d1-temporal-split-201223-train--2024-val--2025-test)
   - [D2: Multi-target decomposition with shared NN backbone](#d2-multi-target-decomposition-with-shared-nn-backbone)
   - [D3: Three-way model comparison, no ensemble](#d3-three-way-model-comparison-no-ensemble)
   - [D4: Attention over game history (all positions)](#d4-attention-over-game-history-all-positions)
   - [D5: Output-constraint stack for zero-inflated, non-negative targets](#d5-output-constraint-stack)
   - [D6: Explicit per-position feature allowlist](#d6-explicit-per-position-feature-allowlist)
   - [D7: EC2 warm instance over Batch/SageMaker for parallel training](#d7-ec2-warm-instance-over-batchsagemaker)
   - [D8: Two Docker images (slim Flask, heavy training)](#d8-two-docker-images)
   - [D9: Warm training host (rollback-path implementation)](#d9-warm-training-host)
   - [D10: Trunk-based CI/CD with test-gated deploys](#d10-trunk-based-cicd-with-test-gated-deploys)
   - [D11: Smoke-test gate + always-stable artifact (manifest v2)](#d11-smoke-test-gate--always-stable-artifact-manifest-v2)
   - [D12: Training-step perf composition (torch.compile rejected on T4)](#d12-training-step-perf-composition-torchcompile-rejected-on-t4)
   - [D13: Spot fan-out via AWS Batch (overrides D7 when BATCH_ACTIVE=true)](#d13-spot-fan-out-via-aws-batch-overrides-d7-when-batch_activetrue)
   - [D14: Serving prediction-cache + post_fork pre-warm](#d14-serving-prediction-cache--post_fork-pre-warm)
4. [Cross-Cutting Consequences](#4-cross-cutting-consequences)
5. [Open Issues / Follow-Ups](#5-open-issues--follow-ups)
6. [References](#6-references)

---

## 1. Context

**Problem.** Predict weekly fantasy football points for individual NFL players across six positions (QB, RB, WR, TE, K, DST) for the 2025 season, using 2012–2024 as training history. Primary output is a per-player point projection (regression); ranking metrics (top-12 hit rate, Spearman correlation) are derived from projections post-hoc.

**Constraints.**
- Solo personal project, ~2 weeks of initial execution.
- Small-sample ML regime: after position filtering and ≥8-games-per-season minimum, roughly 200–600 player-seasons per position — orders of magnitude smaller than datasets most modern NN architectures assume.
- Public data only — `nfl_data_py` ([nflverse](https://github.com/nflverse)) weekly stats, rosters, schedules, snap counts. Snap count coverage starts 2012, which bounds the training window.
- Documenting design decisions with technical trade-offs is what this ADR satisfies.

**Scope.**
- In: per-player weekly projections, three scoring formats (Standard / Half-PPR / Full-PPR), a Flask dashboard for lookup, automated training on AWS Batch.
- Out: lineup optimization, DFS salary-aware construction, betting odds beyond what's embedded as features.

**Forces driving the architecture.**
- The positions are not the same sport — QB and DST share maybe 10% of meaningful features. A one-model-fits-all approach under-fits all of them.
- Fantasy points are a *sum* of component stats (rushing yards, TDs, receptions). Modeling the sum directly wastes structure that modeling the components preserves.
- Training is cheap (~2 min on a GPU per position), but inference must run in a standard ECS container with no CUDA. This asymmetry shapes the whole deployment story.

---

## 2. System Overview

```
                    ┌─────────────────────────┐
  nflverse API ───▶ │ Data ingest + features  │ ─┐
  (2012–2025)       │  src/data, src/features │  │
                    └─────────────────────────┘  │
                                                 ▼
                    ┌──────────────────────────────────────────┐
                    │   Three model families (per position)    │
                    │  ┌────────┐ ┌─────────┐ ┌──────────────┐ │
                    │  │ Ridge  │ │ Multi-  │ │ Attention NN │ │
                    │  │ (base) │ │ HeadNet │ │ (all pos.)   │ │
                    │  └────────┘ └─────────┘ └──────────────┘ │
                    │  + LightGBM (selective positions)        │
                    └──────────────────────────────────────────┘
                                         │
                       Compared, not     │
                       ensembled ──────▶ │
                                         ▼
         ┌──────────────────────┐  ┌────────────────────┐
         │ 6× g4dn Spot (Batch) │  │ Flask app (serve)  │
         │ Dockerfile.train     │  │ Dockerfile (slim)  │
         │ one position / host  │  │ ECS, CPU-only      │
         │ ~25–30 min parallel  │  │                    │
         └──────────────────────┘  └────────────────────┘
                    │                          ▲
                    │   model artifacts → S3 → │
                    └──────────────────────────┘
```

> **Rollback path:** the warm-EC2 implementation ([docs/ec2_design.md](ec2_design.md)) stays provisioned and is reactivated by `gh variable set BATCH_ACTIVE --body "false"` on the next push. D13 explains why the active default flipped to Batch; D7/D9 cover the warm-EC2 fallback.

A training run is triggered by a push to `main`, which invokes [`.github/workflows/train-batch.yml`](../.github/workflows/train-batch.yml) (when `BATCH_ACTIVE=true`): the workflow submits six Batch jobs in parallel against the `ff-gpu-spot` Compute Environment — one per position on its own Spot g4dn.xlarge — blocks until they terminate, verifies fresh `model.tar.gz` artifacts landed in S3 per position, and commits a fresh `benchmark_history/{run_id}.json`. When `BATCH_ACTIVE != 'true'` the [warm-EC2 trainer](../.github/workflows/train-ec2.yml) fires instead and loops the six positions sequentially via SSM. The Flask service is built separately and deployed to ECS on every push to `main`; it reads pre-baked models from S3 and serves projections through a dashboard.

---

## 3. Decision Log

Each decision below follows the same structure: what was decided, the forces at play, options considered, the chosen option's trade-offs, why the rejected alternatives were rejected, and references to code.

---

### D1: Temporal split (2012–23 train / 2024 val / 2025 test)

**Decision.** Split by season, not by random row. Train on 2012–2023, validate on 2024, hold out 2025 for test.

**Context.** Weekly fantasy data is a time series with heavy week-over-week autocorrelation within a player (rolling features by construction). A random train/test split would leak week W+1's rolling stats into week W's label.

**Options considered.**

| Option | Complexity | Leakage risk | Data-efficiency |
|---|---|---|---|
| Random row split | Low | **High** (rolling features leak) | High |
| K-fold time-series CV | Medium | Low | Medium |
| Single season-based holdout (chosen) | Low | Low | Medium |

**Chosen: single season holdout.** Matches how the model will actually be used (next season is unknown; last season is the fairest holdout). Ridge hyperparameter tuning still uses expanding-window CV *inside* the 2012–2023 train window to avoid a single fold's noise — so we get some of the statistical benefit of K-fold without contaminating the test year.

**Rejected.** K-fold over seasons was considered but over-spent our limited post-2012 history and broke the "deployment mirror" intuition — at serve time we're always predicting a future we've never trained on, and a single holdout faithfully simulates that.

**References.** [src/data/split.py:6-37](../src/data/split.py), season constants in [src/config.py](../src/config.py). Knowledge cutoff to 2012 landed in commit `f400a5c`.

---

### D2: Multi-target decomposition with shared NN backbone

**Decision.** Decompose each position's prediction into a small set of *raw NFL stat* targets (yards, TD counts, receptions, interceptions, fumbles_lost) rather than pre-scored fantasy-point buckets. Train one neural network per position with a shared backbone and one head per target. Convert per-target predictions to fantasy points *after* the model runs via a deterministic aggregator ([src/shared/aggregate_targets.py](../src/shared/aggregate_targets.py)) that multiplies each raw-stat prediction by the corresponding coefficient in `src/config.py:SCORING_PPR` (or `SCORING_HALF_PPR` / `SCORING_STANDARD`). The aggregator is only applied at serving and benchmark-reporting time; each head trains on its own raw-stat label.

Current target sets (6 for QB/RB, 4 for WR/TE/K, 10 for DST):

| Pos | Targets |
|---|---|
| QB | `passing_yards`, `rushing_yards`, `passing_tds`, `rushing_tds`, `interceptions`, `fumbles_lost` |
| RB | `rushing_tds`, `receiving_tds`, `rushing_yards`, `receiving_yards`, `receptions`, `fumbles_lost` |
| WR | `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost` |
| TE | `receiving_tds`, `receiving_yards`, `receptions`, `fumbles_lost` |
| K | `fg_yard_points`, `pat_points`, `fg_misses`, `xp_misses` |
| DST | `def_sacks`, `def_ints`, `def_fumble_rec`, `def_fumbles_forced`, `def_safeties`, `def_tds`, `def_blocked_kicks`, `special_teams_tds`, `points_allowed`, `yards_allowed` |

K's four heads (`fg_yard_points`, `pat_points`, `fg_misses`, `xp_misses`) are out of scope for the raw-stat migration — they're already raw counts/sums by construction, just with per-head sign coefficients (`[+1, +1, -1, -1]`) applied at aggregation time. DST, by contrast, *did* migrate: the previous 5-head decomposition (mixing point-scaled buckets like `defensive_scoring` and `pts_allowed_bonus`) was replaced with the 10 raw stats above in commit `cc0c627`, so that DST now follows the same "heads predict raw events → aggregator prices them" discipline as QB/RB/WR/TE. The PA/YA bonuses are applied in the DST-specific branch of the aggregator ([src/shared/aggregate_targets.py:148-169](../src/shared/aggregate_targets.py)) via vectorized tier lookups against `src/dst/targets.py`.

**Context.** Fantasy points collapse heterogeneous events (a receiving TD is structurally very different from a passing yard) into a single scalar. Decomposing lets each head specialize, lets us apply different loss deltas per component, and — critically — keeps MAE reporting interpretable in native stat units ("the model is off by ±18 passing yards, ±0.4 passing TDs per game") rather than in ambiguous point buckets. An earlier iteration of this ADR had targets like `passing_floor = passing_yards × 0.04` and `td_points = pass_TD × 4 + rush_TD × 6` baked in; the migration to raw stats moved all scoring coefficients to one place and decoupled model error from scoring-format choice.

**Options considered.**

| Option | Complexity | MAE interpretability | Scoring-format flexibility |
|---|---|---|---|
| Single-target NN predicting total fantasy points | Low | Low | None (retrain per format) |
| Fantasy-point-component heads (`passing_floor`, `td_points`, …) | Medium | Medium (points, ambiguous) | None (formats baked in) |
| Shared backbone + raw-stat heads + post-aggregation (chosen) | Medium | High (yards / TDs / receptions) | Full (aggregator takes `scoring_format` arg) |

**Chosen: raw-stat heads with post-aggregation.** The backbone learns position-general features ("is this player healthy? on the field? getting opportunity?"). Heads specialize on individual countable events. The aggregator is the single source of truth for turning those events into fantasy points — swap `SCORING_PPR` for `SCORING_STANDARD` without retraining. Training supervises each raw-stat head independently (weighted Huber per target); no aux total-loss is applied, so the aggregator is only used at serving and reporting time.

Zero-inflated targets get a `GatedHead` (BCE gate on `stat > 0` plus a value head for the conditional mean). RB has three gates (`receptions`, `rushing_tds`, `receiving_tds`); WR and TE each have two (`receptions`, `receiving_tds`) — the `receptions` gate was added in the Variant C ablation (PR #96) since TE/WR have non-trivial 0-reception game mass. QB has none — QBs score too often for the zero-inflation argument to hold.

**Rejected.** Single-target models under-fit the structure — every head would implicitly have to learn "what is a TD" separately from "what is a rushing yard." Fantasy-point-component heads (the previous iteration) made MAE hard to reason about — a `td_points` MAE of 4.25 could mean either "off by ~0.7 TDs/game" or "off by ~1 TD/game and a PAT." Raw-stat targets are unambiguous.

**Consequence.** Inference-time totals flow through `src/shared/aggregate_targets.py:predictions_to_fantasy_points` — changing scoring coefficients is a single-file edit. The per-position adjustment functions (`compute_qb_adjustment`, `compute_fumble_adjustment`, etc.) are retired for QB/RB/WR/TE; their effects (interception penalty, fumble penalty) are now direct targets (`interceptions`, `fumbles_lost`) that the aggregator prices in. DST flows through the same aggregator (10 raw stats plus a tiered PA/YA bonus lookup, commit `cc0c627`); K retains its own aggregation path — the 4 heads stay as raw counts and only the sign vector is applied at aggregation time.

The serving layer turns this into a user-facing capability: as of PR #153 (`a533990`) the dashboard exposes a 3-pill PPR / Half-PPR / Standard toggle and caches three per-format prediction columns per model so users can switch without re-fetching. The `?scoring=` query parameter is honored on every endpoint and season-leader / chart / metric aggregations are re-derived from the same raw-stat predictions through the aggregator. The models themselves remain scoring-agnostic — only the aggregation step varies — so this was a pure serving-layer addition, no retraining.

**References.** [src/shared/aggregate_targets.py](../src/shared/aggregate_targets.py) (aggregator + `TARGET_UNITS` + `POINT_EQUIVALENT_MULTIPLIER`), [src/config.py](../src/config.py) (`SCORING_PPR`, `SCORING_HALF_PPR`, `SCORING_STANDARD`), [src/shared/neural_net.py:38-145](../src/shared/neural_net.py) (`MultiHeadNet`, `aggregate_fn` plumbing), [src/shared/training.py](../src/shared/training.py) (`MultiTargetLoss`), per-position `compute_{pos}_targets` in `src/qb/targets.py`, `src/rb/targets.py`, `src/wr/targets.py`, `src/te/targets.py`, and `{POS}/{pos}_config.py` (target lists + loss weights + Huber deltas in raw-stat units). Consolidated in commit `99d7086`; raw-stat migration follows.

---

### D3: Three-way model comparison, no ensemble

**Decision.** Train and report Ridge (L2 linear), multi-head NN, and LightGBM independently per position. Do not ensemble or stack.

**Context.** A core goal is comparing multiple model architectures quantitatively. Ensembling would dominate any single model's MAE, but it would also muddle the question the project is trying to answer.

**Options considered.**

| Option | What it answers | What it costs |
|---|---|---|
| Ensemble (weighted average) | "Lowest MAE possible" | Loses the per-architecture comparison |
| Stacking (meta-model) | Same as ensemble + meta-risk | Extra CV pass, minimal gain at this sample size |
| Independent comparison (chosen) | "Which architecture wins, and why" | Leaves some accuracy on the table |

**Chosen: independent comparison.** Ridge is the honest baseline (same feature matrix, same scaling, just L2 regression). The NN is the headline "custom architecture" deliverable. LightGBM is the "what could a boosted tree do with the same inputs" reference point. Reporting all three per-position surfaces real trade-offs: Ridge wins on stability and interpretability; NN pulls ahead on WR/TE where interactions matter; LightGBM is competitive where there's enough data and falls apart on K/DST.

**Rejected.** Ensembling was considered and rejected because it would obscure exactly the finding the project is trying to produce. (A future production system would of course blend these — that's a follow-up, not this ADR.)

**References.** [src/shared/models.py](../src/shared/models.py) (`RidgeMultiTarget`, `LightGBMMultiTarget`), [src/shared/neural_net.py](../src/shared/neural_net.py), [src/shared/pipeline.py](../src/shared/pipeline.py). LightGBM added in commit `f343c20`.

---

### D4: Attention over game history (all positions)

**Decision.** Every position (QB, RB, WR, TE, K, DST) replaces pure rolling features with a variable-length game-history branch processed by learned-query attention, fused with the standard static feature vector. All six positions set `{POS}_TRAIN_ATTENTION_NN = True` and share the `d_model=32` / `n_heads=2` attention shape; K additionally wraps a per-kick inner attention pool inside the outer per-game sequence.

**Context.** Rolling means lose order — "three good games then a bad one" looks identical to "one bad then three good." Attention over the last N games lets the model weight recent games higher, attend more to games pre-injury, and in principle learn role-change signals (backup becomes starter) that a fixed window can't capture.

**Options considered.**

| Option | Signal captured | Sample-efficient? |
|---|---|---|
| Pure rolling features | Mean/variance only | Yes |
| LSTM / Transformer over history | Order + interactions | No (overfits on ~300 rows) |
| Rolling + attention-pool over history (chosen for skill positions) | Both | Marginal |
| Rolling + attention-pool over per-game sequence (chosen for DST) | Both | Marginal |
| Rolling + nested (per-kick ⊕ per-game) attention (chosen for K) | Per-kick conditions + per-game order | Marginal |

**Chosen: learned-query attention pool on every position.** A small attention head (`d_model=32`, 2 heads) is cheap enough not to overfit, with positional encoding for recency. Originally the skill positions (QB/RB/WR/TE) shipped with attention and K/DST stayed on rolling-only; subsequent measurements showed that with the right history channels (raw defensive stats for DST, per-kick rows for K) the attention branch paid for itself on both. The implementations differ:

- **QB / RB / WR / TE** — outer attention over prior games; per-game row is the rolling-stat snapshot. Unchanged in spirit from the original skill-position design.
- **DST** (commit `cc0c627`) — outer attention over prior games using a 14-stat per-game sequence (the 10 raw targets plus 4 raw opponent-context columns `opp_scoring`, `opp_fumbles`, `opp_interceptions`, `opp_qb_epa`, see [src/dst/config.py:194-211](../src/dst/config.py)). No gated fusion, no gated TD head.
- **K** (commit `801b61a`) — **nested attention**: the outer attention is over prior games, but each game token is itself produced by an *inner* attention pool over up to `K_ATTN_MAX_KICKS_PER_GAME = 10` kicks within that game (with per-kick features like `kick_distance`, `fg_prob`, `is_q4`, `game_wind`, see [src/k/config.py:118-134](../src/k/config.py)). The inner pool summarizes per-kick conditions; the outer pool weights which prior games matter. This lets the model distinguish "3-for-3 from short range in a dome" from "3-for-3 in a blizzard from 50+" — a signal pure per-game rollups destroy. A subsequent refinement (PR #199, `dff43fb`) dropped the L1-rolling columns (`ATTN_L1_FEATURES`) that K had still been feeding into its attention static branch: once the inner per-kick attention pool was learning per-game aggregates directly, the L1 rollups were redundant signal, and keeping them violated the "no rolling features in the attention static channel" rule the other five positions already followed.

**Rejected.** A full LSTM or Transformer was tried conceptually (see [docs/archive/design_lstm_multihead.md](archive/design_lstm_multihead.md)) but rejected as over-parameterized for this regime. Kept the design doc as an artifact of the consideration. An earlier version of this ADR rejected attention on K and DST — that decision was reversed once the input schemas were redesigned to give the attention branch something real to chew on (raw per-kick rows for K, raw defensive stats rather than pre-rolled windows for DST).

**References.** [src/shared/neural_net.py](../src/shared/neural_net.py) — `GatedTDHead` (L9), `AttentionPool` (L122), `MultiHeadNetWithHistory` (L202, outer-only attention used by QB/RB/WR/TE/DST), `MultiHeadNetWithNestedHistory` (L431, inner-kick ⊕ outer-game attention used by K); [src/k/config.py:102-152](../src/k/config.py) (nested attention, per-kick stats, static allowlist); [src/dst/config.py:169-228](../src/dst/config.py) (DST attention + history stats + static allowlist); [docs/archive/design_lstm_multihead.md](archive/design_lstm_multihead.md). Evolved across commits `b31bdf7` → `c399c12` → `99d7086` → `cc0c627` (DST attention + raw-stat targets) → `801b61a` (K nested attention).

---

### D5: Output-constraint stack

**Decision.** Combine four constraints on NN outputs: (a) Huber loss with per-target deltas, (b) per-head `clamp(min=0)` controlled by a `non_negative_targets` set, (c) a gated TD head that models P(TD>0) and E[TD|TD>0] separately, (d) ±4σ feature clipping after StandardScaler.

**Context.** Fantasy targets have three nasty properties: they're zero-inflated (most players don't score a TD on a given week), non-negative (with one exception — DST `pts_allowed_bonus`, which runs −4 to +10), and have outliers (40+ point games do happen). Vanilla MSE regression with no output bound produces nonsense.

**Options considered.** Rather than a single option table, each constraint has its own rationale, and several replaced earlier bugs:

- **Huber over MSE.** Outlier games dominate MSE gradients. Huber with per-target delta (≈1.5–3.0) caps the penalty.
- **Clamp instead of Softplus.** An earlier version used Softplus on head outputs, which has a floor of `softplus(0) ≈ 0.693`. Across three heads that's a ~2-point floor no player could drop below, and it created a scale mismatch with Ridge's `np.maximum(·, 0)`. Clamp allows exact zeros. (Fixed in commit `fe507e0`.)
- **`non_negative_targets` parameter, not a global clamp.** DST's `pts_allowed_bonus` is legitimately negative when the defense gives up a lot of points. A global clamp broke DST; making the set configurable per-position fixed it.
- **Gated TD head.** TDs are discrete and mostly zero. Binary gate + value head reflects the actual data-generating process. (Added in commit `18170a6`.)
- **Hurdle loss families.** Two zero-inflated value losses available alongside the gate: `hurdle_negbin` (zero-truncated NB-2, fits overdispersed counts like receptions where var/mean ≈ 2) and `hurdle_poisson` (zero-truncated Poisson, fits dispersion-≈1 counts like RB TDs and fumbles_lost). Both train the value head on positives only, scaling by fraction-positive so loss magnitude stays comparable to neighbouring Huber/Poisson heads. `hurdle_poisson` was added 2026-05-20 specifically to mirror Ridge's `gated_ordinal` decomposition for sparse Poisson-shaped count heads.
- **±4σ feature clip.** Test-set outliers were producing z-scores up to ~19, sending NN outputs off a cliff. Clipping after scale catches 0.3% of values and prevents catastrophic extrapolation.

**Chosen rationale.** Each constraint was added in response to a specific observed failure, not as a precaution. This ADR captures them together because they form a *coherent* stack — remove any one and a specific failure mode returns. Choosing *which* hurdle family to use on which head is a per-position config call (see RB ablation in TODO.md archive).

**References.** [src/shared/neural_net.py:61-107](../src/shared/neural_net.py) (`non_negative_targets` set + per-head softplus), [src/shared/training.py](../src/shared/training.py) (`MultiTargetLoss` with Huber; `hurdle_negbin_value_loss` / `hurdle_poisson_value_loss` + their ZTNB/ZTP log-pmfs), [src/dst/config.py:128](../src/dst/config.py) (`NN_NON_NEGATIVE_TARGETS = set(TARGETS)` — after the commit `cc0c627` migration all 10 raw DST heads are non-negative, so the set is simply the full target list; the `pts_allowed_bonus` head that used to warrant DST opting out of the global clamp is no longer a head — its negative values are produced downstream by the tier-lookup in `src/shared/aggregate_targets.py`), feature clipping in [src/shared/pipeline.py](../src/shared/pipeline.py). The `GatedTDHead` is now parameterized over a list of gated targets (`RB` has two: `rushing_tds` + `receiving_tds`; `WR`/`TE` have one: `receiving_tds`; `QB`, `K`, and `DST` have none — see D2). See also the "Fixed" section of [TODO.md](../TODO.md) for each bug history.

---

### D6: Explicit per-position feature allowlist

**Decision.** Every position has an explicit `{POS}_INCLUDE_FEATURES` list. The feature engineer computes ~155 features; the trainer uses *only* what's on the allowlist. Adding a feature to training requires changing a config.

**Context.** Feature leakage is the single most common source of "my model works in training and collapses in production" in time-series ML. Opt-out allowlists (compute everything, exclude the bad ones) are easy to get wrong silently — one new feature that accidentally peeks into the current week breaks everything, and nobody notices until deployment.

**Options considered.**

| Option | Leakage resilience | Convenience | Auditability |
|---|---|---|---|
| All features in → trust the builder | Low | High | Low |
| Opt-out blocklist | Medium | Medium | Medium |
| Opt-in allowlist (chosen) | **High** | Lower | High |

**Chosen: opt-in allowlist.** A reviewer can diff a PR and see exactly what features the model sees. Adding a feature is a deliberate act. The inconvenience (a config edit per experiment) is the point — it forces intentionality.

**Extension: per-position attention-static allowlist.** When D4 grew to cover all six positions, the Ridge/base-NN allowlist turned out not to be enough. The attention branch already consumes temporal signal through its game-history channel, so feeding it the *same* rolling/EWMA/trend features again is both redundant and a leakage risk (the rolling features reach back farther than the attention window, so they silently smuggle older-season information past what's on the visible sequence). Landed in commit `2500ecc`: every position now defines a `{POS}_ATTN_STATIC_FEATURES` list that the pipeline consults when building the attention NN's static channel, distinct from the Ridge/base-NN `{POS}_INCLUDE_FEATURES`. Rolling, EWMA, and trend columns are explicitly kept out. Example: DST's attention static branch sees only 9 columns (`is_home`, `week`, `spread_line`, `total_line`, `rest_days`, `div_game`, `is_dome`, `prior_season_dst_pts_avg`, `prior_season_pts_allowed_avg`) — see [src/dst/config.py:218-228](../src/dst/config.py); K swaps its rolling features out for shift-1 "last-game" features ([src/k/config.py:140-152](../src/k/config.py)) because everything further back is in the inner kick sequence anyway.

K was the lone exception until PR #199 (`dff43fb`): when the convention landed in `2500ecc`, K still carried a separate `ATTN_L1_FEATURES` block that pushed L1-rolling columns into the attention static channel. The nested per-kick attention (D4) was already learning the same per-game aggregates the L1 rollups encoded, so the columns were redundant signal — and they violated the rule the other five positions followed. Removing them made the convention uniform across all six positions; this is the kind of residue a cross-cutting refactor leaves behind, and the lesson is to audit every position's config for it rather than trusting the touched-files list (TODO archive entry on the K `ATTN_L1_FEATURES` violation).

**Extension: context-augmented history tokens.** The static-vs-history split D6 establishes (rolling/EWMA/trend live in static, raw per-game stats live in history) doesn't pin down where *per-game raw context* belongs — Vegas implied totals, home/away, days rest, team-level box score for that game. Under the original RB schema these reached the model only through the static branch (carrying the *target week's* values), which meant the attention encoder had to assume every historical game shared the prediction week's environment. For RB that's a meaningful gap: usage is driven by game script, and a 12-carry game in a -7 road dog spot is different signal than the same line in a +10 home favorite spot. RB's `ATTN_HISTORY_STATS` now includes four game-script columns and eight team box-score columns alongside the player's own per-game stats; D6's "no double-counting" rule is satisfied because these are per-game raw values, not rolling/EWMA/trend. The columns are materialised by `src/shared/team_box_score.py::merge_team_box_score_features` (Vegas/home/rest already plumbed per-row by [merge_schedule_features](../src/shared/weather_features.py)) and consumed automatically by `build_game_history_arrays` once they're listed in the position's `ATTN_HISTORY_STATS`. QB/WR/TE/K/DST configs deliberately do not opt in — the pattern is RB-first, with other positions to follow if benchmarks warrant.

**Rejected.** Opt-out was the earlier pattern and was exactly how the feature-clipping bug and the schedule-features-at-inference bug slipped in. Allowlist refactor landed in commit `18170a6` alongside the gated TD change.

**References.** [src/qb/config.py](../src/qb/config.py) (`QB_INCLUDE_FEATURES`, `QB_ATTN_STATIC_FEATURES`), [src/te/config.py](../src/te/config.py), [src/dst/config.py:218-228](../src/dst/config.py), [src/k/config.py:140-152](../src/k/config.py), [src/shared/pipeline.py](../src/shared/pipeline.py). Weather/Vegas features (from [docs/archive/design_weather_and_odds.md](archive/design_weather_and_odds.md)) are opted in per-position through the same mechanism.

---

### D7: EC2 warm instance over Batch/SageMaker

> **Status (2026-05-20):** superseded by [D13](#d13-spot-fan-out-via-aws-batch-overrides-d7-when-batch_activetrue) as the default. The warm-EC2 implementation in D9 remains the rollback path (`BATCH_ACTIVE=false`); this entry is kept verbatim as the historical decision so D13's trade-off discussion is readable. D13 explicitly addresses the "warm EC2 always wins" framing below — the gap closes once you count parallelism across six positions, not per-position cold-start in isolation.

**Decision.** Train on a single warm EC2 g4dn.xlarge driven by CI. Six per-position training containers run in parallel on the instance via SSM commands, invoked by [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). AWS Batch with Spot is kept as a standby path ([docs/batch_design.md](batch_design.md)), reactivated by setting `BATCH_ACTIVE=true`.

**Context.** Per-position training takes ~2 minutes on a GPU. We went through three iterations: SageMaker first (commit `eedacfc`), then Batch + Spot (`57d52f9` → `ffb3119`), then the warm-EC2 design ([docs/ec2_design.md](ec2_design.md), landed 2026-04-19). Each pivot was driven by the same realization: a 2-minute training job amplifies cold-start overhead, so eliminating it is worth more than the per-hour savings. The follow-up (D13) measured that parallelism dominates cold-start once each position runs on its own host, which inverted the conclusion.

**Options considered.**

| Option | Cold-start | Cost pattern | Operational overhead |
|---|---|---|---|
| Train locally | 0 s | $0 | Blocks laptop ~12 min per full run; no audit trail |
| SageMaker Training Jobs | 3–5 min | $0.53/hr × 6 | Managed, but full cold-start every run |
| AWS Batch + Spot | 30–90 s (with pull-through + SOCI) | $0.16/hr × 6 ≈ $0.03/run | Scales to zero; own the IAM/ECR/queue |
| **EC2 warm instance (chosen)** | ~0 s (already running) | ~$0.53/hr while active, $0 while stopped via idle auto-shutdown | Single host to babysit; SSM is the only control plane |

**Chosen: EC2 warm instance.** The container is pre-pulled; the CUDA drivers are already loaded. `train-ec2.yml` just `aws ec2 start-instances` (no-op if already running) then fans six SSM commands out to the host. Per-run cost is effectively the 2 min of training plus the Actions runtime. An idle auto-shutdown timer ([infra/ec2/auto-shutdown.timer](../infra/ec2/auto-shutdown.timer)) stops the instance after 4 h quiet, bringing the idle cost to zero; the next push pays the start-up tax once and reuses the warm host for the rest of the day.

The commit↔model relationship is now one-to-one: every merge to `main` produces a measured, logged training run. Under Batch, cold-starts dominated observability — the Actions log was mostly "waiting for compute environment."

**Why Batch remains the standby path.** Batch is strictly better when training *dominates* wall time (long jobs) or when we genuinely want $0-idle with no manual stop semantics. For constant fine-tuning on a 2-minute job, the always-on-but-auto-stopped EC2 pattern dominates *sequential* training. We keep the Batch image pipeline live ([.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml)) so switching back is one `BATCH_ACTIVE=true` away.

**Superseded by D13 when BATCH_ACTIVE=true.** D13 rebuts the "warm EC2 always wins" framing of this entry by observing that the warm-EC2 path is *sequential* across positions (one T4 can't fit six concurrent NN runs). Fanning out across six Spot g4dn.xlarge instances — one per position — replaces sum(per-position) with max(per-position) and amortizes the per-host cold-start across the parallelism. The choice between D7's warm-OD path and D13's Spot fan-out is now a one-variable flip; both paths remain runnable.

**Rejected.** SageMaker (`eedacfc` → `57d52f9`): managed overhead without training-time dominance. Kubernetes (GKE/EKS): too much machinery for a single GPU job. Long-lived instance without auto-shutdown: leaves an expensive GPU running unused.

**References.** Active path: [docs/ec2_design.md](ec2_design.md), [infra/ec2/README.md](../infra/ec2/README.md), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml), [src/batch/train.py](../src/batch/train.py) (reused as the in-container entrypoint). Standby path: [docs/batch_design.md](batch_design.md), [src/batch/launch.py](../src/batch/launch.py). Commit arc: `eedacfc` (SageMaker) → `57d52f9` (pivot to Batch) → `ffb3119` (final Batch) → `4b96c41` / `deb3cc7` (EC2 wiring) → `ec5ab17` (SSM polling fix).

---

### D8: Two Docker images

**Decision.** Build and deploy two separate Docker images: a slim `python:3.12-slim` image for the Flask inference service (~150 MB) and a `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` image for GPU training (~5–6 GB). The heavy image is consumed by AWS Batch on the active path (D13) and by the EC2 warm host on the D7 rollback path; both pull from the same ECR tag.

**Context.** Inference runs CPU-only on ECS and does not need CUDA, `torch.cuda.*`, or the pytorch wheel's CUDA libs. Training needs all of them plus `nfl_data_py`, `lightgbm`, and the training scripts. A single image would either bloat inference (slow ECS deploys, higher cold-start) or strip training capability.

**Options considered.**

| Option | Inference image size | Training setup | Ops |
|---|---|---|---|
| One shared image | ~5–6 GB | Easy | Slow ECS deploys |
| Two images (chosen) | 150 MB + 5–6 GB | Explicit split | Two pipelines |
| Multi-stage build | Smaller, but fragile | Complex | Debug-hostile |

**Chosen: two images.** They have different requirements, different deploy cadences, and different failure modes. Keeping them separate means the Flask app can deploy without rebuilding torch, and a training dep bump doesn't ship to prod inference.

The training Dockerfile ([src/batch/Dockerfile.train](../src/batch/Dockerfile.train)) uses *explicit* COPYs rather than `COPY . .` to drop the Flask UI, scratch scripts, and analysis notebooks out of the image — see the comments on lines 25–38 of that file.

**Rejected.** Multi-stage builds that share a base were considered but rejected as debug-hostile: when a training run fails on Batch, the fastest debug path is `docker run` the exact training image locally. A multi-stage build obscures that.

**References.** [Dockerfile](../Dockerfile) (Flask), [src/batch/Dockerfile.train](../src/batch/Dockerfile.train) (Batch), [.dockerignore](../.dockerignore). Landed in commit `0e814a1`.

---

### D9: Warm training host

> **Status (2026-05-20):** implementation for the **rollback path** under D7. The warm host stays provisioned and is reactivated by `gh variable set BATCH_ACTIVE --body "false"` on the next push. Idle cost is ~$8/mo (EBS only) while stopped. The active push-driven trainer is D13's Batch fan-out.

**Decision.** Keep a single g4dn.xlarge EC2 instance warm with the training image already pulled and CUDA drivers loaded. Trigger per-push training from CI via SSM `RunCommand`, stream CloudWatch logs back to Actions, and stop the instance after 4 h of inactivity via a systemd timer. The Batch cold-start stack (ECR pull-through + SOCI + aggressive `.dockerignore`) is independently used by the active D13 path; it also helps the first EC2 image pull during user-data.

**Context.** D7 picked the warm-host pattern; this decision is the implementation. The old Batch design had to fight cold-start (image pull, instance provisioning, Docker startup) because Batch intentionally scales to zero. For a 2-minute training job, every second spent warming up is overhead we pay on every run. Leaving a GPU idle at $0.53/hr is also unacceptable — so the design has to stop the instance when it's genuinely unused.

**Chosen (composed).**

| Component | Mechanism | Why it matters |
|---|---|---|
| Deep Learning AMI (Ubuntu 22.04, PyTorch) | NVIDIA drivers, Docker, SSM agent, ECR credsStore pre-installed | First boot is ~90 s; subsequent starts are ~25 s |
| `ff-training:latest` pre-pulled, cached on root EBS | `docker pull` at user-data time, then again on every `systemctl start` | Container-start from the CI command is ~2 s (image is already present) |
| SSM `RunCommand` as the only control plane | No SSH, no open ingress, IAM-scoped per command | Security: instance has egress-only SG; auditability: every run is a logged SSM invocation |
| Per-position `ff-train` helper on PATH | `train-ec2.yml` fires 6 parallel SSM commands (one per position) | Re-uses the 6-parallel-position pattern from the Batch design unchanged |
| [`auto-shutdown.timer`](../infra/ec2/auto-shutdown.timer) | systemd timer fires every 15 min, stops instance if idle > 4 h | Brings idle cost to zero; next push pays one start-up (~25 s), subsequent pushes are warm |
| [`cloudwatch-agent.json`](../infra/ec2/cloudwatch-agent.json) | Ships `/var/log/ff-train/*.log` to `/ff/training` | Logs survive the instance stop/start cycle |

Net effect on a typical push: if the instance is already warm, training starts within seconds; if it was idle and stopped, the first push eats ~25 s of start-up and every subsequent push that day is warm. The total wall-clock time from `git push` to six `model.tar.gz` in S3 is ~3–5 min, of which ~2 min is actual training.

**Batch cold-start stack (now on the active path).** The Batch design planned three optimizations to minimize cold-start: ECR pull-through cache for the PyTorch base image (helps GHA build time, not runtime pulls from ECR to the Spot host), SOCI lazy loading (container starts before image fully pulled), and aggressive `.dockerignore` + explicit `COPY` in the training Dockerfile (~8 GB → ~5–6 GB image). **As of 2026-05-20, only the `.dockerignore` win and the SOCI index publish are realized** — the snapshotter that consumes the index is not running on the default Batch AMI (`ami-03dd7084ddd63d5d0`), so measured cold-start is ~258 s instead of the targeted ~60–90 s. Full breakdown and snapshotter-activation paths in [docs/batch_design.md](batch_design.md). The `.dockerignore` win is independent of the EC2 choice and also helps the first EC2 image pull during user-data when the rollback path runs.

**Rejected.** Dedicated-instance reserved-pricing: commits to 24/7 usage we don't need. Spot on EC2 with no auto-shutdown: interrupts mid-training. On-demand with no auto-shutdown: burns $0.53/hr through idle weekends. Lambda-backed GPU (not generally available at this size): no GPU, and would add a cold-start problem back.

**References.** [docs/ec2_design.md](ec2_design.md), [infra/ec2/launch-instance.sh](../infra/ec2/launch-instance.sh), [infra/ec2/user-data.sh](../infra/ec2/user-data.sh), [infra/ec2/auto-shutdown.sh](../infra/ec2/auto-shutdown.sh), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). Standby assets: [src/batch/build_and_push.sh](../src/batch/build_and_push.sh), [src/batch/Dockerfile.train](../src/batch/Dockerfile.train). Arc: `4145257` → `8a50eec` → `ffb3119` (Batch cold-start stack) → `4b96c41` → `deb3cc7` (EC2 warm-host implementation) → `ec5ab17` (SSM polling fix).

---

### D10: Trunk-based CI/CD with test-gated deploys

**Decision.** All deployments happen from `main`. Three GitHub Actions workflows: [tests.yml](../.github/workflows/tests.yml) runs on every push and PR; [batch-image.yml](../.github/workflows/batch-image.yml) builds and registers a new Batch job definition revision when training code changes; [deploy.yml](../.github/workflows/deploy.yml) builds and pushes the Flask image to ECS. Both deploy workflows gate on `tests.yml` passing.

**Context.** Solo project, short timeline. Branching models designed for teams add ceremony without benefit. What's actually needed is a ratchet: broken code can't reach production, every push is traceable to a green test run, every image is tagged by SHA for rollback.

**Options considered.**

| Option | Ceremony | Rollback | Fit for solo |
|---|---|---|---|
| Trunk-based + test-gated (chosen) | Low | SHA-tagged images | Excellent |
| Env branches (dev/staging/prod) | High | Revert + redeploy | Overkill |
| Manual deploy | None | Manual | Easy to skip tests |

**Chosen: trunk-based + test-gated.** Images are tagged by `${{ github.sha }}`; all historical tags stay in ECR for rollback. Batch job definitions are registered as new *revisions* (never deregistered), so rolling back is "submit a job with definition-name:revision-N-1."

**Rejected.** Environment branches would add a staging deploy with nothing behind it — in a solo project the "prod monitoring" is the dashboard on my laptop. Manual deploys were the original state; replacing them was the point.

**References.** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml). Landed in commit `ffb3119`.

---

### D11: Smoke-test gate + always-stable artifact (manifest v2)

**Decision.** Every training run lands its new tarball under a versioned `history/{ts}-{sha7}/` key, then runs an in-process smoke test that loads the artifact and runs a deterministic zero-input predict against every head; only on success does the manifest's `stable` pointer advance to the new key. The manifest schema (v2, defined in [src/shared/model_sync.py:46-49](../src/shared/model_sync.py)) tracks `stable`, `current`, `previous`, and a newest-first `history[]` capped at `HISTORY_KEEP_N`, so any past-good artifact can be promoted back manually via [src/scripts/promote.py](../src/scripts/promote.py). ECS reads `stable` at boot, so a failed gate means new tasks keep loading the previous good model while the new (broken) tarball sits in `current` and `history/` for forensics. S3 bucket versioning is layered underneath as defense-in-depth.

**Context.** D10's CI gate catches code regressions before they merge; it does not catch artifact regressions — a model that trains successfully but predicts NaN, or whose feature-column hash drifted past the scaler's, will pass pytest and then silently degrade the live dashboard. The weather/Vegas-missing-at-inference incident (TODO archive) is the canonical example: training-pipeline and serving-pipeline drift shipped past every test and only surfaced when users saw zeros in the dashboard. Prior to D11, any successful S3 upload silently became "live" to the next ECS task that booted. Production safety is a core claim of the project; this decision is what backs it.

**Options considered.**

| Option | Safety | Operator cost | CI/CD fit |
|---|---|---|---|
| Always advance — newest upload is live | Low | None | Matches D10's trunk-based ratchet |
| Manual promote step (operator approval before live) | High | Every push needs a human | Defeats D10 |
| **Smoke-test gate + stable/current/previous/history (chosen)** | High | None on success; rollback is a `promote.py` invocation | Compatible with D10 — gate runs in the same CI job |
| ECS canary deploy (split live traffic) | Very high | Needs traffic-splitting infra | Over-engineered for single-task ECS service |

**Chosen rationale.** The smoke test is *non-fatal*: `current` always advances (so failures are visible in the manifest for post-mortem), but `stable` only moves on success. The frontend resolves the artifact pointer at boot from `stable`, so a failed gate means new ECS tasks keep loading the prior good model while a human investigates. PR #179 (`8c42e88`) closed the operational loop: after a successful train, the workflow now issues `aws ecs update-service --force-new-deployment` so the newly-promoted `stable` is actually consumed by a fresh task instead of staying invisible to the long-running one. The smoke test itself ([src/shared/smoke_test.py](../src/shared/smoke_test.py)) catches the four failure modes that matter at promotion time: pickle/torch.load deserialization errors after class-path drift, state-dict shape mismatches (feature-count drift between training and the runtime registry), scaler `feature_cols_hash` drift caught by `assert_scaler_matches`, and NaN/Inf predictions on benign input from a collapsed head.

**Rejected.** Manual-promote breaks the trunk-based CI/CD ratchet from D10 — every push would need a human. Canary deploys need real traffic-splitting infrastructure absent on a single-task ECS service. Doing nothing and relying on tests proved insufficient (see Context).

**Consequence.** Adds ~5 s per training run for the smoke pass. Introduces a two-channel signal — "did training succeed?" splits into `current` (always advances) and `stable` (conditional). Operator scripts must understand which pointer they're reading: model_sync clients consume `stable`; forensics tooling and the `promote.py` listing read `history[]` and `previous`. Trades "newest model always live" for "no NaN-emitting model ever live" — the weather/Vegas archive entry quantifies what the prior behavior cost. The 5-deep `history` plus S3 bucket versioning means even a "two consecutive bad ships" scenario (`current` and `previous` both broken) is recoverable: `promote.py --list` shows older `history[]` entries, `--to <key>` rewrites the manifest atomically, and the legacy `model.tar.gz` mirror is updated for any pre-manifest consumer.

**References.** [src/shared/smoke_test.py](../src/shared/smoke_test.py) (smoke test entrypoint + `SmokeTestFailed`), [src/shared/model_sync.py](../src/shared/model_sync.py) (manifest v2 schema, `build_manifest`/`load_manifest`/`write_manifest`, `_sync_one` consumer with `stable`-preferred fallback), [src/batch/train.py](../src/batch/train.py) (upload → smoke → promote sequence), [src/scripts/promote.py](../src/scripts/promote.py) (operator rollback CLI), [src/shared/artifact_gc.py](../src/shared/artifact_gc.py) (history pruning), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml) (ECS force-new-deployment after train). Commit arc: `1b20e9e` (versioned history / PR #104) → `e8bf2a7` (promote CLI / PR #122) → `c7fa2d7` (smoke-test gate + bucket versioning / PR #130) → `8c42e88` (ECS force-rollover / PR #179).

---

### D12: Training-step perf composition (torch.compile rejected on T4)

**Decision.** Apply four cheap, orthogonal training-step optimizations on the EC2 training host — Feather-cached parquet reads, `DataLoader(num_workers=2, pin_memory=True)` on CUDA, `torch.backends.cudnn.benchmark = True`, and `optimizer.zero_grad(set_to_none=True)` — and emit phase-level timings as part of every benchmark JSON. *Do not* enable `torch.compile`: on T4 with this workload it costs +32% wall time and the wrapper is short-circuited at the call site.

**Context.** A 2-minute training job × 6 positions × every commit to `main` puts wall-clock training time directly on the critical path between `git push` and serving (D7/D9). The four cheap optimizations are uncontroversial — they target distinct bottlenecks (disk parse, host→device copy, conv-kernel search, gradient-zero traffic) and compose linearly. `torch.compile` was the obvious "free win" candidate but the actual measurement on the EC2 g4dn (PR #189, `3167b56`) showed a consistent regression, and a parallel test of `LGBM_N_JOBS=-1` (PR #188) was unwittingly bundled with the compile regression in the same image, masking its signal entirely until both were unwound (PR #196).

**Options considered.** Multiple perf knobs evaluated against the same EC2 g4dn baseline:

| Knob | Wall-time delta | Why |
|---|---|---|
| Feather-cached parquet + async DataLoader + cuDNN benchmark + `set_to_none=True` (chosen, PR #183) | **−40% to −60%** depending on position | Disk parse amortized across positions; H→D copy overlapped with forward pass; cuDNN picks the best conv kernel per attention shape after warmup; no zero-write traffic on the gradient buffer |
| Static-pad attention sequences — hold `build_game_history_arrays`'s fixed `[n, 17, game_dim]` tensors instead of stripping to variable-length lists for a custom collate (chosen, PR #200) | **−30%** on attention training (115.8 s → 81.2 s on RB) | Default PyTorch collate stacks fixed-shape tensors with zero overhead; masked positions contribute zero regardless, so math is equivalent. K's nested-attention path already used this pattern; PR #200 brought QB/RB/WR/TE/DST in line |
| Disk-backed feature-engineering cache (`.cache/features/`, content-hashed) (chosen, PR #200) | **86×** on prepare_data for cache hits (8.6 s → 0.1 s on RB) | `_prepare_position_data` is deterministic given `(position, train_df, val_df, test_df, cfg)` but is called N_folds × N_trials × N_positions times during Optuna and across CLI re-runs. Bypass with `FF_FEATURE_CACHE_DISABLE=1`; metrics bit-identical across runs |
| CPU/GPU branch overlap inside `run_pipeline()` via `ThreadPoolExecutor` (chosen) | `cpu + gpu` → `max(cpu, gpu)` per position | Ridge/ElasticNet/LightGBM (CPU; BLAS, sklearn, LightGBM C++) and base NN/Attention NN (GPU; CUDA) had no inter-model data dependency until the post-train `comparison` block. Nested closures `_cpu_branch` + `_gpu_branch` capture the existing phases and dispatch concurrently with a 1-worker pool. Threading (not processes) reuses the GIL-releasing properties of BLAS/CUDA already exploited by the inner `joblib.Parallel(prefer="threads")` CV grids. New `train_models_total` phase timing exposes the win |
| `torch.compile(model, dynamic=True)` on every NN forward | **+32%** | T4 (sm_75) has too few SMs to amortize the fused-kernel benefit at this batch size; variable-length history sequences cause Inductor to re-check guards each batch, drowning the gain |
| LightGBM `n_jobs=-1` baked into the training image (PR #188) | "test confounded by `torch.compile`" | Reverted in PR #196 — never genuinely measured because the compile regression masked the signal. Needs a clean re-test now that `torch.compile` is off |

**Chosen rationale.** Keep the four cheap wins; short-circuit `_maybe_compile` in [src/shared/pipeline.py:137-159](../src/shared/pipeline.py) so the path is dead code but the wrapper survives for a future hardware change (the docstring records the measurement and the conditions for re-enabling). LGBM threading is left as an open question — the default `LGBM_N_JOBS=1` is conservative but not currently regressed.

**Rejected.** `torch.compile` on T4 for variable-length sequences — re-evaluate if/when the training host moves to `sm_86+` (A10 or larger, more SMs). Bundling parallel perf experiments into one image — every "perf win" must be measured independently so a co-resident regression doesn't mask the signal (this is the lesson from PRs #188/#189/#196).

**Consequence.** Phase timings (`pipeline.nn_train`, `pipeline.attn_nn_train`, `pipeline.prepare_data`, `pipeline.ridge_tune`, etc.) are now part of every benchmark JSON ([benchmark_history/](../benchmark_history/)), making perf regressions visible position-by-position without an explicit perf-test job. This composes with the per-shard `detect` logic in [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml) so an unrelated regression on a non-changed position is still visible in the next run that touches it. The Feather cache adds one disk file per parquet under [src/batch/train.py](../src/batch/train.py)'s cache path — must be regenerated when a feature schema changes (the cache is invalidated by mtime, so re-running training after a parquet rewrite picks up the change automatically).

**References.** [src/batch/train.py:87-132](../src/batch/train.py) (Feather cache with mtime invalidation), [src/shared/pipeline.py:130](../src/shared/pipeline.py) (`cudnn.benchmark` toggle), [src/shared/pipeline.py:137-159](../src/shared/pipeline.py) (`_maybe_compile` short-circuit + restore instructions), [src/shared/training.py:278-320](../src/shared/training.py) (`num_workers=_NUM_WORKERS, pin_memory=True` across all four `DataLoader` sites), [src/shared/training.py:394](../src/shared/training.py) (`zero_grad(set_to_none=True)`), [src/shared/feature_cache.py](../src/shared/feature_cache.py) (disk-backed feature-engineering cache with content-hashed keys; `FF_FEATURE_CACHE_DISABLE=1` bypass), [src/shared/models.py:30](../src/shared/models.py) (`_LGBM_N_JOBS` env-var-only opt-in, default `1`). Commit arc: `48ef419` (the four wins + phase timings / PR #183) → `4ffad1f` (Inductor `g++` install / PR #187) → `cb3c960` (LGBM bake / PR #188) → `3167b56` (compile short-circuit / PR #189) → `35f0a57` (LGBM bake revert / PR #196) → `349aa4a` (static-pad attention + feature-engineering cache / PR #200).

---

### D13: Spot fan-out via AWS Batch (overrides D7 when BATCH_ACTIVE=true)

**Decision.** When the repo variable `BATCH_ACTIVE=true`, push-to-`main` training fires [`.github/workflows/train-batch.yml`](../.github/workflows/train-batch.yml), which submits six Batch jobs in parallel (one per position) against the `ff-gpu-spot` Compute Environment ([infra/batch/](../infra/batch/)). Each job runs on its own Spot g4dn.xlarge — total wall-clock is `max(per-position)` ≈ 25–30 min for a full retrain. When `BATCH_ACTIVE != 'true'`, D7's warm-OD path remains active. Both workflows have a `workflow_dispatch` break-glass that bypasses the gate.

**Context.** D7's "warm EC2 always wins" framing implicitly assumed *sequential* training: the warm host loops positions in a single SSM command because one T4 can't fit six concurrent NN runs ([docs/ec2_design.md:122](ec2_design.md)). That makes wall-clock `sum(per-position)` ≈ 120–156 min and means a `cancel-in-progress` is required to keep rapid pushes tractable — which means cancelled runs never reach their post-train benchmark commit and their results are lost.

Fanning out across six Spot hosts changes both numbers: wall-clock collapses to `max(per-position)`, and parallel runs from rapid pushes can coexist (Batch queues at the AWS level when the 24-vCPU Spot quota is saturated) so every push contributes a `benchmark_history` record. Cold-start (~258s measured 2026-05-20: ~120s Spot+boot, ~122s full image pull, ~10s container start; ~135s achievable once the SOCI snapshotter is activated on the AMI — see [docs/batch_design.md §2a](batch_design.md)) is paid once per fresh Spot host instead of once per sequential position — parallelism makes the per-host cold-start cost worth paying despite the gap to design target.

**Options considered.**

| Option | Wall-clock (full retrain) | Cost / run | Records every push? |
|---|---|---|---|
| Warm OD g4dn.xlarge, sequential (D7) | ~120–156 min | ~$1.10 (OD × ~2h) | No — `cancel-in-progress` drops cancelled runs |
| Warm OD, 6× parallel SSM commands | OOM | n/a | n/a (rejected, T4 can't host six NN runs) |
| Six Spot g4dn.xlarge, parallel (chosen for `BATCH_ACTIVE=true`) | ~25–30 min | ~$0.40 (Spot × ~30 min × 6) | Yes — no `concurrency:` gate; runs coexist |
| Six On-Demand g4dn.xlarge, parallel | ~25–30 min | ~$1.60 (OD × ~30 min × 6) | Yes | (rejected on cost) |
| ECS RunTask on Spot capacity provider | ~25–30 min | similar to Batch | Yes | (rejected — Batch scaffolding 80% built; ECS reinvents the wheel) |

**Chosen rationale.** AWS Batch was the original architecture ([docs/batch_design.md](batch_design.md)) before the pivot to warm EC2 cited cold-start dominating a 2-min training job (D7). Two factors closed that gap: (a) the cold-start blocker is mitigated by the SOCI index publish + the `ff-batch-lt` launch template that installs `soci-snapshotter-grpc` v0.13.0 on the AL2 Spot host via [infra/batch/userdata.sh](../infra/batch/userdata.sh); design target was ~60–90s, baseline 2026-05-20 measured ~258s (snapshotter inactive), expected ~135s once the launch template is attached to the live CE (post-merge measurement pending — see [docs/batch_design.md §2a](batch_design.md)); (b) more importantly, *parallelism dominates cold-start at six positions* — paying ~135s of cold-start once per Spot host while six positions train concurrently (~255s end-to-end including ~2 min training) is much cheaper than paying ~0s of cold-start six times in sequence on one host (~120 min total wall-clock). The Batch scaffolding (`src/batch/launch.py` with ThreadPoolExecutor + Spot-aware retry + `describe_jobs` polling) was 80% built before the EC2 pivot; this decision completes it.

The 24-vCPU Spot quota in us-east-1 is sized exactly for six g4dn.xlarge (`maxVcpus=24` in [infra/batch/setup.sh](../infra/batch/setup.sh)). Concurrent local launches will queue at `RUNNABLE` rather than push over-quota — graceful degradation. Spot interruptions auto-retry up to 3 times via [src/batch/launch.py](../src/batch/launch.py)'s `Host EC2*` / `CannotPullContainerError*` retry pattern (mirrored as the job-definition fallback in [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml) and [infra/batch/setup.sh](../infra/batch/setup.sh) so non-launch.py submitters get the same protection); worst-case wall-clock for one chronically-interrupted position is ~90 min, bounded by the workflow's `timeout-minutes: 180`.

**Rejected.** Keeping warm EC2 as the only path — the +120 min wall-clock penalty and the cancel-in-progress-induced loss of rapid-push benchmark records made it the wrong default once Spot fan-out became viable. Per-position CPU compute environment for K/DST — defers a useful optimization but doubles the provisioned infra surface for ~$0.005/run of compute savings; reconsider if the GPU compute ever becomes constrained. On-Demand fan-out — Spot is ~70% cheaper with bounded interruption-recovery cost.

**Consequence.** `train-batch.yml` has no `concurrency:` block, so parallel runs from rapid pushes coexist; the warm-EC2 path keeps its `cancel-in-progress: true` because the T4-contention constraint that motivated it is still real on that path. The `BATCH_ACTIVE` repo variable becomes a single-toggle switch with bidirectional semantics: in addition to gating [batch-image.yml](../.github/workflows/batch-image.yml)'s job-definition re-registration (legacy semantics), it now selects which training workflow fires on `workflow_run`. `workflow_dispatch` bypasses both gates so either path can be smoke-tested independently of the flag. The Batch infrastructure is provisioned once via [infra/batch/setup.sh](../infra/batch/setup.sh) (idempotent, ~80 lines mirroring [infra/ec2/launch-instance.sh](../infra/ec2/launch-instance.sh)); the cold-start optimizations are wired into [batch-image.yml](../.github/workflows/batch-image.yml) as auto-detected (pull-through cache only used when the rule exists) and `continue-on-error` (SOCI publish step) so neither blocks regular CI.

**References.** [docs/batch_design.md](batch_design.md) (now the active-path doc when `BATCH_ACTIVE=true`), [infra/batch/setup.sh](../infra/batch/setup.sh) + [infra/batch/README.md](../infra/batch/README.md) (provisioning), [infra/batch/userdata.sh](../infra/batch/userdata.sh) (SOCI snapshotter install — Option B activation), [src/batch/launch.py](../src/batch/launch.py) (`--positions`, `--skip-upload`, retry strategy), [src/batch/Dockerfile.train](../src/batch/Dockerfile.train) (`PULL_THROUGH_PREFIX` build-arg), [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml) (pull-through detection + SOCI publish, now load-bearing), [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml) (the CI workflow), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml) (now gated on `vars.BATCH_ACTIVE != 'true'`), [src/scripts/scope_positions.py](../src/scripts/scope_positions.py) + [tests/scripts/test_scope_positions.py](../tests/scripts/test_scope_positions.py) (the shared path → positions helper that both `detect` jobs call). PR arc: #215 (infra/batch/), #216 (cold-start opts in batch-image.yml), #217 (train-batch workflow + EC2 gate + launch.py `--skip-upload`), #283 (SOCI gap documented), this PR (Option B SOCI activation via launch template).

---

### D14: Serving prediction-cache + post_fork pre-warm

**Decision.** Move the serving warm-up off the user request path. Two layers: (1) a gunicorn `post_fork` hook spawns a daemon thread that calls `_ensure_metrics()` *after* the master has bound `:8000`, so warming runs in the background while the worker is already accepting requests; (2) the final assembled predictions + metrics are persisted to `data/serving_cache/{predictions.parquet, metrics.json, fingerprint.json}` and uploaded to `s3://{bucket}/{prefix}/predictions_cache/`. On subsequent container starts `sync_predictions_cache_from_s3` downloads the cache and `_try_hydrate_from_disk` short-circuits the model-load + inference path entirely when the live fingerprint matches the cached one. Fingerprint = SHA256 of `(relpath, size, mtime_ns)` over `src/{pos}/outputs/models/**` + `data/splits/*.parquet` + `data/raw/kicker_kicks_pbp_*.parquet`, so a model retrain or data refresh invalidates automatically — no manual versioning.

**Context.** The Flask app loads six positions' Ridge + NN + Attention-NN + LightGBM artifacts lazily on the first `/api/predictions` call (it cannot easily eager-load: the artifacts are large and only `model_sync` knows when they finish downloading from S3). That first call took 30–60 s, and every ECS task replacement re-exposed the wait to a real user via the full-screen "Loading models & generating predictions…" overlay in [src/serving/templates/index.html:42-49](../src/serving/templates/index.html). PR #148 (`d69f427`) tried pre-warming at module import under `gunicorn --preload` and broke ALB TCP health checks — the import runs *before* `bind()`, so the worker socket stayed unbound for the whole warm. PR #149 (`8ff26be`) reverted it. The captured lesson — "pre-warm in `post_fork` or a background thread, not at module import under `--preload`" — was correct but incomplete: a `post_fork` thread fixes the within-container case (all subsequent users after the warm finishes), but the *first* container after a deploy still serves its first user from a cold compute path. Pairing pre-warm with a deterministic, fingerprint-keyed disk + S3 cache closes that gap.

**Options considered.**

| Approach | First-container cold start | Within-container restart | Notes |
|---|---|---|---|
| Lazy load on first request (status quo) | 30–60 s | 30–60 s | One unlucky user per deploy. Simplest code. |
| Module-level eager warm under `--preload` (PR #148, reverted) | broken — ALB TCP-refused → task unhealthy | n/a (task never serves) | Bind happens after preload import. Strictly worse than lazy. |
| `post_fork` background thread (chosen, part 1) | 30–60 s for users who race the warm; fast after | 30–60 s for users who race; fast after | Hook returns immediately; daemon thread warms while worker accepts requests. Existing `_cache_lock` serializes any racing user request. |
| Disk + S3 prediction cache with fingerprint (chosen, part 2) | near-instant once any prior container has uploaded for the same fingerprint | n/a (worker hydrates at startup) | Predictions are deterministic given the trained models + test split; fingerprint key catches retrains automatically. Best-effort upload + download — failures fall through to recompute. |
| Pre-compute cache in the training pipeline | near-instant | near-instant | Cleanest in principle, but couples training to serving's feature build and breaks the "serving image is independent of training" property of D8. Deferred. |
| Streaming per-position responses to the frontend | unchanged total time, faster perceived | unchanged | UX-only; orthogonal to the actual cold-start cost. Not pursued. |

**Chosen rationale.** Background pre-warm + on-disk cache + S3 sync is the minimum set that makes "every container is fast" a real claim. The post_fork hook respects the PR #148/#149 lesson (warm runs after `bind()`); the cache covers the first-container-after-deploy case the post_fork hook alone can't. Fingerprint-keyed invalidation means the cache is correct-by-construction — a retrain rewrites the model files, the fingerprint changes, the consumer detects the mismatch and falls through to recompute + re-upload. Cache is stored under `data/serving_cache/` (gitignored) and under `s3://{bucket}/{prefix}/predictions_cache/` (env-gated via the existing `FF_MODEL_S3_BUCKET` / `FF_MODEL_S3_PREFIX`). Cross-process write race on cold-container first boot is benign: each worker writes a uniquely-suffixed temp file (`{pid}.{tid}.tmp`) and `os.replace`s it into place, so the final triple is whichever writer's `replace` ran last for each file — both consistent, both populate their own in-memory `_cache` from the compute they already finished.

**Rejected.** (a) Module-level pre-warm under `--preload` — strictly worse than lazy, see PR #148/#149. (b) Pre-computing the cache inside the training pipeline — breaks D8's "serving image is decoupled from training" by forcing training to know the serving feature build. The current design lets serving compute the cache on first boot and the training pipeline stays untouched; revisit if first-boot cost becomes a frequent operational concern. (c) Frontend streaming per-position — addresses perception, not latency, and the disk-cache path now makes even the cold case fast enough that progressive rendering isn't pulling its weight.

**Consequence.** A fresh ECS task with a populated S3 cache hits hydrate-from-disk in `_ensure_metrics` and returns from `/api/predictions` without ever touching `torch.load` / `joblib.load` / `_apply_position_models`. A retrain (new model files → new fingerprint) invalidates the cache automatically on the next consumer; the pre-warm thread recomputes + re-uploads. ALB health checks are unaffected — `/health` ([src/serving/app.py:1723](../src/serving/app.py)) is a cheap dict read and stays fast through the warm window, so the post_fork hook can't recreate the PR #148/#149 unhealthy-task failure. Operational cost: one additional `data/serving_cache/` directory in the container (a few MB) and three S3 objects per deploy. Tests gain a fixture rule — any fixture that touches `src.serving.app._cache` must also monkeypatch `_PREDICTIONS_CACHE_DIR` to a per-test `tmp_path`, or the production cache write at the end of `_compute_metrics_locked` leaks into `<repo>/data/serving_cache/` and accidentally hydrates later tests in the same xdist worker.

**Extension (in-flight per-position refresh).** Boot-time `sync_models_from_s3` only fires once per container life. To pick up new per-position artifacts that land in S3 after boot — the typical case during a parallel Batch fan-out where one position finishes minutes before another — `gunicorn.conf.py::on_starting` now also spawns a daemon thread via [`src/shared/model_sync.py::start_refresh_poller`](../src/shared/model_sync.py). It polls each `models/{POS}/manifest.json` etag every `FF_MODEL_REFRESH_INTERVAL_S` seconds (default **30**, set `0` to disable). On etag advance, `refresh_position` re-runs the same stable→current→previous fallback chain `_sync_one` already uses, but extracts to `models.new/` then atomically swaps via `os.rename(models→models.bak); os.rename(models.new→models); rmtree(models.bak)`, and finally touches `src/{pos.lower()}/outputs/.refreshed_at`. The serving runtime sees the new model on the next request: `_ensure_position_loaded` stats that sentinel and re-runs `_apply_position_models` when its mtime exceeds the value recorded at the last load (`_cache["positions_mtime"][pos]`). `metrics_by_format` is invalidated whenever ANY position refreshes since the aggregate spans all six. The bootstrap-first-call contract (`last_etag=None` records the etag without re-downloading) avoids redundantly re-syncing what `sync_models_from_s3` just downloaded at boot. Fail-soft everywhere — head/get/extract/rename failures log and return `did_refresh=False` with the live `models/` dir intact; the next poll retries. ECS `update-service --force-new-deployment` in [.github/workflows/train-batch.yml:238-255](../.github/workflows/train-batch.yml) is kept as belt-and-suspenders for the first ~2 retrain cycles and can be removed in a follow-up. Net effect: first-finished Batch position is live within one poll interval (~30 s) of its manifest write, decoupling per-position deploy-to-prod from the slowest position's training time + ECS rolling redeploy.

**References.** [gunicorn.conf.py](../gunicorn.conf.py) (`post_fork` daemon thread; `on_starting` refresh-poller spawn), [src/serving/app.py](../src/serving/app.py) (`_PREDICTIONS_CACHE_DIR`, `_compute_models_fingerprint`, `_try_hydrate_from_disk`, `_persist_cache_to_disk`; hydrate wired into `_ensure_metrics`; persist wired into `_compute_metrics_locked`; mtime-aware `_ensure_position_loaded` re-load on sentinel advance), [src/shared/model_sync.py](../src/shared/model_sync.py) (`sync_predictions_cache_from_s3`, `upload_predictions_cache_to_s3`, `refresh_position`, `start_refresh_poller`, `refresh_sentinel_mtime`, `_resolve_manifest_extract`), [Dockerfile](../Dockerfile) (CMD switched to `-c gunicorn.conf.py`, `COPY gunicorn.conf.py` added), [tests/test_app_predictions_cache.py](../tests/test_app_predictions_cache.py) + [tests/test_app_inflight_refresh.py](../tests/test_app_inflight_refresh.py) + [tests/shared/test_model_sync.py](../tests/shared/test_model_sync.py) + [tests/shared/test_model_sync_refresh.py](../tests/shared/test_model_sync_refresh.py), [TODO.md](../TODO.md) archive ("First-request cold start" entry) and predecessor ("Gunicorn `--preload` pre-warm broke ALB health checks" entry — PRs #148/#149).

---

### D15: Attention-NN hyperparameter tuning via Optuna + Batch Spot fan-out

**Decision.** Tune the attention-NN architecture + optimizer knobs with Optuna. Each Spot g4dn.xlarge runs one position's study (`n_jobs=1`); six positions fan out in parallel via the same Batch infrastructure D13 uses for training. The SQLite study DB round-trips to `s3://{bucket}/tune_nn/{pos}/study.db` on every trial completion and on `SIGTERM`, so a Spot reclaim resumes the search on Batch's retry instead of starting over. Loss-config (Huber δ, loss weights, `head_losses`, `gated_targets`) is **excluded** from the search; see Rejected.

**Context.** The existing [src/tuning/tune_lgbm.py](../src/tuning/tune_lgbm.py) runs Optuna for LightGBM on the warm-EC2 path (sequential, ~50 trials × ~5s each ≈ minutes per position). For the attention NN that doesn't compose: a single NN trial is 5–10× longer than an LGBM trial, and the warm path serializes positions. 30 trials × 6 positions × ~2 min/trial on one host ≈ 6 hours wall-clock per full retune; that's slow enough to break the "run it, eyeball it, iterate" loop the LGBM tuner enables. The attention NN was hand-tuned in lieu (edit `src/{pos}/config.py`, rerun `python -m src.{pos}.run_pipeline`, eyeball `benchmark_history/`), which the [ablate_rb_gate.py](../src/tuning/ablate_rb_gate.py) pattern partially formalized for hand-picked variants.

**Options considered.**

| Option | Wall-clock (full retune) | Complexity | Resumes on Spot reclaim? |
|---|---|---|---|
| Hand-tune via `src/{pos}/config.py` edits (status quo) | n/a (per-knob × 6 positions, human-driven) | low | n/a |
| Optuna on the warm-EC2 path like `tune_lgbm.py` | ~6 hr sequential | low | no |
| **Optuna + Batch Spot fan-out, one position per host (chosen)** | ~30–60 min (parallel) | medium | yes (S3 study-DB checkpoint) |
| Ray Tune with distributed sampler | ~30 min (with intra-position parallelism) | high (heavy dep, RDS for shared storage) | yes |
| Per-trial parallelism on one host | n/a — single T4 oversubscribes; trials thrash VRAM | high | n/a |

**Chosen rationale.** Fan-out by *position*, not by *trial*. The 24-vCPU Spot quota (D13) is sized for six g4dn.xlarge exactly; each container runs Optuna `n_jobs=1` because the single T4 oversubscribes if multiple trials race. This is the same shape `train-batch.yml` already proved out: same compute environment, same Docker image (the entry point dispatches on a new `--mode={train,tune}` flag on [src/batch/train.py](../src/batch/train.py)), same `Host EC2*` retry strategy. The S3 study-DB checkpoint closes the Spot-reclaim gap that would otherwise undo trial progress — on startup we pull `study.db` if present and Optuna's `load_if_exists=True` resumes from the last-completed trial; `remaining = n_trials - completed` accounts for the work already done so a retry doesn't re-run finished trials.

The trial objective is `min(history["val_loss"])` from the attention NN training curve (exposed via `result["attn_history"]`) — **val-only, no test contamination**. The pipeline reports val loss per epoch via the `epoch_callback` hook on `MultiHeadTrainer` (the only invasive change inside `src/shared/`), gated to attention trainer kinds so the regular NN's earlier phase doesn't bleed into the pruner's monotonic trajectory. HyperbandPruner (`min_resource=8`) kills clearly-bad trials at low epoch counts; without pruning, 30 NN trials × ~2 min each per position would dominate wall-clock past the Spot retry budget.

**Rejected.**
- **Searching loss-config (`HUBER_DELTAS`, `LOSS_WEIGHTS`, `head_losses`, `gated_targets`)**: `LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS` is a coupling, not two independent axes ([src/qb/config.py](../src/qb/config.py)). Searching deltas + deriving weights also blows up the dimensionality past what ~30 trials resolve. Hand-tune loss-config via `ablate_rb_gate.py` instead. Captured as a Stop rule in [CLAUDE.md](../CLAUDE.md).
- **Ray Tune**: heavyweight dep; the 24-vCPU Spot quota means the only parallelism worth distributing is across positions, which Batch already does without a new framework. Reconsider if the quota grows past one g4dn per position.
- **Warm-EC2 sequential like `tune_lgbm.py`**: 6× too slow for NN trial wall-clock. The LGBM tuner stays on that path because LGBM trials are short enough that the simpler infrastructure wins.
- **Shared PostgreSQL Optuna storage** (one study across hosts, true parallel sampling): operational overhead (RDS instance, connection strings, race-condition handling) unjustified for the solo-project scale. Per-position SQLite + S3 checkpoint gives the resilience without the moving parts.
- **`scheduler_type` in the search space** (`onecycle` ↔ `cosine_warm_restarts`): switching scheduler families requires the matching scheduler-specific cfg keys (`onecycle_max_lr`, `cosine_t0`, etc.) to be present, which the position configs don't guarantee. Out of scope until the cfg builder enforces both sets.

**Consequence.** A new opt-in workflow ([.github/workflows/retune-nn-batch.yml](../.github/workflows/retune-nn-batch.yml)) is `workflow_dispatch`-only — tuning is a research operation, not part of the push-driven training cadence. NO ECS refresh, NO `benchmark_history` commit (the workflow produces config *recommendations* printed as paste-ready `{POS}_ATTN_*` / `{POS}_NN_*` constants; operators hand-paste winners into the affected `src/{pos}/config.py` and let the normal training path produce the actual model on the next push). The matrix-per-position structure means one Spot failure surfaces independently in the Actions UI without cancelling the others; the aggregator job (`if: always()`) collates whatever made it to S3. All six positions are supported once K/DST's `run()` signatures accept `config=`. Container dispatch goes through a single `--mode={train,tune}` flag on `src/batch/train.py` so there's no second job definition, no second Docker image — D8's "two images, not six" property is preserved.

**References.** [src/tuning/tune_nn.py](../src/tuning/tune_nn.py) (Optuna study, HyperbandPruner, `_S3Checkpoint`, SIGTERM handler), [src/tuning/aggregate_results.py](../src/tuning/aggregate_results.py) (per-position merge), [src/tuning/launch_tune.py](../src/tuning/launch_tune.py) (Batch job submitter; reuses `wait_for_jobs` + `RETRY_STRATEGY` from [src/batch/launch.py](../src/batch/launch.py)), [src/batch/train.py](../src/batch/train.py) (`--mode=tune` dispatch), [src/shared/training.py](../src/shared/training.py) (`epoch_callback` hook on `MultiHeadTrainer`), [src/shared/pipeline.py](../src/shared/pipeline.py) (`_ATTENTION_TRAINERS` gate + `result["attn_history"]` exposure), [.github/workflows/retune-nn-batch.yml](../.github/workflows/retune-nn-batch.yml) (matrix per position + aggregator job). PR arc: #269 (MVP tuner + epoch_callback + pre-PR hook content-inspector), #272 (Batch fan-out + `--mode=tune` + `--checkpoint-s3` + workflow), this PR (K/DST `run(config=)` extension + ADR).

---

## 4. Cross-Cutting Consequences

**What becomes easier.**
- *Parallel iteration per position.* A change to RB features affects only RB's training job, only RB's models, only RB's tests. D2 (multi-head), D6 (allowlist), and D13 (parallel Batch jobs on the active path) compose into position-independent evolution.
- *Reproducible serving.* The Flask image is immutable and SHA-tagged (D10); models are baked in, not pulled at runtime. No "it worked yesterday" class of bugs.
- *Audit trail for leakage.* The allowlist (D6) plus the temporal split (D1) plus the ±4σ clip (D5) means any new feature has to survive three independent checks before it affects training.
- *Visibility into perf regressions.* Phase-level timings emitted under D12 make a slow-down at any pipeline stage visible in the next benchmark JSON without an explicit perf-test job.
- *Bounded blast radius for a bad artifact.* D11's smoke-test gate + always-stable manifest pointer means a NaN-emitting or shape-mismatched model lands in `current`/`history` but never replaces the live `stable` pointer; rollback is a single `promote.py` invocation against the manifest, no S3 surgery required.

**What becomes harder.**
- *Six configuration surfaces instead of one.* Each position has its own config, targets, loss weights. A framework-level change (e.g., a new regularizer) needs propagation to six places. This is a deliberate trade (D2/D6) but real.
- *Training-inference skew.* The Flask app must run the same preprocessing as the training pipeline, or models get zeros for features they were trained on. This happened once already (weather/Vegas features missing at inference, fixed after the fact — see [TODO.md](../TODO.md)).
- *Two images to maintain.* D8 doubles the Dockerfile surface; a requirements bump in one does not automatically propagate to the other.
- *Two-channel artifact pointer to reason about.* Under D11, "did training succeed?" splits into `current` (always advances) and `stable` (advances on smoke-pass only). Operator scripts and consumers must read the right pointer for their use case — model_sync clients want `stable`, forensics tooling wants `history[]`/`previous`.

**What we'll need to revisit.**
- K-position features use cross-season rolling windows (see [TODO.md](../TODO.md) "Open"). Technically a leakage source; currently justified by the specialist-role-stability argument, but worth re-measuring once the 2025 season completes.
- Single-format (PPR) training models. The scoring-format flexibility is only at the fantasy-points *computation* layer; the models themselves are trained on PPR. Retraining per format is straightforward but not automated.
- No lineup-construction layer. Per-player projections alone aren't a DFS product.

---

## 5. Open Issues / Follow-Ups

From [TODO.md](../TODO.md) "Open" section, mapped to decisions:

1. **K cross-season rolling leakage** — related to D1 (temporal split) and D5 (per-position non-negative targets). Requires either collecting more K games or accepting the bias.
2. **PPR-only training** — related to D2 (multi-target). Needs a training-matrix flag for scoring format and re-running six position pipelines.
3. **No lineup optimizer** — out of scope for this ADR; tracked as a follow-up project, not a revision of D1–D10.

---

## 6. References

### Source files by subsystem

- **Data & features:** [src/data/loader.py](../src/data/loader.py), [src/data/split.py](../src/data/split.py), [src/data/preprocessing.py](../src/data/preprocessing.py), [src/features/engineer.py](../src/features/engineer.py), [src/shared/weather_features.py](../src/shared/weather_features.py).
- **Models:** [src/shared/models.py](../src/shared/models.py) (Ridge, LightGBM, ordinal, two-stage, baselines), [src/shared/neural_net.py](../src/shared/neural_net.py) (MultiHeadNet, attention, gated TD head).
- **Training:** [src/shared/training.py](../src/shared/training.py) (MultiTargetLoss, trainer, schedulers), [src/shared/pipeline.py](../src/shared/pipeline.py) (pipeline orchestrator).
- **Per-position configs:** `src/qb/config.py`, `src/rb/config.py`, `src/wr/config.py`, `src/te/config.py`, `src/k/config.py`, `src/dst/config.py`.
- **Serving:** [src/serving/app.py](../src/serving/app.py), [Dockerfile](../Dockerfile).
- **Training infra (active, Batch):** [infra/batch/](../infra/batch/), [src/batch/launch.py](../src/batch/launch.py), [src/batch/train.py](../src/batch/train.py), [src/batch/Dockerfile.train](../src/batch/Dockerfile.train), [src/batch/build_and_push.sh](../src/batch/build_and_push.sh), [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml), [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml).
- **Training infra (rollback, EC2):** [infra/ec2/](../infra/ec2/), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). Reuses the same `src/batch/train.py` entrypoint and Dockerfile.
- **CI:** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [train-ec2.yml](../.github/workflows/train-ec2.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml).

### Related design docs

- [docs/method_contracts.md](method_contracts.md) — function signatures + data-layer contracts.
- [docs/batch_design.md](batch_design.md) — Batch + Spot fan-out design, cold-start optimizations, cost breakdown (authoritative for D13, the active push-driven path).
- [docs/ec2_design.md](ec2_design.md) — warm-host training design (authoritative for the D7 / D9 rollback path).
- [infra/aws/README.md](../infra/aws/README.md) — ECS + ALB + domain runbook (authoritative for D8 serving ops).
- [infra/ec2/README.md](../infra/ec2/README.md) — EC2 warm-host runbook (authoritative for D9 ops).
- [docs/archive/design_weather_and_odds.md](archive/design_weather_and_odds.md) — weather/Vegas feature rationale (folded into D6).
- [docs/archive/design_lstm_multihead.md](archive/design_lstm_multihead.md) — LSTM exploration, kept as artifact of the rejection under D4.
- [docs/archive/design_xgboost_ensemble.md](archive/design_xgboost_ensemble.md) — ensembling consideration, rejected under D3.
- [docs/expert_comparison.md](expert_comparison.md) — benchmark against published projections (evaluation evidence).
- [TODO.md](../TODO.md) — issue log (fixed + open).

### Commit timeline of inflection points

| Commit | Phase | What changed |
|---|---|---|
| `974f00d` | Prototype | Monolithic local pipeline |
| `f400a5c` | Data | Knowledge cutoff set to 2012 (D1) |
| `99d7086` | Modeling | Attention V4 + model consolidation (D2, D4) |
| `fe507e0` | Modeling | Softplus → clamp output constraints (D5) |
| `18170a6` | Modeling | Gated TD head + allowlist refactor (D5, D6) |
| `f343c20` | Modeling | LightGBM added (D3) |
| `eedacfc` | Infra | SageMaker attempt |
| `57d52f9` | Infra | Pivot to AWS Batch (D7) |
| `0e814a1` | Infra | Docker optimization — two images (D8) |
| `4145257`, `8a50eec` | Infra | Batch cold-start stack (D9) |
| `ffb3119` | Infra | CI/CD + test gating + final batch infra (D10) |
| `cc0c627` | Modeling | DST attention NN + migrate DST targets from 5 mixed-bucket heads to 10 raw stats (D2, D4, D5) |
| `2500ecc` | Modeling | Per-position `{POS}_ATTN_STATIC_FEATURES` allowlist — rolling/EWMA/trend features blocked from the attention-NN static branch (D4, D6) |
| `801b61a` | Modeling | K nested attention — outer over games + inner per-kick pool; attention now covers all six positions (D4) |
| `c7fa2d7` | Infra | Smoke-test gate + manifest v2 (stable/current/previous + history[5]) + S3 bucket versioning (D11) |
| `8c42e88` | Infra | ECS force-rollover after train so promoted artifacts get loaded (D11 closure) |
| `a533990` | Serving | PPR / Half-PPR / Standard end-to-end scoring switch (D2 extension) |
| `20cda09`, `668fa81` | Repo | Layout reorganization; `src/{POS}/` → `src/{pos}/` rename + symbol prefix drop |
| `48ef419` | Training | Feather cache + async DataLoader + cuDNN benchmark + phase timings (D12 wins) |
| `3167b56` | Training | `torch.compile` short-circuit — +32% on T4 (D12 rejection) |
| `0c66171` | Modeling | K/DST eval totals use signed/tiered aggregator (was reporting bogus `total_r2`) |
| `dff43fb` | Modeling | Drop K `ATTN_L1_FEATURES` — K now matches DST/skill convention (D4, D6) |
| `349aa4a` | Training | Static-pad attention (−30% attn train) + disk-backed feature-engineering cache (−86× on prepare_data hits) (D12) |
| `056423b` | Serving | Benchmark History tab — per-PR rows fetched from S3 at boot, auto-updates without redeploy |
| PR #215 | Infra | `infra/batch/` provisioning scripts (CE/JQ/JD/IAM) for Spot fan-out (D13) |
| PR #216 | Infra | `batch-image.yml` cold-start opts — ECR pull-through cache + SOCI v2 lazy-load (D13) |
| PR #217 | Infra | `train-batch.yml` workflow + `train-ec2.yml` gate on `BATCH_ACTIVE` + `launch.py --skip-upload` (D13) |
