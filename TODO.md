# Bugs & Potential Issues

Tracking known issues and uncertainties in the project. Resolved issues are kept as an archive at the bottom — each entry includes the lesson learned, which has repeatedly been useful context for reviewers and future work.

**Last reviewed: 2026-05-20.**

---

## Open

### [ACKNOWLEDGED] K features use cross-season rolling windows
- **File:** `src/k/features.py:27`
- **What:** Kicker features group by `["player_id"]` only (no season reset), so rolling windows span across seasons. A kicker's late-season 2024 stats influence early-2025 predictions.
- **Rationale:** Kickers have stable multi-year careers and small per-season sample sizes (~17 games), so cross-season windows provide more signal. Comment above the `grp` assignment records this rationale.
- **Risk:** If a kicker changes teams or role between seasons, stale cross-season features could mislead the model. Likely small impact.

### [LOW] Optuna parallel trials wired but disabled by default
- **File:** `src/tuning/tune_lgbm.py:448` (`study.optimize(n_jobs=args.n_jobs)`), CLI flag definition at `:378`.
- **What:** `--n-jobs` exists (default `1`) and is plumbed into `study.optimize`. The default keeps trials serial — nothing flips it on yet. Optuna's thread-based trial parallelism is safe with the existing SQLite storage at `:432`.
- **Why not flipped:** Needs a wall-clock benchmark on the tuning host to confirm a win, and the value must be coordinated with the LGBM tree-learning `n_jobs=-1` (set on Linux in `src/shared/models.py:_LGBM_N_JOBS`) to avoid oversubscribing a 4-vCPU box. On EC2 g4dn picking `--n-jobs=2` (with LGBM at `-1`) is the obvious starting point; `--n-jobs=-1` will fight the per-trial tree-learning threads.

### [LOW] `_cache` dict grows without eviction
- **File:** `app.py:63`
- **What:** `_get_data()` caches results in a module-level dict (serialized by `_cache_lock` since #31). The cache is never cleared. Not a real problem in practice (server restarts frequently), but worth noting.

### [LOW] `drop_last=True` silently discards training samples
- **File:** `src/shared/training.py:167, 196, 449`
- **What:** Last incomplete batch is dropped in all three DataLoaders (attention, multi-target, history-multi-target). With batch_size=512 (WR) and 32,521 training rows, 121 rows (~0.4%) are never seen. Standard practice, but combined with early stopping means those rows never contribute.

### [LOW] K targets overwrite `fantasy_points` column
- **File:** `src/k/targets.py:31`
- **What:** `df["fantasy_points"] = df["fg_yard_points"] + df["pat_points"] - df["fg_misses"] - df["xp_misses"]` overwrites the original column. Safe if called once, but calling twice would use the already-computed value. Not a current bug, just fragile.

### [LOW] Redundant NaN handling in feature engineering
- **Files:** All `*_features.py` files
- **What:** Pattern like `(a / b).fillna(0)` followed by `df.loc[b == 0, col] = 0` is redundant — fillna already handled the division-by-zero case. Not wrong, just noisy.

### [UNCERTAIN] K/DST index collision in `_get_data()`
- **File:** `app.py:593-606`
- **What:** K/DST test rows are appended to `results` with `offset = results.index.max() + 1`. Assumes the general test data has a well-behaved index. If the general test parquet has gaps, K/DST indices could collide. Probably safe in practice since parquet preserves sequential indices.

### [UNCERTAIN] Team share features computed per-split
- **Files:** `src/rb/features.py:74`, `src/wr/features.py:58`, `src/te/features.py:51`
- **What:** Team carry/target shares are computed within each split independently (`compute_team_*_totals` runs on each split's own data). A player's share could differ between train and test if their teammates are distributed differently across splits. By design (prevents leakage), but the share values won't be globally consistent.

### [LOW] Unify the three `detect` jobs into one shared helper
- **Files:** `.github/workflows/train-batch.yml` (detect), `.github/workflows/train-ec2.yml` (detect), `.github/workflows/tests.yml` (detect)
- **What:** `train-batch.yml` and `train-ec2.yml` already share `src/scripts/scope_positions.py` (pinned by `tests/scripts/test_scope_positions.py`). `tests.yml`'s detect job has divergent semantics — docs/markdown stripping, a `shared` shard fallback, and a different global-trigger list (`conftest.py`, `pyproject.toml`, `tests/_pipeline_e2e_utils.py`, etc.) — so it still computes its scoping in inline bash.
- **Why not now:** The tests detect mapping is genuinely different (it emits *shard names*, including a `shared` shard, and falls back to all-shards on uncertainty). Sharing one helper would need a config-driven mapping, not the train detect's hardcoded list. Worth doing once the train side's contract has bedded in, but no rush.

---

## Archive (Fixed)

Kept for the lessons-learned value — each entry captures a debug-to-root-cause cycle and a one-line takeaway that's been useful when modifying related code.

### [FIXED] K is at the noise floor — feature engineering on existing inputs will not improve MAE
- **Files:** `src/analysis/analysis_k_feature_audit.py` (new in PR #231, `df82235`), `src/analysis/analysis_k_signal_floor.py` (new in PR #234, `92a087c`). Outputs (gitignored) at `analysis_output/k_feature_audit.{json,png}` and `analysis_output/k_signal_floor.{json,png}`.
- **What:** Every K model family was posting negative R² and clustered MAE around the same number — Ridge 4.079 / NN 4.128 / Attn 4.132 / LGBM 4.061 in [benchmark_history/2026-05-20T06-39-23_4af6a9d.json](benchmark_history/2026-05-20T06-39-23_4af6a9d.json). Two competing explanations: (a) multicollinearity collapsing the design matrix, or (b) the input feature set genuinely doesn't carry enough signal about per-game K performance to beat naive baselines. Distinguishing (a) from (b) is necessary before proposing any feature additions, since (a) is fixed by dropping redundant columns and (b) is not fixable from the same data sources.
- **Fix:** Diagnosed and documented as noise floor — no code change to `src/k/`. **PR #231 (multicollinearity audit)** on training rows (4574 K games) found the feature matrix is well-conditioned: condition number 8.33, max VIF 8.49 on `total_k_pts_L3`, max |Pearson| 0.778 for `(fg_attempts_L3, total_k_pts_L3)`, zero columns meeting the conservative drop bar (`|r| > 0.95` AND `VIF > 10` AND pre-registered). Top |corr| with leak-free target `fg_yards_made` was only 0.157 across all 19 features — i.e. no single feature explains more than ~2.5% of the target's variance. **PR #234 (signal-floor diagnostic)** on 2025 test rows (n=529) confirmed (b) directly: a 2-feature `vegas_only` Ridge (`implied_team_total + is_dome`) hits test MAE 3.986, an in-script Ridge on all 19 features hits 3.957 (only 0.029 better), and **every production model is *worse* than the 2-feature baseline** (Ridge 4.079 / NN 4.128 / Attn 4.132 / LGBM 4.061). The categorical recommendation emitted was `accept_noise_floor`.
- **Lesson:** When MAE is flat across model families *and* the design matrix is well-conditioned *and* the best model barely beats a 2-feature Vegas baseline, the bottleneck is the **input data**, not the model or the feature engineering. Adding more rolling/EWMA/trend variants of existing K features will not move the floor — the per-game variance is dominated by inputs the project doesn't currently measure (long-snapper / holder identity, individual-game stadium altitude and precipitation, opponent block rate, kicker-specific surface preferences, etc.). Future K work should not propose feature additions from existing data sources without first introducing a new input modality, and any K MAE improvement claim should be measured against the `vegas_only` floor (~3.99) not against the position mean (~4.09). The audit scripts double as regression checks: re-run after any K feature change to confirm the floor hasn't moved.

### [FIXED] K underprojection: stale PBP cache survived `fg_yards_made` schema addition
- **Files:** `src/k/data.py` (`_REQUIRED_PBP_COLUMNS`, `_cached_pbp_is_current`, schema-gate at the cache-hit branch), `tests/k/test_data_loaders.py` (updated `test_reconstruct_weekly_from_pbp_cache_hit`, new `test_reconstruct_weekly_from_pbp_stale_cache_regenerates`, new `test_compute_targets_fg_yard_points_non_zero_after_load`). Cache regen: deleted local `data/raw/kicker_pbp_2015_2024.parquet` (Apr-19) and `s3://ff-predictor-training/data/raw/kicker_pbp_2015_2024.parquet`.
- **What:** K projections collapsed to ~3–4 fpts vs. an actual starting-K range of 7–10 fpts. Every model family (Ridge, NN, attention NN, LightGBM) converged on the same `fg_yard_points` MAE = 6.815 / R² = -1.785 ([benchmark_history/2026-05-20T05-07-41_47b61dd.json](benchmark_history/2026-05-20T05-07-41_47b61dd.json)) — degenerate predict-zero behavior. Root cause: the Apr-19 `kicker_pbp_2015_2024.parquet` (local + S3) was written before the `fg_yards_made` aggregation was added to `reconstruct_kicker_weekly_from_pbp`, so the cache had every other column but not that one. `reconstruct_kicker_weekly_from_pbp` short-circuited on the cached parquet (no schema check), `pd.concat` aligned columns and gave 2015-2024 rows `fg_yards_made = NaN`, `compute_targets.fillna(0) * 0.1` silently zeroed `fg_yard_points` across the entire train+val range, the model learned "kickers never score FG yards", and the 2025 test set (with real values, mean ≈ 6.81) blew up. AWS Batch reproduced the bug because `sync_raw_data` ([src/batch/train.py:191-209](src/batch/train.py:191)) downloads the stale cache from S3.
- **Fix:** Added `_REQUIRED_PBP_COLUMNS` (frozenset of every column the current aggregation produces) and `_cached_pbp_is_current` (reads the parquet schema via `pyarrow.parquet.read_schema` — no row reads — and returns False if any required column is missing). Cache-hit branch gated on both `os.path.exists(cache_path)` AND `_cached_pbp_is_current(cache_path)`, so a schema-drifted cache is logged and regenerated rather than served. Deleted both stale caches to immediately unstick production until the new code ships. Added a stale-cache regen test (mocked PBP path runs, asserts log line surfaces the missing column) and a `load_data → compute_targets → assert (fg_yard_points > 0).mean() > 0.5` distribution test so the same class of bug fails CI even when CI never hits the real cache.
- **Lesson:** Caches survive schema changes. A loader that short-circuits on file existence alone will keep serving the old schema until something downstream notices — which `fillna(0)` actively prevents from being an exception. Gate cache hits on a column-set check, not a file-exists check, whenever the producer's schema is allowed to evolve. Companion lesson: synthetic-fixture unit tests (`fg_yards_made: 55.0` baked into every test row) don't catch real-data feed bugs — distribution assertions on the post-`compute_targets` frame catch zero-collapse failures that look fine through the per-target unit tests.

### [FIXED] First-request cold start: `/api/predictions` blocked 30–60s after every ECS task replacement
- **Files:** `gunicorn.conf.py` (new — `post_fork` hook), `src/serving/app.py` (`_PREDICTIONS_CACHE_DIR`, `_compute_models_fingerprint`, `_try_hydrate_from_disk`, `_persist_cache_to_disk`, plumbed through `_ensure_metrics` / `_compute_metrics_locked`), `src/shared/model_sync.py` (`sync_predictions_cache_from_s3`, `upload_predictions_cache_to_s3`), `Dockerfile` (`CMD` switched to `-c gunicorn.conf.py`; `COPY gunicorn.conf.py`). Builds on the predecessor attempt in PRs #148 (`d69f427`) / #149 (`8ff26be`).
- **What:** The Flask UI overlay (`templates/index.html:42-49`) blocked behind `/api/predictions`, which lazily fanned out across six positions on first call — `joblib.load` + `torch.load` + feature build + inference for each — and took ~30–60 s on a fresh container. Every ECS task replacement exposed a single user to that wait. The earlier predecessor PR #148 tried pre-warming under `gunicorn --preload` and broke ALB health checks (entry "Gunicorn `--preload` pre-warm broke ALB health checks" above); since the revert in #149 the recommended path (`post_fork` hook or background thread) had not been re-attempted.
- **Fix:** Two-layer change. (1) `gunicorn.conf.py` `post_fork` hook spawns a daemon thread that calls `_ensure_metrics()` after the master has already bound `:8000` — health checks pass immediately, warming runs async, the existing `_cache_lock` serializes any user request that races the warm. (2) Predictions are persisted to `data/serving_cache/{predictions.parquet, metrics.json, fingerprint.json}` and uploaded to `s3://{bucket}/{prefix}/predictions_cache/`. On startup, `sync_predictions_cache_from_s3` pulls the cache and `_try_hydrate_from_disk` short-circuits the model-load + inference path entirely when the fingerprint matches. Fingerprint = SHA256 of `(relpath, size, mtime_ns)` over `src/{pos}/outputs/models/**` + `data/splits/*.parquet` + `data/raw/kicker_kicks_pbp_*.parquet`, so a fresh model retrain or data refresh invalidates automatically. New tests: `tests/test_app_predictions_cache.py` (fingerprint, persist+hydrate round-trip, mismatch, missing files, atomic write, `post_fork` thread spawn) and `tests/shared/test_model_sync.py` (`test_predcache_*` round-trip + error swallowing). Existing fixtures (`app_module`, `degraded_mode_app`) gained a `_PREDICTIONS_CACHE_DIR` patch to a per-test `tmp_path` so production cache writes can't leak between tests and accidentally hydrate later ones.
- **Lesson:** When a feature's hot path is deterministic given a small set of inputs, persist the output keyed on a fingerprint of those inputs — invalidation falls out naturally and the next cold start is free. The PR #148/#149 lesson ("pre-warm in `post_fork`, not at module import under `--preload`") remained correct but was incomplete: pre-warm helps within a container but doesn't help the *first* container after a deploy. Pairing post_fork with disk + S3 cache is what makes "every fresh container is fast" a real claim. Also worth noting: any module-level state that writes to a non-`tmp_path` location at compute time will silently leak between tests through shared `_cache` fixtures; the fix is to anchor write destinations on a monkeypatched constant rather than a hardcoded path.

### [TESTED, REJECTED] RB hurdle_poisson on sparse-count heads regressed FP MAE
- **Files:** `src/shared/training.py` (`ztp_log_prob`, `hurdle_poisson_value_loss`, `SUPPORTED_HEAD_LOSSES` expansion, dispatch in `MultiTargetLoss.forward`), `src/shared/pipeline.py` (hurdle-family downgrade path extended to cover `hurdle_poisson` for non-`GatedHead` models), `src/tuning/ablate_rb_gate.py` (Variants D / E / Bf + new per-target-sum decision rule). Benchmark JSON at `benchmark_history/ablations/2026-05-20T03-42-20_fdcba72_rb_td_gate.json`.
- **What:** RB attention NN was losing to Ridge `gated_ordinal` on all three sparse-count heads (rushing_tds NN 0.305 vs Ridge 0.234, receiving_tds 0.108 vs 0.064, fumbles_lost 0.074 vs 0.071 in the same-seed Variant C run). Ridge's gated_ordinal wins because it trains Stage-2 (ordinal regressor) *only on positives*, while the NN's Poisson NLL trains the value head on all samples — dragging mu toward the marginal (mostly-zero) mean. Added `hurdle_poisson` (zero-truncated Poisson on positives + BCE gate) as the NN-side architectural mirror.
- **Ablation result (seed=42, same-run Ridge baseline):**
  ```
  Variant            FP MAE   Rush TD   Rec TD   Fum lost   CntSum
  A (huber+gate)      4.418    0.271    0.081    0.037      0.389
  B (Poisson/no gt)   4.330    0.316    0.109    0.076      0.501
  C (Poisson+gate)    4.316    0.305    0.108    0.074      0.487  ← current ship
  D (hurdle_poisson)  4.522    0.254    0.067    0.074      0.395
  E (D + fumbles)     4.479    0.249    0.066    0.039      0.353
  Bf (C + fum gate)   4.305    0.314    0.099    0.072      0.485
  Ridge (gated_ord.)    -      0.234    0.064    0.071      0.369
  ```
  Variant E wins per-target MAE on all three count heads — beating Ridge on `fumbles_lost` (-0.032) and matching on both TD heads — but regresses aggregate FP MAE +0.163 vs Variant C. The yards heads also drift slightly (cross-target gradient interaction from the loss-family swap).
- **Decision:** Held back from shipping. The per-target wins are real but the +0.163 FP MAE regression is well above the project's de-facto sensitivity threshold (~0.05 pt/game). Empirical dispersion diagnostic confirmed ZTP was the right family — rushing_tds 1.16, receiving_tds 1.06, fumbles_lost 1.01 — so the experiment was correctly designed, the trade-off is just unfavorable. Kept the `hurdle_poisson` primitive available in `src/shared/training.py` and the D/E/Bf variants in `src/tuning/ablate_rb_gate.py` so future work can re-test if the aggregate-FP balance shifts.
- **Lesson:** Optimizing on per-target MAE for sparse heads can directly trade against aggregate FP MAE through cross-target gradient interaction in a shared-backbone model. A per-head architectural win doesn't compose linearly to a FP-MAE win — the head whose calibration improves may absorb gradient capacity that previously supported a different head's calibration. When evaluating a new loss family, *always* measure FP MAE alongside per-target MAE; the constrained criterion (per-target ≤ Ridge + tolerance, lowest FP MAE among qualifying) is the safer default than either metric alone.

### [FIXED] Push-to-`main` training was sequential across positions; one T4 capped full retrain at ~120 min
- **Files:** `.github/workflows/train-batch.yml` (new), `.github/workflows/train-ec2.yml` (gate added), `.github/workflows/batch-image.yml` (PULL_THROUGH_PREFIX + SOCI), `src/batch/launch.py` (`--skip-upload`), `infra/batch/setup.sh` + companion IAM JSONs + `infra/batch/README.md` + `infra/batch/teardown.sh`, plus updates to `docs/ARCHITECTURE.md` (D13), `docs/batch_design.md`, `docs/ec2_design.md`, `CLAUDE.md`. PRs #215, #216, #217.
- **What:** D7's warm-OD g4dn.xlarge ran the six positions sequentially in a single SSM command (`for POS in $POSITIONS; do /usr/local/bin/ff-train $POS $SEED; done`), because one T4 can't host concurrent NN training jobs. Wall-clock for a full retrain was ~120–156 min, and the workflow's `concurrency: cancel-in-progress: true` meant rapid pushes dropped in-flight runs before they could write a benchmark record.
- **Fix:** When `BATCH_ACTIVE=true`, push-driven training now fires `train-batch.yml` which calls `python -m src.batch.launch --positions $POSITIONS --seed $SEED --skip-upload`. `launch.py` submits six Batch jobs in parallel against the `ff-gpu-spot` Compute Environment (one g4dn.xlarge Spot per position) and blocks until all terminate. Wall-clock collapses to `max(per-position)` ≈ 25–30 min. Cold-start (~120s → ~60–90s on a fresh Spot host) is amortized by parallelism. `train-batch.yml` has no `concurrency:` block, so rapid pushes coexist and every push contributes a benchmark record. The warm-EC2 path remains as a one-flag rollback (`gh variable set BATCH_ACTIVE --body "false"`).
- **Lesson:** When the per-instance GPU is the bottleneck, scaling vertically (warm OD) hits a hard ceiling that scaling horizontally (per-position Spot) doesn't. The Batch path was originally rejected (D7) because cold-start dominated a 2-minute training job, but that math implicitly assumed *sequential* training. Once each position trains on its own host, max(per-position) replaces sum(per-position) and the per-host cold-start cost is parallelism-amortized down to a small fraction of total time. Cold-start optimizations (pull-through cache, SOCI v2) close the remaining gap. Bonus: the cancel-in-progress workaround for T4 contention goes away when there's no contention, so every push records a benchmark.

### [FIXED] Attention sequences were stripped to variable-length lists then re-padded per batch
- **Files:** `src/shared/pipeline.py` (dataset construction for QB/RB/WR/TE/DST attention paths), `src/shared/training.py` (custom collate removed in favor of default collate). PR #200 (`349aa4a`).
- **What:** `build_game_history_arrays` already returned fixed `[n, 17, game_dim]` zero-padded tensors plus boolean masks. The pipeline then stripped them back to a Python list of variable-length tensors, and a custom collate re-padded each batch to its local max. The net cost was an extra Python-level transform on every batch (the default PyTorch collate stacks fixed-shape tensors with zero overhead) and a denser-than-needed padding scheme when local maxes happened to exceed 17.
- **Fix:** Hold the fixed-shape tensors directly across QB/RB/WR/TE/DST (K's nested-attention path already used this pattern). Masked positions contribute zero to attention regardless of how many of them there are, so math is equivalent; only the per-batch RNG state shifts. Measured on a same-hardware stash comparison: `attn_nn_train` 115.8 s → 81.2 s on RB (~30% faster), per-target MAE within seed variance.
- **Lesson:** When a downstream consumer already produces fixed-shape outputs, don't reshape them back into variable-length structures just because earlier code did so. The K path was the canonical example for the other five positions — D12's "compose orthogonal cheap wins" only works if you periodically look for inconsistencies across positions and align them.

### [FIXED] `_prepare_position_data` re-ran feature engineering on every Optuna trial and CLI invocation
- **Files:** `src/shared/feature_cache.py` (new), `src/shared/pipeline.py` (wraps `_prepare_position_data` with the cache). PR #200 (`349aa4a`).
- **What:** `_prepare_position_data` is deterministic given `(position, train_df, val_df, test_df, cfg)` but was called N_folds × N_trials × N_positions times during Optuna and many times across CLI re-runs. The 8.6 s per call on RB was a real chunk of wall time, especially on local iteration loops.
- **Fix:** New `src/shared/feature_cache.py` wraps the call with an in-memory LRU + parquet/pickle disk cache under `.cache/features/`, keyed on SHA-256 of DataFrame content hashes + relevant cfg keys. Verified: first RB run miss (prepare_data=8.6 s), second run disk hit (prepare_data=0.1 s, **86× faster**). All metrics bit-identical across runs. Bypass with `FF_FEATURE_CACHE_DISABLE=1` when the cache itself is suspect.
- **Lesson:** A determinism check ("does this function produce the same output given the same input?") is the green light to memoize, but the key has to capture every input that varies. Content-hashing the DataFrames plus the relevant cfg subset is the safe key — hashing just the input paths or just `cfg` would silently return stale results when an upstream parquet rewrote.

### [FIXED] K `ATTN_L1_FEATURES` violated the per-position attention-static convention
- **Files:** `src/k/config.py` (`ATTN_L1_FEATURES` block removed).
- **What:** When the per-position `{POS}_ATTN_STATIC_FEATURES` allowlist landed in commit `2500ecc` (PR #140), every position was supposed to keep rolling/EWMA/trend features *out* of the attention NN's static channel — they're already represented in the game-history sequence and double-feeding leaks older-season signal past the attention window. K was the lone holdout: it kept a separate `ATTN_L1_FEATURES` block that pushed L1-rolling columns back into the static branch, on the theory the inner per-kick attention pool needed help. Subsequent measurement (PR #199) showed the inner pool was already learning those aggregates directly, making the L1 block redundant signal and a soft-leak risk.
- **Fix:** Dropped `ATTN_L1_FEATURES` from K's config; K now matches the QB/RB/WR/TE/DST pattern of no rolling features in the attention static channel.
- **Lesson:** When a cross-cutting convention lands across positions, audit *every* position's config for residue — not just the obviously-affected ones. Six configuration surfaces (D6 cross-cutting consequence) means six places to look, and a rollback to "one position is an exception" is exactly the kind of drift the convention was meant to prevent.

### [FIXED] LightGBM `n_jobs=-1` "perf win" test was confounded by a co-resident `torch.compile` regression
- **Files:** `src/batch/Dockerfile.train` (`LGBM_N_JOBS=-1` env removed in PR #196), `src/shared/models.py:30` (`_LGBM_N_JOBS` reads env, defaults to `1`).
- **What:** PR #188 (`cb3c960`) baked `LGBM_N_JOBS=-1` into the training image to try multi-core LightGBM on the EC2 g4dn. The benchmark didn't move, and the next commit (PR #189, `3167b56`) traced an unrelated +32% wall-time regression to `torch.compile` being enabled on T4. With both changes live in the same image, the LightGBM threading question was never genuinely measured — the compile regression masked any signal LightGBM might have shown.
- **Fix:** PR #196 (`35f0a57`) reverted the `LGBM_N_JOBS=-1` bake, returning the image to env-var-only opt-in (default `1`). The torch.compile short-circuit lands separately under D12. LGBM threading is left as an open perf question to measure cleanly later.
- **Lesson:** Don't ship two perf experiments in the same image. Every "perf win" must be isolated from co-resident regressions, or its signal is impossible to read. The auto-memory entry on "Check git log for SHA perf regressions" formalizes this — before crediting a perf result, search `git log` for nearby PRs that touched the same path.

### [FIXED] `torch.compile` cost +32% wall time on T4 (sm_75) with variable-length history
- **Files:** `src/shared/pipeline.py:137-159` (`_maybe_compile` short-circuited; wrapper survives for a future hardware change with restore instructions in the docstring).
- **What:** On the EC2 g4dn (T4, sm_75), wrapping the multi-head attention NN with `torch.compile(model, dynamic=True)` produced a consistent +32% wall-time regression. Root causes: T4 has too few SMs to amortize the fused-kernel benefit at this batch size, and the variable-length history sequences (different number of prior games per player-week) trigger Inductor guard re-checks on every batch, drowning whatever speedup the fused kernels deliver.
- **Fix:** Short-circuit `_maybe_compile` to a no-op, keeping the wrapper in place so a future move to `sm_86+` (A10 or larger, more SMs) can restore the call with one line. The docstring records the measurement, the conditions to re-evaluate, and the exact line to uncomment.
- **Lesson:** `torch.compile` is not a "free win" on small/older GPUs with dynamic-shape inputs — measure on the actual training hardware before keeping it. The four other perf knobs that landed in the same PR series (Feather parquet cache, async DataLoader, cuDNN benchmark, `zero_grad(set_to_none=True)`) all paid off on the same machine, so the loss was specifically `torch.compile`, not the perf-bundle. Phase-level timings (D12) now make this kind of regression visible in the next benchmark JSON without an explicit perf test.

### [FIXED] Train freshness check anchored on `now()` instead of train-start
- **File:** `.github/workflows/train-ec2.yml` (PR #197, `1f5cd68`).
- **What:** A workflow step that verified "all six positions produced a fresh tarball" computed its freshness threshold from `date -u +%s` *at check time*. Because the check ran after training completed, the threshold tightened as the job ran longer — a slow training run could fail its own freshness check by waiting for itself. The threshold needs to be anchored on when the training step *started*, not when the check runs.
- **Fix:** Captured the train-start timestamp into an output and referenced it in the freshness check.
- **Lesson:** Time-based freshness windows in CI must use the train-job start time as origin, not `now()`. Any computation that uses "current time" mid-workflow drifts in the wrong direction relative to the work it's validating.

### [FIXED] Serving tab choice didn't survive a page refresh
- **Files:** `src/serving/templates/`, JS tab-switch handler (PR #198, `d92362c`).
- **What:** The dashboard's tab switcher (predictions / wiki / etc.) held active state in JS memory, so refreshing the page reset to the default tab. Deep links to a specific tab didn't work.
- **Fix:** Mirror the active tab into `location.hash`; restore the tab from the hash on page load.
- **Lesson:** When a single-page-app has multiple visible "modes," put the mode in the URL hash, not in JS memory. Cheap to add and gives you deep linking for free.

### [FIXED] Wiki tables overflowed the page on narrow viewports
- **Files:** `src/serving/templates/`, wiki-tab CSS (PR #194, `5b2d880`).
- **What:** The Wiki tab (PR #138, `ce4543e`) renders repo markdown into the app. Markdown tables can be arbitrarily wide; without a container constraint they pushed the page layout past the viewport on narrow screens.
- **Fix:** Wrap each markdown table in an `overflow-x: auto` container.
- **Lesson:** When rendering external/user-provided markdown, native markdown→HTML doesn't constrain table width — wrap tables in a scrollable container at render time.

### [FIXED] Multiple serving-path layout breakages after the `src/{POS}/` → `src/{pos}/` rename
- **Files:** `src/serving/app.py` (`_repo_root`, `model_sync` path construction, `_attn_kwargs_static` keying). Fixed across PRs #155 (`5f7f8da`), #160 (`d7d86b5`), #164 (`f154ea8`), #171 (`795117e`).
- **What:** PR #154 (`668fa81`) renamed `src/{POS}/` → `src/{pos}/` and dropped the `{pos}_` prefix from files and symbols across 175 files. The mechanical refactor was correct; what it missed was *dynamic* path/key construction in the serving layer — places where code built `src/{POS_UPPER}/outputs/...` paths by string-concatenating an uppercase position string, or built dict keys like `f"{POS}_ATTN_STATIC_FEATURES"` expecting the old prefix. Four follow-up PRs were needed to track all of them down (model_sync local layout, output directories, `_repo_root` after the file move, and the attn kwargs keying).
- **Fix:** Each PR fixed one consumer; collectively they restored the serving path. The benchmark/pipeline path was untouched because it used the registry pattern, which had already been migrated to lowercase.
- **Lesson:** A mechanical rename will still catch downstream consumers that build paths or keys *dynamically* (string concat, dict keys from `f"{POS}_"` templates). After a rename PR lands, do not declare it complete until a full local serving boot + smoke-test passes — `pytest` alone misses the runtime path-construction sites because the renaming made the static analysis green.

### [FIXED] K attention scaler metadata was missing the full `attn_static_features` list
- **Files:** `src/k/run_pipeline.py` and scaler save path (PR #145, `e01507b`).
- **What:** K's attention NN scaler was being saved with a truncated `feature_cols` meta — only a subset of the actual `attn_static_features` it had been fit on. At inference, `assert_scaler_matches` (the canonical training/inference skew check) compared the truncated meta against the runtime feature list and either filtered columns or raised, depending on the runtime path. Either way the served K model was operating with a different feature set than the trained one.
- **Fix:** Write the full `attn_static_features` list into the scaler meta so `assert_scaler_matches` sees the same set the scaler was actually fit on.
- **Lesson:** Scaler metadata must always be written with the full feature list it was fit on. After a rename or schema change, rebuild the artifact rather than letting a stale meta file silently filter columns at inference. This is exactly the failure mode D11's smoke test now catches (`assert_scaler_matches` check inside `src/shared/smoke_test.py`).

### [FIXED] K/DST evaluation totals reported a nonsense aggregate `total_r2`
- **Files:** `src/shared/evaluation.py`, `src/shared/aggregate_targets.py` (PR #178, `0c66171`).
- **What:** `compute_target_metrics` computed an aggregate "total" metric for each position by summing per-head predictions, but for K it did an unsigned sum (treating `fg_misses` and `xp_misses` as if they added to fantasy points instead of subtracting), and for DST it skipped the PA/YA tier lookup that converts `points_allowed`/`yards_allowed` into the bonus dollars. The reported K `total_r2` was `-1.65` against a fictitious unsigned-K-points unit space; DST's totals were correct in shape but wrong in scale. Per-target metrics (per-head MAE/R²) were always correct — only the aggregate metric was bogus, and only for K/DST.
- **Fix:** Added a position-aware aggregator (`_k_predictions_to_fantasy_points` for K mirroring DST's pattern) and routed `compute_target_metrics` through `predictions_to_fantasy_points` for both K and DST. Per-target metrics unchanged; total metrics now match what `app.py` shows in the dashboard.
- **Lesson:** Any aggregated metric must route through the same aggregator that serving uses (`predictions_to_fantasy_points`), or it doesn't measure what the dashboard shows. The eval table in README's "Evaluation" section now reports the corrected K/DST MAE — which is naturally higher (K MAE went from a fictitious 3.6 to a real 6.7) because the previous numbers existed in the wrong unit space.

### [FIXED] Gunicorn `--preload` pre-warm broke ALB health checks during task replacement
- **Files:** gunicorn launch config / Dockerfile CMD. Tried in PR #148 (`d69f427`), reverted in PR #149 (`8ff26be`).
- **What:** To prevent 503s on ECS task replacement, PR #148 tried pre-warming the data + model caches at module import time under `gunicorn --preload`. Under `--preload`, the import happens *before* the worker binds its socket — so the ALB's TCP health check saw connection-refused for the entire pre-warm duration and marked the new task unhealthy before it could serve a single request. The intended fix (skip the 503 window) created an "unhealthy task" window that was strictly worse.
- **Fix:** Reverted (`8ff26be`). The right place for pre-warm work is a `post_fork` hook or a background thread that fires *after* the worker binds its socket — that path hasn't been re-attempted yet.
- **Lesson:** Under `gunicorn --preload`, module-import work runs before `bind()`. Anything slower than the ALB's TCP health-check timeout will produce a TCP-refused window that fails the deploy. Pre-warm in `post_fork` or a background thread, not at module import. (Captured in auto-memory as the "no module-level pre-warm under --preload" rule.)

### [FIXED] ECS `services-stable` waiter timed out under AZ rebalancing + grace-period drift
- **Files:** `.github/workflows/deploy.yml:76-99` (rewritten to use `aws-actions/amazon-ecs-render-task-definition@v1` + `amazon-ecs-deploy-task-definition@v2` with `wait-for-minutes: 20`), `infra/aws/bootstrap.sh:318` (+`--availability-zone-rebalancing DISABLED` on create-service).
- **What:** Deploy run [24770004436](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/actions/runs/24770004436) failed with `Waiter ServicesStable failed: Max attempts exceeded`. The ECS service had `availabilityZoneRebalancing=ENABLED` (an AWS-defaulted setting), so after a normal rolling deploy landed, ECS started a second task in another AZ to rebalance. During the rebalance, `runningCount=2, desiredCount=1`, and the waiter's default 40×15 s = 10 min budget expired before the extra task drained. A separate drift — `healthCheckGracePeriodSeconds=0` on the live service vs. `120` in bootstrap.sh — made the new task more likely to fail health checks during cold start and prolonged the window.
- **Fix:** Disabled AZ rebalancing on the service (single-replica can't span AZs anyway) and restored grace period to 120. Reconciled the live service with `aws ecs update-service --availability-zone-rebalancing DISABLED --health-check-grace-period-seconds 120`. Updated bootstrap.sh so a future recreate produces the same state. Also rewrote the deploy step to use the official ECS deploy actions with `wait-for-minutes: 20`, replacing the raw `aws ecs wait services-stable` call — so unrelated future slowdowns have more headroom.
- **Lesson:** `aws ecs wait services-stable` assumes `runningCount == desiredCount` holds steady. Any feature that transiently exceeds `desiredCount` (AZ rebalancing, capacity provider rebalance, deployment circuit breaker retries) will race the 10-min waiter on cold services. Two guardrails: (1) don't leave AWS auto-enabled features in place without understanding their deploy-time side effects — diff `describe-services` against `bootstrap.sh` after AWS service updates; (2) treat `bootstrap.sh` as source of truth and reconcile drift explicitly rather than letting it accumulate.

### [FIXED] NN aux total-loss removed entirely (was whitelisted per-position)
- **Files:** `src/shared/training.py` (MultiTargetLoss, MultiHeadTrainer), `src/shared/neural_net.py` (all three `MultiHeadNet*` variants), `src/shared/models.py` (Ridge/LightGBM `predict`), `src/shared/pipeline.py` (`_prepare_position_data`, four NN construction sites, benchmark attachments, baseline), `src/shared/evaluation.py` (`compute_target_metrics`, `plot_pred_vs_actual`), every position's config + `run_*_pipeline.py` (+ TINY variants), every position's training/model/neural_net test.
- **What:** Training used a `w_total · Huber(preds["total"], targets["total"])` aux term where `preds["total"] = sum(heads)` for QB/RB/WR/TE and `aggregate_fn(heads)` for K/DST (gated by a hardcoded `_FANTASY_POINTS_AUX_POSITIONS = {"K", "DST"}` whitelist). Since `preds["total"]` is a derived quantity with no extra parameters, the aux term was a calibration regularizer on the head sum — per-head Huber losses already constrained each head's mean, so the aux was mostly redundant. K's entry in the whitelist was also dead code: `run_k_pipeline.py` never set `aggregate_fn`, so `_nn_aggregate_fn("K", cfg)` always returned `None`.
- **Fix:** Dropped `w_total`, `huber_total`, the whitelist, `aggregate_fn` constructor param on `MultiHeadNet*`, the `preds["total"]` emission from Ridge/LightGBM/NN `predict`, and every `"total"` entry from `*_LOSS_WEIGHTS` / `*_HUBER_DELTAS` / `*_LOSS_W_TOTAL`. `aggregate_fn` stays registered for serving (`app.py:_combine_total`) and for benchmark-ranking reporting (`pos_test["pred_{model}_total"] = aggregate_fn(preds)` so DST ranking metrics compare on fantasy-point scale, not yards-dominated raw sum). Early stopping in `MultiHeadTrainer` moved from `val_mae_total` to a loss-weighted mean of per-target MAEs.
- **Lesson:** When a "loss term" is really just a calibration constraint on a derived quantity, ask whether the per-component losses already provide that signal. For multi-head regression with properly rebalanced per-head weights (see the `2.0/δ` convention entry below), the sum-of-heads aux is usually redundant — and it adds a failure mode where future contributors feel pressure to keep `sum(heads) ≈ fantasy_points` (which the earlier QB double-count regression already demonstrated can go wrong).

### [FIXED] Total aux loss double-counted adjustments
- **Files:** `src/shared/pipeline.py:208-211`
- **What:** Training total target was `fantasy_points` (includes INT/fumble penalties), but the model predicts `sum(heads)` (clean targets only). The total aux loss trained heads to absorb penalties. Then at inference, adjustments were added *again* via `adj.values`. Net effect: ~1.9 pts/game double-counted penalty for QB.
- **Fix:** Changed total target to `sum(pos_train[t].values for t in targets)`.
- **Lesson:** When a loss term compares a derived quantity (sum of heads) to a label, the label must match the derivation exactly. Any mismatch between what the model produces and what it's trained against will leak into predictions.

### [FIXED] Softplus floor inflated low-scoring predictions (twice)
- **Files:** `src/shared/neural_net.py:106, :407, :589` (all three `MultiHeadNet*` variants)
- **What:** `softplus(0) ~ 0.693` per head creates a floor that compounds across heads: ~2.08 pts for a 3-head position, ~2.8 pts for K's 4-head sum, etc. No player could be predicted below the floor. Also created a scale mismatch with Ridge (`np.maximum(..., 0)` allows exact zeros), biasing the ensemble.
- **First fix:** Replaced with `torch.clamp(val, min=0.0)` so heads can emit exact zeros.
- **Regression:** Commit `845d93b` reverted the clamp back to `F.softplus` at lines 106 and 407 and extended the softplus floor into the new `MultiHeadNetWithNestedHistory` at line 589 (so K inherited the bug). Also dropped a separate `torch.clamp(td_pred, min=0.0)` around GatedTDHead's output — left unchanged this time because GatedTDHead's `sigmoid × softplus(value)` is already in `[0, +∞)` and doesn't need an outer clamp.
- **Re-fix:** Restored `torch.clamp(val, min=0.0)` at all three sites.
- **Lesson:** Non-negativity constraints on output layers must allow exact zeros. Softplus is good in hidden layers (smooth gradient), but on outputs its floor compounds across heads. When porting a new model variant, re-check which output transforms the existing variants use — don't copy the pre-fix pattern. Consider a shared `_apply_non_negative` helper so a future reviewer can't silently diverge one variant.

### [FIXED] No feature clipping after StandardScaler
- **Files:** `src/shared/pipeline.py:311-313`, `app.py:320`
- **What:** Test features could produce z-scores up to 19.5 — far outside the training distribution. NN predictions were unpredictable for these inputs.
- **Fix:** Added `np.clip(..., -4, 4)` after all `StandardScaler.transform()` calls.
- **Lesson:** Always clip scaled features. StandardScaler assumes train/test distributions are similar, but outliers in test data can produce extreme z-scores. Clip at +/-4 (catches 0.3-0.4% of values, prevents catastrophic extrapolation).

### [FIXED] `kicker_week_split` does not exist — app.py crashed on import
- **File:** `app.py:62, 262`
- **What:** Imported `kicker_week_split` from `K.k_data`, but the function was renamed to `kicker_season_split`. App crashed immediately with `ImportError`.
- **Fix:** Changed import and call site to `kicker_season_split`.
- **Lesson:** When renaming functions, grep for all call sites across the project — not just the file where the function is defined.

### [FIXED] DST `pts_allowed_bonus` clamped to >= 0, but target ranges from -4 to +10
- **Files:** `src/shared/neural_net.py`, `src/dst/config.py`, `src/dst/run_pipeline.py`, `src/serving/app.py`
- **What:** The softplus-to-clamp fix applied `clamp(min=0)` globally to all heads. But DST's `pts_allowed_bonus` ranges from -4 (35+ points allowed) to +10 (shutout). The model couldn't predict negative tiers.
- **Fix:** Added `non_negative_targets` parameter to `MultiHeadNet.__init__` (defaults to all targets). DST config specifies `{"defensive_scoring", "td_points"}`, leaving `pts_allowed_bonus` unconstrained.
- **Lesson:** Output constraints must be per-head when targets have different valid ranges. A global clamp works for most positions but breaks any target that can legitimately be negative.

### [FIXED] Pipeline evaluation added adjustment to predictions but not to the total target
- **Files:** `src/shared/pipeline.py:305, 349` (and equivalent in `run_cv_pipeline`)
- **What:** After fixing the total aux loss target to `sum(targets)`, evaluation still added `adj_test.values` to Ridge and NN total predictions. This compared `sum(preds) + adj` against `sum(targets)` — the adjustment inflated the evaluation error.
- **Fix:** Removed `+ adj_test.values` from evaluation totals. Adjustment is only applied at inference in `src/serving/app.py`.
- **Lesson:** When changing a training target, trace all downstream consumers — evaluation metrics, ensemble computation, and plotting all need to stay consistent. This was a cascading side-effect of Fix 1 that we caught.

### [FIXED] `run_cv_pipeline` missing `non_negative_targets` on MultiHeadNet
- **File:** `src/shared/pipeline.py:804`
- **What:** `_train_nn` and `_train_attention_nn` both pass `non_negative_targets` to `MultiHeadNet`, but `run_cv_pipeline` constructed its own `MultiHeadNet` without it. DST's `pts_allowed_bonus` (range [-4, +10]) was incorrectly clamped to >= 0 during CV.
- **Fix:** Added `non_negative_targets=cfg.get("nn_non_negative_targets")` to the CV pipeline's `MultiHeadNet` call.
- **Lesson:** When the same model is constructed in multiple code paths, all paths must pass the same kwargs. `_train_nn` is the reference — any manual `MultiHeadNet(...)` call elsewhere must mirror it.

### [FIXED] Dead `adj_val`/`adj_test` variables after adjustment removal
- **File:** `src/shared/pipeline.py:778, 921-922`
- **What:** After removing `+ adj_val.values` from CV and holdout totals (Fix 6), the `adj_val` and `adj_test` variables were still computed but never used.
- **Fix:** Deleted the dead lines.

### [FIXED] Weather/Vegas features missing at inference in `src/serving/app.py`
- **File:** `app.py:310-311`
- **What:** Training pipeline (`_prepare_position_data`) merged schedule features, but `src/serving/app.py`'s inference path (`_apply_position_models`) did not. Models trained with 12 weather/Vegas features received zeros at serving time.
- **Fix:** Added `merge_schedule_features(_df)` calls in `_apply_position_models` before feature computation.
- **Lesson:** Any feature engineering done in the training pipeline must also be done in the inference/serving path. Diff the two code paths when adding new features.

### [FIXED] ReDoS risk in `/api/predictions` search and `int(week)` crash
- **File:** `app.py:496-498`
- **What:** User search input was passed directly to `str.contains()` as a regex pattern (ReDoS risk). Also, `int(week)` could crash with `ValueError` on invalid input.
- **Fix:** Added `regex=False` to `str.contains()` and wrapped `int(week)` in try/except with 400 response.

### [FIXED] Predictions are always PPR but API serves multiple scoring formats
- **File:** `app.py:505-561`
- **What:** `/api/predictions` accepts a `scoring` param (standard, half_ppr, ppr). It selects the correct *actual* column, but `ridge_pred` and `nn_pred` are always trained on PPR targets. When a user selects "standard" scoring, actuals change but predictions don't — the comparison is apples-to-oranges.
- **Fix:** Added `scoring_note` field to API response when scoring != ppr. UI already displays a "PPR Scoring" badge and has no scoring selector, so users aren't misled. Training separate models per format is out of scope.
- **Impact:** API consumers now get a clear warning.

### [FIXED] RB test fixture missing `receiving_epa` and `receiving_air_yards` columns
- **File:** `tests/rb/test_features.py:10-52`
- **What:** `_make_player_games()` fixture didn't include `receiving_epa` or `receiving_air_yards` columns added with features 10-11. The feature list `RB_FEATURE_COLS` also had stale names (`first_down_rate_L3` instead of split `rushing_first_down_rate_L3` / `receiving_first_down_rate_L3`). 11 tests failed with `KeyError: 'receiving_epa'`.
- **Fix:** Added missing columns to the fixture and updated `RB_FEATURE_COLS` to match the current 11 features in `rb_config.py`.
- **Lesson:** When adding new features to `*_features.py`, update the corresponding test fixture and expected feature list.

### [FIXED] DST prior-season feature alignment used `.values`
- **File:** `src/dst/features.py:74-86`
- **What:** Prior-season features were merged via `season+1` then assigned back using `.values` (strips index). If the merge reordered rows, assignments would be silently misaligned.
- **Fix:** Changed to index-preserving merge: `reset_index()` → merge → `set_index("index")` → loc-based assignment using the original index.

### [FIXED] Feature column filtering could silently drop features at inference
- **File:** `app.py:308-311`
- **What:** `feature_cols = [c for c in feature_cols if c in pos_train.columns]` filters to available columns. If a feature from training is missing, the model gets fewer features than expected → dimension mismatch → crash. The crash would happen, but the error message wouldn't identify which feature is missing.
- **Fix:** Added count comparison and warning log when columns are dropped.

### [FIXED] No API error handling
- **Files:** `app.py:85-90`
- **What:** All API routes lacked try/except. If `_get_data()` or model loading failed, the user saw a generic 500 with no useful message.
- **Fix:** Added Flask `@app.errorhandler(Exception)` that returns JSON `{"error": ...}` for `/api/` routes. Logs full traceback to console.

### [FIXED] Huber delta asymmetry across targets starved count heads
- **Files:** Position config files (`*_config.py`)
- **What:** Pre-rebalance loss weights were roughly equal across heads, so yards targets (δ ≈ 15–30) dominated count-head gradients (δ ≈ 0.25–0.5) by ~20–2500× per sample, collapsing the count heads toward their mean. The DST `pts_allowed_bonus` head also had a too-forgiving delta relative to its range, and the old QB `td_points` delta was too small relative to its point scale.
- **Fix:** (1) Rebalanced NN loss weights to ≈ `2.0 / huber_delta` per head across RB (`d229830`), QB (`4ac478f`), WR (`35e611b`), and TE (`a03f795`). (2) DST targets were migrated to 10 raw stats (`cc0c627`), retiring `pts_allowed_bonus` entirely; QB's `td_points` was likewise replaced by split `passing_tds`/`rushing_tds` heads with δ = 0.5 and matching w = 4.0.
- **Lesson:** Huber δ and loss weight are coupled — changing one without the other either starves or drowns a head. Encode the pairing in the config (`2.0/δ`) and re-derive the weight whenever δ moves. See CLAUDE.md "Loss weights are tuned inverse-to-Huber-delta".
