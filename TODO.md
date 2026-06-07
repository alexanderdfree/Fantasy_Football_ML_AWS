# Bugs & Potential Issues

Tracking known issues and uncertainties in the project. Resolved issues are split into **[todo/fixed-archive.md](todo/fixed-archive.md)** — each archived entry includes the lesson learned, which has repeatedly been useful context for reviewers and future work. **Read the archive before proposing changes near anything it mentions** — most "obvious" fixes have been tried.

**Last reviewed: 2026-06-01.**

---

## Open

### [AUDIT] Attention NN architecture audit — verified-sound + live/dormant map
- **Doc:** [todo/attention_nn_architecture_audit.md](todo/attention_nn_architecture_audit.md) — read-only correctness/design audit of the production attention NN across all six positions (`src/shared/neural_net.py`, `src/shared/training.py`, six `config.py`).
- **What:** **No confirmed correctness bugs.** Architecture is learned-query attention pooling (not a transformer; `attn_self_layers=0` everywhere), numerically careful (`clamp(min=0)` non-negativity, all-padding `nan_to_num` guards, season-opener handling), and cross-position-consistent (`nn_non_negative_targets`, `attn_d_model=32/n_heads=2` everywhere). Headline finding is a **live-vs-dormant capability map**: ALiBi / learned-temperature / self-attention / query-conditioning / gated-fusion / entropy-reg / K-V-projection / opp-history are all implemented + test-covered but enabled by **no** `POSITION_CONFIG`.
- **Leads (not scheduled):** (1) adopt-or-delete the dormant extensions — already tracked as the PR #662 ablation-harness item below; (2) one low-risk numerical note: `GatedHead.value_log_alpha` is unbounded (`neural_net.py:268`) with no max clamp on `alpha` — optional `log_alpha`-clamp hardening, but it edits `src/shared/` → 6-position retrain (not `[docs-only]`).
- **Also records two subagent-flagged "bugs" as verified false positives** (empty-batch checkpoint; batcher seeding) so they don't get re-raised.
- **Status:** audit complete + committed (docs-only). Leads are candidates, not scheduled changes.

### [ANALYSIS] ATTN week-by-week & subgroup accuracy — patterns + leads
- **Findings doc:** [todo/attn_accuracy_findings.md](todo/attn_accuracy_findings.md); reusable diagnostic [src/analysis/attn_weekly_accuracy.py](src/analysis/attn_weekly_accuracy.py) (read-only, multi-seed). Rerun: `python -m src.analysis.attn_weekly_accuracy --report report.md --figdir figs`.
- **What (robust across seeds 42+123, 7,177 test rows/seed):** the four models are ~tied overall (LightGBM 4.142 ≈ ATTN 4.158 < NN 4.198 < Ridge 4.307); the trustworthy signal is **per-position direction** — ATTN is the **best model on DST** and the **worst-of-the-learned-models on K** (Ridge wins K cleanly), loses by a hair on QB/RB/TE, and is a seed-coin-flip on WR. The dominant error structure is **regression-to-the-mean shared by all models** (over-predict Q1 by +2.7–3.5, under-predict Q4 ceiling games by −7 to −8); ATTN is the best-calibrated *learned* model at both tails.
- **Leads (not yet acted on):** (1) K should not headline the attention NN — Ridge is robustly better; (2) the universal Q4 under-shoot dwarfs any model choice — a ceiling-aware/quantile loss head (tuned per the `LOSS_WEIGHTS ≈ 2.0/δ` coupling, not independently) would move accuracy more than swapping models; (3) DST is where attention genuinely pays off.
- **Status:** analysis complete + committed (docs-only). Leads above are candidates, not scheduled changes.

### [PRIORITY] Extend NN Optuna grid: add batch-size rung 1024 + widen LR ceiling
- **Plan doc:** [todo/increase_batch_size_plan_priority.md](todo/increase_batch_size_plan_priority.md) — full rationale (VRAM math, per-position steps/epoch + drop_last table, accepted consequences). Read it before starting.
- **File(s):** [src/tuning/tune_nn.py](src/tuning/tune_nn.py) `_sample_overrides` (lines ~179–186); [tests/tuning/test_tune_nn.py](tests/tuning/test_tune_nn.py) (assertions at lines 105, 107, 109, 110).
- **What:** Investigated "increase NN batch size + adjust LR accordingly." Key reframings: optimizer is **AdamW** (so √-scaling, not SGD linear scaling) and batch×LR are **already jointly Optuna-searched** — so "adjust LR" = widen the LR ceiling so the larger-batch optimum isn't clipped, then let TPE find the pairing. VRAM is slack (~69K-param model, peak ≤~250 MB at batch 1024); the binding constraint is steps/epoch + `drop_last` tail on the *smallest* position (QB ~7.7K rows), since the grid is position-agnostic.
- **Change (4 remaining steps; grid code unchanged so far):** (1) both batch grids `[128,256,512]` → `[128,256,512,1024]`; (2) both LR ranges `1e-4..5e-3` → `1e-4..1e-2` (log) + a comment block recording the rationale; (3) update the 4 test assertions (grid membership + the `<= 5e-3` → `<= 1e-2` bound — widening **up** does break the old bound); (4) when the grid lands, update the `[LOW] drop_last=True` entry below to note the ~7% QB tail now reachable @1024, and add an ARCHITECTURE.md `Update history` line. Note: PR #743 only updated tune_nn's scheduler-output plumbing so sampled attention scheduler values still route to the attention-specific config keys introduced by the production default change; it did **not** widen this grid.
- **Decisions (from user):** extend grid but **do not re-run the tuner**; add **one shared rung (1024)** only — *not* 2048, *not* position-aware; keep objective pure `min(val_loss)` (no speed tiebreak). 2048 / position-aware grid remain out of scope. Production attention defaults were hard-pinned separately from the completed batch/LR ablation, so this item is now only about keeping the Optuna grid able to rediscover larger batches in future tuner runs.
- **Verify:** `ruff check . && ruff format --check .` + `pytest tests/tuning/test_tune_nn.py -m unit`. No GPU / pipeline run needed (`.md`-only handoff today; the tuner run is a later user-triggered step).
- **Status:** Plan + this note committed (docs-only). Grid/range change **not yet made** — pick up at step 1.

### GPU launch-bound levers (CUDA graph built; single-process streams still planned)
- **Plan doc:** [todo/gpu_launch_bound_levers.md](todo/gpu_launch_bound_levers.md) — the diagnosis (`-j` sweep + phase data proving launch-boundedness), the MPS dead-end (Linux-only; unavailable on both WSL2 *and* native Windows), and two opt-in levers with friction points + A/B gates. Read it first.
- **What:** local 6-position parallel training is **GPU launch/host-bound** at full concurrency — the core pool (#670) cut the LightGBM stage 15–90× but total wall-clock stayed flat (~242s) because the shared-GPU attention-NN training (~200s/pos, launch-bound) dominates and `-j6` is already optimal (`-j6` 242s < `-j3` 244s < `-j2` 269s). **CPU allocation is not the lever** — don't re-optimize it for local wall-clock.
- **Levers:** (A) **CUDA graphs** — **autodetect-ON for sm_80+** as of 2026-06-05 (`FF_CUDA_GRAPH=0` is the force-off override); speed win is real, graph-vs-graph is clean, graph-vs-eager is not bit-comparable, so the sm_80+ metric path is intentionally non-byte-identical (ADR-0017). (B) **single-process + per-position CUDA streams** — the MPS substitute; bigger refactor that would supersede the subprocess model + the core pool.
- **AWS Batch tuning update:** Batch NN tuning now uses the Linux-only NVIDIA MPS path where it is actually available: each of the six existing g6.xlarge position jobs launches `--parallel-backend auto --n-jobs 3`; inside the container `detect_platform()` resolves native-Linux L4/g6 to MPS, while Macs and 5080 hosts keep the existing thread backend unless MPS is explicitly forced. Batch tune jobs set `FF_CUDA_GRAPH=1`, keep FP16 AMP default, and force `FF_COMPILE=0`. The study namespace is separated as `scheduler_v2_mps_graph` so graph-enabled tuning does not resume from eager studies. K's nested-history trainer explicitly no-ops CUDA graph capture so all six positions can share the same graph-on tune workflow.
- **Recommended:** CUDA graph benchmarkability is settled: graphs are the default on sm_80+, so use graphed-vs-graphed rebaselines (set `FF_CUDA_GRAPH=0` to recover an eager baseline). Consider Lever B only if the remaining local wall-clock bottleneck justifies the larger stream/orchestrator refactor.
- **Stop-rule:** `torch.compile` is measured-rejected (#641, +169% on 5080); a hand-rolled graph sidesteps the dynamic-shape recompile but still must clear the A/B.
- **Status:** Lever A built/measured/**shipped autodetect-ON for sm_80+ production training (2026-06-05, PR #874 follow-up)**; AWS Batch tune path wires graph-on + NVIDIA MPS as a tuner-only profile. Lever B remains planned only for local Windows/WSL where NVIDIA MPS is unavailable.

### [PRIORITY] Tune `attn_weight_decay` (+ optional `attn_patience`) in the Optuna NN search space
- **Plan:** [todo/tune-nn-attn-weight-decay-patience_priority.md](todo/tune-nn-attn-weight-decay-patience_priority.md) — full implementation plan (exact ranges, `_PARAM_TO_CONST` + test edits, verification). **Pick this up to execute** (line numbers there are approximate post-rebase — re-grep the symbols).
- **File(s):** `src/tuning/tune_nn.py` (`_sample_overrides` search space + `_PARAM_TO_CONST`); `tests/tuning/test_tune_nn.py` (`_EXPECTED_KEYS`, range assertions, config-line round-trip).
- **What:** The attention tuner samples 6 attn knobs + 6 base-NN knobs but omits `attn_weight_decay` — an asymmetry, since the base NN's `nn_weight_decay` *is* tuned (`tune_nn.py:185`) while the attn branch's weight decay stays pinned at the per-position default. Both cfg keys already flow to the attention trainer (`pipeline.py`: `weight_decay=cfg.get("attn_weight_decay", ...)`, `patience=cfg.get("attn_patience", ...)`), so the change is contained to the tuner + its tests — no `POSITION_CONFIG`/pipeline edits.
- **Primary (clean win):** add `attn_weight_decay` → `trial.suggest_float("attn_weight_decay", 1e-5, 1e-3, log=True)`, mirroring `nn_weight_decay`; brackets both current defaults (`5e-5`, `3e-4`).
- **Optional (deprioritized):** `attn_patience` is weakly identified by the `min(val_loss)` objective — patience only moves *where* the deterministic trajectory stops, and `min` over a longer prefix is always ≤ a shorter one, so the objective is monotone in patience and TPE drifts it to the range max. Add only with a tight 15–30 band, or skip. See the plan's caveat.
- **Scheduler note:** scheduler search is no longer part of this follow-up. `src/tuning/tune_nn.py` now samples `scheduler_type` plus matching conditional keys (`cosine_t0`/`cosine_t_mult`/`cosine_eta_min` or `onecycle_max_lr`/`onecycle_pct_start`) and validates stale/mismatched scheduler payloads before training or config output.
- **Scope / no-retrain:** offline `workflow_dispatch`-only tuner; no model/feature/target/loss change → no production-metric change and no six-position retrain. The tuner emits paste-ready `{POS}_ATTN_*` constants for an operator to hand-paste into `src/{pos}/config.py` later.
- **Why not now:** deferred to a separate session per request; this entry + the committed plan carry the full context. Related: the `[LOW] Optuna parallel trials` note below.

### [PRIORITY] Local parallel multi-position training/tuning on one GPU (RTX 5080)
- **Doc:** [`todo/LOCAL_MULTI_POSITION_GPU_priority.md`](todo/LOCAL_MULTI_POSITION_GPU_priority.md) — full feasibility findings, no-code recipe, and design sketch (Options B/C).
- **What:** Running QB/RB/WR/TE/K/DST concurrently on one local 5080 is feasible — VRAM is a non-issue (~0.5–1 GB/process, ~3–6 GB for all six of 16 GB; the 5080 is already supported via `requirements-gpu.txt`'s `torch==2.11.0+cu128`). The real constraint is CPU oversubscription (Ridge/ElasticNet CV + LightGBM), capped via `OMP_NUM_THREADS`/`LGBM_N_JOBS`. Today the benchmark loop (`src/benchmarking/benchmark.py:399`) is sequential and tuning is one position at a time.
- **Next:** Build **Option B** — a parallel runner (`src/scripts/run_local_parallel.py` or a `--max-workers` flag on `benchmark.py`) using `ProcessPoolExecutor` + per-process thread caps; reuse `benchmark.py::run_one(pos)`. **Not** under `src/batch/` (triggers a 6-position retrain). See the doc for Option C (parallel tuning sweep).

### [PRIORITY] PCA-before-Ridge for QB/TE — validate then maybe ship (harness ready, NOT yet run)
- **Priority handoff.** Likely lands **flat** (see "Prior") — pick up to close the loop, not as a metric win. A complete, smoke-verified validation harness exists; a future session just runs it on fresh splits and reads the numbers.
- **Context:** the feature-collinearity audit ([#594](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/594), [src/analysis/analysis_feature_audit.py](src/analysis/analysis_feature_audit.py) + [findings](src/analysis/feature_audit_findings.md)) measured that **QB and TE feed a ~1e15-condition feature matrix straight into Ridge** (`ridge_pca_components=None`), while WR/RB/DST run PCA-before-Ridge (30/80/20). The hypothetical-PCA probe said ~53–56 components @ 99% variance would bring QB/TE to a well-conditioned ~17. Open question: does PCA-before-Ridge actually **lower QB/TE test MAE**, or does tuned Ridge L2 already absorb the conditioning so unsupervised PCA only adds bias (truncating low-variance PCs can drop real signal)?
- **Harness (DONE, smoke-verified): [src/tuning/ablate_ridge_pca.py](src/tuning/ablate_ridge_pca.py)** + [tests/tuning/test_ablate_ridge_pca.py](tests/tuning/test_ablate_ridge_pca.py). Ridge-only A/B at several `pca_n`, reporting val(2024)+test(2025) total MAE vs the no-PCA baseline. **Exact** A/B: PCA feeds only Ridge (NN/attention/LightGBM consume raw scaled features) and Ridge is deterministic — no seed noise, no GPU.
- **To run (another session):**
  1. **Fresh splits required.** Worktree `data/splits` is usually a symlink to the parent's STALE shared splits (missing recently-added features → `KeyError`). Rebuild into a LOCAL dir — do NOT write through the symlink (corrupts the parent). Exact recipe in the harness module docstring (`test -L data/splits && rm data/splits; mkdir -p data/splits;` then the SETUP heredoc, reuses cached `data/raw`).
  2. **Thread caps mandatory** — without them it hangs (joblib×BLAS oversubscription; it ran ~1h once): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m src.tuning.ablate_ridge_pca QB TE`
  3. A `pca_n` counts only if it beats `None` on **both** seasons by more than benchmark noise (±~2%).
- **Two gotchas already paid for:** (1) **Must go through `run_pipeline`** — a standalone sweep calling `add_specific_features` directly (à la [src/wr/benchmark_ridge_variants.py](src/wr/benchmark_ridge_variants.py)) MISSES the weather/Vegas/contextual columns (`is_dome`, `implied_opp_total`, `wind_adjusted`, `is_divisional`, `temp_adjusted`) that `merge_schedule_features` adds inside `build_position_features` at pipeline time → `KeyError`. The harness uses a Ridge-only cfg override to avoid this train/inference drift. (2) **No Ridge-only production retrain** — editing `src/{pos}/config.py` scopes a **full GPU retrain** of that position via [src/scripts/scope_positions.py](src/scripts/scope_positions.py); no per-model granularity, `[docs-only]` forbidden (PCA is behavioral). Shipping burns a GPU retrain whose NN output is identical.
- **Ship criterion (all must hold):** a single `pca_n` improves **both** val and test beyond noise AND **Ridge is the served (best) model** for that position in the latest `benchmark_history` (else the served prediction never sees the gain). If criteria fail, record a `[TESTED, REJECTED]` archive entry with the val+test table (mirrors the draft-capital precedent) and leave configs unchanged.
- **Prior (why low priority):** #594 concluded collinearity "isn't hurting predictions" because tuned Ridge L2 + the robust NN/tree models already handle it; even where PCR was tuned the gain was small (WR −0.094, RB ~−0.002 MAE) and Ridge isn't the served best-model everywhere. Expect flat. Validate to confirm, don't assume a win.

### [PRIORITY] Run the attention-architecture ablation harness (built in PR #662, NOT yet run)
- **Doc (start here):** [todo/run_attn_arch_harness_priority.md](todo/run_attn_arch_harness_priority.md) — run commands, how to read the decision table, adopt/delete criteria.
- **What:** PR #662 added [src/tuning/ablate_attn_arch.py](src/tuning/ablate_attn_arch.py), a config-toggle ablation for the **seven default-OFF attention extensions** (#109/#112/#115/#116/#117/#120/#121 — all on `main`, enabled by no position, **never benchmarked**). It mirrors `ablate_backbone_norm.py` (multi-seed, `--position`, Ridge data-identity sentinel, mean±std). Smoke-tested but deliberately not run yet. This entry is the run + adopt/delete follow-up.
- **How:** screen the Tier 1–2 flags (`alibi`/`alibi_only`/`seqdrop`/`temp`/`swiglu`) on RB at ≥8 seeds, promote survivors to QB (small-sample stress test); judge on paired Δ vs baseline + the Ridge sentinel (single-seed is directional only). Tier-4 `selfattn` is expected to regress (the "larger regressed on 15K-sample positions" stop-rule) — confirm once, then delete its dead code.
- **Decision → action:** a flag that beats baseline beyond noise (RB + QB) → enable in that `POSITION_CONFIG` + add an ADR-0004 changelog line (**fires a GPU retrain — not `[docs-only]`**); flat/regression → leave OFF or delete the dead branch + tests (numerically inert → Ridge-identity → training-skipped).
- **Scope:** running the harness is offline `src/tuning/` (no retrain); only *adopting* a winner edits a position config.
- **Status:** harness merged (PR #662); not yet run. Pick up from the doc.

### [PRIORITY] Shared ablation runner with intelligent multithreading (foundation built)
- **Doc (design):** [todo/shared_ablation_multithreading_priority.md](todo/shared_ablation_multithreading_priority.md) — full design, contention model, verification.
- **What:** [src/tuning/ablation_runner.py](src/tuning/ablation_runner.py) now provides shared `AblationJob` / `AblationResult`, serial-by-default `run_grid`, process-worker support, thread caps, dry-run formatting, paired-delta helpers, and history writes. The first consumer is [src/tuning/ablate_batch_lr.py](src/tuning/ablate_batch_lr.py). Existing `ablate_*` scripts still have their hand-rolled loops.
- **"Intelligent" =** cap per-worker BLAS/joblib threads (`OPENBLAS/OMP/MKL/VECLIB=1` — the PCA harness *hangs* without them) and bound workers to physical cores on CPU; on CUDA, serialize / VRAM-limit the GPU-resident phase (reuse `detect_platform()`); **warm the seed/flag-independent feature cache once before fan-out** to avoid a concurrent-build race. `ProcessPoolExecutor`, reuse the position `run()`.
- **Scope / no-retrain:** `src/tuning/` only — **NOT** `src/batch/` or `src/shared/` (both fire a 6-position retrain). Determinism preserved: each `(variant, seed)` re-seeds internally → order-independent, so a parallel run must be byte-identical to serial (assert it). The **complement** of the `[PRIORITY] Local parallel multi-position training/tuning on one GPU` entry above (positions vs variants/seeds) — factor the contention primitives into one shared util both import.
- **Status:** foundation implemented for new ablations. Follow-up remains to migrate older `src/tuning/ablate_*.py` scripts and add any GPU-aware worker cap once a non-timing parallel CUDA sweep needs it.

### [OPEN] QB min-games relaxation — served Ridge benefits but attention regresses established QBs (excluded from the RB/WR/TE ship)
- **Context:** the train-only min-games relaxation shipped for RB/WR/TE ([todo/fixed-archive.md](todo/fixed-archive.md): *"[FIXED] Train-only `MIN_GAMES_PER_SEASON` filter…"*). QB was **excluded** — its full-model picture is mixed, unlike the clean skill-position wins. Harness: [src/tuning/ablate_min_games.py](src/tuning/ablate_min_games.py).
- **Evidence (QB {1,3,6} × 8 full-model seeds):** relaxing 6→1 helps the cold-start subgroup on *every* model (`would_filter` Δ: ridge +1.016, attn +1.220±0.472, lgbm +1.502±0.386 — low-vol MAE ~7–8 → ~6), **but the attention NN regresses established QBs** (`kept` Δ attn **−0.404±0.158**, lgbm −0.101±0.030; ridge/nn ≈0). QB's *served* model is **Ridge** (best ALL MAE 6.54), which relaxes cleanly (ALL 6.54→6.35, `kept` +0.010) — so QB-via-Ridge passes the ship criterion, but the served best-model is close enough (attn 6.61 vs ridge 6.54 at thr6) that a retrain could flip it to attention and make the −0.404 regression user-facing. **Threshold 3 is unusable** (attn ALL 6.95±0.75 / `kept` 7.18±1.01 — worse than both 1 and 6).
- **Why deferred:** the single-seed deterministic view said "QB ALL +0.264, ship it"; the 8-seed full-model run surfaced the attention `kept` regression — a clean instance of validate-on-the-served-model-not-a-proxy. Forcing QB into the retrain risks regressing established-QB predictions on a best-model flip.
- **To pick up:** decide per-*target* (not per-position) — relax QB only where the served model is regression-free (Ridge/nn), or pair it with a fix for the attention `kept` regression (the rookie early-game calibration entry above targets the same early-career QB rows). Judge `would_filter` vs `kept` per served model, ≥8 seeds. The `min_games_per_season` `PositionConfig` knob already exists — set QB's once clean.

### [ACKNOWLEDGED] K features use cross-season rolling windows
- **File:** `src/k/features.py:27`
- **What:** Kicker features group by `["player_id"]` only (no season reset), so rolling windows span across seasons. A kicker's late-season 2024 stats influence early-2025 predictions.
- **Rationale:** Kickers have stable multi-year careers and small per-season sample sizes (~17 games), so cross-season windows provide more signal. Comment above the `grp` assignment records this rationale.
- **Risk:** If a kicker changes teams or role between seasons, stale cross-season features could mislead the model. Likely small impact.

### [LOW] `_cache` dict grows without eviction
- **File:** `src/serving/app.py:88` (`_cache = {}`)
- **What:** `_get_data()` caches results in a module-level dict (serialized by `_cache_lock` since #31). The cache is never cleared. Not a real problem in practice (server restarts frequently), but worth noting.

### [LOW] `drop_last=True` silently discards training samples
- **File:** the train-DataLoader constructions in `src/shared/training.py` (grep `drop_last=True`).
- **What:** Last incomplete batch is dropped in every training DataLoader (attention, multi-target, history-multi-target, plus the CV and per-kick variants). With batch_size=512 (WR) and 32,521 training rows, 121 rows (~0.4%) are never seen. Standard practice, but combined with early stopping means those rows never contribute.

### [LOW] K targets overwrite `fantasy_points` column
- **File:** `src/k/targets.py:31`
- **What:** `df["fantasy_points"] = df["fg_yard_points"] + df["pat_points"] - df["fg_misses"] - df["xp_misses"]` overwrites the original column. Safe if called once, but calling twice would use the already-computed value. Not a current bug, just fragile.

### [LOW] Redundant NaN handling in feature engineering
- **Files:** All `*_features.py` files
- **What:** Pattern like `(a / b).fillna(0)` followed by `df.loc[b == 0, col] = 0` is redundant — fillna already handled the division-by-zero case. Not wrong, just noisy.

### [UNCERTAIN] K/DST index collision in `_get_data()`
- **File:** `src/serving/app.py:1223`
- **What:** K/DST test rows are appended to `results` with `offset = results.index.max() + 1`. Assumes the general test data has a well-behaved index. If the general test parquet has gaps, K/DST indices could collide. Probably safe in practice since parquet preserves sequential indices.

### [UNCERTAIN] Team share features computed per-split
- **Files:** `src/rb/features.py:96`, `src/wr/features.py:60`, `src/te/features.py:59`
- **What:** Team carry/target shares are computed within each split independently (`compute_team_*_totals` runs on each split's own data). A player's share could differ between train and test if their teammates are distributed differently across splits. By design (prevents leakage), but the share values won't be globally consistent.

### [LOW] Drop `RidgeMultiTarget.predict_total` / `ElasticNetMultiTarget.predict_total`
- **Files:** `src/shared/models.py:350-352, 464-466`; consumers in `tests/{qb,rb,wr,te,k,dst}/test_models.py`, `tests/shared/test_elasticnet.py`.
- **What:** `predict_total` returns an unweighted raw-stat sum (yards + TDs + receptions, all summed as if commensurate). No production callers in `src/`; only tests reference it. A future caller using it for ranking would regress the ~1.9 pt/game double-count fix in the Fixed archive.
- **Why not now:** Deletion touches shared model helpers and per-position model tests, tripping the benchmark-freshness gate; this Tier A pass intentionally avoided pipeline/benchmark evidence.

### [LOW] Defer per-position `CONFIG = build_pipeline_config(...)` to function level
- **Files:** All `src/{pos}/run_pipeline.py`.
- **What:** Each position builds its `CONFIG` at module import, which eagerly imports `data.py` / `features.py` / `targets.py` before `run_pipeline()` is called. Import-time failures surface in confusing places (the importing test, not the position's own module). Move to a function-local build or `functools.cached_property`.
- **Why not now:** Touches 6 `run_pipeline.py` files; the W.SHARED-A worker (PR #314) couldn't touch per-position files under its boundary. Carry-over from L-S17.

### [LOW] 63 fully-empty columns carried in every unified split (dead weight)
- **Files:** `src/data/split.py` (`temporal_split`), `src/features/engineer.py` (`build_features`).
- **What:** A comprehensive data profile (`src/analysis/analysis_training_data_profile.py`) found 63 columns are 100% null in every split: all K/DST raw stats (`fg_*`, `pat_*`, `gwfg_*`, `def_*`), return stats (`punt_returns`, `kickoff_returns`, `*_return_yards`), `game_id`, `passing_cpoe`, `penalties`, `penalty_yards`, and `fumble_recovery_*`. These are present because `load_raw_data` merges them from nflverse (they are populated for other table shapes) but they carry no signal for the QB/RB/WR/TE models that read the unified splits. Train split alone wastes ~15–20 MB and these columns muddy any automated feature scan (`analysis_feature_audit.py`, `analysis_data_completeness.py`).
- **Fix:** either drop them in `temporal_split` (pass a column allowlist) or have `build_features` filter to the union of all six positions' `get_feature_columns()` plus identifier/target columns. Whichever path, update the `Verify engineered columns` step in `refresh-splits.yml` to assert absence.
- **Stop-rule:** the empty columns are genuinely empty (not partial: K/DST self-load their own frames); dropping them should produce a Δ=0 Ridge MAE on a fresh benchmark run (use as the inertness tell).

### [LOW] data/splits schema non-uniform: test(2025) carries 56 raw columns absent in train/val
- **Files:** `src/data/split.py`, `src/data/loader.py`.
- **What:** The same data profile found 56 columns (e.g. `game_id`, all `def_*`, `misc_yards`, `fumble_recovery_*`) that are 100% null in train (2012–2023) and val (2024) but fully populated in test (2025). The 2025 split was built from a newer nflverse schema that includes these fields. Currently benign — none are whitelisted model features — but a future feature reading any of these would silently train on all-null and evaluate on real data (a model would see the column as all-zero at train time and then real values at test time, a hidden covariate shift).
- **Fix:** standardise the column set across all three splits at build time — either zero-fill the train/val rows for the new columns, or exclude the new columns from the split parquets entirely. Pair with the `[LOW] 63 fully-empty columns` fix above so both cleanups happen in one split-builder edit.

---

## Archive (Fixed)

Moved to **[todo/fixed-archive.md](todo/fixed-archive.md)** to keep this file lean (~49K tokens of frozen history, ~87% of the old TODO.md). Append new resolved entries there using the `### [FIXED] Title` + **File(s) / What / Fix / Lesson** format. Frozen-history split mirrors the ADR's [docs/architecture-history.md](docs/architecture-history.md).
