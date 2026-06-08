# [PRIORITY] Shared parallel A/B / ablation harness (device-autodetect)

**Status: BUILT.** [src/tuning/ab_harness.py](../src/tuning/ab_harness.py) (library + CLI + worker),
[src/tuning/ab_example.py](../src/tuning/ab_example.py) (copy-me template + smoke target),
[tests/tuning/test_ab_harness.py](../tests/tuning/test_ab_harness.py). Motivated by the
2026-06-08 role-inheritance A/B ([todo/rotowire_gap_remediation.md](rotowire_gap_remediation.md)).
The design below is preserved for reference; **what actually shipped** is summarized here.

## What shipped
- **Spec = a module** exposing `VARIANTS: list[Variant]` (+ optional `POSITIONS` / `SEEDS` /
  `metric_fn` / `BASELINE`). `Variant(name, cfg_mutator=None, frame_injector=None,
  expect_ridge_identical=None, label="")`. A spec runs itself via `ab_main(__spec__.name)` in
  `__main__`; programmatic callers use `run_ab(spec, …)`.
- **CLI:** `python -m src.tuning.<spec> [--positions …] [--seeds …] [--only V …] [-j N]
  [--sequential] [--device cpu|cuda|…] [--feature-cache] [--list]`.
- **Two execution modes, same cell results:** sequential **in-process** (low-core / CI / the
  unit-test path) and parallel **subprocess-per-cell** (the perf path; chdir isolation is
  process-global, so a cell must be its own process). The parallel path *composes*
  `parallel_train.physical_cores` + `core_pool.start_coordinator` — nothing new in `src/shared/`.
- **Jobs autodetect** (`resolve_jobs`, gate = `FF_DEVICE` + `detect_platform()`): CUDA → `min(cells,6)`
  (GPU-launch-bound, share the one GPU); CPU+`cpu_count≥12` → one cell per **physical** core (pool
  fair-shares the joblib/LGBM stages, BLAS pinned to 1); MPS / small boxes → sequential. `FF_AB_JOBS`
  / `-j` override; workers are `nice`-d (`FF_AB_NICE`, default 10 — the owner games on this box).
- **Artifact isolation:** every cell `chdir`s into a private tmp dir with `data/` symlinked in, so the
  hard-coded `{pos}/outputs` writes land there and the served artifacts are never touched. The
  feature cache is **`FF_FEATURE_CACHE_DISABLE`-d by default** (it keys on data, not code → a sibling
  variant's features could be silently reused → false `Δ=0`); `--feature-cache` opts back in.
- **Aggregation:** mean±std per (position, variant, model, metric) + Δ-vs-baseline (order-independent)
  + the **Ridge-invariance sentinel** keyed on `expect_ridge_identical` (`False` ⇒ a feature MUST move
  Ridge; `True` ⇒ an NN-only change must NOT; `None` ⇒ report-only). Default `metric_fn` =
  `cohort_analysis.per_model_metrics` on `result["test_df"]`.

### Caveat found while building — frame injection is QB/RB/WR/TE-only
**K and DST `run(seed, config)` build their own splits internally** (PBP-reconstructed kicker /
defense data) — they take no `train/val/test` args. So `frame_injector` variants only work for the
four skill positions; `cfg_mutator` variants work for all six. The harness detects the `run`
signature and raises a clear error if a `frame_injector` variant targets K/DST. (This is the kind of
subprocess-only break unit tests can't see — caught by the smoke, not CI.)

### Verified (smoke, on the 5080)
- **TE** baseline + `+season_recency` (frame injection) × 2 seeds, `-j4`: 4 cells ran concurrently
  (~90s wall vs ~330s sequential); Ridge moved `Δ=-0.0028` ⇒ sentinel `expect=differ` → "data changed";
  served `te/outputs` byte-identical before/after; clean ±std table.
- **K** baseline + `nn_dropout=0` (cfg-only, K's no-frames path) × 1 seed: Ridge `Δ=+0.00000` ⇒ sentinel
  `expect=identical` → "data-identical"; served `k/outputs` byte-identical.

### Follow-ups (not in this PR)
- **Reproduce the role-inheritance A/B** (verification item 1) is deferred — it needs the
  role-inheritance *feature* itself (Phase 1 of [rotowire_gap_remediation.md](rotowire_gap_remediation.md)),
  which isn't built yet. The harness is ready to run it (RB/WR, frame-injection shape).
- **Port the existing `ablate_*` scripts** (`ablate_rb_gate`, `ablate_injury_features`,
  `ablate_backbone_norm`, …) onto the harness so they parallelize + isolate too. Mechanical; left out
  here to keep the PR to the harness itself.

## Why (the anti-pattern this kills)
That A/B was a hand-rolled `/tmp` script looping `for s in 42 123 7; do … done`:
- **Sequential** — 6 fully-independent cells (baseline vs +variant × 3 seeds) ran one at a
  time, ~20–30 s each, on `FF_DEVICE=cpu` with the **5080 idle and ~14 of 16 cores idle**.
- **Clobbered served artifacts** — every `run()` writes to the hardcoded `{pos}/outputs/models`
  ([src/shared/pipeline.py](../src/shared/pipeline.py):~1285, no cfg override), so each A/B
  overwrote the production RB models → **3 manual regenerations**.
- **Bespoke** — config-clone + frame-injection + cohort metric + seed-aggregate were reinvented
  inline, with no leakage guard (the first cut shipped a ~60%-inflating season-mean leak).

This should be **one reusable, parallel, artifact-isolated** harness.

## What already exists — REUSE, don't reinvent
The parallel + ablation machinery is already in the tree:
- **Device-autodetect fan-out:** [`src/benchmarking/parallel_train.py`](../src/benchmarking/parallel_train.py)
  `orchestrate(positions, jobs, …)` + `_default_jobs(n)` (CUDA backend AND `cpu_count≥12` → parallel,
  else sequential) + `physical_cores()` (16 on the 9950X3D, SMT-safe). `benchmark.py` already
  auto-delegates to it.
- **Fair-share core leasing:** [`src/shared/core_pool.py`](../src/shared/core_pool.py)
  `start_coordinator()` / `lease_cores(stage)` (`fair_cap = ceil(total/active)`); per-worker BLAS
  caps (`OMP/MKL/OPENBLAS_NUM_THREADS=1`) so joblib/LGBM own the parallel axis. No-op when the pool
  env is unset (byte-identical standalone).
- **Device/platform:** `detect_platform()` ([src/shared/platform_detect.py](../src/shared/platform_detect.py)
  → `backend`/`sm`/`compute_capability`/`cpu_count`); `requested_device()`/`FF_DEVICE`;
  `_lgbm_n_jobs()`/`LGBM_N_JOBS`; `scripts/train-local-parallel.sh` + `scripts/wsl-env.sh`.
- **Ablation pattern to generalize:** [`src/tuning/ablate_rb_gate.py`](../src/tuning/ablate_rb_gate.py),
  `ablate_backbone_norm.py`, `ablate_injury_features.py` — variants = `{name: (label, cfg-mutator-fn)}`,
  `run(seed, config=cfg)`, metrics from `result["test_df"]` / `result["{model}_metrics"]`, plus the
  **Ridge-invariance sentinel** (baseline Ridge MAE identical across variants on a seed ⇒ data-identity).
- **Artifact isolation that already works:** [tests/_pipeline_e2e_utils.py](../tests/_pipeline_e2e_utils.py):271-304
  — chdir to a tmp dir + symlink `data/`, run, restore cwd → all `{pos}/outputs` writes land in the
  tmp dir. This is the fix for the clobber (the output path is hardcoded, so chdir-isolation is the lever).
- **Subgroup metrics:** `cohort_analysis.label_ascension_rows`/`per_model_metrics`/`bucket_model_table`;
  `evaluation.compute_metrics`; `src/analysis/rmse_gap_decomposition.py`.

## Design — `src/tuning/ab_harness.py` (read-only; `src/tuning/` → no retrain trigger)
- **Variant** = `(name, cfg_mutator: cfg→cfg, frame_injector: (train,val,test)→(train,val,test) | None)`.
  Baseline = identity. Feature A/B = inject columns + extend `get_feature_columns_fn` / `attn_static_features`.
  Loss A/B = mutate `head_losses` / `loss_weights`. Arch A/B = flag overrides.
- **Metric** = `fn(result) → dict` (reuse cohort / `compute_metrics` / `rmse_gap_decomposition`).
- **Cell grid** = positions × variants × seeds; each cell = one isolated `run(seed, config, frames)`.
- **Parallel execution:** compose `parallel_train`/`core_pool` to run cells concurrently (a cell is the
  same unit `parallel_train` fans positions over — an independent `run()`). Each cell runs in its own
  **chdir + symlinked-`data/` tmp dir → never clobbers `{pos}/outputs`.**
- **Aggregate:** mean±std across seeds per (position, variant, model, metric); Δ vs baseline; the
  **Ridge-invariance assert** per seed; paired significance optional.
- **Output:** a variant×model×metric table (mean±std, Δ vs baseline) + a one-line "what was
  capped/dropped" log (no silent truncation).

## Parallelization spec (gated on `detect_platform()`; env override `FF_AB_JOBS`)
- **5080 (CUDA, sm_120) — GPU-launch-bound:** the bottleneck is kernel launches, not CPU (measured
  #670: `-j6` optimal, core-pool wall-clock-neutral). Run ~`min(cells, 6)` concurrent `run()` processes
  **sharing the single GPU** (`FF_DEVICE=cuda`); the GPU multiplexes. Do **not** pin to 1.
- **9950X3D (CPU A/Bs, `FF_DEVICE=cpu`) — CPU-bound:** fan out across the **16 physical cores** (not 32
  SMT — LightGBM SMT penalty is 16–27×) via the core pool, per-worker BLAS=1 + LGBM fair-share;
  `jobs ≈ 16 / cores-per-cell`.
- **Autodetect, don't hardcode:** derive backend + worker count + thread caps from `detect_platform()`
  (`backend`, `cpu_count`, `sm`) — reuse/extend `_default_jobs`. `FF_AB_JOBS` overrides. **CPU-considerate**
  (owner games on this box): `nice`, capped threads, honor `--jobs`.
- The gate is `FF_DEVICE` + backend, **not** the box: a CPU A/B on the 5080 box uses the CPU pool; a
  CUDA A/B uses the GPU-launch-bound pool.

## Stop-rules / gotchas
- **Artifact isolation is mandatory** (chdir + symlink `data/`) — an A/B must never clobber served
  `{pos}/outputs` (this session's footgun).
- **Frame-injectors must be pre-kickoff / leakage-safe** — the harness can't enforce it, and the
  Ridge-invariance sentinel won't catch it (a leak lives in the feature, not the split). The
  role-inheritance season-mean leak (~60% inflation) is the cautionary tale; document loudly.
- **Keep it in `src/tuning/`** (not `src/batch/` → 6-pos retrain; not `src/shared/` → retrain). The
  harness is read-only tooling that *composes* the benchmarking primitives — don't lift its parallel
  core into `src/shared/`.

## Verification
- Re-run the role-inheritance A/B through the harness → reproduces the manual leakage-clean numbers
  (RB lgbm Δ−0.048 MAE / +0.52 cohort).
- Measure parallel speedup vs sequential on both machines (GPU-launch-bound 5080 wave; 16-core CPU pool).
- Smoke: 2 variants × 2 seeds × 1 position completes with **served `{pos}/outputs` untouched** + a clean
  ±std table.
