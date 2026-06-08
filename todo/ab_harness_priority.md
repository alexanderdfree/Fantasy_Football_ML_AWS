# [PRIORITY] Shared parallel A/B / ablation harness (device-autodetect)

**Status:** planned, not built. Motivated by the 2026-06-08 role-inheritance A/B
([todo/rotowire_gap_remediation.md](rotowire_gap_remediation.md)).

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
