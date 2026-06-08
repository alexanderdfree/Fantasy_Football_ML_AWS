# [PRIORITY] Shared ablation runner with intelligent multithreading (built + all scripts migrated)

**Status update:** the shared runner exists at
[`src/tuning/ablation_runner.py`](../src/tuning/ablation_runner.py), and **all six `ablate_*`
scripts are now migrated onto it** (`ablate_batch_lr` first, then `rb_gate` / `ridge_pca` /
`min_games` / `injury_features` / `backbone_norm` / `attn_arch` in #1054). Only the priority
ablation *runs* remain (tracked in #1055).

## Built
- `AblationJob` / `AblationResult` dataclasses for one position x seed x variant run.
- `run_grid(jobs, max_workers=1, preserve_order=True)`, serial by default for timing-sensitive
  ablations, with `ProcessPoolExecutor` support for non-timing sweeps.
- Per-worker BLAS/joblib thread caps, dry-run table formatting, seed/variant parsing,
  paired-delta mean/std helpers, and `benchmark_history/ablations/` writing.
- Unit tests in [`tests/tuning/test_ablation_runner.py`](../tests/tuning/test_ablation_runner.py).
- **Autodetect unified with the A/B harness** (the parallel-runner sibling, see below):
  `resolve_max_workers("auto")` delegates to [`ab_harness.resolve_jobs`](../src/tuning/ab_harness.py)
  — one source of truth for "how many jobs on this box" (CUDA→6 sharing the one GPU; **9950X3D→16
  physical cores**, was serial before; MPS/small→1; `FF_AB_JOBS` override). Timing ablations pass an
  explicit `--max-workers 1`.
- **Output isolation** (`_isolated_outputs`): each job runs `chdir`-ed into a tmp dir with `data/`
  **and `.cache/`** symlinked in, so the hardcoded `{pos}/outputs` writes never clobber served
  artifacts while the warm-once/primed feature cache still pays off (the A/B harness disables the
  cache instead; ablations keep it — only the model-artifact writes are redirected).

## Still To Do
- **Run the priority ablations** — QB batch-size A/B (≥8 seeds), attention-architecture flag pass,
  PCA-before-Ridge for QB/TE. Tracked in **#1055** (the harnesses are migrated, so the runs are cheap).
- Longer-term convergence (open): fold ablations into [`ab_harness.py`](../src/tuning/ab_harness.py)
  as cfg-mutator variants and retire the `ablation_runner.py` / `ab_harness.py` overlap — both now
  share the autodetect + isolation.
- Add an explicit feature-cache warmup hook only when migrating a script that can safely prime
  features without changing the measured quantity. The batch/LR ablation keeps timing clean by
  defaulting to serial execution.

## Done
- **All six `ablate_*` scripts migrated onto `run_grid`** (#1054): `rb_gate`/`ridge_pca`/`min_games`/
  `injury_features`/`backbone_norm`/`attn_arch` build `AblationJob` grids + run via the runner with
  the shared autodetect + output-isolation; each keeps its variant definitions + decision table local.
  `backbone_norm`'s LayerNorm monkeypatch is applied inside the run_fn (spawn-worker-safe).

## Original Design Constraints
**"Intelligent" = contention-aware, not a blind thread pool:**
- **CPU-bound stages** (Ridge/ElasticNet alpha-CV + LightGBM) are joblib/BLAS-heavy. The runner
  must cap per-worker math threads (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=
  VECLIB_MAXIMUM_THREADS=1`) and bound `max_workers` to physical cores ÷ per-run thread budget —
  otherwise it oversubscribes and *slows down* (the PCA harness hangs without these caps; it ran
  ~1 h once). `ProcessPoolExecutor`, not threads (GIL + native libs).
- **GPU-bound stage** (the NN / attention training): on CUDA, concurrent runs contend for one
  device. The runner must detect CUDA (reuse `detect_platform()` / `cuda_enabled()`) and either
  serialize the GPU-resident phase or cap concurrent CUDA processes by per-process VRAM (the
  per-process VRAM/thread math is already worked out in
  [LOCAL_MULTI_POSITION_GPU_priority.md](LOCAL_MULTI_POSITION_GPU_priority.md)). On CPU-only boxes
  (this Mac) it's pure-CPU → just the thread-cap path above.
- **Warm the feature cache once before fan-out.** The feature cache is seed- and flag-independent,
  so N workers launching simultaneously would each trigger (or race) an identical feature build.
  Do one warm-up `run()` (or an explicit cache prime) on the main process first, then fan out — the
  workers all hit the warm cache. Avoids the concurrent-build write race.

## Constraints
- Lives in `src/tuning/` — **NOT** `src/batch/` (fires a 6-position retrain) and **NOT**
  `src/shared/` (also a 6-position retrain via the path-based detect job). Pure offline tooling.
- Determinism preserved: each `(variant, seed)` re-seeds internally inside `run()`, so results are
  order-independent — parallelism doesn't change any number, only wall-clock. The Ridge
  data-identity sentinel each script already prints still validates that.
- Reuse the position `run(seed=…, config=cfg)` (don't reimplement models); reuse
  `benchmark_utils.append_to_history` for the JSON write.

## Relationship to the sibling parallel-GPU entry
This is the **complement** of `[PRIORITY] Local parallel multi-position training/tuning on one GPU`
([LOCAL_MULTI_POSITION_GPU_priority.md](LOCAL_MULTI_POSITION_GPU_priority.md)): that parallelizes
across **positions** (six independent pipelines at once); this parallelizes the **variant × seed**
grid *within a single ablation* on one position. Same contention primitives (thread caps, VRAM
budget, `ProcessPoolExecutor`) — factor them into one shared util both can import.

## Verify
- `ruff check . && ruff format --check .`
- `pytest tests/tuning/test_ablation_runner.py tests/tuning/test_ablate_batch_lr.py -m unit`
- `python -m src.tuning.ablate_batch_lr --dry-run --positions QB RB WR TE K DST`
