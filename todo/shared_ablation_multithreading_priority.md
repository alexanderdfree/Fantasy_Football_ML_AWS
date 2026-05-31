# [PRIORITY] Shared ablation runner with intelligent multithreading (NOT implemented)

**Priority handoff — design note only, no code yet.** Every ablation script under `src/tuning/`
(`ablate_attn_arch.py`, `ablate_backbone_norm.py`, `ablate_rb_gate.py`, `ablate_ridge_pca.py`,
`ablate_min_games.py`, `ablate_injury_features.py`) re-implements the same **sequential**
`for seed: for variant: run()` loop. A proper ≥8-seed sweep is therefore `N_variants × N_seeds ×
~1.5 min` wall-clock, serial — e.g. the attention sweep (9 variants × 8 seeds) is ~1.8 h on RB.
There is no shared parallel executor. Build one.

## Proposed
A shared helper, e.g. `src/tuning/ablation_runner.py`, exposing something like
`run_grid(position, variants, seeds, *, max_workers=None) -> list[row]` that fans the
`(variant, seed)` grid out across workers and returns the same per-run rows the scripts already
print. Each existing `ablate_*` keeps its own variant definitions + decision-table printing and
just calls the runner instead of its hand-rolled loop.

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

## Verify (when implemented)
- A parallel run and a serial run on the same `(position, variants, seeds)` must produce
  byte-identical rows (re-seed determinism) — assert it in a smoke test.
- Wall-clock < serial by ~min(cores, N_runs)× on CPU-only, with the Ridge sentinel still OK.
- `ruff check . && ruff format --check .` + a `tests/tuning/test_ablation_runner.py` smoke test
  (mock `run` to avoid a real train) following the existing `tests/tuning/test_ablate_*` pattern.

**Status:** NOT implemented — design/handoff note. A future session builds the runner, then
migrates the `ablate_*` scripts onto it one at a time.
