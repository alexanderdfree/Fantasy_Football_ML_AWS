# Running multiple positions concurrently on a local GPU (RTX 5080)

> Handoff / feasibility note. Tracked from `TODO.md` (Open). A later session can
> implement the tooling in "Optional tooling" without re-deriving any of this.

## Context

The training pipeline was designed around AWS (six g4dn.xlarge **T4** Spot
instances, one position per host — `docs/batch_design.md`). Locally, the
benchmark loop runs positions **sequentially** and tuning runs **one position at
a time**. With a local RTX 5080 (16 GB, Blackwell `sm_120`) now in the picture,
the question is whether QB/RB/WR/TE/K/DST can train or tune **concurrently on the
one local GPU** to speed up local iteration.

**Answer: yes, comfortably.** The GPU is the *last* resource you run out of.

## TL;DR

- The 5080 is already supported: `requirements-gpu.txt` pins `torch==2.11.0+cu128`
  (the `cu128` wheel is what carries `sm_120` kernels; the AWS `cu126` path tops
  out at `sm_90`). `SETUP.md` documents the install + expected `torch.cuda` output.
- VRAM is a non-issue: each position's attention NN is ~3–8K params; a concurrent
  process costs **~0.5–1 GB** (dominated by its CUDA context, not the model). Six
  at once ≈ **3–6 GB**, well inside 16 GB.
- **The real bottleneck is CPU, not the GPU** — Ridge/ElasticNet CV (joblib) and
  LightGBM (`src/shared/models.py`, CPU-only, multi-threaded) run per position.
  Six concurrently will oversubscribe cores unless you cap per-process threads.
- Nothing runs positions concurrently locally **today**; doing so is just
  "launch N processes." A convenience runner is net-new but small.

## Findings

### Hardware / software support
- Device selection: `src/shared/pipeline.py:124` `_nn_device()` returns a plain
  `torch.device("cuda")` — **no `cuda:0` pinning**. With a single GPU every
  process lands on the 5080 automatically; no device-targeting code needed.
- Mixed precision: `src/shared/training.py:597` uses **FP16** autocast + GradScaler,
  device-gated (`device.type == "cuda"`). FP16 is native/full-throughput on
  Blackwell. (BF16 was tried and reverted because the production T4 lacks it.)

### VRAM footprint (per position)
- Attention NN: `d_model=32`, `n_heads=2`, 17-game history, 4–10 target heads →
  **~3–8K params** (`src/shared/neural_net.py`). Model + optimizer + activations +
  GPU-resident dataset = tens of MB.
- Per **process** the dominant cost is the CUDA context (~0.5 GB of cuBLAS/cuDNN).
  So **~0.5–1 GB per concurrent process**, **~3–6 GB for all six**.

### The real bottleneck is CPU
- Each position also trains classical models on CPU. Six concurrent jobs ×
  multi-threaded LightGBM/joblib will thrash cores. **Cap threads per process**
  with `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` and
  `LGBM_N_JOBS` (the pipeline already reads `LGBM_N_JOBS` via `_lgbm_n_jobs()`).

### What exists today
- `src/benchmarking/benchmark.py:399` — `for pos in positions:` (sequential).
- `src/tuning/tune_nn.py` — one position per invocation, but **already** runs
  `--n-jobs` Optuna trials concurrently on the GPU (default 2; SETUP.md notes
  up to 3 fits on the 5080). So within-position GPU concurrency is a solved thing.
- AWS Batch gives each position its **own** instance/GPU — there is no existing
  notion of >1 position on a single GPU to reuse.

## How to do it today (no code changes)

Train/benchmark all six in parallel, capping CPU threads so the classical-ML
stages don't thrash (tune the numbers to your core count — e.g. 6 jobs × 2
threads = 12):

```zsh
for pos in qb rb wr te k dst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LGBM_N_JOBS=2 \
    python -m src.$pos.run_pipeline &
done
wait
```

Parallel tuning across positions (each already trial-parallel via `--n-jobs`, so
keep `--n-jobs` low when stacking positions — total GPU trials = positions ×
n_jobs; ~6 GB at 12 trials is still fine, but cap LightGBM threads):

```zsh
for pos in QB RB WR TE K DST; do
  LGBM_N_JOBS=2 python -m src.tuning.tune_nn $pos --n-jobs 1 &
done
wait
```

## Optional tooling (handoff — pick one to build next)

### Option B — parallel train/benchmark runner
- A small launcher (`src/scripts/run_local_parallel.py`) **or** a
  `--parallel/--max-workers` flag on `src/benchmarking/benchmark.py`.
- Use `ProcessPoolExecutor` / `subprocess.Popen` + a semaphore (separate
  processes, not threads — each position needs its own CUDA context; process
  isolation also sidesteps torch global state).
- Compute per-process thread caps from `os.cpu_count() // max_workers` and set
  `OMP/MKL/LGBM` env in each child. Reuse `benchmark.py::run_one(pos)` / the
  per-position `run_pipeline`. Keep `--max-workers 1` as the sequential fallback.
- **Do not** put this under `src/batch/` (any non-`tune`/`ablate` file there
  triggers a full 6-position retrain — CLAUDE.md). `src/scripts/` or
  `src/benchmarking/` is the correct home.

### Option C — parallel tuning sweep
- A wrapper that launches `tune_nn` (and/or `tune_lgbm`) for several positions
  under one **global** concurrency cap (semaphore over positions × n_jobs), so an
  overnight sweep covers all six instead of one. Same CPU-thread-capping concern.

## File references
- `src/shared/pipeline.py:124` — `_nn_device()` (device selection, no pinning)
- `src/shared/training.py:597` — FP16 AMP, device-gated
- `src/shared/neural_net.py` — attention NN (`d_model=32`, `n_heads=2`)
- `src/shared/models.py` — `LightGBMMultiTarget` (CPU-only), `_lgbm_n_jobs()`
- `src/benchmarking/benchmark.py:399` — sequential position loop
- `src/tuning/tune_nn.py` — Optuna, `--n-jobs` trial parallelism
- `requirements-gpu.txt` / `SETUP.md` — `torch==2.11.0+cu128`, RTX 5080 install
- `docs/batch_design.md` — AWS one-GPU-per-position design (contrast)

## Status
- **Implemented.** Option B shipped as `src/benchmarking/parallel_train.py` (subprocess
  fan-out) + the work-conserving `src/shared/core_pool.py` (the CPU-thread-capping policy is
  a lease-`ceil(cores/active)`-per-stage core pool, not a fixed split). As of **2026-06-07**
  the `--parallel/--max-workers`-on-`benchmark.py` variant this note also floated is wired:
  `python -m src.benchmarking.benchmark` **autodetects** the parallel fan-out on a many-core
  CUDA box (sequential elsewhere; `-j N` / `--sequential` override), lazily delegating to
  `parallel_train.orchestrate`. Option C (parallel tuning sweep) remains for a later session.
- The original plan-mode scratch file lives outside the repo at
  `~/.claude/plans/can-multiple-positions-train-tune-cuddly-backus.md`
  (`.claude/plans/` is gitignored); this doc is the committed copy.
