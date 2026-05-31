# shellcheck shell=bash
# Local CPU-parallelism profile for WSL2 / Linux dev boxes with many cores
# (tuned for a 16-core / 32-thread Ryzen 9 9950X3D + RTX 5080, but safe on any
# multi-core Linux host). Source it before a local run/tune:
#
#   source scripts/wsl-env.sh
#   python -m src.qb.run_pipeline            # single position, full pipeline
#   python -m src.benchmarking.benchmark RB  # benchmark one position
#
# WHY (and why this is NOT the Windows OPENBLAS guard):
#   On native Windows, OPENBLAS_NUM_THREADS=1 is REQUIRED to avoid a
#   0xC0000005 access-violation crash in the Ridge-PCA alpha CV (concurrent
#   threaded LAPACK eigh). WSL2 is Linux, so that crash does NOT apply here.
#   We still cap BLAS to 1 thread for a different, throughput reason: the
#   Ridge/ElasticNet alpha CV fans alphas over joblib.Parallel(n_jobs=-1,
#   prefer="threads") (src/shared/pipeline.py). If each of those joblib
#   threads also spawns a full BLAS thread-pool, you get
#   joblib_threads * BLAS_threads oversubscription on the same cores --
#   the exact pathology the AWS Batch job-definition caps fix (ADR D13 /
#   TODO.md) addresses on the 4-vCPU T4 host. Capping BLAS to 1 lets the
#   OUTER joblib axis own the 16-core fan-out cleanly.
#
# Idempotent and override-friendly: each var is only set if you haven't
# already exported it, so `OPENBLAS_NUM_THREADS=2 source scripts/wsl-env.sh`
# (or any prior export) wins.

# --- BLAS: 1 thread/process so joblib owns the alpha-CV fan-out ---------------
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

# --- LightGBM: use the physical cores for a single-position run ---------------
# Default LGBM_N_JOBS is 1 (a macOS nested-OpenMP segfault guard, see
# src/shared/models.py). On a 16-core Linux box that leaves 15 cores idle
# during the LightGBM stage of a bare `run_pipeline`. 16 = physical cores,
# NOT the 32 logical -- LightGBM regresses under SMT/hyperthreading.
#
# NOTE: for `python -m src.tuning.tune_lgbm` do the OPPOSITE -- leave
# LGBM_N_JOBS=1 and let --n-jobs (defaults to min(cpu,16)=16) parallelise the
# Optuna TRIALS instead, so you don't get 16 trials * 16 LGBM threads = 256.
# tune_lgbm's own _guard_lgbm_threads enforces this, but exporting 16 here
# would fight it; unset it before tuning:  unset LGBM_N_JOBS
: "${LGBM_N_JOBS:=16}"
export LGBM_N_JOBS

echo "[wsl-env] BLAS threads=1 (OMP/MKL/OPENBLAS/NUMEXPR), LGBM_N_JOBS=${LGBM_N_JOBS}" >&2
echo "[wsl-env] for tuning: 'unset LGBM_N_JOBS' then 'python -m src.tuning.tune_lgbm <POS> --n-jobs 16'" >&2
