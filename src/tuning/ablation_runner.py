"""Shared helpers for offline tuning ablations.

This module is intentionally small and lives under ``src/tuning`` so ablation
scripts can share seed/variant loops without moving experiment-only machinery
into ``src/shared``. ``max_workers=1`` is available when an ablation needs clean
per-job timing; local CUDA sweeps can use ``resolve_max_workers("auto", ...)`` to
fan out small NN jobs across the single GPU while keeping worker processes
isolated.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from typing import Any

from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso

HISTORY_DIR = os.path.join("benchmark_history", "ablations")
_THREAD_CAP_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class AblationJob:
    """One seed x variant x position ablation run."""

    position: str
    seed: int
    variant: str
    label: str
    run_fn: Callable[[AblationJob], dict[str, Any] | AblationResult]
    base_cfg: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AblationResult:
    """Normalised output from one ablation job."""

    position: str
    seed: int
    variant: str
    metrics: dict[str, Any]
    timings: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None = None


def parse_seed_list(raw: str) -> list[int]:
    """Parse a comma-separated seed list."""

    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def select_variants(
    raw: str | None,
    available: dict[str, Any],
    default: tuple[str, ...] | list[str],
) -> list[str]:
    """Parse and validate a comma-separated variant list."""

    if raw is None or not raw.strip():
        selected = list(default)
    elif raw.strip().lower() == "all":
        selected = list(available)
    else:
        selected = [part.strip() for part in raw.split(",") if part.strip()]

    unknown = [variant for variant in selected if variant not in available]
    if unknown:
        raise ValueError(f"unknown variant(s): {unknown}; choose from {sorted(available)}")
    if not selected:
        raise ValueError("at least one variant is required")
    return selected


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    """Mean/sample-std summary for decision tables."""

    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def fmt_mean_std(values: list[float]) -> str:
    """Format a ``mean±std`` summary string for decision tables."""

    summary = mean_std(values)
    if not summary["n"]:
        return "n/a"
    return f"{summary['mean']:.4f}±{summary['std']:.4f}"


def _metric_value(result: AblationResult, metric_key: str) -> float:
    cur: Any = result.metrics
    for part in metric_key.split("."):
        cur = cur[part]
    return float(cur)


def paired_deltas(
    results: list[AblationResult],
    *,
    variant: str,
    baseline_variant: str,
    metric_key: str,
    position: str | None = None,
) -> list[float]:
    """Per-seed ``variant - baseline`` deltas for a metric."""

    by_key = {
        (result.position, result.seed, result.variant): result
        for result in results
        if result.error is None and (position is None or result.position == position)
    }
    pairs = []
    for pos, seed, var in sorted(by_key):
        if var != variant:
            continue
        baseline = by_key.get((pos, seed, baseline_variant))
        current = by_key[(pos, seed, var)]
        if baseline is not None:
            pairs.append(_metric_value(current, metric_key) - _metric_value(baseline, metric_key))
    return pairs


def format_dry_run_table(jobs: list[AblationJob]) -> str:
    """Return a compact table describing planned jobs."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for job in jobs:
        run_kind = str(job.metadata.get("run_kind", "experiment"))
        key = (job.position, run_kind)
        rec = rows.setdefault(key, {"seeds": set(), "variants": set(), "jobs": 0})
        rec["seeds"].add(job.seed)
        rec["variants"].add(job.variant)
        rec["jobs"] += 1

    lines = [f"Planned ablation jobs: {len(jobs)}"]
    lines.append(f"{'position':<8} {'kind':<20} {'seeds':<18} {'variants':<42} {'jobs':>5}")
    lines.append("-" * 99)
    for (position, run_kind), rec in sorted(rows.items()):
        seed_text = ",".join(str(seed) for seed in sorted(rec["seeds"]))
        variant_text = ",".join(sorted(rec["variants"]))
        lines.append(
            f"{position:<8} {run_kind:<20} {seed_text:<18} {variant_text:<42} {rec['jobs']:>5}"
        )
    return "\n".join(lines)


def _cap_worker_threads(lgbm_n_jobs: int | None = None) -> None:
    for name in _THREAD_CAP_VARS:
        os.environ[name] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    # LightGBM has its own ``LGBM_N_JOBS`` (not covered by the BLAS/OMP caps). When fanning
    # out (``max_workers > 1``) each worker must get a bounded share or N workers × an
    # env-set ``LGBM_N_JOBS`` (e.g. 16 via wsl-env.sh) oversubscribes the box — the doc's
    # "physical cores ÷ per-run thread budget" rule. Serial runs leave it untouched so one
    # ablation still uses every core for LightGBM.
    if lgbm_n_jobs is not None:
        os.environ["LGBM_N_JOBS"] = str(max(1, lgbm_n_jobs))


def resolve_max_workers(raw: str | int, *, job_count: int) -> int:
    """Resolve a CLI worker count, sharing the A/B harness's device autodetect.

    ``auto`` delegates to :func:`src.tuning.ab_harness.resolve_jobs` so the ablation
    runner and the A/B harness agree on parallelism on every box: CUDA ⇒ up to 6 jobs
    sharing the one GPU (#670); the 9950X3D CPU box ⇒ one job per physical core (per-worker
    BLAS pinned to 1, never SMT); MPS / small hosts ⇒ serial. ``FF_AB_JOBS`` overrides. An
    explicit integer is honoured (clamped to ``job_count``); ``< 1`` raises.

    Note ``auto`` now **parallelises on CPU too** (it used to stay serial there). A
    timing-sensitive ablation that needs clean per-run wall-clock should pass an explicit
    ``--max-workers 1`` rather than relying on ``auto``.
    """
    from src.tuning.ab_harness import resolve_jobs

    job_count = max(1, int(job_count))
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return resolve_jobs(job_count, None)  # shared autodetect (reads FF_AB_JOBS)
    workers = int(raw)
    if workers < 1:
        raise ValueError("max_workers must be >= 1")
    return resolve_jobs(job_count, workers)


def _error_result(job: AblationJob, exc: BaseException, *, tb: str | None = None) -> AblationResult:
    metadata = dict(job.metadata)
    if tb:
        metadata["traceback"] = tb
    return AblationResult(
        position=job.position,
        seed=job.seed,
        variant=job.variant,
        metrics={},
        timings={},
        metadata=metadata,
        error=f"{type(exc).__name__}: {exc}",
    )


def _coerce_result(job: AblationJob, payload: dict[str, Any] | AblationResult) -> AblationResult:
    if isinstance(payload, AblationResult):
        return payload
    metadata = {**job.metadata, **dict(payload.get("metadata") or {})}
    return AblationResult(
        position=job.position,
        seed=job.seed,
        variant=job.variant,
        metrics=dict(payload.get("metrics") or {}),
        timings=dict(payload.get("timings") or {}),
        metadata=metadata,
        error=payload.get("error"),
    )


def _safe_log_name(job: AblationJob, idx: int) -> str:
    parts = [
        f"{idx:04d}",
        job.position.upper(),
        str(job.seed),
        job.variant,
        str(job.metadata.get("run_kind", "experiment")),
    ]
    return "_".join(part.replace(os.sep, "-") for part in parts) + ".log"


def _with_log_path(result: AblationResult, log_path: str | None) -> AblationResult:
    if not log_path:
        return result
    metadata = dict(result.metadata)
    metadata["log_path"] = log_path
    return replace(result, metadata=metadata)


def _append_error_to_log(log_path: str | None, tb: str) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as logf:
        print("\n=== Job error ===", file=logf)
        print(tb, file=logf)


@contextmanager
def _isolated_outputs(data_dir: str):
    """chdir into a private tmp dir with ``data/`` and ``.cache/`` symlinked to the
    originals, so the pipeline's hard-coded ``{pos}/outputs`` writes land in the tmp dir
    and never clobber the served artifacts.

    Unlike the A/B harness (which *disables* the feature cache for correctness), the
    ablation runner symlinks ``.cache/`` through to the shared/primed cache so the
    warm-once-before-fan-out optimisation (e.g. ``ablate_batch_lr._prime_feature_cache``)
    still pays off — only the model-artifact writes are redirected. Mirrors the isolation
    in ``src/tuning/ab_harness.run_cell``.
    """
    orig = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="ff-ablation-")
    try:
        os.chdir(tmp)
        os.symlink(data_dir, os.path.join(tmp, "data"))
        os.symlink(os.path.join(orig, ".cache"), os.path.join(tmp, ".cache"))
        yield
    finally:
        os.chdir(orig)
        shutil.rmtree(tmp, ignore_errors=True)


def _run_job(
    job: AblationJob,
    log_path: str | None = None,
    data_dir: str | None = None,
    lgbm_n_jobs: int | None = None,
) -> AblationResult:
    """Run one job, capturing failures. When ``data_dir`` points at a real ``data/`` dir,
    the job runs output-isolated (``log_path`` must be absolute so the log still lands in
    the orchestrator's log dir, which ``run_grid`` guarantees). ``lgbm_n_jobs`` bounds
    LightGBM's per-worker threads under fan-out (``run_grid`` sets it; ``None`` = serial)."""
    _cap_worker_threads(lgbm_n_jobs)

    def _body() -> AblationResult:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
                return _coerce_result(job, job.run_fn(job))
        return _coerce_result(job, job.run_fn(job))

    try:
        if data_dir and os.path.isdir(data_dir):
            with _isolated_outputs(data_dir):
                result = _body()
        else:
            result = _body()
        return _with_log_path(result, log_path)
    except Exception as exc:  # noqa: BLE001 - ablation runners should capture per-job failures
        tb = traceback.format_exc()
        _append_error_to_log(log_path, tb)
        return _with_log_path(_error_result(job, exc, tb=tb), log_path)


def run_grid(
    jobs: list[AblationJob],
    *,
    max_workers: int = 1,
    preserve_order: bool = True,
    log_dir: str | None = None,
    progress: bool = False,
) -> list[AblationResult]:
    """Run an ablation grid and return one result per job.

    ``max_workers=1`` is intentionally serial. Larger values use a
    ``ProcessPoolExecutor`` and are best reserved for non-timing diagnostics.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    # Absolute log paths survive each job's output-isolation chdir; data_dir is the real
    # splits dir symlinked into every job's tmp cwd (see _isolated_outputs).
    log_paths = [
        os.path.abspath(os.path.join(log_dir, _safe_log_name(job, idx))) if log_dir else None
        for idx, job in enumerate(jobs)
    ]
    data_dir = os.path.abspath("data")
    # Under fan-out, bound LightGBM's per-worker threads to physical_cores ÷ workers so N
    # workers don't oversubscribe (serial keeps the env default → all cores for one run).
    lgbm_n_jobs: int | None = None
    if max_workers > 1:
        from src.benchmarking.parallel_train import physical_cores

        lgbm_n_jobs = max(1, len(physical_cores()) // max_workers)
    if max_workers == 1:
        results = []
        for idx, job in enumerate(jobs):
            result = _run_job(job, log_paths[idx], data_dir)
            if progress:
                status = "ERROR" if result.error else "ok"
                suffix = f" log={log_paths[idx]}" if log_paths[idx] else ""
                print(
                    f"[ablation] {idx + 1}/{len(jobs)} {job.position} "
                    f"seed={job.seed} variant={job.variant} {status}{suffix}",
                    flush=True,
                )
            results.append(result)
        return results

    ordered: list[AblationResult | None] = [None] * len(jobs)
    completed: list[AblationResult] = []
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
        futures = {
            pool.submit(_run_job, job, log_paths[idx], data_dir, lgbm_n_jobs): (idx, job)
            for idx, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            idx, job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - include process/bootstrap failures per job
                result = _error_result(job, exc)
            result = _with_log_path(result, log_paths[idx])
            if progress:
                status = "ERROR" if result.error else "ok"
                suffix = f" log={log_paths[idx]}" if log_paths[idx] else ""
                print(
                    f"[ablation] {idx + 1}/{len(jobs)} {job.position} "
                    f"seed={job.seed} variant={job.variant} {status}{suffix}",
                    flush=True,
                )
            if preserve_order:
                ordered[idx] = result
            else:
                completed.append(result)

    if preserve_order:
        return [result for result in ordered if result is not None]
    return completed


def result_to_dict(result: AblationResult) -> dict[str, Any]:
    return asdict(result)


def write_history(
    name: str,
    results: list[AblationResult],
    *,
    metadata: dict[str, Any] | None = None,
    history_dir: str = HISTORY_DIR,
) -> str:
    """Write ablation results to ``benchmark_history/ablations``."""

    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{name}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": name,
        "results": [result_to_dict(result) for result in results],
    }
    if metadata:
        entry.update(metadata)
    return append_to_history(history_dir, entry)
