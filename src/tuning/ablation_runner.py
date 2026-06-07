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
import statistics
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
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
_LOCAL_CUDA_WORKERS = 6


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


def _cap_worker_threads() -> None:
    for name in _THREAD_CAP_VARS:
        os.environ[name] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"


def resolve_max_workers(raw: str | int, *, job_count: int) -> int:
    """Resolve a CLI worker count.

    ``auto`` is intentionally conservative off the many-core local CUDA box:
    serial keeps timing clean on CPU/MPS/small AWS GPU hosts, while the RTX 5080
    development box can run several tiny attention-NN ablation jobs at once
    without VRAM pressure.
    """

    if isinstance(raw, int):
        workers = raw
    elif str(raw).strip().lower() == "auto":
        workers = 1
        try:
            from src.shared.platform_detect import detect_platform

            plat = detect_platform()
            if plat.backend == "cuda" and (plat.cpu_count or 0) >= 12:
                workers = _LOCAL_CUDA_WORKERS
        except Exception:  # noqa: BLE001 - platform detection is best-effort
            workers = 1
    else:
        workers = int(str(raw))

    if workers < 1:
        raise ValueError("max_workers must be >= 1")
    return min(workers, max(1, int(job_count)))


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


def _run_job(job: AblationJob, log_path: str | None = None) -> AblationResult:
    _cap_worker_threads()
    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
                result = _coerce_result(job, job.run_fn(job))
        else:
            result = _coerce_result(job, job.run_fn(job))
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
    log_paths = [
        os.path.join(log_dir, _safe_log_name(job, idx)) if log_dir else None
        for idx, job in enumerate(jobs)
    ]
    if max_workers == 1:
        results = []
        for idx, job in enumerate(jobs):
            result = _run_job(job, log_paths[idx])
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
            pool.submit(_run_job, job, log_paths[idx]): (idx, job) for idx, job in enumerate(jobs)
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
