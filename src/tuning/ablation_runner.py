"""Shared helpers for offline tuning ablations.

This module is intentionally small and lives under ``src/tuning`` so ablation
scripts can share seed/variant loops without moving experiment-only machinery
into ``src/shared``. The default runner is serial because several ablations
measure wall-clock. ``max_workers > 1`` is available for non-timing sweeps and
uses process workers so model training stays isolated per run.
"""

from __future__ import annotations

import os
import statistics
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso

HISTORY_DIR = os.path.join("benchmark_history", "ablations")
_THREAD_CAP_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
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
        os.environ.setdefault(name, "1")


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


def _run_job(job: AblationJob) -> AblationResult:
    _cap_worker_threads()
    try:
        return _coerce_result(job, job.run_fn(job))
    except Exception as exc:  # noqa: BLE001 - ablation runners should capture per-job failures
        return _error_result(job, exc, tb=traceback.format_exc())


def run_grid(
    jobs: list[AblationJob],
    *,
    max_workers: int = 1,
    preserve_order: bool = True,
) -> list[AblationResult]:
    """Run an ablation grid and return one result per job.

    ``max_workers=1`` is intentionally serial. Larger values use a
    ``ProcessPoolExecutor`` and are best reserved for non-timing diagnostics.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_workers == 1:
        return [_run_job(job) for job in jobs]

    ordered: list[AblationResult | None] = [None] * len(jobs)
    completed: list[AblationResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_job, job): (idx, job) for idx, job in enumerate(jobs)}
        for future in as_completed(futures):
            idx, job = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - include process/bootstrap failures per job
                result = _error_result(job, exc)
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
