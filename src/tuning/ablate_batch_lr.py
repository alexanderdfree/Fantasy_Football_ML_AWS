"""Batch-size + learning-rate ablation for attention NNs.

The driving question is whether attention training can get faster by increasing
batch size without repeating the historical accuracy regression. This harness
holds architecture/loss/features fixed, varies only attention batch size and
the coupled LR/scheduler scale, and reports paired multi-seed MAE deltas plus
attention training time.

Examples:
    python -m src.tuning.ablate_batch_lr --dry-run
    python -m src.tuning.ablate_batch_lr --positions QB WR --seeds 42,43
    python -m src.tuning.ablate_batch_lr --ridge-sentinel-preflight
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

from src.shared.registry import get_config, get_runner
from src.tuning.ablation_runner import (
    AblationJob,
    AblationResult,
    format_dry_run_table,
    mean_std,
    paired_deltas,
    parse_seed_list,
    resolve_max_workers,
    run_grid,
    select_variants,
    write_history,
)

ABLATION_NAME = "batch_lr_attention"
DEFAULT_LOG_DIR = os.path.join("logs", "ablations", ABLATION_NAME)
DEFAULT_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
# 3 seeds by default (AGENTS.md "Default to 3 seeds for FP-MAE A/Bs"): 8 was too
# slow for a full six-position × five-variant sweep. Bump via --seeds (e.g.
# 42,43,44,45,46) when a variant's MAE delta lands inside the seed band and the
# decision hinges on it.
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_VARIANTS = ("baseline", "b2_lr1", "b2_lrsqrt", "b2_lrlin", "b4_lrsqrt")
BASELINE = "baseline"
SENTINEL_VARIANTS = ("baseline", "b2_lrsqrt")
MAX_BATCH_SIZE = 1024
MAE_TOLERANCE = 0.02
MIN_SPEEDUP_FOR_RECOMMENDATION = 0.10


@dataclass(frozen=True)
class BatchLrVariant:
    key: str
    label: str
    batch_multiplier: int
    lr_scale: float


VARIANTS: dict[str, BatchLrVariant] = {
    "baseline": BatchLrVariant("baseline", "production batch/LR", 1, 1.0),
    "b2_lr1": BatchLrVariant("b2_lr1", "2x batch, LR unchanged", 2, 1.0),
    "b2_lrsqrt": BatchLrVariant("b2_lrsqrt", "2x batch, LR sqrt-scaled", 2, math.sqrt(2.0)),
    "b2_lrlin": BatchLrVariant("b2_lrlin", "2x batch, LR linearly scaled", 2, 2.0),
    "b4_lrsqrt": BatchLrVariant("b4_lrsqrt", "4x batch, LR sqrt-scaled", 4, 2.0),
}


def _scheduler_fields(cfg: dict[str, Any]) -> dict[str, Any]:
    sched_type = cfg.get("scheduler_type")
    keys = ["scheduler_type"]
    if sched_type == "onecycle":
        keys.extend(["onecycle_max_lr", "attn_onecycle_max_lr", "onecycle_pct_start"])
    elif sched_type == "cosine_warm_restarts":
        keys.extend(["cosine_t0", "cosine_t_mult", "cosine_eta_min", "attn_cosine_eta_min"])
    elif sched_type == "plateau":
        keys.extend(["plateau_factor", "plateau_patience"])
    return {key: cfg.get(key) for key in keys if key in cfg}


def _scale_batch(batch_size: int, multiplier: int) -> int:
    return min(int(batch_size) * int(multiplier), MAX_BATCH_SIZE)


def _make_cfg(
    base_cfg: dict[str, Any],
    variant: BatchLrVariant,
    *,
    ridge_sentinel: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deep-copied config plus run metadata for one variant."""

    cfg = copy.deepcopy(base_cfg)
    base_batch = int(cfg.get("attn_batch_size", cfg["nn_batch_size"]))
    base_lr = float(cfg.get("attn_lr", cfg["nn_lr"]))
    base_scheduler = _scheduler_fields(cfg)

    cfg["attn_batch_size"] = _scale_batch(base_batch, variant.batch_multiplier)
    cfg["attn_lr"] = base_lr * variant.lr_scale

    sched_type = cfg.get("scheduler_type")
    if sched_type == "onecycle" and "onecycle_max_lr" in cfg:
        base_max_lr = cfg.get("attn_onecycle_max_lr", cfg["onecycle_max_lr"])
        cfg["attn_onecycle_max_lr"] = float(base_max_lr) * variant.lr_scale
    elif sched_type == "cosine_warm_restarts" and "cosine_eta_min" in cfg:
        base_eta_min = cfg.get("attn_cosine_eta_min", cfg["cosine_eta_min"])
        cfg["attn_cosine_eta_min"] = float(base_eta_min) * variant.lr_scale

    cfg["train_attention_nn"] = True
    cfg["train_base_nn"] = False
    cfg["train_elasticnet"] = False
    cfg["train_lightgbm"] = False
    cfg["train_ridge"] = bool(ridge_sentinel)

    metadata = {
        "variant_label": variant.label,
        "batch_multiplier": variant.batch_multiplier,
        "lr_scale": variant.lr_scale,
        "base_attn_batch_size": base_batch,
        "effective_attn_batch_size": cfg["attn_batch_size"],
        "base_attn_lr": base_lr,
        "effective_attn_lr": cfg["attn_lr"],
        "base_scheduler": base_scheduler,
        "effective_scheduler": _scheduler_fields(cfg),
        "ridge_sentinel": bool(ridge_sentinel),
    }
    return cfg, metadata


def _safe_min(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _safe_sum(values: list[float]) -> float | None:
    return float(sum(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _extract_run_payload(
    result: dict[str, Any],
    *,
    targets: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    attn = result.get("attn_nn_metrics")
    if not attn or "total" not in attn:
        raise RuntimeError(f"missing attention metrics in result keys: {sorted(result)}")

    attn_history = result.get("attn_history") or {}
    val_losses = [float(v) for v in (attn_history.get("val_loss") or [])]
    epoch_secs = [float(v) for v in (attn_history.get("epoch_sec") or [])]
    peak_mem = [float(v) for v in (attn_history.get("peak_mem_gb") or [])]
    phase_seconds = result.get("phase_seconds") or {}
    ridge = result.get("ridge_metrics") or {}

    metrics = {
        "attn_fp_mae": float(attn["total"]["mae"]),
        "attn_target_mae": {
            target: float(attn[target]["mae"]) for target in targets if target in attn
        },
        "min_attn_val_loss": _safe_min(val_losses),
        "epochs": len(val_losses) or len(epoch_secs),
        "ridge_fp_mae": (
            float(ridge["total"]["mae"]) if isinstance(ridge.get("total"), dict) else None
        ),
    }
    timings = {
        "attn_nn_train_sec": (
            float(phase_seconds["attn_nn_train"])
            if phase_seconds.get("attn_nn_train") is not None
            else None
        ),
        "epoch_sec_median": _safe_median(epoch_secs),
        "epoch_sec_sum": _safe_sum(epoch_secs),
        "peak_mem_gb_max": _safe_max(peak_mem),
    }
    return {"metrics": metrics, "timings": timings, "metadata": metadata}


def _execute_batch_lr_job(job: AblationJob) -> dict[str, Any]:
    variant = VARIANTS[job.variant]
    ridge_sentinel = bool(job.metadata.get("ridge_sentinel", False))
    cfg, run_metadata = _make_cfg(job.base_cfg, variant, ridge_sentinel=ridge_sentinel)
    run_metadata.update(job.metadata)

    run_fn = get_runner(job.position)
    result = run_fn(seed=job.seed, config=cfg)
    return _extract_run_payload(result, targets=cfg["targets"], metadata=run_metadata)


def _prime_feature_cache(position: str, *, log_dir: str | None = None) -> None:
    """Warm seed/variant-independent feature data before process fan-out."""

    base_cfg = get_config(position)
    cfg = copy.deepcopy(base_cfg)
    cfg["train_attention_nn"] = False
    cfg["train_base_nn"] = False
    cfg["train_elasticnet"] = False
    cfg["train_lightgbm"] = False
    cfg["train_ridge"] = False
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"cache_prime_{position.upper()}.log")
        print(f"[cache] priming {position.upper()} feature cache (log: {log_path})")
        with open(log_path, "w") as logf, redirect_stdout(logf), redirect_stderr(logf):
            get_runner(position)(seed=0, config=cfg)
    else:
        print(f"[cache] priming {position.upper()} feature cache before parallel ablation jobs")
        get_runner(position)(seed=0, config=cfg)


def _build_jobs(
    *,
    positions: list[str],
    seeds: list[int],
    variants: list[str],
    ridge_sentinel_preflight: bool,
) -> tuple[list[AblationJob], list[AblationJob]]:
    preflight: list[AblationJob] = []
    experiment: list[AblationJob] = []

    for position in positions:
        base_cfg = get_config(position)
        if ridge_sentinel_preflight:
            for variant_key in SENTINEL_VARIANTS:
                variant = VARIANTS[variant_key]
                preflight.append(
                    AblationJob(
                        position=position,
                        seed=seeds[0],
                        variant=variant_key,
                        label=variant.label,
                        run_fn=_execute_batch_lr_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "sentinel_preflight", "ridge_sentinel": True},
                    )
                )

        for seed in seeds:
            for variant_key in variants:
                variant = VARIANTS[variant_key]
                experiment.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant_key,
                        label=variant.label,
                        run_fn=_execute_batch_lr_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "experiment", "ridge_sentinel": False},
                    )
                )

    return preflight, experiment


def ridge_sentinel_by_position(results: list[AblationResult], *, atol: float = 1e-9) -> dict:
    out: dict[str, dict[str, Any]] = {}
    preflight = [r for r in results if r.metadata.get("run_kind") == "sentinel_preflight"]
    positions = sorted({r.position for r in preflight})
    for position in positions:
        rows = [r for r in preflight if r.position == position]
        vals = [
            float(r.metrics["ridge_fp_mae"])
            for r in rows
            if r.error is None and r.metrics.get("ridge_fp_mae") is not None
        ]
        if len(vals) < 2:
            out[position] = {"ok": None, "max_spread": None, "n": len(vals)}
            continue
        spread = max(vals) - min(vals)
        out[position] = {"ok": spread <= atol, "max_spread": spread, "n": len(vals)}
    return out


def _median_timing(
    results: list[AblationResult], position: str, variant: str, key: str
) -> float | None:
    vals = [
        float(r.timings[key])
        for r in results
        if r.position == position
        and r.variant == variant
        and r.error is None
        and r.metadata.get("run_kind") == "experiment"
        and r.timings.get(key) is not None
    ]
    return _safe_median(vals)


def _variant_metric_values(
    results: list[AblationResult], position: str, variant: str, metric_key: str
) -> list[float]:
    vals = []
    for result in results:
        if (
            result.position == position
            and result.variant == variant
            and result.error is None
            and result.metadata.get("run_kind") == "experiment"
        ):
            vals.append(float(result.metrics[metric_key]))
    return vals


def summarize_results(
    results: list[AblationResult],
    *,
    variants: list[str],
    sentinels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a decision summary keyed by position."""

    experiment_results = [
        result for result in results if result.metadata.get("run_kind") == "experiment"
    ]
    positions = sorted({result.position for result in experiment_results})
    summary: dict[str, Any] = {}
    for position in positions:
        baseline_time = _median_timing(experiment_results, position, BASELINE, "attn_nn_train_sec")
        pos_summary = {
            "sentinel": (sentinels or {}).get(position),
            "variants": {},
            "recommendation": None,
        }
        eligible: list[tuple[str, float]] = []
        for variant in variants:
            mae_vals = _variant_metric_values(experiment_results, position, variant, "attn_fp_mae")
            deltas = (
                [0.0 for _ in mae_vals]
                if variant == BASELINE
                else paired_deltas(
                    experiment_results,
                    variant=variant,
                    baseline_variant=BASELINE,
                    metric_key="attn_fp_mae",
                    position=position,
                )
            )
            train_time = _median_timing(experiment_results, position, variant, "attn_nn_train_sec")
            speedup = (
                (baseline_time - train_time) / baseline_time
                if baseline_time and train_time is not None
                else None
            )
            delta_stats = mean_std(deltas)
            mean_delta = delta_stats["mean"]
            std_delta = delta_stats["std"]
            not_clearly_worse = mean_delta is not None and (
                mean_delta <= 0.0 or (std_delta is not None and mean_delta <= std_delta)
            )
            is_eligible = bool(
                variant == BASELINE
                or (mean_delta is not None and mean_delta <= MAE_TOLERANCE and not_clearly_worse)
            )
            rec = {
                "attn_fp_mae": mean_std(mae_vals),
                "delta_vs_baseline": delta_stats,
                "attn_nn_train_sec_median": train_time,
                "speedup_vs_baseline": speedup,
                "eligible": is_eligible,
            }
            pos_summary["variants"][variant] = rec
            if variant != BASELINE and is_eligible and speedup is not None:
                eligible.append((variant, speedup))

        eligible.sort(key=lambda item: item[1], reverse=True)
        if eligible and eligible[0][1] >= MIN_SPEEDUP_FOR_RECOMMENDATION:
            variant, speedup = eligible[0]
            if position == "K":
                pos_summary["recommendation"] = (
                    f"{variant} is fastest eligible ({speedup:.1%}), but K requires a separate "
                    "served-model competitiveness check before any production recommendation."
                )
            else:
                pos_summary["recommendation"] = (
                    f"{variant} is fastest eligible ({speedup:.1%} faster than baseline)."
                )
        else:
            pos_summary["recommendation"] = "No eligible variant clears the 10% speedup bar."
        summary[position] = pos_summary
    return summary


def _fmt_stat(stat: dict[str, Any]) -> str:
    if stat.get("n", 0) == 0 or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.4f}+/-{stat['std']:.4f}"


def print_summary(summary: dict[str, Any], variants: list[str]) -> None:
    print("\nBatch/LR attention ablation summary")
    print("=" * 96)
    for position, pos_summary in summary.items():
        sentinel = pos_summary.get("sentinel")
        if sentinel:
            print(
                f"\n{position}  ridge sentinel: ok={sentinel['ok']} spread={sentinel['max_spread']}"
            )
        else:
            print(f"\n{position}")
        print(
            f"{'variant':<12} {'attn FP MAE':>18} {'delta':>18} "
            f"{'attn train s':>14} {'speedup':>10} {'eligible':>10}"
        )
        print("-" * 90)
        for variant in variants:
            rec = pos_summary["variants"].get(variant)
            if not rec:
                continue
            speedup = rec["speedup_vs_baseline"]
            speedup_text = f"{speedup:.1%}" if speedup is not None else "n/a"
            time_text = (
                f"{rec['attn_nn_train_sec_median']:.1f}"
                if rec["attn_nn_train_sec_median"] is not None
                else "n/a"
            )
            print(
                f"{variant:<12} {_fmt_stat(rec['attn_fp_mae']):>18} "
                f"{_fmt_stat(rec['delta_vs_baseline']):>18} {time_text:>14} "
                f"{speedup_text:>10} {str(rec['eligible']):>10}"
            )
        print(f"Recommendation: {pos_summary['recommendation']}")


def _parse_positions(raw_positions: list[str]) -> list[str]:
    positions = [pos.upper() for pos in raw_positions]
    unknown = [pos for pos in positions if pos not in DEFAULT_POSITIONS]
    if unknown:
        raise ValueError(f"unknown position(s): {unknown}; choose from {list(DEFAULT_POSITIONS)}")
    return positions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(DEFAULT_POSITIONS),
        help="Positions to run (default: all six)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 42,43,44; add more for borderline deltas)",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants, or 'all'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument("--no-history", action="store_true", help="Do not write history JSON")
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers for experiment jobs. Use an integer or 'auto' "
            "(default: local many-core CUDA -> 6, otherwise 1; pass 1 for clean timing)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-job logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--ridge-sentinel-preflight",
        action="store_true",
        help="Run one Ridge-enabled baseline/b2_lrsqrt preflight per position",
    )
    args = parser.parse_args(argv)

    try:
        positions = _parse_positions(args.positions)
        seeds = parse_seed_list(args.seeds)
        variants = select_variants(args.variants, VARIANTS, DEFAULT_VARIANTS)
    except ValueError as exc:
        parser.error(str(exc))
    if BASELINE not in variants:
        parser.error("baseline must be included so paired deltas can be computed")

    preflight_jobs, experiment_jobs = _build_jobs(
        positions=positions,
        seeds=seeds,
        variants=variants,
        ridge_sentinel_preflight=args.ridge_sentinel_preflight,
    )
    all_jobs = preflight_jobs + experiment_jobs
    try:
        max_workers = resolve_max_workers(args.max_workers, job_count=len(experiment_jobs))
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(format_dry_run_table(all_jobs))
        print(f"\nExperiment workers: {max_workers} ({args.max_workers})")
        print(f"Job logs: {args.log_dir}")
        return

    results: list[AblationResult] = []
    if preflight_jobs:
        print("\nRunning Ridge sentinel preflight serially...")
        results.extend(run_grid(preflight_jobs, max_workers=1, log_dir=args.log_dir, progress=True))

    if max_workers > 1:
        for position in positions:
            _prime_feature_cache(position, log_dir=args.log_dir)

    print("\nRunning batch/LR experiment jobs...")
    results.extend(
        run_grid(experiment_jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)
    )

    sentinels = ridge_sentinel_by_position(results)
    summary = summarize_results(results, variants=variants, sentinels=sentinels)
    print_summary(summary, variants)

    if not args.no_history:
        write_history(
            ABLATION_NAME,
            results,
            metadata={
                "positions": positions,
                "seeds": seeds,
                "variants": variants,
                "max_workers": max_workers,
                "log_dir": args.log_dir,
                "ridge_sentinel_preflight": bool(args.ridge_sentinel_preflight),
                "summary": summary,
            },
        )

    errors = [result for result in results if result.error]
    if errors:
        for result in errors:
            print(
                f"ERROR {result.position} seed={result.seed} variant={result.variant}: {result.error}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
