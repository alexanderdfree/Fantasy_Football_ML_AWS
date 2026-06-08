"""Ridge-only Principal-Component-Regression (PCR) ablation for any position.

WHY THIS EXISTS
---------------
The 2026-05 feature-collinearity audit (PR #594, ``analysis_feature_audit.py``)
showed QB/TE feed a ~1e15-condition matrix straight into Ridge (no
``ridge_pca_components`` set), while WR/RB/DST run PCA-before-Ridge. Open
question: does adding PCA-before-Ridge to QB/TE actually lower test MAE, or does
tuned Ridge's L2 already handle the conditioning so PCA only adds bias?

PCA feeds ONLY the Ridge path (verified: NN/attention/LightGBM consume the raw
scaled+clipped features — see ``src/shared/pipeline.py`` ``_cpu_branch`` vs
``_gpu_branch``). Ridge is deterministic. So a Ridge-only A/B at several
``pca_n`` is an EXACT validation — no NN seed noise, no GPU needed.

WHY IT GOES THROUGH ``run_pipeline`` (do NOT hand-roll the feature matrix)
-------------------------------------------------------------------------
A standalone sweep that loads the split + calls ``add_specific_features``
directly (like ``src/wr/benchmark_ridge_variants.py``) MISSES the
weather/Vegas/contextual columns (``is_dome``, ``implied_opp_total``,
``wind_adjusted``, ``is_divisional``, ``temp_adjusted``). Those are merged in by
``merge_schedule_features`` INSIDE ``build_position_features`` at pipeline time —
they are NOT in the base parquet splits. A hand-rolled sweep crashes with
``KeyError: ['is_dome', ...] not in index`` (or, worse, silently audits a
matrix missing 5 production features). This harness calls the real
``run_pipeline`` with a Ridge-only config override, so the feature matrix is
byte-identical to what production Ridge sees. (This is the training/inference
drift CLAUDE.md warns about — don't reintroduce it.)

PERFORMANCE — READ BEFORE RUNNING (this once hung for an hour)
-------------------------------------------------------------
``_tune_ridge_alphas_cv`` fans every alpha over ``joblib.Parallel(n_jobs=-1,
prefer="threads")``, and each Ridge solve calls multi-threaded BLAS. Run
WITHOUT thread caps, that oversubscribes (joblib threads x BLAS threads) and
thrashes. ALWAYS run with:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python -m src.tuning.ablate_ridge_pca --positions QB TE

With caps, one Ridge-only ``run_pipeline`` call is ~30-90s (alpha tuning is the
cost). The full default sweep = 2 positions x N pca_n x 2 seasons (val+test)
calls; budget accordingly, or narrow ``--variants``.

DATA — needs CURRENT splits (the local symlink is usually stale)
---------------------------------------------------------------
Reads ``data/splits/{train,val,test}.parquet`` (relative ``SPLITS_DIR``). In a
worktree ``data/splits`` is typically a symlink to the parent's STALE shared
splits (missing recently-added features -> ``KeyError`` in feature building).
Rebuild fresh splits into a LOCAL dir first (do NOT write through the symlink —
it corrupts the parent's shared data):

    test -L data/splits && rm data/splits        # remove ONLY the symlink
    mkdir -p data/splits
    OPENBLAS_NUM_THREADS=1 python - <<'PY'
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.features.engineer import build_features
    from src.data.split import temporal_split
    temporal_split(build_features(preprocess(load_raw_data())))   # uses cached data/raw
    PY

Restore the symlink afterwards if you want the shared splits back.

VAL + TEST (robustness against single-season noise)
---------------------------------------------------
``run_pipeline`` evaluates Ridge on whatever ``test_df`` you pass, and Ridge
fits on TRAIN only (val is for NN early-stopping, which is off here). So this
harness gets BOTH holdout seasons faithfully: once with the real 2025 test, and
once passing the 2024 val frame AS the test frame. A pca_n only "wins" if it
beats the no-PCA baseline on BOTH seasons (don't headline a single-season flip).
Both MAEs are stored in a single AblationResult so the decision table can apply
the consistent-improver rule in one pass.

DECIDING / SHIPPING
-------------------
Report is val_MAE + test_MAE at each pca_n, delta vs None. Ship a config change
(add ``ridge_pca_components=<n>`` to ``src/{pos}/config.py``) ONLY if a single
pca_n improves BOTH seasons by more than benchmark noise AND Ridge is the served
(best) model for that position in the latest ``benchmark_history`` — otherwise it
burns a full GPU retrain (config edits scope a full retrain; there is no
Ridge-only production retrain path) for a gain the served prediction never sees.
See the TODO.md note "[OPEN] PCA-before-Ridge for QB/TE" for the full handoff.

Usage::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python -m src.tuning.ablate_ridge_pca --positions QB TE
    # dry-run to see the plan:
    python -m src.tuning.ablate_ridge_pca --dry-run
    # custom pca grid via variants:
    python -m src.tuning.ablate_ridge_pca --positions TE --variants none,pca80,pca60,pca40
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any

import pandas as pd

from src.config import SPLITS_DIR
from src.shared.registry import get_config, get_runner
from src.tuning.ablation_runner import (
    AblationJob,
    AblationResult,
    fmt_mean_std,
    format_dry_run_table,
    mean_std,
    parse_seed_list,
    resolve_max_workers,
    run_grid,
    select_variants,
    write_history,
)

ABLATION_NAME = "ridge_pca"
DEFAULT_LOG_DIR = os.path.join("logs", "ablations", ABLATION_NAME)
DEFAULT_POSITIONS = ("QB", "TE")
# Ridge is deterministic — seed noise is zero, so a single seed is sufficient.
# Multi-seed is preserved so the harness is consistent with the runner shape
# and a caller can confirm determinism by passing multiple seeds and checking
# that std≈0 in the output.
DEFAULT_SEEDS = (42,)
BASELINE_VARIANT = "none"

# Default pca_n values to sweep (None = no-PCA baseline, always first). Chosen
# around the audit's hypothetical-PCA recommendation (~53-56 components @ 99%
# variance for QB/TE) plus a spread of more aggressive truncations.
# Variant keys encode pca_n: "none" -> None, "pcaN" -> N.
VARIANTS: dict[str, int | None] = {
    "none": None,
    "pca80": 80,
    "pca60": 60,
    "pca55": 55,
    "pca50": 50,
    "pca40": 40,
    "pca30": 30,
    "pca20": 20,
}
DEFAULT_VARIANTS = tuple(VARIANTS.keys())


def _make_cfg(
    base_cfg: dict[str, Any],
    pca_n: int | None,
) -> dict[str, Any]:
    """Return a deep-copied Ridge-only config with ``ridge_pca_components`` set.

    PCA feeds ONLY the Ridge path. Disabling NN/attention/LightGBM/ElasticNet
    makes the A/B exact (byte-identical Ridge feature matrix) and GPU-free.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["train_base_nn"] = False
    cfg["train_attention_nn"] = False
    cfg["train_lightgbm"] = False
    cfg["train_elasticnet"] = False
    cfg["train_ridge"] = True
    cfg["ridge_pca_components"] = pca_n
    return cfg


def _execute_ridge_pca_job(job: AblationJob) -> dict[str, Any]:
    """Execute one (position, seed, pca_n) job.

    Evaluates Ridge on BOTH holdout frames — val-as-2024-test and the real 2025
    test — and stores both MAEs in ``metrics`` so the decision table can apply
    the consistent-improver rule without needing a second job.
    """
    pca_n: int | None = job.base_cfg.get("_job_pca_n")  # sentinel injected by _build_jobs
    cfg = _make_cfg(job.base_cfg, pca_n)

    train = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    val = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
    test = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    run_fn = get_runner(job.position)

    # Val-as-test: ridge fits on train, evaluates on val frame (2024 season).
    result_val = run_fn(train_df=train, val_df=val, test_df=val, seed=job.seed, config=cfg)
    rm_val = result_val.get("ridge_metrics")
    if not rm_val or "total" not in rm_val:
        raise RuntimeError(
            f"{job.position}: run_pipeline returned no ridge_metrics['total'] "
            f"(keys={list(rm_val) if rm_val else None}). Did train_ridge get disabled?"
        )
    val_mae = float(rm_val["total"]["mae"])

    # Real test: ridge fits on train, evaluates on real 2025 test frame.
    result_test = run_fn(train_df=train, val_df=val, test_df=test, seed=job.seed, config=cfg)
    rm_test = result_test.get("ridge_metrics")
    if not rm_test or "total" not in rm_test:
        raise RuntimeError(
            f"{job.position}: run_pipeline returned no ridge_metrics['total'] for test frame "
            f"(keys={list(rm_test) if rm_test else None})."
        )
    test_mae = float(rm_test["total"]["mae"])

    phase_seconds = result_test.get("phase_seconds") or {}

    metrics = {
        "val_mae": val_mae,
        "test_mae": test_mae,
        "pca_n": pca_n,
    }
    timings = {
        "ridge_train_sec": (
            float(phase_seconds["ridge_train"])
            if phase_seconds.get("ridge_train") is not None
            else None
        ),
    }
    metadata = dict(job.metadata)
    metadata["pca_n"] = pca_n
    return {"metrics": metrics, "timings": timings, "metadata": metadata}


def _build_jobs(
    *,
    positions: list[str],
    seeds: list[int],
    variants: list[str],
) -> list[AblationJob]:
    jobs: list[AblationJob] = []
    for position in positions:
        base_cfg = get_config(position)
        for seed in seeds:
            for variant_key in variants:
                pca_n = VARIANTS[variant_key]
                # Inject pca_n into base_cfg via a private sentinel key so the
                # spawned worker process can recover it without a closure
                # (closures are not picklable across ProcessPoolExecutor workers).
                job_cfg = dict(base_cfg)
                job_cfg["_job_pca_n"] = pca_n
                jobs.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant_key,
                        label=f"pca_n={pca_n}",
                        run_fn=_execute_ridge_pca_job,
                        base_cfg=job_cfg,
                        metadata={
                            "run_kind": "experiment",
                            "pca_n": pca_n,
                            "variant_label": f"pca_n={pca_n}",
                        },
                    )
                )
    return jobs


def summarize_results(
    results: list[AblationResult],
    *,
    variants: list[str],
) -> dict[str, Any]:
    """Build a decision summary keyed by position.

    Consistent improver = beats the ``none`` (no-PCA) baseline on BOTH val
    and test MAE across all seeds. Decision mirrors the original sequential
    harness's rule exactly.
    """
    experiment_results = [
        result for result in results if result.metadata.get("run_kind") == "experiment"
    ]
    positions = sorted({result.position for result in experiment_results})
    summary: dict[str, Any] = {}

    for position in positions:
        # Collect per-variant mean val/test MAE across seeds.
        baseline_rows = [
            r
            for r in experiment_results
            if r.position == position and r.variant == BASELINE_VARIANT and r.error is None
        ]
        baseline_val_vals = [float(r.metrics["val_mae"]) for r in baseline_rows]
        baseline_test_vals = [float(r.metrics["test_mae"]) for r in baseline_rows]
        base_val_mean = (
            float(sum(baseline_val_vals) / len(baseline_val_vals)) if baseline_val_vals else None
        )
        base_test_mean = (
            float(sum(baseline_test_vals) / len(baseline_test_vals)) if baseline_test_vals else None
        )

        pos_summary: dict[str, Any] = {
            "variants": {},
            "consistent_improvers": [],
            "recommendation": None,
        }

        for variant_key in variants:
            rows = [
                r
                for r in experiment_results
                if r.position == position and r.variant == variant_key and r.error is None
            ]
            val_vals = [float(r.metrics["val_mae"]) for r in rows]
            test_vals = [float(r.metrics["test_mae"]) for r in rows]
            pca_n = VARIANTS[variant_key]

            val_stat = mean_std(val_vals)
            test_stat = mean_std(test_vals)

            val_delta_vals = (
                [v - bv for v, bv in zip(val_vals, baseline_val_vals, strict=False)]
                if baseline_val_vals and val_vals
                else []
            )
            test_delta_vals = (
                [v - bv for v, bv in zip(test_vals, baseline_test_vals, strict=False)]
                if baseline_test_vals and test_vals
                else []
            )

            val_delta = mean_std(val_delta_vals)
            test_delta = mean_std(test_delta_vals)

            is_baseline = variant_key == BASELINE_VARIANT
            val_mean = val_stat.get("mean")
            test_mean = test_stat.get("mean")
            consistent = bool(
                not is_baseline
                and val_mean is not None
                and test_mean is not None
                and base_val_mean is not None
                and base_test_mean is not None
                and val_mean < base_val_mean
                and test_mean < base_test_mean
            )

            pos_summary["variants"][variant_key] = {
                "pca_n": pca_n,
                "val_mae": val_stat,
                "test_mae": test_stat,
                "val_delta_vs_baseline": val_delta,
                "test_delta_vs_baseline": test_delta,
                "consistent_improver": consistent,
            }
            if consistent:
                pos_summary["consistent_improvers"].append(variant_key)

        # Pick best consistent improver by test_mae mean.
        improvers = [
            (vk, pos_summary["variants"][vk]["test_mae"]["mean"])
            for vk in pos_summary["consistent_improvers"]
            if pos_summary["variants"][vk]["test_mae"].get("mean") is not None
        ]
        if improvers:
            best_key, best_test = min(improvers, key=lambda x: x[1])
            best_pca_n = VARIANTS[best_key]
            best_rec = pos_summary["variants"][best_key]
            val_d = best_rec["val_delta_vs_baseline"].get("mean")
            test_d = best_rec["test_delta_vs_baseline"].get("mean")
            val_d_str = f"{val_d:+.4f}" if val_d is not None else "n/a"
            test_d_str = f"{test_d:+.4f}" if test_d is not None else "n/a"
            pos_summary["recommendation"] = (
                f"consistent improvers: {[VARIANTS[k] for k in pos_summary['consistent_improvers']]}; "
                f"best by test MAE = pca_n={best_pca_n} "
                f"(val {val_d_str}, test {test_d_str})"
            )
        else:
            pos_summary["recommendation"] = (
                "NO pca_n beats None on BOTH val and test (PCA not worth shipping here)."
            )

        summary[position] = pos_summary
    return summary


def print_summary(summary: dict[str, Any], variants: list[str]) -> None:
    print("\nRidge-PCR ablation summary")
    print("=" * 78)
    for position, pos_summary in summary.items():
        print(f"\n{'=' * 78}\n=== {position} ===\n{'=' * 78}")
        print(
            f"{'pca_n':>6} {'val_MAE':>18} {'val_dN':>18} {'test_MAE':>18} {'test_dN':>18} {'improver':>10}"
        )
        print("-" * 96)
        for variant_key in variants:
            rec = pos_summary["variants"].get(variant_key)
            if not rec:
                continue
            pca_n = rec["pca_n"]
            label = "None" if pca_n is None else str(pca_n)
            val_str = fmt_mean_std(
                [rec["val_mae"]["mean"]] if rec["val_mae"].get("mean") is not None else []
            )
            test_str = fmt_mean_std(
                [rec["test_mae"]["mean"]] if rec["test_mae"].get("mean") is not None else []
            )
            vd_mean = rec["val_delta_vs_baseline"].get("mean")
            td_mean = rec["test_delta_vs_baseline"].get("mean")
            vd_str = f"{vd_mean:+.4f}" if vd_mean is not None else ""
            td_str = f"{td_mean:+.4f}" if td_mean is not None else ""
            improver_str = "YES" if rec["consistent_improver"] else ""
            print(
                f"{label:>6} {val_str:>18} {vd_str:>18} {test_str:>18} {td_str:>18} {improver_str:>10}"
            )
        print(f"  -> {pos_summary['recommendation']}")


def _parse_positions(raw_positions: list[str]) -> list[str]:
    all_valid = ("QB", "RB", "WR", "TE", "K", "DST")
    positions = [pos.upper() for pos in raw_positions]
    unknown = [pos for pos in positions if pos not in all_valid]
    if unknown:
        raise ValueError(f"unknown position(s): {unknown}; choose from {list(all_valid)}")
    return positions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(DEFAULT_POSITIONS),
        help="Positions to run (default: QB TE)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=(
            "Comma-separated seeds (default: 42; Ridge is deterministic so "
            "std≈0 — add seeds to confirm)"
        ),
    )
    parser.add_argument(
        "--variants",
        default=None,
        help=(
            "Comma-separated variant keys or 'all'. "
            f"Available: {', '.join(VARIANTS)}. Default: all."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument("--no-history", action="store_true", help="Do not write history JSON")
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers. Use an integer or 'auto' "
            "(default: local CUDA box -> up to 6; otherwise 1). "
            "Ridge-only is CPU-bound so fan-out is beneficial."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-job logs (default: {DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        positions = _parse_positions(args.positions)
        seeds = parse_seed_list(args.seeds)
        variants = select_variants(args.variants, VARIANTS, DEFAULT_VARIANTS)
    except ValueError as exc:
        parser.error(str(exc))

    if BASELINE_VARIANT not in variants:
        parser.error(
            f"'{BASELINE_VARIANT}' (no-PCA baseline) must be included so "
            "consistent-improver deltas can be computed"
        )

    jobs = _build_jobs(positions=positions, seeds=seeds, variants=variants)
    try:
        max_workers = resolve_max_workers(args.max_workers, job_count=len(jobs))
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(format_dry_run_table(jobs))
        print(f"\nExperiment workers: {max_workers} ({args.max_workers})")
        print(f"Job logs: {args.log_dir}")
        return

    print("\nRunning Ridge-PCR ablation jobs...")
    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    summary = summarize_results(results, variants=variants)
    print_summary(summary, variants=variants)
    print("\nABLATION_DONE")

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
                "summary": summary,
            },
        )

    errors = [result for result in results if result.error]
    if errors:
        for result in errors:
            print(
                f"ERROR {result.position} seed={result.seed} "
                f"variant={result.variant}: {result.error}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
