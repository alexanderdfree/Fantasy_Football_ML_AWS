"""Ablation: does the train-only ``MIN_GAMES_PER_SEASON`` filter help or hurt?

The production pipeline filters TRAIN rows to players with ``>= MIN_GAMES_PER_SEASON``
games in a season ([src/shared/pipeline.py] ``_prepare_position_data_uncached``,
the ``games_per_season >= MIN_GAMES_PER_SEASON`` line) but leaves VAL/TEST
unfiltered. So the model trains only on established players yet is evaluated on
everyone — a train->test covariate shift the split design itself manufactures,
landing on the low-volume / cold-start subgroup the model never saw.

This script sweeps the threshold and reports per-subgroup fantasy-point MAE so we
can see whether relaxing the filter helps the low-volume test rows
(``would_filter``, < 6 games in the test season) and at what cost to the
established rows (``kept``, >= 6 games).

Investigation-only:
  * It overrides ``cfg["min_games_per_season"]`` per run (the production knob added
    alongside this tool) rather than editing production code.
  * It lives in ``src/tuning/`` (no retrain trigger; scope_positions.py).
  * It writes results to ``benchmark_history/ablations/``.

The swept threshold flows through ``cfg["min_games_per_season"]``. The pipeline
reads that cfg key (the global ``MIN_GAMES_PER_SEASON`` is only the ``None``
fallback), so overriding the cfg — not monkeypatching the global, which a position's
own ``min_games_per_season`` would shadow — is what moves the filter.
``src/shared/feature_cache.py`` includes the key in the cache fingerprint, so
threshold variants get distinct cache entries and don't collide (no cache-disable
needed).

Two-phase protocol:
    # Phase 1 — deterministic Ridge+LGBM signal, full threshold curve, 1 seed
    python -m src.tuning.ablate_min_games --positions RB WR

    # Phase 2 — confirm the best (NN/attention) model on the decisive pair
    python -m src.tuning.ablate_min_games --positions RB WR --variants thr1,thr6 --seeds 42,43,44,45,46,47,48,49

Variants
--------
Each variant encodes a min-games threshold:
  thr1  → min_games_per_season = 1
  thr3  → min_games_per_season = 3
  thr6  → min_games_per_season = 6  (production default)
  thr8  → min_games_per_season = 8

Notes:
  * ``--seeds`` is a comma-separated list (e.g. ``42,43,44``) — the SAME list
    across every threshold, so threshold comparisons are paired. Deterministic
    models (Ridge, LightGBM) show std~=0; the seed spread is the NN/attention
    signal.
  * Default positions are RB/WR (highest low-volume fraction). K/DST are inert —
    DST has no <6-game team-seasons and K floors its own ``min_games`` in
    ``src/k/data.py`` before the shared filter, so the shared knob barely bites; the
    UNRELIABLE train-row flag marks K/DST (their ``filter_fn`` is identity on the
    shared skill-split). Trust the per-bucket test deltas (from the real pipeline).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.config import SPLITS_DIR
from src.shared.registry import get_config, get_runner
from src.tuning.ablation_runner import (
    AblationJob,
    AblationResult,
    format_dry_run_table,
    parse_seed_list,
    resolve_max_workers,
    run_grid,
    select_variants,
    write_history,
)

ABLATION_NAME = "min_games_filter"
DEFAULT_LOG_DIR = os.path.join("logs", "ablations", ABLATION_NAME)
DEFAULT_POSITIONS = ("RB", "WR")
DEFAULT_SEEDS = (42,)
BASELINE_VARIANT = "thr6"

# Threshold variants: key -> min_games_per_season value
VARIANTS: dict[str, int] = {
    "thr1": 1,
    "thr3": 3,
    "thr6": 6,
    "thr8": 8,
}
DEFAULT_VARIANTS = ("thr1", "thr3", "thr6", "thr8")

# Buckets printed in the headline tables (the rest feed the JSON + finer curve).
KEY_BUCKETS = ["ALL", "would_filter(<6)", "kept(>=6)"]


def _bucket_masks(g: pd.Series) -> dict[str, pd.Series]:
    """Test-side cohorts keyed by the player's game count IN THE TEST SEASON,
    computed with the SAME groupby the train filter uses. ``would_filter`` is the
    subgroup the production threshold (6) strips from TRAIN — the regime the model
    is under-exposed to but still evaluated on."""
    return {
        "ALL": pd.Series(True, index=g.index),
        "would_filter(<6)": g < 6,
        "kept(>=6)": g >= 6,
        "g=1": g == 1,
        "g=2-3": g.between(2, 3),
        "g=4-5": g.between(4, 5),
        "g=6-9": g.between(6, 9),
        "g>=10": g >= 10,
    }


def _pred_total_cols(test_df: pd.DataFrame) -> list[str]:
    """Per-model fantasy-point total columns attached by run_pipeline
    (``pred_ridge_total``, ``pred_nn_total``, ``pred_lgbm_total``,
    ``pred_attn_nn_total``, ...). Discovered dynamically so the table adapts to
    whichever models trained. ``pred_baseline`` lacks the ``_total`` suffix and is
    excluded."""
    return sorted(c for c in test_df.columns if c.startswith("pred_") and c.endswith("_total"))


def _short(pred_col: str) -> str:
    return pred_col[len("pred_") : -len("_total")]


def _subgroup_mae(test_df: pd.DataFrame, targets: list[str], agg: Any) -> dict[str, dict]:
    """Per-(bucket, model) fantasy-point MAE on the test set.

    The actual FP total reuses ``cfg["aggregate_fn"]`` applied to the actual stat
    columns — the same ``_total`` aggregator run_pipeline uses for predictions — so
    actual and predicted totals are scored like-for-like (production scoring, not a
    reimplementation). MAE of a vector is computed directly; that is identical to
    ``compute_metrics(...)["mae"]`` but avoids r2-on-tiny-bucket noise.
    """
    g = test_df.groupby(["player_id", "season"])["week"].transform("count")
    if agg is not None:
        actual_total = np.asarray(
            agg({t: test_df[t].to_numpy(dtype=float) for t in targets}), dtype=float
        )
    else:  # no aggregator registered — fall back to sum(heads) (skill-position shape)
        actual_total = np.sum([test_df[t].to_numpy(dtype=float) for t in targets], axis=0)

    pred_cols = _pred_total_cols(test_df)
    out: dict[str, dict] = {}
    for bname, mask in _bucket_masks(g).items():
        m = mask.to_numpy()
        n = int(m.sum())
        rec: dict[str, float] = {"n": n}
        if n:
            at = actual_total[m]
            for pc in pred_cols:
                rec[pc] = float(np.mean(np.abs(test_df[pc].to_numpy(dtype=float)[m] - at)))
        out[bname] = rec
    return out


def _train_rows_by_threshold(
    cfg: dict[str, Any], thresholds: list[int]
) -> tuple[dict[int, int], bool]:
    """Surviving TRAIN rows at each threshold — replicates the production filter
    (filter-to-position, then ``groupby(player_id, season).week.count() >= thr``) so
    the table shows how much each threshold strips.

    Returns ``(counts, reliable)``. ``reliable`` is False for K/DST: their
    ``filter_fn`` is identity on the shared (skill-only) ``train.parquet`` because
    K/DST load their real training data separately — so the count would reflect the
    *skill* split, not the position's true train population. For those, trust the
    per-bucket *test* deltas (which come from the real pipeline run), not this count.
    QB/RB/WR/TE counts are the true post-filter train sizes.
    """
    train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
    if "season_type" in train_df.columns:
        train_df = train_df[train_df["season_type"] == "REG"]
    pos_train = cfg["filter_fn"](train_df)
    reliable = len(pos_train) < len(train_df)  # filter_fn actually subset to a position
    g = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    return {int(thr): int((g >= thr).sum()) for thr in thresholds}, reliable


def _make_cfg(
    base_cfg: dict[str, Any],
    threshold: int,
    *,
    skip_nn: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deep-copied config with ``min_games_per_season`` set to ``threshold``.

    When ``skip_nn`` is True the NN/attention heads run at minimal epochs (Phase-1
    fast path); the degraded NN preds are excluded from the summary table by
    ``_execute_min_games_job``. The deterministic Ridge + LightGBM stay at full
    production config — not a weakened proxy.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["min_games_per_season"] = threshold
    if skip_nn:
        cfg["nn_epochs"] = 2
        cfg["train_attention_nn"] = False
        cfg["train_elasticnet"] = False
    metadata: dict[str, Any] = {
        "threshold": threshold,
        "skip_nn": skip_nn,
        "run_kind": "experiment",
    }
    return cfg, metadata


def _execute_min_games_job(job: AblationJob) -> dict[str, Any]:
    """Run one pipeline call and extract per-subgroup fantasy-point MAE."""
    threshold = VARIANTS[job.variant]
    skip_nn = bool(job.metadata.get("skip_nn", False))
    cfg, run_metadata = _make_cfg(job.base_cfg, threshold, skip_nn=skip_nn)
    run_metadata.update(job.metadata)

    run_fn = get_runner(job.position)
    result = run_fn(seed=job.seed, config=cfg)
    test_df = result["test_df"]
    buckets = _subgroup_mae(test_df, cfg["targets"], cfg.get("aggregate_fn"))
    pred_cols = _pred_total_cols(test_df)

    if skip_nn:
        # Skip-nn fast path: the base NN ran at minimal epochs (degraded) and
        # attention was skipped — report only the deterministic Ridge/LightGBM/ENet.
        pred_cols = [c for c in pred_cols if "nn" not in _short(c)]
        buckets = {
            b: {k: v for k, v in rec.items() if k == "n" or k in pred_cols}
            for b, rec in buckets.items()
        }

    metrics: dict[str, Any] = {
        "buckets": buckets,
        "pred_cols": pred_cols,
    }
    timings: dict[str, Any] = {}
    phase_seconds = result.get("phase_seconds") or {}
    if phase_seconds:
        timings["phase_seconds"] = {k: float(v) for k, v in phase_seconds.items()}

    return {"metrics": metrics, "timings": timings, "metadata": run_metadata}


def _build_jobs(
    *,
    positions: list[str],
    seeds: list[int],
    variants: list[str],
    skip_nn: bool,
) -> list[AblationJob]:
    jobs: list[AblationJob] = []
    for position in positions:
        base_cfg = get_config(position)
        for seed in seeds:
            for variant_key in variants:
                jobs.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant_key,
                        label=f"thr={VARIANTS[variant_key]}",
                        run_fn=_execute_min_games_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "experiment", "skip_nn": skip_nn},
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# Summary helpers (preserved from the original implementation)
# ---------------------------------------------------------------------------


def _aggregate_seeds(seed_runs: list[dict]) -> tuple[dict, list[str]]:
    """Collapse per-seed bucket dicts to mean/std per (bucket, model)."""
    pred_cols = seed_runs[0]["pred_cols"]
    agg: dict[str, dict] = {}
    for bname in seed_runs[0]["buckets"]:
        rec: dict = {"n": seed_runs[0]["buckets"][bname]["n"]}
        for pc in pred_cols:
            vals = [sr["buckets"][bname][pc] for sr in seed_runs if pc in sr["buckets"][bname]]
            if vals:
                rec[pc] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_seeds": len(vals),
                }
        agg[bname] = rec
    return agg, pred_cols


def _fmt(cell: dict | None, multi_seed: bool) -> str:
    if cell is None:
        return f"{'-':>11}"
    if multi_seed:
        return f"{cell['mean']:>7.3f}±{cell['std']:.3f}"
    return f"{cell['mean']:>11.3f}"


def _results_to_seed_runs(
    results: list[AblationResult],
    position: str,
    variant: str,
) -> list[dict]:
    """Re-package AblationResult metrics back into the per-seed dict shape that
    ``_aggregate_seeds`` and ``print_position_summary`` expect."""
    runs = []
    for r in results:
        if r.position == position and r.variant == variant and r.error is None:
            runs.append(
                {
                    "buckets": r.metrics.get("buckets", {}),
                    "pred_cols": r.metrics.get("pred_cols", []),
                }
            )
    return runs


def print_position_summary(
    pos: str,
    train_rows: dict,
    per_threshold: dict,
    baseline_threshold: int,
    train_rows_reliable: bool = True,
) -> None:
    """Print position-level threshold comparison table.

    ``per_threshold`` maps each threshold int to a list of per-seed run-dicts
    (shape: ``{"buckets": ..., "pred_cols": [...]}``) — same as the original
    implementation.
    """
    thresholds = sorted(per_threshold)
    agg_by_thr = {thr: _aggregate_seeds(per_threshold[thr])[0] for thr in thresholds}
    pred_cols = _aggregate_seeds(per_threshold[thresholds[0]])[1]
    multi_seed = len(per_threshold[thresholds[0]]) > 1
    models = [_short(pc) for pc in pred_cols]

    print(f"\n{'#' * 96}")
    print(
        f"# {pos.upper()} — min-games filter ablation"
        f"{'  (mean±std over seeds)' if multi_seed else ''}"
    )
    print(f"{'#' * 96}")
    caveat = (
        ""
        if train_rows_reliable
        else "  (UNRELIABLE — filter_fn identity on shared skill-split; trust test deltas)"
    )
    print(
        "train rows by threshold: "
        + "  ".join(f"{thr}={train_rows[thr]}" for thr in thresholds)
        + caveat
    )

    for bucket in KEY_BUCKETS:
        n = agg_by_thr[thresholds[0]][bucket]["n"]
        print(f"\n  bucket = {bucket}   [test n={n}]")
        header = f"  {'thresh':<8}" + "".join(f"{m:>13}" for m in models)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for thr in thresholds:
            row = agg_by_thr[thr][bucket]
            cells = "".join(_fmt(row.get(pc), multi_seed) + "  " for pc in pred_cols)
            tag = "  <- baseline" if thr == baseline_threshold else ""
            print(f"  {thr:<8}{cells}{tag}")

    # Verdict: Δ = MAE(baseline) - MAE(min threshold). Positive => relaxing helps.
    lo = thresholds[0]
    if baseline_threshold in agg_by_thr and lo != baseline_threshold:
        print(
            f"\n  VERDICT ({pos.upper()}): relaxing filter {baseline_threshold} -> {lo} "
            f"(Δ = MAE@{baseline_threshold} − MAE@{lo}; + = relaxing HELPS)"
        )
        for bucket in ["would_filter(<6)", "kept(>=6)"]:
            base, low = agg_by_thr[baseline_threshold][bucket], agg_by_thr[lo][bucket]
            parts = []
            for pc in pred_cols:
                if pc in base and pc in low:
                    d = base[pc]["mean"] - low[pc]["mean"]
                    noise = (
                        (base[pc]["std"] ** 2 + low[pc]["std"] ** 2) ** 0.5 if multi_seed else 0.0
                    )
                    parts.append(
                        f"{_short(pc)} Δ={d:+.3f}" + (f"±{noise:.3f}" if multi_seed else "")
                    )
            body = "  ".join(parts) if parts else "n=0 (no rows — filter inert for this bucket)"
            print(f"    {bucket:<18} {body}")
        print(
            "    (filter HURTS low-volume if would_filter Δ>0 beyond noise AND kept Δ not meaningfully <0)"
        )


def _summarize_results(
    results: list[AblationResult],
    positions: list[str],
    variants: list[str],
    baseline_threshold: int,
    skip_nn: bool,
) -> dict[str, Any]:
    """Build the per-position summary dict written to history."""
    out: dict[str, Any] = {}
    for pos in positions:
        thresholds = sorted(VARIANTS[v] for v in variants)
        per_threshold: dict[int, list[dict]] = {}
        for variant in variants:
            thr = VARIANTS[variant]
            seed_runs = _results_to_seed_runs(results, pos, variant)
            if seed_runs:
                per_threshold[thr] = seed_runs
        if not per_threshold:
            continue
        out[pos] = {
            "aggregated": {
                thr: _aggregate_seeds(per_threshold[thr])[0]
                for thr in thresholds
                if thr in per_threshold
            },
        }
    return out


def _parse_positions(raw_positions: list[str]) -> list[str]:
    from src.shared.registry import ALL_POSITIONS

    positions = [pos.upper() for pos in raw_positions]
    unknown = [pos for pos in positions if pos not in ALL_POSITIONS]
    if unknown:
        raise ValueError(f"unknown position(s): {unknown}; choose from {sorted(ALL_POSITIONS)}")
    return positions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(DEFAULT_POSITIONS),
        help="Positions to run (default: RB WR)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=(
            "Comma-separated seeds, same list across thresholds (default: 42). "
            "Deterministic models show std~=0; spread is NN signal. "
            "Use e.g. --seeds 42,43,44 for multi-seed A/Bs."
        ),
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Comma-separated threshold variants to sweep, or 'all'. "
            "Choices: thr1 thr3 thr6 thr8 (default: all four). "
            "Example: --variants thr1,thr6"
        ),
    )
    parser.add_argument(
        "--baseline-threshold",
        type=int,
        default=6,
        help="Threshold to diff against in the verdict (default: 6, production)",
    )
    parser.add_argument(
        "--skip-nn",
        action="store_true",
        help=(
            "Phase-1 fast path: train only the production Ridge+LightGBM "
            "(NN/attention skipped, not weakened). Deterministic and ~seed-invariant."
        ),
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers for jobs. Use an integer or 'auto' "
            "(default: local many-core CUDA -> 6, otherwise 1; pass 1 for clean timing)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs without running them",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing benchmark_history/ablations/",
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

    # Derive the baseline variant key from the --baseline-threshold integer.
    baseline_threshold: int = args.baseline_threshold
    baseline_variant_key = next((k for k, v in VARIANTS.items() if v == baseline_threshold), None)
    if baseline_variant_key is not None and baseline_variant_key not in variants:
        # Silently include baseline threshold in the variant set so paired verdicts work.
        variants = list(variants) + [baseline_variant_key]

    jobs = _build_jobs(
        positions=positions,
        seeds=seeds,
        variants=variants,
        skip_nn=args.skip_nn,
    )

    try:
        max_workers = resolve_max_workers(args.max_workers, job_count=len(jobs))
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(format_dry_run_table(jobs))
        print(f"\nExperiment workers: {max_workers} ({args.max_workers})")
        print(f"Job logs: {args.log_dir}")
        return

    print(f"\nRunning min-games filter ablation jobs ({len(jobs)} total)...")
    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    # Build per_threshold dicts for the summary printer (original shape).
    for pos in positions:
        per_threshold: dict[int, list[dict]] = {}
        for variant in variants:
            thr = VARIANTS[variant]
            seed_runs = _results_to_seed_runs(results, pos, variant)
            if seed_runs:
                per_threshold[thr] = seed_runs

        # Train-row sentinel: load once per position for the selected thresholds.
        cfg = get_config(pos)
        selected_thresholds = sorted(VARIANTS[v] for v in variants)
        try:
            train_rows, train_rows_reliable = _train_rows_by_threshold(cfg, selected_thresholds)
        except Exception:  # noqa: BLE001 — sentinel is informational, never blocks results
            train_rows = {thr: -1 for thr in selected_thresholds}
            train_rows_reliable = False

        if per_threshold:
            print_position_summary(
                pos, train_rows, per_threshold, baseline_threshold, train_rows_reliable
            )

    errors = [r for r in results if r.error]
    if errors:
        for r in errors:
            print(f"ERROR {r.position} seed={r.seed} variant={r.variant}: {r.error}")

    if not args.no_history:
        summary = _summarize_results(results, positions, variants, baseline_threshold, args.skip_nn)
        write_history(
            ABLATION_NAME,
            results,
            metadata={
                "positions": positions,
                "seeds": seeds,
                "variants": variants,
                "baseline_threshold": baseline_threshold,
                "skip_nn": args.skip_nn,
                "max_workers": max_workers,
                "log_dir": args.log_dir,
                "summary": summary,
            },
        )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
