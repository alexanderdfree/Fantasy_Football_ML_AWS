"""Ablation: do the injury / return features actually help, or are they
benchmark-flat? (operator-only CLI).

Companion to :mod:`src.analysis.injury_subgroup_error`. That script slices the
*current* model's error by injury/return subgroup; this one trains each position
twice — **with** and **without** the injury/return features — and reports the MAE
delta both overall and *within the returning subgroup*. The subgroup delta is the
point: per the draft-capital lesson (TODO.md ``[TESTED, REJECTED]``) a feature can
be flat in overall MAE yet move a ~15%-of-rows subgroup, so an overall-only
comparison can't settle "does it help".

Features removed (whichever are present for the position; RB carries no
``is_returning_from_absence``):
    game_status, practice_status, days_rest, is_returning_from_absence

Removal hits **both** model paths so the comparison is honest:
  * linear/tree — wrap ``get_feature_columns_fn`` so its column list excludes them
    (the pipeline slices ``X = df[cols]``);
  * attention NN — filter ``attn_static_features`` (the static-branch whitelist;
    these features sit in the ``contextual`` category, which feeds it).

Standing caveat: ``game_status`` is 96.8% constant in the 2025 test split (Out /
Doubtful self-eliminate — preprocessing drops no-play rows), so its *overall* MAE
delta is near-unmeasurable. The real signal is the ``returning`` subgroup and
``days_rest`` — read the per-subgroup deltas, not just GLOBAL.

Runs the full production config per variant (no skip-NN proxy — a reduced model
can flip the sign; see AGENTS.md). Multi-seed, with parallel workers on
many-core CUDA boxes.

Examples:
    python -m src.tuning.ablate_injury_features --dry-run
    python -m src.tuning.ablate_injury_features                          # QB RB WR TE
    python -m src.tuning.ablate_injury_features --positions QB RB        # subset
    python -m src.tuning.ablate_injury_features --seeds 42,43,44 --max-workers auto
    python -m src.tuning.ablate_injury_features --seeds 7 --no-history
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

from src.analysis.analysis_rb_lgbm_disagreement import (  # reuse pure helpers
    available_models,
    per_model_metrics,
)
from src.analysis.injury_subgroup_error import SMALL_N, SUBGROUP_SPECS
from src.shared.registry import get_config, get_runner
from src.tuning.ablation_runner import (
    AblationJob,
    AblationResult,
    fmt_mean_std,
    format_dry_run_table,
    parse_seed_list,
    resolve_max_workers,
    run_grid,
    select_variants,
    write_history,
)

ABLATION_NAME = "injury_features"
DEFAULT_LOG_DIR = "logs/ablations/injury_features"

# Default to the skill positions: largest returning subgroups and the designation
# features live in their `contextual` category. Extend to K/DST only if A shows
# an effect (per the plan).
DEFAULT_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_SEEDS = (42,)

# Variant keys: "with" = baseline (production cfg), "without" = injury features dropped.
VARIANT_WITH = "with"
VARIANT_WITHOUT = "without"
DEFAULT_VARIANTS = (VARIANT_WITH, VARIANT_WITHOUT)
VARIANTS: dict[str, str] = {
    VARIANT_WITH: "with injury/return features (baseline)",
    VARIANT_WITHOUT: "WITHOUT injury/return features",
}

# The injury/return features under test. Whichever are present for a given
# position are removed; absent names are no-ops.
INJURY_FEATURES = frozenset(
    {"game_status", "practice_status", "days_rest", "is_returning_from_absence"}
)

# Subgroups shown in the per-position delta table (subset of A's specs, in order).
_DELTA_SUBGROUPS = ["global", "returning", "ret_1wk", "ret_2wk", "questionable"]


# ---------------------------------------------------------------------------
# Config surgery helpers — preserved exactly from the original script
# ---------------------------------------------------------------------------


def _drop_injury_features(cfg: dict) -> dict:
    """Deep-copy ``cfg`` and strip the injury/return features from both model
    paths. Module-level callables are atomic to ``deepcopy`` (the wrapped
    function still calls the original), matching the ablate_rb_gate.py pattern."""
    cfg = copy.deepcopy(cfg)
    orig_get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [c for c in orig_get_cols() if c not in INJURY_FEATURES]
    cfg["attn_static_features"] = [
        c for c in cfg["attn_static_features"] if c not in INJURY_FEATURES
    ]
    return cfg


def _dropped_summary(base_cfg: dict, drop_cfg: dict) -> dict:
    """Which features the ablation actually removed from each path. A 0-feature
    drop silently invalidates the comparison (the MAE Δ=0.0000 smell), so this is
    printed and checked before the (expensive) runs."""
    base_cols = set(base_cfg["get_feature_columns_fn"]())
    drop_cols = set(drop_cfg["get_feature_columns_fn"]())
    return {
        "linear_tree_dropped": sorted(base_cols - drop_cols),
        "attn_static_dropped": sorted(
            set(base_cfg["attn_static_features"]) - set(drop_cfg["attn_static_features"])
        ),
    }


def _subgroup_metrics(df: Any, models: dict[str, str]) -> dict:
    """Per-subgroup per-model metrics on ``df``, reusing A's subgroup definitions
    so 'with' and 'without' are sliced identically."""
    out: dict[str, dict] = {}
    n_total = len(df)
    for key, label, needed, mask_fn in SUBGROUP_SPECS:
        if needed is not None and needed not in df.columns:
            continue
        sub = df[mask_fn(df)]
        out[key] = {
            "label": label.strip(),
            "n": len(sub),
            "pct": round(100 * len(sub) / n_total, 1) if n_total else 0.0,
            "models": per_model_metrics(sub, models),
        }
    return out


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _execute_injury_job(job: AblationJob) -> dict[str, Any]:
    """Run one (position, seed, variant) job and return metrics/timings/metadata."""
    base_cfg = job.base_cfg

    # Apply the feature-drop for the "without" variant; leave "with" untouched.
    if job.variant == VARIANT_WITHOUT:
        cfg = _drop_injury_features(base_cfg)
    else:
        cfg = copy.deepcopy(base_cfg)

    run_fn = get_runner(job.position)
    result = run_fn(seed=job.seed, config=cfg)

    df = result["test_df"].copy()
    models = available_models(df)
    subgroups = _subgroup_metrics(df, models)

    # Capture phase_seconds timing when available.
    phase_seconds = result.get("phase_seconds") or {}
    timings = {k: float(v) for k, v in phase_seconds.items() if v is not None}

    # Build dropped summary so it surfaces in the history record.
    if job.variant == VARIANT_WITHOUT:
        dropped = _dropped_summary(base_cfg, cfg)
    else:
        dropped = {"linear_tree_dropped": [], "attn_static_dropped": []}

    metadata = {
        **job.metadata,
        "variant_label": VARIANTS[job.variant],
        "dropped": dropped,
        "model_list": list(models),
    }

    return {
        "metrics": {"subgroups": subgroups},
        "timings": timings,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Job grid construction
# ---------------------------------------------------------------------------


def _build_jobs(
    *,
    positions: list[str],
    seeds: list[int],
    variants: list[str],
) -> list[AblationJob]:
    jobs: list[AblationJob] = []
    for position in positions:
        base_cfg = get_config(position)
        drop_cfg = _drop_injury_features(base_cfg)
        dropped = _dropped_summary(base_cfg, drop_cfg)

        # Print drop summary once per position (before any job is spawned).
        print(
            f"[{position}] dropped from linear/tree : {dropped['linear_tree_dropped']}",
            flush=True,
        )
        print(
            f"[{position}] dropped from attn static  : {dropped['attn_static_dropped']}",
            flush=True,
        )
        if not dropped["linear_tree_dropped"] and not dropped["attn_static_dropped"]:
            print(
                f"[{position}] WARNING: no injury features present to drop"
                f" — ablation is a NO-OP for {position}.",
                flush=True,
            )

        for seed in seeds:
            for variant in variants:
                jobs.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant,
                        label=VARIANTS[variant],
                        run_fn=_execute_injury_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "experiment"},
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# Result aggregation + decision table printing
# ---------------------------------------------------------------------------


def _result_map(
    results: list[AblationResult],
) -> dict[tuple[str, int, str], AblationResult]:
    """Index results by (position, seed, variant) for paired lookups."""
    return {(r.position, r.seed, r.variant): r for r in results if r.error is None}


def _best_model_from_result(result: AblationResult) -> str:
    """Best model by global MAE in a single result."""
    g = result.metrics["subgroups"]["global"]["models"]
    return min(g, key=lambda k: g[k]["mae"])


def _subgroup_mae(result: AblationResult, subgroup_key: str, model: str) -> float | None:
    sg = result.metrics.get("subgroups", {}).get(subgroup_key)
    if not sg:
        return None
    return sg["models"].get(model, {}).get("mae")


def _print_position_table(
    pos: str,
    results: list[AblationResult],
    seeds: list[int],
) -> None:
    """Print per-position Δ = without − with table (global all-models + best-model
    by subgroup), averaged across seeds with mean±std formatting."""

    rmap = _result_map(results)
    pos_results_with = [
        rmap[(pos, s, VARIANT_WITH)] for s in seeds if (pos, s, VARIANT_WITH) in rmap
    ]
    if not pos_results_with:
        print(f"\n[{pos}] No successful 'with' results — skipping table.")
        return

    # Determine best model from the first successful 'with' result.
    best = _best_model_from_result(pos_results_with[0])

    print(f"\n{'=' * 86}")
    print(f"{pos} injury/return ablation — Δ = without − with  (positive ⇒ features HELP)")
    print(f"best model (baseline GLOBAL MAE): {best}")
    print("=" * 86)

    # 1) All models, GLOBAL — is removal benchmark-flat across the board?
    print("\n  GLOBAL — every model (mean±std across seeds):")
    print(f"    {'model':14}{'with':>18}{'without':>18}{'Δ (mean±std)':>18}")

    # Collect model names from first with result.
    all_models = list(pos_results_with[0].metrics["subgroups"]["global"]["models"])
    for model_name in all_models:
        with_maes: list[float] = []
        without_maes: list[float] = []
        for seed in seeds:
            r_with = rmap.get((pos, seed, VARIANT_WITH))
            r_without = rmap.get((pos, seed, VARIANT_WITHOUT))
            if r_with is not None:
                v = _subgroup_mae(r_with, "global", model_name)
                if v is not None:
                    with_maes.append(v)
            if r_without is not None:
                v = _subgroup_mae(r_without, "global", model_name)
                if v is not None:
                    without_maes.append(v)
        deltas = [b - a for a, b in zip(with_maes, without_maes, strict=False)]
        print(
            f"    {model_name:14}{fmt_mean_std(with_maes):>18}"
            f"{fmt_mean_std(without_maes):>18}{fmt_mean_std(deltas):>18}"
        )

    # 2) Best model across subgroups — where (if anywhere) does it move?
    print(f"\n  {best} — by subgroup (mean±std across seeds):")
    print(f"    {'subgroup':28}{'n':>6}{'with':>18}{'without':>18}{'Δ (mean±std)':>18}")
    for sg_key in _DELTA_SUBGROUPS:
        # Use first with-result to check presence / n.
        first_sg = pos_results_with[0].metrics["subgroups"].get(sg_key)
        if not first_sg or first_sg["n"] == 0:
            continue
        with_maes = []
        without_maes = []
        for seed in seeds:
            r_with = rmap.get((pos, seed, VARIANT_WITH))
            r_without = rmap.get((pos, seed, VARIANT_WITHOUT))
            if r_with is not None:
                v = _subgroup_mae(r_with, sg_key, best)
                if v is not None:
                    with_maes.append(v)
            if r_without is not None:
                v = _subgroup_mae(r_without, sg_key, best)
                if v is not None:
                    without_maes.append(v)
        deltas = [b - a for a, b in zip(with_maes, without_maes, strict=False)]
        n = first_sg["n"]
        label = first_sg["label"][:28]
        flag = f"  [small-n<{SMALL_N}]" if n < SMALL_N else ""
        print(
            f"    {label:28}{n:>6}{fmt_mean_std(with_maes):>18}"
            f"{fmt_mean_std(without_maes):>18}{fmt_mean_std(deltas):>18}{flag}"
        )


def _print_cross_summary(
    results: list[AblationResult],
    positions: list[str],
    seeds: list[int],
) -> None:
    """Cross-position summary: Δ(without − with) FP MAE, best model."""
    print(f"\n{'=' * 78}")
    print("CROSS-POSITION SUMMARY — Δ(without − with) FP MAE, best model")
    print("positive Δ ⇒ removing the features RAISED MAE ⇒ the features HELP that slice")
    print("=" * 78)
    print(f"{'Pos':<5}{'model':12}{'global Δ':>18}{'returning Δ':>18}{'n_ret':>7}")
    print("-" * 60)

    rmap = _result_map(results)
    for pos in positions:
        pos_with = [rmap[(pos, s, VARIANT_WITH)] for s in seeds if (pos, s, VARIANT_WITH) in rmap]
        if not pos_with:
            continue
        best = _best_model_from_result(pos_with[0])

        global_deltas: list[float] = []
        ret_deltas: list[float] = []
        n_ret = 0
        for seed in seeds:
            r_with = rmap.get((pos, seed, VARIANT_WITH))
            r_without = rmap.get((pos, seed, VARIANT_WITHOUT))
            if r_with is None or r_without is None:
                continue
            a_g = _subgroup_mae(r_with, "global", best)
            b_g = _subgroup_mae(r_without, "global", best)
            if a_g is not None and b_g is not None:
                global_deltas.append(b_g - a_g)

            a_r = _subgroup_mae(r_with, "returning", best)
            b_r = _subgroup_mae(r_without, "returning", best)
            if a_r is not None and b_r is not None:
                ret_deltas.append(b_r - a_r)

            # n_ret from the first seed's 'with' result (consistent across seeds).
            if n_ret == 0:
                sg = r_with.metrics["subgroups"].get("returning")
                if sg:
                    n_ret = sg["n"]

        print(
            f"{pos:<5}{best:12}{fmt_mean_std(global_deltas):>18}"
            f"{fmt_mean_std(ret_deltas):>18}{n_ret:>7}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_positions(raw: list[str]) -> list[str]:
    positions = [p.upper() for p in raw]
    unknown = [p for p in positions if p not in DEFAULT_POSITIONS]
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
        help="Positions to run (default: QB RB WR TE)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 42)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants (with,without) or 'all' (default: both)",
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers. Use an integer or 'auto' "
            "(default: local many-core CUDA -> 6, otherwise 1)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-job logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="skip writing results to benchmark_history/ablations/",
    )
    args = parser.parse_args(argv)

    try:
        positions = _parse_positions(args.positions)
        seeds = parse_seed_list(args.seeds)
        variants = select_variants(args.variants, VARIANTS, DEFAULT_VARIANTS)
    except ValueError as exc:
        parser.error(str(exc))

    if VARIANT_WITH not in variants:
        parser.error(
            f"'{VARIANT_WITH}' (baseline) must be included so paired deltas can be computed"
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

    print(f"\nRunning injury-feature ablation jobs ({len(jobs)} total)...")
    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    # Print per-position decision tables.
    for pos in positions:
        _print_position_table(pos, results, seeds)

    # Print cross-position summary.
    _print_cross_summary(results, positions, seeds)

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
            },
        )

    errors = [r for r in results if r.error]
    if errors:
        for r in errors:
            print(
                f"ERROR {r.position} seed={r.seed} variant={r.variant}: {r.error}",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
