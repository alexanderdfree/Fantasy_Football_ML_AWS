"""Multi-variant ablation: RB sparse-count head losses (TDs + fumbles_lost).

Runs the RB pipeline with deep-copied config overrides, prints a side-by-side
table of FP MAE + per-head MAE for rushing_tds / receiving_tds / fumbles_lost,
and writes the run metadata as a standalone JSON file under
``benchmark_history/ablations/``.

Variants:
    A   — Huber + gate on TDs (pre-PR-2 baseline, no fumble gate)
    B   — Poisson NLL, no TD gate (PR #96 config)
    C   — Poisson NLL + gate on TDs (current shipping)
    D   — hurdle_poisson on both TDs (gate + ZTP positives-only value loss)
    E   — D + hurdle_poisson on fumbles_lost (gate + ZTP on all three counts)
    Bf  — current C config + BCE gate on fumbles_lost (Poisson NLL kept)

Original decision rule (variants A/B/C): keep gate on TDs only if A or C beats
B by >= 0.05 pt/game on fantasy-point MAE.

New decision rule (variants D/E/Bf): pick lowest sum of per-target MAEs across
{rushing_tds, receiving_tds, fumbles_lost}. FP MAE reported but not gating.

Usage:
    python -m src.tuning.ablate_rb_gate               # all variants
    python -m src.tuning.ablate_rb_gate --seeds 7     # override seed
    python -m src.tuning.ablate_rb_gate --variants C  # run one variant only
    python -m src.tuning.ablate_rb_gate --dry-run     # show plan without running
    python -m src.tuning.ablate_rb_gate --max-workers 2  # parallel workers
"""

from __future__ import annotations

import argparse
import copy
import sys
from typing import Any

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

ABLATION_NAME = "rb_td_gate"
DEFAULT_POSITIONS = ("RB",)
DEFAULT_SEEDS = (42,)


def _apply_variant_a(cfg: dict) -> dict:
    """Variant A: pre-PR-2 baseline — Huber + gate on TDs, receptions not gated."""
    cfg = copy.deepcopy(cfg)
    cfg["head_losses"] = {
        "rushing_tds": "huber",
        "receiving_tds": "huber",
        "rushing_yards": "huber",
        "receiving_yards": "huber",
        "receptions": "huber",
        "fumbles_lost": "huber",
    }
    cfg["gated_targets"] = ["rushing_tds", "receiving_tds"]
    cfg["loss_weights"] = {
        "rushing_tds": 4.0,
        "receiving_tds": 4.0,
        "rushing_yards": 0.133,
        "receiving_yards": 0.133,
        "receptions": 1.0,
        "fumbles_lost": 4.0,
    }
    cfg["huber_deltas"] = {
        "rushing_tds": 0.5,
        "receiving_tds": 0.5,
        "rushing_yards": 15.0,
        "receiving_yards": 15.0,
        "receptions": 2.0,
        "fumbles_lost": 0.5,
    }
    cfg["nn_head_hidden_overrides"] = {"rushing_tds": 64, "receiving_tds": 64}
    return cfg


def _apply_variant_b(cfg: dict) -> dict:
    """Variant B: Poisson NLL on TDs with no gate on TDs — the PR #96 shipping
    config before the TD-gate-restoration PR. Explicitly forces
    ``gated_targets = ["receptions"]`` so this variant stays meaningful even
    as the live CONFIG's gated_targets list evolves."""
    cfg = copy.deepcopy(cfg)
    cfg["gated_targets"] = ["receptions"]
    return cfg


def _apply_variant_c(cfg: dict) -> dict:
    """Variant C: Poisson NLL on TDs + BCE gate on each TD head on top of
    the reception hurdle. Matches the current shipping RB config after the
    TD-gate-restoration PR. ``head_losses`` stays as PR #96 shipped — TDs
    on ``poisson_nll``; the BCE gate loss is added in addition via
    ``gated_targets``."""
    cfg = copy.deepcopy(cfg)
    cfg["gated_targets"] = ["receptions", "rushing_tds", "receiving_tds"]
    return cfg


def _apply_variant_d(cfg: dict) -> dict:
    """Variant D: hurdle_poisson on both TDs. Architectural mirror of Ridge's
    gated_ordinal — Stage-1 BCE gate + Stage-2 zero-truncated-Poisson value
    loss trained on positives only. Empirical RB TD dispersion ≈ 1.06–1.16,
    so ZTP is the right family (NegBin would burn capacity on alpha≈0)."""
    cfg = copy.deepcopy(cfg)
    cfg["head_losses"] = {
        **cfg["head_losses"],
        "rushing_tds": "hurdle_poisson",
        "receiving_tds": "hurdle_poisson",
    }
    cfg["gated_targets"] = ["receptions", "rushing_tds", "receiving_tds"]
    return cfg


def _apply_variant_e(cfg: dict) -> dict:
    """Variant E: D + hurdle_poisson on fumbles_lost. Empirical dispersion
    ≈ 1.006 (essentially pure Poisson) but the zero rate is 0.948 — the
    positives-only value loss should help the same way it should for TDs."""
    cfg = _apply_variant_d(cfg)
    cfg["head_losses"]["fumbles_lost"] = "hurdle_poisson"
    cfg["gated_targets"] = [
        "receptions",
        "rushing_tds",
        "receiving_tds",
        "fumbles_lost",
    ]
    return cfg


def _apply_variant_bf(cfg: dict) -> dict:
    """Variant Bf: current shipping (C) + BCE gate on fumbles_lost, keeping
    Poisson NLL on the value head. Isolates the gate-only contribution from
    the positives-only-loss contribution that E provides."""
    cfg = copy.deepcopy(cfg)
    cfg["gated_targets"] = [
        "receptions",
        "rushing_tds",
        "receiving_tds",
        "fumbles_lost",
    ]
    return cfg


VARIANTS: dict[str, tuple[str, Any]] = {
    "A": ("Huber + gate on TDs (pre-PR-2 baseline)", _apply_variant_a),
    "B": ("Poisson NLL, no TD gate (PR #96 config)", _apply_variant_b),
    "C": ("Poisson NLL + gate on TDs (current shipping)", _apply_variant_c),
    "D": ("hurdle_poisson on both TDs", _apply_variant_d),
    "E": ("hurdle_poisson on TDs + fumbles_lost", _apply_variant_e),
    "Bf": ("C + BCE gate on fumbles_lost (Poisson kept)", _apply_variant_bf),
}

# Ordered list used as the default run order and for sorted display.
_VARIANT_ORDER = ["A", "B", "C", "D", "E", "Bf"]


def _make_cfg(base_cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    """Deep-copy base_cfg and apply the named variant mutator."""
    _label, fn = VARIANTS[variant]
    return fn(base_cfg)


def _execute_rb_gate_job(job: AblationJob) -> dict[str, Any]:
    """Run one seed × variant job and return the metrics/timings/metadata payload."""
    cfg = _make_cfg(job.base_cfg, job.variant)
    run_fn = get_runner(job.position)
    result = run_fn(seed=job.seed, config=cfg)

    attn = result.get("attn_nn_metrics") or result.get("metrics", {}).get("attn_nn")
    if attn is None:
        raise RuntimeError(
            f"Variant {job.variant}: could not find attn_nn_metrics in result keys "
            f"{sorted(result.keys())}"
        )

    label, _fn = VARIANTS[job.variant]
    count_mae_sum = (
        attn["rushing_tds"]["mae"] + attn["receiving_tds"]["mae"] + attn["fumbles_lost"]["mae"]
    )
    gate_aucs = {
        t: attn[t].get("gate_auc")
        for t in attn
        if isinstance(attn.get(t), dict) and "gate_auc" in attn[t]
    }

    metrics = {
        "fp_mae": float(attn["total"]["mae"]),
        "fp_rmse": float(attn["total"]["rmse"]),
        "rushing_tds_mae": float(attn["rushing_tds"]["mae"]),
        "receiving_tds_mae": float(attn["receiving_tds"]["mae"]),
        "fumbles_lost_mae": float(attn["fumbles_lost"]["mae"]),
        "receptions_mae": float(attn["receptions"]["mae"]),
        "rushing_yards_mae": float(attn["rushing_yards"]["mae"]),
        "receiving_yards_mae": float(attn["receiving_yards"]["mae"]),
        "count_target_mae_sum": float(count_mae_sum),
        "gate_aucs": gate_aucs,
    }
    timings: dict[str, Any] = {}
    phase_seconds = result.get("phase_seconds") or {}
    if phase_seconds.get("attn_nn_train") is not None:
        timings["attn_nn_train_sec"] = float(phase_seconds["attn_nn_train"])

    metadata = dict(job.metadata)
    metadata["variant_label"] = label

    return {"metrics": metrics, "timings": timings, "metadata": metadata}


def _results_to_rows(results: list[AblationResult]) -> list[dict[str, Any]]:
    """Convert AblationResults into the legacy per-variant summary row dicts.

    When multiple seeds are present the numeric metrics are averaged; the
    ``gate_aucs`` from the first non-error result for each variant are used
    (they are qualitative, not averaged).
    """
    from collections import defaultdict

    by_variant: dict[str, list[AblationResult]] = defaultdict(list)
    for result in results:
        if result.error is None:
            by_variant[result.variant].append(result)

    _SCALAR_KEYS = (
        "fp_mae",
        "fp_rmse",
        "rushing_tds_mae",
        "receiving_tds_mae",
        "fumbles_lost_mae",
        "receptions_mae",
        "rushing_yards_mae",
        "receiving_yards_mae",
        "count_target_mae_sum",
    )

    rows = []
    for variant in _VARIANT_ORDER:
        variant_results = by_variant.get(variant)
        if not variant_results:
            continue
        label = variant_results[0].metadata.get("variant_label", VARIANTS[variant][0])
        seed = variant_results[0].seed if len(variant_results) == 1 else None

        row: dict[str, Any] = {"variant": variant, "label": label, "seed": seed}
        for key in _SCALAR_KEYS:
            vals = [float(r.metrics[key]) for r in variant_results if key in r.metrics]
            row[key] = float(sum(vals) / len(vals)) if vals else float("nan")
            if len(variant_results) > 1:
                row[f"{key}_stats"] = mean_std(vals)

        # gate_aucs: use the first result (qualitative, not averaged)
        row["gate_aucs"] = variant_results[0].metrics.get("gate_aucs", {})
        rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    """Print the side-by-side table and apply both decision rules.

    Accepts either the old list-of-dicts shape (single-seed) or the output of
    ``_results_to_rows`` (multi-seed mean values).
    """
    multi_seed = any(r.get("fp_mae_stats") for r in rows)

    print(f"\n{'=' * 96}")
    print("RB sparse-count head ablation — summary")
    print(f"{'=' * 96}")

    if multi_seed:
        print(
            f"{'Var':<4}{'FP MAE (mean±std)':>22}{'Rush TD':>14}{'Rec TD':>14}"
            f"{'Fum lost':>14}{'CntSum':>14}{'Rec':>10}{'RushYd':>12}{'RecYd':>12}"
        )
        print("-" * 116)

        def _ms(row: dict, key: str) -> str:
            stats = row.get(f"{key}_stats")
            if stats and stats.get("mean") is not None:
                return fmt_mean_std([stats["mean"]])
            return f"{row[key]:.3f}"

        for r in rows:
            fp_str = fmt_mean_std(
                [v["mean"] for v in [r.get("fp_mae_stats", {})] if v.get("mean") is not None]
                if r.get("fp_mae_stats")
                else [r["fp_mae"]]
            )
            print(
                f"{r['variant']:<4}{fp_str:>22}{_ms(r, 'rushing_tds_mae'):>14}"
                f"{_ms(r, 'receiving_tds_mae'):>14}{_ms(r, 'fumbles_lost_mae'):>14}"
                f"{_ms(r, 'count_target_mae_sum'):>14}{_ms(r, 'receptions_mae'):>10}"
                f"{_ms(r, 'rushing_yards_mae'):>12}{_ms(r, 'receiving_yards_mae'):>12}"
            )
    else:
        print(
            f"{'Var':<4}{'FP MAE':>9}{'Rush TD':>10}{'Rec TD':>10}{'Fum lost':>10}"
            f"{'CntSum':>9}{'Rec':>8}{'RushYd':>9}{'RecYd':>9}"
        )
        print("-" * 80)
        for r in rows:
            print(
                f"{r['variant']:<4}{r['fp_mae']:>9.3f}{r['rushing_tds_mae']:>10.3f}"
                f"{r['receiving_tds_mae']:>10.3f}{r['fumbles_lost_mae']:>10.3f}"
                f"{r['count_target_mae_sum']:>9.3f}{r['receptions_mae']:>8.3f}"
                f"{r['rushing_yards_mae']:>9.3f}{r['receiving_yards_mae']:>9.3f}"
            )

    if any(r["gate_aucs"] for r in rows):
        print("\nGate AUCs (attention NN only, gated targets only):")
        for r in rows:
            if r["gate_aucs"]:
                auc_str = ", ".join(
                    f"{t}={auc:.3f}" if auc is not None else f"{t}=n/a"
                    for t, auc in r["gate_aucs"].items()
                )
                print(f"  {r['variant']}: {auc_str}")

    # Original A/B/C decision rule (preserved for back-compat).
    by_var = {r["variant"]: r for r in rows}
    if {"A", "B", "C"} <= set(by_var):
        a, b, c = by_var["A"]["fp_mae"], by_var["B"]["fp_mae"], by_var["C"]["fp_mae"]
        margin_a = b - a
        margin_c = b - c
        print(f"\nFP-MAE margin vs B (positive = gate helps): A={margin_a:+.3f}, C={margin_c:+.3f}")
        if max(margin_a, margin_c) >= 0.05:
            print("Original decision (A/B/C, FP MAE): keep gate on TDs.")
        else:
            print("Original decision (A/B/C, FP MAE): drop gate on TDs.")

    # New decision rule: lowest sum of per-target MAEs on count heads.
    if len(rows) > 1:
        ranked = sorted(rows, key=lambda r: r["count_target_mae_sum"])
        winner = ranked[0]
        print(
            f"\nCount-target decision: variant {winner['variant']} wins on "
            f"sum(rushing_tds + receiving_tds + fumbles_lost) MAE = "
            f"{winner['count_target_mae_sum']:.4f} ({winner['label']})."
        )
        print("Ranking by count_target_mae_sum (lowest = best):")
        for r in ranked:
            print(
                f"  {r['variant']:<4} sum={r['count_target_mae_sum']:.4f}  fp_mae={r['fp_mae']:.3f}"
            )


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
            for variant in variants:
                label, _fn = VARIANTS[variant]
                jobs.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant,
                        label=label,
                        run_fn=_execute_rb_gate_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "experiment"},
                    )
                )
    return jobs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(DEFAULT_POSITIONS),
        help="Positions to run (default: RB)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: 42)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants to run, or 'all' (default: all six A,B,C,D,E,Bf)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing results to benchmark_history/ablations/",
    )
    parser.add_argument(
        "--max-workers",
        default="auto",
        help=(
            "Process workers. Use an integer or 'auto' "
            "(default: local many-core CUDA -> 6, otherwise 1; pass 1 for clean timing)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for per-job logs (default: no per-job logs)",
    )
    # Legacy single-seed shorthand kept for back-compat with old invocations.
    parser.add_argument("--seed", type=int, default=None, help="Single seed override (use --seeds)")
    parser.add_argument(
        "--only",
        choices=sorted(VARIANTS),
        default=None,
        help="Run a single variant only (shorthand for --variants X)",
    )
    args = parser.parse_args(argv)

    # Resolve seed(s): --seed overrides --seeds for single-seed back-compat.
    seeds_str = str(args.seed) if args.seed is not None else args.seeds
    try:
        seeds = parse_seed_list(seeds_str)
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve variants: --only takes precedence over --variants.
    variants_raw = args.only if args.only is not None else args.variants
    try:
        variants = select_variants(variants_raw, VARIANTS, _VARIANT_ORDER)
    except ValueError as exc:
        parser.error(str(exc))

    positions = [pos.upper() for pos in args.positions]

    jobs = _build_jobs(positions=positions, seeds=seeds, variants=variants)
    try:
        max_workers = resolve_max_workers(args.max_workers, job_count=len(jobs))
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(format_dry_run_table(jobs))
        print(f"\nExperiment workers: {max_workers} ({args.max_workers})")
        if args.log_dir:
            print(f"Job logs: {args.log_dir}")
        return

    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    rows = _results_to_rows(results)
    print_summary(rows)

    if not args.no_history:
        write_history(
            ABLATION_NAME,
            results,
            metadata={
                "positions": positions,
                "seeds": seeds,
                "variants": variants,
                "max_workers": max_workers,
            },
        )

    errors = [r for r in results if r.error]
    if errors:
        for r in errors:
            print(f"ERROR {r.position} seed={r.seed} variant={r.variant}: {r.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
