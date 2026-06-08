"""Ablation: BatchNorm vs LayerNorm in the NN static-feature backbone.

The project uses ``BatchNorm1d`` in exactly one place — the shared static-feature
backbone MLP (``src/shared/neural_net.py::_build_backbone``: ``Linear -> BN ->
ReLU -> Dropout``) reused by every ``MultiHeadNet*`` variant — and ``LayerNorm``
throughout the attention-pooling path (per-game encoder, per-target history
norms; plus the optional, off-by-default Pre-LN ``SelfAttentionBlock``). The
BatchNorm choice was inherited from the first RB prototype (``113f780``),
consolidated in #93 (``9ead4f9``), and never A/B'd.

This script settles it empirically. It runs the real position pipeline twice
under an identical seed and identical splits — once with the stock BatchNorm
backbone, once with a LayerNorm backbone injected by monkeypatching
``_build_backbone`` at runtime — and prints a side-by-side decision table.

The norm layer is the ONLY thing that differs between variants:
  * Ridge/ElasticNet are backbone-free, so their FP MAE MUST be identical across
    variants for a given seed. That equality is asserted as a *data-identity
    sentinel* (a mismatch means the two runs did not see the same data/seed, so
    the deltas are meaningless).
  * LightGBM is backbone-free too, so it is disabled (``train_lightgbm=False``)
    to save wall-clock without affecting the measured quantity (the NN deltas).
  * NN configs are left at PRODUCTION values — no epoch/data shrink. A reduced
    proxy can flip the sign of a small effect (see project memory).

Both the base MLP NN (``nn_metrics``) and the production attention NN
(``attn_nn_metrics``) go through ``_build_backbone``, so deltas are reported for
both; the attention NN is the headline.

Usage:
    python -m src.tuning.ablate_backbone_norm                         # WR, seed 42, both
    python -m src.tuning.ablate_backbone_norm --seeds 42,7,123        # multi-seed
    python -m src.tuning.ablate_backbone_norm --variants bn           # one variant
    python -m src.tuning.ablate_backbone_norm --positions QB          # different position
    python -m src.tuning.ablate_backbone_norm --dry-run               # print plan, no training
    python -m src.tuning.ablate_backbone_norm --max-workers 2         # parallel workers
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from typing import Any

import torch.nn as nn

import src.shared.neural_net as nn_mod
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

ABLATION_NAME = "backbone_norm"
DEFAULT_LOG_DIR = "logs/ablations/backbone_norm"

# A mean Δ FP MAE smaller than this (or within seed noise) counts as "no
# meaningful difference" — ~0.5% of a typical ~4 pt/game baseline.
FLAT_NOISE_THRESHOLD = 0.02

VARIANTS: dict[str, str] = {
    "bn": "BatchNorm1d backbone (stock / production)",
    "ln": "LayerNorm backbone (monkeypatched)",
}
DEFAULT_VARIANTS = ("bn", "ln")
DEFAULT_POSITION = "WR"


def _build_backbone_layernorm(input_dim, hidden_dims, dropout):
    """Drop-in LayerNorm replacement for the stock BatchNorm1d backbone.

    Mirrors ``src.shared.neural_net._build_backbone`` exactly except for the norm
    layer. ``LayerNorm(hidden_dim)`` normalizes over the feature dim per sample —
    batch-size independent, identical train/eval (no running stats) — unlike
    ``BatchNorm1d``, which normalizes each feature across the batch and applies
    frozen running stats at eval/serving.
    """
    blocks: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        blocks.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim
    return nn.Sequential(*blocks)


def _make_cfg(base_cfg: dict) -> dict:
    """Deep-copy the production config and disable LightGBM.

    LightGBM has no backbone and is invariant to the norm swap, so dropping it
    saves ~2-5 min/run without touching the measured quantity. Everything else
    stays at production values.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["train_lightgbm"] = False
    return cfg


def _execute_backbone_norm_job(job: AblationJob) -> dict[str, Any]:
    """Run one backbone-norm job; apply the ln monkeypatch inside the worker.

    This function is called inside a ``ProcessPoolExecutor`` spawn worker for
    the ``ln`` variant, so the monkeypatch must live here — module-level patches
    in the orchestrator process are not inherited by spawn workers. The
    try/finally guarantees the patch is reverted even if the run raises, keeping
    each worker's module state clean for any future job it might run.
    """
    cfg = _make_cfg(job.base_cfg)
    run_fn = get_runner(job.position)
    targets = cfg.get("targets", [])

    if job.variant == "ln":
        original = nn_mod._build_backbone
        nn_mod._build_backbone = _build_backbone_layernorm
        try:
            result = run_fn(seed=job.seed, config=cfg)
        finally:
            nn_mod._build_backbone = original
    else:
        result = run_fn(seed=job.seed, config=cfg)

    return _extract_run_payload(result, targets=targets, variant=job.variant, seed=job.seed)


def _extract_run_payload(
    result: dict[str, Any],
    *,
    targets: list[str],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    attn = result.get("attn_nn_metrics")
    base = result.get("nn_metrics")
    ridge = result.get("ridge_metrics")
    if attn is None or base is None or ridge is None:
        raise RuntimeError(
            f"variant {variant!r}: missing metrics (attn/base/ridge) in result keys "
            f"{sorted(result.keys())}"
        )
    metrics: dict[str, Any] = {
        "attn_fp_mae": float(attn["total"]["mae"]),
        "base_fp_mae": float(base["total"]["mae"]),
        "ridge_fp_mae": float(ridge["total"]["mae"]),
        "attn_targets": {t: float(attn[t]["mae"]) for t in targets if t in attn},
        "base_targets": {t: float(base[t]["mae"]) for t in targets if t in base},
    }
    return {
        "metrics": metrics,
        "timings": {},
        "metadata": {"variant": variant, "seed": seed},
    }


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
                jobs.append(
                    AblationJob(
                        position=position,
                        seed=seed,
                        variant=variant_key,
                        label=VARIANTS[variant_key],
                        run_fn=_execute_backbone_norm_job,
                        base_cfg=base_cfg,
                        metadata={"run_kind": "experiment"},
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# Summary / decision output — reproduces the original tables from AblationResult
# ---------------------------------------------------------------------------


def _by_key(results: list[AblationResult]) -> dict[tuple[str, int], AblationResult]:
    """Index successful results by (variant, seed)."""
    return {(r.variant, r.seed): r for r in results if r.error is None}


def print_summary(results: list[AblationResult], targets: list[str]) -> bool:
    """Print the decision table. Returns the data-identity sentinel result (True = OK).

    Accepts either a list of ``AblationResult`` objects (new parallel path) or a
    list of plain dicts (legacy path kept for test compatibility); the dict path
    converts each dict into a minimal ``AblationResult`` in-place.
    """
    # Normalise: support the old plain-dict format used by some tests.
    normalised: list[AblationResult] = []
    for item in results:
        if isinstance(item, dict):
            normalised.append(
                AblationResult(
                    position="",
                    seed=item["seed"],
                    variant=item["variant"],
                    metrics={
                        "attn_fp_mae": item["attn_fp_mae"],
                        "base_fp_mae": item["base_fp_mae"],
                        "ridge_fp_mae": item["ridge_fp_mae"],
                        "attn_targets": item.get("attn_targets", {}),
                        "base_targets": item.get("base_targets", {}),
                    },
                    timings={},
                    metadata={},
                )
            )
        else:
            normalised.append(item)

    by = _by_key(normalised)
    seeds = sorted({r.seed for r in normalised if r.error is None})
    have_both = all(("bn", s) in by and ("ln", s) in by for s in seeds)

    print(f"\n{'=' * 88}")
    print("Backbone-norm ablation — BatchNorm vs LayerNorm in the static backbone")
    print(f"{'=' * 88}")

    # --- Per-seed headline FP MAE (attention NN + base NN) ---
    print(f"\n{'seed':>6}{'model':>9}{'BN MAE':>11}{'LN MAE':>11}{'Δ(LN-BN)':>12}")
    print("-" * 49)
    for s in seeds:
        bn_r, ln_r = by.get(("bn", s)), by.get(("ln", s))
        if not (bn_r and ln_r):
            continue
        for model, key in (("attn", "attn_fp_mae"), ("base", "base_fp_mae")):
            bn_val = float(bn_r.metrics[key])
            ln_val = float(ln_r.metrics[key])
            print(f"{s:>6}{model:>9}{bn_val:>11.4f}{ln_val:>11.4f}{ln_val - bn_val:>+12.4f}")

    # --- Data-identity sentinel: Ridge is backbone-free → must match per seed ---
    print("\nData-identity sentinel (Ridge FP MAE — must be identical across variants):")
    sentinel_ok = True
    for s in seeds:
        bn_r, ln_r = by.get(("bn", s)), by.get(("ln", s))
        if not (bn_r and ln_r):
            sentinel_ok = False
            continue
        bn_ridge = float(bn_r.metrics["ridge_fp_mae"])
        ln_ridge = float(ln_r.metrics["ridge_fp_mae"])
        diff = abs(ln_ridge - bn_ridge)
        ok = diff < 1e-9
        sentinel_ok = sentinel_ok and ok
        flag = "OK" if ok else "*** MISMATCH — variants saw different data/seed ***"
        print(f"  seed {s}: bn={bn_ridge:.6f} ln={ln_ridge:.6f} (Δ={diff:.2e}) {flag}")

    # --- Aggregate across seeds ---
    print("\nAggregate across seeds (mean ± sample-std):")
    for model, key in (("attn NN", "attn_fp_mae"), ("base NN", "base_fp_mae")):
        bn_vals = [float(by[("bn", s)].metrics[key]) for s in seeds if ("bn", s) in by]
        ln_vals = [float(by[("ln", s)].metrics[key]) for s in seeds if ("ln", s) in by]
        deltas = [
            float(by[("ln", s)].metrics[key]) - float(by[("bn", s)].metrics[key])
            for s in seeds
            if ("ln", s) in by and ("bn", s) in by
        ]
        print(
            f"  {model:<8} BN={fmt_mean_std(bn_vals)}  LN={fmt_mean_std(ln_vals)}  "
            f"Δ(LN-BN)={fmt_mean_std(deltas)}"
        )

    # --- Per-target attention-NN MAE (mean across seeds) ---
    print("\nPer-target attention-NN MAE (mean across seeds):")
    print(f"  {'target':<18}{'BN':>11}{'LN':>11}{'Δ(LN-BN)':>12}")
    for t in targets:
        bn_vals = [
            float(by[("bn", s)].metrics["attn_targets"][t])
            for s in seeds
            if ("bn", s) in by and t in by[("bn", s)].metrics.get("attn_targets", {})
        ]
        ln_vals = [
            float(by[("ln", s)].metrics["attn_targets"][t])
            for s in seeds
            if ("ln", s) in by and t in by[("ln", s)].metrics.get("attn_targets", {})
        ]
        if not bn_vals or not ln_vals:
            continue
        bnm, lnm = statistics.mean(bn_vals), statistics.mean(ln_vals)
        print(f"  {t:<18}{bnm:>11.4f}{lnm:>11.4f}{lnm - bnm:>+12.4f}")

    _verdict(seeds, by, sentinel_ok, have_both)
    return sentinel_ok


def _verdict(
    seeds: list[int],
    by: dict[tuple[str, int], AblationResult],
    sentinel_ok: bool,
    have_both: bool,
) -> None:
    print(f"\n{'-' * 88}")
    if not seeds or not have_both:
        print("VERDICT: incomplete — need both variants (bn and ln) for the same seeds.")
        return
    if not sentinel_ok:
        print(
            "VERDICT: SENTINEL FAILED — Ridge MAE differs across variants, so this is not a "
            "clean single-variable comparison. Fix data/seed handling before trusting the NN deltas."
        )
        return

    deltas = [
        float(by[("ln", s)].metrics["attn_fp_mae"]) - float(by[("bn", s)].metrics["attn_fp_mae"])
        for s in seeds
    ]
    mean_d = statistics.mean(deltas)
    winner = "LayerNorm" if mean_d < 0 else "BatchNorm"
    if len(deltas) < 2:
        print(
            f"VERDICT (attention NN, single seed {seeds[0]}): Δ(LN-BN) FP MAE = {mean_d:+.4f} "
            f"→ {winner} ahead by {abs(mean_d):.4f}."
        )
        print(
            "  Directional only — re-run with ≥3 seeds to separate signal from seed noise "
            "before concluding."
        )
        return

    sd = statistics.stdev(deltas)
    print(
        f"VERDICT (attention NN, headline): mean Δ(LN-BN) FP MAE = {mean_d:+.4f} ± {sd:.4f} "
        f"over {len(deltas)} seeds."
    )
    if abs(mean_d) <= sd or abs(mean_d) < FLAT_NOISE_THRESHOLD:
        print(
            "  FLAT: the gap is within seed noise. BatchNorm (production) is fine for the tabular "
            "backbone; no change warranted — document the rationale + this result."
        )
    elif winner == "LayerNorm":
        print(
            f"  LayerNorm ahead by {abs(mean_d):.4f} (> noise {sd:.4f}). Worth pursuing — re-run on "
            "a 2nd position, then consider the backbone_norm config-flag ship (accepts a retrain)."
        )
    else:
        print(
            f"  BatchNorm ahead by {abs(mean_d):.4f} (> noise {sd:.4f}). Production choice confirmed "
            "best; keep it and document."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=[DEFAULT_POSITION],
        help=f"Position(s) to ablate (default: {DEFAULT_POSITION})",
    )
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42)")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to run, or 'all' (default: bn,ln)",
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
            "(default: CUDA many-core -> up to 6, otherwise 1)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for per-job logs (default: {DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        seeds = parse_seed_list(args.seeds)
        variants = select_variants(args.variants, VARIANTS, DEFAULT_VARIANTS)
        positions = [p.upper() for p in args.positions]
    except ValueError as exc:
        parser.error(str(exc))

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

    results = run_grid(jobs, max_workers=max_workers, log_dir=args.log_dir, progress=True)

    # Collect targets from the first position's config (for the per-target table).
    targets: list[str] = []
    if positions:
        import contextlib

        with contextlib.suppress(Exception):
            targets = list(get_config(positions[0]).get("targets", []))

    print_summary(results, targets)

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
