"""Ablation: BatchNorm vs LayerNorm in the NN static-feature backbone.

The project uses ``BatchNorm1d`` in exactly one place — the shared static-feature
backbone MLP (``src/shared/neural_net.py::_build_backbone``: ``Linear -> BN ->
ReLU -> Dropout``) reused by every ``MultiHeadNet*`` variant — and ``LayerNorm``
throughout the attention path (per-game encoder, Pre-LN ``SelfAttentionBlock``,
per-target history norms). The BatchNorm choice was inherited from the first RB
prototype (``113f780``), consolidated in #93 (``9ead4f9``), and never A/B'd.

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
    python -m src.tuning.ablate_backbone_norm                   # WR, seed 42, both
    python -m src.tuning.ablate_backbone_norm --seeds 42,7,123  # multi-seed
    python -m src.tuning.ablate_backbone_norm --only bn         # one variant
    python -m src.tuning.ablate_backbone_norm --position QB     # different position
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import statistics
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.shared.neural_net as nn_mod  # noqa: E402
from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso  # noqa: E402

ABLATION_NAME = "backbone_norm"
HISTORY_DIR = "benchmark_history"

VARIANTS = {
    "bn": "BatchNorm1d backbone (stock / production)",
    "ln": "LayerNorm backbone (monkeypatched)",
}


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


def run_variant(variant: str, seed: int, run_fn, base_cfg: dict, targets: list[str]) -> dict:
    cfg = _make_cfg(base_cfg)
    print(f"\n{'=' * 72}")
    print(f"Variant {variant!r} (seed {seed}): {VARIANTS[variant]}")
    print(f"{'=' * 72}")
    if variant == "ln":
        original = nn_mod._build_backbone
        nn_mod._build_backbone = _build_backbone_layernorm
        try:
            result = run_fn(seed=seed, config=cfg)
        finally:
            nn_mod._build_backbone = original  # never leak the patch
    else:
        result = run_fn(seed=seed, config=cfg)
    return _extract(result, variant, seed, targets)


def _extract(result: dict, variant: str, seed: int, targets: list[str]) -> dict:
    attn = result.get("attn_nn_metrics")
    base = result.get("nn_metrics")
    ridge = result.get("ridge_metrics")
    if attn is None or base is None or ridge is None:
        raise RuntimeError(
            f"variant {variant}: missing metrics (attn/base/ridge) in result keys "
            f"{sorted(result.keys())}"
        )
    return {
        "variant": variant,
        "seed": seed,
        "attn_fp_mae": attn["total"]["mae"],
        "base_fp_mae": base["total"]["mae"],
        "ridge_fp_mae": ridge["total"]["mae"],
        "attn_targets": {t: attn[t]["mae"] for t in targets},
        "base_targets": {t: base[t]["mae"] for t in targets},
    }


def _fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f}±{sd:.4f}"


def print_summary(rows: list[dict], targets: list[str]) -> bool:
    """Print the decision table. Returns the data-identity sentinel result."""
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["variant"], r["seed"]): r for r in rows}
    have_both = all(("bn", s) in by and ("ln", s) in by for s in seeds)

    print(f"\n{'=' * 88}")
    print("Backbone-norm ablation — BatchNorm vs LayerNorm in the static backbone")
    print(f"{'=' * 88}")

    # --- Per-seed headline FP MAE (attention NN + base NN) ---
    print(f"\n{'seed':>6}{'model':>9}{'BN MAE':>11}{'LN MAE':>11}{'Δ(LN-BN)':>12}")
    print("-" * 49)
    for s in seeds:
        bn, ln = by.get(("bn", s)), by.get(("ln", s))
        if not (bn and ln):
            continue
        for model, key in (("attn", "attn_fp_mae"), ("base", "base_fp_mae")):
            print(f"{s:>6}{model:>9}{bn[key]:>11.4f}{ln[key]:>11.4f}{ln[key] - bn[key]:>+12.4f}")

    # --- Data-identity sentinel: Ridge is backbone-free → must match per seed ---
    print("\nData-identity sentinel (Ridge FP MAE — must be identical across variants):")
    sentinel_ok = True
    for s in seeds:
        bn, ln = by.get(("bn", s)), by.get(("ln", s))
        if not (bn and ln):
            sentinel_ok = False
            continue
        diff = abs(ln["ridge_fp_mae"] - bn["ridge_fp_mae"])
        ok = diff < 1e-9
        sentinel_ok = sentinel_ok and ok
        flag = "OK" if ok else "*** MISMATCH — variants saw different data/seed ***"
        print(
            f"  seed {s}: bn={bn['ridge_fp_mae']:.6f} ln={ln['ridge_fp_mae']:.6f} "
            f"(Δ={diff:.2e}) {flag}"
        )

    # --- Aggregate across seeds ---
    print("\nAggregate across seeds (mean ± sample-std):")
    for model, key in (("attn NN", "attn_fp_mae"), ("base NN", "base_fp_mae")):
        bn_vals = [by[("bn", s)][key] for s in seeds if ("bn", s) in by]
        ln_vals = [by[("ln", s)][key] for s in seeds if ("ln", s) in by]
        deltas = [
            by[("ln", s)][key] - by[("bn", s)][key]
            for s in seeds
            if ("ln", s) in by and ("bn", s) in by
        ]
        print(
            f"  {model:<8} BN={_fmt_mean_std(bn_vals)}  LN={_fmt_mean_std(ln_vals)}  "
            f"Δ(LN-BN)={_fmt_mean_std(deltas)}"
        )

    # --- Per-target attention-NN MAE (mean across seeds) ---
    print("\nPer-target attention-NN MAE (mean across seeds):")
    print(f"  {'target':<18}{'BN':>11}{'LN':>11}{'Δ(LN-BN)':>12}")
    for t in targets:
        bn_vals = [by[("bn", s)]["attn_targets"][t] for s in seeds if ("bn", s) in by]
        ln_vals = [by[("ln", s)]["attn_targets"][t] for s in seeds if ("ln", s) in by]
        if not bn_vals or not ln_vals:
            continue
        bnm, lnm = statistics.mean(bn_vals), statistics.mean(ln_vals)
        print(f"  {t:<18}{bnm:>11.4f}{lnm:>11.4f}{lnm - bnm:>+12.4f}")

    _verdict(seeds, by, sentinel_ok, have_both)
    return sentinel_ok


def _verdict(seeds: list[int], by: dict, sentinel_ok: bool, have_both: bool) -> None:
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

    deltas = [by[("ln", s)]["attn_fp_mae"] - by[("bn", s)]["attn_fp_mae"] for s in seeds]
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
    if abs(mean_d) <= sd or abs(mean_d) < 0.02:
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


def _write_ablation(rows: list[dict], position: str, seeds: list[int]) -> None:
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{ABLATION_NAME}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": ABLATION_NAME,
        "position": position,
        "seeds": seeds,
        "variants": rows,
    }
    append_to_history(os.path.join(HISTORY_DIR, "ablations"), entry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--position", default="WR", help="Position to ablate (default: WR)")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42)")
    parser.add_argument(
        "--only", choices=sorted(VARIANTS), help="Run a single variant (default: both)"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing results to benchmark_history/ablations/",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [args.only] if args.only else ["bn", "ln"]

    mod = importlib.import_module(f"src.{args.position.lower()}.run_pipeline")
    base_cfg, run_fn = mod.CONFIG, mod.run
    targets = base_cfg["targets"]

    # Run all variants for one seed before advancing — each run re-seeds
    # internally, and the feature cache is seed-independent so engineering
    # happens once and every run trains on identical features.
    rows = [run_variant(v, s, run_fn, base_cfg, targets) for s in seeds for v in variants]

    print_summary(rows, targets)
    if not args.no_history:
        _write_ablation(rows, args.position.upper(), seeds)


if __name__ == "__main__":
    main()
