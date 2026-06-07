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
    python -m src.tuning.ablate_rb_gate           # all variants
    python -m src.tuning.ablate_rb_gate --seed 7  # override seed
    python -m src.tuning.ablate_rb_gate --only D  # run one variant only
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rb.run_pipeline import CONFIG, run  # noqa: E402
from src.tuning.history import append_ablation_run

ABLATION_NAME = "rb_td_gate"
HISTORY_DIR = "benchmark_history"


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


VARIANTS = {
    "A": ("Huber + gate on TDs (pre-PR-2 baseline)", _apply_variant_a),
    "B": ("Poisson NLL, no TD gate (PR #96 config)", _apply_variant_b),
    "C": ("Poisson NLL + gate on TDs (current shipping)", _apply_variant_c),
    "D": ("hurdle_poisson on both TDs", _apply_variant_d),
    "E": ("hurdle_poisson on TDs + fumbles_lost", _apply_variant_e),
    "Bf": ("C + BCE gate on fumbles_lost (Poisson kept)", _apply_variant_bf),
}


def run_variant(variant: str, seed: int) -> dict:
    label, fn = VARIANTS[variant]
    cfg = fn(CONFIG)
    print(f"\n{'=' * 72}")
    print(f"Variant {variant}: {label}")
    print(f"{'=' * 72}")
    result = run(seed=seed, config=cfg)

    # compute_target_metrics stores per-model metrics on ``result["model_metrics"]``
    # or similar — read the attention-NN entry since that's the gated path.
    # Fall back to the result's summary if the schema is different.
    attn = result.get("attn_nn_metrics") or result.get("metrics", {}).get("attn_nn")
    if attn is None:
        raise RuntimeError(
            f"Variant {variant}: could not find attn_nn_metrics in result keys "
            f"{sorted(result.keys())}"
        )

    summary = {
        "variant": variant,
        "label": label,
        "seed": seed,
        "fp_mae": attn["total"]["mae"],
        "fp_rmse": attn["total"]["rmse"],
        "rushing_tds_mae": attn["rushing_tds"]["mae"],
        "receiving_tds_mae": attn["receiving_tds"]["mae"],
        "fumbles_lost_mae": attn["fumbles_lost"]["mae"],
        "receptions_mae": attn["receptions"]["mae"],
        "rushing_yards_mae": attn["rushing_yards"]["mae"],
        "receiving_yards_mae": attn["receiving_yards"]["mae"],
        "count_target_mae_sum": (
            attn["rushing_tds"]["mae"] + attn["receiving_tds"]["mae"] + attn["fumbles_lost"]["mae"]
        ),
        # Gate diagnostics on the gated targets for this variant.
        "gate_aucs": {
            t: attn[t].get("gate_auc")
            for t in attn
            if isinstance(attn.get(t), dict) and "gate_auc" in attn[t]
        },
    }
    return summary


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 96}")
    print("RB sparse-count head ablation — summary")
    print(f"{'=' * 96}")
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


def _write_ablation(rows: list[dict]) -> None:
    append_ablation_run(ABLATION_NAME, {"variants": rows}, history_dir=HISTORY_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=sorted(VARIANTS),
        help="Run a single variant (default: run all variants)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip writing results to benchmark_history/ablations/",
    )
    args = parser.parse_args()

    variants = [args.only] if args.only else sorted(VARIANTS)
    rows = [run_variant(v, args.seed) for v in variants]
    print_summary(rows)
    if not args.no_history:
        _write_ablation(rows)


if __name__ == "__main__":
    main()
