"""Three-way ablation: RB TD head — Huber+gate vs Poisson(no gate) vs Poisson+gate.

Runs the RB pipeline three times with deep-copied config overrides, prints a
side-by-side table of fantasy-point MAE + per-head TD MAE + gate AUC, and
writes the run metadata as a standalone JSON file under
``benchmark_history/ablations/``.

Decision rule (from the PR 2 plan): keep the gate on TDs only if variant A or
C beats B by >= 0.05 pts/game on fantasy-point MAE. Otherwise land variant B
permanently (which is what PR 2's RB config already does).

Usage:
    python scripts/ablate_rb_gate.py           # full three-way
    python scripts/ablate_rb_gate.py --seed 7  # override seed
    python scripts/ablate_rb_gate.py --only B  # run one variant only
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rb.run_pipeline import RB_CONFIG, run_rb_pipeline  # noqa: E402
from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso  # noqa: E402

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
    as the live RB_CONFIG's gated_targets list evolves."""
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


VARIANTS = {
    "A": ("Huber + gate on TDs (pre-PR-2 baseline)", _apply_variant_a),
    "B": ("Poisson NLL, no TD gate (PR #96 config)", _apply_variant_b),
    "C": ("Poisson NLL + gate on TDs (current shipping)", _apply_variant_c),
}


def run_variant(variant: str, seed: int) -> dict:
    label, fn = VARIANTS[variant]
    cfg = fn(RB_CONFIG)
    print(f"\n{'=' * 72}")
    print(f"Variant {variant}: {label}")
    print(f"{'=' * 72}")
    result = run_rb_pipeline(seed=seed, config=cfg)

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
        "receptions_mae": attn["receptions"]["mae"],
        # Gate diagnostics on the gated targets for this variant.
        "gate_aucs": {
            t: attn[t].get("gate_auc")
            for t in attn
            if isinstance(attn.get(t), dict) and "gate_auc" in attn[t]
        },
    }
    return summary


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print("RB TD-gate ablation — summary")
    print(f"{'=' * 72}")
    print(
        f"{'Var':<4}{'FP MAE':>10}{'FP RMSE':>10}{'Rush TD MAE':>14}"
        f"{'Rec TD MAE':>14}{'Rec MAE':>10}"
    )
    print("-" * 62)
    for r in rows:
        print(
            f"{r['variant']:<4}{r['fp_mae']:>10.3f}{r['fp_rmse']:>10.3f}"
            f"{r['rushing_tds_mae']:>14.3f}{r['receiving_tds_mae']:>14.3f}"
            f"{r['receptions_mae']:>10.3f}"
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

    # Decision rule.
    by_var = {r["variant"]: r for r in rows}
    if {"A", "B", "C"} <= set(by_var):
        a, b, c = by_var["A"]["fp_mae"], by_var["B"]["fp_mae"], by_var["C"]["fp_mae"]
        margin_a = b - a
        margin_c = b - c
        print(f"\nFP-MAE margin vs B (positive = gate helps): A={margin_a:+.3f}, C={margin_c:+.3f}")
        if max(margin_a, margin_c) >= 0.05:
            print("Decision: keep a gate on TDs — exceeds 0.05 pt/game threshold.")
        else:
            print("Decision: drop gate on TDs (variant B wins) — below 0.05 pt/game threshold.")


def _write_ablation(rows: list[dict]) -> None:
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{ABLATION_NAME}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": "ablation",
        "name": ABLATION_NAME,
        "variants": rows,
    }
    append_to_history(os.path.join(HISTORY_DIR, "ablations"), entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=sorted(VARIANTS),
        help="Run a single variant (default: run all three)",
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
