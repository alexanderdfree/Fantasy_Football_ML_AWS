"""Head-to-head: our model vs. NFL.com expert projections.

Where :mod:`src.analysis.analysis_nflcom_baseline` scores NFL.com *against
actuals* (so you eyeball it next to the model's benchmark numbers), this script
puts the **model and the expert on the same player-weeks** and adjudicates the
gap properly:

  - Restricts both forecasters to the *intersection* of player-weeks where each
    has a projection (same-sample), scoring both against the single ground-truth
    ``fantasy_points`` column — so the errors are genuinely paired.
  - Reports **MAE and RMSE** for each side. We headline MAE (fantasy scores are
    heavy-tailed, so errors are closer to Laplacian than Gaussian) but report
    RMSE too — RMSE *favors* the expert's implicit squared-error objective, so
    winning on it is the stronger claim.
  - Reports **ranking** quality (top-K hit-rate + Spearman) — fantasy is a
    selection problem, and ranking metrics are loss-agnostic, which sidesteps the
    "you trained on Huber, they on MSE" objection entirely.
  - Attaches a **paired significance test**: a player-clustered bootstrap CI on
    ΔMAE / ΔRMSE (primary) plus a Diebold-Mariano test (named companion), so a
    0.1-pt gap over one season isn't mistaken for a real edge.

Loss ≠ metric: the model trains with Huber, the expert is squared-error-oriented,
but the comparison happens entirely on held-out errors under shared metrics, so
the training-loss mismatch is irrelevant here. See the methodology memo for the
Gneiting (2011) / Taggart (2022) basis.

Scope: this compares against NFL.com only (the single expert source in the repo;
there is no consensus/FantasyPros/ESPN feed). DST is skipped (no NFL.com DST
projections). K is totals-only and flagged (NFL.com K is standard-scoring totals,
not PPR-decomposable).

Outputs:
  analysis_output/expert_comparison.json  -- per-position head-to-head + significance
  stdout                                  -- pretty-printed comparison table

Operator usage:
  python -m src.analysis.analysis_expert_comparison                  # default = TEST_SEASONS, PPR, all positions
  python -m src.analysis.analysis_expert_comparison --positions QB RB
  python -m src.analysis.analysis_expert_comparison --scoring-format half_ppr --n-boot 2000

Note: the default model loader runs each position's full training pipeline
(``src.{pos}.run_pipeline.run()``) to obtain held-out predictions, so a full run
trains every requested position (minutes). ``--positions`` subsets the work.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

# Reuse the NFL.com projection helpers + serializer from the baseline script
# (same package). These are the single source of truth for turning NFL.com's raw
# projections into our PPR fantasy-point scale and for which positions are
# skipped / totals-only.
from src.analysis.analysis_nflcom_baseline import (
    _SKIPPED_POSITIONS,
    _TOTALS_ONLY_POSITIONS,
    _json_default,
    _project_nflcom_to_ppr,
)
from src.analysis.significance import (
    diebold_mariano_test,
    paired_bootstrap_metric_ci,
)
from src.config import TEST_SEASONS, TOP_K_RANKING
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.shared.evaluation import compute_metrics, compute_ranking_metrics

EVAL_SEASONS_DEFAULT: tuple[int, ...] = tuple(TEST_SEASONS) if TEST_SEASONS else (2025,)
TARGET_POSITIONS_DEFAULT: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMAT_DEFAULT = "ppr"
OUTPUT_DIR_DEFAULT = "analysis_output"
N_BOOT_DEFAULT = 1000
SEED_DEFAULT = 0

# Production model column, in preference order. The attention NN is the shipped
# model for all six positions; pred_nn_total is the fallback when attention is
# disabled (e.g. a CONFIG_TINY run).
_MODEL_PRED_COLS: tuple[str, ...] = ("pred_attn_nn_total", "pred_nn_total")
_KEY_COLS = ["player_id", "season", "week"]


# ---------- Model-prediction sourcing ----------------------------------------


def _default_model_preds(
    pos: str, eval_seasons: Sequence[int], scoring_format: str
) -> pd.DataFrame:
    """Run the position's pipeline and return its held-out ``test_df``.

    The pipeline's test split is fixed by config (``TEST_SEASONS``); ``eval_seasons``
    is honored downstream by filtering the join, so passing a season the pipeline
    didn't test on simply yields an empty overlap (surfaced as a skip).
    """
    module = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    result = module.run()
    test_df = result.get("test_df")
    if test_df is None:
        raise KeyError(f"{pos} run() result has no 'test_df'")
    return test_df


# ---------- Per-position comparison ------------------------------------------


def _wilcoxon_abs_error(e_model: np.ndarray, e_expert: np.ndarray) -> dict | None:
    """Wilcoxon signed-rank on |model error| - |expert error| (robustness check).

    Returns ``None`` when every paired difference is zero (Wilcoxon is undefined
    there) — e.g. degenerate fixtures.
    """
    from scipy.stats import wilcoxon

    diff = np.abs(e_model) - np.abs(e_expert)
    if np.allclose(diff, 0.0):
        return None
    res = wilcoxon(diff)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def _compare_one_position(
    pos: str,
    model_df: pd.DataFrame,
    nflcom_full: pd.DataFrame,
    eval_seasons: Sequence[int],
    scoring_format: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Build the head-to-head block for one position (model vs NFL.com)."""
    eval_set = {int(s) for s in eval_seasons}

    model_col = next((c for c in _MODEL_PRED_COLS if c in model_df.columns), None)
    if model_col is None:
        return {
            "position": pos,
            "skipped": True,
            "reason": f"model test_df has no prediction column {list(_MODEL_PRED_COLS)}",
        }
    needed = set(_KEY_COLS) | {"fantasy_points", model_col}
    missing = needed - set(model_df.columns)
    if missing:
        return {
            "position": pos,
            "skipped": True,
            "reason": f"model test_df missing {sorted(missing)}",
        }

    model = model_df[list(needed)].copy()
    model = model[model["season"].astype(int).isin(eval_set)]
    model["player_id"] = model["player_id"].astype(str)
    model["season"] = model["season"].astype(int)
    model["week"] = model["week"].astype(int)

    expert = _project_nflcom_to_ppr(nflcom_full, pos, scoring_format)
    if expert.empty:
        return {"position": pos, "skipped": True, "reason": f"no NFL.com projections for {pos}"}
    expert = expert[_KEY_COLS + ["nflcom_pred_total"]].copy()
    expert["player_id"] = expert["player_id"].astype(str)
    expert["season"] = expert["season"].astype(int)
    expert["week"] = expert["week"].astype(int)
    expert = expert[expert["season"].isin(eval_set)]

    joined = model.merge(expert, on=_KEY_COLS, how="inner")
    if joined.empty:
        return {
            "position": pos,
            "skipped": True,
            "reason": f"no (player_id, season, week) overlap for {pos}",
        }

    actual = joined["fantasy_points"].to_numpy(dtype=float)
    model_pred = joined[model_col].to_numpy(dtype=float)
    expert_pred = joined["nflcom_pred_total"].to_numpy(dtype=float)
    e_model = model_pred - actual
    e_expert = expert_pred - actual

    model_metrics = compute_metrics(actual, model_pred)
    expert_metrics = compute_metrics(actual, expert_pred)

    model_rank = compute_ranking_metrics(
        joined, pred_col=model_col, true_col="fantasy_points", top_k=TOP_K_RANKING
    )
    expert_rank = compute_ranking_metrics(
        joined, pred_col="nflcom_pred_total", true_col="fantasy_points", top_k=TOP_K_RANKING
    )

    groups = joined["player_id"].to_numpy()
    boot_mae = paired_bootstrap_metric_ci(
        e_model, e_expert, metric="mae", groups=groups, n_boot=n_boot, seed=seed
    )
    boot_rmse = paired_bootstrap_metric_ci(
        e_model, e_expert, metric="rmse", groups=groups, n_boot=n_boot, seed=seed
    )

    return {
        "position": pos,
        "n_matched": int(len(joined)),
        "model_col": model_col,
        "totals_only": pos in _TOTALS_ONLY_POSITIONS,
        "model": {
            "mae": model_metrics["mae"],
            "rmse": model_metrics["rmse"],
            "r2": model_metrics["r2"],
            "top_k_hit_rate": model_rank["season_avg_hit_rate"],
            "spearman": model_rank["season_avg_spearman"],
        },
        "expert": {
            "mae": expert_metrics["mae"],
            "rmse": expert_metrics["rmse"],
            "r2": expert_metrics["r2"],
            "top_k_hit_rate": expert_rank["season_avg_hit_rate"],
            "spearman": expert_rank["season_avg_spearman"],
        },
        # delta = model - expert; negative ⇒ model better.
        "delta_mae": {"value": boot_mae["delta"], "ci_lo": boot_mae["lo"], "ci_hi": boot_mae["hi"]},
        "delta_rmse": {
            "value": boot_rmse["delta"],
            "ci_lo": boot_rmse["lo"],
            "ci_hi": boot_rmse["hi"],
        },
        "dm_mae": diebold_mariano_test(e_model, e_expert, power=1),
        "dm_rmse": diebold_mariano_test(e_model, e_expert, power=2),
        "bootstrap_mae": boot_mae,
        "bootstrap_rmse": boot_rmse,
        "wilcoxon_abs_error": _wilcoxon_abs_error(e_model, e_expert),
        "top_k": int(TOP_K_RANKING),
    }


# ---------- Printing ---------------------------------------------------------


def _fmt(x: float, width: int = 8, prec: int = 3) -> str:
    """Right-aligned float, or a placeholder for NaN/None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return f"{'-':>{width}}"
    return f"{x:>{width}.{prec}f}"


def _print_summary_table(
    blocks: list[dict], eval_seasons: Sequence[int], scoring_format: str
) -> None:
    season_label = "+".join(str(s) for s in eval_seasons)
    print("\n" + "=" * 100)
    print(f"Model vs NFL.com — head-to-head ({scoring_format.upper()}, {season_label})")
    print("  ΔMAE = model − expert (negative ⇒ model better); CI is a player-clustered bootstrap")
    print("=" * 100)
    header = (
        f"{'Pos':<5}{'n':>6}{'M_MAE':>8}{'E_MAE':>8}{'ΔMAE':>8}"
        f"{'[95% CI]':>18}{'DM_p':>8}{'M_top':>7}{'E_top':>7}{'M_ρ':>7}{'E_ρ':>7}"
    )
    print(header)
    print("-" * 100)
    for b in blocks:
        pos = b["position"]
        if b.get("skipped") or "model" not in b:
            print(f"{pos:<5}{'(skipped)':>10}  {b.get('reason', '')}")
            continue
        ci = f"[{b['delta_mae']['ci_lo']:.2f},{b['delta_mae']['ci_hi']:.2f}]"
        flag = " *K totals-only" if b.get("totals_only") else ""
        print(
            f"{pos:<5}{b['n_matched']:>6d}"
            f"{_fmt(b['model']['mae'])}{_fmt(b['expert']['mae'])}"
            f"{_fmt(b['delta_mae']['value'])}{ci:>18}"
            f"{_fmt(b['dm_mae']['p_value'])}"
            f"{_fmt(b['model']['top_k_hit_rate'], 7)}{_fmt(b['expert']['top_k_hit_rate'], 7)}"
            f"{_fmt(b['model']['spearman'], 7)}{_fmt(b['expert']['spearman'], 7)}{flag}"
        )
    print("=" * 100)


# ---------- Entry point ------------------------------------------------------


def main(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    scoring_format: str = SCORING_FORMAT_DEFAULT,
    positions: Sequence[str] = TARGET_POSITIONS_DEFAULT,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
    model_preds_loader=None,
    nflcom_loader=None,
) -> dict:
    """Run the model-vs-NFL.com comparison, print + write JSON, return the result.

    ``model_preds_loader(pos, eval_seasons, scoring_format) -> DataFrame`` and
    ``nflcom_loader(seasons) -> DataFrame`` are injectable for tests; production
    callers leave them at their defaults.
    """
    eval_seasons = tuple(int(s) for s in eval_seasons)
    if model_preds_loader is None:
        model_preds_loader = _default_model_preds
    if nflcom_loader is None:
        nflcom_loader = load_nflcom_with_gsis_id

    # The default model loader sources predictions from the pipeline's held-out
    # test_df, which is scored in the pipeline's configured format (PPR for the
    # shipped models). This script re-scores only the *expert* side to
    # ``scoring_format``; re-scoring it to a non-PPR format while the model side
    # stays PPR would be apples-to-oranges, so warn rather than silently mislead.
    if scoring_format != "ppr" and model_preds_loader is _default_model_preds:
        print(
            f"WARNING: --scoring-format={scoring_format} only re-scores the NFL.com (expert) "
            "side. The model side reflects the pipeline's configured scoring (PPR for the shipped "
            "models), so a non-PPR head-to-head is only valid if the pipeline is also run in that "
            "format."
        )

    print(f"\nLoading NFL.com projections for seasons {list(eval_seasons)}...")
    nflcom_full = nflcom_loader(list(eval_seasons))

    blocks: list[dict] = []
    for pos in positions:
        print(f"\n{'#' * 60}\n# {pos}\n{'#' * 60}")
        if pos in _SKIPPED_POSITIONS:
            blocks.append(
                {
                    "position": pos,
                    "skipped": True,
                    "reason": "NFL.com has no DST projections in hvpkod/NFL-Data",
                }
            )
            continue
        model_df = model_preds_loader(pos, eval_seasons, scoring_format)
        blocks.append(
            _compare_one_position(
                pos, model_df, nflcom_full, eval_seasons, scoring_format, n_boot, seed
            )
        )

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_seasons": list(eval_seasons),
        "scoring": scoring_format,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "expert_source": "nflcom",
        "positions": {b["position"]: b for b in blocks},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "expert_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nWrote {out_path}")

    _print_summary_table(blocks, eval_seasons, scoring_format)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Head-to-head: our model vs NFL.com expert projections (MAE/RMSE + ranking + paired significance)"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(EVAL_SEASONS_DEFAULT),
        help=f"Seasons to evaluate; should match the model's test split (default: TEST_SEASONS = {list(EVAL_SEASONS_DEFAULT)})",
    )
    parser.add_argument(
        "--scoring-format",
        default=SCORING_FORMAT_DEFAULT,
        choices=["ppr", "half_ppr", "standard"],
        help="Scoring format for the NFL.com side. NOTE: the model side reflects the pipeline's "
        "configured scoring (PPR for shipped models), so non-PPR is only valid if the pipeline is "
        "also run in that format (default: ppr).",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(TARGET_POSITIONS_DEFAULT),
        choices=list(TARGET_POSITIONS_DEFAULT),
        metavar="POS",
        help="Positions to evaluate (default: all). DST is skipped (no NFL.com DST).",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT, help="Bootstrap replicates")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Bootstrap PRNG seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        eval_seasons=tuple(args.seasons),
        scoring_format=args.scoring_format,
        positions=tuple(args.positions),
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
