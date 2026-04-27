"""Compare NFL.com weekly fantasy projections against our position models.

NFL.com is treated as a reference baseline for the test season (2025 by default).
For each position, we run the standard pipeline once, then for the same
(player_id, season, week) cells compute MAE / RMSE / R² of:

  1. Our model's predicted fantasy points (PPR; best-of attn_nn / nn / ridge).
  2. NFL.com's projected raw stats × our PPR scoring (apples-to-apples).
  3. Reference: NFL.com's own ``PlayerWeekProjectedPts`` (their standard scoring).
  4. Reference: ``SeasonAverageBaseline`` recomputed on the matched subset.

Outputs:
  analysis_output/nflcom_baseline_comparison.json -- full per-position breakout
  stdout                                         -- pretty-printed comparison table

Notes:
  - DST is hard-skipped (hvpkod has no DST/Defense file in the upstream archive).
  - K only gets a totals comparison (NFL.com's K projections are per-distance-
    bucket FG attempts; doesn't decompose to our K raw-stat targets).
  - Scoring format is parameterized so callers can rerun under half-PPR /
    standard without changing code.

Operator usage:
  python -m src.analysis.analysis_nflcom_baseline                    # default 2025 test, PPR
  python -m src.analysis.analysis_nflcom_baseline --positions QB RB
  python -m src.analysis.analysis_nflcom_baseline --scoring-format half_ppr
  python -m src.analysis.analysis_nflcom_baseline --force-refresh-nflcom
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import TEST_SEASONS
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline
from src.shared.aggregate_targets import (
    POSITION_TARGET_MAP,
    TARGET_UNITS,
    predictions_to_fantasy_points,
)
from src.shared.registry import get_inference_spec, get_runner

EVAL_SEASON_DEFAULT = TEST_SEASONS[0] if TEST_SEASONS else 2025
TARGET_POSITIONS_DEFAULT: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMAT_DEFAULT = "ppr"
OUTPUT_DIR_DEFAULT = "analysis_output"

# Positions where NFL.com cannot be projected per-target. K's projection schema
# is per-distance-bucket FG attempts; DST has no upstream file at all.
_TOTALS_ONLY_POSITIONS = {"K"}
_SKIPPED_POSITIONS = {"DST"}
_TIE_TOLERANCE_PTS = 0.01


# ---------- Internal helpers -------------------------------------------------


_MODEL_PRIORITY: tuple[tuple[str, str], ...] = (
    ("attn_nn", "Attention NN"),
    ("nn", "Multi-Head NN"),
    ("ridge", "Ridge Multi-Target"),
)


def _select_model_predictions(pipeline_result: dict, pos: str) -> tuple[dict, str]:
    """Pick the best-available model's per-target preds.

    Prefers the model with the lowest ``metrics["total"]["mae"]`` among those
    present in ``per_target_preds``. Falls back to a fixed
    attn_nn -> nn -> ridge priority order when metrics are missing or
    unusable. K and DST never run attn_nn; they fall back automatically.
    """
    per_target_preds = pipeline_result["per_target_preds"]

    candidates: list[tuple[float, int, str, str]] = []
    for priority_idx, (model_key, model_label) in enumerate(_MODEL_PRIORITY):
        if model_key not in per_target_preds:
            continue
        metrics = pipeline_result.get(f"{model_key}_metrics")
        if not isinstance(metrics, dict):
            continue
        total = metrics.get("total")
        if not isinstance(total, dict):
            continue
        mae = total.get("mae")
        try:
            mae_value = float(mae)
        except (TypeError, ValueError):
            continue
        if np.isnan(mae_value):
            continue
        # Tuple sort: lowest MAE first, priority tie-broken by index (lower wins).
        candidates.append((mae_value, priority_idx, model_key, model_label))

    if candidates:
        _, _, best_key, best_label = min(candidates)
        return per_target_preds[best_key], best_label

    # No usable metrics — fall back to fixed priority for whatever preds exist.
    for model_key, model_label in _MODEL_PRIORITY:
        if model_key in per_target_preds:
            return per_target_preds[model_key], model_label
    raise RuntimeError(f"No model predictions found in pipeline_result for {pos}")


def _project_nflcom_to_ppr(
    nflcom_df: pd.DataFrame, pos: str, scoring_format: str = "ppr"
) -> pd.DataFrame:
    """Apply ``predictions_to_fantasy_points`` to NFL.com's projected raw stats.

    Returns a frame keyed by (player_id, season, week) with columns:
      - nflcom_pred_total: the PPR-aggregated projection
      - nflcom_pred_<target>: per-target projections (QB/RB/WR/TE only)
      - nflcom_projected_pts: NFL.com's own (standard-scoring) projection, kept
        as a reference column for the ``nflcom_native`` metric.
    """
    pos_df = nflcom_df[nflcom_df["position"] == pos].copy()
    pos_df = pos_df[pos_df["player_id"].notna()]
    if pos_df.empty:
        return pos_df

    out = (
        pos_df[["player_id", "season", "week", "nflcom_projected_pts"]]
        .reset_index(drop=True)
        .copy()
    )

    if pos in _TOTALS_ONLY_POSITIONS:
        # K uses NFL.com's own projection as the head-to-head value.
        out["nflcom_pred_total"] = pos_df["nflcom_projected_pts"].to_numpy()
        return out

    targets = list(POSITION_TARGET_MAP[pos].keys())
    pred_dict = {t: pos_df[t].to_numpy() for t in targets}
    out["nflcom_pred_total"] = predictions_to_fantasy_points(pos, pred_dict, scoring_format)
    for t in targets:
        out[f"nflcom_pred_{t}"] = pos_df[t].to_numpy()
    return out


def _attach_model_predictions(
    test_df: pd.DataFrame,
    model_preds: dict,
    pos: str,
    scoring_format: str = "ppr",
) -> pd.DataFrame:
    """Add ``model_pred_total`` (and per-target ``model_pred_<t>`` for skill positions)
    columns to a copy of test_df. Position arrays are aligned with test_df row order
    by construction (pipeline emits per-row predictions in test_df order)."""
    out = test_df.copy().reset_index(drop=True)
    if pos in POSITION_TARGET_MAP:  # QB/RB/WR/TE
        out["model_pred_total"] = predictions_to_fantasy_points(pos, model_preds, scoring_format)
        for t, vals in model_preds.items():
            out[f"model_pred_{t}"] = np.asarray(vals)
    elif pos == "DST":
        # Routed via _dst_predictions_to_fantasy_points (uses tier lookups).
        out["model_pred_total"] = predictions_to_fantasy_points(pos, model_preds)
    else:
        # K: aggregate via the per-position signed sum. K's penalty heads
        # (fg_misses, xp_misses) carry positive raw values but contribute
        # negatively to fantasy points; a plain sum would systematically
        # overstate the projection. The canonical sign vector lives in the
        # registry's inference spec so we use the same source of truth as the
        # serving path.
        out["model_pred_total"] = _aggregate_k_predictions(model_preds)
    return out


def _aggregate_k_predictions(preds: dict) -> np.ndarray:
    """Apply K's signed-sum aggregator to a per-target prediction dict.

    Mirrors ``src.k.targets.compute_targets`` and the registry's K
    ``target_signs``: ``fantasy_points = fg_yard_points + pat_points
    - fg_misses - xp_misses``. Falls back to a plain sum only if the
    registry doesn't expose signs (defensive — shouldn't happen for K).
    """
    spec = get_inference_spec("K")
    signs = spec.get("target_signs")
    if not signs:
        # Defensive fallback: treat every head as additive (matches old behavior).
        return sum(np.asarray(v) for v in preds.values())
    total = None
    for target, sign in signs.items():
        if target not in preds:
            continue
        contribution = np.asarray(preds[target]) * float(sign)
        total = contribution if total is None else total + contribution
    if total is None:
        raise RuntimeError(
            "K aggregation: model_preds had no overlap with target_signs keys; "
            f"got {list(preds)} vs {list(signs)}"
        )
    return total


def _decide_winner(model_mae: float, nflcom_mae: float) -> str:
    if abs(model_mae - nflcom_mae) < _TIE_TOLERANCE_PTS:
        return "tie"
    return "model" if model_mae < nflcom_mae else "nflcom"


def _weekly_breakout(joined: pd.DataFrame) -> list[dict]:
    weekly = []
    for week in sorted(joined["week"].unique()):
        wk = joined[joined["week"] == week]
        if len(wk) == 0:
            continue
        weekly.append(
            {
                "week": int(week),
                "n": int(len(wk)),
                "model_mae": float(np.mean(np.abs(wk["fantasy_points"] - wk["model_pred_total"]))),
                "nflcom_ppr_mae": float(
                    np.mean(np.abs(wk["fantasy_points"] - wk["nflcom_pred_total"]))
                ),
            }
        )
    return weekly


def _per_target_breakout(joined: pd.DataFrame, pos: str) -> dict:
    if pos in _TOTALS_ONLY_POSITIONS or pos not in POSITION_TARGET_MAP:
        return {}
    out = {}
    for t in POSITION_TARGET_MAP[pos]:
        if t not in joined.columns:
            continue
        actual = joined[t].to_numpy()
        model_p = joined[f"model_pred_{t}"].to_numpy()
        nflcom_p = joined[f"nflcom_pred_{t}"].to_numpy()
        out[t] = {
            "model_mae": float(np.mean(np.abs(actual - model_p))),
            "nflcom_mae": float(np.mean(np.abs(actual - nflcom_p))),
            "unit": TARGET_UNITS.get(t, "pts"),
        }
    return out


def _compute_position_comparison(
    pos: str,
    pipeline_result: dict,
    nflcom_df: pd.DataFrame,
    eval_season: int,
    scoring_format: str = "ppr",
) -> dict:
    """Build the comparison dict for one position. See module docstring for shape."""
    test_df = pipeline_result["test_df"]
    test_df = test_df[test_df["season"] == eval_season].copy()
    if test_df.empty:
        return {
            "position": pos,
            "skipped": True,
            "reason": f"No rows in test_df for season {eval_season}",
        }

    model_preds, model_label = _select_model_predictions(pipeline_result, pos)
    # The pipeline emits per-target preds aligned with the full test_df it built.
    # If we filter test_df by season we have to apply the same mask to preds —
    # which is one-to-one only if the pipeline's test_df is exactly the eval window.
    full_test = pipeline_result["test_df"].reset_index(drop=True)
    season_mask = (full_test["season"] == eval_season).to_numpy()
    preds_filtered = {t: np.asarray(v)[season_mask] for t, v in model_preds.items()}

    test_with_model = _attach_model_predictions(test_df, preds_filtered, pos, scoring_format)

    nflcom_pred_df = _project_nflcom_to_ppr(nflcom_df, pos, scoring_format)
    if nflcom_pred_df.empty:
        return {
            "position": pos,
            "skipped": True,
            "reason": f"No NFL.com projection rows for {pos} in season {eval_season}",
        }

    join_cols = ["player_id", "season", "week"]
    joined = test_with_model.merge(nflcom_pred_df, on=join_cols, how="inner")
    if joined.empty:
        raise RuntimeError(
            f"No (player_id, season, week) overlap between test_df and NFL.com "
            f"projections for {pos}/{eval_season}. Check upstream URL or join keys."
        )

    fp_truth = joined["fantasy_points"].to_numpy()
    metrics_model = compute_metrics(fp_truth, joined["model_pred_total"].to_numpy())
    metrics_nflcom_ppr = compute_metrics(fp_truth, joined["nflcom_pred_total"].to_numpy())
    metrics_nflcom_native = compute_metrics(fp_truth, joined["nflcom_projected_pts"].to_numpy())

    # Recompute SeasonAverageBaseline on the matched subset for fair triangulation.
    season_avg_preds = SeasonAverageBaseline().predict(joined)
    metrics_season_avg = compute_metrics(fp_truth, season_avg_preds)

    return {
        "position": pos,
        "eval_window": f"{eval_season}_test",
        "scoring": scoring_format,
        "n_test_total": int(len(test_df)),
        "n_matched": int(len(joined)),
        "match_rate": float(len(joined) / len(test_df)),
        "model_label": model_label,
        "metrics": {
            "model": metrics_model,
            "nflcom_ppr": metrics_nflcom_ppr,
            "nflcom_native": metrics_nflcom_native,
            "season_avg": metrics_season_avg,
        },
        "per_target": _per_target_breakout(joined, pos),
        "weekly": _weekly_breakout(joined),
        "who_won": _decide_winner(metrics_model["mae"], metrics_nflcom_ppr["mae"]),
    }


def _compute_summary(per_pos_results: list[dict]) -> dict:
    """Roll up per-position 'who_won' tags + headline numbers."""
    who_won = {r["position"]: r.get("who_won", "n/a") for r in per_pos_results}
    headlines = {}
    for r in per_pos_results:
        if r.get("skipped"):
            continue
        headlines[r["position"]] = {
            "model_mae": r["metrics"]["model"]["mae"],
            "nflcom_ppr_mae": r["metrics"]["nflcom_ppr"]["mae"],
            "season_avg_mae": r["metrics"]["season_avg"]["mae"],
        }
    return {"who_won": who_won, "headlines": headlines}


def _print_comparison_table(per_pos_results: list[dict]) -> None:
    print("\n" + "=" * 96)
    print("NFL.com Baseline Comparison")
    print("=" * 96)
    print(
        f"{'Pos':<5} {'Model':<22} {'Model MAE':>10} {'NFL.com MAE':>12} "
        f"{'Season Avg':>11} {'Winner':>8} {'Match%':>8}"
    )
    print("-" * 96)
    for r in per_pos_results:
        if r.get("skipped"):
            print(f"{r['position']:<5} {'(skipped)':<22} {r.get('reason', '')}")
            continue
        m = r["metrics"]
        print(
            f"{r['position']:<5} {r['model_label']:<22} "
            f"{m['model']['mae']:>10.3f} {m['nflcom_ppr']['mae']:>12.3f} "
            f"{m['season_avg']['mae']:>11.3f} {r['who_won']:>8} "
            f"{r['match_rate']:>7.1%}"
        )
    print("=" * 96)


# ---------- Entry point ------------------------------------------------------


def main(
    eval_season: int = EVAL_SEASON_DEFAULT,
    scoring_format: str = SCORING_FORMAT_DEFAULT,
    positions: Sequence[str] = TARGET_POSITIONS_DEFAULT,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    *,
    force_refresh_nflcom: bool = False,
    pipeline_runner=None,
    nflcom_loader=None,
) -> dict:
    """Run the comparison, print + write JSON, return the result dict.

    ``pipeline_runner`` and ``nflcom_loader`` are injectable for tests; production
    callers should leave them at their defaults.
    """
    if pipeline_runner is None:
        pipeline_runner = get_runner
    if nflcom_loader is None:
        nflcom_loader = load_nflcom_with_gsis_id

    print(f"\nLoading NFL.com projections for season {eval_season}...")
    nflcom_full = nflcom_loader(seasons=[eval_season], force_refresh=force_refresh_nflcom)
    nflcom_eval = nflcom_full[nflcom_full["season"] == eval_season]

    per_pos_results: list[dict] = []
    for pos in positions:
        print(f"\n{'#' * 60}\n# {pos}\n{'#' * 60}")
        if pos in _SKIPPED_POSITIONS:
            per_pos_results.append(
                {
                    "position": pos,
                    "skipped": True,
                    "reason": "NFL.com has no DST projections in hvpkod/NFL-Data",
                }
            )
            continue
        runner = pipeline_runner(pos)
        pipeline_result = runner()
        per_pos_results.append(
            _compute_position_comparison(
                pos, pipeline_result, nflcom_eval, eval_season, scoring_format
            )
        )

    summary = _compute_summary(per_pos_results)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_window": f"{eval_season}_test",
        "scoring": scoring_format,
        "positions": {r["position"]: r for r in per_pos_results},
        "summary": summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "nflcom_baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    _print_comparison_table(per_pos_results)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare our position models against NFL.com weekly projections"
    )
    parser.add_argument("--eval-season", type=int, default=EVAL_SEASON_DEFAULT)
    parser.add_argument(
        "--scoring-format",
        default=SCORING_FORMAT_DEFAULT,
        choices=["ppr", "half_ppr", "standard"],
    )
    parser.add_argument("--positions", nargs="+", default=list(TARGET_POSITIONS_DEFAULT))
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-refresh-nflcom", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        eval_season=args.eval_season,
        scoring_format=args.scoring_format,
        positions=tuple(args.positions),
        output_dir=args.output_dir,
        force_refresh_nflcom=args.force_refresh_nflcom,
    )
