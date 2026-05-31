"""Generate the committed expert-comparison summary for the serving "Comparison" tab.

Writes ``src/serving/comparison_experts.json`` — for each position, the
``{mae, rmse, r2, n}`` of each EXPERT (NFL.com, RotoWire via Sleeper) scored
against actuals, on (a) all matched player-weeks and (b) the top-30-per-position
subset (ranked by actual fantasy points). Also emits the top-30 ``player_id`` sets
(so the serving app slices the *same* players for its live model column) and
source metadata for the tab's explanation block.

The MODEL column is deliberately NOT in this file: the serving app computes our
model's MAE/RMSE/R² live from loaded models (``_ensure_metrics`` →
``metrics_by_format``), so it auto-updates on every retrain. Experts are fixed
historical projections, so they're the only static half.

No model training: actuals come from nflverse ``stats_player_week`` (offense + K,
via the NFL.com-baseline helpers) and the DST team-build, scored through the same
PPR aggregator used for the projections — apples-to-apples.

Operator usage:
  python -m src.analysis.build_comparison_summary
  python -m src.analysis.build_comparison_summary --seasons 2025 --top-n 30
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.analysis.analysis_expert_comparison import (
    _EXPERT_PRED_COL,
    _project_sleeper_to_ppr,
)
from src.analysis.analysis_nflcom_baseline import (
    _actuals_for_position,
    _aggregate_actuals_to_ppr,
    _json_default,
    _load_actuals,
    _project_nflcom_to_ppr,
)
from src.analysis.sleeper_loader import load_sleeper_with_gsis_id
from src.config import TEST_SEASONS
from src.data.nflcom_loader import load_nflcom_with_gsis_id
from src.dst.data import build_data as build_dst_data
from src.dst.targets import compute_targets as compute_dst_targets
from src.shared.evaluation import compute_metrics

EVAL_SEASONS_DEFAULT: tuple[int, ...] = tuple(TEST_SEASONS) if TEST_SEASONS else (2025,)
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMAT = "ppr"
TOP_N_DEFAULT = 30
_KEYS = ["player_id", "season", "week"]

# Which expert covers which position (mirrors analysis_expert_comparison): NFL.com
# has no DST projections; RotoWire/Sleeper is offense + DST (K is totals-only).
_NFLCOM_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K"})
_ROTOWIRE_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "DST"})

_NFLCOM_NOTE = (
    "NFL.com weekly projections from the hvpkod/NFL-Data archive, scored through "
    "our PPR aggregator. Curated to likely-to-play players (no DST projections)."
)
_ROTOWIRE_NOTE = (
    "RotoWire projections via Sleeper's unofficial API — a single provider, not a "
    "consensus. Unprojected roster placeholders are dropped (K is out of scope)."
)

# Repo root = three levels up from this file (src/analysis/build_comparison_summary.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH_DEFAULT = os.path.join(_REPO_ROOT, "src", "serving", "comparison_experts.json")


def _round_metrics(actual: np.ndarray, pred: np.ndarray) -> dict | None:
    """compute_metrics on the paired arrays, rounded; ``None`` if empty."""
    if len(actual) == 0:
        return None
    m = compute_metrics(np.asarray(actual, dtype=float), np.asarray(pred, dtype=float))
    r2 = m["r2"]
    r2 = None if (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else round(float(r2), 4)
    return {
        "mae": round(float(m["mae"]), 4),
        "rmse": round(float(m["rmse"]), 4),
        "r2": r2,
        "n": int(len(actual)),
    }


def _position_actuals(
    pos: str, offense_actuals: pd.DataFrame, dst_actuals: pd.DataFrame, eval_seasons: Sequence[int]
) -> pd.DataFrame:
    """Per-position [player_id, season, week, actual_pts] scored through the aggregator.

    Offense + K use the nflverse stats_player_week feed (the NFL.com-baseline path);
    DST uses the team-build's tier-scored ``fantasy_points``. Returns an empty frame
    if the position has no actuals.
    """
    if pos == "DST":
        return dst_actuals
    pos_df = _actuals_for_position(offense_actuals, pos, eval_seasons)
    if pos_df.empty:
        return pd.DataFrame(columns=[*_KEYS, "actual_pts"])
    pos_df = pos_df.copy()
    pos_df["actual_pts"] = _aggregate_actuals_to_ppr(pos_df, pos, SCORING_FORMAT)
    return pos_df[[*_KEYS, "actual_pts"]]


def _dst_actuals(eval_seasons: Sequence[int]) -> pd.DataFrame:
    """DST actual fantasy points keyed by team (== the model's DST player_id)."""
    raw = compute_dst_targets(build_dst_data())
    eval_set = {int(s) for s in eval_seasons}
    df = raw[raw["season"].astype(int).isin(eval_set)].copy()
    out = df[["team", "season", "week", "fantasy_points"]].rename(
        columns={"team": "player_id", "fantasy_points": "actual_pts"}
    )
    return out.reset_index(drop=True)


def _top_n_ids(actuals: pd.DataFrame, top_n: int) -> list[str]:
    """Top-N player_ids for one position, ranked by total actual fantasy points."""
    if actuals.empty:
        return []
    totals = actuals.groupby("player_id")["actual_pts"].sum().sort_values(ascending=False)
    return [str(pid) for pid in totals.head(top_n).index.tolist()]


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def _expert_subsets(
    actuals: pd.DataFrame, projection: pd.DataFrame, pred_col: str, top_ids: set[str]
) -> dict:
    """Join one expert's projection to actuals; return all/top30 metric blocks."""
    if actuals.empty or projection is None or projection.empty:
        return {"all": None, "top30": None}
    a = _normalize_keys(actuals[[*_KEYS, "actual_pts"]])
    e = _normalize_keys(projection[[*_KEYS, pred_col]])
    joined = a.merge(e, on=_KEYS, how="inner")
    if joined.empty:
        return {"all": None, "top30": None}
    all_block = _round_metrics(joined["actual_pts"].to_numpy(), joined[pred_col].to_numpy())
    top = joined[joined["player_id"].isin(top_ids)]
    top_block = _round_metrics(top["actual_pts"].to_numpy(), top[pred_col].to_numpy())
    return {"all": all_block, "top30": top_block}


def build_summary(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
    *,
    nflcom_loader=None,
    sleeper_loader=None,
    actuals_loader=None,
    dst_actuals_loader=None,
) -> dict:
    """Build the committed expert-summary dict. Loaders are injectable for tests."""
    eval_seasons = tuple(int(s) for s in eval_seasons)
    nflcom_loader = nflcom_loader or load_nflcom_with_gsis_id
    sleeper_loader = sleeper_loader or load_sleeper_with_gsis_id
    actuals_loader = actuals_loader or _load_actuals
    dst_actuals_loader = dst_actuals_loader or _dst_actuals

    print(f"Loading actuals (offense + K) for {list(eval_seasons)}...")
    offense_actuals = actuals_loader(list(eval_seasons))
    print("Building DST actuals (team-level)...")
    dst_actuals = dst_actuals_loader(eval_seasons)

    print(f"Loading NFL.com projections for {list(eval_seasons)}...")
    nflcom_full = nflcom_loader(seasons=list(eval_seasons))
    print(f"Loading RotoWire (Sleeper) projections for {list(eval_seasons)}...")
    sleeper_full = sleeper_loader(list(eval_seasons))

    top30_ids: dict[str, list[str]] = {}
    subsets: dict[str, dict] = {"all": {}, "top30": {}}

    for pos in POSITIONS:
        actuals = _position_actuals(pos, offense_actuals, dst_actuals, eval_seasons)
        ids = _top_n_ids(actuals, top_n)
        top30_ids[pos] = ids
        id_set = set(ids)

        if pos in _NFLCOM_POSITIONS:
            nfl_proj = _project_nflcom_to_ppr(nflcom_full, pos, SCORING_FORMAT)
            nfl_blocks = _expert_subsets(actuals, nfl_proj, "nflcom_pred_total", id_set)
        else:
            nfl_blocks = {"all": None, "top30": None}

        if pos in _ROTOWIRE_POSITIONS:
            rw_proj = _project_sleeper_to_ppr(sleeper_full, pos, SCORING_FORMAT)
            rw_blocks = _expert_subsets(actuals, rw_proj, _EXPERT_PRED_COL, id_set)
        else:
            rw_blocks = {"all": None, "top30": None}

        subsets["all"][pos] = {"nflcom": nfl_blocks["all"], "rotowire": rw_blocks["all"]}
        subsets["top30"][pos] = {"nflcom": nfl_blocks["top30"], "rotowire": rw_blocks["top30"]}
        print(
            f"  {pos:<4} all: nflcom={nfl_blocks['all']} rotowire={rw_blocks['all']} "
            f"(top30 ids: {len(ids)})"
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scoring": SCORING_FORMAT,
        "test_seasons": list(eval_seasons),
        "top_n": int(top_n),
        "rank_basis": "actual_ppr_fantasy_points",
        "experts_meta": {
            "model": {"train": "2012-2023", "val": "2024", "test": "2025"},
            "nflcom": {"label": "NFL.com", "note": _NFLCOM_NOTE, "seasons": "2025"},
            "rotowire": {"label": "RotoWire", "note": _ROTOWIRE_NOTE, "seasons": "2025"},
        },
        "top30_ids": top30_ids,
        "subsets": subsets,
    }


def main(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
    output_path: str = OUTPUT_PATH_DEFAULT,
) -> dict:
    result = build_summary(eval_seasons, top_n)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nWrote {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the committed expert-comparison summary for the serving Comparison tab"
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=list(EVAL_SEASONS_DEFAULT))
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--output-path", default=OUTPUT_PATH_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(eval_seasons=tuple(args.seasons), top_n=args.top_n, output_path=args.output_path)
