"""Compute NFL.com's projection accuracy vs. actuals.

For each (player_id, season, week) cell where NFL.com has a projection AND
nflverse has actual game stats, computes NFL.com's MAE / RMSE / R² on
PPR-scored fantasy points (NFL.com's projected raw stats × our PPR aggregator
vs. actual raw stats × the same aggregator -- apples-to-apples).

This script does NOT train any models. Compare the numbers it prints to our
model's MAE in ``docs/expert_comparison.md`` "Our Model Performance" table.

Outputs:
  analysis_output/nflcom_baseline.json -- per-position metrics, per-season + pooled
  stdout                              -- pretty-printed table

Notes:
  - DST is hard-skipped (no NFL.com DST projections in hvpkod's archive).
  - K is totals-only: NFL.com's K projection is per-distance-bucket FG attempts
    which doesn't decompose to our raw-stat targets, so we use NFL.com's own
    ``nflcom_projected_pts`` (standard scoring, format-invariant for K) vs the
    actual ``fantasy_points`` column from nflverse.
  - Scoring format is parameterized so callers can rerun under half-PPR /
    standard without changing code.

Operator usage:
  python -m src.analysis.analysis_nflcom_baseline                 # default = TEST_SEASONS, PPR
  python -m src.analysis.analysis_nflcom_baseline --seasons 2024 2025
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
from src.shared.aggregate_targets import (
    POSITION_TARGET_MAP,
    TARGET_UNITS,
    predictions_to_fantasy_points,
)

EVAL_SEASONS_DEFAULT: tuple[int, ...] = tuple(TEST_SEASONS) if TEST_SEASONS else (2025,)
TARGET_POSITIONS_DEFAULT: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMAT_DEFAULT = "ppr"
OUTPUT_DIR_DEFAULT = "analysis_output"

# Positions where NFL.com can't be decomposed per-target. K's projection
# schema is per-distance-bucket FG attempts; DST has no upstream file at all.
_TOTALS_ONLY_POSITIONS = {"K"}
_SKIPPED_POSITIONS = {"DST"}

# nflverse stats_player_week parquets — one per season, covers all positions
# including K (which nfl_data_py.import_weekly_data omits).
_NFLVERSE_PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.parquet"
)

# Extra columns retained for K so we can compute actual K fantasy points
# directly from raw stats (the parquet's ``fantasy_points`` column is offensive-
# only and reports ~0 for kickers).
_K_RAW_COLUMNS = ("fg_made_distance", "pat_made", "fg_missed", "pat_missed")


# ---------- Internal helpers -------------------------------------------------


def _json_default(o):
    """Strict serializer for ``json.dump`` — handles numpy scalars + arrays."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable: {o!r}")


def _load_actuals(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch weekly actual stats for the requested seasons from nflverse.

    Reads one parquet per season directly from the nflverse stats_player_week
    release. Covers QB/RB/WR/TE/K offensive + kicker stats from 2012+ (DST
    actuals live in stats_team and are not used here — DST is skipped). The
    rename step normalizes columns to match NFL.com's schema (``interceptions``
    instead of ``passing_interceptions``).
    """
    seasons = sorted(set(int(s) for s in seasons))
    if not seasons:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    for s in seasons:
        df = pd.read_parquet(_NFLVERSE_PLAYER_WEEK_URL.format(season=s))
        df = df.rename(
            columns={
                "team": "recent_team",
                "passing_interceptions": "interceptions",
            }
        )
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    # Derive a single ``fumbles_lost`` column from the three nflverse fumble
    # subcolumns — matches ``src.data.loader.compute_fantasy_points``.
    fumble_cols = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
    out["fumbles_lost"] = sum(out[c].fillna(0) if c in out.columns else 0.0 for c in fumble_cols)
    return out


def _actuals_for_position(
    actuals: pd.DataFrame, pos: str, eval_seasons: Sequence[int]
) -> pd.DataFrame:
    """Filter actuals to one position + season range, keep the cols we need."""
    eval_set = set(int(s) for s in eval_seasons)
    df = actuals[(actuals["position"] == pos) & (actuals["season"].isin(eval_set))].copy()
    if df.empty:
        return df
    # Coerce raw-stat columns; missing columns (e.g. K's offensive stats)
    # become zero rather than NaN so the aggregator doesn't choke.
    for target in POSITION_TARGET_MAP.get(pos, {}):
        if target not in df.columns:
            df[target] = 0.0
        else:
            df[target] = df[target].fillna(0.0)
    # K needs its raw kick stats kept around so we can compute actual K
    # fantasy points (the precomputed ``fantasy_points`` column is offensive-
    # only — kickers report ~0 there).
    extra_cols: list[str] = []
    if pos == "K":
        for c in _K_RAW_COLUMNS:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
            else:
                df[c] = 0.0
            extra_cols.append(c)
    return df[
        ["player_id", "season", "week", "position", "fantasy_points"]
        + [t for t in POSITION_TARGET_MAP.get(pos, {})]
        + extra_cols
    ].reset_index(drop=True)


def _project_nflcom_to_ppr(
    nflcom_df: pd.DataFrame, pos: str, scoring_format: str = "ppr"
) -> pd.DataFrame:
    """Apply ``predictions_to_fantasy_points`` to NFL.com's projected raw stats.

    Returns a frame keyed by (player_id, season, week) with columns:
      - nflcom_pred_total: the PPR-aggregated projection (their raw stats × our scoring)
      - nflcom_pred_<target>: per-target projections (QB/RB/WR/TE only)
      - nflcom_projected_pts: NFL.com's own (standard-scoring) projection, kept
        as a reference for callers that want to compare native vs aggregated.
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
        out["nflcom_pred_total"] = pos_df["nflcom_projected_pts"].to_numpy()
        return out

    targets = list(POSITION_TARGET_MAP[pos].keys())
    pred_dict = {t: pos_df[t].to_numpy() for t in targets}
    out["nflcom_pred_total"] = predictions_to_fantasy_points(pos, pred_dict, scoring_format)
    for t in targets:
        out[f"nflcom_pred_{t}"] = pos_df[t].to_numpy()
    return out


def _aggregate_actuals_to_ppr(actuals: pd.DataFrame, pos: str, scoring_format: str) -> np.ndarray:
    """Re-score actuals through our aggregator so they're apples-to-apples
    with the NFL.com projections.

    QB/RB/WR/TE: ``predictions_to_fantasy_points`` with the position's target map.
    K: computed directly from raw FG/PAT stats per ``src.k.targets.compute_targets``
       (``fg_made_distance × 0.1 + pat_made − fg_missed − pat_missed``). The
       precomputed ``fantasy_points`` column on the parquet is offensive-only and
       reads ~0 for kickers, so we can't use it.
    """
    if pos == "K":
        # Falls back to 0.0 columns from _actuals_for_position if the parquet
        # is missing any of these (older seasons can have partial schema).
        fg_yards = actuals["fg_made_distance"].to_numpy()
        pat_made = actuals["pat_made"].to_numpy()
        fg_missed = actuals["fg_missed"].to_numpy()
        pat_missed = actuals["pat_missed"].to_numpy()
        return fg_yards * 0.1 + pat_made - fg_missed - pat_missed
    target_map = POSITION_TARGET_MAP[pos]
    pred_dict = {t: actuals[t].to_numpy() for t in target_map}
    return predictions_to_fantasy_points(pos, pred_dict, scoring_format)


def _per_target_breakout(joined: pd.DataFrame, pos: str) -> dict:
    if pos in _TOTALS_ONLY_POSITIONS or pos not in POSITION_TARGET_MAP:
        return {}
    out = {}
    for t in POSITION_TARGET_MAP[pos]:
        if t not in joined.columns or f"nflcom_pred_{t}" not in joined.columns:
            continue
        actual = joined[t].to_numpy()
        nflcom_p = joined[f"nflcom_pred_{t}"].to_numpy()
        out[t] = {
            "mae": float(np.mean(np.abs(actual - nflcom_p))),
            "rmse": float(np.sqrt(np.mean((actual - nflcom_p) ** 2))),
            "unit": TARGET_UNITS.get(t, "pts"),
        }
    return out


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
                "mae": float(np.mean(np.abs(wk["actual_pts"] - wk["nflcom_pred_total"]))),
            }
        )
    return weekly


def _compute_season_block(
    pos: str,
    nflcom_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    season: int,
    scoring_format: str,
) -> dict:
    """Compute per-season metrics for one (position, season)."""
    nflcom_season = nflcom_df[nflcom_df["season"] == season]
    actuals_season = actuals_df[actuals_df["season"] == season]
    if nflcom_season.empty or actuals_season.empty:
        return {
            "season": season,
            "skipped": True,
            "reason": "no NFL.com or no actuals rows for this season",
        }

    nflcom_pred = _project_nflcom_to_ppr(nflcom_season, pos, scoring_format)
    if nflcom_pred.empty:
        return {
            "season": season,
            "skipped": True,
            "reason": f"no NFL.com projection rows for {pos}/{season}",
        }

    actuals_pos = _actuals_for_position(actuals_season, pos, [season])
    if actuals_pos.empty:
        return {
            "season": season,
            "skipped": True,
            "reason": f"no actuals rows for {pos}/{season}",
        }
    actuals_pos["actual_pts"] = _aggregate_actuals_to_ppr(actuals_pos, pos, scoring_format)

    joined = actuals_pos.merge(nflcom_pred, on=["player_id", "season", "week"], how="inner")
    if joined.empty:
        return {
            "season": season,
            "skipped": True,
            "reason": f"no (player_id, season, week) overlap for {pos}/{season}",
        }

    metrics = compute_metrics(
        joined["actual_pts"].to_numpy(), joined["nflcom_pred_total"].to_numpy()
    )
    return {
        "season": int(season),
        "n_matched": int(len(joined)),
        "metrics": metrics,
        "per_target": _per_target_breakout(joined, pos),
        "weekly": _weekly_breakout(joined),
        "_joined": joined,  # private — used by the pooled rollup, stripped before JSON
    }


def _compute_position_comparison(
    pos: str,
    nflcom_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    eval_seasons: Sequence[int],
    scoring_format: str = "ppr",
) -> dict:
    """Build the per-position block (per_season + pooled)."""
    per_season: dict[str, dict] = {}
    pooled_frames: list[pd.DataFrame] = []
    for season in eval_seasons:
        block = _compute_season_block(pos, nflcom_df, actuals_df, season, scoring_format)
        joined = block.pop("_joined", None)
        per_season[str(int(season))] = block
        if joined is not None and not joined.empty:
            pooled_frames.append(joined)

    if not pooled_frames:
        return {"position": pos, "per_season": per_season, "pooled": {"skipped": True}}

    pooled = pd.concat(pooled_frames, ignore_index=True)
    pooled_metrics = compute_metrics(
        pooled["actual_pts"].to_numpy(), pooled["nflcom_pred_total"].to_numpy()
    )
    return {
        "position": pos,
        "per_season": per_season,
        "pooled": {
            "n_matched": int(len(pooled)),
            "metrics": pooled_metrics,
            "per_target": _per_target_breakout(pooled, pos),
        },
    }


def _print_summary_table(per_pos_results: list[dict], eval_seasons: Sequence[int]) -> None:
    season_label = "+".join(str(s) for s in eval_seasons)
    print("\n" + "=" * 72)
    print(f"NFL.com Projection Accuracy ({season_label})")
    print("=" * 72)
    print(f"{'Pos':<5} {'n':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 72)
    for r in per_pos_results:
        pos = r["position"]
        pooled = r.get("pooled", {})
        if pooled.get("skipped") or "metrics" not in pooled:
            print(f"{pos:<5} {'(skipped)':<22} {pooled.get('reason', '')}")
            continue
        m = pooled["metrics"]
        print(
            f"{pos:<5} {pooled['n_matched']:>6d} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}"
        )
    print("=" * 72)


# ---------- Entry point ------------------------------------------------------


def main(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    scoring_format: str = SCORING_FORMAT_DEFAULT,
    positions: Sequence[str] = TARGET_POSITIONS_DEFAULT,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    *,
    force_refresh_nflcom: bool = False,
    nflcom_loader=None,
    actuals_loader=None,
) -> dict:
    """Run the NFL.com baseline analysis, print + write JSON, return the result.

    ``nflcom_loader`` and ``actuals_loader`` are injectable for tests; production
    callers should leave them at their defaults.
    """
    eval_seasons = tuple(int(s) for s in eval_seasons)
    if nflcom_loader is None:
        nflcom_loader = load_nflcom_with_gsis_id
    if actuals_loader is None:
        actuals_loader = _load_actuals

    print(f"\nLoading NFL.com projections for seasons {list(eval_seasons)}...")
    nflcom_full = nflcom_loader(seasons=list(eval_seasons), force_refresh=force_refresh_nflcom)

    print(f"Loading actuals for seasons {list(eval_seasons)}...")
    actuals_full = actuals_loader(list(eval_seasons))

    per_pos_results: list[dict] = []
    for pos in positions:
        print(f"\n{'#' * 60}\n# {pos}\n{'#' * 60}")
        if pos in _SKIPPED_POSITIONS:
            per_pos_results.append(
                {
                    "position": pos,
                    "skipped": True,
                    "reason": "NFL.com has no DST projections in hvpkod/NFL-Data",
                    "per_season": {},
                    "pooled": {"skipped": True, "reason": "DST skipped"},
                }
            )
            continue
        per_pos_results.append(
            _compute_position_comparison(
                pos, nflcom_full, actuals_full, eval_seasons, scoring_format
            )
        )

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_seasons": list(eval_seasons),
        "scoring": scoring_format,
        "positions": {r["position"]: r for r in per_pos_results},
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "nflcom_baseline.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nWrote {out_path}")

    _print_summary_table(per_pos_results, eval_seasons)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NFL.com's projection accuracy vs actuals (no model retraining)"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(EVAL_SEASONS_DEFAULT),
        help=f"Seasons to evaluate (default: TEST_SEASONS = {list(EVAL_SEASONS_DEFAULT)})",
    )
    parser.add_argument(
        "--scoring-format",
        default=SCORING_FORMAT_DEFAULT,
        choices=["ppr", "half_ppr", "standard"],
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(TARGET_POSITIONS_DEFAULT),
        choices=list(TARGET_POSITIONS_DEFAULT),
        metavar="POS",
        help="Positions to evaluate (default: all). DST is hard-skipped.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--force-refresh-nflcom", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        eval_seasons=tuple(args.seasons),
        scoring_format=args.scoring_format,
        positions=tuple(args.positions),
        output_dir=args.output_dir,
        force_refresh_nflcom=args.force_refresh_nflcom,
    )
