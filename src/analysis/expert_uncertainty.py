"""Per-source uncertainty for the expert projection feeds (NFL.com, RotoWire).

Neither expert source ships an uncertainty field — both publish a single point
projection per player-week. This script *derives* one: over a multi-season
archive, it scores each source's projection against the realized outcome and
reports the **residual standard deviation** (σ) per source × position, alongside
bias, MAE, RMSE and R². σ is the "how noisy is this source" number — the spread
of `projection − actual` once the systematic bias is removed.

Residual convention (matches :func:`src.analysis.analysis_nflcom_baseline._compare_position`):
``residual = projection − actual``, so **bias > 0 ⇒ the source over-projects**.
σ is the sample std (``ddof=1``), the conventional standard-deviation estimator;
note ``RMSE² = bias² + population_variance`` (σ with ``ddof=0``), so σ and RMSE
carry the same information split differently — σ isolates the noise RMSE conflates
with bias.

Why multi-season: a single season's σ is a noisy estimate of a source's intrinsic
spread. The loaders pull the whole archive (NFL.com ≈2015–2025, RotoWire ≈2018–2025)
keyed by (player_id, season, week), so a ~8-season panel is free. A per-season
breakdown is emitted too, so year-to-year stability is visible.

Caveats (carried in the output ``note`` and surfaced in the serving UI):
  - **RotoWire/Sleeper provenance.** Sleeper's historical projections may be
    backfilled/look-ahead (see ``_SLEEPER_NOTE`` in analysis_expert_comparison),
    which would *understate* σ. NFL.com (genuine weekly hvpkod snapshots) is the
    trustworthy headline; sanity-check RotoWire's σ against NFL.com's.
  - **K is totals-only** for NFL.com (standard scoring, not PPR-decomposable) —
    its block is flagged ``totals_only``. RotoWire skips K.
  - **DST** has no NFL.com feed; its σ is RotoWire-only.

The model side is deliberately absent here: the model has no clean multi-season
*held-out* residuals (every season but the 2025 test split was trained on), so a
multi-season model σ would be leakage. This characterizes the *expert sources*.

Operator usage:
  python -m src.analysis.expert_uncertainty
  python -m src.analysis.expert_uncertainty --seasons 2023 2024 --scoring-format ppr
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.analysis.analysis_expert_comparison import _EXPERT_PRED_COL, _project_sleeper_to_ppr
from src.analysis.analysis_nflcom_baseline import (
    _TOTALS_ONLY_POSITIONS,
    _json_default,
    _load_actuals,
    _project_nflcom_to_ppr,
)
from src.analysis.build_comparison_summary import (
    _KEYS,
    _NFLCOM_POSITIONS,
    _ROTOWIRE_POSITIONS,
    POSITIONS,
    SCORING_FORMAT,
    _dst_actuals,
    _normalize_keys,
    _position_actuals,
)
from src.analysis.sleeper_loader import load_sleeper_with_gsis_id
from src.data.nflcom_loader import load_nflcom_with_gsis_id

# 2018 is the first season Sleeper/RotoWire publishes genuine projections (earlier
# is placeholder junk). NFL.com reaches back to ~2015 — pass --seasons explicitly
# for an NFL.com-only deep-history run. Through 2025 (the last completed season).
RELIABILITY_SEASONS_DEFAULT: tuple[int, ...] = tuple(range(2018, 2026))
OUTPUT_DIR_DEFAULT = "analysis_output"

_NOTE = (
    "Residual σ = std(projection − actual), sample std (ddof=1); bias > 0 ⇒ the "
    "source over-projects. NFL.com is genuine weekly snapshots; RotoWire (Sleeper) "
    "history is of unverified provenance — it may be backfilled, which would understate "
    "its σ — so read it as a sanity check, not gospel. K is NFL.com totals-only "
    "(standard scoring); DST is RotoWire-only."
)


def _residual_block(actual: np.ndarray, pred: np.ndarray) -> dict | None:
    """{n, mae, rmse, r2, bias, sigma} for paired arrays; ``None`` if empty.

    ``residual = pred − actual`` (bias > 0 ⇒ over-projection); ``sigma`` is the
    sample std (ddof=1), matching ``analysis_nflcom_baseline``'s ``std_resid``.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = int(len(actual))
    if n == 0:
        return None
    # compute_metrics is imported transitively; recompute mae/rmse/r2 locally to
    # avoid a second import hop and keep the block self-contained.
    from src.shared.evaluation import compute_metrics

    m = compute_metrics(actual, pred)
    resid = pred - actual
    bias = float(np.mean(resid))
    sigma = float(np.std(resid, ddof=1)) if n > 1 else 0.0
    r2 = m["r2"]
    r2 = None if (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else round(float(r2), 4)
    return {
        "n": n,
        "mae": round(float(m["mae"]), 4),
        "rmse": round(float(m["rmse"]), 4),
        "r2": r2,
        "bias": round(bias, 4),
        "sigma": round(sigma, 4),
    }


def _join_actuals(
    actuals: pd.DataFrame | None, projection: pd.DataFrame | None, pred_col: str
) -> pd.DataFrame | None:
    """Inner-join one source's projection to actuals on (player_id, season, week)."""
    if actuals is None or actuals.empty or projection is None or projection.empty:
        return None
    a = _normalize_keys(actuals[[*_KEYS, "actual_pts"]])
    e = _normalize_keys(projection[[*_KEYS, pred_col]])
    joined = a.merge(e, on=_KEYS, how="inner")
    return joined if not joined.empty else None


def _reliability_from_joined(
    joined: pd.DataFrame | None, pred_col: str, *, totals_only: bool = False
) -> dict | None:
    """Pooled residual block + a per-season breakdown for one (source, position)."""
    if joined is None:
        return None
    block = _residual_block(joined["actual_pts"].to_numpy(), joined[pred_col].to_numpy())
    if block is None:
        return None
    per_season: dict[int, dict] = {}
    for season, grp in joined.groupby("season"):
        sblk = _residual_block(grp["actual_pts"].to_numpy(), grp[pred_col].to_numpy())
        if sblk is not None:
            per_season[int(season)] = sblk
    block["per_season"] = per_season
    block["seasons"] = sorted(per_season)
    if totals_only:
        block["totals_only"] = True
    return block


def compute_expert_reliability(
    seasons: Sequence[int] = RELIABILITY_SEASONS_DEFAULT,
    *,
    scoring_format: str = SCORING_FORMAT,
    nflcom_loader=None,
    sleeper_loader=None,
    actuals_loader=None,
    dst_actuals_loader=None,
) -> dict:
    """Per-source × position residual σ over ``seasons``. Loaders injectable for tests.

    Returns ``{seasons, scoring, residual_convention, note, positions: {POS:
    {nflcom: block|None, rotowire: block|None}}}`` where each block is the
    :func:`_reliability_from_joined` shape.
    """
    seasons = tuple(int(s) for s in seasons)
    nflcom_loader = nflcom_loader or load_nflcom_with_gsis_id
    sleeper_loader = sleeper_loader or load_sleeper_with_gsis_id
    actuals_loader = actuals_loader or _load_actuals
    dst_actuals_loader = dst_actuals_loader or _dst_actuals

    # Calling conventions mirror build_comparison_summary.build_summary exactly, so
    # the same injected loaders work in both.
    offense_actuals = actuals_loader(list(seasons))
    dst_actuals = dst_actuals_loader(seasons)
    nflcom_full = nflcom_loader(seasons=list(seasons))
    sleeper_full = sleeper_loader(list(seasons))

    positions_out: dict[str, dict] = {}
    for pos in POSITIONS:
        actuals = _position_actuals(pos, offense_actuals, dst_actuals, seasons)

        nfl_block = None
        if pos in _NFLCOM_POSITIONS:
            nfl_proj = _project_nflcom_to_ppr(nflcom_full, pos, scoring_format)
            nfl_block = _reliability_from_joined(
                _join_actuals(actuals, nfl_proj, "nflcom_pred_total"),
                "nflcom_pred_total",
                totals_only=pos in _TOTALS_ONLY_POSITIONS,
            )

        rw_block = None
        if pos in _ROTOWIRE_POSITIONS:
            rw_proj = _project_sleeper_to_ppr(sleeper_full, pos, scoring_format)
            rw_block = _reliability_from_joined(
                _join_actuals(actuals, rw_proj, _EXPERT_PRED_COL), _EXPERT_PRED_COL
            )

        positions_out[pos] = {"nflcom": nfl_block, "rotowire": rw_block}

    return {
        "seasons": list(seasons),
        "scoring": scoring_format,
        "residual_convention": "projection_minus_actual",
        "note": _NOTE,
        "positions": positions_out,
    }


def _print_summary(result: dict) -> None:
    seasons = result.get("seasons", [])
    span = f"{min(seasons)}–{max(seasons)}" if seasons else "?"
    print("\n" + "=" * 78)
    print(
        f"Expert projection reliability — residual σ ({result.get('scoring', '').upper()}, {span})"
    )
    print("  σ = std(projection − actual); bias > 0 ⇒ source over-projects")
    print("=" * 78)
    header = f"{'Pos':<5}{'src':<10}{'n':>7}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'σ':>8}"
    for pos, cell in result.get("positions", {}).items():
        for src in ("nflcom", "rotowire"):
            block = cell.get(src)
            if pos == "QB" and src == "nflcom":
                print(header)
                print("-" * 78)
            if block is None:
                print(f"{pos:<5}{src:<10}{'(none)':>7}")
                continue
            flag = " *totals-only" if block.get("totals_only") else ""
            print(
                f"{pos:<5}{src:<10}{block['n']:>7d}{block['mae']:>8.3f}"
                f"{block['rmse']:>8.3f}{block['bias']:>8.3f}{block['sigma']:>8.3f}{flag}"
            )
    print("=" * 78)
    print(f"note: {result.get('note', '')}")


def main(
    seasons: Sequence[int] = RELIABILITY_SEASONS_DEFAULT,
    scoring_format: str = SCORING_FORMAT,
    output_dir: str = OUTPUT_DIR_DEFAULT,
) -> dict:
    """Compute, print, and write ``analysis_output/expert_uncertainty.json``."""
    result = compute_expert_reliability(seasons, scoring_format=scoring_format)
    result["generated_at"] = datetime.now(UTC).isoformat()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "expert_uncertainty.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nWrote {out_path}")
    _print_summary(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-source residual σ for the expert projection feeds (NFL.com, RotoWire)"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(RELIABILITY_SEASONS_DEFAULT),
        help=f"Seasons to pool (default: {list(RELIABILITY_SEASONS_DEFAULT)})",
    )
    parser.add_argument(
        "--scoring-format", default=SCORING_FORMAT, choices=["ppr", "half_ppr", "standard"]
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        seasons=tuple(args.seasons),
        scoring_format=args.scoring_format,
        output_dir=args.output_dir,
    )
