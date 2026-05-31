"""Unit tests for src/analysis/expert_uncertainty.py + the build_comparison_summary
enrichment that embeds its output.

The default loaders hit the network — out of scope for unit tests. These inject
stub NFL.com / Sleeper / actuals loaders so the residual-σ math, per-season
breakdown, position-coverage gaps (NFL.com no-DST, RotoWire no-K), the totals-only
K flag, and the wiring into ``build_comparison_summary.build_summary`` are all
exercised on synthetic frames. ``_residual_block`` is also pinned directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import build_comparison_summary as bcs
from src.analysis import expert_uncertainty as mod

pytestmark = pytest.mark.unit

_QB_TARGETS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
)
_DST_TARGETS = (
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
    "points_allowed",
    "yards_allowed",
)
_K_RAW = ("fg_made_distance", "pat_made", "fg_missed", "pat_missed")


def _qb_raw(i: int, wk: int) -> dict:
    """Varied QB raw stats so aggregated totals differ row-to-row."""
    return {
        "passing_yards": 220.0 + 5 * i + wk,
        "rushing_yards": 10.0 + i,
        "passing_tds": 1.0 + (i % 3),
        "rushing_tds": float(i % 2),
        "interceptions": float((i + wk) % 2),
        "fumbles_lost": 0.0,
    }


def _nflcom_loader(seasons):
    """Raw NFL.com frame: QB (P1..P4) + K (K1..K2), all requested seasons, weeks 1-2."""
    rows = []
    for s in seasons:
        for wk in (1, 2):
            for i in range(1, 5):
                row = {
                    "position": "QB",
                    "player_id": f"P{i}",
                    "season": int(s),
                    "week": wk,
                    "nflcom_projected_pts": 14.0 + i,
                }
                row.update(_qb_raw(i, wk))
                rows.append(row)
            for i in range(1, 3):
                rows.append(
                    {
                        "position": "K",
                        "player_id": f"K{i}",
                        "season": int(s),
                        "week": wk,
                        "nflcom_projected_pts": 8.0 + i,
                    }
                )
    return pd.DataFrame(rows)


def _sleeper_loader(seasons):
    """Gsis-joined Sleeper frame: QB (P1..P4) + DST (T1..T2), weeks 1-2."""
    rows = []
    for s in seasons:
        for wk in (1, 2):
            for i in range(1, 5):
                row = {"position": "QB", "player_id": f"P{i}", "season": int(s), "week": wk}
                row.update(_qb_raw(i + 1, wk))  # offset so Sleeper ≠ NFL.com
                rows.append(row)
            for i in range(1, 3):
                row = {"position": "DST", "player_id": f"T{i}", "season": int(s), "week": wk}
                for j, t in enumerate(_DST_TARGETS):
                    row[t] = float((i + j + wk) % 3)
                row["points_allowed"] = 14.0 + i
                row["yards_allowed"] = 300.0 + 10.0 * i
                rows.append(row)
    return pd.DataFrame(rows)


def _actuals_loader(seasons):
    """nflverse-shaped offense actuals: QB + K rows with raw stat columns."""
    rows = []
    for s in seasons:
        for wk in (1, 2):
            for i in range(1, 5):
                row = {
                    "player_id": f"P{i}",
                    "season": int(s),
                    "week": wk,
                    "position": "QB",
                    "fantasy_points": 12.0 + i,
                }
                row.update(_qb_raw(i, wk + 1))  # actual ≠ either projection
                rows.append(row)
            for i in range(1, 3):
                row = {
                    "player_id": f"K{i}",
                    "season": int(s),
                    "week": wk,
                    "position": "K",
                    "fantasy_points": 7.0 + i,
                }
                for c in _K_RAW:
                    row[c] = 0.0
                row["fg_made_distance"] = 90.0 + i
                row["pat_made"] = 2.0
                rows.append(row)
    return pd.DataFrame(rows)


def _dst_actuals_loader(seasons):
    """_dst_actuals shape: [player_id, season, week, actual_pts] keyed by team."""
    rows = []
    for s in seasons:
        for wk in (1, 2):
            for i in range(1, 3):
                rows.append(
                    {
                        "player_id": f"T{i}",
                        "season": int(s),
                        "week": wk,
                        "actual_pts": 6.0 + i + wk,
                    }
                )
    return pd.DataFrame(rows)


def _reliability(seasons=(2025,)):
    return mod.compute_expert_reliability(
        seasons,
        nflcom_loader=_nflcom_loader,
        sleeper_loader=_sleeper_loader,
        actuals_loader=_actuals_loader,
        dst_actuals_loader=_dst_actuals_loader,
    )


# --------------------------------------------------------------------------- #
# _residual_block — pure math
# --------------------------------------------------------------------------- #


def test_residual_block_math_and_identity():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 33.0, 38.0])  # resid = [+2,-2,+3,-2]
    b = mod._residual_block(actual, pred)
    resid = pred - actual
    assert b["n"] == 4
    assert b["bias"] == pytest.approx(round(float(np.mean(resid)), 4))
    assert b["sigma"] == pytest.approx(round(float(np.std(resid, ddof=1)), 4))  # sample std
    assert b["mae"] == pytest.approx(round(float(np.mean(np.abs(resid))), 4))
    # RMSE² = bias² + population variance (ddof=0) — σ and RMSE carry the same info.
    assert b["rmse"] ** 2 == pytest.approx(b["bias"] ** 2 + np.std(resid, ddof=0) ** 2, abs=1e-3)


def test_residual_block_empty_is_none():
    assert mod._residual_block(np.array([]), np.array([])) is None


def test_residual_block_single_row_sigma_zero():
    b = mod._residual_block(np.array([10.0]), np.array([13.0]))
    assert b["n"] == 1 and b["bias"] == 3.0 and b["sigma"] == 0.0


# --------------------------------------------------------------------------- #
# compute_expert_reliability — coverage + shape
# --------------------------------------------------------------------------- #


def test_qb_has_both_sources_with_full_block():
    qb = _reliability()["positions"]["QB"]
    for src in ("nflcom", "rotowire"):
        block = qb[src]
        assert {"n", "mae", "rmse", "r2", "bias", "sigma", "per_season", "seasons"} <= set(block)
        assert block["n"] == 8  # P1..P4 × 2 weeks
        assert block["sigma"] >= 0.0


def test_position_coverage_gaps():
    """NFL.com has no DST; RotoWire has no K — those cells are None, the covered
    sibling is present."""
    pos = _reliability()["positions"]
    assert pos["DST"]["nflcom"] is None
    assert pos["DST"]["rotowire"] is not None  # RotoWire covers DST
    assert pos["K"]["rotowire"] is None
    assert pos["K"]["nflcom"] is not None  # NFL.com covers K (totals-only)


def test_kicker_flagged_totals_only():
    assert _reliability()["positions"]["K"]["nflcom"].get("totals_only") is True
    # Non-totals positions don't carry the flag.
    assert "totals_only" not in _reliability()["positions"]["QB"]["nflcom"]


def test_per_season_breakdown_spans_requested_seasons():
    qb = _reliability(seasons=(2024, 2025))["positions"]["QB"]["nflcom"]
    assert qb["seasons"] == [2024, 2025]
    assert set(qb["per_season"]) == {2024, 2025}
    assert qb["n"] == 16  # 8 per season × 2 seasons
    for season_block in qb["per_season"].values():
        assert season_block["n"] == 8


def test_top_level_metadata():
    out = _reliability(seasons=(2024, 2025))
    assert out["seasons"] == [2024, 2025]
    assert out["residual_convention"] == "projection_minus_actual"
    assert "provenance" in out["note"].lower()  # RotoWire caveat rides along


# --------------------------------------------------------------------------- #
# build_comparison_summary enrichment — the block reaches the committed dict
# --------------------------------------------------------------------------- #


def test_build_summary_embeds_expert_reliability():
    summary = bcs.build_summary(
        eval_seasons=(2025,),
        top_n=2,
        nflcom_loader=_nflcom_loader,
        sleeper_loader=_sleeper_loader,
        actuals_loader=_actuals_loader,
        dst_actuals_loader=_dst_actuals_loader,
        reliability_seasons=(2025,),
    )
    rel = summary["expert_reliability"]
    assert set(rel["positions"]) == set(bcs.POSITIONS)
    assert rel["positions"]["QB"]["nflcom"]["sigma"] >= 0.0
    assert rel["positions"]["DST"]["nflcom"] is None
    assert rel["positions"]["K"]["rotowire"] is None
