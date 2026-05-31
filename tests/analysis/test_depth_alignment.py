"""Tests for src/analysis/audit_depth_alignment.py.

The module's ``main()`` fetches weekly stats + raw depth charts over many seasons
(slow, network-dependent) — out of scope for unit tests. These cover the pure
detection functions against a synthetic frame with a *deliberate* one-week stale
depth chart (chart labeled week W lists week W-1's starter), asserting that:

  1. as-is (no relabel), the audit classifies it ``SHIFTED_BACK_1``;
  2. relabeling the chart ``week -= 1`` flips it to ``ALIGNED`` and lifts the overall
     starter-match — i.e. the loader's ``_fetch_depth`` correction is the right one.

A future regression that re-introduced the stale alignment (or broke the detector)
fails here without needing the real nflverse data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import audit_depth_alignment as ada

_SEASON = 2099
_N_TEAMS = 40  # > _MIN_TRANSITIONS so classify() returns a verdict, not INSUFFICIENT_DATA


def _synthetic_stale_by_one() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (weekly, depth) where each team changes QB at week 3 and the depth chart
    is stale by exactly one week: chart[W] rank-1 == the true week-(W-1) starter."""
    weekly_rows, depth_rows = [], []
    for t in range(_N_TEAMS):
        team = f"TM{t:02d}"
        starter_a, starter_b = f"A{t:02d}", f"B{t:02d}"

        def true_starter(week: int, a=starter_a, b=starter_b) -> str:
            return a if week <= 2 else b  # QB change at week 3

        # Actual results: one unambiguous starter per team-week (max attempts).
        for w in range(1, 6):
            weekly_rows.append(
                {
                    "season": _SEASON,
                    "week": w,
                    "recent_team": team,
                    "season_type": "REG",
                    "position": "QB",
                    "player_id": true_starter(w),
                    "attempts": 30,
                }
            )
        # Stale chart: chart[W] names the prior week's starter (chart[1] := week-0 = A).
        # Provide week 6 too so the week-=1 fix can backfill week 5.
        for w in range(1, 7):
            stale = starter_a if w - 1 <= 2 else starter_b
            depth_rows.append(
                {
                    "season": _SEASON,
                    "week": w,
                    "club_code": team,
                    "game_type": "REG",
                    "position": "QB",
                    "formation": "Offense",
                    "gsis_id": stale,
                    "depth_team": "1",
                }
            )
            depth_rows.append(  # a rank-2 QB so the min-rank pick is meaningful
                {
                    "season": _SEASON,
                    "week": w,
                    "club_code": team,
                    "game_type": "REG",
                    "position": "QB",
                    "formation": "Offense",
                    "gsis_id": f"C{t:02d}",
                    "depth_team": "2",
                }
            )
    return pd.DataFrame(weekly_rows), pd.DataFrame(depth_rows)


@pytest.mark.unit
def test_audit_detects_stale_by_one_week():
    weekly, depth = _synthetic_stale_by_one()
    starters = ada.qb_starters(weekly)
    rates = ada.alignment_rates(starters, ada.chart_rank1_qb(depth, week_shift=0))

    assert rates["n_transitions"] == _N_TEAMS
    # At QB-change weeks the chart names the PRIOR week's starter, not the current one.
    assert rates["transition_prev"] == pytest.approx(1.0)
    assert rates["transition_current"] == pytest.approx(0.0)
    assert ada.classify(rates).startswith("SHIFTED_BACK_1")


@pytest.mark.unit
def test_week_minus_one_relabel_realigns():
    weekly, depth = _synthetic_stale_by_one()
    starters = ada.qb_starters(weekly)
    as_is = ada.alignment_rates(starters, ada.chart_rank1_qb(depth, week_shift=0))
    fixed = ada.alignment_rates(starters, ada.chart_rank1_qb(depth, week_shift=-1))

    # The -1 relabel (the loader's fix) flips the verdict and lifts overall match.
    assert fixed["transition_current"] == pytest.approx(1.0)
    assert fixed["transition_prev"] == pytest.approx(0.0)
    assert fixed["overall_current"] > as_is["overall_current"]
    assert ada.classify(fixed).startswith("ALIGNED")


@pytest.mark.unit
def test_qb_starters_filters_non_qb_and_low_attempts():
    weekly = pd.DataFrame(
        [
            {
                "season": _SEASON,
                "week": 1,
                "recent_team": "TM",
                "season_type": "REG",
                "position": "QB",
                "player_id": "starter",
                "attempts": 30,
            },
            {
                "season": _SEASON,
                "week": 1,
                "recent_team": "TM",
                "season_type": "REG",
                "position": "QB",
                "player_id": "mop_up",
                "attempts": 4,
            },  # below min_attempts
            {
                "season": _SEASON,
                "week": 1,
                "recent_team": "TM",
                "season_type": "REG",
                "position": "RB",
                "player_id": "rb",
                "attempts": 99,
            },  # not a QB
        ]
    )
    out = ada.qb_starters(weekly, min_attempts=10)
    assert list(out["starter_id"]) == ["starter"]
