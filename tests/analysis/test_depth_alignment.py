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


# ── CI guard functions (#2 alignment + #3 week-range) ───────────────────────────


def _synthetic_caches(stale: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic (depth, rosters, weekly) caches: 40 teams, weeks 1–5, QB change at week 3.
    Only the actual starter appears in ``weekly`` (the benched QB did NOT play); BOTH QBs
    appear in ``rosters`` (rostered) and ``depth`` (ranked). If ``stale``, the chart's
    rank-1 QB lags the starter by a week — so on transition weeks the chart's rank-1 is the
    *benched* QB, which only the cache-join (via rosters) can recover."""
    depth_rows, roster_rows, weekly_rows = [], [], []
    for t in range(_N_TEAMS):
        team = f"TM{t:02d}"
        a, b = f"A{t:02d}", f"B{t:02d}"

        def true_starter(w: int, a=a, b=b) -> str:
            return a if w <= 2 else b

        for w in range(1, 6):
            starter = true_starter(w)
            chart1 = true_starter(w - 1 if stale else w)  # stale -> prior week's starter
            weekly_rows.append(  # only the actual starter plays
                {
                    "player_id": starter,
                    "season": _SEASON,
                    "week": w,
                    "recent_team": team,
                    "season_type": "REG",
                    "position": "QB",
                    "attempts": 30,
                }
            )
            for qb in (a, b):
                roster_rows.append(  # both QBs rostered (incl. the benched one)
                    {"player_id": qb, "season": _SEASON, "week": w, "team": team, "position": "QB"}
                )
                depth_rows.append(
                    {
                        "gsis_id": qb,
                        "season": _SEASON,
                        "week": w,
                        "formation": "Offense",
                        "depth_team": "1" if qb == chart1 else "2",
                    }
                )
    return pd.DataFrame(depth_rows), pd.DataFrame(roster_rows), pd.DataFrame(weekly_rows)


@pytest.mark.unit
def test_gate_check_alignment_fails_on_stale_passes_on_inertia():
    # Stale fingerprint (prior-dominant transitions) -> gate fails.
    stale = {
        "n_transitions": 40,
        "transition_current": 0.23,
        "transition_prev": 0.71,
        "overall_current": 0.87,
    }
    ok, reason = ada.gate_check_alignment(stale)
    assert not ok and reason.startswith("SHIFTED_BACK_1")

    # Healthy post-fix inertia (current ≈ prior, high overall) -> AMBIGUOUS but OK.
    healthy = {
        "n_transitions": 40,
        "transition_current": 0.50,
        "transition_prev": 0.44,
        "overall_current": 0.91,
    }
    ok, _ = ada.gate_check_alignment(healthy)
    assert ok

    # Same transition shape but overall collapses -> gate fails on the secondary band.
    low_overall = {**healthy, "overall_current": 0.70}
    ok, reason = ada.gate_check_alignment(low_overall)
    assert not ok and "overall" in reason


@pytest.mark.unit
def test_chart_rank1_from_caches_recovers_benched_starter():
    # On a stale transition week the chart's rank-1 is the benched prior starter (A00),
    # who is rostered but did NOT play week 3 — the cache-join must still recover him
    # (the who-played bias that made a splits-based check false-negative).
    depth, rosters, weekly = _synthetic_caches(stale=True)
    chart = ada.chart_rank1_from_caches(depth, rosters)
    wk3 = chart[(chart["week"] == 3) & (chart["team"] == "TM00")]
    assert list(wk3["chart_qb_id"]) == ["A00"]
    assert "A00" not in set(weekly.loc[weekly["week"] == 3, "player_id"])  # A00 benched wk3


@pytest.mark.unit
def test_alignment_from_caches_aligned_vs_stale(tmp_path):
    for stale, expect_ok in [(False, True), (True, False)]:
        depth, rosters, weekly = _synthetic_caches(stale=stale)
        depth.to_parquet(tmp_path / "depth_charts_2099_2099.parquet")
        rosters.to_parquet(tmp_path / "rosters_2099_2099.parquet")
        weekly.to_parquet(tmp_path / "weekly_2099_2099.parquet")
        rates = ada.alignment_from_caches(str(tmp_path))
        ok, reason = ada.gate_check_alignment(rates)
        assert ok is expect_ok
        if stale:  # the PR #595 bug re-introduced — caught even though the prior starter sat
            assert reason.startswith("SHIFTED_BACK_1")


@pytest.mark.unit
def test_alignment_from_caches_missing_cache_returns_none(tmp_path):
    # A missing cache -> None so the CI step logs a loud skip instead of crashing.
    assert ada.alignment_from_caches(str(tmp_path)) is None


def _write_depth_and_schedule(tmp_path, depth_weeks, sched_weeks, season=_SEASON):
    pd.DataFrame(
        {
            "gsis_id": ["x"] * len(depth_weeks),
            "season": [season] * len(depth_weeks),
            "week": depth_weeks,
            "formation": ["Offense"] * len(depth_weeks),
            "depth_team": ["1"] * len(depth_weeks),
        }
    ).to_parquet(tmp_path / f"depth_charts_{season}_{season}.parquet")
    pd.DataFrame(
        {
            "season": [season] * len(sched_weeks),
            "week": sched_weeks,
            "game_type": ["REG"] * len(sched_weeks),
        }
    ).to_parquet(tmp_path / f"schedules_{season}_{season}.parquet")


@pytest.mark.unit
def test_depth_week_range_check(tmp_path):
    # Absent caches -> graceful skip.
    assert ada.depth_week_range_check(str(tmp_path)) == (True, [])

    # In-range depth (weeks 1–18 for an 18-week schedule), incl. an expected week-0 row
    # from the relabel -> clean (upper-bound-only check, no min-week false positive).
    _write_depth_and_schedule(tmp_path, [0] + list(range(1, 19)), list(range(1, 19)))
    ok, offenders = ada.depth_week_range_check(str(tmp_path))
    assert ok and offenders == []

    # Depth overruns the schedule (week 19 for an 18-week season) -> flagged.
    _write_depth_and_schedule(tmp_path, list(range(1, 20)), list(range(1, 19)))
    ok, offenders = ada.depth_week_range_check(str(tmp_path))
    assert not ok and offenders[0]["season"] == _SEASON and offenders[0]["depth_max_week"] == 19
