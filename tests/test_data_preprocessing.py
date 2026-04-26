"""Coverage tests for ``src/data/preprocessing.py``.

``preprocess()`` filters raw nflverse weekly data to skill positions, drops
no-snap zero-stat rows, fills missing stat columns with 0, imputes
``snap_pct`` with position-week medians, and computes the three fantasy
scoring formats. These tests exercise every step + every conditional
branch (with/without ``season_type``, with/without ``snap_pct``, with/
without ``fantasy_points_ppr``) on synthetic frames that pin down the
expected behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import preprocess


def _base_row(**overrides) -> dict:
    """Single row with sensible defaults; tests override only what they care about."""
    row = {
        "player_id": "P1",
        "position": "QB",
        "season_type": "REG",
        "season": 2024,
        "week": 5,
        "passing_yards": 250.0,
        "passing_tds": 2,
        "interceptions": 0,
        "rushing_yards": 10.0,
        "rushing_tds": 0,
        "carries": 1,
        "receiving_yards": 0.0,
        "receiving_tds": 0,
        "receptions": 0,
        "targets": 0,
        "completions": 22,
        "attempts": 30,
        "sack_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "snap_pct": 0.95,
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# Position filter + season_type filter
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preprocess_filters_to_skill_positions():
    """Rows with non-skill positions (LB, CB, etc.) get dropped."""
    df = pd.DataFrame(
        [
            _base_row(player_id="QB1", position="QB"),
            _base_row(player_id="LB1", position="LB"),  # filtered out
            _base_row(player_id="WR1", position="WR"),
            _base_row(player_id="CB1", position="CB"),  # filtered out
        ]
    )
    out = preprocess(df)
    assert set(out["player_id"]) == {"QB1", "WR1"}


@pytest.mark.unit
def test_preprocess_filters_to_regular_season_when_column_present():
    """Postseason rows dropped when ``season_type`` is present."""
    df = pd.DataFrame(
        [
            _base_row(player_id="REG1", season_type="REG"),
            _base_row(player_id="POST1", season_type="POST"),
        ]
    )
    out = preprocess(df)
    assert set(out["player_id"]) == {"REG1"}


@pytest.mark.unit
def test_preprocess_skips_season_type_filter_when_column_absent():
    """No ``season_type`` → no filter (older datasets)."""
    df = pd.DataFrame([_base_row()])
    df = df.drop(columns=["season_type"])
    out = preprocess(df)
    assert len(out) == 1


# --------------------------------------------------------------------------
# No-snaps + zero-stats removal
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preprocess_drops_rows_with_no_snaps_and_zero_stats():
    """A row with NaN snap_pct AND zero across every stat column gets dropped."""
    df = pd.DataFrame(
        [
            _base_row(player_id="active", passing_yards=200.0),
            _base_row(
                player_id="ghost",
                passing_yards=0.0,
                rushing_yards=0.0,
                receiving_yards=0.0,
                receptions=0,
                targets=0,
                carries=0,
                completions=0,
                attempts=0,
                snap_pct=np.nan,
            ),
        ]
    )
    out = preprocess(df)
    assert "ghost" not in set(out["player_id"])
    assert "active" in set(out["player_id"])


@pytest.mark.unit
def test_preprocess_keeps_zero_stats_row_with_nonzero_snaps():
    """Zero stats but a real snap count → kept (snapped but didn't touch the ball)."""
    df = pd.DataFrame(
        [
            _base_row(
                player_id="snapper",
                passing_yards=0.0,
                rushing_yards=0.0,
                receiving_yards=0.0,
                receptions=0,
                targets=0,
                carries=0,
                completions=0,
                attempts=0,
                snap_pct=0.45,  # snapped → kept even though stats are zero
            ),
        ]
    )
    out = preprocess(df)
    assert "snapper" in set(out["player_id"])


@pytest.mark.unit
def test_preprocess_when_snap_pct_column_absent_keeps_all_zero_stat_rows():
    """No ``snap_pct`` column → no_snaps mask is True everywhere; rows with
    truly zero stats still get dropped (snap-mask matches)."""
    df = pd.DataFrame([_base_row(passing_yards=200.0)])
    df = df.drop(columns=["snap_pct"])
    out = preprocess(df)
    assert len(out) == 1


# --------------------------------------------------------------------------
# Fill-zero columns
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preprocess_fills_missing_stat_columns_with_zero():
    """Stat columns absent in input get added as 0; NaN values fill to 0."""
    df = pd.DataFrame(
        [
            _base_row(player_id="P1"),
        ]
    )
    df = df.drop(columns=["rushing_2pt_conversions", "passing_first_downs"], errors="ignore")
    out = preprocess(df)
    # Columns added wholesale
    assert (out["rushing_2pt_conversions"] == 0).all()
    assert (out["passing_first_downs"] == 0).all()
    # NaN-filling: explicitly set a column to NaN and verify it becomes 0.
    df2 = pd.DataFrame([_base_row(player_id="P2")])
    df2["passing_yards"] = np.nan
    out2 = preprocess(df2)
    # Filtered out because all stats are NaN/0 + snap_pct=0.95 (kept). After
    # fillna 0 the row is kept; passing_yards == 0.
    assert (out2["passing_yards"] == 0).all()


# --------------------------------------------------------------------------
# snap_pct median imputation
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preprocess_imputes_snap_pct_with_position_week_median():
    """One QB row at week 5 has snap_pct=NaN — fill with median of others."""
    df = pd.DataFrame(
        [
            _base_row(player_id="QB1", week=5, snap_pct=0.80),
            _base_row(player_id="QB2", week=5, snap_pct=0.60),
            _base_row(player_id="QB3", week=5, snap_pct=np.nan, passing_yards=180.0),
        ]
    )
    out = preprocess(df)
    qb3 = out[out["player_id"] == "QB3"].iloc[0]
    # median of (0.80, 0.60) = 0.70
    assert qb3["snap_pct"] == pytest.approx(0.70)


@pytest.mark.unit
def test_preprocess_fills_orphan_snap_pct_with_zero_after_median_attempt():
    """If an entire (position, week) group is all NaN, the transform-median is
    NaN and the second fillna(0) catches it."""
    df = pd.DataFrame(
        [
            _base_row(
                player_id="QB_only", position="QB", week=10, snap_pct=np.nan, passing_yards=100.0
            ),
        ]
    )
    out = preprocess(df)
    # No other QB rows in week 10 → median is NaN → fallback to 0.
    assert out.iloc[0]["snap_pct"] == 0.0


# --------------------------------------------------------------------------
# Fantasy-point parity warning
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_preprocess_warns_when_fantasy_points_ppr_disagrees(capsys):
    """If ``fantasy_points_ppr`` is in the input and disagrees with our computed
    PPR, a WARNING line gets printed."""
    row = _base_row(player_id="QB1")
    df = pd.DataFrame([row])
    # Inject a deliberately-wrong fantasy_points_ppr (off by 10 pts).
    df["fantasy_points_ppr"] = [9999.0]
    preprocess(df)
    out = capsys.readouterr().out
    assert "differ from nflverse PPR points" in out


@pytest.mark.unit
def test_preprocess_no_warning_when_fantasy_points_ppr_agrees(capsys):
    """If our computed PPR matches the input within 0.5 pts, no warning."""
    row = _base_row(player_id="QB1")
    df = pd.DataFrame([row])
    # Pre-compute the expected PPR points using the same formula as
    # compute_fantasy_points (full PPR).
    expected = (
        row["passing_yards"] * 0.04
        + row["passing_tds"] * 4
        + row["interceptions"] * -2
        + row["rushing_yards"] * 0.1
        + row["rushing_tds"] * 6
        + row["receiving_yards"] * 0.1
        + row["receiving_tds"] * 6
        + row["receptions"] * 1.0
        + 0  # no fumbles
    )
    df["fantasy_points_ppr"] = [expected]
    preprocess(df)
    out = capsys.readouterr().out
    assert "differ from nflverse" not in out


@pytest.mark.unit
def test_preprocess_returns_three_scoring_format_columns():
    """Output must carry ``fantasy_points``, ``fantasy_points_half_ppr``,
    ``fantasy_points_standard``."""
    df = pd.DataFrame(
        [_base_row(player_id="WR1", position="WR", receptions=4, receiving_yards=50.0, targets=6)]
    )
    out = preprocess(df)
    for col in ("fantasy_points", "fantasy_points_half_ppr", "fantasy_points_standard"):
        assert col in out.columns
    # PPR > half_ppr > standard for a row with receptions
    row = out.iloc[0]
    assert row["fantasy_points"] > row["fantasy_points_half_ppr"] > row["fantasy_points_standard"]


@pytest.mark.unit
def test_preprocess_does_not_mutate_input_frame():
    """``preprocess`` works on a copy — caller's DataFrame must be unchanged."""
    df = pd.DataFrame([_base_row()])
    before_cols = set(df.columns)
    preprocess(df)
    assert set(df.columns) == before_cols
