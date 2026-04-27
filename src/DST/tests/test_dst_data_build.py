"""Coverage tests for ``DST/dst_data.py::build_dst_data``.

``build_dst_data`` reads three inputs — weekly player stats, schedules,
and team-week stats — and performs the bulk of the 400-line merge/
aggregation logic that powered this file's 6% Codecov number. This
file exercises the full pipe with synthetic parquet fixtures and a
stubbed ``load_team_week_stats`` so the nflverse fetch is never touched.

What it deliberately does NOT check: the exact numerical values. The
goal is branch coverage (every merge, every fillna, every rolling
window), not a regression test on a fabricated dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Synthetic fixtures --------------------------------------------------

_TEAMS = ["BUF", "KC", "SF", "DAL"]
_SEASONS = [2024]
_WEEKS = list(range(1, 4))  # 3 weeks — enough for L5 rolling shift to populate.


def _make_weekly(seed: int = 0) -> pd.DataFrame:
    """Tiny weekly player-stats frame with just the columns build_dst_data reads."""
    rng = np.random.default_rng(seed)
    rows = []
    # One QB + one non-QB per team per week; cycle opponents so every team sees
    # every other team at least once per season.
    for s in _SEASONS:
        for w in _WEEKS:
            for i, t in enumerate(_TEAMS):
                opp = _TEAMS[(i + w) % len(_TEAMS)]
                for pos, pid in [("QB", f"{t}_QB"), ("WR", f"{t}_WR")]:
                    rows.append(
                        {
                            "player_id": pid,
                            "position": pos,
                            "recent_team": t,
                            "opponent_team": opp,
                            "season": s,
                            "week": w,
                            "sacks": int(rng.integers(0, 3)),
                            "interceptions": int(rng.integers(0, 2)),
                            "sack_fumbles_lost": 0,
                            "rushing_fumbles_lost": 0,
                            "receiving_fumbles_lost": 0,
                            "special_teams_tds": 0,
                            "passing_epa": float(rng.uniform(-5, 5)) if pos == "QB" else 0.0,
                            "attempts": int(rng.integers(10, 40)) if pos == "QB" else 0,
                            "rushing_yards": float(rng.integers(0, 80)),
                        }
                    )
    return pd.DataFrame(rows)


def _make_schedules() -> pd.DataFrame:
    """Schedules with the cols build_dst_data indexes into, in REG game_type."""
    rows = []
    for s in _SEASONS:
        for w in _WEEKS:
            for i in range(0, len(_TEAMS), 2):
                home, away = _TEAMS[i], _TEAMS[(i + 1) % len(_TEAMS)]
                rows.append(
                    {
                        "season": s,
                        "week": w,
                        "home_team": home,
                        "away_team": away,
                        "home_score": 24,
                        "away_score": 17,
                        "spread_line": -3.5,
                        "total_line": 44.5,
                        "home_rest": 7,
                        "away_rest": 7,
                        "div_game": 0,
                        "roof": "outdoors" if w % 2 == 0 else "dome",
                        "game_type": "REG",
                    }
                )
    return pd.DataFrame(rows)


def _make_team_stats(seed: int = 1) -> pd.DataFrame:
    """Team-week stats with the defensive + offensive cols that feed build_dst_data."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in _SEASONS:
        for w in _WEEKS:
            for t in _TEAMS:
                rows.append(
                    {
                        "team": t,
                        "season": s,
                        "week": w,
                        "def_tds": int(rng.integers(0, 2)),
                        "def_safeties": 0,
                        "def_fumbles_forced": int(rng.integers(0, 3)),
                        "passing_yards": float(rng.integers(150, 400)),
                        "rushing_yards": float(rng.integers(50, 180)),
                        "fg_blocked": 0,
                        "pat_blocked": 0,
                    }
                )
    return pd.DataFrame(rows)


# --- Fixture wrapper ----------------------------------------------------


@pytest.fixture()
def synthetic_parquets(tmp_path, monkeypatch):
    """Write the three parquets build_dst_data expects + patch CACHE_DIR/SEASONS.

    Also stubs ``load_team_week_stats`` + ``nfl.import_team_desc`` so no
    network traffic happens. Returns the tmp cache directory for debugging.
    """
    import src.DST.dst_data as dst_data

    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()

    weekly = _make_weekly()
    schedules = _make_schedules()
    team_stats = _make_team_stats()

    weekly.to_parquet(cache_dir / f"weekly_{_SEASONS[0]}_{_SEASONS[-1]}.parquet")
    schedules.to_parquet(cache_dir / f"schedules_{_SEASONS[0]}_{_SEASONS[-1]}.parquet")

    monkeypatch.setattr(dst_data, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(dst_data, "SEASONS", _SEASONS)
    monkeypatch.setattr(dst_data, "load_team_week_stats", lambda seasons: team_stats)

    # Stub nfl_data_py import_team_desc so the logo lookup branch runs without
    # a network call. The except-Exception fallback branch is covered by a
    # separate test that forces import_team_desc to raise.
    fake_team_desc = pd.DataFrame(
        {"team_abbr": _TEAMS, "team_logo_espn": [f"https://logo/{t}.png" for t in _TEAMS]}
    )
    monkeypatch.setattr(dst_data.nfl, "import_team_desc", lambda: fake_team_desc)

    return cache_dir


# --- Tests --------------------------------------------------------------


@pytest.mark.unit
def test_build_dst_data_rows_per_team_week(synthetic_parquets):
    """Every (season, week, team) combo in schedules must yield one DST row."""
    from src.DST.dst_data import build_dst_data

    df = build_dst_data()
    # 4 teams × 3 weeks × 1 season = 12 rows
    assert len(df) == len(_TEAMS) * len(_WEEKS) * len(_SEASONS)
    assert set(df["team"].unique()) == set(_TEAMS)
    assert set(df["season"].unique()) == set(_SEASONS)


@pytest.mark.unit
def test_build_dst_data_schema(synthetic_parquets):
    """Output schema must include derived defensive + context columns."""
    from src.DST.dst_data import build_dst_data

    df = build_dst_data()
    required = {
        "points_allowed",
        "spread_line",
        "total_line",
        "rest_days",
        "div_game",
        "is_home",
        "is_dome",
        "opponent_team",
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_tds",
        "def_safeties",
        "def_fumbles_forced",
        "def_blocked_kicks",
        "yards_allowed",
        "special_teams_tds",
        "opp_scoring_L5",
        "opp_scoring_L3",
        "opp_turnovers_L5",
        "opp_sacks_allowed_L5",
        "opp_qb_epa_L5",
        "opp_qb_int_rate_L5",
        "opp_qb_sack_rate_L5",
        "opp_qb_rush_yds_L5",
        "opp_scoring",
        "opp_fumbles",
        "opp_interceptions",
        "opp_qb_epa",
        "player_id",
        "player_display_name",
        "player_name",
        "position",
        "recent_team",
        "headshot_url",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing derived columns: {sorted(missing)}"


@pytest.mark.unit
def test_build_dst_data_no_nans_in_fills(synthetic_parquets):
    """All columns in the fillna block must be NaN-free post build."""
    from src.DST.dst_data import build_dst_data

    df = build_dst_data()
    fill_cols = [
        "def_sacks",
        "def_ints",
        "def_fumble_rec",
        "def_tds",
        "def_safeties",
        "def_fumbles_forced",
        "def_blocked_kicks",
        "special_teams_tds",
        "yards_allowed",
        "spread_line",
        "total_line",
        "is_home",
        "rest_days",
        "div_game",
        "is_dome",
        "opp_scoring_L5",
        "opp_scoring_L3",
        "opp_turnovers_L5",
        "opp_sacks_allowed_L5",
        "opp_qb_epa_L5",
        "opp_qb_int_rate_L5",
        "opp_qb_sack_rate_L5",
        "opp_qb_rush_yds_L5",
        "opp_scoring",
        "opp_fumbles",
        "opp_interceptions",
        "opp_qb_epa",
    ]
    for col in fill_cols:
        assert df[col].notna().all(), f"{col} still has NaN after build"


@pytest.mark.unit
def test_build_dst_data_is_dome_maps_roof(synthetic_parquets):
    """Roof = 'dome' must set is_dome=1, 'outdoors' must set is_dome=0.

    Our synthetic schedule alternates roof per week, so both values show up.
    """
    from src.DST.dst_data import build_dst_data

    df = build_dst_data()
    assert df["is_dome"].isin([0, 1]).all()
    assert df["is_dome"].sum() > 0  # at least one dome row
    assert (df["is_dome"] == 0).sum() > 0  # and at least one outdoors row


@pytest.mark.unit
def test_build_dst_data_logo_fallback_on_nfl_error(synthetic_parquets, monkeypatch):
    """If ``nfl.import_team_desc`` raises, headshot_url defaults to empty str
    (covers the ``except Exception`` branch at the bottom of build_dst_data)."""
    import src.DST.dst_data as dst_data

    def _boom():
        raise RuntimeError("nflverse down")

    monkeypatch.setattr(dst_data.nfl, "import_team_desc", _boom)

    df = dst_data.build_dst_data()
    assert (df["headshot_url"] == "").all()


@pytest.mark.unit
def test_filter_to_dst_is_identity():
    """``filter_to_dst`` must return a copy equal to the input frame."""
    from src.DST.dst_data import filter_to_dst

    df = pd.DataFrame({"team": ["KC", "BUF"], "season": [2024, 2024], "week": [1, 1]})
    out = filter_to_dst(df)
    pd.testing.assert_frame_equal(out, df)
    # Copy, not the same reference
    assert out is not df
