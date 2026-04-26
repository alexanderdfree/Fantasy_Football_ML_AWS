"""Coverage tests for ``src/data/loader.py``.

Mocks every ``nfl_data_py`` helper + HTTP parquet read so the whole loader
chain runs in-process without network traffic. Covers both the cache-hit
shortcuts (pre-written parquet) and the fresh-fetch branches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    compute_all_scoring_formats,
    compute_fantasy_points,
    load_raw_data,
    load_team_week_stats,
)

# --------------------------------------------------------------------------
# load_team_week_stats
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_load_team_week_stats_cache_hit(tmp_path):
    """Pre-written parquet at the expected path → no network call, just a read."""
    seasons = [2022, 2023]
    cache_path = tmp_path / f"team_stats_{seasons[0]}_{seasons[-1]}.parquet"
    fake = pd.DataFrame({"team": ["KC"], "season": [2022], "week": [1]})
    fake.to_parquet(cache_path)

    out = load_team_week_stats(seasons, cache_dir=str(tmp_path))
    assert len(out) == 1
    assert out.iloc[0]["team"] == "KC"


@pytest.mark.unit
def test_load_team_week_stats_default_seasons_fallback(tmp_path, monkeypatch):
    """``seasons=None`` uses ``SEASONS`` from config. We monkeypatch
    ``pd.read_parquet`` inside the loader so the URL fetches return a stub."""
    import src.data.loader as loader

    def _fake_read_parquet(path, *args, **kwargs):
        # Every URL hit returns a single-row stub.
        return pd.DataFrame({"team": ["X"], "season": [2020], "week": [1]})

    monkeypatch.setattr(loader.pd, "read_parquet", _fake_read_parquet)
    monkeypatch.setattr(loader, "SEASONS", [2020, 2021])

    out = load_team_week_stats(cache_dir=str(tmp_path))
    # Two URL fetches (one per season) → concat with 2 rows.
    assert len(out) == 2


@pytest.mark.unit
def test_load_team_week_stats_network_fetches_and_caches(tmp_path, monkeypatch, capsys):
    """Fresh fetch: 2 of 3 seasons succeed, 1 raises — result merges + caches,
    the failure is logged via WARNING."""
    import src.data.loader as loader

    def _fake_read_parquet(path, *args, **kwargs):
        s = str(path)
        if "2022" in s:
            raise RuntimeError("404 missing")
        return pd.DataFrame(
            {"team": ["KC"], "season": [2021 if "2021" in s else 2023], "week": [1]}
        )

    monkeypatch.setattr(loader.pd, "read_parquet", _fake_read_parquet)

    out = loader.load_team_week_stats([2021, 2022, 2023], cache_dir=str(tmp_path))
    assert len(out) == 2  # 2021 + 2023 survived
    # Cache file must be written
    assert (tmp_path / "team_stats_2021_2023.parquet").exists()
    # Warning printed for the failing season
    assert "team_stats fetch failed for 2022" in capsys.readouterr().out


@pytest.mark.unit
def test_load_team_week_stats_all_fail_returns_empty_and_does_not_cache(tmp_path, monkeypatch):
    """Every season 404s → return empty DF, but do NOT poison the cache."""
    import src.data.loader as loader

    def _always_404(path, *args, **kwargs):
        raise RuntimeError("404 everywhere")

    monkeypatch.setattr(loader.pd, "read_parquet", _always_404)

    out = loader.load_team_week_stats([2020], cache_dir=str(tmp_path))
    assert out.empty
    # No cache file written.
    assert not (tmp_path / "team_stats_2020_2020.parquet").exists()


# --------------------------------------------------------------------------
# load_raw_data — full chain with every nfl_data_py helper mocked
# --------------------------------------------------------------------------


def _mock_all_nfl_helpers(monkeypatch):
    """Stub every nfl.* helper load_raw_data calls. Returns the expected
    DataFrame shapes."""
    import src.data.loader as loader

    # weekly_data: old-style columns
    def _fake_weekly(seasons):
        rows = []
        for s in seasons:
            for pid in range(3):
                rows.append(
                    {
                        "player_id": f"P{pid:02d}",
                        "season": s,
                        "week": 1,
                        "position": "QB",
                        "recent_team": "KC",
                    }
                )
        return pd.DataFrame(rows)

    # rosters: player_id/season/position/plus an extra object col with mixed dtype
    def _fake_rosters(seasons):
        return pd.DataFrame(
            {
                "player_id": [f"P{i:02d}" for i in range(3)],
                "season": [seasons[0]] * 3,
                "position": ["QB", "WR", "RB"],
                "jersey_number": ["12", "88", "21"],  # object dtype
            }
        )

    def _fake_schedules(seasons):
        return pd.DataFrame(
            {"season": seasons, "week": [1] * len(seasons), "home_team": ["KC"] * len(seasons)}
        )

    def _fake_snap_counts(seasons):
        return pd.DataFrame(
            {
                "pfr_player_id": ["pfr1"],
                "season": [seasons[0]],
                "week": [1],
                "offense_pct": [0.95],
            }
        )

    def _fake_ids():
        return pd.DataFrame(
            {
                "pfr_id": ["pfr1"],
                "gsis_id": ["P00"],
            }
        )

    def _fake_injuries(seasons):
        return pd.DataFrame(
            {
                "gsis_id": ["P00"],
                "season": [seasons[0]],
                "week": [1],
                "practice_status": ["Full Participation in Practice"],
                "report_status": ["Questionable"],
            }
        )

    def _fake_depth_charts(seasons):
        return pd.DataFrame(
            {
                "gsis_id": ["P00"] * 2,
                "season": [seasons[0]] * 2,
                "week": [1] * 2,
                "formation": ["Offense", "Defense"],
                "depth_team": ["1", "1"],
            }
        )

    monkeypatch.setattr(loader.nfl, "import_weekly_data", _fake_weekly)
    monkeypatch.setattr(loader.nfl, "import_seasonal_rosters", _fake_rosters)
    monkeypatch.setattr(loader.nfl, "import_schedules", _fake_schedules)
    monkeypatch.setattr(loader.nfl, "import_snap_counts", _fake_snap_counts)
    monkeypatch.setattr(loader.nfl, "import_ids", _fake_ids)
    monkeypatch.setattr(loader.nfl, "import_injuries", _fake_injuries)
    monkeypatch.setattr(loader.nfl, "import_depth_charts", _fake_depth_charts)


@pytest.mark.unit
def test_load_raw_data_fresh_fetch_old_seasons_only(tmp_path, monkeypatch):
    """Happy path: old-style seasons (≤2024) → nfl.import_weekly_data only,
    no 2025+ URL branch. All six caches get written."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    df = loader.load_raw_data([2022, 2023], cache_dir=str(tmp_path))
    # Merge chain must end with these enrichment columns:
    assert "snap_pct" in df.columns
    assert "practice_status" in df.columns
    assert "game_status" in df.columns
    assert "depth_chart_rank" in df.columns
    # Schedules attached as attrs metadata.
    assert "schedules" in df.attrs
    # Every parquet cache exists.
    for name in ("weekly", "rosters", "schedules", "snap_counts", "injuries", "depth_charts"):
        assert (tmp_path / f"{name}_2022_2023.parquet").exists()


@pytest.mark.unit
def test_load_raw_data_cache_hit_short_circuit(tmp_path, monkeypatch):
    """Pre-written caches → the loader skips every nfl.* call and reads from disk."""
    import src.data.loader as loader

    def _boom(*a, **k):
        raise AssertionError("nfl_data_py was called despite cache hit")

    # Stub everything to scream if called.
    for name in (
        "import_weekly_data",
        "import_seasonal_rosters",
        "import_schedules",
        "import_snap_counts",
        "import_injuries",
        "import_depth_charts",
    ):
        monkeypatch.setattr(loader.nfl, name, _boom)
    # import_ids is called inside the snap-merge try/except; it's ok for it to fire.
    monkeypatch.setattr(
        loader.nfl,
        "import_ids",
        lambda: pd.DataFrame({"pfr_id": ["pfr1"], "gsis_id": ["P00"]}),
    )

    seasons = [2022, 2023]
    # Pre-write every cache.
    for name, df in [
        (
            "weekly",
            pd.DataFrame(
                {
                    "player_id": ["P00"],
                    "season": [2022],
                    "week": [1],
                    "position": ["QB"],
                    "recent_team": ["KC"],
                }
            ),
        ),
        (
            "rosters",
            pd.DataFrame({"player_id": ["P00"], "season": [2022], "position": ["QB"]}),
        ),
        (
            "schedules",
            pd.DataFrame({"season": [2022], "week": [1], "home_team": ["KC"]}),
        ),
        (
            "snap_counts",
            pd.DataFrame(
                {
                    "pfr_player_id": ["pfr1"],
                    "season": [2022],
                    "week": [1],
                    "offense_pct": [0.9],
                }
            ),
        ),
        (
            "injuries",
            pd.DataFrame(
                {
                    "gsis_id": ["P00"],
                    "season": [2022],
                    "week": [1],
                    "practice_status": ["Full Participation in Practice"],
                    "report_status": ["Questionable"],
                }
            ),
        ),
        (
            "depth_charts",
            pd.DataFrame(
                {
                    "gsis_id": ["P00"],
                    "season": [2022],
                    "week": [1],
                    "formation": ["Offense"],
                    "depth_team": ["1"],
                }
            ),
        ),
    ]:
        df.to_parquet(tmp_path / f"{name}_{seasons[0]}_{seasons[-1]}.parquet")

    out = loader.load_raw_data(seasons, cache_dir=str(tmp_path))
    # Enrichment columns still land from the merge path.
    assert "snap_pct" in out.columns
    assert "depth_chart_rank" in out.columns


@pytest.mark.unit
def test_load_raw_data_snap_merge_exception_falls_back_to_nan(tmp_path, monkeypatch, capsys):
    """If import_ids raises, snap_pct defaults to NaN (except-branch coverage)."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    def _bad_ids():
        raise RuntimeError("nflverse id-map missing")

    monkeypatch.setattr(loader.nfl, "import_ids", _bad_ids)

    df = loader.load_raw_data([2023], cache_dir=str(tmp_path))
    assert "snap_pct" in df.columns
    assert df["snap_pct"].isna().all()
    assert "Snap count merge failed" in capsys.readouterr().out


@pytest.mark.unit
def test_load_raw_data_new_season_url_branch(tmp_path, monkeypatch):
    """Seasons ≥ 2025 go through the nflverse release URL + column rename."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    # For 2025 the loader does pd.read_parquet(url). Intercept.
    def _fake_url_read_parquet(path, *args, **kwargs):
        if isinstance(path, str) and "stats_player_week_2025" in path:
            return pd.DataFrame(
                {
                    "player_id": ["P00"],
                    "season": [2025],
                    "week": [1],
                    "position": ["QB"],
                    "team": ["KC"],  # will be renamed to recent_team
                    "passing_interceptions": [1],  # → interceptions
                    "sacks_suffered": [2],  # → sacks
                    "sack_yards_lost": [14],  # → sack_yards
                }
            )
        # Otherwise delegate to the real pd.read_parquet (cache reads).
        return pd.read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(loader.pd, "read_parquet", _fake_url_read_parquet)

    df = loader.load_raw_data([2024, 2025], cache_dir=str(tmp_path))
    # 2024 from import_weekly_data + 2025 from the URL branch
    assert 2025 in df["season"].values
    # Renamed columns must land.
    assert "recent_team" in df.columns
    assert "interceptions" in df.columns


@pytest.mark.unit
def test_load_raw_data_no_snap_seasons_returns_empty_snap_df(tmp_path, monkeypatch):
    """When seasons are all < 2012, snap_counts ends up empty — the loader
    must still finish (empty-frame merge branch)."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)
    # Force import_snap_counts NOT to be called by picking pre-2012 seasons.

    # Override weekly to produce pre-2012 rows (simulate old nfl data).
    monkeypatch.setattr(
        loader.nfl,
        "import_weekly_data",
        lambda seasons: pd.DataFrame(
            {
                "player_id": ["P00"],
                "season": [seasons[0]],
                "week": [1],
                "position": ["QB"],
                "recent_team": ["KC"],
            }
        ),
    )

    df = loader.load_raw_data([2010, 2011], cache_dir=str(tmp_path))
    assert "snap_pct" in df.columns


# --------------------------------------------------------------------------
# compute_fantasy_points / compute_all_scoring_formats
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_fantasy_points_default_scoring():
    """No scoring dict → uses SCORING (full PPR)."""
    df = pd.DataFrame(
        {
            "passing_yards": [300.0],
            "passing_tds": [2],
            "interceptions": [1],
            "rushing_yards": [20.0],
            "rushing_tds": [0],
            "receptions": [0],
            "receiving_yards": [0.0],
            "receiving_tds": [0],
            "sack_fumbles_lost": [1],
            "rushing_fumbles_lost": [0],
            "receiving_fumbles_lost": [0],
        }
    )
    pts = compute_fantasy_points(df)
    # 300*0.04 + 2*4 + 1*-2 + 20*0.1 + 0 + 1*-2 = 12 + 8 - 2 + 2 - 2 = 18
    assert pts.iloc[0] == pytest.approx(18.0)


@pytest.mark.unit
def test_compute_all_scoring_formats_adds_three_columns():
    df = pd.DataFrame(
        {
            "passing_yards": [0.0],
            "passing_tds": [0],
            "interceptions": [0],
            "rushing_yards": [0.0],
            "rushing_tds": [0],
            "receptions": [3],
            "receiving_yards": [30.0],
            "receiving_tds": [0],
            "sack_fumbles_lost": [0],
            "rushing_fumbles_lost": [0],
            "receiving_fumbles_lost": [0],
        }
    )
    out = compute_all_scoring_formats(df)
    assert "fantasy_points_standard" in out.columns
    assert "fantasy_points_half_ppr" in out.columns
    assert "fantasy_points" in out.columns
    # PPR > half_PPR > standard because of the 3 receptions.
    assert out["fantasy_points"].iloc[0] > out["fantasy_points_half_ppr"].iloc[0]
    assert out["fantasy_points_half_ppr"].iloc[0] > out["fantasy_points_standard"].iloc[0]
