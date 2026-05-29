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
    _is_espn_offense,
    _normalize_espn_depth,
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

    # Capture the real read_parquet for cache round-trip verification — the
    # monkeypatch below replaces loader.pd.read_parquet, which is the same
    # object as pd.read_parquet, so the test itself can't call the real one
    # by name.
    real_read_parquet = pd.read_parquet

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
    assert set(out["season"].tolist()) == {2021, 2023}
    # Cache file must be written and round-trip the merged frame.
    cache_path = tmp_path / "team_stats_2021_2023.parquet"
    assert cache_path.exists()
    cached = real_read_parquet(cache_path)
    assert len(cached) == 2
    assert set(cached["season"].tolist()) == {2021, 2023}
    # Warning printed for the failing season
    assert "team_stats fetch failed for 2022" in capsys.readouterr().out


@pytest.mark.unit
def test_load_team_week_stats_all_fail_returns_empty_and_does_not_cache(
    tmp_path, monkeypatch, capsys
):
    """Every season 404s → return empty DF, but do NOT poison the cache."""
    import src.data.loader as loader

    def _always_404(path, *args, **kwargs):
        raise RuntimeError("404 everywhere")

    monkeypatch.setattr(loader.pd, "read_parquet", _always_404)

    out = loader.load_team_week_stats([2020], cache_dir=str(tmp_path))
    assert out.empty
    # No cache file written.
    assert not (tmp_path / "team_stats_2020_2020.parquet").exists()
    # The failing season was logged so an operator can see why the result is empty.
    assert "team_stats fetch failed for 2020" in capsys.readouterr().out


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

    # Empty-but-typed PBP frame so reconstruct_redzone_from_pbp (newly invoked
    # by load_raw_data) doesn't hit the network. Aggregator references these
    # columns by name; an empty frame with the right schema yields an empty
    # red-zone result, which the loader gracefully treats as "all zeros".
    def _fake_pbp(seasons, cols):
        return pd.DataFrame(
            {
                "season": pd.Series([], dtype="int64"),
                "season_type": pd.Series([], dtype="object"),
                "week": pd.Series([], dtype="int64"),
                "posteam": pd.Series([], dtype="object"),
                "rusher_player_id": pd.Series([], dtype="object"),
                "receiver_player_id": pd.Series([], dtype="object"),
                "pass_attempt": pd.Series([], dtype="int64"),
                "yardline_100": pd.Series([], dtype="float64"),
            }
        )

    monkeypatch.setattr(loader.nfl_source, "weekly_data", _fake_weekly)
    monkeypatch.setattr(loader.nfl_source, "rosters", _fake_rosters)
    monkeypatch.setattr(loader.nfl_source, "schedules", _fake_schedules)
    monkeypatch.setattr(loader.nfl_source, "snap_counts", _fake_snap_counts)
    monkeypatch.setattr(loader.nfl_source, "player_ids", _fake_ids)
    monkeypatch.setattr(loader.nfl_source, "injuries", _fake_injuries)
    monkeypatch.setattr(loader.nfl_source, "depth_charts", _fake_depth_charts)
    # reconstruct_redzone_from_pbp lives in src.data.redzone_pbp and pulls
    # nfl_data_py via its own module-level binding, so stub it there too.
    import src.data.redzone_pbp as redzone_pbp

    monkeypatch.setattr(redzone_pbp.nfl_source, "pbp_data", _fake_pbp)


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
    # Every parquet cache exists. Schedules are persisted to disk for
    # downstream consumers (src/k/data.py::load_data,
    # src/shared/weather_features.py::_load_schedules) — not attached via
    # df.attrs, which doesn't survive to_parquet.
    # Every parquet cache exists.
    for name in ("weekly", "rosters", "schedules", "snap_counts", "injuries", "depth_charts"):
        assert (tmp_path / f"{name}_2022_2023.parquet").exists()


@pytest.mark.unit
@pytest.mark.parametrize("depth_order", [["1", "3"], ["3", "1"]])
def test_load_raw_data_depth_chart_rank_picks_min_deterministically(
    tmp_path, monkeypatch, depth_order
):
    """When a player has multiple Offense-formation rows in the same week with
    different ``depth_team`` values, the merged ``depth_chart_rank`` must be
    the minimum (best rank) regardless of input row order — guards against
    the ``agg('last')`` non-determinism the previous loader had."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    # Override depth charts: two Offense rows for the same player-week with
    # different depth_team values. The parametrize swaps row order so we
    # exercise both possible "last" answers under the old code.
    def _two_row_depth_charts(seasons):
        return pd.DataFrame(
            {
                "gsis_id": ["P00", "P00"],
                "season": [seasons[0]] * 2,
                "week": [1] * 2,
                "formation": ["Offense", "Offense"],
                "depth_team": depth_order,
            }
        )

    monkeypatch.setattr(loader.nfl_source, "depth_charts", _two_row_depth_charts)

    df = loader.load_raw_data([2022, 2023], cache_dir=str(tmp_path))
    p00_row = df[df["player_id"] == "P00"].iloc[0]
    assert p00_row["depth_chart_rank"] == 1.0, (
        f"depth_chart_rank should be 1 (best rank held that week) regardless "
        f"of input row order; got {p00_row['depth_chart_rank']} for input "
        f"order {depth_order}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "pos_grp,expected",
    [
        ("3WR 1TE", True),
        ("2WR 2TE", True),  # unseen offensive personnel label — must survive
        ("Empty", True),
        ("Base 4-3 D", False),
        ("Base 3-4 D", False),
        ("Special Teams", False),
    ],
)
def test_is_espn_offense_selects_offense_negatively(pos_grp, expected):
    """_is_espn_offense excludes defensive fronts (' D') and 'Special Teams';
    everything else (including unseen offensive personnel labels) is offense."""
    mask = _is_espn_offense(pd.Series([pos_grp]))
    assert bool(mask.iloc[0]) is expected


@pytest.mark.unit
def test_normalize_espn_depth_asof_maps_snapshots_to_weeks():
    """The daily ESPN feed (no week column) collapses to one row per
    (player, week) using the latest snapshot at/before that week's kickoff:
    a preseason snapshot feeds week 1; a mid-season snapshot supersedes it for
    later weeks. Defensive rows are dropped; depth_team is the str rank."""
    espn = pd.DataFrame(
        {
            "dt": [
                "2025-08-20T10:00:00Z",  # preseason → latest ≤ wk1 kickoff
                "2025-09-10T10:00:00Z",  # between wk1 and wk2 → applies to wk2
                "2025-08-20T10:00:00Z",  # defense — must be excluded
            ],
            "team": ["KC", "KC", "KC"],
            "gsis_id": ["P00", "P00", "D99"],
            "pos_grp": ["3WR 1TE", "3WR 1TE", "Base 4-3 D"],
            "pos_slot": [9, 9, 1],
            "pos_rank": [2, 1, 1],
        }
    )
    schedules = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "gameday": ["2025-09-07", "2025-09-14"],
            "home_team": ["KC", "KC"],
            "away_team": ["LV", "DEN"],
        }
    )

    out = _normalize_espn_depth(espn, schedules, 2025).sort_values("week")

    assert list(out.columns) == ["gsis_id", "season", "week", "formation", "depth_team"]
    assert "D99" not in out["gsis_id"].values  # defense excluded
    assert out[out["week"] == 1].iloc[0]["depth_team"] == "2"  # preseason rank carries to wk1
    assert out[out["week"] == 2].iloc[0]["depth_team"] == "1"  # 09-10 snapshot applies to wk2
    assert (out["formation"] == "Offense").all()
    # str rank survives the legacy pd.to_numeric coercion the loader merge applies.
    assert pd.to_numeric(out["depth_team"]).tolist() == [2, 1]


@pytest.mark.unit
def test_normalize_espn_depth_empty_when_schedule_incomplete():
    """If the schedule lacks columns the as-of join needs, normalization returns
    an empty (but correctly-typed) frame so the loader's -1 sentinel +
    consumer-side impute cover the gap rather than crashing."""
    espn = pd.DataFrame(
        {
            "dt": ["2025-09-04T10:00:00Z"],
            "team": ["KC"],
            "gsis_id": ["P00"],
            "pos_grp": ["3WR 1TE"],
            "pos_slot": [9],
            "pos_rank": [1],
        }
    )
    bad_sched = pd.DataFrame({"season": [2025], "week": [1], "home_team": ["KC"]})
    out = _normalize_espn_depth(espn, bad_sched, 2025)
    assert out.empty
    assert list(out.columns) == ["gsis_id", "season", "week", "formation", "depth_team"]


@pytest.mark.unit
def test_load_raw_data_espn_depth_lands_real_rank(tmp_path, monkeypatch):
    """End-to-end: a 2025 season fetches the ESPN depth feed via URL, as-of
    joins it onto the schedule, and lands a REAL depth_chart_rank (not the -1
    sentinel) on the merged frame — the data the -1 consumer impute stood in for."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    # Complete schedule so the as-of join produces rows (the shared mock's
    # _fake_schedules omits gameday/game_type).
    def _full_schedules(seasons):
        return pd.DataFrame(
            {
                "season": [2025],
                "week": [1],
                "game_type": ["REG"],
                "gameday": ["2025-09-07"],
                "home_team": ["KC"],
                "away_team": ["LV"],
            }
        )

    monkeypatch.setattr(loader.nfl_source, "schedules", _full_schedules)

    url_calls: list[str] = []

    def _fake_url_read_parquet(path, *args, **kwargs):
        s = str(path)
        if "stats_player_week_2025" in s:
            url_calls.append(s)
            return pd.DataFrame(
                {
                    "player_id": ["P00"],
                    "season": [2025],
                    "week": [1],
                    "position": ["QB"],
                    "team": ["KC"],
                }
            )
        if "depth_charts_2025" in s:
            url_calls.append(s)
            return pd.DataFrame(
                {
                    "dt": ["2025-09-04T10:00:00Z"],  # ≤ wk1 kickoff 2025-09-07
                    "team": ["KC"],
                    "gsis_id": ["P00"],
                    "pos_grp": ["3WR 1TE"],
                    "pos_slot": [9],
                    "pos_rank": [1],
                }
            )
        return pd.read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(loader.pd, "read_parquet", _fake_url_read_parquet)

    df = loader.load_raw_data([2025], cache_dir=str(tmp_path))

    assert any("depth_charts_2025" in u for u in url_calls), "ESPN depth URL was not read"
    p00 = df[(df["player_id"] == "P00") & (df["season"] == 2025)]
    assert len(p00) == 1
    assert p00.iloc[0]["depth_chart_rank"] == 1.0  # real ESPN rank, not the -1 sentinel


@pytest.mark.unit
def test_load_raw_data_cache_hit_short_circuit(tmp_path, monkeypatch):
    """Pre-written caches → the loader skips every nfl.* call and reads from disk."""
    import src.data.loader as loader

    boom_calls: list[str] = []

    def _make_boom(name):
        def _boom(*a, **k):
            # Why: the cache short-circuit is the contract this test pins. We
            # collect call sites instead of raising so a regression yields a
            # readable list rather than an opaque first-failure.
            boom_calls.append(name)
            raise AssertionError(f"nfl_data_py.{name} was called despite cache hit")

        return _boom

    # Stub everything to scream if called.
    for name in (
        "weekly_data",
        "rosters",
        "schedules",
        "snap_counts",
        "injuries",
        "depth_charts",
    ):
        monkeypatch.setattr(loader.nfl_source, name, _make_boom(name))
    # import_pbp_data fires from src.data.redzone_pbp; stub it to scream too
    # so the cache short-circuit covers the new red-zone fetch path.
    import src.data.redzone_pbp as redzone_pbp

    monkeypatch.setattr(redzone_pbp.nfl_source, "pbp_data", _make_boom("import_pbp_data"))
    # import_ids is called inside the snap-merge try/except; it's ok for it to fire.
    ids_calls: list[None] = []

    def _ids():
        ids_calls.append(None)
        return pd.DataFrame({"pfr_id": ["pfr1"], "gsis_id": ["P00"]})

    monkeypatch.setattr(loader.nfl_source, "player_ids", _ids)

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

    # Pre-write the red-zone PBP cache with the full required schema so the
    # cache short-circuit fires and import_pbp_data isn't called.
    pd.DataFrame(
        {
            "player_id": ["P00"],
            "season": [2022],
            "week": [1],
            "recent_team": ["KC"],
            "redzone_carries": [0],
            "redzone_targets": [0],
            "inside10_carries": [0],
            "inside5_carries": [0],
            "redzone_target_share": [0.0],
        }
    ).to_parquet(tmp_path / f"redzone_pbp_{seasons[0]}_{seasons[-1]}.parquet")

    out = loader.load_raw_data(seasons, cache_dir=str(tmp_path))
    # Enrichment columns still land from the merge path.
    assert "snap_pct" in out.columns
    assert "depth_chart_rank" in out.columns
    # The cache short-circuit must skip every nfl.* fetch helper.
    assert boom_calls == [], f"nfl_data_py was hit despite caches: {boom_calls}"
    # import_ids fires inside the snap-merge try block — confirm the path took
    # the merge branch (not the bare-except fallback) on a cache hit.
    assert len(ids_calls) == 1
    # Player from cache propagates through the merge.
    assert "P00" in out["player_id"].tolist()


@pytest.mark.unit
def test_load_raw_data_snap_merge_exception_falls_back_to_nan(tmp_path, monkeypatch, capsys):
    """If import_ids raises, snap_pct defaults to NaN (except-branch coverage)."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    def _bad_ids():
        raise RuntimeError("nflverse id-map missing")

    monkeypatch.setattr(loader.nfl_source, "player_ids", _bad_ids)

    df = loader.load_raw_data([2023], cache_dir=str(tmp_path))
    assert "snap_pct" in df.columns
    assert df["snap_pct"].isna().all()
    assert "Snap count merge failed" in capsys.readouterr().out


@pytest.mark.unit
def test_load_raw_data_unified_weekly_path_handles_old_and_new_seasons(tmp_path, monkeypatch):
    """After the nflreadpy migration there is no 2025-only URL branch: every
    season flows through ``nfl_source.weekly_data`` (one code path). A mixed
    old+new season request must return rows for both. The legacy-schema rename
    moved into the shim and is covered by tests/data/test_nfl_source.py."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)

    # 2025 depth charts come from the nflverse ESPN `depth_charts` release via a
    # direct pd.read_parquet(url); intercept it. Weekly is unified through the
    # nf_source shim (mocked above) so there is no weekly URL. _fake_schedules
    # lacks gameday/game_type, so _normalize_espn_depth returns empty and 2025
    # depth falls to the -1 sentinel; this test only asserts the unified weekly path.
    def _fake_url_read_parquet(path, *args, **kwargs):
        if isinstance(path, str) and "depth_charts_2025" in path:
            return pd.DataFrame(
                {
                    "dt": ["2025-09-04T10:00:00Z"],
                    "team": ["KC"],
                    "gsis_id": ["P00"],
                    "pos_grp": ["3WR 1TE"],
                    "pos_slot": [9],
                    "pos_rank": [1],
                }
            )
        # Otherwise delegate to the real pd.read_parquet (cache reads).
        return pd.read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(loader.pd, "read_parquet", _fake_url_read_parquet)
    df = loader.load_raw_data([2024, 2025], cache_dir=str(tmp_path))
    assert 2024 in df["season"].values
    assert 2025 in df["season"].values
    # recent_team is the harmonised name the rest of the pipeline keys on
    # (produced by the shim; _fake_weekly already emits it).
    assert "recent_team" in df.columns


@pytest.mark.unit
def test_load_raw_data_no_snap_seasons_returns_empty_snap_df(tmp_path, monkeypatch):
    """When seasons are all < 2012, snap_counts ends up empty — the loader
    must still finish (empty-frame merge branch)."""
    import src.data.loader as loader

    _mock_all_nfl_helpers(monkeypatch)
    # Force import_snap_counts NOT to be called by picking pre-2012 seasons.

    # Override weekly to produce pre-2012 rows (simulate old nfl data).
    monkeypatch.setattr(
        loader.nfl_source,
        "weekly_data",
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
