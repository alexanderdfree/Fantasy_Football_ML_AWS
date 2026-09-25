"""Unit tests for the nflreadpy → pandas boundary shim (``src/data/nfl_source.py``).

These mock ``nflreadpy``'s ``load_*`` functions with small Polars frames so the
shim's schema-reconciliation logic (renames, ``player_id`` aliasing, PBP column
projection, Polars→pandas conversion) is exercised without any network access.
This is where the legacy-weekly rename contract lives after the migration unified
the loader's weekly path (see tests/test_data_loader.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import src.data.nfl_source as nfl_source


@pytest.mark.unit
def test_weekly_data_harmonizes_modern_schema(monkeypatch):
    """The modern ``stats_player`` columns get renamed to the legacy weekly
    names the pipeline keys on, and the result is a pandas frame."""
    modern = pl.DataFrame(
        {
            "player_id": ["P00"],
            "season": [2023],
            "week": [1],
            "team": ["KC"],
            "passing_interceptions": [1],
            "sacks_suffered": [2],
            "sack_yards_lost": [14],
        }
    )
    monkeypatch.setattr(
        nfl_source._nflreadpy,
        "load_player_stats",
        lambda seasons, summary_level="week": modern,
    )

    out = nfl_source.weekly_data([2023])

    assert isinstance(out, pd.DataFrame)
    assert {"recent_team", "interceptions", "sacks", "sack_yards"} <= set(out.columns)
    # Pre-rename names must not survive (they'd diverge from downstream schema).
    assert {"team", "passing_interceptions", "sacks_suffered", "sack_yards_lost"}.isdisjoint(
        out.columns
    )
    assert out.iloc[0]["recent_team"] == "KC"
    assert out.iloc[0]["interceptions"] == 1


@pytest.mark.unit
def test_rosters_adds_player_id_from_gsis(monkeypatch):
    """``load_rosters`` keys by ``gsis_id``; the shim adds the ``player_id``
    alias the loader's roster merge expects."""
    ros = pl.DataFrame(
        {"gsis_id": ["00-0000001"], "season": [2023], "position": ["QB"], "team": ["KC"]}
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_rosters", lambda seasons: ros)

    out = nfl_source.rosters([2023])

    assert isinstance(out, pd.DataFrame)
    assert "player_id" in out.columns
    assert out.iloc[0]["player_id"] == "00-0000001"


@pytest.mark.unit
def test_pbp_data_selects_only_requested_columns(monkeypatch):
    """``pbp_data`` projects to the requested columns before converting — this
    replaces nfl_data_py's removed ``columns=``/``downcast=`` params."""
    pbp = pl.DataFrame(
        {
            "season": [2023],
            "season_type": ["REG"],
            "week": [1],
            "posteam": ["KC"],
            "yardline_100": [5],
            "rusher_player_id": ["R1"],
            "receiver_player_id": [None],
            "pass_attempt": [0],
            "play_type": ["run"],
            "two_point_attempt": [0],
            "extra_unused_col": [99],
        }
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", lambda seasons: pbp)

    out = nfl_source.pbp_data([2023], nfl_source.PBP_REDZONE_COLS)

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(nfl_source.PBP_REDZONE_COLS)
    assert "extra_unused_col" not in out.columns


@pytest.mark.unit
def test_pbp_data_tolerates_missing_columns(monkeypatch):
    """A season missing a requested PBP column degrades to the available subset
    rather than raising (mirrors the per-year try/except in the consumers)."""
    pbp = pl.DataFrame({"season": [2023], "week": [1], "posteam": ["KC"]})
    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", lambda seasons: pbp)

    out = nfl_source.pbp_data([2023], nfl_source.PBP_REDZONE_COLS)

    assert set(out.columns) <= set(nfl_source.PBP_REDZONE_COLS)
    assert "season" in out.columns


@pytest.mark.unit
def test_teams_passthrough_has_logo_columns(monkeypatch):
    """``load_teams`` already exposes ``team_abbr`` + ``team_logo_espn``; the
    shim passes them through unchanged so dst/data.py's logo map is untouched."""
    teams = pl.DataFrame({"team_abbr": ["KC"], "team_logo_espn": ["https://example.test/kc.png"]})
    monkeypatch.setattr(nfl_source._nflreadpy, "load_teams", lambda: teams)

    out = nfl_source.teams()

    assert isinstance(out, pd.DataFrame)
    assert {"team_abbr", "team_logo_espn"} <= set(out.columns)
    assert out.iloc[0]["team_logo_espn"].endswith("kc.png")


@pytest.mark.unit
def test_rosters_weekly_adds_player_id_from_gsis(monkeypatch):
    """``load_rosters_weekly`` is the per-(player, week) status frame the
    inheritance out-set consumes (#1106); like ``rosters``, the shim adds the
    ``player_id`` alias from ``gsis_id``."""
    ros = pl.DataFrame(
        {
            "gsis_id": ["00-0000001", "00-0000001"],
            "season": [2023, 2023],
            "week": [1, 2],
            "position": ["RB", "RB"],
            "team": ["KC", "KC"],
            "status": ["ACT", "INA"],
        }
    )
    monkeypatch.setattr(nfl_source._nflreadpy, "load_rosters_weekly", lambda seasons: ros)

    out = nfl_source.rosters_weekly([2023])

    assert isinstance(out, pd.DataFrame)
    assert "player_id" in out.columns
    assert len(out) == 2  # one row per week, not per season
    assert set(out["status"]) == {"ACT", "INA"}


@pytest.mark.unit
def test_native_int_seasons_coerces_numpy_int():
    """The boundary helper yields native Python ints from numpy-int input."""
    out = nfl_source._native_int_seasons([np.int64(2023), np.int64(2024)])
    assert out == [2023, 2024]
    assert all(type(s) is int for s in out)


@pytest.mark.unit
def test_rosters_coerces_numpy_int_seasons_for_strict_loader(monkeypatch):
    """nflreadpy's ``load_rosters`` validates seasons with a strict
    ``not isinstance(season, int)`` guard that REJECTS numpy.int64 (the NFL.com
    Season Leaders all-null bug — a DataFrame-derived ``season.astype(int).unique()``
    list is numpy.int64). The shim must coerce to native int before the call so
    such a list still loads instead of raising "Season must be between …"."""
    captured: dict[str, list[int]] = {}

    def _strict_load_rosters(seasons):
        for s in seasons:  # mirror nflreadpy 0.1.5's exact guard
            if not isinstance(s, int):
                raise ValueError("Season must be between 1920 and 2026")
        captured["seasons"] = list(seasons)
        return pl.DataFrame({"gsis_id": ["00-0000001"], "season": [int(seasons[0])]})

    monkeypatch.setattr(nfl_source._nflreadpy, "load_rosters", _strict_load_rosters)

    # numpy.int64 seasons would raise without the boundary coercion.
    out = nfl_source.rosters([np.int64(2023), np.int64(2024)])

    assert isinstance(out, pd.DataFrame)
    assert captured["seasons"] == [2023, 2024]
    assert all(type(s) is int for s in captured["seasons"])


def _hung(seasons):
    raise ConnectionError(
        "Failed to download play_by_play_2012.parquet: HTTPSConnectionPool(...): "
        "Read timed out. (read timeout=120)"
    )


def _http_failure(status):
    import requests

    class _Response:
        status_code = status

    def loader(seasons):
        cause = requests.exceptions.HTTPError(f"{status} error", response=_Response())
        raise ConnectionError(f"Failed to download: {status}") from cause

    return loader


@pytest.mark.unit
def test_pbp_data_retries_a_hung_download_then_succeeds(monkeypatch):
    """The first request after the session idles hangs until the read timeout;
    the next attempt runs on a fresh connection (seven refresh-splits runs)."""
    pbp = pl.DataFrame({"season": [2012], "week": [1], "posteam": ["KC"], "epa": [0.1]})
    calls: list[list[int]] = []
    naps: list[float] = []

    def flaky(seasons):
        calls.append(list(seasons))
        if len(calls) == 1:
            _hung(seasons)
        return pbp

    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", flaky)
    monkeypatch.setattr(nfl_source, "_sleep", naps.append)
    out = nfl_source.pbp_data([2012], ("season", "week", "posteam"))
    assert list(out.columns) == ["season", "week", "posteam"] and len(out) == 1
    assert calls == [[2012], [2012]]
    assert naps == [nfl_source._PBP_RETRY_BACKOFF_S]


@pytest.mark.unit
def test_pbp_download_gives_up_after_the_configured_retries(monkeypatch):
    attempts: list[list[int]] = []

    def always_hangs(seasons):
        attempts.append(list(seasons))
        _hung(seasons)

    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", always_hangs)
    monkeypatch.setattr(nfl_source, "_sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        nfl_source._load_pbp_with_retry([2013])
    assert len(attempts) == nfl_source._PBP_MAX_RETRIES + 1


@pytest.mark.unit
def test_pbp_download_does_not_retry_a_missing_season_file_but_does_retry_a_5xx(monkeypatch):
    """nflreadpy wraps a 404 in the same ConnectionError as a hang; the response on
    the cause chain tells them apart."""
    attempts: list[list[int]] = []
    missing = _http_failure(404)

    def track_missing(seasons):
        attempts.append(list(seasons))
        missing(seasons)

    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", track_missing)
    monkeypatch.setattr(nfl_source, "_sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        nfl_source._load_pbp_with_retry([1999])
    assert len(attempts) == 1

    served: list[list[int]] = []
    gateway = _http_failure(502)

    def flaky_gateway(seasons):
        served.append(list(seasons))
        if len(served) == 1:
            gateway(seasons)
        return pl.DataFrame({"season": [2012]})

    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", flaky_gateway)
    out = nfl_source._load_pbp_with_retry([2012])
    assert len(served) == 2 and out.height == 1


@pytest.mark.unit
def test_pbp_download_does_not_retry_other_errors(monkeypatch):
    def bad_input(seasons):
        raise ValueError("Failed to parse data")

    monkeypatch.setattr(nfl_source._nflreadpy, "load_pbp", bad_input)
    monkeypatch.setattr(nfl_source, "_sleep", lambda s: None)
    with pytest.raises(ValueError):
        nfl_source._load_pbp_with_retry([2012])
