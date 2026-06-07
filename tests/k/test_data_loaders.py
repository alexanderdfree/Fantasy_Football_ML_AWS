"""Coverage tests for ``src/k/data.py``.

Mocks the ``src.data.nfl_source.pbp_data`` shim with synthetic per-play DataFrames so the
full PBP → weekly-kicker aggregation pipeline runs in-process. Also tests
the cache-hit shortcut (pre-written parquet) and the 2025 weekly+backfill
branch via tmp-path parquet fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _kicker_pbp_cache_row(player_id: str, season: int, week: int, recent_team: str = "KC") -> dict:
    """One row matching the schema written by ``reconstruct_kicker_weekly_from_pbp``.

    Tests pre-write a parquet of these to exercise the cache-hit branch without
    invoking the real PBP aggregation.
    """
    return {
        "player_id": player_id,
        "player_name": player_id,
        "recent_team": recent_team,
        "season": season,
        "week": week,
        "position": "K",
        "season_type": "REG",
        "fg_att": 3,
        "fg_made": 2,
        "fg_missed": 1,
        "fg_made_0_19": 0,
        "fg_made_20_29": 1,
        "fg_made_30_39": 1,
        "fg_made_40_49": 0,
        "fg_made_50_59": 0,
        "fg_made_60_": 0,
        "fg_missed_40_49": 1,
        "fg_missed_50_59": 0,
        "fg_missed_60_": 0,
        "fg_yards_made": 55.0,
        "avg_fg_distance": 35.0,
        "avg_fg_prob": 0.85,
        "q4_fg_att": 1,
        "q4_fg_made": 1,
        "long_fg_att": 1,
        "long_fg_made": 0,
        "game_wind": 5.0,
        "game_temp": 60.0,
        "roof": "outdoors",
        "surface": "grass",
        "is_dome": 0,
        "pat_att": 3,
        "pat_made": 3,
        "pat_missed": 0,
        # Sentinel proving this row was written by the post-XP-venue-backfill
        # code path; the cache schema gate (`_REQUIRED_PBP_COLUMNS`) rejects
        # parquets without it so stale caches regenerate. Test fixtures must
        # include it to walk the cache-hit branch.
        "_xp_venue_backfilled": True,
    }


# --------------------------------------------------------------------------
# Synthetic PBP frame — covers FG + XP rows and field required by the loader
# --------------------------------------------------------------------------


def _synthetic_pbp(season: int, n_fg: int = 6, n_xp: int = 4) -> pd.DataFrame:
    """Build a PBP-shaped DataFrame matching what ``nfl_source.pbp_data`` emits."""
    rng = np.random.default_rng(season)
    rows = []
    # Field goal attempts (alternate made/missed so every distance bucket + clutch branch fires)
    for i in range(n_fg):
        d = [15, 25, 35, 45, 55, 61][i % 6]
        rows.append(
            {
                "season": season,
                "season_type": "REG",
                "week": (i % 3) + 1,
                "posteam": ["KC", "BUF"][i % 2],
                "kicker_player_id": f"K{i % 2:02d}",
                "kicker_player_name": f"Kicker {i % 2}",
                "play_id": 1000 + i,
                "field_goal_attempt": 1,
                "extra_point_attempt": 0,
                "field_goal_result": "made" if i % 2 == 0 else "missed",
                "extra_point_result": None,
                "kick_distance": d,
                "score_differential": -3 + i,
                "qtr": 3 + (i % 2),
                "fg_prob": float(rng.uniform(0.5, 0.95)),
                "wind": float(rng.integers(0, 15)),
                "temp": 55.0,
                "roof": "outdoors" if i % 2 == 0 else "dome",
                "surface": "grass",
            }
        )
    # Extra points
    for i in range(n_xp):
        rows.append(
            {
                "season": season,
                "season_type": "REG",
                "week": (i % 3) + 1,
                "posteam": ["KC", "BUF"][i % 2],
                "kicker_player_id": f"K{i % 2:02d}",
                "kicker_player_name": f"Kicker {i % 2}",
                "play_id": 2000 + i,
                "field_goal_attempt": 0,
                "extra_point_attempt": 1,
                "field_goal_result": None,
                "extra_point_result": "good" if i % 2 == 0 else "failed",
                "kick_distance": 33,
                "score_differential": 0,
                "qtr": 2,
                "fg_prob": 0.99,
                "wind": 0.0,
                "temp": 55.0,
                "roof": "dome",
                "surface": "turf",
            }
        )
    # Non-kicker play (should be filtered out)
    rows.append(
        {
            "season": season,
            "season_type": "REG",
            "week": 1,
            "posteam": "KC",
            "kicker_player_id": None,
            "kicker_player_name": None,
            "play_id": 3000,
            "field_goal_attempt": 0,
            "extra_point_attempt": 0,
            "field_goal_result": None,
            "extra_point_result": None,
            "kick_distance": np.nan,
            "score_differential": 0,
            "qtr": 1,
            "fg_prob": np.nan,
            "wind": 0.0,
            "temp": 55.0,
            "roof": "outdoors",
            "surface": "grass",
        }
    )
    # Playoff row — must be filtered by season_type
    rows.append(
        {
            "season": season,
            "season_type": "POST",
            "week": 20,
            "posteam": "KC",
            "kicker_player_id": "K00",
            "kicker_player_name": "Kicker 0",
            "play_id": 4000,
            "field_goal_attempt": 1,
            "extra_point_attempt": 0,
            "field_goal_result": "made",
            "extra_point_result": None,
            "kick_distance": 40,
            "score_differential": 0,
            "qtr": 4,
            "fg_prob": 0.85,
            "wind": 5.0,
            "temp": 60.0,
            "roof": "outdoors",
            "surface": "grass",
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tests — reconstruct_kicker_weekly_from_pbp
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_reconstruct_weekly_from_pbp_happy_path(tmp_path, monkeypatch):
    import src.k.data as k_data

    def _fake_pbp(seasons, cols):
        return _synthetic_pbp(seasons[0])

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _fake_pbp)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))
    # 2 kickers x 3 weeks = 6 potential rows (some may merge differently)
    assert len(out) > 0
    assert "player_id" in out.columns
    assert "position" in out.columns
    assert (out["position"] == "K").all()
    assert "is_dome" in out.columns
    # Cache file must now exist
    assert (tmp_path / "kicker_pbp_2020_2020.parquet").exists()


@pytest.mark.unit
def test_reconstruct_weekly_from_pbp_xp_only_game_gets_venue(tmp_path, monkeypatch):
    """A kicker who only attempted XPs (no FGs) in a game must still get
    roof/surface populated — pulled from the XP plays' venue columns.

    Regression guard: the original aggregation sourced venue only from FG
    plays' groupby, leaving XP-only kicker-weeks with NaN roof/surface.
    The signal-floor diagnostic flagged this as 55 NaN rows on the 2025
    test set; this test prevents the FG-only sourcing from coming back.
    """
    import src.k.data as k_data

    def _xp_only_pbp(seasons, cols):
        """Synthetic PBP where K01 has only XPs (no FGs)."""
        yr = seasons[0]
        rows = []
        for i in range(3):
            rows.append(
                {
                    "season": yr,
                    "season_type": "REG",
                    "week": 1,
                    "posteam": "KC",
                    "kicker_player_id": "K01",
                    "kicker_player_name": "Kicker 1",
                    "play_id": 2000 + i,
                    "field_goal_attempt": 0,
                    "extra_point_attempt": 1,
                    "field_goal_result": None,
                    "extra_point_result": "good",
                    "kick_distance": 33,
                    "score_differential": 0,
                    "qtr": 2,
                    "fg_prob": 0.99,
                    "wind": 10.0,
                    "temp": 65.0,
                    "roof": "outdoors",
                    "surface": "grass",
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _xp_only_pbp)
    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))

    k01 = out[out["player_id"] == "K01"].iloc[0]
    assert k01["fg_att"] == 0  # confirms it's the XP-only branch
    assert k01["pat_att"] == 3
    # Venue must be populated from XP plays, not NaN.
    assert k01["roof"] == "outdoors"
    assert k01["surface"] == "grass"
    assert k01["is_dome"] == 0
    # recent_team and player_name must also come from XP plays (the FG groupby
    # had no rows so they would default to NaN without the XP fallback).
    assert k01["recent_team"] == "KC"
    assert k01["player_name"] == "Kicker 1"


@pytest.mark.unit
def test_reconstruct_weekly_from_pbp_cache_hit(tmp_path, monkeypatch):
    """Pre-existing cache parquet with the current schema → no PBP call, just
    a load-and-return."""
    import src.k.data as k_data

    # Pre-write the cache file with the full current schema (required by the
    # `_cached_pbp_is_current` guard added after the Apr-19 stale-cache bug).
    cache_path = tmp_path / "kicker_pbp_2021_2021.parquet"
    pd.DataFrame([_kicker_pbp_cache_row("K01", 2021, 1)]).to_parquet(cache_path)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("pbp_data was called despite cache hit")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _should_not_be_called)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2021], cache_dir=str(tmp_path))
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "K01"


@pytest.mark.unit
def test_reconstruct_weekly_pbp_dome_games_get_65f_0_wind(tmp_path, monkeypatch):
    """Dome games in raw PBP have NaN temp/wind ~100% of the time. The
    reconstruction must rewrite those to (65.0 F, 0.0 mph) to mirror the
    canonical temp_adjusted/wind_adjusted handling in weather_features.py —
    not leave them at 0.0 F (the old blanket fillna behaviour) which
    confounded dome games with extreme-cold outdoor games."""
    import src.k.data as k_data

    def _pbp_with_nan_dome_weather(seasons, cols):
        # Two games: one dome (NaN weather, as nfl_data_py emits) and one
        # outdoor (real weather). Each game gets one FG attempt.
        return pd.DataFrame(
            [
                {  # Dome FG: NaN temp/wind in raw PBP
                    "season": seasons[0],
                    "season_type": "REG",
                    "week": 1,
                    "posteam": "MIN",
                    "kicker_player_id": "K_DOME",
                    "kicker_player_name": "Dome Kicker",
                    "play_id": 1001,
                    "field_goal_attempt": 1,
                    "extra_point_attempt": 0,
                    "field_goal_result": "made",
                    "extra_point_result": None,
                    "kick_distance": 35,
                    "score_differential": 0,
                    "qtr": 3,
                    "fg_prob": 0.85,
                    "wind": float("nan"),
                    "temp": float("nan"),
                    "roof": "dome",
                    "surface": "turf",
                },
                {  # Outdoor FG: real weather
                    "season": seasons[0],
                    "season_type": "REG",
                    "week": 1,
                    "posteam": "BUF",
                    "kicker_player_id": "K_OUT",
                    "kicker_player_name": "Outdoor Kicker",
                    "play_id": 1002,
                    "field_goal_attempt": 1,
                    "extra_point_attempt": 0,
                    "field_goal_result": "made",
                    "extra_point_result": None,
                    "kick_distance": 42,
                    "score_differential": 0,
                    "qtr": 3,
                    "fg_prob": 0.75,
                    "wind": 12.0,
                    "temp": 28.0,
                    "roof": "outdoors",
                    "surface": "grass",
                },
            ]
        )

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _pbp_with_nan_dome_weather)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))

    dome_row = out[out["player_id"] == "K_DOME"].iloc[0]
    outdoor_row = out[out["player_id"] == "K_OUT"].iloc[0]

    assert dome_row["is_dome"] == 1
    assert dome_row["game_temp"] == 65.0, (
        f"dome game_temp must be rewritten to 65.0 F (got {dome_row['game_temp']})"
    )
    assert dome_row["game_wind"] == 0.0, (
        f"dome game_wind must be rewritten to 0.0 mph (got {dome_row['game_wind']})"
    )

    assert outdoor_row["is_dome"] == 0
    assert outdoor_row["game_temp"] == 28.0, (
        "outdoor game_temp must be preserved, not overwritten by dome default"
    )
    assert outdoor_row["game_wind"] == 12.0, (
        "outdoor game_wind must be preserved, not overwritten by dome default"
    )


@pytest.mark.unit
def test_backfill_2025_pbp_dome_games_get_65f_0_wind(monkeypatch):
    """2025 PBP backfill must apply the same dome rewrite as the historical
    reconstruction: dome rows -> (65.0 F, 0.0 mph)."""
    import src.k.data as k_data

    def _pbp_2025_dome(seasons, cols):
        return pd.DataFrame(
            [
                {
                    "season": 2025,
                    "season_type": "REG",
                    "week": 1,
                    "posteam": "ATL",
                    "kicker_player_id": "K_DOME_2025",
                    "kicker_player_name": "Dome 2025",
                    "play_id": 5001,
                    "field_goal_attempt": 1,
                    "extra_point_attempt": 0,
                    "field_goal_result": "made",
                    "extra_point_result": None,
                    "kick_distance": 33,
                    "score_differential": 0,
                    "qtr": 4,
                    "fg_prob": 0.9,
                    "wind": float("nan"),
                    "temp": float("nan"),
                    "roof": "dome",
                    "surface": "turf",
                },
            ]
        )

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _pbp_2025_dome)

    # recent_team is required: post-PR #239/#241, the 2025 backfill merges
    # venue/weather via a second pass keyed on (season, week, recent_team)
    # — independent of the FG-stats merge keyed on player_id.
    k_df = pd.DataFrame(
        [
            {
                "player_id": "K_DOME_2025",
                "recent_team": "ATL",
                "season": 2025,
                "week": 1,
                "avg_fg_distance": float("nan"),
                "avg_fg_prob": float("nan"),
                "q4_fg_att": float("nan"),
                "q4_fg_made": float("nan"),
                "long_fg_att": float("nan"),
                "long_fg_made": float("nan"),
                "game_wind": float("nan"),
                "game_temp": float("nan"),
                "roof": float("nan"),
                "surface": float("nan"),
                "is_dome": float("nan"),
                "fg_yards_made": float("nan"),
            }
        ]
    )

    k_data._backfill_2025_pbp_columns(k_df, [2025])

    row = k_df.iloc[0]
    assert row["is_dome"] == 1
    assert row["game_temp"] == 65.0
    assert row["game_wind"] == 0.0


@pytest.mark.unit
def test_reconstruct_weekly_from_pbp_stale_cache_regenerates(tmp_path, monkeypatch, capsys):
    """A cache parquet missing required columns (e.g. ``fg_yards_made`` added
    after the cache was written) must be ignored and the PBP path re-run.

    Regression guard for the Apr-19 stale-cache bug where the cache survived a
    schema change and silently zeroed `fg_yard_points` for the entire training
    range — collapsing K projections from 8-10 fpts down to ~3 fpts.
    """
    import src.k.data as k_data

    # Pre-write a parquet that looks plausible but is missing fg_yards_made.
    stale_cache = tmp_path / "kicker_pbp_2020_2020.parquet"
    pd.DataFrame({"player_id": ["K01"], "season": [2020], "week": [1]}).to_parquet(stale_cache)

    monkeypatch.setattr(
        k_data.nfl_source, "pbp_data", lambda seasons, cols: _synthetic_pbp(seasons[0])
    )

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))

    # The PBP path ran — output reflects the synthetic frame's many rows, not
    # the 1-row stale cache.
    assert len(out) > 1
    assert "fg_yards_made" in out.columns
    # Log line surfaces the bad schema so future debugging is obvious.
    captured = capsys.readouterr().out
    assert "Stale cache" in captured
    assert "fg_yards_made" in captured


@pytest.mark.unit
def test_reconstruct_weekly_from_pbp_pre_xp_venue_cache_rejected(tmp_path, monkeypatch, capsys):
    """A cache parquet from before the XP-venue backfill landed (lacks the
    ``_xp_venue_backfilled`` sentinel column) must be regenerated so the
    historical 589 XP-only kicker-weeks pick up roof/surface from XP plays.
    Otherwise the train cache would diverge from the test-time
    schedules-fallback backfill in `load_data` and create a train/test
    distribution mismatch on `is_dome`.
    """
    import src.k.data as k_data

    # Pre-write a cache that looks fully populated but predates the XP-venue
    # fix — every required column except the sentinel.
    stale_cache = tmp_path / "kicker_pbp_2020_2020.parquet"
    cache_row = _kicker_pbp_cache_row("K01", 2020, 1)
    del cache_row["_xp_venue_backfilled"]
    pd.DataFrame([cache_row]).to_parquet(stale_cache)

    monkeypatch.setattr(
        k_data.nfl_source, "pbp_data", lambda seasons, cols: _synthetic_pbp(seasons[0])
    )

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))

    # The PBP path ran (multi-row synthetic output, not the 1-row cache).
    assert len(out) > 1
    # Regenerated cache now has the sentinel.
    assert "_xp_venue_backfilled" in out.columns
    captured = capsys.readouterr().out
    assert "Stale cache" in captured
    assert "_xp_venue_backfilled" in captured


@pytest.mark.unit
def test_reconstruct_weekly_pbp_skips_failing_seasons(tmp_path, monkeypatch, capsys):
    """If ``pbp_data`` throws (e.g. upstream 502), the per-year body is
    skipped, a WARNING is logged, and the partial result is NOT cached so the
    next call doesn't treat a partial frame as authoritative."""
    import src.k.data as k_data

    def _bad(seasons, cols):
        raise RuntimeError(f"pbp fetch boom for {seasons}")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _bad)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020], cache_dir=str(tmp_path))
    assert out.empty
    assert "PBP weekly extraction failed" in capsys.readouterr().out
    # No poisoned cache.
    assert not (tmp_path / "kicker_pbp_2020_2020.parquet").exists()


@pytest.mark.unit
def test_reconstruct_weekly_pbp_partial_failure_skips_cache(tmp_path, monkeypatch, capsys):
    """If only some seasons fail, returned frame contains the survivors but
    the combined cache key is NOT written (it would silently claim coverage
    of the failed years)."""
    import src.k.data as k_data

    def _selective(seasons, cols):
        yr = seasons[0]
        if yr == 2021:
            raise RuntimeError("upstream 502 for 2021")
        return _synthetic_pbp(yr)

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _selective)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2020, 2021], cache_dir=str(tmp_path))
    # 2020 survives; 2021 was dropped.
    assert not out.empty
    assert 2020 in out["season"].values
    assert 2021 not in out["season"].values
    captured = capsys.readouterr().out
    assert "PBP weekly extraction failed for 2021" in captured
    assert "not caching partial result" in captured
    assert not (tmp_path / "kicker_pbp_2020_2021.parquet").exists()


# --------------------------------------------------------------------------
# Tests — reconstruct_kicker_kicks_from_pbp
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_reconstruct_kicks_from_pbp_happy_path(tmp_path, monkeypatch):
    import src.k.data as k_data

    monkeypatch.setattr(
        k_data.nfl_source, "pbp_data", lambda seasons, cols: _synthetic_pbp(seasons[0])
    )
    out = k_data.reconstruct_kicker_kicks_from_pbp([2020], cache_dir=str(tmp_path))
    assert "is_fg" in out.columns
    assert "is_xp" in out.columns
    assert out["is_fg"].sum() > 0
    assert out["is_xp"].sum() > 0
    # Cache persisted.
    assert (tmp_path / "kicker_kicks_pbp_2020_2020.parquet").exists()


@pytest.mark.unit
def test_reconstruct_kicks_from_pbp_cache_hit(tmp_path, monkeypatch):
    """Cache-hit path: a parquet whose schema covers every column the current
    aggregation produces is returned as-is without invoking nflverse.
    """
    import src.k.data as k_data

    cache_path = tmp_path / "kicker_kicks_pbp_2022_2022.parquet"
    # Schema must cover ``_KICKS_SCHEMA`` so ``_cached_kick_pbp_is_current``
    # accepts the cache and the function takes the cache-hit branch.
    pd.DataFrame(
        {
            "player_id": ["K01"],
            "season": [2022],
            "week": [1],
            "play_id": [101],
            "is_fg": [1],
            "is_xp": [0],
            "kick_distance": [35.0],
            "kick_made": [1],
            "fg_prob": [0.85],
            "is_q4": [0],
            "score_diff": [-3.0],
            "game_wind": [5.0],
        }
    ).to_parquet(cache_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("pbp_data was called despite cache hit")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _should_not_be_called)
    out = k_data.reconstruct_kicker_kicks_from_pbp([2022], cache_dir=str(tmp_path))
    assert len(out) == 1


@pytest.mark.unit
def test_reconstruct_kicks_from_pbp_stale_cache_regenerates(tmp_path, monkeypatch, capsys):
    """Cache parquet missing required schema columns (e.g. ``play_id`` added
    after the cache was written) is rejected and PBP re-aggregated.

    Mirrors the weekly-cache regression guard but for the per-kick cache —
    a stale per-kick cache would silently break the deterministic
    most-recent kick truncation that ``build_nested_kick_history`` relies on.
    """
    import src.k.data as k_data

    cache_path = tmp_path / "kicker_kicks_pbp_2022_2022.parquet"
    # Old-style cache: only the first few columns from a pre-``play_id`` version.
    pd.DataFrame(
        {"player_id": ["K01"], "season": [2022], "week": [1], "is_fg": [1], "is_xp": [0]}
    ).to_parquet(cache_path)

    # PBP call must fire (cache is stale).
    called = {"n": 0}

    def _stub_pbp(seasons, cols):
        called["n"] += 1
        raise RuntimeError("synthetic boom — we just need to know PBP was tried")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _stub_pbp)
    _ = k_data.reconstruct_kicker_kicks_from_pbp([2022], cache_dir=str(tmp_path))
    assert called["n"] == 1
    assert "Stale kick cache" in capsys.readouterr().out


@pytest.mark.unit
def test_reconstruct_kicks_pbp_skips_failing_seasons(tmp_path, monkeypatch, capsys):
    """If ``pbp_data`` throws for a season, we log and continue."""
    import src.k.data as k_data

    def _bad(seasons, cols):
        raise RuntimeError(f"pbp fetch boom for {seasons}")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _bad)
    # Every season fails, so the result is the empty frame the function returns.
    out = k_data.reconstruct_kicker_kicks_from_pbp([2020], cache_dir=str(tmp_path))
    assert out.empty
    assert "per-kick PBP extraction failed" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Tests — load_data (cache-hit shortcut path only; the full PBP path
# is exercised by reconstruct_* tests above).
# --------------------------------------------------------------------------


@pytest.fixture()
def _cached_pbp(tmp_path, monkeypatch):
    """Pre-write kicker_pbp cache so load_data skips the PBP chain."""
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("src.config.CACHE_DIR", str(tmp_path))

    # Restrict SEASONS so cache path key is predictable.
    monkeypatch.setattr(k_data, "SEASONS", [2022, 2023, 2024])
    monkeypatch.setattr(k_data, "MIN_GAMES", 1)  # don't filter our tiny frame

    # Enforce the docstring's "without touching nflverse" claim — without this
    # guard, a default-arg pitfall on cache_dir silently caused cache misses
    # and the test passed only because nflverse usually returned valid data.
    def _no_network(*args, **kwargs):
        raise AssertionError("nfl_source.pbp_data must not be called when the PBP cache hits")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _no_network)

    # Synthetic kicker weekly cache (2022-2024).
    kicker_rows = [
        _kicker_pbp_cache_row(player_id=f"K{wk}", season=season, week=wk)
        for season in [2022, 2023, 2024]
        for wk in range(1, 4)
    ]
    pd.DataFrame(kicker_rows).to_parquet(tmp_path / "kicker_pbp_2022_2024.parquet")

    # Schedule parquet — needs SEASONS[0]-SEASONS[-1] key to match load_data.
    sched_path = tmp_path / f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    sched_rows = []
    for season in [2022, 2023, 2024]:
        for wk in range(1, 4):
            sched_rows.append(
                {
                    "season": season,
                    "week": wk,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "spread_line": -3.0,
                    "total_line": 47.0,
                    "game_type": "REG",
                }
            )
    pd.DataFrame(sched_rows).to_parquet(sched_path)

    return tmp_path


@pytest.mark.unit
def test_load_kicker_data_uses_pbp_cache(_cached_pbp):
    """With a pre-written PBP cache and no 2025 weekly, load_data
    walks the cache-hit branch and merges schedules without touching nflverse."""
    from src.k.data import load_data

    df = load_data()
    assert len(df) > 0
    assert "is_home" in df.columns
    assert "implied_team_total" in df.columns
    # Every row must have total_line + implied_team_total post fillna.
    assert df["total_line"].notna().all()
    assert df["implied_team_total"].notna().all()


@pytest.mark.unit
def test_compute_targets_fg_yard_points_non_zero_after_load(_cached_pbp):
    """End-to-end guard: load_data() → compute_targets() must produce non-zero
    `fg_yard_points` across the training data.

    Regression guard for the Apr-19 stale-cache bug where the cache survived a
    schema change. `fg_yards_made` came back as NaN, `compute_targets` ran
    `fillna(0)`, and the model trained on all-zero `fg_yard_points` →
    catastrophic generalization gap on 2025 test data (R² = -1.79).
    """
    from src.k.data import load_data
    from src.k.targets import compute_targets

    df = compute_targets(load_data())
    # Fixture row ships with fg_yards_made=55.0 → fg_yard_points = 5.5 on
    # every kicker-game; assert the column isn't silently zeroed.
    assert (df["fg_yard_points"] > 0).mean() > 0.5, (
        "fg_yard_points collapsed to zero — cache schema check likely broke "
        "or compute_targets stopped reading fg_yards_made"
    )


@pytest.mark.unit
def test_load_kicker_data_fills_missing_is_home_with_zero(tmp_path, monkeypatch):
    """Schedule merge can leave NaN ``is_home`` for kicker rows whose
    ``recent_team`` doesn't appear in any schedule row (e.g. mid-season
    trade with stale schedule cache). Those rows must fall through to
    ``is_home = 0`` instead of breaking downstream features."""
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("src.config.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(k_data, "SEASONS", [2022])
    monkeypatch.setattr(k_data, "MIN_GAMES", 1)

    def _no_network(*args, **kwargs):
        raise AssertionError("pbp_data must not be called when the cache hits")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _no_network)

    # PBP cache contains a kicker on team "ZZZ" that the schedule doesn't cover.
    pd.DataFrame([_kicker_pbp_cache_row("K01", 2022, 1, recent_team="ZZZ")]).to_parquet(
        tmp_path / "kicker_pbp_2022_2022.parquet"
    )

    # Schedule covers KC vs BUF only — kicker on ZZZ won't match.
    sched_path = tmp_path / f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "season": [2022],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BUF"],
            "spread_line": [-3.0],
            "total_line": [47.0],
            "game_type": ["REG"],
        }
    ).to_parquet(sched_path)

    df = k_data.load_data()
    # Unmatched recent_team falls through to is_home = 0 (line 274 fallback).
    assert (df["is_home"] == 0).all()
    assert df["is_home"].notna().all()


# --------------------------------------------------------------------------
# Tests — filter_to_position + season_split
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_to_k_is_identity():
    from src.k.data import filter_to_position

    df = pd.DataFrame({"player_id": ["K1"], "season": [2024], "week": [1]})
    out = filter_to_position(df)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df  # copy, not same ref


@pytest.mark.unit
def test_kicker_season_split_splits_by_year(capsys):
    """Train rows for a (player_id, season) below ``MIN_GAMES`` are dropped
    (train-only filter, mirroring other positions). Val/test rows are kept
    regardless of season-game-count."""
    from src.k.data import MIN_GAMES, season_split

    # K1: enough games per train season to survive the MIN_GAMES filter.
    # K2: too few games in train (will be dropped from train) but kept in val/test.
    rows = []
    for wk in range(1, MIN_GAMES + 2):  # >= MIN_GAMES weeks for K1 in 2022 and 2023
        rows.append({"player_id": "K1", "season": 2022, "week": wk})
        rows.append({"player_id": "K1", "season": 2023, "week": wk})
    rows += [
        {"player_id": "K1", "season": 2024, "week": 1},
        {"player_id": "K1", "season": 2025, "week": 1},
        {"player_id": "K2", "season": 2023, "week": 1},  # below MIN_GAMES → dropped
        {"player_id": "K2", "season": 2024, "week": 1},  # val: kept
        {"player_id": "K2", "season": 2025, "week": 1},  # test: kept
    ]
    df = pd.DataFrame(rows)
    train, val, test = season_split(df)
    assert set(train["season"].unique()) <= {2022, 2023}
    assert set(val["season"].unique()) == {2024}
    assert set(test["season"].unique()) == {2025}
    # K2 dropped from train (only 1 game in 2023), kept in val/test.
    assert "K2" not in set(train["player_id"].unique())
    assert "K2" in set(val["player_id"].unique())
    assert "K2" in set(test["player_id"].unique())
    # Print statement fires
    assert "K cross-season split" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Tests — load_kicks (minimal: cache-hit branch)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_backfill_2025_pbp_columns_updates_in_place(monkeypatch):
    """``_backfill_2025_pbp_columns`` must overwrite PBP-derived columns on
    rows whose ``season`` is in the backfill list — existing NaNs should
    become populated values once the fake PBP frame fires."""
    import src.k.data as k_data

    monkeypatch.setattr(
        k_data.nfl_source, "pbp_data", lambda seasons, cols: _synthetic_pbp(seasons[0])
    )

    # k_df: 2 kickers x 2 weeks in 2025 with everything NaN/None.
    # recent_team is provided so the (season, week, recent_team) venue lookup
    # populates roof/surface even for XP-only kicker-weeks (the FG-derived
    # path keyed on (player_id, season, week) would otherwise leave them NaN).
    rows = []
    for pid, team in (("K00", "KC"), ("K01", "BUF")):
        for wk in (1, 2):
            rows.append(
                {
                    "player_id": pid,
                    "recent_team": team,
                    "season": 2025,
                    "week": wk,
                    "avg_fg_distance": float("nan"),
                    "avg_fg_prob": float("nan"),
                    "q4_fg_att": float("nan"),
                    "q4_fg_made": float("nan"),
                    "long_fg_att": float("nan"),
                    "long_fg_made": float("nan"),
                    "game_wind": float("nan"),
                    "game_temp": float("nan"),
                    "roof": float("nan"),
                    "surface": float("nan"),
                    "is_dome": float("nan"),
                    "fg_yards_made": float("nan"),
                }
            )
    k_df = pd.DataFrame(rows)

    k_data._backfill_2025_pbp_columns(k_df, [2025])
    # At least one row must have been populated from the synthetic PBP.
    # (Exact values depend on the synthetic data — we just check that the
    # backfill actually writes.)
    assert k_df["roof"].notna().sum() > 0


@pytest.mark.unit
def test_backfill_2025_pbp_logs_warning_on_failure(monkeypatch, capsys):
    """If ``pbp_data`` raises, _backfill logs a warning and leaves
    k_df untouched (swallowed by the outer try/except)."""
    import src.k.data as k_data

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _boom)

    k_df = pd.DataFrame(
        {
            "player_id": ["K01"],
            "season": [2025],
            "week": [1],
            "roof": [float("nan")],
            "surface": [float("nan")],
        }
    )
    k_data._backfill_2025_pbp_columns(k_df, [2025])
    assert "2025 PBP backfill failed" in capsys.readouterr().out


@pytest.mark.unit
def test_backfill_2025_pbp_early_returns_when_no_matching_seasons():
    """``k_df`` without any rows matching ``seasons`` → no-op early return."""
    import src.k.data as k_data

    k_df = pd.DataFrame(
        {"player_id": ["K01"], "season": [2023], "week": [1], "roof": [float("nan")]}
    )
    # No 2025 rows → function must early-return without importing PBP.
    k_data._backfill_2025_pbp_columns(k_df, [2025])
    # k_df unchanged (no Wobbly side effects).
    assert k_df.iloc[0]["season"] == 2023


@pytest.mark.unit
def test_load_kicker_data_includes_2025_weekly_branch(tmp_path, monkeypatch):
    """``load_data`` must walk the 2025-weekly branch when SEASONS
    contains 2025. We pre-seed the weekly parquet, skip real PBP via the
    cache-hit shortcut, and stub _backfill so it's a no-op."""
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(k_data, "SEASONS", [2024, 2025])
    monkeypatch.setattr(k_data, "MIN_GAMES", 1)

    # Enforce the docstring's "skip real PBP via the cache-hit shortcut" claim:
    # neither reconstruct_kicker_weekly_from_pbp (cache-hit) nor _backfill_2025
    # (stubbed below) should reach the network.
    def _no_network(*args, **kwargs):
        raise AssertionError(
            "nfl_source.pbp_data must not be called: 2024 must hit cache, "
            "and the 2025 backfill is stubbed to no-op"
        )

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _no_network)

    # 2024 PBP cache
    pd.DataFrame([_kicker_pbp_cache_row("K01", 2024, 1)]).to_parquet(
        tmp_path / "kicker_pbp_2024_2024.parquet"
    )

    # Weekly frame with a 2025 K row
    weekly_path = tmp_path / f"weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "player_id": ["K01", "WR01"],
            "player_name": ["Kicker 01", "Wide 01"],
            "recent_team": ["KC", "KC"],
            "season": [2025, 2025],
            "week": [1, 1],
            "position": ["K", "WR"],
            "season_type": ["REG", "REG"],
            "fg_att": [3.0, 0.0],
            "fg_made": [2.0, 0.0],
            "fg_missed": [1.0, 0.0],
            "pat_att": [3.0, 0.0],
            "pat_made": [3.0, 0.0],
            "pat_missed": [0.0, 0.0],
        }
    ).to_parquet(weekly_path)

    # Schedule parquet (same key format as the existing fixture).
    sched_path = tmp_path / f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "season": [2024, 2025],
            "week": [1, 1],
            "home_team": ["KC", "KC"],
            "away_team": ["BUF", "BUF"],
            "spread_line": [-3.0, -3.0],
            "total_line": [47.0, 47.0],
            "game_type": ["REG", "REG"],
        }
    ).to_parquet(sched_path)

    # Stub _backfill_2025_pbp_columns so it's a no-op (covered separately).
    monkeypatch.setattr(k_data, "_backfill_2025_pbp_columns", lambda df, seasons: None)

    df = k_data.load_data()
    # Has rows from both 2024 (PBP) and 2025 (weekly).
    assert 2024 in df["season"].values
    assert 2025 in df["season"].values
    # 2025 weekly WR row must have been filtered out.
    assert (df["player_id"] != "WR01").all()


@pytest.mark.unit
def test_load_data_backfills_venue_for_xp_only_games(tmp_path, monkeypatch):
    """XP-only kicker-weeks (no FG attempts) must still get roof/surface from
    the schedules-merge fallback. Regression guard for the 55 NaN test rows
    that the K signal-floor diagnostic flagged: XP-only games had NaN roof
    because both the FG-aggregation path and the PBP-derived backfill key on
    FG plays only.
    """
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(k_data, "SEASONS", [2024, 2025])
    monkeypatch.setattr(k_data, "MIN_GAMES", 1)

    def _no_network(*args, **kwargs):
        raise AssertionError("pbp_data must not be called when the cache hits")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _no_network)

    # 2024 PBP cache (one FG-attempted row).
    pd.DataFrame([_kicker_pbp_cache_row("K00", 2024, 1)]).to_parquet(
        tmp_path / "kicker_pbp_2024_2024.parquet"
    )

    # 2025 weekly with two kickers: K01 (FG + XP) and K02 (XP-only).
    weekly_path = tmp_path / f"weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "player_id": ["K01", "K02"],
            "player_name": ["Kicker 01", "Kicker 02"],
            "recent_team": ["KC", "BUF"],
            "season": [2025, 2025],
            "week": [1, 1],
            "position": ["K", "K"],
            "season_type": ["REG", "REG"],
            "fg_att": [3.0, 0.0],
            "fg_made": [2.0, 0.0],
            "fg_missed": [1.0, 0.0],
            "pat_att": [3.0, 4.0],
            "pat_made": [3.0, 4.0],
            "pat_missed": [0.0, 0.0],
        }
    ).to_parquet(weekly_path)

    # Schedule parquet with roof/surface populated for both games.
    sched_path = tmp_path / f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "season": [2024, 2025, 2025],
            "week": [1, 1, 1],
            "home_team": ["KC", "KC", "BUF"],
            "away_team": ["BUF", "DET", "MIA"],
            "spread_line": [-3.0, -3.0, -3.0],
            "total_line": [47.0, 47.0, 47.0],
            "game_type": ["REG", "REG", "REG"],
            "roof": ["outdoors", "dome", "outdoors"],
            "surface": ["grass", "matrixturf", "a_turf"],
        }
    ).to_parquet(sched_path)

    # Stub the 2025 PBP backfill — we want to prove the schedules fallback
    # alone is sufficient to populate roof/surface for the XP-only kicker.
    monkeypatch.setattr(k_data, "_backfill_2025_pbp_columns", lambda df, seasons: None)

    df = k_data.load_data()
    df_25 = df[df["season"] == 2025]
    # Both 2025 kickers must have roof + surface populated (no NaN).
    assert df_25["roof"].notna().all(), "XP-only kicker's roof was not backfilled"
    assert df_25["surface"].notna().all(), "XP-only kicker's surface was not backfilled"
    # is_dome must be derived from the backfilled roof.
    k02_row = df_25[df_25["player_id"] == "K02"].iloc[0]
    assert k02_row["roof"] == "outdoors"
    assert k02_row["surface"] == "a_turf"
    assert k02_row["is_dome"] == 0


@pytest.mark.unit
def test_load_data_treats_empty_string_surface_as_missing(tmp_path, monkeypatch):
    """Surface from PBP is occasionally an empty string (e.g. 2025 KC@LAC wk 1).
    The schedules fallback must replace those just like it replaces NaN.
    """
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(k_data, "SEASONS", [2024, 2025])
    monkeypatch.setattr(k_data, "MIN_GAMES", 1)

    def _no_network(*args, **kwargs):
        raise AssertionError("pbp_data must not be called when the cache hits")

    monkeypatch.setattr(k_data.nfl_source, "pbp_data", _no_network)

    pd.DataFrame([_kicker_pbp_cache_row("K00", 2024, 1)]).to_parquet(
        tmp_path / "kicker_pbp_2024_2024.parquet"
    )

    # 2025 weekly with surface="" (empty string from PBP).
    weekly_path = tmp_path / f"weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "player_id": ["K01"],
            "player_name": ["Kicker 01"],
            "recent_team": ["KC"],
            "season": [2025],
            "week": [1],
            "position": ["K"],
            "season_type": ["REG"],
            "fg_att": [3.0],
            "fg_made": [2.0],
            "fg_missed": [1.0],
            "pat_att": [3.0],
            "pat_made": [3.0],
            "pat_missed": [0.0],
            "roof": ["dome"],
            "surface": [""],  # empty-string surface — must be replaced
        }
    ).to_parquet(weekly_path)

    sched_path = tmp_path / f"schedules_{SEASONS[0]}_{SEASONS[-1]}.parquet"
    pd.DataFrame(
        {
            "season": [2024, 2025],
            "week": [1, 1],
            "home_team": ["KC", "KC"],
            "away_team": ["BUF", "BUF"],
            "spread_line": [-3.0, -3.0],
            "total_line": [47.0, 47.0],
            "game_type": ["REG", "REG"],
            "roof": ["outdoors", "dome"],
            "surface": ["grass", "matrixturf"],
        }
    ).to_parquet(sched_path)

    monkeypatch.setattr(k_data, "_backfill_2025_pbp_columns", lambda df, seasons: None)

    df = k_data.load_data()
    df_25 = df[df["season"] == 2025]
    # surface must have been replaced from "" to the schedules value.
    assert (df_25["surface"] != "").all()
    assert df_25.iloc[0]["surface"] == "matrixturf"


@pytest.mark.unit
def test_load_kicker_kicks_with_stubbed_reconstruct(monkeypatch):
    """load_kicks delegates to ``reconstruct_kicker_kicks_from_pbp``;
    stub that out and verify the merge + is_home fill logic."""
    import src.k.data as k_data

    stub_kicks = pd.DataFrame(
        {
            "player_id": ["K01", "K01", "K02"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "is_fg": [1, 0, 1],
            "is_xp": [0, 1, 0],
            "kick_distance": [35.0, 0.0, 48.0],
            "kick_made": [1, 1, 0],
            "fg_prob": [0.85, 0.0, 0.6],
            "is_q4": [0, 1, 0],
            "score_diff": [-3.0, 0.0, 7.0],
            "game_wind": [5.0, 0.0, 10.0],
        }
    )
    monkeypatch.setattr(k_data, "reconstruct_kicker_kicks_from_pbp", lambda s: stub_kicks)

    k_df = pd.DataFrame(
        {
            "player_id": ["K01", "K01", "K02"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 1],
            "is_home": [1, 0, 1],
        }
    )
    out = k_data.load_kicks(k_df)
    assert len(out) == 3
    assert "is_home" in out.columns
    assert out["is_home"].isin([0, 1]).all()
