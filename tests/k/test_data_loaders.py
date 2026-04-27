"""Coverage tests for ``K/k_data.py``.

Mocks ``nfl.import_pbp_data`` with synthetic per-play DataFrames so the
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
# Synthetic PBP frame — covers FG + XP rows and field required by the loader
# --------------------------------------------------------------------------


def _synthetic_pbp(season: int, n_fg: int = 6, n_xp: int = 4) -> pd.DataFrame:
    """Build a PBP-shaped DataFrame matching what ``nfl.import_pbp_data`` emits."""
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

    def _fake_pbp(seasons, downcast=True):
        return _synthetic_pbp(seasons[0])

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _fake_pbp)

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
def test_reconstruct_weekly_from_pbp_cache_hit(tmp_path, monkeypatch):
    """Pre-existing cache parquet → no PBP call, just a load-and-return."""
    import src.k.data as k_data

    # Pre-write the cache file.
    cache_path = tmp_path / "kicker_pbp_2021_2021.parquet"
    pd.DataFrame({"player_id": ["K01"], "season": [2021], "week": [1]}).to_parquet(cache_path)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("import_pbp_data was called despite cache hit")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _should_not_be_called)

    out = k_data.reconstruct_kicker_weekly_from_pbp([2021], cache_dir=str(tmp_path))
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "K01"


@pytest.mark.unit
def test_reconstruct_weekly_pbp_skips_failing_seasons(tmp_path, monkeypatch, capsys):
    """If ``import_pbp_data`` throws (e.g. upstream 502), the per-year body is
    skipped, a WARNING is logged, and the partial result is NOT cached so the
    next call doesn't treat a partial frame as authoritative."""
    import src.k.data as k_data

    def _bad(seasons, downcast=True):
        raise RuntimeError(f"pbp fetch boom for {seasons}")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _bad)

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

    def _selective(seasons, downcast=True):
        yr = seasons[0]
        if yr == 2021:
            raise RuntimeError("upstream 502 for 2021")
        return _synthetic_pbp(yr)

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _selective)

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
        k_data.nfl, "import_pbp_data", lambda seasons, downcast=True: _synthetic_pbp(seasons[0])
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
    import src.k.data as k_data

    cache_path = tmp_path / "kicker_kicks_pbp_2022_2022.parquet"
    pd.DataFrame(
        {"player_id": ["K01"], "season": [2022], "week": [1], "is_fg": [1], "is_xp": [0]}
    ).to_parquet(cache_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("import_pbp_data was called despite cache hit")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _should_not_be_called)
    out = k_data.reconstruct_kicker_kicks_from_pbp([2022], cache_dir=str(tmp_path))
    assert len(out) == 1


@pytest.mark.unit
def test_reconstruct_kicks_pbp_skips_failing_seasons(tmp_path, monkeypatch, capsys):
    """If ``import_pbp_data`` throws for a season, we log and continue."""
    import src.k.data as k_data

    def _bad(seasons, downcast=True):
        raise RuntimeError(f"pbp fetch boom for {seasons}")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _bad)
    # Every season fails, so the result is the empty frame the function returns.
    out = k_data.reconstruct_kicker_kicks_from_pbp([2020], cache_dir=str(tmp_path))
    assert out.empty
    assert "per-kick PBP extraction failed" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Tests — load_kicker_data (cache-hit shortcut path only; the full PBP path
# is exercised by reconstruct_* tests above).
# --------------------------------------------------------------------------


@pytest.fixture()
def _cached_pbp(tmp_path, monkeypatch):
    """Pre-write kicker_pbp cache so load_kicker_data skips the PBP chain."""
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("src.config.CACHE_DIR", str(tmp_path))

    # Restrict K_SEASONS so cache path key is predictable.
    monkeypatch.setattr(k_data, "K_SEASONS", [2022, 2023, 2024])
    monkeypatch.setattr(k_data, "K_MIN_GAMES", 1)  # don't filter our tiny frame

    # Enforce the docstring's "without touching nflverse" claim — without this
    # guard, a default-arg pitfall on cache_dir silently caused cache misses
    # and the test passed only because nflverse usually returned valid data.
    def _no_network(*args, **kwargs):
        raise AssertionError("nfl.import_pbp_data must not be called when the PBP cache hits")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _no_network)

    # Synthetic kicker weekly cache (2022-2024).
    kicker_rows = []
    for season in [2022, 2023, 2024]:
        for wk in range(1, 4):
            kicker_rows.append(
                {
                    "player_id": f"K{wk}",
                    "player_name": f"Kicker {wk}",
                    "recent_team": "KC",
                    "season": season,
                    "week": wk,
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
                    "max_fg_distance": 45.0,
                    "avg_fg_prob": 0.85,
                    "clutch_fg_att": 1,
                    "clutch_fg_made": 1,
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
                }
            )
    pd.DataFrame(kicker_rows).to_parquet(tmp_path / "kicker_pbp_2022_2024.parquet")

    # Schedule parquet — needs SEASONS[0]-SEASONS[-1] key to match load_kicker_data.
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
    """With a pre-written PBP cache and no 2025 weekly, load_kicker_data
    walks the cache-hit branch and merges schedules without touching nflverse."""
    from src.k.data import load_kicker_data

    df = load_kicker_data()
    assert len(df) > 0
    assert "is_home" in df.columns
    assert "implied_team_total" in df.columns
    # Every row must have total_line + implied_team_total post fillna.
    assert df["total_line"].notna().all()
    assert df["implied_team_total"].notna().all()


# --------------------------------------------------------------------------
# Tests — filter_to_k + kicker_season_split
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_to_k_is_identity():
    from src.k.data import filter_to_k

    df = pd.DataFrame({"player_id": ["K1"], "season": [2024], "week": [1]})
    out = filter_to_k(df)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df  # copy, not same ref


@pytest.mark.unit
def test_kicker_season_split_splits_by_year(capsys):
    from src.k.data import kicker_season_split

    df = pd.DataFrame(
        {
            "player_id": ["K1"] * 6,
            "season": [2022, 2023, 2024, 2025, 2023, 2025],
            "week": [1, 1, 1, 1, 2, 2],
        }
    )
    train, val, test = kicker_season_split(df)
    assert set(train["season"].unique()) <= {2022, 2023}
    assert set(val["season"].unique()) == {2024}
    assert set(test["season"].unique()) == {2025}
    # Print statement fires
    assert "K cross-season split" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Tests — load_kicker_kicks (minimal: cache-hit branch)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_backfill_2025_pbp_columns_updates_in_place(monkeypatch):
    """``_backfill_2025_pbp_columns`` must overwrite PBP-derived columns on
    rows whose ``season`` is in the backfill list — existing NaNs should
    become populated values once the fake PBP frame fires."""
    import src.k.data as k_data

    monkeypatch.setattr(
        k_data.nfl, "import_pbp_data", lambda seasons, downcast=True: _synthetic_pbp(seasons[0])
    )

    # k_df: 2 kickers x 2 weeks in 2025 with everything NaN/None
    rows = []
    for pid in ("K00", "K01"):
        for wk in (1, 2):
            rows.append(
                {
                    "player_id": pid,
                    "season": 2025,
                    "week": wk,
                    "avg_fg_distance": float("nan"),
                    "max_fg_distance": float("nan"),
                    "avg_fg_prob": float("nan"),
                    "clutch_fg_att": float("nan"),
                    "clutch_fg_made": float("nan"),
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
    """If ``import_pbp_data`` raises, _backfill logs a warning and leaves
    k_df untouched (swallowed by the outer try/except)."""
    import src.k.data as k_data

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _boom)

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
    """``load_kicker_data`` must walk the 2025-weekly branch when K_SEASONS
    contains 2025. We pre-seed the weekly parquet, skip real PBP via the
    cache-hit shortcut, and stub _backfill so it's a no-op."""
    import src.k.data as k_data
    from src.config import SEASONS

    monkeypatch.setattr(k_data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(k_data, "K_SEASONS", [2024, 2025])
    monkeypatch.setattr(k_data, "K_MIN_GAMES", 1)

    # Enforce the docstring's "skip real PBP via the cache-hit shortcut" claim:
    # neither reconstruct_kicker_weekly_from_pbp (cache-hit) nor _backfill_2025
    # (stubbed below) should reach the network.
    def _no_network(*args, **kwargs):
        raise AssertionError(
            "nfl.import_pbp_data must not be called: 2024 must hit cache, "
            "and the 2025 backfill is stubbed to no-op"
        )

    monkeypatch.setattr(k_data.nfl, "import_pbp_data", _no_network)

    # 2024 PBP cache
    pd.DataFrame(
        {
            "player_id": ["K01"],
            "player_name": ["K01"],
            "recent_team": ["KC"],
            "season": [2024],
            "week": [1],
            "position": ["K"],
            "season_type": ["REG"],
            "fg_att": [3],
            "fg_made": [2],
            "fg_missed": [1],
            "fg_made_0_19": [0],
            "fg_made_20_29": [1],
            "fg_made_30_39": [1],
            "fg_made_40_49": [0],
            "fg_made_50_59": [0],
            "fg_made_60_": [0],
            "fg_missed_40_49": [1],
            "fg_missed_50_59": [0],
            "fg_missed_60_": [0],
            "fg_yards_made": [55.0],
            "avg_fg_distance": [35.0],
            "max_fg_distance": [45.0],
            "avg_fg_prob": [0.85],
            "clutch_fg_att": [1],
            "clutch_fg_made": [1],
            "q4_fg_att": [1],
            "q4_fg_made": [1],
            "long_fg_att": [1],
            "long_fg_made": [0],
            "game_wind": [5.0],
            "game_temp": [60.0],
            "roof": ["outdoors"],
            "surface": ["grass"],
            "is_dome": [0],
            "pat_att": [3],
            "pat_made": [3],
            "pat_missed": [0],
        }
    ).to_parquet(tmp_path / "kicker_pbp_2024_2024.parquet")

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

    df = k_data.load_kicker_data()
    # Has rows from both 2024 (PBP) and 2025 (weekly).
    assert 2024 in df["season"].values
    assert 2025 in df["season"].values
    # 2025 weekly WR row must have been filtered out.
    assert (df["player_id"] != "WR01").all()


@pytest.mark.unit
def test_load_kicker_kicks_with_stubbed_reconstruct(monkeypatch):
    """load_kicker_kicks delegates to ``reconstruct_kicker_kicks_from_pbp``;
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
    out = k_data.load_kicker_kicks(k_df)
    assert len(out) == 3
    assert "is_home" in out.columns
    assert out["is_home"].isin([0, 1]).all()
