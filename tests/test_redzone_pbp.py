"""Coverage tests for ``src/data/redzone_pbp.py``.

Mocks ``nfl.import_pbp_data`` with synthetic per-play DataFrames so the
full PBP -> per-game-redzone-aggregate pipeline runs in-process. Pins the
schema-gated cache pattern (mirrors K's, see the
``[FIXED] K underprojection`` TODO archive entry — a cache that survives a
schema change silently zeros downstream features when ``fillna(0)`` runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# Synthetic PBP frame — covers rushing + receiving rows across yardline
# bands so the aggregation logic exercises every yardline gate (<=20, <=10,
# <=5) and the team-redzone-pass-attempts denominator.
# --------------------------------------------------------------------------


def _synthetic_pbp(season: int) -> pd.DataFrame:
    """Build a PBP-shaped DataFrame matching what ``nfl.import_pbp_data`` emits.

    Layout per (season, week):
      * KC week 1: two rushes by RB-A (yardlines 4, 15) + two pass attempts to
        WR-X (yardlines 8, 18) + one pass attempt to WR-Y (yardline 30, NOT RZ).
      * KC week 2: one rush by RB-A (yardline 22, NOT RZ) + one rush by RB-B
        (yardline 3) + one pass attempt to WR-Y (yardline 12).
      * BUF week 1: one rush by RB-C (yardline 50, NOT RZ) + zero pass attempts
        in the RZ (lets us assert redzone_target_share=0 when denom=0).
      * POST/playoff row (must be filtered out by season_type).
    """
    rows = []

    def row(**kw):
        base = {
            "season": season,
            "season_type": "REG",
            "rusher_player_id": np.nan,
            "receiver_player_id": np.nan,
            "pass_attempt": 0,
            "yardline_100": np.nan,
        }
        base.update(kw)
        rows.append(base)

    # --- KC week 1 ---
    row(week=1, posteam="KC", rusher_player_id="RB-A", yardline_100=4)  # inside 5
    row(week=1, posteam="KC", rusher_player_id="RB-A", yardline_100=15)  # in RZ, not in 10
    row(week=1, posteam="KC", receiver_player_id="WR-X", pass_attempt=1, yardline_100=8)
    row(week=1, posteam="KC", receiver_player_id="WR-X", pass_attempt=1, yardline_100=18)
    row(week=1, posteam="KC", receiver_player_id="WR-Y", pass_attempt=1, yardline_100=30)

    # --- KC week 2 ---
    row(week=2, posteam="KC", rusher_player_id="RB-A", yardline_100=22)  # OUT of RZ
    row(week=2, posteam="KC", rusher_player_id="RB-B", yardline_100=3)  # inside 5
    row(week=2, posteam="KC", receiver_player_id="WR-Y", pass_attempt=1, yardline_100=12)

    # --- BUF week 1 ---
    row(week=1, posteam="BUF", rusher_player_id="RB-C", yardline_100=50)

    # --- Playoff row that must be filtered ---
    row(
        season_type="POST",
        week=20,
        posteam="KC",
        rusher_player_id="RB-A",
        yardline_100=1,
    )

    return pd.DataFrame(rows)


def _redzone_cache_row(
    player_id: str,
    season: int,
    week: int,
    recent_team: str = "KC",
) -> dict:
    """One row matching the schema written by ``reconstruct_redzone_from_pbp``.

    Tests pre-write a parquet of these to exercise the cache-hit branch
    without invoking the real PBP aggregation.
    """
    return {
        "player_id": player_id,
        "season": season,
        "week": week,
        "recent_team": recent_team,
        "redzone_carries": 1,
        "redzone_targets": 1,
        "inside10_carries": 1,
        "inside5_carries": 0,
        "redzone_target_share": 0.25,
    }


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_reconstruct_redzone_from_pbp_happy_path(tmp_path, monkeypatch):
    """Aggregation arithmetic must match hand-computed expectations for the
    synthetic frame (rushing + receiving yardline gates, RZ denominator)."""
    import src.data.redzone_pbp as rz

    monkeypatch.setattr(
        rz.nfl, "import_pbp_data", lambda seasons, downcast=True: _synthetic_pbp(seasons[0])
    )

    out = rz.reconstruct_redzone_from_pbp([2020], cache_dir=str(tmp_path))

    # Cache file must now exist.
    assert (tmp_path / "redzone_pbp_2020_2020.parquet").exists()

    # Every required schema column lands.
    required = {
        "player_id",
        "season",
        "week",
        "recent_team",
        "redzone_carries",
        "redzone_targets",
        "inside10_carries",
        "inside5_carries",
        "redzone_target_share",
    }
    assert required <= set(out.columns)

    # Playoff (season_type='POST') row must be filtered.
    rb_a_post = out[(out["player_id"] == "RB-A") & (out["week"] == 20)]
    assert len(rb_a_post) == 0

    # RB-A week 1 KC: 2 rushes (y=4, y=15) -> rz_carries=2, in10=1, in5=1.
    rb_a_w1 = out[
        (out["player_id"] == "RB-A") & (out["week"] == 1) & (out["recent_team"] == "KC")
    ].iloc[0]
    assert rb_a_w1["redzone_carries"] == 2
    assert rb_a_w1["inside10_carries"] == 1
    assert rb_a_w1["inside5_carries"] == 1

    # WR-X week 1 KC: 2 RZ targets (y=8, y=18).
    wr_x_w1 = out[(out["player_id"] == "WR-X") & (out["week"] == 1)].iloc[0]
    assert wr_x_w1["redzone_targets"] == 2

    # WR-Y week 1 KC: y=30, NOT in RZ -> redzone_targets=0.
    wr_y_w1 = out[
        (out["player_id"] == "WR-Y") & (out["week"] == 1) & (out["recent_team"] == "KC")
    ].iloc[0]
    assert wr_y_w1["redzone_targets"] == 0

    # Team RZ-pass-attempts for KC week 1 = 2 (WR-X x2; WR-Y at y=30 doesn't
    # qualify). So WR-X redzone_target_share = 2/2 = 1.0.
    assert wr_x_w1["redzone_target_share"] == pytest.approx(1.0)


@pytest.mark.unit
def test_redzone_target_share_zero_when_team_has_no_redzone_passes(tmp_path, monkeypatch):
    """BUF week 1 has zero RZ pass attempts (only a non-RZ rush). The
    rushing-only row for RB-C must carry redzone_target_share=0, not NaN."""
    import src.data.redzone_pbp as rz

    monkeypatch.setattr(
        rz.nfl, "import_pbp_data", lambda seasons, downcast=True: _synthetic_pbp(seasons[0])
    )

    out = rz.reconstruct_redzone_from_pbp([2020], cache_dir=str(tmp_path))

    rb_c = out[out["player_id"] == "RB-C"]
    assert len(rb_c) == 1
    # RB-C had no RZ targets and BUF had 0 RZ pass attempts; share = 0.
    assert rb_c.iloc[0]["redzone_target_share"] == pytest.approx(0.0)


@pytest.mark.unit
def test_reconstruct_redzone_from_pbp_cache_hit(tmp_path, monkeypatch):
    """Pre-existing cache parquet with the current schema -> no PBP call, just
    a load-and-return."""
    import src.data.redzone_pbp as rz

    cache_path = tmp_path / "redzone_pbp_2021_2021.parquet"
    pd.DataFrame([_redzone_cache_row("RB-A", 2021, 1)]).to_parquet(cache_path)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("import_pbp_data was called despite cache hit")

    monkeypatch.setattr(rz.nfl, "import_pbp_data", _should_not_be_called)

    out = rz.reconstruct_redzone_from_pbp([2021], cache_dir=str(tmp_path))
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "RB-A"


@pytest.mark.unit
def test_reconstruct_redzone_stale_cache_regenerates(tmp_path, monkeypatch, capsys):
    """A cache parquet missing a required column must be ignored and the PBP
    path re-run. Regression guard for the same class of bug as the
    ``[FIXED] K underprojection`` TODO archive entry, where a cache survived
    a schema addition and silently zeroed targets across the training range."""
    import src.data.redzone_pbp as rz

    # Pre-write a parquet that looks plausible but is missing redzone_target_share.
    stale_cache = tmp_path / "redzone_pbp_2020_2020.parquet"
    pd.DataFrame(
        {
            "player_id": ["RB-A"],
            "season": [2020],
            "week": [1],
            "recent_team": ["KC"],
            "redzone_carries": [1],
            "redzone_targets": [0],
            "inside10_carries": [0],
            "inside5_carries": [0],
            # redzone_target_share intentionally missing
        }
    ).to_parquet(stale_cache)

    monkeypatch.setattr(
        rz.nfl, "import_pbp_data", lambda seasons, downcast=True: _synthetic_pbp(seasons[0])
    )

    out = rz.reconstruct_redzone_from_pbp([2020], cache_dir=str(tmp_path))

    # The PBP path ran — output reflects the synthetic frame's multiple rows,
    # not the 1-row stale cache.
    assert len(out) > 1
    assert "redzone_target_share" in out.columns

    # Log line surfaces the bad schema so future debugging is obvious.
    captured = capsys.readouterr().out
    assert "Stale cache" in captured
    assert "redzone_target_share" in captured


@pytest.mark.unit
def test_reconstruct_redzone_skips_failing_seasons(tmp_path, monkeypatch, capsys):
    """If ``import_pbp_data`` throws for every season (e.g. upstream 502), the
    partial result is NOT cached so the next call doesn't treat it as
    authoritative. Returns empty frame with the full schema."""
    import src.data.redzone_pbp as rz

    def _bad(seasons, downcast=True):
        raise RuntimeError(f"pbp fetch boom for {seasons}")

    monkeypatch.setattr(rz.nfl, "import_pbp_data", _bad)

    out = rz.reconstruct_redzone_from_pbp([2020], cache_dir=str(tmp_path))

    # No cache file written (partial result must not poison the cache).
    assert not (tmp_path / "redzone_pbp_2020_2020.parquet").exists()

    # Returns an empty frame with the required schema so callers can merge
    # without crashing.
    assert out.empty
    required = {
        "player_id",
        "season",
        "week",
        "recent_team",
        "redzone_carries",
        "redzone_targets",
        "inside10_carries",
        "inside5_carries",
        "redzone_target_share",
    }
    assert required <= set(out.columns)

    captured = capsys.readouterr().out
    assert "red-zone PBP extraction failed" in captured


@pytest.mark.unit
def test_reconstruct_redzone_partial_failure_does_not_cache(tmp_path, monkeypatch, capsys):
    """If at least one season fetch fails, the loader emits the partial result
    but does NOT cache it (same poisoning-prevention rule as K's loader)."""
    import src.data.redzone_pbp as rz

    call_count = {"n": 0}

    def _flaky(seasons, downcast=True):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _synthetic_pbp(seasons[0])
        raise RuntimeError("upstream 502")

    monkeypatch.setattr(rz.nfl, "import_pbp_data", _flaky)

    out = rz.reconstruct_redzone_from_pbp([2020, 2021], cache_dir=str(tmp_path))

    # First season's rows present; second season skipped.
    assert (out["season"] == 2020).any()
    assert not (out["season"] == 2021).any()
    # Partial result must not be cached.
    assert not (tmp_path / "redzone_pbp_2020_2021.parquet").exists()
    captured = capsys.readouterr().out
    assert "Skipped seasons [2021]" in captured
