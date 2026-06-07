"""Tests for ``src/features/engineer.py`` helpers not covered by the
position-specific or opp-defense suites."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import GAME_HISTORY_STATS, build_features, build_game_history_arrays


def _traded_player_frame() -> pd.DataFrame:
    """One WR, four 2023 games, traded KC->BUF after week 2 (a mid-season
    ``stint`` change). snap_pct rises 0.5->0.8 across the weeks."""
    rows = []
    for wk, (team, snap) in enumerate([("KC", 0.5), ("KC", 0.6), ("BUF", 0.7), ("BUF", 0.8)], 1):
        rows.append(
            dict(
                player_id="P1",
                player_name="P1",
                position="WR",
                season=2023,
                week=wk,
                recent_team=team,
                opponent_team="DAL",
                snap_pct=snap,
                targets=5,
                carries=0,
                receptions=3,
                receiving_yards=40,
                rushing_yards=0,
                passing_yards=0,
                attempts=0,
                completions=0,
                interceptions=0,
                fumbles_lost=0,
                passing_tds=0,
                rushing_tds=0,
                receiving_tds=0,
                receiving_air_yards=50,
                receiving_yards_after_catch=20,
                receiving_first_downs=2,
                receiving_epa=1.0,
                rushing_epa=0.0,
                rushing_first_downs=0,
                sacks=0,
                sack_yards=0,
                fantasy_points=7.0,
            )
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_build_game_history_arrays_raises_on_missing_columns():
    """Requesting a history stat that isn't in ``df.columns`` now raises a
    KeyError naming the missing columns and pointing at the splits-regen
    workflow. Before this guard, missing columns were silently filtered and
    the trained ``game_encoder.0.weight`` shape diverged from the inference
    registry's ``game_dim``, surfacing only at smoke-test time (PR #235's
    redzone-history stats vs. an Apr-16 splits parquet)."""
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2023, 2023],
            "week": [1, 2],
            "rushing_yards": [50.0, 70.0],
        }
    )
    with pytest.raises(KeyError, match=r"history_stats columns missing from df"):
        build_game_history_arrays(
            df,
            history_stats=["rushing_yards", "redzone_carries", "redzone_targets"],
            max_seq_len=4,
        )


@pytest.mark.unit
def test_build_game_history_arrays_error_message_lists_missing_and_points_at_regen():
    """The KeyError body must (a) list the missing columns so the operator
    knows exactly which cols to add and (b) point at refresh-splits.yml /
    SETUP.md so the fix path is one click away."""
    df = pd.DataFrame(
        {
            "player_id": ["p1"],
            "season": [2023],
            "week": [1],
            "rushing_yards": [10.0],
        }
    )
    with pytest.raises(KeyError) as exc:
        build_game_history_arrays(
            df,
            history_stats=["rushing_yards", "redzone_carries"],
            max_seq_len=2,
        )
    message = str(exc.value)
    assert "redzone_carries" in message
    assert "refresh-splits" in message or "SETUP.md" in message
    # ``attn_history_stats`` (the PositionConfig field name) must appear so an
    # operator who greps for the config field lands on this error.
    assert "attn_history_stats" in message


@pytest.mark.unit
def test_build_game_history_arrays_empty_df_with_missing_cols_raises():
    """The missing-cols check runs BEFORE the ``n == 0`` early return, so an
    empty DataFrame whose schema doesn't include the requested stats still
    raises rather than silently returning zero arrays. Locks in the (correct)
    new semantics: a configuration error doesn't get hidden by an empty
    split."""
    df = pd.DataFrame()  # no rows, no columns
    with pytest.raises(KeyError, match=r"attn_history_stats columns missing"):
        build_game_history_arrays(
            df,
            history_stats=["rushing_yards"],
            max_seq_len=4,
        )


@pytest.mark.unit
def test_build_game_history_arrays_empty_df_with_cols_returns_zero_shape():
    """Empty rows but required columns present → returns zero-shaped arrays.
    Distinguishes the legitimate empty-split case from the misconfiguration
    case in the previous test."""
    df = pd.DataFrame({"player_id": [], "season": [], "week": [], "rushing_yards": []})
    X_history, mask = build_game_history_arrays(
        df,
        history_stats=["rushing_yards"],
        max_seq_len=4,
    )
    assert X_history.shape == (0, 4, 1)
    assert mask.shape == (0, 4)


@pytest.mark.unit
def test_build_game_history_arrays_happy_path_unchanged():
    """Sanity check that the guard doesn't alter the happy-path return
    shape: when every requested col exists, the function returns
    ``(X_history, history_mask)`` with the expected dimensions."""
    df = pd.DataFrame(
        {
            "player_id": ["p1", "p1", "p1"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "rushing_yards": [10.0, 20.0, 30.0],
            "receiving_yards": [5.0, 15.0, 25.0],
        }
    )
    X_history, mask = build_game_history_arrays(
        df,
        history_stats=["rushing_yards", "receiving_yards"],
        max_seq_len=5,
    )
    assert X_history.shape == (3, 5, 2)
    assert X_history.dtype == np.float32
    assert mask.shape == (3, 5)
    assert mask.dtype == np.bool_


@pytest.mark.unit
def test_snap_pct_static_lag_is_stint_aware():
    """#677: the static ``snap_pct`` (prior week's snap %) must reset at a
    mid-season team change. A traded player's first game with the new team must
    NOT inherit the old team's snap share — the old non-stint-aware
    ``groupby(player_id, season).shift(1)`` carried it over."""
    out = build_features(_traded_player_frame()).sort_values("week")
    # Raw per-game values preserved verbatim (these feed the attention history).
    assert "snap_pct_raw" in out.columns
    np.testing.assert_allclose(out["snap_pct_raw"].to_numpy(), [0.5, 0.6, 0.7, 0.8], atol=1e-9)
    # Static feature = stint-aware prior-week snap %. Week 3 is the first BUF
    # game: stint-aware -> 0.0 (no prior game in the BUF stint), NOT 0.6 (the
    # KC week-2 carryover the non-stint-aware lag produced).
    np.testing.assert_allclose(out["snap_pct"].to_numpy(), [0.0, 0.5, 0.0, 0.7], atol=1e-9)
    week3 = out[out["week"] == 3]["snap_pct"].iloc[0]
    assert week3 == 0.0  # explicit guard against the old carryover (would be 0.6)


@pytest.mark.unit
def test_snap_pct_history_uses_raw_not_prelagged():
    """#788: the attention game-history default uses ``snap_pct_raw`` (un-lagged)
    rather than the pre-lagged static ``snap_pct``. ``build_game_history_arrays``
    applies its own leakage-safe shift, so feeding it the lagged column would
    double-lag the snap-share signal in the sequence."""
    assert "snap_pct_raw" in GAME_HISTORY_STATS
    assert "snap_pct" not in GAME_HISTORY_STATS
