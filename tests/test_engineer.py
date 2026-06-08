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


@pytest.mark.unit
def test_inheritance_features_next_man_up():
    """``_build_inheritance_features`` (the production mirror of the validated A/B
    src/tuning/ab_history_token.py): when a higher-role same-position teammate is Out, the
    top-available teammate inherits the vacated role. RB ranks on ``snap_pct_raw``, WR on
    ``targets``; only same-position, same-team, higher-ranked OUT players count, and only
    for the top-available player. Role is prior-to-W (leakage-safe)."""
    from src.features.engineer import _build_inheritance_features

    rows = []
    # KC RBs weeks 1-2: A lead (snap .8), B second (.4), C third (.1). Week 3: A is Out.
    for wk in (1, 2):
        for pid, snap in (("A", 0.8), ("B", 0.4), ("C", 0.1)):
            rows.append(
                dict(
                    player_id=pid,
                    position="RB",
                    recent_team="KC",
                    season=2023,
                    week=wk,
                    snap_pct_raw=snap,
                    targets=0.0,
                )  # fmt: skip
            )
    for pid, snap in (("B", 0.45), ("C", 0.1)):  # week 3: A absent (Out); .45 must be ignored
        rows.append(
            dict(
                player_id=pid,
                position="RB",
                recent_team="KC",
                season=2023,
                week=3,
                snap_pct_raw=snap,
                targets=0.0,
            )  # fmt: skip
        )
    # KC WRs weeks 1-2: D lead (targets 9), E second (4). Week 3: D Out -> E inherits.
    for wk in (1, 2):
        for pid, tgt in (("D", 9.0), ("E", 4.0)):
            rows.append(
                dict(
                    player_id=pid,
                    position="WR",
                    recent_team="KC",
                    season=2023,
                    week=wk,
                    snap_pct_raw=0.0,
                    targets=tgt,
                )  # fmt: skip
            )
    rows.append(
        dict(
            player_id="E",
            position="WR",
            recent_team="KC",
            season=2023,
            week=3,
            snap_pct_raw=0.0,
            targets=5.0,
        )  # fmt: skip
    )
    df = pd.DataFrame(rows)
    inj = pd.DataFrame(
        [
            dict(gsis_id="A", position="RB", team="KC", season=2023, week=3, report_status="Out"),
            dict(gsis_id="D", position="WR", team="KC", season=2023, week=3, report_status="Out"),
        ]
    )

    g = _build_inheritance_features(df, inj).set_index(["player_id", "week"])

    # Week 2 (roles differentiated by week-1, nobody Out): the lead is top, no inheritance.
    assert g.loc[("A", 2), "is_top_available"] == 1.0
    assert g.loc[("B", 2), "is_top_available"] == 0.0
    assert g.loc[("A", 2), "inherited_opportunity"] == 0.0
    # Week 3 RB: A Out (prior role .8), B is now top-available and inherits A's role; C does not.
    assert g.loc[("B", 3), "is_top_available"] == 1.0
    assert g.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)
    assert g.loc[("C", 3), "is_top_available"] == 0.0
    assert g.loc[("C", 3), "inherited_opportunity"] == 0.0
    # Week 3 WR: D Out (prior targets-role 9), E inherits via the `targets` proxy.
    assert g.loc[("E", 3), "is_top_available"] == 1.0
    assert g.loc[("E", 3), "inherited_opportunity"] == pytest.approx(9.0)

    # injuries_df=None -> columns still emitted, inherited_opportunity all zero (no out-set).
    none_out = _build_inheritance_features(df, None)
    assert "is_top_available" in none_out.columns
    assert (none_out["inherited_opportunity"] == 0.0).all()
