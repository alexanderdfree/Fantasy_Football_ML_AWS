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


def _traded_with_varying_stats_frame() -> pd.DataFrame:
    """Traded WR ``P1`` (KC weeks 1-2 → BUF weeks 3-4) with *distinct* per-week
    stats so a stint-aware trailing feature differs visibly from the old
    cross-team blend, plus a teammate ``T1`` on each team so air-yard shares
    aren't trivially 1.0.

    P1 targets: KC 10, 20 → BUF 2, 4. Air yards (P1, T1):
    KC wk1 (90, 10), wk2 (80, 20); BUF wk3 (30, 70), wk4 (40, 60) → P1 raw
    air-yards share 0.90, 0.80, 0.30, 0.40.
    """
    spec = [
        # (player_id, week, team, targets, p_air_yards, team_air_yards_partner)
        ("P1", 1, "KC", 10, 90),
        ("T1", 1, "KC", 1, 10),
        ("P1", 2, "KC", 20, 80),
        ("T1", 2, "KC", 1, 20),
        ("P1", 3, "BUF", 2, 30),
        ("T1", 3, "BUF", 1, 70),
        ("P1", 4, "BUF", 4, 40),
        ("T1", 4, "BUF", 1, 60),
    ]
    rows = []
    for pid, wk, team, tgts, air in spec:
        rows.append(
            dict(
                player_id=pid,
                player_name=pid,
                position="WR",
                season=2023,
                week=wk,
                recent_team=team,
                opponent_team="DAL",
                snap_pct=0.6,
                targets=tgts,
                carries=0,
                receptions=2,
                receiving_yards=float(air) * 0.6,
                rushing_yards=0,
                passing_yards=0,
                attempts=0,
                completions=0,
                interceptions=0,
                fumbles_lost=0,
                passing_tds=0,
                rushing_tds=0,
                receiving_tds=0,
                receiving_air_yards=float(air),
                receiving_yards_after_catch=10,
                receiving_first_downs=1,
                receiving_epa=1.0,
                rushing_epa=0.0,
                rushing_first_downs=0,
                sacks=0,
                sack_yards=0,
                fantasy_points=float(tgts),
            )
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_rolling_features_are_stint_aware():
    """#722: rolling / EWMA / trend windowed features must reset at a mid-season
    team change. A traded player's first weeks at the new team must NOT blend
    the prior team's trailing games. Positive control on ``P1`` (KC 10, 20 →
    BUF 2, 4 targets):

      * week 3 (first BUF game): stint-aware ``rolling_mean_targets_L3`` has no
        prior BUF game → NaN. The OLD non-stint-aware grouping produced the KC
        blend (10+20)/2 = 15.0.
      * week 4: stint-aware mean = the lone prior BUF game (2.0). The OLD
        grouping blended KC: (10+20+2)/3 ≈ 10.67.
    """
    out = build_features(_traded_with_varying_stats_frame())
    p1 = out[out["player_id"] == "P1"].sort_values("week")
    by_week = dict(zip(p1["week"], p1["rolling_mean_targets_L3"], strict=True))

    # Week 3: stint-fresh → NaN (no prior game in the BUF stint), NOT the 15.0
    # KC blend the old groupby produced.
    assert pd.isna(by_week[3])
    # Week 4: only the BUF week-3 game (targets=2) is in the trailing window —
    # NOT 10.67, which would require blending the KC weeks.
    assert by_week[4] == pytest.approx(2.0)
    # EWMA resets too (week 4 sees only BUF week 3 → 2.0, not a KC-blended value).
    ewma4 = p1[p1["week"] == 4]["ewma_targets_L3"].iloc[0]
    assert ewma4 == pytest.approx(2.0)


@pytest.mark.unit
def test_air_yards_share_lag_is_stint_aware():
    """#666: the ``air_yards_share`` lag must reset at a mid-season team change,
    matching its sibling share features (``target_share`` / ``carry_share``). A
    traded player's first game with the new team must NOT carry the previous
    team's last-week air-yards share. P1 raw air-yards share = 0.90, 0.80
    (KC) → 0.30, 0.40 (BUF); the lagged feature:

      * week 3 (first BUF game): stint-aware → 0.0 (no prior BUF game), NOT the
        0.80 KC week-2 carryover the old non-stint-aware lag produced.
      * week 4: the lagged BUF week-3 raw share (0.30).
    """
    out = build_features(_traded_with_varying_stats_frame())
    p1 = out[out["player_id"] == "P1"].sort_values("week")
    by_week = dict(zip(p1["week"], p1["air_yards_share"], strict=True))

    assert by_week[3] == pytest.approx(0.0)  # stint-fresh, NOT 0.80 (KC carryover)
    assert by_week[4] == pytest.approx(0.30)  # lagged BUF week-3 raw share


@pytest.mark.unit
def test_build_features_routes_schedule_reads_through_module_attr(monkeypatch):
    """build_features must reach schedules via the ``weather_features`` MODULE
    attribute, not a from-import copy of ``_load_schedules``: the
    tests/conftest.py session stub (and any per-test patch) only rebinds the
    module attr. The old copy bypassed patches and — once another test cleared
    ``_schedule_cache`` (e.g. test_weather_features' raw-``None`` reset) — read
    the real ``data/raw`` parquet: ``FileNotFoundError`` on boxes without raw
    data, surfacing as the sporadic per-worker "xdist race" flake."""
    import src.shared.weather_features as wf

    calls = []
    orig = wf._load_schedules

    def _spy():
        calls.append(1)
        return orig()

    monkeypatch.setattr(wf, "_load_schedules", _spy)
    # Force the function path (a populated module cache would mask a bypassing
    # from-import copy on boxes where the real parquet exists).
    monkeypatch.setattr(wf, "_schedule_cache", None)

    build_features(_traded_player_frame())

    assert calls, "build_features bypassed the patched weather_features._load_schedules"


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
    top-available teammate inherits the vacated role. RB ranks on ``snap_pct_raw``, WR/TE on
    ``targets``, QB on ff_opportunity ``total_fantasy_points_exp``; only same-position, same-team,
    higher-ranked OUT players count, and only for the top-available player. Role is prior-to-W."""
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
    # KC QBs weeks 1-2: P starter (exp-FP 18.0), Q backup (2.0). Week 3: P Out -> Q inherits via
    # the ff_opportunity ``total_fantasy_points_exp`` proxy (a QB's role isn't snap%/targets).
    for wk in (1, 2):
        for pid, exp in (("P", 18.0), ("Q", 2.0)):
            rows.append(
                dict(
                    player_id=pid,
                    position="QB",
                    recent_team="KC",
                    season=2023,
                    week=wk,
                    total_fantasy_points_exp=exp,
                )  # fmt: skip
            )
    rows.append(
        dict(
            player_id="Q",
            position="QB",
            recent_team="KC",
            season=2023,
            week=3,
            total_fantasy_points_exp=2.5,
        )  # fmt: skip
    )
    df = pd.DataFrame(rows)
    inj = pd.DataFrame(
        [
            dict(gsis_id="A", position="RB", team="KC", season=2023, week=3, report_status="Out"),
            dict(gsis_id="D", position="WR", team="KC", season=2023, week=3, report_status="Out"),
            dict(gsis_id="P", position="QB", team="KC", season=2023, week=3, report_status="Out"),
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
    # Week 3 QB: P Out (prior exp-FP role 18.0), Q is now top-available and inherits via the
    # ff_opportunity ``total_fantasy_points_exp`` proxy.
    assert g.loc[("P", 2), "is_top_available"] == 1.0
    assert g.loc[("Q", 2), "inherited_opportunity"] == 0.0
    assert g.loc[("Q", 3), "is_top_available"] == 1.0
    assert g.loc[("Q", 3), "inherited_opportunity"] == pytest.approx(18.0)

    # injuries_df=None -> columns still emitted, inherited_opportunity all zero (no out-set).
    none_out = _build_inheritance_features(df, None)
    assert "is_top_available" in none_out.columns
    assert (none_out["inherited_opportunity"] == 0.0).all()


def test_inheritance_ir_roster_status_out_set():
    """#1106: a starter lost to IR (or scratched on game day) is absent from BOTH the weekly frame
    and the injury report, so the vacancy is invisible unless rosters ``status`` in {"RES", "INA"}
    is folded into the out-set. With ``rosters_df`` the next man up inherits the out starter's role
    (for RES and INA); a truly-active status stays 0; without ``rosters_df`` it stays 0."""
    from src.features.engineer import _build_inheritance_features

    rows = []
    # KC RBs: A leads (snap .8) weeks 1-2, B backs up (.4) weeks 1-3. Week 3: A is on IR -> no
    # weekly row and NOT on the injury report; B plays and should inherit A's vacated role.
    for wk in (1, 2):
        rows.append(
            dict(
                player_id="A",
                position="RB",
                recent_team="KC",
                season=2023,
                week=wk,
                snap_pct_raw=0.8,
                targets=0.0,
            )  # fmt: skip
        )
    for wk in (1, 2, 3):
        rows.append(
            dict(
                player_id="B",
                position="RB",
                recent_team="KC",
                season=2023,
                week=wk,
                snap_pct_raw=0.4,
                targets=0.0,
            )  # fmt: skip
        )
    df = pd.DataFrame(rows)
    # Only the RES row matters; A on IR in week 3.
    rosters = pd.DataFrame(
        [dict(player_id="A", position="RB", team="KC", season=2023, week=3, status="RES")]
    )

    # No injury report -> the only out-set source is rosters RES.
    g = _build_inheritance_features(df, None, rosters_df=rosters).set_index(["player_id", "week"])
    assert g.loc[("B", 3), "is_top_available"] == 1.0
    assert g.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)

    # Without rosters_df the IR vacancy is invisible -> inheritance stays 0 (old behavior).
    g0 = _build_inheritance_features(df, None).set_index(["player_id", "week"])
    assert g0.loc[("B", 3), "inherited_opportunity"] == 0.0

    # Game-day INA is also folded in (leakage-audited pre-kickoff inactives list).
    ina = rosters.assign(status="INA")
    g1 = _build_inheritance_features(df, None, rosters_df=ina).set_index(["player_id", "week"])
    assert g1.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)

    # A truly-active status is still excluded.
    act = rosters.assign(status="ACT")
    g2 = _build_inheritance_features(df, None, rosters_df=act).set_index(["player_id", "week"])
    assert g2.loc[("B", 3), "inherited_opportunity"] == 0.0


def test_inheritance_out_set_normalizes_legacy_team_codes():
    """#1269: raw injuries/rosters carry pre-relocation codes (OAK/SD/STL) for
    relocated-franchise seasons while the weekly frame's ``recent_team`` is
    retroactively modern (LV/LAC/LA), so the out-set key must be normalized —
    an unnormalized key silently never matches and ``inherited_opportunity``
    stays 0 exactly when a starter is Out."""
    from src.features.engineer import _build_inheritance_features

    def _rb_frame(team, season):
        rows = []
        # A leads (snap .8) weeks 1-2; B backs up (.4) weeks 1-3; A absent week 3.
        for wk in (1, 2):
            for pid, snap in (("A", 0.8), ("B", 0.4)):
                rows.append(
                    dict(
                        player_id=pid,
                        position="RB",
                        recent_team=team,
                        season=season,
                        week=wk,
                        snap_pct_raw=snap,
                        targets=0.0,
                    )  # fmt: skip
                )
        rows.append(
            dict(
                player_id="B",
                position="RB",
                recent_team=team,
                season=season,
                week=3,
                snap_pct_raw=0.45,
                targets=0.0,
            )  # fmt: skip
        )
        return pd.DataFrame(rows)

    # 2019 Raiders: weekly recent_team is retroactively "LV"; the raw injury
    # report says "OAK". A Out week 3 → B must still inherit A's 0.8 role.
    df = _rb_frame("LV", 2019)
    inj = pd.DataFrame(
        [dict(gsis_id="A", position="RB", team="OAK", season=2019, week=3, report_status="Out")]
    )
    g = _build_inheritance_features(df, inj).set_index(["player_id", "week"])
    assert g.loc[("B", 3), "is_top_available"] == 1.0
    assert g.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)

    # Same through the rosters RES/INA out-set: 2016 Chargers ("SD" → "LAC").
    df2 = _rb_frame("LAC", 2016)
    rosters = pd.DataFrame(
        [dict(player_id="A", position="RB", team="SD", season=2016, week=3, status="RES")]
    )
    g2 = _build_inheritance_features(df2, None, rosters_df=rosters).set_index(["player_id", "week"])
    assert g2.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)

    # 2012-2015 rosters use GAMEBOOK codes the shared relocation map never sees:
    # the Rams roster code is "SL" (not "STL"), plus ARZ/BLT/CLV/HST. The
    # out-set map must cover those too (#1269 review find — 5 teams × 2013-2015
    # otherwise kept the silently-never-matches failure).
    for roster_code, weekly_code in (("SL", "LA"), ("ARZ", "ARI")):
        df_g = _rb_frame(weekly_code, 2014)
        rosters_g = pd.DataFrame(
            [
                dict(
                    player_id="A",
                    position="RB",
                    team=roster_code,
                    season=2014,
                    week=3,
                    status="RES",
                )
            ]
        )
        g_g = _build_inheritance_features(df_g, None, rosters_df=rosters_g).set_index(
            ["player_id", "week"]
        )
        assert g_g.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8), roster_code

    # Modern codes pass through the normalization unchanged (identity on non-legacy).
    df3 = _rb_frame("KC", 2023)
    inj3 = pd.DataFrame(
        [dict(gsis_id="A", position="RB", team="KC", season=2023, week=3, report_status="Out")]
    )
    g3 = _build_inheritance_features(df3, inj3).set_index(["player_id", "week"])
    assert g3.loc[("B", 3), "inherited_opportunity"] == pytest.approx(0.8)


def test_inheritance_prior_season_fallback_week1():
    """#1106 finding B: in Week 1 (no current-season history) the within-season expanding role is
    identically 0, which silently zeroed the vacancy signal. ``role_before`` now falls back to the
    player's prior-season mean role, so a newly-out prior-season starter is still inherited by the
    next man up on the opener."""
    from src.features.engineer import _build_inheritance_features

    rows = []
    # Prior season (2022): A is the lead RB (snap .8), B the backup (.4).
    for wk in (1, 2, 3):
        for pid, snap in (("A", 0.8), ("B", 0.4)):
            rows.append(
                dict(
                    player_id=pid,
                    position="RB",
                    recent_team="KC",
                    season=2022,
                    week=wk,
                    snap_pct_raw=snap,
                    targets=0.0,
                )  # fmt: skip
            )
    # 2023 Week-1 opener: A is Out (absent), B plays. No current-season history exists yet.
    rows.append(
        dict(
            player_id="B",
            position="RB",
            recent_team="KC",
            season=2023,
            week=1,
            snap_pct_raw=0.5,
            targets=0.0,
        )  # fmt: skip
    )
    df = pd.DataFrame(rows)
    inj = pd.DataFrame(
        [dict(gsis_id="A", position="RB", team="KC", season=2023, week=1, report_status="Out")]
    )

    g = _build_inheritance_features(df, inj).set_index(["player_id", "season", "week"])
    # B is the only present back -> top-available; A (prior-season role .8) is Out and ranked above
    # B (prior-season role .4), so B inherits A's prior-season role on the opener.
    assert g.loc[("B", 2023, 1), "is_top_available"] == 1.0
    assert g.loc[("B", 2023, 1), "inherited_opportunity"] == pytest.approx(0.8)

    # Without a prior season the fallback is absent -> Week-1 inheritance stays 0 (old behavior).
    df_no_prior = df[df["season"] == 2023].copy()
    g2 = _build_inheritance_features(df_no_prior, inj).set_index(["player_id", "season", "week"])
    assert g2.loc[("B", 2023, 1), "inherited_opportunity"] == 0.0


def test_inheritance_season_granular_rosters_warns_and_noops(caplog):
    """Granularity tripwire (#1106 activation bug, 2026-06-11): the SEASONAL
    load_rosters frame also carries week/status columns, so the column guard
    passes — but with ~1 row per player-season the out-set registers ~no
    vacancies. The block must warn loudly so the inert path is visible."""
    import logging

    from src.features.engineer import _build_inheritance_features

    rows = []
    for wk in (1, 2):
        rows.append(
            dict(
                player_id="A",
                position="RB",
                recent_team="KC",
                season=2023,
                week=wk,
                snap_pct_raw=0.8,
                targets=0.0,
            )  # fmt: skip
        )
    for wk in (1, 2, 3):
        rows.append(
            dict(
                player_id="B",
                position="RB",
                recent_team="KC",
                season=2023,
                week=wk,
                snap_pct_raw=0.4,
                targets=0.0,
            )  # fmt: skip
        )
    df = pd.DataFrame(rows)
    # Season-granular: ONE row for A with a week column (the seasonal frame
    # shape) — has every guard column yet cannot describe the week-3 absence.
    seasonal = pd.DataFrame(
        [dict(player_id="A", position="RB", team="KC", season=2023, week=1, status="RES")]
    )

    with caplog.at_level(logging.WARNING, logger="src.features.engineer"):
        g = _build_inheritance_features(df, None, rosters_df=seasonal).set_index(
            ["player_id", "week"]
        )
    assert any("season-granular" in r.message for r in caplog.records)
    # And the missing-columns variant warns too (no silent skip).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.features.engineer"):
        _build_inheritance_features(df, None, rosters_df=seasonal.drop(columns=["week"]))
    assert any("missing columns" in r.message for r in caplog.records)
    # Week-3 vacancy invisible to the seasonal frame -> B inherits nothing.
    assert g.loc[("B", 3), "inherited_opportunity"] == 0.0


# --- Cold-start opp_* prior-season fallback (_fill_opp_prior_season) ----------


@pytest.mark.unit
def test_fill_opp_prior_season_backs_opener_with_prior_mean():
    """The helper fills only season-opener NaNs, with the *prior season's*
    per-game mean of the base stat for the same entity (leakage-safe: the S
    fill must use S-1 data ONLY, never same-season rows), leaves non-NaN rows
    untouched, leaves a no-prior opener NaN (catch-all 0 is the last resort),
    and drops its temp column."""
    from src.features.engineer import _fill_opp_prior_season

    agg = pd.DataFrame(
        {
            "entity": ["A", "A", "A", "A"],
            "season": [2020, 2020, 2021, 2021],
            "week": [1, 2, 1, 2],
            "base": [10.0, 20.0, 30.0, 40.0],
            # within-season shift(1).rolling: openers (week 1) are NaN
            "out": [np.nan, 10.0, np.nan, 30.0],
        }
    )
    res = _fill_opp_prior_season(agg, ["entity"], {"out": "base"}).set_index(["season", "week"])

    # 2021 opener backed by the 2020 per-game mean = (10+20)/2 = 15.0.
    # That it is 15.0 (not 25.0 = mean over all four rows) proves the fill uses
    # S-1 data only — no same-season / future leakage.
    assert res.loc[(2021, 1), "out"] == pytest.approx(15.0)
    # First-season opener has no S-1 -> stays NaN (flows to the catch-all 0).
    assert pd.isna(res.loc[(2020, 1), "out"])
    # Non-opener rows are untouched (keep their within-season rolling value).
    assert res.loc[(2020, 2), "out"] == pytest.approx(10.0)
    assert res.loc[(2021, 2), "out"] == pytest.approx(30.0)
    # The temp prior-season column is dropped.
    assert not any(c.startswith("_prior_") for c in res.columns)


def _opp_matchup_frame() -> pd.DataFrame:
    """One RB facing opponent OPP across 2020 (wk1-3) and a 2021 wk1 opener.
    Per-game points allowed to RB = the RB's fantasy_points; receptions drive
    recv_fantasy so the multi-column fill_map is exercised too."""
    rows = []
    schedule = [(2020, 1, 10.0, 2), (2020, 2, 14.0, 3), (2020, 3, 12.0, 4), (2021, 1, 99.0, 9)]
    for season, week, fp, rec in schedule:
        rows.append(
            dict(
                player_id="RB1",
                position="RB",
                season=season,
                week=week,
                opponent_team="OPP",
                fantasy_points=fp,
                receptions=rec,
                receiving_yards=0.0,
                receiving_tds=0,
                rushing_yards=0.0,
                rushing_tds=0,
            )
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_matchup_opener_uses_prior_season_mean():
    """``_build_matchup_features`` backs a season-opener's opp_* matchup
    features with the opponent's prior-season per-game mean instead of leaving
    them NaN (which the catch-all maps to a biased-low 0)."""
    from src.features.engineer import _build_matchup_features

    out = _build_matchup_features(_opp_matchup_frame()).set_index(["season", "week"])

    # 2021 opener: prior-season (2020) mean pts allowed to RB = (10+14+12)/3 = 12.0
    assert out.loc[(2021, 1), "opp_fantasy_pts_allowed_to_pos"] == pytest.approx(12.0)
    # recv component falls back too: 2020 mean receptions = (2+3+4)/3 = 3.0
    assert out.loc[(2021, 1), "opp_recv_pts_allowed_to_pos"] == pytest.approx(3.0)
    # The opener now ranks (not NaN); single opponent -> rank 1.
    assert out.loc[(2021, 1), "opp_def_rank_vs_pos"] == pytest.approx(1.0)
    # Within-season rows keep their shift(1).rolling value (2020 wk2 sees wk1=10).
    assert out.loc[(2020, 2), "opp_fantasy_pts_allowed_to_pos"] == pytest.approx(10.0)
    # First-season opener has no prior -> NaN survives to the catch-all 0.
    assert pd.isna(out.loc[(2020, 1), "opp_fantasy_pts_allowed_to_pos"])


@pytest.mark.unit
def test_defense_matchup_opener_uses_prior_season_mean(monkeypatch):
    """``_build_defense_matchup_features`` backs a season-opener's opp_def_*_L5
    (offense-derived, site 2) and opp_def_pts_allowed_L5 (schedule-derived,
    site 3) with the defense's prior-season per-game mean."""
    import src.shared.weather_features as wf
    from src.features.engineer import _build_defense_matchup_features

    # Schedule: DEF (always the away team) allows the home score. 2020 mean
    # points allowed = (20+10)/2 = 15.0; a 2021 wk1 opener game must exist so
    # the merged row carries the fallback rather than dropping to the catch-all.
    # spread_line/total_line are unused by these assertions but required by the
    # implied_team_total lookup the same builder runs.
    def _g(season, week, home, hs):
        return dict(
            season=season,
            week=week,
            away_team="DEF",
            home_team=home,
            away_score=0,
            home_score=hs,
            spread_line=0.0,
            total_line=40.0,
        )

    sched = pd.DataFrame([_g(2020, 1, "HA", 20), _g(2020, 2, "HB", 10), _g(2021, 1, "HC", 99)])
    monkeypatch.setattr(wf, "_schedule_cache", sched)
    monkeypatch.setattr(wf, "_load_schedules", lambda: sched)

    # Offense facing DEF: 2020 per-game sacks 2 & 4 (mean 3.0), plus a 2021
    # wk1 opener whose within-season rolling is NaN -> falls back to 3.0.
    plays = [(2020, 1, 2), (2020, 2, 4), (2021, 1, 9)]
    df = pd.DataFrame(
        [
            dict(
                player_id="QB1",
                season=season,
                week=week,
                opponent_team="DEF",
                recent_team="OFF",
                sacks=sk,
                passing_yards=200.0,
                passing_tds=1,
                interceptions=0,
                rushing_yards=10.0,
            )
            for season, week, sk in plays
        ]
    )

    out = _build_defense_matchup_features(df).set_index(["season", "week"])

    # Site 2: opener opp_def_sacks_L5 = 2020 mean sacks = 3.0 (not 0).
    assert out.loc[(2021, 1), "opp_def_sacks_L5"] == pytest.approx(3.0)
    # Site 3: opener opp_def_pts_allowed_L5 = 2020 mean points allowed = 15.0.
    assert out.loc[(2021, 1), "opp_def_pts_allowed_L5"] == pytest.approx(15.0)
    # Within-season 2020 wk2 keeps its rolling value (sees wk1 sacks=2).
    assert out.loc[(2020, 2), "opp_def_sacks_L5"] == pytest.approx(2.0)
