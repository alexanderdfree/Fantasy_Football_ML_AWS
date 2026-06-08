"""Tests for the opp-OFFENSE attention-history builder in
``src/features/engineer.py``.

Mirrors ``tests/test_opp_defense_history.py`` for the inverse semantic:
- DST faces an offense, so the parallel branch attends over the opposing
  *offense*'s per-game form.
- The per-game frame is keyed by ``recent_team`` (the team whose offense
  produced the stats), renamed to ``opponent_team`` on output so the
  downstream history-array builder finds the row for whoever a given
  DST is facing.
- ``off_pts_scored`` is sourced from schedules (the team's own score in
  that game), mirroring ``def_pts_allowed``.
"""

import pandas as pd
import pytest

from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    OPP_OFFENSE_HISTORY_STATS,
    build_opp_defense_history_arrays,
    build_opp_defense_per_game_df,
    build_opp_offense_per_game_df,
)


def _synthetic_offense_df(
    teams=("X", "Y"),
    weeks=(1, 2, 3, 4),
    players=("p1", "p2"),
    season=2023,
):
    """Two offenses, four weeks, two players per offense per week.

    Each player-row contributes fixed offensive stats so the per-game
    aggregate equals ``n_players * stat_per_player``, making expected
    values trivial to verify.
    """
    rows = []
    for team in teams:
        for week in weeks:
            for pid in players:
                rows.append(
                    {
                        "player_id": pid,
                        "position": "QB",
                        "recent_team": team,
                        "opponent_team": "Z" if team == "X" else "W",
                        "season": season,
                        "week": week,
                        "passing_yards": 150.0,
                        "passing_tds": 1.0,
                        "rushing_yards": 20.0,
                        "rushing_tds": 0.5,
                        "interceptions": 0.5,
                        "sack_fumbles_lost": 0.0,
                        "rushing_fumbles_lost": 0.5,
                        "receiving_fumbles_lost": 0.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestBuildOppOffensePerGameDf:
    def test_aggregation_shape_and_columns(self):
        df = _synthetic_offense_df()
        out = build_opp_offense_per_game_df(df)
        # 2 offenses × 4 weeks = 8 rows.
        assert len(out) == 8
        for col in OPP_OFFENSE_HISTORY_STATS:
            assert col in out.columns
        assert set(out.columns) >= {"opponent_team", "season", "week"}
        # The recent_team key was renamed; downstream lookup is by opponent_team.
        assert "recent_team" not in out.columns

    def test_aggregated_stats_match_sum(self):
        df = _synthetic_offense_df()
        out = build_opp_offense_per_game_df(df).sort_values(["opponent_team", "season", "week"])
        # Each offense sees 2 players × per-player stat each week.
        assert (out["off_pass_yards"] == 300.0).all()
        assert (out["off_pass_tds"] == 2.0).all()
        assert (out["off_rush_yards"] == 40.0).all()
        assert (out["off_rush_tds"] == 1.0).all()
        assert (out["off_ints"] == 1.0).all()
        # 2 players × (sack_fumbles_lost 0 + rushing_fumbles_lost 0.5 + receiving 0)
        assert (out["off_fumbles_lost"] == 1.0).all()

    def test_recent_team_renamed_to_opponent_team(self):
        """Downstream history-array builder keys on opponent_team — confirm
        the rename happened so the lookup chain works."""
        df = _synthetic_offense_df()
        out = build_opp_offense_per_game_df(df)
        assert set(out["opponent_team"].unique()) == {"X", "Y"}

    def test_missing_columns_returns_empty_frame(self):
        df = pd.DataFrame(
            {"player_id": ["p1"], "recent_team": ["X"], "season": [2023], "week": [1]}
        )
        out = build_opp_offense_per_game_df(df)
        assert len(out) == 0
        # Schema is still present so downstream callers don't KeyError.
        for col in OPP_OFFENSE_HISTORY_STATS:
            assert col in out.columns

    def test_all_nan_recent_team_returns_empty(self):
        df = _synthetic_offense_df()
        df = df.assign(recent_team=pd.NA)
        out = build_opp_offense_per_game_df(df)
        assert len(out) == 0
        for col in OPP_OFFENSE_HISTORY_STATS:
            assert col in out.columns

    def test_missing_schedules_falls_back_to_zero_pts(self, monkeypatch):
        """If _load_schedules can't find the parquet, off_pts_scored falls back
        to 0 but the player-derived columns survive."""

        def _raise():
            raise FileNotFoundError("schedules parquet absent")

        # The helper now routes through _load_schedules (#757); patch it to raise
        # so the FileNotFoundError zero-fill fallback is exercised. (Patching
        # CACHE_DIR no longer reaches the read — _load_schedules owns it.)
        monkeypatch.setattr("src.features.engineer._load_schedules", _raise)
        df = _synthetic_offense_df()
        out = build_opp_offense_per_game_df(df)
        assert (out["off_pts_scored"] == 0.0).all()
        # Player-derived columns are still populated.
        assert (out["off_pass_yards"] == 300.0).all()

    def test_fumbles_lost_combines_three_sources(self):
        """off_fumbles_lost must equal sack + rushing + receiving fumbles_lost."""
        df = _synthetic_offense_df()
        # Bump receiving fumbles to a non-zero value to spread the sum across
        # all three sources.
        df = df.assign(receiving_fumbles_lost=0.5, sack_fumbles_lost=0.25)
        out = build_opp_offense_per_game_df(df).sort_values(["opponent_team", "season", "week"])
        # 2 players × (0.25 + 0.5 + 0.5) = 2.5 per team-week.
        assert (out["off_fumbles_lost"] == 2.5).all()

    def test_offense_read_must_drop_postseason_rows(self, tmp_path):
        """Audit #424: the raw weekly cache the DST opp-offense branch reads
        carries postseason rows (it is written unfiltered by
        ``src/data/loader.py``), unlike every other REG-only signal. The
        pipeline / serving read now applies the same ``season_type == "REG"``
        filter ``_read_split`` uses before aggregating. This pins that contract:
        a POST game must not inflate the opp-offense per-game aggregate.
        """
        df = _synthetic_offense_df(weeks=(1, 2))
        # Append a postseason row for team X, week 1 (overlapping REG week number,
        # exactly the case that silently double-counts without a season_type
        # filter). REG rows have season_type REG.
        df["season_type"] = "REG"
        post = df[(df["recent_team"] == "X") & (df["week"] == 1)].copy()
        post["season_type"] = "POST"
        post["passing_yards"] = 999.0  # detectable inflation if it leaks through
        weekly = pd.concat([df, post], ignore_index=True)

        cache = tmp_path / "weekly.parquet"
        weekly.to_parquet(cache)

        # Replicate the pipeline/serving read seam (src/shared/pipeline.py,
        # src/serving/core.py): read, then drop non-REG rows before aggregation.
        read = pd.read_parquet(cache)
        if "season_type" in read.columns:
            read = read[read["season_type"] == "REG"].copy()

        out = build_opp_offense_per_game_df(read).sort_values(["opponent_team", "season", "week"])
        # Team X week 1 reflects only the two REG players (2 × 150 = 300), NOT
        # the 999-yard POST row — proving the postseason game was excluded.
        x_w1 = out[(out["opponent_team"] == "X") & (out["week"] == 1)]
        assert (x_w1["off_pass_yards"] == 300.0).all()


@pytest.mark.unit
class TestDispatchTable:
    def test_dispatch_resolves_both_kinds(self):
        """OPP_ATTN_PER_GAME_BUILDERS is the single source of truth shared
        by training (pipeline.py) and inference (serving/app.py). Drift guard."""
        assert OPP_ATTN_PER_GAME_BUILDERS["defense"] is build_opp_defense_per_game_df
        assert OPP_ATTN_PER_GAME_BUILDERS["offense"] is build_opp_offense_per_game_df

    def test_unknown_kind_raises_keyerror(self):
        with pytest.raises(KeyError):
            OPP_ATTN_PER_GAME_BUILDERS["something_else"]


@pytest.mark.unit
class TestEndToEndChain:
    """Drift detector: offense-builder output flows through
    build_opp_defense_history_arrays (the shared sequence builder) and
    produces non-zero values. If either the kind dispatch or the source
    frame picks the wrong path in pipeline/serving, the resulting tensor
    would be all-zero — this test catches that class of bug."""

    def test_chain_yields_nonzero_history(self, monkeypatch):
        # Force the no-schedules fallback so off_pts_scored is zero but all other
        # off_* columns flow through with real values. The helper routes through
        # _load_schedules now (#757), so patch that to raise.
        def _raise():
            raise FileNotFoundError("schedules parquet absent")

        monkeypatch.setattr("src.features.engineer._load_schedules", _raise)
        df = _synthetic_offense_df()
        per_game = build_opp_offense_per_game_df(df)

        # A DST facing team X in week 4 must see 3 prior games (weeks 1-3)
        # of opponent_team=X's offensive output. The mask + tensor are
        # built by the shared history-array helper, same path pipeline.py
        # uses regardless of kind.
        target_df = pd.DataFrame(
            [{"player_id": "dst_x_w4", "opponent_team": "X", "season": 2023, "week": 4}]
        )
        X_opp, mask = build_opp_defense_history_arrays(
            target_df, per_game, OPP_OFFENSE_HISTORY_STATS, max_seq_len=5
        )
        assert mask[0].sum() == 3
        # off_pass_yards is column 0 of OPP_OFFENSE_HISTORY_STATS; non-zero
        # means the offense aggregation flowed all the way through.
        assert X_opp[0, : mask[0].sum(), 0].mean() > 0
