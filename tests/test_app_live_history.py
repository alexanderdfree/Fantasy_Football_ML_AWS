"""Live-season ingestion, career context, and unknown practice imputation."""

import pandas as pd
import pytest

from src.serving import core
from src.serving import upcoming_week as live

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def missing_live_snaps(monkeypatch):
    def unavailable(seasons):
        raise ConnectionError("snap_counts_2026.parquet: 404")

    monkeypatch.setattr(live.nfl_source, "snap_counts", unavailable)
    monkeypatch.setattr(live.live_qbr, "recover_qbr", lambda frame, *args: (frame, {}))
    recover_snap_counts = live.live_snap_counts.recover_snap_counts
    monkeypatch.setattr(
        live.live_snap_counts, "recover_snap_counts", lambda frame, *args: (frame, {})
    )
    return recover_snap_counts


@pytest.fixture
def history_inputs(monkeypatch, tmp_path):
    """Run the real live-history boundary with small, isolated source frames."""

    def prepare(*, season=2026, snap_values=(0.9, None, None, 0.0)):
        current = pd.DataFrame(
            {
                "player_id": ["qb1", "rb1", "wr1", "native_zero", "future", "unplayed"],
                "season": [season] * 6,
                "season_type": ["REG"] * 6,
                "week": [6, 6, 6, 6, 7, 6],
                "position": ["QB", "RB", "WR", "TE", "WR", "RB"],
                "recent_team": ["SEA", "SEA", "NE", "NE", "SEA", "IND"],
                "opponent_team": ["NE", "NE", "SEA", "SEA", "NE", "CHI"],
                "attempts": [30, 0, 0, 0, 0, 0],
                "carries": [1, 10, 0, 0, 0, 1],
                "targets": [0, 2, 5, 0, 1, 0],
                "special_teams_tds": [0, 0, 0, 1, 0, 0],
                "snap_pct": [*snap_values, None, None],
                "qbr_total": [78.6, None, None, None, None, None],
                "pts_added": [3.0, None, None, None, None, None],
            }
        )
        old = current.iloc[[0]].assign(player_id="archived", season=season - 1, week=1)
        schedule = pd.DataFrame(
            {
                "season": [season, season],
                "week": [6, 6],
                "game_type": ["REG", "REG"],
                "game_id": [f"{season}_06_NE_SEA", f"{season}_06_CHI_IND"],
                "gameday": [f"{season}-10-18", f"{season}-10-18"],
                "home_team": ["SEA", "IND"],
                "away_team": ["NE", "CHI"],
                "home_score": [13, None],
                "away_score": [10, None],
            }
        )

        def load(seasons, cache_dir=None):
            if cache_dir:
                signature = f"{seasons[0]}_{seasons[-1]}"
                current[["player_id", "season", "week"]].assign(
                    **{column: 0.0 for column in live.FF_OPP_FEATURE_COLUMNS}
                ).to_parquet(f"{cache_dir}/ff_opportunity_{signature}.parquet")
            return current.copy()

        teams = current.rename(columns={"recent_team": "team"})[
            ["season", "week", "team"]
        ].drop_duplicates()
        monkeypatch.setattr(live, "load_raw_data", load)
        monkeypatch.setattr(live, "load_team_week_stats", lambda *args, **kwargs: teams.copy())
        monkeypatch.setattr(live, "preprocess", lambda frame: frame.copy())
        monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(live, "_history_cache", old)
        monkeypatch.setattr(live, "_history_seasons", tuple(range(live.SEASONS[0], season)))
        return current, schedule

    return prepare


def _snap_issues(history):
    quality = live.upcoming_status.data_quality(
        {"injuries": {"status": "available"}, "history": history}
    )
    return [issue for issue in quality["issues"] if issue["source"] == "snap_counts"]


def _sleeper_snap_payload(player_id="10", team="SEA", opponent="NE", *, snaps=50, total=100):
    return {
        "category": "stat",
        "sport": "nfl",
        "season": "2026",
        "week": 6,
        "season_type": "regular",
        "date": "2026-10-18",
        "team": team,
        "opponent": opponent,
        "game_id": "sleeper-game",
        "player_id": player_id,
        "player": {"position": "RB" if player_id == "10" else "TE"},
        "stats": {"gp": 1, "off_snp": snaps, "tm_off_snp": total},
    }


def _stub_sleeper_response(monkeypatch, payload):
    monkeypatch.setattr(
        live.live_snap_counts.nfl_source,
        "player_ids",
        lambda: pd.DataFrame({"sleeper_id": ["10", "11"], "gsis_id": ["rb1", "native_zero"]}),
    )

    def get_json(url):
        return payload, {
            "url": url,
            "retrieved_at": "2026-10-19T06:00:00+00:00",
            "sha256": "a" * 64,
        }

    monkeypatch.setattr(live.live_snap_counts, "_get_json", get_json)


def test_snap_recovery_precedes_qbr_and_counts_only_filled_completed_games(
    monkeypatch, history_inputs
):
    _, schedule = history_inputs()
    receipt = {"provider": "Sleeper actual statistics", "recovered_rows": 1, "errors": []}
    calls = []

    def recover(frame, schedules, season):
        assert set(frame.player_id) == {"qb1", "rb1", "wr1", "native_zero"}
        assert season == 2026
        pd.testing.assert_frame_equal(schedules, schedule)
        calls.append("snaps")
        result = frame.copy()
        result.loc[result.player_id.eq("rb1"), "snap_pct"] = 0.5
        return result, receipt

    def qbr(frame, *args):
        assert frame.set_index("player_id").loc["rb1", "snap_pct"] == 0.5
        calls.append("qbr")
        return frame, {}

    monkeypatch.setattr(live.live_snap_counts, "recover_snap_counts", recover)
    monkeypatch.setattr(live.live_qbr, "recover_qbr", qbr)

    result = live._load_history(2026, 6, schedule)

    assert calls == ["snaps", "qbr"]
    assert set(result.player_id) == {"archived", "qb1", "rb1", "wr1", "native_zero"}
    history = result.attrs["live_history_sources"]
    assert history["snap_counts_recovery"] == receipt
    assert history["player_rows"] == 4
    assert history["coverage"]["snap_counts"] == live.upcoming_status.coverage(3, 4)
    assert "3 of 4 completed player-games" in _snap_issues(history)[0]["message"]


def test_actual_snap_fallback_preserves_native_zero_and_residual_gap_warning(
    monkeypatch, history_inputs, missing_live_snaps
):
    _, schedule = history_inputs()
    monkeypatch.setattr(live.live_snap_counts, "recover_snap_counts", missing_live_snaps)
    _stub_sleeper_response(
        monkeypatch,
        [_sleeper_snap_payload(), _sleeper_snap_payload("11", "NE", "SEA", snaps=80)],
    )

    result = live._load_history(2026, 6, schedule)

    values = result.set_index("player_id")["snap_pct"]
    assert values["rb1"] == 0.5
    assert values["native_zero"] == 0.0
    assert values["qb1"] == 0.9
    assert pd.isna(values["wr1"])
    history = result.attrs["live_history_sources"]
    assert history["snap_counts_recovery"]["recovered_rows"] == 1
    assert history["coverage"]["snap_counts"] == live.upcoming_status.coverage(3, 4)
    assert _snap_issues(history)


@pytest.mark.parametrize("provider_error", [False, True], ids=["empty_source", "source_outage"])
def test_unavailable_optional_snap_provider_retains_missing_history_and_warning(
    monkeypatch, history_inputs, missing_live_snaps, provider_error
):
    _, schedule = history_inputs(snap_values=(None, None, None, None))
    monkeypatch.setattr(live.live_snap_counts, "recover_snap_counts", missing_live_snaps)
    _stub_sleeper_response(monkeypatch, [])
    if provider_error:

        def unavailable(url):
            raise ConnectionError("Sleeper temporarily unavailable")

        monkeypatch.setattr(live.live_snap_counts, "_get_json", unavailable)

    result = live._load_history(2026, 6, schedule)

    current = result[result.season.eq(2026)]
    assert len(current) == 4
    assert current.snap_pct.isna().all()
    history = result.attrs["live_history_sources"]
    assert history["coverage"]["snap_counts"] == live.upcoming_status.coverage(0, 4)
    assert history["snap_counts_recovery"]["recovered_rows"] == 0
    assert "0 of 4 completed player-games" in _snap_issues(history)[0]["message"]
    if provider_error:
        assert history["snap_counts_recovery"]["errors"]


def test_historical_history_replay_never_queries_live_snap_fallback(monkeypatch, history_inputs):
    _, schedule = history_inputs(season=2025)
    monkeypatch.setattr(
        live.live_snap_counts,
        "recover_snap_counts",
        lambda *args: pytest.fail("Historical replay must not recover today's snap observations"),
    )
    monkeypatch.setattr(
        live.nfl_source,
        "snap_counts",
        lambda *args: pytest.fail("Historical replay must retain its sealed native snap source"),
    )
    monkeypatch.setattr(
        live.live_qbr,
        "recover_qbr",
        lambda *args: pytest.fail("Historical replay must not recover today's QBR observations"),
    )

    result = live._load_history(2025, 6, schedule)

    history = result.attrs["live_history_sources"]
    assert history["snap_counts_recovery"] == {}
    assert history["coverage"]["snap_counts"] == live.upcoming_status.coverage(2, 4)
    assert _snap_issues(history)


def test_supplementary_coverage_counts_completed_observations_not_nonempty_files():
    current = pd.DataFrame(
        {
            "player_id": ["qb", "rb", "wr", "idle"],
            "season": [2026] * 4,
            "week": [1] * 4,
            "position": ["QB", "RB", "WR", "TE"],
            "attempts": [30, 0, 0, 0],
            "carries": [1, 10, 0, 0],
            "targets": [0, 2, 5, 0],
            "snap_pct": [0.9, None, 0.6, None],
            "qbr_total": [78.6, None, None, None],
            "pts_added": [3.0, None, None, None],
        }
    )
    opportunity = current.iloc[:2][["player_id", "season", "week"]].assign(
        **{col: 0.0 for col in live.FF_OPP_FEATURE_COLUMNS}
    )
    result = live._history_source_coverage(current, opportunity)
    assert result == {
        "snap_counts": {"status": "partial", "observed": 2, "expected": 4},
        "ff_opportunity": {"status": "partial", "observed": 2, "expected": 3},
        "qbr": {"status": "available", "observed": 1, "expected": 1},
    }
    # Complete-looking keys with absent source statistics are still unavailable.
    missing_values = opportunity.assign(**{live.FF_OPP_FEATURE_COLUMNS[0]: float("nan")})
    assert live._history_source_coverage(current, missing_values)["ff_opportunity"] == {
        "status": "unavailable",
        "observed": 0,
        "expected": 3,
    }


@pytest.mark.parametrize("opportunity_week", [-1, None, 1, 2])
def test_current_season_is_fresh_and_team_tokens_advance(monkeypatch, tmp_path, opportunity_week):
    old = pd.DataFrame(
        {"player_id": ["old"], "season": [2025], "week": [1], "recent_team": ["SEA"]}
    )
    current = pd.DataFrame(
        {
            "player_id": ["A", "B", "future"],
            "season": [2026] * 3,
            "week": [1, 1, 2],
            "recent_team": ["SEA", "NE", "SEA"],
        }
    )
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [13],
            "away_score": [10],
        }
    )
    calls = []

    def load(seasons, cache_dir=None):
        calls.append((seasons, cache_dir))
        if cache_dir and opportunity_week != -1:
            opportunity = current.loc[
                current.week.eq(opportunity_week), ["player_id", "season", "week"]
            ].assign(**{column: 0.0 for column in live.FF_OPP_FEATURE_COLUMNS})
            opportunity.to_parquet(f"{cache_dir}/ff_opportunity_2026_2026.parquet")
        return old.copy() if len(seasons) > 1 else current.copy()

    def teams(seasons, cache_dir=None):
        frame = old if len(seasons) > 1 else current
        return frame.rename(columns={"recent_team": "team"}).drop(columns="player_id")

    monkeypatch.setattr(live, "load_raw_data", load)
    monkeypatch.setattr(live, "load_team_week_stats", teams)
    monkeypatch.setattr(live, "preprocess", lambda df: df)
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "_history_cache", None)
    monkeypatch.setattr(live, "_history_seasons", None)
    one = live._load_history(2026, 1, schedule)
    two = live._load_history(2026, 1, schedule)
    assert set(one.player_id) == {"old", "A", "B"}
    pd.testing.assert_frame_equal(one, two)
    assert calls[0][0][-1] == 2025
    assert calls[1][0] == calls[2][0] == [2026]
    assert calls[1][1] != calls[2][1]  # no stale current-season file reuse
    assert one.attrs["live_history_sources"]["snap_counts"] == "unavailable"
    assert one.attrs["live_history_sources"]["player_rows"] == 2
    assert one.attrs["live_history_sources"]["ff_opportunity"] == (
        "available" if opportunity_week == 1 else "unavailable"
    )
    assert one.attrs["live_history_sources"]["ff_opportunity_rows"] == (
        2 if opportunity_week == 1 else 0
    )
    assert one.attrs["live_history_sources"]["qbr_observed_rows"] == 0
    team_cache = pd.read_parquet(tmp_path / "team_stats_2012_2025.parquet")
    assert set(team_cache.loc[team_cache.season.eq(2026), "team"]) == {"SEA", "NE"}
    assert team_cache.week.max() == 1


def test_current_season_missing_completed_team_fails(monkeypatch, tmp_path):
    old = pd.DataFrame(
        {"player_id": ["old"], "season": [2025], "week": [1], "recent_team": ["SEA"]}
    )
    monkeypatch.setattr(live, "_history_cache", old)
    monkeypatch.setattr(live, "_history_seasons", tuple(range(live.SEASONS[0], 2026)))
    monkeypatch.setattr(live, "preprocess", lambda df: df)
    monkeypatch.setattr(live, "load_raw_data", lambda *a, **k: old.iloc[:0])
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [13],
            "away_score": [10],
        }
    )
    with pytest.raises(RuntimeError, match="not available yet"):
        live._load_history(2026, 1, schedule)


def test_rollover_publishes_intervening_history_to_fixed_consumers(monkeypatch, tmp_path):
    history = pd.DataFrame(
        {"player_id": ["A"], "season": [2026], "week": [1], "recent_team": ["SEA"]}
    )
    archived_schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [27],
            "away_score": [20],
            "temp": [51.0],
            "wind": [18.0],
        }
    )
    archived_schedule.iloc[:0].to_parquet(tmp_path / "schedules_2012_2025.parquet")
    teams = pd.DataFrame({"season": [2026], "week": [1], "team": ["SEA"], "passing_yards": [310]})
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live.weather_features, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(live.weather_features, "_schedule_cache", pd.DataFrame())
    monkeypatch.setattr(live, "_history_cache", None)
    monkeypatch.setattr(live, "_history_seasons", None)

    def load(seasons, cache_dir=None):
        if seasons == live.SEASONS:
            return history.iloc[:0]
        assert seasons == [2026]
        archived_schedule.to_parquet(f"{cache_dir}/schedules_2026_2026.parquet")
        return history

    monkeypatch.setattr(live, "load_raw_data", load)
    monkeypatch.setattr(live, "load_team_week_stats", lambda *a, **kw: teams)
    monkeypatch.setattr(live, "preprocess", lambda frame: frame)
    opener = archived_schedule.assign(season=2027, home_score=float("nan"), away_score=float("nan"))
    pd.testing.assert_frame_equal(live._load_history(2027, 1, opener), history)
    # Read through the same fixed-range paths used by schedule features and
    # attention team tokens, even though there are no live-year games yet.
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "schedules_2012_2025.parquet"), archived_schedule
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "team_stats_2012_2025.parquet").reset_index(drop=True), teams
    )


def test_inference_keeps_heldout_year_without_repeating_fit_years(monkeypatch):
    frame = pd.DataFrame(
        {
            "player_id": ["A"] * 3,
            "position": ["QB"] * 3,
            "recent_team": ["TEN"] * 3,
            "season": [2024, 2025, 2026],
            "week": [1, 1, 1],
            "_is_upcoming": [False, False, True],
        }
    )
    train = pd.DataFrame({"season": [2023]})
    val = pd.DataFrame({"season": [2024]})
    monkeypatch.setattr(live.core, "_ensure_base_data", lambda: None)
    monkeypatch.setattr(
        live.app_pkg.current_state(), "cache", {"splits": {"QB": (train, val, frame)}}
    )
    seen = []
    monkeypatch.setattr(
        live.core, "_apply_position_models", lambda tr, va, te, p, r: seen.extend(te.season)
    )
    roster = pd.DataFrame({"player_id": ["A"], "espn_name": ["A"], "espn_id": ["1"]})
    slate = pd.DataFrame({"recent_team": ["TEN"], "spread_line": [1.0], "total_line": [40.0]})
    out = live.run_upcoming_inference(frame, roster, slate, 2026, 1)
    assert seen == [2025, 2026]
    assert out.season.tolist() == [2026]


def test_unknown_practice_uses_filtered_training_mean(monkeypatch):
    from src.prediction import frames

    reg = {
        "targets": [],
        "model_dir": "unused",
        "filter_fn": lambda df: df.copy(),
        "compute_targets_fn": lambda df: df,
        "min_games_per_season": 2,
        "get_feature_columns_fn": lambda: ["practice_status"],
    }
    train = pd.DataFrame(
        {
            "player_id": ["A", "A", "B"],
            "season": [2023] * 3,
            "week": [1, 2, 1],
            "practice_status": [2.0, 1.0, 0.0],
        }
    )
    test = pd.DataFrame(
        {
            "player_id": ["X", "Y"],
            "season": [2026, 2026],
            "week": [1, 1],
            "practice_status": [float("nan"), 1.0],
            "_practice_status_missing": [True, False],
        }
    )

    class StopAfterPreparation(Exception):
        pass

    def capture(tr, va, te, *args, **kwargs):
        assert te.practice_status.tolist() == [1.5, 1.0]
        raise StopAfterPreparation

    monkeypatch.setattr(frames, "build_position_features", capture)
    with pytest.raises(StopAfterPreparation):
        frames.prepare_position_frame("QB", train, train.iloc[:0], test, reg)
