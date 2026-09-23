"""No-fit contracts for independent, pinned-release scoring reconstruction."""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.analysis import audit_development_truth as truth
from src.shared.comparison_scoring import ACTUAL_BASIS, comparison_actuals
from src.shared.comparison_truth import SOURCE_AVAILABLE, SOURCE_BASIS
from src.training.context import current_context

pytestmark = pytest.mark.unit


def _write(root, relative, frame):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _skill_rows(position):
    frame = pd.DataFrame(
        {
            "player_id": ["prior", "observed", "missing", "playoff"],
            "position": position,
            "season": [2021, 2022, 2022, 2022],
            "week": [1, 1, 1, 19],
            "season_type": ["REG", "REG", "REG", "POST"],
            "receiving_yards": [10.0, 20.0, 0.0, 2000.0],
            "receptions": 1.0,
            "pos_WR": int(position == "WR"),
            SOURCE_AVAILABLE: [True, True, False, True],
            SOURCE_BASIS: f"{position}:{ACTUAL_BASIS}",
        }
    )
    for column in (
        "passing_yards",
        "rushing_yards",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "interceptions",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
    ):
        frame[column] = 0.0
    frame["fantasy_points"] = frame.receiving_yards * 0.1 + frame.receptions
    return frame


def _write_splits(root, frame):
    for path, rows in zip(
        truth.required_files("WR"), (frame.iloc[:1], frame.iloc[1:2], frame.iloc[2:]), strict=True
    ):
        _write(root, path, rows)


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_skill_truth_uses_targets_and_preserves_certified_missingness(tmp_path, position):
    original = _skill_rows(position)
    _write_splits(tmp_path, original)
    actual = truth.build_truth(position.lower(), tmp_path)
    targets = importlib.import_module(f"src.{position.lower()}.targets")
    expected = (
        targets.compute_targets(original.iloc[:3]).sort_values(truth.KEYS).reset_index(drop=True)
    )
    pd.testing.assert_series_equal(
        comparison_actuals(actual, position), comparison_actuals(expected, position)
    )
    assert actual.player_id.tolist() == ["missing", "observed", "prior"]
    assert "pos_WR" not in actual
    assert actual.loc[actual.player_id.eq("missing"), "actual_projected_total"].isna().all()
    assert current_context() is None


@pytest.mark.parametrize("problem", ["mask", "basis", "duplicate"])
def test_skill_truth_fails_closed_without_observation_provenance(tmp_path, problem):
    rows = _skill_rows("WR")
    if problem == "mask":
        rows = rows.drop(columns=[SOURCE_AVAILABLE])
    elif problem == "basis":
        rows[SOURCE_BASIS] = "stale"
    else:
        rows.loc[2, "player_id"] = "observed"
    _write_splits(tmp_path, rows)
    with pytest.raises(ValueError, match="certification|Duplicate"):
        truth.build_truth("WR", tmp_path)


def test_truth_validates_only_explicit_assessment_and_prior_seasons(tmp_path):
    rows = _skill_rows("TE")
    old = rows.iloc[[0, 0]].copy()
    old["season"] = 2017
    _write_splits(tmp_path, pd.concat([old, rows], ignore_index=True))
    with pytest.raises(ValueError, match="Duplicate"):
        truth.build_truth("TE", tmp_path)
    scoped = truth.build_truth("TE", tmp_path, seasons=[2021, 2022])
    assert set(scoped.season) == {2021, 2022}
    assert len(scoped) == 3


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_missing_dependency_rejected_before_load(tmp_path, position, monkeypatch):
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("read before preflight"))
    with pytest.raises(FileNotFoundError, match="Pinned truth dependency"):
        truth.build_truth(position, tmp_path)


def _native_inputs(root, position):
    for relative in truth.required_files(position):
        _write(root, relative, pd.DataFrame({"sentinel": [relative]}))


def _kicker_raw():
    return pd.DataFrame(
        {
            "player_id": ["k1", "k2"],
            "position": "K",
            "season": 2022,
            "week": 1,
            "season_type": "REG",
            "fg_yards_made": [40.0, np.nan],
            "pat_made": 2.0,
            "fg_missed": 1.0,
            "pat_missed": 0.0,
        }
    )


def test_kicker_native_contract_and_pre_target_fill_mask(tmp_path, monkeypatch):
    from src.k import data

    _native_inputs(tmp_path, "K")
    _write(
        tmp_path,
        truth._KICKER,
        pd.DataFrame({column: [0] for column in data._REQUIRED_PBP_COLUMNS}),
    )
    backfill = pd.DataFrame({"pinned": [2025]})
    monkeypatch.setattr(
        data,
        "_load_backfill_pbp",
        lambda season: backfill if season == 2025 else pytest.fail("wrong year"),
    )

    def load(**kwargs):
        assert kwargs["impute_context"] is False
        assert kwargs["pbp"] is backfill
        assert kwargs["weekly"].sentinel.iloc[0] == truth._WEEKLY
        assert kwargs["schedules"].sentinel.iloc[0] == truth._SCHEDULES
        assert current_context().raw_root == tmp_path / "raw"
        assert current_context().artifact_sink is None
        return _kicker_raw()

    monkeypatch.setattr(data, "load_data", load)
    result = truth.build_truth("K", tmp_path)
    assert result.fg_yard_points.tolist() == [4.0, 0.0]
    assert result.actual_projected_total.iloc[0] == 5.0
    assert pd.isna(result.actual_projected_total.iloc[1])
    assert current_context() is None


def test_kicker_stale_cache_cannot_rebuild(tmp_path, monkeypatch):
    from src.k import data

    _native_inputs(tmp_path, "K")
    monkeypatch.setattr(data, "load_data", lambda **kw: pytest.fail("stale loader invoked"))
    with pytest.raises(ValueError, match="stale schema"):
        truth.build_truth("K", tmp_path)


def test_dst_injected_native_contract_and_shared_components(tmp_path, monkeypatch):
    from src.dst import data
    from src.shared.aggregate_targets import DST_TARGETS

    _native_inputs(tmp_path, "DST")
    scoring = pd.DataFrame(
        {
            "team": "LA",
            "season": range(2012, 2026),
            "week": 1,
            "def_tds": 0,
            "special_teams_tds": 0,
            "def_punt_blocks": 0,
        }
    )
    _write(tmp_path, truth._DST_SCORING, scoring)

    def build(**kwargs):
        assert kwargs["allow_scoring_fetch"] is False
        assert kwargs["impute_context"] is False
        assert kwargs["weekly"].sentinel.iloc[0] == truth._WEEKLY
        assert kwargs["team_stats"].sentinel.iloc[0] == truth._TEAM_STATS
        pd.testing.assert_frame_equal(kwargs["scoring_events"], scoring)
        frame = pd.DataFrame({target: [0.0, 0.0] for target in DST_TARGETS})
        frame["player_id"] = ["LA", "LV"]
        frame["season"] = 2022
        frame["week"] = 1
        frame["season_type"] = "REG"
        frame["points_allowed"] = [0.0, 50.0]
        frame["yards_allowed"] = 300.0
        return frame

    monkeypatch.setattr(data, "build_data", build)
    result = truth.build_truth("DST", tmp_path)
    assert result.actual_projected_total.nunique() == 1
    assert result.fantasy_points.nunique() == 2
    assert result.player_id.tolist() == ["LA", "LV"]


def test_source_guard_blocks_swallowed_fetch_and_restores_functions():
    from src.data import nfl_source
    from src.data.release import DataReleaseError

    original = nfl_source.pbp_data
    with pytest.raises(DataReleaseError, match="attempted"):
        with truth._local_sources_only():
            with pytest.raises(DataReleaseError, match="cannot fetch"):
                nfl_source.pbp_data([2025], ())
            assert nfl_source.teams().empty
    assert nfl_source.pbp_data is original


def test_dependency_identities_and_unsupported_position():
    assert "raw/kicker_backfill_pbp_v1_2025.parquet" in truth.required_files("K")
    assert "raw/dst_scoring_pbp_v1_2012_2025.parquet" in truth.required_files("DST")
    assert all("kicks_pbp" not in path for path in truth.required_files("K"))
    with pytest.raises(ValueError, match="Unsupported"):
        truth.required_files("P")
