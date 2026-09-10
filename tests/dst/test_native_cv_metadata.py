"""The native DST loader must satisfy the real cross-validation fold contract."""

import pandas as pd
import pytest

from tests.dst.test_data_build import _make_schedules, _make_team_stats, _make_weekly

pytestmark = pytest.mark.unit


@pytest.fixture
def native_inputs(tmp_path, monkeypatch):
    from src import config
    from src.dst import data

    seasons = list(range(2019, 2026))

    def extend(frame):
        return pd.concat([frame.assign(season=year) for year in seasons], ignore_index=True)

    weekly = extend(_make_weekly())
    schedules = extend(_make_schedules())
    postseason = schedules.assign(week=19, game_type="POST", home_score=99, away_score=98)
    schedules = pd.concat([schedules, postseason], ignore_index=True)
    stats = extend(_make_team_stats())
    weekly.to_parquet(tmp_path / "weekly_2019_2025.parquet")
    schedules.to_parquet(tmp_path / "schedules_2019_2025.parquet")
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "SEASONS", seasons)
    monkeypatch.setattr(data, "load_team_week_stats", lambda seasons: stats.copy())
    monkeypatch.setattr(
        data.nfl_source,
        "teams",
        lambda: pd.DataFrame({"team_abbr": [], "team_logo_espn": []}),
    )


def test_native_loader_labels_only_regular_season_rows(native_inputs):
    from src.dst.data import build_data

    frame = build_data()
    assert frame["season_type"].eq("REG").all()
    assert frame["week"].le(3).all()
    assert len(frame) == 7 * 3 * 4


def test_public_native_cv_builds_real_folds_before_training(native_inputs, monkeypatch):
    from src.config import CV_VAL_SEASONS
    from src.dst.run_pipeline import run_cv
    from src.shared import pipeline

    real_folds = pipeline.expanding_window_folds
    captured = []

    def capture_folds(frame):
        result = real_folds(frame)
        captured.extend(result)
        return result

    class ReachedTrainingBoundary(Exception):
        pass

    def stop_before_training(*args, **kwargs):
        raise ReachedTrainingBoundary

    monkeypatch.setattr(pipeline, "expanding_window_folds", capture_folds)
    monkeypatch.setattr(pipeline, "_prepare_train_val", stop_before_training)
    with pytest.raises(ReachedTrainingBoundary):
        run_cv(seed=11)

    assert len(captured) == len(CV_VAL_SEASONS)
    for (_, train, val), season in zip(captured, CV_VAL_SEASONS, strict=True):
        assert not train.empty and not val.empty
        assert train["season"].lt(season).all()
        assert val["season"].eq(season).all()
        assert train["season_type"].eq("REG").all()
        assert val["season_type"].eq("REG").all()


def test_native_metadata_preserves_numerical_targets_and_features(native_inputs):
    from src.dst.data import build_data
    from src.dst.features import compute_features
    from src.dst.targets import compute_targets

    frame = build_data()
    legacy = compute_targets(frame.drop(columns="season_type", errors="ignore"))
    labeled = compute_targets(frame)
    compute_features(legacy)
    compute_features(labeled)
    pd.testing.assert_frame_equal(labeled.drop(columns="season_type", errors="ignore"), legacy)
