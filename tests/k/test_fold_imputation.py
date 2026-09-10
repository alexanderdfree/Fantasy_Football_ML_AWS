"""Native K folds fit missing Vegas context on their own training cohort."""

import numpy as np
import pandas as pd
import pytest

from tests.k.conftest import _build_tiny_kicks
from tests.k.test_data_loaders import _kicker_pbp_cache_row

pytestmark = pytest.mark.unit
CONTEXT = ["total_line", "implied_team_total"]


@pytest.fixture
def native_sources(tmp_path, monkeypatch):
    from src import config
    from src.k import data, run_pipeline

    seasons = list(range(2019, 2025))
    weekly = pd.DataFrame(
        [
            _kicker_pbp_cache_row(player, season, week, team)
            for season in seasons
            for week in range(1, 7)
            for player, team in (("K1", "KC"), ("K2", "BUF"))
        ]
    )
    weekly.to_parquet(tmp_path / "kicker_pbp_2019_2024.parquet")
    pd.DataFrame(columns=["season", "week", "game_type", "position", "st_snaps"]).to_parquet(
        tmp_path / f"snap_counts_{config.SEASONS[0]}_{config.SEASONS[-1]}.parquet"
    )
    schedules = pd.DataFrame(
        [
            dict(
                season=season,
                week=week,
                game_type="REG",
                home_team="KC",
                away_team="BUF",
                total_line=40.0 + 0.5 * (season - 2019) + week,
                spread_line=3.0,
                roof="outdoors",
                surface="grass",
            )
            for season in seasons
            for week in range(1, 7)
        ]
    )
    schedules.loc[schedules.season.eq(2019) & schedules.week.eq(1), "total_line"] = np.nan
    current = {"schedules": schedules}
    real_load = data.load_data

    def load(**kwargs):
        frame = real_load(seasons=seasons, schedules=current["schedules"], **kwargs)
        # K consumes none of the auxiliary team-box columns. Its own real
        # schedule merge above must run, including its missing-value handling.
        return frame.assign(_team_box_score_merged=True)

    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data, "load_data", load)
    monkeypatch.setattr(run_pipeline, "load_data", load)
    monkeypatch.setattr(
        data,
        "reconstruct_kicker_kicks_from_pbp",
        lambda seasons: _build_tiny_kicks(weekly).drop(columns="is_home"),
    )
    monkeypatch.setattr(data.nfl_source, "pbp_data", lambda *a: pytest.fail("unexpected PBP fetch"))
    monkeypatch.setenv("FF_FEATURE_CACHE_DISABLE", "1")
    return current, schedules


def _prepared_origin():
    from src.benchmarking.benchmark import _rolling_origin_inputs
    from src.shared.pipeline import _prepare_position_data

    origins, cfg = _rolling_origin_inputs("K")
    _, train, val, test = next(row for row in origins if row[0] == 2023)
    return _prepare_position_data("K", cfg, train, val, test)


def test_actual_origin_training_matrix_ignores_heldout_vegas_lines(native_sources):
    current, schedules = native_sources
    baseline = _prepared_origin()
    changed = schedules.copy()
    changed.loc[changed.season.eq(2023), "total_line"] += 200
    current["schedules"] = changed
    perturbed = _prepared_origin()
    np.testing.assert_array_equal(baseline[0], perturbed[0])
    np.testing.assert_array_equal(baseline[1], perturbed[1])
    for name in baseline[3]:
        np.testing.assert_array_equal(baseline[3][name], perturbed[3][name])


def test_origin_imputed_training_rows_respond_to_training_values(native_sources):
    current, schedules = native_sources
    baseline = _prepared_origin()
    changed = schedules.copy()
    changed.loc[changed.season.le(2021), "total_line"] += 10
    current["schedules"] = changed
    perturbed = _prepared_origin()
    missing = baseline[6].season.eq(2019) & baseline[6].week.eq(1)
    columns = [baseline[9].index(column) for column in CONTEXT]
    np.testing.assert_allclose(
        perturbed[0][missing][:, columns] - baseline[0][missing][:, columns],
        np.tile([10.0, 5.0], (int(missing.sum()), 1)),
        rtol=0,
        atol=0,
    )


def test_fully_observed_origin_is_unchanged_by_heldout_values(native_sources):
    current, schedules = native_sources
    healthy = schedules.fillna({"total_line": 43.0})
    current["schedules"] = healthy
    baseline = _prepared_origin()
    changed = healthy.copy()
    changed.loc[changed.season.eq(2023), "total_line"] += 200
    current["schedules"] = changed
    np.testing.assert_array_equal(baseline[0], _prepared_origin()[0])


def test_public_cv_preserves_missing_context_until_each_real_fold(native_sources, monkeypatch):
    from src.data.split import expanding_window_folds
    from src.k import run_pipeline as native
    from src.shared.pipeline import _prepare_position_data

    inspected = []

    def inspect_cv(position, cfg, full, test, seed):
        assert position == "K" and seed == 7
        missing = full.season.eq(2019) & full.week.eq(1)
        assert full.loc[missing, CONTEXT].isna().all().all()
        for _, train, val in expanding_window_folds(full, min_train_season=2015):
            prepared = _prepare_position_data(position, cfg, train, val)
            mask = prepared[6].season.eq(2019) & prepared[6].week.eq(1)
            for column in CONTEXT:
                assert prepared[6].loc[mask, column].eq(train[column].median()).all()
            inspected.append(prepared)
        # Shared CV uses this same config at its final refit boundary.
        train = full[full.season.isin(native.TRAIN_SEASONS)]
        val = full[full.season.isin(native.VAL_SEASONS)]
        final = _prepare_position_data(position, cfg, train, val, test)
        mask = final[6].season.eq(2019) & final[6].week.eq(1)
        for column in CONTEXT:
            assert final[6].loc[mask, column].eq(train[column].median()).all()
        return "checked"

    original_fill = native.CONFIG["fill_nans_fn"]
    monkeypatch.setattr(native, "run_cv_pipeline", inspect_cv)
    assert native.run_cv(seed=7) == "checked"
    assert len(inspected) == 4
    assert native.CONFIG["fill_nans_fn"] is original_fill


def test_lgbm_cv_prepares_deferred_k_context(native_sources, monkeypatch):
    from src.k.run_pipeline import CONFIG
    from src.tuning import tune_lgbm

    original = tune_lgbm._prepare_position_data
    seen = []

    def inspect(position, cfg, train, val, *args, **kwargs):
        missing = train.season.eq(2019) & train.week.eq(1)
        assert train.loc[missing, CONTEXT].isna().all().all()
        prepared = original(position, cfg, train, val, *args, **kwargs)
        seen.append(prepared)
        return prepared

    monkeypatch.setattr(tune_lgbm, "_prepare_position_data", inspect)
    folds, _ = tune_lgbm._prepare_cv_folds("K", CONFIG)
    assert len(folds) == len(seen) == 4


def test_default_loader_preserves_its_existing_fill_and_optional_deferral(native_sources):
    from src.k import data

    default = data.load_data()
    raw = data.load_data(impute_context=False)
    other = [column for column in default if column not in CONTEXT]
    pd.testing.assert_frame_equal(default[other], raw[other])
    missing = raw.season.eq(2019) & raw.week.eq(1)
    assert raw.loc[missing, CONTEXT].isna().all().all()
    for column in CONTEXT:
        expected = raw.loc[raw.season.le(2023), column].median()
        assert default.loc[missing, column].eq(expected).all()


def test_fold_with_no_observed_context_never_fits_from_validation():
    from src.k.data import impute_context_from_train

    train = pd.DataFrame({column: [np.nan, np.nan] for column in CONTEXT})
    val = pd.DataFrame({column: [np.nan, 100.0] for column in CONTEXT})
    filled = impute_context_from_train(val, fit_on=train)
    assert filled.iloc[0].eq(0.0).all()
    assert filled.iloc[1].eq(100.0).all()
    assert train.isna().all().all() and val.iloc[0].isna().all()
