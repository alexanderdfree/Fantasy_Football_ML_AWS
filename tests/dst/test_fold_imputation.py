"""Native DST folds fit missing contextual features on their own training rows."""

import numpy as np
import pandas as pd
import pytest

from tests.dst.test_data_build import _make_schedules, _make_team_stats, _make_weekly

pytestmark = pytest.mark.unit


@pytest.fixture
def native_sources(monkeypatch):
    from src.dst import data, run_pipeline

    def extend(frame):
        six_weeks = pd.concat([frame, frame.assign(week=frame["week"] + 3)])
        return pd.concat(
            [six_weeks.assign(season=season) for season in range(2019, 2026)], ignore_index=True
        )

    weekly = extend(_make_weekly())
    schedules = extend(_make_schedules())
    team_stats = extend(_make_team_stats())
    build_data = data.build_data
    current = {"schedules": schedules}

    def build(**kwargs):
        frame = build_data(
            weekly=weekly, schedules=current["schedules"], team_stats=team_stats, **kwargs
        )
        # Native schedule/box context is already supplied. REG metadata is an
        # independent loader correction, tested in test_native_cv_metadata.
        return frame.assign(_schedule_merged=True, _team_box_score_merged=True, season_type="REG")

    monkeypatch.setattr(data, "build_data", build)
    monkeypatch.setattr(run_pipeline, "build_data", build)
    monkeypatch.setattr(
        data.nfl_source, "teams", lambda: pd.DataFrame({"team_abbr": [], "team_logo_espn": []})
    )
    monkeypatch.setenv("FF_FEATURE_CACHE_DISABLE", "1")
    return current, schedules


def _prepared_origin():
    from src.benchmarking.benchmark import _rolling_origin_inputs
    from src.shared.pipeline import _prepare_position_data

    origins, cfg = _rolling_origin_inputs("DST")
    year, train, val, test = next(origin for origin in origins if origin[0] == 2023)
    assert year == 2023
    return _prepare_position_data("DST", cfg, train, val, test)


def test_rolling_origin_prepared_inputs_ignore_holdout_scores(native_sources):
    current, schedules = native_sources
    baseline = _prepared_origin()
    perturbed = schedules.copy()
    perturbed.loc[perturbed["season"].eq(2023), ["home_score", "away_score"]] += 10
    current["schedules"] = perturbed
    changed = _prepared_origin()
    # Neither train nor validation inputs can see the test year's scores.
    np.testing.assert_array_equal(baseline[0], changed[0])
    np.testing.assert_array_equal(baseline[1], changed[1])
    # The first held-out week has no prior same-season scoring history either.
    first_week = baseline[8]["week"].eq(1).to_numpy()
    np.testing.assert_array_equal(baseline[2][first_week], changed[2][first_week])


def test_rolling_origin_imputation_responds_to_training_scores(native_sources):
    current, schedules = native_sources
    baseline = _prepared_origin()
    perturbed = schedules.copy()
    perturbed.loc[perturbed["season"].eq(2020), ["home_score", "away_score"]] += 10
    current["schedules"] = perturbed
    changed = _prepared_origin()
    col = baseline[9].index("opp_scoring_L3")
    first_week = baseline[7]["week"].eq(1).to_numpy()
    assert np.all(changed[1][first_week, col] > baseline[1][first_week, col])


def test_native_cv_defers_context_until_each_train_partition(native_sources, monkeypatch):
    from src.data.split import expanding_window_folds
    from src.dst import run_pipeline as native
    from src.shared.pipeline import _prepare_position_data

    captured = {}

    def inspect_cv(position, cfg, full, test, seed):
        assert position == "DST" and seed == 7
        assert full.loc[full["week"].eq(1), "opp_scoring_L3"].isna().all()
        captured["cfg"] = cfg
        captured["folds"] = []
        for _, train, val in expanding_window_folds(full):
            prepared = _prepare_position_data(position, cfg, train, val)
            opener = prepared[7]["week"].eq(1)
            expected = prepared[6]["points_allowed"].mean()
            assert prepared[7].loc[opener, "opp_scoring_L3"].eq(expected).all()
            captured["folds"].append(prepared)
        # The shared CV final refit uses the same cfg with the final train split.
        train = full[full["season"].isin(native.TRAIN_SEASONS)]
        val = full[full["season"].isin(native.VAL_SEASONS)]
        captured["final"] = _prepare_position_data(position, cfg, train, val, test)
        return "checked"

    original_fill = native.CONFIG["fill_nans_fn"]
    monkeypatch.setattr(native, "run_cv_pipeline", inspect_cv)
    assert native.run_cv(seed=7) == "checked"
    assert len(captured["folds"]) == 4
    assert native.CONFIG["fill_nans_fn"] is original_fill


def test_lgbm_cv_uses_deferred_native_context(native_sources, monkeypatch):
    from src.dst.run_pipeline import CONFIG
    from src.tuning import tune_lgbm

    real_prepare = tune_lgbm._prepare_position_data
    seen = []

    def inspect_prepare(position, cfg, train, val, *args, **kwargs):
        assert train.loc[train["week"].eq(1), "opp_scoring_L3"].isna().all()
        prepared = real_prepare(position, cfg, train, val, *args, **kwargs)
        seen.append(prepared)
        return prepared

    monkeypatch.setattr(tune_lgbm, "_prepare_position_data", inspect_prepare)
    folds, _ = tune_lgbm._prepare_cv_folds("DST", CONFIG)
    assert len(folds) == len(seen) == 4


def test_default_context_fill_matches_explicit_production_train(native_sources):
    from src.config import TRAIN_SEASONS
    from src.dst.data import build_data, impute_context_from_train

    default = build_data()
    raw = build_data(impute_context=False)
    filled = impute_context_from_train(raw, fit_on=raw[raw["season"].isin(TRAIN_SEASONS)])
    pd.testing.assert_frame_equal(default, filled)


def test_fold_context_never_falls_back_to_holdout_values():
    from src.dst.data import impute_context_from_train

    cols = [
        "spread_line",
        "total_line",
        "points_allowed",
        "opp_scoring_L3",
        "opp_scoring_L5",
        "opp_turnovers_L5",
        "opp_sacks_allowed_L5",
        "opp_qb_int_rate_L5",
        "opp_qb_sack_rate_L5",
        "opp_qb_rush_yds_L5",
    ]
    train = pd.DataFrame({c: [np.nan] for c in cols})
    holdout = pd.DataFrame({c: [99.0, np.nan] for c in cols})
    filled = impute_context_from_train(holdout, fit_on=train)
    assert filled.loc[1].drop("points_allowed").eq(0).all()
    assert filled.loc[0].eq(99).all()
    assert pd.isna(filled.loc[1, "points_allowed"])
