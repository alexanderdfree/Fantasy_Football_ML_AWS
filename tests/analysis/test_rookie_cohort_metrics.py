"""Unit tests for src/analysis/rookie_cohort_metrics.py.

The CLI ``main()`` runs the full pipeline (slow, data-dependent) and pulls the
splits, so it is out of scope here. These cover the pure helpers against tiny
synthetic frames so a future change to the cohort/metric logic is caught by CI,
and assert that importing the module does not fire the pipeline (``run`` is
imported lazily inside ``main()``) — the drift class CLAUDE.md flags for
operator-only CLIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import rookie_cohort_metrics as a

pytestmark = pytest.mark.unit


def _min_season() -> pd.Series:
    """R debuts in the test season (rookie); V is a 2020 veteran."""
    train = pd.DataFrame({"player_id": ["V", "V"], "season": [2020, 2021]})
    test = pd.DataFrame({"player_id": ["R", "V"], "season": [2025, 2025]})
    return a.player_min_season([train, None, test])


def _test_df() -> pd.DataFrame:
    """Pipeline-style test_df: rookie R (4 games, week 4 skipped) + veteran V.

    LightGBM is the best model overall and rookies are its harder cohort, so the
    metric helpers have a clear, hand-checkable signal to assert against.
    """
    return pd.DataFrame(
        {
            "player_id": ["R", "R", "R", "R", "V", "V"],
            "season": [2025, 2025, 2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 5, 1, 2],
            "fantasy_points": [2.0, 3.0, 4.0, 10.0, 15.0, 16.0],
            "pred_ridge_total": [4.0, 5.0, 6.0, 9.0, 14.0, 16.0],
            "pred_lgbm_total": [3.0, 4.0, 5.0, 10.0, 15.0, 15.0],
            "pred_nn_total": [5.0, 6.0, 7.0, 8.0, 13.0, 14.0],
            "pred_attn_nn_total": [6.0, 7.0, 8.0, 12.0, 16.0, 17.0],
        }
    )


def test_import_is_cheap_and_defers_pipeline():
    # run() is imported lazily inside main()/_load_run, so a bare import cannot
    # pull torch / the pipeline.
    assert not hasattr(a, "run")
    for fn in (
        "player_min_season",
        "label_rookie_rows",
        "available_models",
        "cohort_model_table",
        "bias_corrected_mae",
        "best_model",
        "rookie_gap",
        "main",
    ):
        assert callable(getattr(a, fn))
    assert a.ACTUAL == "fantasy_points"
    assert a.MODELS["LightGBM"] == "pred_lgbm_total"
    assert a.EARLY_GAMES == 3
    assert a.DEFAULT_POSITIONS == ["QB", "RB", "WR", "TE"]
    assert {a.ROOKIE_EARLY, a.ROOKIE_REST, a.VETERAN, a.UNKNOWN} == {
        "rookie_early",
        "rookie_rest",
        "veteran",
        "unknown",
    }


def test_player_min_season_takes_earliest_and_skips_bad_frames():
    ms = _min_season()
    assert ms["V"] == 2020
    assert ms["R"] == 2025
    # No usable frame -> empty Series, no crash.
    assert a.player_min_season([None, pd.DataFrame({"x": [1]})]).empty


def test_label_rookie_rows_splits_early_by_games_played_not_week_number():
    labels = a.label_rookie_rows(_test_df(), _min_season())
    # R's first 3 games played (weeks 1,2,3) are early; the 4th (week 5, after a
    # skipped week 4) is rest — proving it counts games, not week numbers.
    assert labels.tolist() == [
        "rookie_early",
        "rookie_early",
        "rookie_early",
        "rookie_rest",
        "veteran",
        "veteran",
    ]


def test_label_rookie_rows_respects_early_games_and_missing_column_guard():
    # early_games=2 -> only the first two rookie games are "early".
    labels = a.label_rookie_rows(_test_df(), _min_season(), early_games=2)
    assert labels.tolist()[:4] == ["rookie_early", "rookie_early", "rookie_rest", "rookie_rest"]
    # A missing required column degrades the whole column to "unknown".
    assert a.label_rookie_rows(
        _test_df().drop(columns=["week"]), _min_season()
    ).unique().tolist() == ["unknown"]
    # Empty min_season -> all unknown (no rookie signal available).
    assert a.label_rookie_rows(_test_df(), pd.Series(dtype="int64")).unique().tolist() == [
        "unknown"
    ]


def test_label_is_robust_to_a_nonunique_index():
    df = _test_df()
    df.index = [0, 0, 0, 0, 1, 1]  # force a non-unique index
    labels = a.label_rookie_rows(df, _min_season())
    assert labels.tolist() == [
        "rookie_early",
        "rookie_early",
        "rookie_early",
        "rookie_rest",
        "veteran",
        "veteran",
    ]


def test_available_models_filters_absent_columns():
    df = _test_df()  # has ridge/lgbm/nn/attn but not enet
    models = a.available_models(df)
    assert "ElasticNet" not in models
    assert set(models) == {"Ridge", "LightGBM", "NN", "Attention NN"}


def test_cohort_model_table_reports_per_bucket_mae_and_bias():
    df = _test_df()
    df[a.ROOKIE_BUCKET] = a.label_rookie_rows(df, _min_season())
    tbl = a.cohort_model_table(df)
    assert set(tbl["bucket"]) == {"rookie_early", "rookie_rest", "veteran"}
    assert set(tbl["model"]) == {"Ridge", "LightGBM", "NN", "Attention NN"}
    assert {"mae", "mae_corr", "rmse", "bias"} <= set(tbl.columns)
    # LightGBM rookie_early: preds (3,4,5) vs actual (2,3,4) -> errors all +1, so MAE 1,
    # bias +1, RMSE 1, n=3, and mae_corr 0 (the entire error is systematic bias —
    # nothing remains once the cohort mean is removed).
    lgbm_early = tbl[(tbl["model"] == "LightGBM") & (tbl["bucket"] == "rookie_early")].iloc[0]
    assert lgbm_early["n"] == 3
    assert lgbm_early["mae"] == pytest.approx(1.0)
    assert lgbm_early["bias"] == pytest.approx(1.0)
    assert lgbm_early["rmse"] == pytest.approx(1.0)
    assert lgbm_early["mae_corr"] == pytest.approx(0.0)


def test_bias_corrected_mae_isolates_systematic_error():
    df = _test_df()
    df[a.ROOKIE_BUCKET] = a.label_rookie_rows(df, _min_season())
    corr = a.bias_corrected_mae(df, a.ACTUAL, "pred_lgbm_total", a.ROOKIE_BUCKET)
    # rookie_early errors are a constant +1 -> pure bias, nothing left after centering.
    assert corr["rookie_early"] == pytest.approx(0.0)
    # veteran errors [0,-1] -> mean -0.5, centered |+0.5|,|-0.5| -> 0.5 irreducible spread.
    assert corr["veteran"] == pytest.approx(0.5)


def test_best_model_picks_lowest_overall_mae():
    # Hand-checked overall MAE: LightGBM 0.667 < Ridge 1.333 < NN 2.5 < Attn 2.667.
    name, mae = a.best_model(_test_df())
    assert name == "LightGBM"
    assert mae == pytest.approx(4 / 6)


def test_rookie_gap_quantifies_the_selected_models_cohort_delta():
    df = _test_df()
    df[a.ROOKIE_BUCKET] = a.label_rookie_rows(df, _min_season())
    r_mae, v_mae, e_mae = a.rookie_gap(df, "pred_lgbm_total")
    # rookie errors (1,1,1,0)->0.75 ; veteran (0,1)->0.5 ; early (1,1,1)->1.0.
    assert r_mae == pytest.approx(0.75)
    assert v_mae == pytest.approx(0.5)
    assert e_mae == pytest.approx(1.0)
    # Empty cohort -> NaN, no crash.
    vets_only = df[df[a.ROOKIE_BUCKET] == a.VETERAN].copy()
    assert np.isnan(a.rookie_gap(vets_only, "pred_lgbm_total")[0])
