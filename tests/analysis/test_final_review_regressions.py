"""Unavailable truth/forecasts, format parity, and source-cache recovery."""

from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_expert_comparison as compare
from src.analysis import topn_expert_gap as topn
from src.data import nflcom_loader as nfl
from src.serving import expert_sources as sleeper
from src.shared.comparison_scoring import comparison_actuals, scoring_components
from tests.analysis.test_sleeper_loader import _fake_reader
from tests.test_nflcom_loader import _fixture_reader_qb_only

pytestmark = pytest.mark.unit


def frame():
    return pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "receiving_yards": [100.0, 200.0, 300.0],
            "receptions": 5.0,
            "receiving_tds": 0.0,
            "fumbles_lost": 0.0,
            "fantasy_points": [15.0, 25.0, 35.0],
            "pred_ridge_total": [15.0, 25.0, 35.0],
            "pred_ridge_receiving_yards": [100.0, 200.0, 300.0],
            "pred_nn_total": [15.0, 25.0, 35.0],
            "pred_ridge_receptions": 5.0,
            "pred_ridge_receiving_tds": 0.0,
            "pred_ridge_fumbles_lost": 0.0,
        }
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_head_to_head_uses_same_finite_rows_and_preserves_zero(invalid):
    models = frame()
    models.loc[0, "pred_nn_total"] = 0.0
    expert = models[["player_id", "season", "week"]].assign(expert_pred_total=[0.0, invalid, 35.0])
    models.loc[1, "pred_nn_total"] = 9999.0
    result = compare._compare_one_position("WR", models, expert, "example", [2025], "ppr", 20, 42)
    assert result["n_matched"] == 2
    assert result["model"]["mae"] == result["expert"]["mae"] == 7.5
    expert["expert_pred_total"] = invalid
    unavailable = compare._compare_one_position(
        "WR", models, expert, "example", [2025], "ppr", 20, 42
    )
    assert unavailable["status"] == "unavailable" and unavailable["n_matched"] == 0


def test_duplicate_player_weeks_are_rejected():
    models = frame()
    expert = models[["player_id", "season", "week"]].assign(expert_pred_total=1.0)
    with pytest.raises(pd.errors.MergeError):
        compare._compare_one_position(
            "WR", models, pd.concat([expert, expert.iloc[:1]]), "example", [2025], "ppr", 20, 42
        )


@pytest.mark.parametrize(
    "scoring,expected",
    [("ppr", [15, 25, 35]), ("half_ppr", [12.5, 22.5, 32.5]), ("standard", [10, 20, 30])],
)
def test_fresh_model_loaders_rescore_raw_heads(monkeypatch, scoring, expected):
    monkeypatch.setattr(compare, "get_runner", lambda _: lambda: {"test_df": frame()})
    monkeypatch.setattr("src.wr.run_pipeline.run", lambda: {"test_df": frame()})
    assert compare._default_model_preds("WR", [2025], scoring).pred_ridge_total.tolist() == expected
    assert (
        topn._fresh_model_predictions("WR", [2025], scoring).pred_ridge_total.tolist() == expected
    )


def test_certified_truth_keeps_missing_pre_imputation_rows_unavailable():
    data = frame()
    data["actual_projected_total"] = [15.0, np.nan, 35.0]
    data.attrs["actual_projected_total_metadata"] = {
        "basis": "configured_target_aggregation_v1",
        "targets": list(scoring_components("WR")),
        "scoring_format": "ppr",
    }
    assert comparison_actuals(data, "WR").isna().tolist() == [False, True, False]
    standard = comparison_actuals(data, "WR", "standard")
    assert standard.iloc[0] == 10 and pd.isna(standard.iloc[1])
    data.attrs["actual_projected_total_metadata"]["targets"].append("rushing_yards")
    assert comparison_actuals(data, "WR").tolist() == [15.0, 25.0, 35.0]


def test_local_nonpositive_forecasts_with_stats_survive(tmp_path):
    source = topn.local_expert_source(topn.LocalExpertSpec("local", tmp_path / "local.csv"))
    raw = frame().assign(projected_points=[0.0, -1.0, 0.0], pred_receiving_yards=[10.0, 10.0, 0.0])
    output = source.project(raw, "WR", "ppr")
    assert output.player_id.tolist() == ["a", "b"]
    assert output.expert_pred_total.tolist() == [0.0, -1.0]


def test_zero_hit_season_selection_is_zero_f1():
    actual = frame()
    source = topn.SourceMeta(name="model", label="Model", kind="model", native_col="pred_total")
    forecasts = actual.assign(pred_total=[100.0, 50.0, 0.0])
    rows, _ = topn.season_selection_rows("WR", source, forecasts, actual, top_ns=[1])
    assert rows[0]["f1"] == 0.0


def test_nfl_transient_partial_result_is_never_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl.time, "sleep", lambda _: None)
    attempts = []

    def partial(url):
        attempts.append(url)
        if "/2/" in url:
            raise HTTPError(url, 503, "transient", None, None)
        return _fixture_reader_qb_only(url)

    result = nfl.load_nflcom_projections(
        [2024], cache_dir=str(tmp_path), weeks=[1, 2], reader=partial
    )
    assert result.attrs[nfl._FETCH_COMPLETE_ATTR] is False
    assert not list(tmp_path.glob("*.parquet"))
    healed = nfl.load_nflcom_projections(
        [2024],
        cache_dir=str(tmp_path),
        weeks=[1, 2],
        reader=lambda url: _fixture_reader_qb_only(url.replace("/2/", "/1/")),
    )
    assert healed.attrs[nfl._FETCH_COMPLETE_ATTR] is True
    assert set(healed.week) == {1, 2}
    assert list(tmp_path.glob("*.parquet"))


def test_sleeper_partial_fetch_recovers_and_sparse_seasons_do_not_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(sleeper.time, "sleep", lambda _: None)

    def partial(url):
        if "/2025/" in url:
            raise HTTPError(url, 503, "transient", None, None)
        return _fake_reader(url)

    result = sleeper.load_sleeper_projections(
        [2023, 2025], str(tmp_path), weeks=[1], reader=partial
    )
    assert result.attrs[sleeper._FETCH_COMPLETE_ATTR] is False
    assert not list(tmp_path.glob("*.parquet"))
    sparse = sleeper.load_sleeper_projections(
        [2023, 2025], str(tmp_path), weeks=[1], reader=_fake_reader
    )
    full = sleeper.load_sleeper_projections(
        [2023, 2024, 2025], str(tmp_path), weeks=[1], reader=_fake_reader
    )
    assert set(sparse.season) == {2023, 2025}
    assert set(full.season) == {2023, 2024, 2025}
    assert len(list(tmp_path.glob("*.parquet"))) == 2
