import copy

import pandas as pd
import pytest

from src.features.practice_context import PRACTICE_CONTEXT_FEATURES, REASON_FEATURES
from src.tuning import ab_practice_context as spec
from src.tuning.ab_harness import build_cells, resolve_spec

pytestmark = pytest.mark.unit


def test_grid_is_eager_four_positions_three_arms_three_seeds_and_same_injector():
    resolved = resolve_spec(spec)
    assert len(build_cells(resolved)) == 36
    assert resolved.supports_stacked is False
    assert resolved.baseline == "baseline"
    assert {v.frame_injector for v in spec.VARIANTS} == {spec.inject_context}
    assert spec.VARIANTS[0].cfg_mutator is None


def test_only_treatment_features_enter_both_model_paths():
    original = {
        "get_feature_columns_fn": lambda: ["practice_status"],
        "attn_static_features": ["practice_status"],
    }
    for variant, columns in zip(
        spec.VARIANTS[1:], (REASON_FEATURES, PRACTICE_CONTEXT_FEATURES), strict=True
    ):
        config = variant.cfg_mutator(copy.deepcopy(original))
        assert config["get_feature_columns_fn"]() == ["practice_status", *columns]
        assert config["attn_static_features"] == ["practice_status", *columns]
        assert variant.expect_ridge_identical is False
    assert original["attn_static_features"] == ["practice_status"]


def test_injection_uses_pinned_cache_and_never_silently_runs_an_empty_treatment(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(spec, "raw_data_dir", lambda _: tmp_path)
    path = tmp_path / f"injuries_{min(spec.SEASONS)}_{max(spec.SEASONS)}.parquet"
    frame = pd.DataFrame(
        {"player_id": ["a"], "season": [2025], "week": [1], "recent_team": ["BAL"]}
    )
    injuries = pd.DataFrame(
        {"gsis_id": ["a"], "season": [2025], "week": [1], "practice_primary_injury": ["Ankle"]}
    )
    injuries.to_parquet(path)
    results = spec.inject_context(frame, frame, frame)
    assert all(result.practice_lower_body.iloc[0] == 1 for result in results)
    assert not set(PRACTICE_CONTEXT_FEATURES) & set(frame)
    injuries["practice_primary_injury"] = ""
    injuries.to_parquet(path)
    with pytest.raises(ValueError, match="no injury-location signal"):
        spec.inject_context(frame, frame, frame)


def test_metrics_include_bias_sparse_groups_and_unavailable_protected_cohorts():
    frame = pd.DataFrame(
        {
            "fantasy_points": [10, 20],
            "pred_ridge_total": [12, 16],
            "game_status": [0.5, 1],
            "is_returning_from_absence": [1, 0],
        }
    )
    frame["rushing_yards"] = [100.0, 200.0]
    for target in ("rushing_tds", "receiving_tds", "receiving_yards", "receptions", "fumbles_lost"):
        frame[target] = 0.0
    # Full fantasy totals are not an allowed fallback for component scoring.
    frame["fantasy_points"] = 999.0
    for name in PRACTICE_CONTEXT_FEATURES:
        frame[name] = 0.0
    frame["practice_lower_body"] = [1.0, 0.0]
    output = spec.metric_fn({"test_df": frame}, "RB")
    assert output["Ridge"]["bias"] == -1
    assert output["Ridge @injured"]["bias"] == 2
    assert output["injured coverage"] == {"n": 1, "sparse": 1}
    assert output["weekly_reference_top24 coverage"]["available"] == 0
    assert output["rest_only coverage"]["n"] == 0
    assert output["scoring coverage"] == {"n": 2, "unavailable_actuals": 0}
    frame.loc[0, "fumbles_lost"] = float("nan")
    missing = spec.metric_fn({"test_df": frame}, "RB")
    assert missing["scoring coverage"] == {"n": 1, "unavailable_actuals": 1}
    assert missing["Ridge"]["n"] == 1
