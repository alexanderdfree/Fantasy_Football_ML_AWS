"""Contract tests for pregame selection, common truth, and serialized cohort reporting."""

import json
import time

import numpy as np
import pandas as pd
import pytest

from src.shared.evaluation_cohorts import (
    REFERENCE_VERSION,
    build_cohorts,
    merge_cohorts,
    reference_selection,
    regular_season_rows,
    seasonal_top_mask,
)

pytestmark = pytest.mark.unit


def frame(n=30):
    return pd.DataFrame(
        {
            "player_id": [f"p{i:02}" for i in range(n)],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "fantasy_points": np.arange(n, dtype=float),
            "receiving_yards": np.arange(n, dtype=float) * 10,
            "receiving_tds": 0.0,
            "receptions": 0.0,
            "fumbles_lost": 0.0,
            "fg_yard_points": np.arange(n, dtype=float),
            "pat_points": 0.0,
            "fg_misses": 0.0,
            "xp_misses": 0.0,
            "prior_season_mean_shared_component_points": np.arange(n, dtype=float),
            "pred_ridge_total": np.arange(n, dtype=float) + 1,
            "pred_nn_total": np.arange(n, dtype=float) - 2,
        }
    )


def reference(n=30):
    return pd.DataFrame(
        {
            "player_id": [f"p{i:02}" for i in range(n)],
            "position": "WR",
            "season": 2025,
            "week": 1,
            "reference_rank": np.arange(1, n + 1),
            "reference_version": REFERENCE_VERSION,
        }
    )


def test_weekly_reference_cannot_change_with_actuals_or_model_forecasts():
    original = frame()
    first, _ = reference_selection("WR", original, reference(), 24)
    changed = original.assign(
        fantasy_points=999 - original.fantasy_points,
        pred_ridge_total=-100 * original.pred_ridge_total,
    )
    second, _ = reference_selection("WR", changed, reference(), 24)
    assert original.loc[first, "player_id"].tolist() == changed.loc[second, "player_id"].tolist()
    assert original.loc[first, "player_id"].tolist() == [f"p{i:02}" for i in range(24)]


def test_missing_actuals_do_not_promote_reference_rank_25():
    observed = frame().iloc[1:]
    mask, meta = reference_selection("WR", observed, reference(), 24)
    assert int(mask.sum()) == 23
    assert "p24" not in set(observed.loc[mask, "player_id"])
    assert meta["reference_n"] == 24


def test_seasonal_selection_excludes_playoffs_and_keeps_full_fantasy_truth():
    regular = frame().assign(season_type="REG")
    playoff = regular.iloc[[0]].assign(season_type="POST", week=19, fantasy_points=10000)
    data = regular_season_rows(pd.concat([regular, playoff], ignore_index=True))
    assert len(data) == 30
    selected = data[seasonal_top_mask(data, 24)]
    assert "p00" not in set(selected.player_id)
    assert "p29" in set(selected.player_id)


def test_legacy_week_numbers_exclude_postseason_without_type_column():
    data = pd.DataFrame({"season": [2020, 2020, 2025, 2025], "week": [17, 18, 18, 19]})
    assert regular_season_rows(data).index.tolist() == [0, 2]


def test_cohorts_have_distinct_definitions_and_shared_component_errors():
    data = frame()
    data["fantasy_points"] += 7  # e.g. WR rushing points outside the receiving heads
    block = build_cohorts("WR", data, reference=reference())
    assert block["weekly_reference_top24"]["n"] == 24
    assert block["elite_top24"]["n"] == 24
    assert block["weekly_reference_top24"]["cohort_hash"] != block["elite_top24"]["cohort_hash"]
    assert block["weekly_reference_top24"]["models"]["Ridge"]["bias"] == 1
    assert block["weekly_reference_top24"]["models"]["Ridge"]["mae"] == 1
    assert block["weekly_reference_top24"]["actual_basis"] == "shared_projected_components_v1"
    assert block["weekly_actual_top24"]["models"]["Ridge"]["hit_rate"] == 1


def test_missing_reference_and_prior_information_are_explicit(monkeypatch):
    monkeypatch.setattr("src.shared.evaluation_cohorts.load_reference", lambda: None)
    block = build_cohorts("WR", frame().drop(columns="prior_season_mean_shared_component_points"))
    for name in ("elite_top24", "weekly_reference_top24"):
        assert block[name]["status"] == "unavailable"
        assert block[name]["n"] is None
        assert block[name]["reason"]
    assert "seasonal_actual_top24" in block


def test_k_does_not_use_offensive_prior_scores():
    data = frame().assign(position="K", prior_season_mean_fantasy_points=0)
    past = frame().assign(position="K", season=2024)
    without = build_cohorts("K", data, reference=reference())
    assert without["elite_top24"]["status"] == "unavailable"
    with_native_totals = build_cohorts("K", data, prior_frames=(past,), reference=reference())
    assert with_native_totals["elite_top24"]["n"] == 24


def test_full_fantasy_prior_mean_is_not_a_shared_component_prior():
    data = frame().rename(
        columns={"prior_season_mean_shared_component_points": "prior_season_mean_fantasy_points"}
    )
    block = build_cohorts("WR", data, reference=reference())
    assert block["elite_top24"]["reason"] == "prior_season_scores_missing"


def test_batch_extract_merge_and_history_summary_preserve_identical_cohorts():
    from src.batch.train import _extract_metrics, _merged_split_metrics
    from src.shared.benchmark_utils import summarize_pipeline_result

    data = frame()
    all_cohorts = build_cohorts("WR", data, reference=reference())
    total = {"total": {"mae": 1.0, "r2": 0.5, "rmse": 1.0}}
    cpu = {
        "ridge_metrics": total,
        "cohorts": build_cohorts("WR", data.drop(columns="pred_nn_total"), reference=reference()),
    }
    nn = {
        "nn_metrics": total,
        "cohorts": build_cohorts(
            "WR", data.drop(columns="pred_ridge_total"), reference=reference()
        ),
    }
    serialized = [_extract_metrics("WR", result) for result in (nn, cpu)]
    serialized = json.loads(json.dumps(serialized))
    merged = _merged_split_metrics("WR", "test", *serialized, {}, time.monotonic())
    summary = summarize_pipeline_result("WR", json.loads(json.dumps(merged)))
    assert summary["cohorts"] == all_cohorts


def test_split_merge_rejects_different_truth_or_reference_membership():
    a = build_cohorts("WR", frame().drop(columns="pred_nn_total"), reference=reference())
    b = build_cohorts(
        "WR",
        frame().drop(columns="pred_ridge_total").assign(receiving_yards=9999),
        reference=reference(),
    )
    with pytest.raises(ValueError, match="cohort mismatch"):
        merge_cohorts(a, b)


def test_rolling_origin_serialization_retains_each_origins_cohorts():
    from src.benchmarking.benchmark import finalize_rolling_origin

    cohort = build_cohorts("WR", frame(), reference=reference())
    rows = [(year, {"position": "WR", "ridge_mae": 1, "cohorts": cohort}) for year in (2024, 2025)]
    result = json.loads(json.dumps(finalize_rolling_origin("WR", rows)))
    assert all(row["cohorts"] == cohort for row in result["rolling_origin"]["per_origin"])
