"""Reporting unknown actuals must survive the unchanged training fills."""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import preprocess
from src.shared.aggregate_targets import DST_TARGETS, K_TARGETS, POSITION_TARGET_MAP
from src.shared.comparison_scoring import score_actual_components, scoring_components
from src.shared.comparison_truth import (
    ACTUAL_METADATA,
    SOURCE_AVAILABLE,
    comparison_source_availability,
)
from src.shared.feature_build import build_position_features

pytestmark = pytest.mark.unit


def raw_frame(position):
    columns = {
        *(name for targets in POSITION_TARGET_MAP.values() for name in targets),
        *DST_TARGETS,
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "fg_yards_made",
        "pat_made",
        "fg_missed",
        "pat_missed",
        "rushing_2pt_conversions",
        "receiving_2pt_conversions",
        "fantasy_points",
    }
    return pd.DataFrame({column: [0.0, 0.0] for column in columns}, index=[9, 3]).assign(
        position=position,
        player_id=["P1", "P2"],
        season=2025,
        week=1,
        season_type="REG",
        snap_pct=1,
    )


@pytest.mark.parametrize(
    "position,missing",
    [
        ("QB", "passing_yards"),
        ("RB", "rushing_yards"),
        ("WR", "receptions"),
        ("TE", "receiving_tds"),
        ("K", "fg_missed"),
        ("DST", "def_sacks"),
    ],
)
def test_target_fills_preserve_fitting_values_but_not_comparison_truth(position, missing):
    builder = importlib.import_module(f"src.{position.lower()}.targets").compute_targets
    raw = raw_frame(position)
    baseline = builder(raw)
    raw.loc[3, missing] = np.nan
    actual = builder(raw)
    targets = {**POSITION_TARGET_MAP, "K": K_TARGETS, "DST": DST_TARGETS}[position]
    pd.testing.assert_frame_equal(actual[list(targets)], baseline[list(targets)])
    assert (
        actual.loc[9, "actual_projected_total"]
        == score_actual_components(baseline, position).loc[9]
    )
    assert np.isnan(actual.loc[3, "actual_projected_total"])
    assert actual[SOURCE_AVAILABLE].tolist() == [True, False]
    assert actual.attrs[ACTUAL_METADATA] == {
        "basis": "configured_target_aggregation_v1",
        "targets": list(scoring_components(position)),
        "scoring_format": "ppr",
    }
    # K/DST providers and shared preparation may call the target builder again.
    assert np.isnan(builder(actual).loc[3, "actual_projected_total"])
    assert "actual_projected_total" not in raw


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_absent_or_nonfinite_raw_component_is_unavailable(position):
    raw = raw_frame(position)
    component = (
        "fg_missed"
        if position == "K"
        else "def_sacks"
        if position == "DST"
        else "receiving_fumbles_lost"
    )
    assert not comparison_source_availability(raw.drop(columns=component), position).any()
    raw.loc[3, component] = np.inf
    assert comparison_source_availability(raw, position).tolist() == [True, False]


def test_preprocessing_mask_survives_raw_nan_fills_and_row_reordering():
    from src.wr.targets import compute_targets

    raw = raw_frame("WR")
    raw.loc[3, "receiving_fumbles_lost"] = np.nan
    prepared = preprocess(raw)
    assert prepared.loc[3, "receiving_fumbles_lost"] == 0
    # General feature engineering concatenates new blocks and drops attrs.
    prepared = pd.concat([prepared, pd.DataFrame({"other": 1}, index=prepared.index)], axis=1)
    assert not prepared.attrs
    result = compute_targets(prepared.iloc[::-1])
    assert result[SOURCE_AVAILABLE].tolist() == [False, True]
    assert np.isnan(result.iloc[0].actual_projected_total)
    assert result.iloc[1].actual_projected_total == 0


def test_feature_preparation_restores_metadata_after_concat_and_reordering(monkeypatch):
    from src.wr.targets import compute_targets

    monkeypatch.setattr("src.shared.feature_build.merge_schedule_features", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.shared.feature_build.merge_team_box_score_features", lambda *a, **k: None
    )
    raw = raw_frame("WR")
    raw.loc[3, "receiving_yards"] = np.nan
    targeted = compute_targets(raw)

    def features(*frames, **kwargs):
        return tuple(
            pd.concat([frame, pd.DataFrame({"feature": 1}, index=frame.index)], axis=1).iloc[::-1]
            for frame in frames
        )

    cfg = {
        "add_features_fn": features,
        "fill_nans_fn": lambda a, b, c, *args: (a, b, c),
        "specific_features": [],
    }
    outputs = build_position_features(
        targeted.copy(), targeted.copy(), targeted.copy(), cfg, ["receiving_yards"]
    )
    for frame in outputs:
        assert frame.attrs[ACTUAL_METADATA] == targeted.attrs[ACTUAL_METADATA]
        assert frame[SOURCE_AVAILABLE].tolist() == [False, True]
        assert np.isnan(frame.iloc[0].actual_projected_total)
        assert frame.iloc[1].actual_projected_total == 0
        assert frame.receiving_yards.eq(0).all()


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_reporting_columns_never_enter_production_features(position):
    from src.shared.registry import get_config

    config = get_config(position)
    for names in (
        config["get_feature_columns_fn"](),
        config["attn_static_features"],
        config["attn_history_stats"],
    ):
        assert not any(name.startswith("actual_projected_") for name in names)


def test_kicker_transforms_and_dst_excluded_points_allowed():
    from src.dst.targets import compute_targets as dst_targets
    from src.k.targets import compute_targets as k_targets

    kicker = raw_frame("K").assign(fg_yards_made=45.0, pat_made=2, fg_missed=1, pat_missed=1)
    assert k_targets(kicker).actual_projected_total.eq(4.5).all()
    defense = raw_frame("DST").assign(points_allowed=np.nan, yards_allowed=350.0)
    assert dst_targets(defense).actual_projected_total.eq(-1.0).all()
    defense.loc[3, "yards_allowed"] = np.nan
    assert np.isnan(dst_targets(defense).loc[3, "actual_projected_total"])
