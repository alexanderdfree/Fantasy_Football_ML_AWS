import importlib

import pandas as pd
import pytest

from src.features.practice_context import (
    PRACTICE_CONTEXT_FEATURES,
    attach_historical_context,
    attach_observation_features,
    reason_features,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Not injury related - resting player"], {"practice_rest_only"}),
        (["Knee", "Not injury related - resting player"], {"practice_lower_body"}),
        (["Hamstring / rest"], {"practice_lower_body"}),
        (
            ["Knee, Shoulder", "Concussion"],
            {"practice_lower_body", "practice_upper_body", "practice_head"},
        ),
        (["Illness", "Ankle"], {"practice_illness", "practice_lower_body"}),
        (["Undisclosed"], {"practice_other"}),
        ([None, "  ", "Note"], {"practice_reason_unknown"}),
    ],
)
def test_reason_categories_are_multi_hot_and_never_discount_mixed_rest(labels, expected):
    features = reason_features(labels)
    assert {name for name, value in features.items() if value} == expected


def test_unknown_report_differs_from_published_healthy_absence():
    assert sum(reason_features([], coverage="published_absence").values()) == 0
    assert reason_features([], coverage="unknown")["practice_reason_unknown"] == 1


def test_historical_and_live_transforms_match_and_preserve_frame_identity():
    frame = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "a"],
            "season": [2025] * 4,
            "week": [2] * 4,
            "recent_team": ["BAL", "BAL", "NE", "BAL"],
            "rushing_yards": [1, 2, 3, 1],
        },
        index=[10, 10, 4, 5],
    )
    original = frame.copy()
    injuries = pd.DataFrame(
        {
            "gsis_id": ["a", "a"],
            "season": [2025] * 2,
            "week": [2] * 2,
            "team": ["BAL"] * 2,
            "practice_primary_injury": ["Knee", "Not injury related - resting player"],
            "practice_status": ["Limited Participation in Practice"] * 2,
        }
    )
    historical = attach_historical_context(frame, injuries)
    observations = [
        {
            "player_id": pid,
            "season": 2025,
            "week": 2,
            "coverage": coverage,
            "injury_descriptions": labels,
        }
        for pid, coverage, labels in (
            ("a", "reported", ["Knee", "Not injury related - resting player"]),
            ("b", "published_absence", []),
            ("c", "unknown", []),
        )
    ]
    live = attach_observation_features(frame, observations)
    pd.testing.assert_frame_equal(historical, live)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(historical[original.columns], original)
    assert list(historical["practice_rest_only"]) == [0, 0, 0, 0]
    assert list(historical["practice_reason_unknown"]) == [0, 0, 1, 0]


def test_other_weeks_postseason_and_unknown_team_reports_do_not_supply_coverage():
    frame = pd.DataFrame(
        {"player_id": ["a"], "season": [2025], "week": [2], "recent_team": ["BAL"]}
    )
    injuries = pd.DataFrame(
        {
            "gsis_id": ["a", "b"],
            "season": [2025] * 2,
            "week": [1, 2],
            "team": ["BAL"] * 2,
            "game_type": ["REG", "POST"],
            "practice_primary_injury": ["Knee"] * 2,
            "practice_status": ["Limited Participation in Practice"] * 2,
        }
    )
    result = attach_historical_context(frame, injuries)
    assert result["practice_reason_unknown"].iloc[0] == 1
    assert result["practice_lower_body"].iloc[0] == 0


def test_duplicate_live_identities_are_rejected():
    frame = pd.DataFrame({"player_id": ["a"], "season": [2025], "week": [2]})
    row = {"player_id": "a", "season": 2025, "week": 2, "coverage": "reported"}
    with pytest.raises(ValueError, match="duplicate"):
        attach_observation_features(frame, [row, row])


@pytest.mark.parametrize("position", ["qb", "rb", "wr", "te", "k", "dst"])
def test_candidate_features_are_disabled_in_every_production_model(position):
    cfg = importlib.import_module(f"src.{position}.run_pipeline").CONFIG
    assert not set(PRACTICE_CONTEXT_FEATURES) & set(cfg["get_feature_columns_fn"]())
    assert not set(PRACTICE_CONTEXT_FEATURES) & set(cfg["attn_static_features"])
