"""No-fit positive controls for isolated WR/TE stint-window reconstruction."""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.tuning import audit_development as audit

pytestmark = pytest.mark.unit


def _frame(position):
    return pd.DataFrame(
        [
            dict(
                player_id="receiver",
                position=position,
                season=2022,
                week=week,
                recent_team="KC" if week <= 3 else "SF",
                receiving_yards=45,
                receptions=4,
                targets=6,
                carries=0,
                receiving_tds=0,
                receiving_air_yards=60,
                receiving_yards_after_catch=20,
                receiving_epa=1,
                receiving_first_downs=3,
                redzone_targets=1,
                redzone_target_share=0.25 if week <= 3 else 0.75,
            )
            for week in range(1, 7)
        ]
    )


@pytest.mark.parametrize("position", ["WR", "TE"])
def test_candidate_changes_only_windows_and_reset_recovers_main(position):
    module = importlib.import_module(f"src.{position.lower()}.features")
    original = module._compute_features
    baseline, changed, restored = (_frame(position) for _ in range(3))
    try:
        audit.select_changes("baseline")
        module._compute_features(baseline)
        audit.select_changes("stint_reset")
        module._compute_features(changed)
        assert np.isnan(changed.loc[changed.week.eq(4), "redzone_target_share_L3"].iloc[0])
        assert baseline.loc[baseline.week.eq(4), "redzone_target_share_L3"].iloc[0] == 0.25
        assert changed.loc[changed.week.eq(5), "redzone_target_share_L3"].iloc[0] == 0.75
        changed_columns = {"opportunity_index_L3", "redzone_target_share_L3"}
        pd.testing.assert_frame_equal(
            baseline.drop(columns=list(changed_columns)),
            changed.drop(columns=list(changed_columns)),
        )
        audit.select_changes("baseline_rep")
        assert module._compute_features is original
        module._compute_features(restored)
        pd.testing.assert_frame_equal(baseline, restored)
    finally:
        audit.select_changes("baseline")


def test_stint_spec_allows_ridge_changes():
    from src.tuning import ab_audit_stint_development as spec

    assert spec.POSITIONS == ["WR", "TE"]
    assert spec.SUPPORTS_STACKED is False
    candidate = next(v for v in spec.VARIANTS if v.name == "stint_reset")
    assert candidate.expect_ridge_identical is False
    assert candidate.frame_injector is audit.origin_frames
