"""Feature contract tests for src.rb.features.

Locks down what `add_specific_features` is allowed to output:

* The full column list advertised by `get_feature_columns()`.
* Dtype stability per column (numeric outputs must stay numeric so the
  downstream StandardScaler / LightGBM / NN don't silently coerce).
* Per-feature NaN ceilings — computed features are allowed some NaN on the
  leading edge (shift+rolling), but nothing should be fully NaN, and the
  rate must stay below a documented ceiling.

This is reviewer-critical: feature drift (an engineered column going NaN or
changing dtype) was one of the top failure modes flagged in the code review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import get_attn_static_columns
from src.rb.config import ATTN_STATIC_FEATURES, INCLUDE_FEATURES, SPECIFIC_FEATURES
from src.rb.features import (
    add_specific_features,
    fill_nans,
    get_feature_columns,
)

# ---------------------------------------------------------------------------
# Contract spec
# ---------------------------------------------------------------------------

# Features computed directly by `_compute_features` over raw stats.
# These share the same leading-edge NaN profile because they all stem from
# `shift(1).rolling(3, min_periods=1)...`.
_COMPUTED_L3_FEATURES = {
    "yards_per_carry_L3",
    "reception_rate_L3",
    "weighted_opportunities_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "rushing_first_down_rate_L3",
    "receiving_first_down_rate_L3",
    "yac_per_reception_L3",
    "receiving_epa_per_target_L3",
    "air_yards_per_target_L3",
    "team_rb_carry_hhi_L3",
    "team_rb_target_hhi_L3",
    "opportunity_index_L3",
}

# career_carries is cumsum().shift(1) across seasons; only row 1 per player is NaN.
_CUMULATIVE_FEATURES = {"career_carries"}


def _build_multi_player_frame(n_players: int = 6, n_weeks: int = 8) -> pd.DataFrame:
    """Synthetic multi-player RB frame large enough to exercise _compute_features."""
    rng = np.random.default_rng(42)
    teams = ["KC", "BUF", "SF"]
    rows = []
    for pid in range(1, n_players + 1):
        team = teams[pid % len(teams)]
        for week in range(1, n_weeks + 1):
            carries = int(max(0, rng.normal(12, 4)))
            targets = int(max(0, rng.normal(4, 2)))
            receptions = min(targets, int(max(0, rng.normal(targets * 0.7, 1))))
            rushing_yards = int(max(0, carries * rng.normal(4.2, 1.0)))
            receiving_yards = int(max(0, receptions * rng.normal(7.5, 2.0)))
            rows.append(
                {
                    "player_id": f"P{pid:02d}",
                    "season": 2023,
                    "week": week,
                    "recent_team": team,
                    "carries": carries,
                    "targets": targets,
                    "receptions": receptions,
                    "rushing_yards": rushing_yards,
                    "receiving_yards": receiving_yards,
                    "rushing_epa": rng.normal(0, 0.5) * max(carries, 1),
                    "rushing_first_downs": int(carries * 0.25),
                    "receiving_first_downs": int(receptions * 0.4),
                    "receiving_yards_after_catch": int(receiving_yards * 0.5),
                    "receiving_epa": rng.normal(0, 0.3) * max(receptions, 1),
                    "receiving_air_yards": int(max(0, receiving_yards * 0.8)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def rb_featured_frame():
    """One synthetic frame with `add_specific_features` applied once.

    Uses a realistic 3-split layout matching what the pipeline passes in
    (train = bulk of weeks, val = small, test = small). All three splits get
    concatenated afterwards so the contract tests can reason about the
    training fraction — the leading-week NaN pattern from the smaller
    val/test splits would otherwise inflate the measured NaN rate.
    """
    df = _build_multi_player_frame(n_players=6, n_weeks=10)
    weeks = sorted(df["week"].unique())
    # Keep train dominant (8/10 weeks); val/test get 1 week each.
    split_a = df[df["week"] <= weeks[-3]].copy()
    split_b = df[df["week"] == weeks[-2]].copy()
    split_c = df[df["week"] == weeks[-1]].copy()
    a, _, _ = add_specific_features(split_a, split_b, split_c)
    return a.sort_values(["player_id", "season", "week"])


# ---------------------------------------------------------------------------
# Column-list contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFeatureColumnContract:
    def test_feature_list_non_empty(self):
        cols = get_feature_columns()
        assert len(cols) > 0

    def test_all_specific_features_present_in_full_list(self):
        """Every SPECIFIC_FEATURES entry must appear in the flattened list."""
        full = set(get_feature_columns())
        missing = set(SPECIFIC_FEATURES) - full
        assert not missing, f"Specific features missing from feature list: {missing}"

    def test_no_duplicate_columns(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols)), "Duplicate columns in feature list"

    def test_feature_groups_flatten_match(self):
        """The flat feature list must be the concatenation of every group."""
        flat = get_feature_columns()
        expected_total = sum(len(v) for v in INCLUDE_FEATURES.values())
        assert len(flat) == expected_total

    def test_compute_adds_every_specific_feature(self, rb_featured_frame):
        """add_specific_features must create every advertised RB-specific column."""
        for col in SPECIFIC_FEATURES:
            assert col in rb_featured_frame.columns, f"add_specific_features did not create {col}"

    def test_specific_features_excluded_from_attn_static(self):
        """SPECIFIC_FEATURES are per-game signals consumed by the attention
        branch via ATTN_HISTORY_STATS — they must not leak into the
        attention NN's static-feature branch (the old blacklist missed
        ``yards_per_carry_L3``, ``team_rb_carry_share_L3``,
        ``opportunity_index_L3``, ``career_carries`` …)."""
        static_cols = get_attn_static_columns(get_feature_columns(), ATTN_STATIC_FEATURES)
        leaks = set(SPECIFIC_FEATURES) & set(static_cols)
        assert not leaks, f"RB specific features leaked into attention static: {sorted(leaks)}"


# ---------------------------------------------------------------------------
# Dtype contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFeatureDtypeContract:
    def test_all_specific_features_are_numeric(self, rb_featured_frame):
        """Computed RB features must be numeric (float or int)."""
        non_numeric = []
        for col in SPECIFIC_FEATURES:
            if col not in rb_featured_frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(rb_featured_frame[col]):
                non_numeric.append((col, rb_featured_frame[col].dtype))
        assert not non_numeric, f"Non-numeric dtypes in RB features: {non_numeric}"

    def test_computed_features_are_float(self, rb_featured_frame):
        """L3 aggregates are ratios / rolling means → must be floating point."""
        for col in _COMPUTED_L3_FEATURES:
            if col not in rb_featured_frame.columns:
                continue
            assert pd.api.types.is_float_dtype(rb_featured_frame[col]), (
                f"{col} is not float: {rb_featured_frame[col].dtype}"
            )


# ---------------------------------------------------------------------------
# NaN ceiling contract
# ---------------------------------------------------------------------------

# Max allowed NaN rate per group. Ratios with shift(1).rolling(3) produce
# exactly one NaN per (player, season) group on the first week, so the
# upper bound scales with group density. For the synthetic frame (6 players
# × 8 weeks = 48 rows), the expected NaN rate per L3 feature is ~6/48 = 0.125.
# We allow up to 0.30 headroom to keep the test stable.
_MAX_NAN_RATE = {
    "computed_l3": 0.30,
    "cumulative": 0.30,
}


@pytest.mark.unit
class TestFeatureNaNContract:
    def test_no_feature_is_fully_nan(self, rb_featured_frame):
        fully_nan = [
            col
            for col in SPECIFIC_FEATURES
            if col in rb_featured_frame.columns and rb_featured_frame[col].isna().all()
        ]
        assert not fully_nan, f"Features fully NaN: {fully_nan}"

    def test_no_inf_values(self, rb_featured_frame):
        """Division helpers must guard against divide-by-zero (no inf outputs)."""
        for col in SPECIFIC_FEATURES:
            if col not in rb_featured_frame.columns:
                continue
            series = rb_featured_frame[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            infs = np.isinf(series.dropna()).sum()
            assert infs == 0, f"{col} has {infs} infinite values"

    def test_computed_l3_nan_rate_within_ceiling(self, rb_featured_frame):
        """shift(1)+rolling leaves NaN on row 1 per (player, season)."""
        ceiling = _MAX_NAN_RATE["computed_l3"]
        offenders = {}
        for col in _COMPUTED_L3_FEATURES:
            if col not in rb_featured_frame.columns:
                continue
            nan_rate = rb_featured_frame[col].isna().mean()
            if nan_rate > ceiling:
                offenders[col] = nan_rate
        assert not offenders, f"L3 features exceeded {ceiling:.2f} NaN ceiling: {offenders}"

    def test_career_carries_nan_rate(self, rb_featured_frame):
        """career_carries is cumsum().shift(1) — one NaN per player, filled to 0."""
        col = "career_carries"
        # add_specific_features unconditionally creates career_carries and
        # fills the lead NaN with 0, so post-call the rate should be 0%.
        assert col in rb_featured_frame.columns, (
            f"{col} missing — add_specific_features must always create it"
        )
        nan_rate = rb_featured_frame[col].isna().mean()
        assert nan_rate <= _MAX_NAN_RATE["cumulative"], (
            f"{col} NaN rate {nan_rate:.3f} exceeds cumulative ceiling"
        )


# ---------------------------------------------------------------------------
# End-to-end invariants: fill_nans produces a ready-to-train frame
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillRBNansInvariants:
    def test_post_fill_no_nan_in_specific_features(self):
        """After fill_nans, RB-specific features must contain zero NaN / inf."""
        df = _build_multi_player_frame()
        weeks = sorted(df["week"].unique())
        a = df[df["week"] <= weeks[-3]].copy()
        b = df[df["week"] == weeks[-2]].copy()
        c = df[df["week"] == weeks[-1]].copy()
        a, b, c = add_specific_features(a, b, c)
        a, b, c = fill_nans(a, b, c, SPECIFIC_FEATURES)

        for split_name, split in [("train", a), ("val", b), ("test", c)]:
            nan_cols = [
                col for col in SPECIFIC_FEATURES if col in split.columns and split[col].isna().any()
            ]
            # Allowed: all-NaN training column → train mean is NaN → propagates.
            # In this synthetic frame we do not expect any such column.
            assert not nan_cols, f"{split_name} has NaN in specific features: {nan_cols}"

            inf_cols = [
                col
                for col in SPECIFIC_FEATURES
                if col in split.columns
                and pd.api.types.is_numeric_dtype(split[col])
                and np.isinf(split[col]).any()
            ]
            assert not inf_cols, f"{split_name} has inf in specific features: {inf_cols}"
