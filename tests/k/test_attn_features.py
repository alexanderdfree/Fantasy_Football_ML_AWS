"""Tests for K attention-NN L1 features.

L1 = shift(1) — the previous game's raw stat. The attention NN's static branch
consumes these instead of the L3/L5 rolling aggregates (attention handles the
longer-term smoothing). L1 columns must NEVER appear in ALL_FEATURES, since
that would leak them into Ridge and the base NN.
"""

import numpy as np
import pandas as pd
import pytest

from src.k.config import ALL_FEATURES, ATTN_L1_FEATURES
from src.k.features import compute_features


@pytest.mark.unit
class TestL1Features:
    def test_all_l1_features_created(self, make_kicker_games):
        df = make_kicker_games(n_weeks=6)
        compute_features(df)
        for col in ATTN_L1_FEATURES:
            assert col in df.columns, f"Missing L1 feature: {col}"

    def test_l1_features_excluded_from_k_all_features(self):
        """Critical: L1 features must stay out of ALL_FEATURES so Ridge
        and the base NN never see them."""
        all_features = set(ALL_FEATURES)
        for col in ATTN_L1_FEATURES:
            assert col not in all_features, (
                f"L1 feature {col} leaked into ALL_FEATURES — Ridge/base NN "
                f"would train on it, violating the attention-branch isolation."
            )

    def test_week1_l1_is_zero(self, make_kicker_games):
        """First game of a player has no prior row — L1 should fill to 0."""
        df = make_kicker_games(n_weeks=3)
        compute_features(df)
        week1 = df[df["week"] == 1].iloc[0]
        for col in ATTN_L1_FEATURES:
            assert week1[col] == 0.0, f"{col} at week 1 = {week1[col]}"

    def test_fg_attempts_l1_is_prev_week(self, make_kicker_games):
        """fg_attempts_L1 at week N = fg_att at week N-1."""
        df = make_kicker_games(n_weeks=4, fg_att=3)
        compute_features(df)
        for wk in range(2, 5):
            expected = df[df["week"] == wk - 1]["fg_att"].iloc[0]
            actual = df[df["week"] == wk]["fg_attempts_L1"].iloc[0]
            assert actual == pytest.approx(expected), (
                f"fg_attempts_L1 at week {wk}: expected {expected}, got {actual}"
            )

    def test_fg_accuracy_l1_is_prev_ratio(self, make_kicker_games):
        """fg_accuracy_L1 at week N = fg_made / fg_att at week N-1."""
        df = make_kicker_games(n_weeks=4, fg_att=4, fg_made=3)
        compute_features(df)
        expected = 3 / 4
        for wk in range(2, 5):
            actual = df[df["week"] == wk]["fg_accuracy_L1"].iloc[0]
            assert actual == pytest.approx(expected), (
                f"fg_accuracy_L1 at week {wk}: expected {expected}, got {actual}"
            )

    def test_fg_accuracy_l1_zero_denom_safe(self, make_kicker_games):
        """Previous week with 0 attempts must yield 0, not NaN/Inf."""
        df = make_kicker_games(n_weeks=3, fg_att=0, fg_made=0)
        compute_features(df)
        for wk in [2, 3]:
            val = df[df["week"] == wk]["fg_accuracy_L1"].iloc[0]
            assert np.isfinite(val), f"fg_accuracy_L1 at week {wk} is not finite: {val}"
            assert val == 0.0

    def test_l1_features_numeric_and_finite(self, make_kicker_games):
        """All L1 features must be finite after compute_features."""
        df = make_kicker_games(n_weeks=6)
        compute_features(df)
        for col in ATTN_L1_FEATURES:
            vals = df[col].to_numpy(dtype=float)
            assert np.all(np.isfinite(vals)), f"{col} has NaN/Inf: {vals}"

    def test_l1_no_current_week_leakage(self, make_kicker_games):
        """L1 at week N must not equal the current week's stat when prev differs."""
        # Build a varying series: fg_att = [1, 5, 1, 5, 1]
        df = pd.concat(
            [
                make_kicker_games(n_weeks=1, fg_att=1).assign(week=1),
                make_kicker_games(n_weeks=1, fg_att=5).assign(week=2),
                make_kicker_games(n_weeks=1, fg_att=1).assign(week=3),
                make_kicker_games(n_weeks=1, fg_att=5).assign(week=4),
                make_kicker_games(n_weeks=1, fg_att=1).assign(week=5),
            ],
            ignore_index=True,
        )
        compute_features(df)
        # At week 2, fg_att=5 but fg_attempts_L1 should be 1 (from week 1)
        w2 = df[df["week"] == 2].iloc[0]
        assert w2["fg_att"] == 5
        assert w2["fg_attempts_L1"] == 1
        # At week 3, fg_att=1 but fg_attempts_L1 should be 5 (from week 2)
        w3 = df[df["week"] == 3].iloc[0]
        assert w3["fg_att"] == 1
        assert w3["fg_attempts_L1"] == 5
