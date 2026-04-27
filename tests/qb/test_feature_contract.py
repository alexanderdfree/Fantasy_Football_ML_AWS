"""QB feature contract — catches silently dropped features.

Asserts that add_specific_features produces every column listed in
SPECIFIC_FEATURES with the correct dtype and within per-feature NaN
ceilings. get_feature_columns() is the authoritative source of the full
feature catalog; this contract covers the subset that QB owns (specific
features); shared rolling/EWMA/etc. features are built upstream.

A silent schema drift (e.g. a feature renamed without updating the whitelist)
would break this test first.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import get_attn_static_columns
from src.qb.config import ATTN_STATIC_FEATURES, INCLUDE_FEATURES, SPECIFIC_FEATURES
from src.qb.features import add_specific_features, get_feature_columns

# Per-feature NaN ceilings — the QB pipeline uses .fillna(0) so output NaN
# fraction must be 0. Anything else indicates a regression in fill logic.
NAN_CEILING = 0.0


def _make_qb_season(n_players=5, n_weeks=17, seed=42):
    """Synthetic QB-season data with the raw columns needed by _compute_features."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_players + 1):
        for wk in range(1, n_weeks + 1):
            rows.append(
                {
                    "player_id": f"QB{pid}",
                    "season": 2023,
                    "week": wk,
                    "completions": int(rng.integers(10, 30)),
                    "attempts": int(rng.integers(20, 45)),
                    "passing_yards": float(rng.integers(150, 400)),
                    "passing_tds": int(rng.integers(0, 4)),
                    "interceptions": int(rng.integers(0, 3)),
                    "sacks": int(rng.integers(0, 5)),
                    "rushing_yards": float(rng.integers(0, 60)),
                    "passing_epa": float(rng.uniform(-10, 20)),
                    "passing_air_yards": float(rng.integers(100, 350)),
                    "carries": int(rng.integers(0, 8)),
                    "passing_first_downs": int(rng.integers(5, 20)),
                    "rushing_first_downs": int(rng.integers(0, 3)),
                    "rushing_epa": float(rng.uniform(-3, 5)),
                    "passing_yards_after_catch": float(rng.integers(50, 200)),
                    "sack_yards": float(rng.integers(0, 30)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def qb_feature_df():
    """DataFrame with QB-specific features computed."""
    train = _make_qb_season(n_players=5, n_weeks=17, seed=42)
    val = _make_qb_season(n_players=3, n_weeks=17, seed=43)
    test = _make_qb_season(n_players=3, n_weeks=17, seed=44)
    train, val, test = add_specific_features(train, val, test)
    return train


@pytest.mark.unit
class TestQBFeatureContract:
    def test_all_specific_features_present(self, qb_feature_df):
        """add_specific_features must produce every column in SPECIFIC_FEATURES."""
        missing = [c for c in SPECIFIC_FEATURES if c not in qb_feature_df.columns]
        assert not missing, (
            f"Missing {len(missing)} QB-specific features: {missing}. "
            "A silent rename or deletion has broken the feature contract."
        )

    def test_specific_features_numeric_dtype(self, qb_feature_df):
        """All specific features must be numeric (float)."""
        for col in SPECIFIC_FEATURES:
            dtype = qb_feature_df[col].dtype
            assert np.issubdtype(dtype, np.number), f"{col} has non-numeric dtype {dtype}"

    def test_specific_features_nan_ceiling(self, qb_feature_df):
        """Pipeline fills NaN with 0; every specific feature must be fully dense."""
        n = len(qb_feature_df)
        for col in SPECIFIC_FEATURES:
            nan_frac = qb_feature_df[col].isna().mean()
            assert nan_frac <= NAN_CEILING, (
                f"{col} NaN fraction {nan_frac:.3f} exceeds ceiling {NAN_CEILING} "
                f"({int(nan_frac * n)} / {n} rows)"
            )

    def test_specific_features_no_inf(self, qb_feature_df):
        """No feature should contain inf/-inf — all are ratios with zero-guards."""
        for col in SPECIFIC_FEATURES:
            assert not qb_feature_df[col].isin([np.inf, -np.inf]).any(), (
                f"{col} contains inf/-inf — zero-division guard regressed"
            )

    def test_feature_columns_source_of_truth(self):
        """get_feature_columns() must include every SPECIFIC_FEATURES entry."""
        all_cols = get_feature_columns()
        missing = [c for c in SPECIFIC_FEATURES if c not in all_cols]
        assert not missing, (
            f"SPECIFIC_FEATURES entries missing from get_feature_columns(): "
            f"{missing}. INCLUDE_FEATURES['specific'] must list them."
        )

    def test_no_duplicate_feature_columns(self):
        """get_feature_columns() returns unique names (duplicates would silently mask)."""
        cols = get_feature_columns()
        duplicates = [c for c in set(cols) if cols.count(c) > 1]
        assert not duplicates, f"Duplicate feature columns: {duplicates}"

    def test_rate_features_non_negative(self, qb_feature_df):
        """Rate features are non-negative — division by zero is guarded."""
        rate_cols = [
            "completion_pct_L3",
            "td_rate_L3",
            "int_rate_L3",
            "sack_rate_L3",
            "qb_rushing_share_L3",
            "deep_ball_rate_L3",
            "pass_first_down_rate_L3",
            "rush_first_down_rate_L3",
            "yac_rate_L3",
        ]
        for col in rate_cols:
            values = qb_feature_df[col]
            assert values.min() >= 0.0, f"{col} has negative values (min={values.min()})"

    def test_specific_features_excluded_from_attn_static(self):
        """SPECIFIC_FEATURES are per-game signals consumed by the attention
        branch via ATTN_HISTORY_STATS — they must not leak into the
        attention NN's static-feature branch (the old blacklist missed them)."""
        static_cols = get_attn_static_columns(get_feature_columns(), ATTN_STATIC_FEATURES)
        leaks = set(SPECIFIC_FEATURES) & set(static_cols)
        assert not leaks, f"QB specific features leaked into attention static: {sorted(leaks)}"

    def test_feature_categories_documented(self):
        """INCLUDE_FEATURES dict contains every documented category."""
        expected = {
            "rolling",
            "prior_season",
            "ewma",
            "trend",
            "share",
            "matchup",
            "defense",
            "contextual",
            "weather_vegas",
            "specific",
        }
        assert expected == set(INCLUDE_FEATURES.keys()), (
            f"Category mismatch: {set(INCLUDE_FEATURES.keys()) ^ expected}"
        )
