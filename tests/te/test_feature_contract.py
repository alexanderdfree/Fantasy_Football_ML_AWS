"""Feature contract tests for TE.te_features.add_specific_features.

Asserts:
  - the 8 TE-specific engineered columns are present on every split,
  - outputs are numeric (float-compatible) and finite after nan-fill,
  - NaN ceilings are enforced (no column allowed to go all-NaN on non-empty input),
  - `get_feature_columns()` is the source of truth for the flattened whitelist,
  - the contract survives a fill_nans() round-trip.

A broken contract here means silent model degradation: the model training code
reads columns from `get_feature_columns()`, so a dropped or renamed feature
silently becomes zero-filled in the training matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import get_attn_static_columns
from src.te.config import ATTN_STATIC_FEATURES, SPECIFIC_FEATURES
from src.te.features import (
    add_specific_features,
    fill_nans,
    get_feature_columns,
)

pytestmark = pytest.mark.unit


# The exact 8 columns documented in te_config.SPECIFIC_FEATURES.
# If a feature is added/removed, both the config and this contract must move together.
EXPECTED_TE_SPECIFIC = [
    "yards_per_reception_L3",
    "reception_rate_L3",
    "yac_per_reception_L3",
    "team_te_target_share_L3",
    "receiving_epa_per_target_L3",
    "receiving_first_down_rate_L3",
    "air_yards_per_target_L3",
    "td_rate_per_target_L3",
]


def _make_minimal_split(
    player_ids=("T1", "T2", "T3"),
    season: int = 2023,
    n_weeks: int = 5,
    team: str = "KC",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a TE split with every raw column touched by _compute_features."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in player_ids:
        for wk in range(1, n_weeks + 1):
            rows.append(
                {
                    "player_id": pid,
                    "season": season,
                    "week": wk,
                    "recent_team": team,
                    "receptions": int(rng.poisson(4)),
                    "targets": int(rng.poisson(6)),
                    "receiving_yards": float(rng.normal(55, 15)),
                    "receiving_air_yards": float(rng.normal(70, 20)),
                    "receiving_yards_after_catch": float(rng.normal(20, 8)),
                    "receiving_epa": float(rng.normal(1.0, 1.5)),
                    "receiving_first_downs": int(rng.poisson(2)),
                    "receiving_tds": int(rng.binomial(1, 0.2)),
                }
            )
    return pd.DataFrame(rows)


class TestTESpecificFeatureContract:
    def test_specific_features_constant_matches_config(self):
        """The exported specific-feature list matches the config source."""
        assert list(SPECIFIC_FEATURES) == EXPECTED_TE_SPECIFIC

    def test_all_specific_columns_present_after_add(self):
        train = _make_minimal_split(seed=1)
        val = _make_minimal_split(player_ids=("V1",), seed=2)
        test = _make_minimal_split(player_ids=("X1",), seed=3)

        t, v, x = add_specific_features(train, val, test)
        for col in EXPECTED_TE_SPECIFIC:
            for split_name, df in [("train", t), ("val", v), ("test", x)]:
                assert col in df.columns, f"{col} missing from {split_name}"

    def test_specific_columns_are_numeric(self):
        train = _make_minimal_split(seed=1)
        val = _make_minimal_split(player_ids=("V1",), seed=2)
        test = _make_minimal_split(player_ids=("X1",), seed=3)
        t, v, x = add_specific_features(train, val, test)
        for col in EXPECTED_TE_SPECIFIC:
            for df in (t, v, x):
                assert pd.api.types.is_numeric_dtype(df[col]), (
                    f"{col} is not numeric (dtype={df[col].dtype})"
                )

    def test_no_inf_after_fill_nans(self):
        """fill_nans must eliminate inf and NaN from specific columns."""
        train = _make_minimal_split(seed=1)
        val = _make_minimal_split(player_ids=("V1",), seed=2)
        test = _make_minimal_split(player_ids=("X1",), seed=3)
        t, v, x = add_specific_features(train, val, test)
        t, v, x = fill_nans(t, v, x, EXPECTED_TE_SPECIFIC)
        for col in EXPECTED_TE_SPECIFIC:
            for df in (t, v, x):
                assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} has inf after fill"
                assert not df[col].isna().any(), f"{col} has NaN after fill"

    def test_specific_columns_not_all_nan(self):
        """Every specific column produces at least one non-NaN value on realistic input."""
        train = _make_minimal_split(n_weeks=6, seed=1)
        val = _make_minimal_split(player_ids=("V1",), n_weeks=6, seed=2)
        test = _make_minimal_split(player_ids=("X1",), n_weeks=6, seed=3)
        t, _, _ = add_specific_features(train, val, test)
        for col in EXPECTED_TE_SPECIFIC:
            assert t[col].notna().any(), f"{col} is entirely NaN on non-empty input"


class TestTEFeatureColumnsSourceOfTruth:
    def test_get_te_feature_columns_returns_nonempty_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_te_feature_columns_includes_all_specific(self):
        """SPECIFIC_FEATURES are a subset of the flattened feature list."""
        cols = get_feature_columns()
        for c in EXPECTED_TE_SPECIFIC:
            assert c in cols, f"{c} missing from flattened TE feature list"

    def test_get_te_feature_columns_is_unique(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols)), "duplicate column in get_feature_columns()"

    def test_get_te_feature_columns_is_deterministic(self):
        """Calling twice yields the exact same ordered list."""
        assert get_feature_columns() == get_feature_columns()

    def test_specific_features_excluded_from_attn_static(self):
        """SPECIFIC_FEATURES are per-game signals consumed by the attention
        branch via ATTN_HISTORY_STATS — they must not leak into the
        attention NN's static-feature branch."""
        static_cols = get_attn_static_columns(get_feature_columns(), ATTN_STATIC_FEATURES)
        leaks = set(SPECIFIC_FEATURES) & set(static_cols)
        assert not leaks, f"TE specific features leaked into attention static: {sorted(leaks)}"
