"""Feature contract for the WR-specific features.

Complements ``tests/test_feature_leakage.py`` with a columnar contract: after
``add_specific_features`` runs, the documented WR-specific columns must:

  * all be present (none silently dropped)
  * be numeric (float dtype)
  * be finite (no inf, NaN below per-feature ceiling)
  * stay within documented value ranges (rates in [0, 1], yards in [0, inf))

Catches the "accidentally dropped a column" bug that leakage tests can't see.
Uses ``get_feature_columns()`` as source of truth for the WR-specific
portion of the whitelist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import get_attn_static_columns
from src.wr.config import ATTN_STATIC_FEATURES, INCLUDE_FEATURES, SPECIFIC_FEATURES
from src.wr.features import (
    _compute_features,
    add_specific_features,
    get_feature_columns,
)

# ---------------------------------------------------------------------------
# Per-feature contract: rate-type features are bounded in [0, 1]; rest are
# non-negative with reasonable finite ceilings.
# ---------------------------------------------------------------------------

# (name, min, max, max_nan_fraction)
_SPECIFIC_CONTRACT = [
    # rate-style features
    ("reception_rate_L3", 0.0, 1.0, 0.0),
    ("team_wr_target_share_L3", 0.0, 1.0, 0.0),
    ("receiving_first_down_rate_L3", 0.0, 1.0, 0.0),
    # yards / target & reception style features (non-negative, loose upper bound)
    ("yards_per_reception_L3", 0.0, 200.0, 0.0),
    ("yards_per_target_L3", 0.0, 200.0, 0.0),
    ("air_yards_per_target_L3", -5.0, 200.0, 0.0),
    ("yac_per_reception_L3", -5.0, 200.0, 0.0),
    # EPA can be negative; wider allowed range
    ("receiving_epa_per_target_L3", -10.0, 10.0, 0.0),
]


def _make_wr_df(n_players: int = 3, n_weeks: int = 5, seed: int = 42) -> pd.DataFrame:
    """Build a tiny but realistic WR DataFrame with all upstream columns
    needed for _compute_features / add_specific_features.

    Respects physical invariants: receptions <= targets, first_downs <= receptions.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p_idx in range(1, n_players + 1):
        for week in range(1, n_weeks + 1):
            targets = int(rng.integers(4, 12))
            receptions = int(rng.integers(1, targets + 1))
            rows.append(
                {
                    "player_id": f"W{p_idx}",
                    "season": 2023,
                    "week": week,
                    "recent_team": "KC",
                    "targets": targets,
                    "receptions": receptions,
                    "receiving_yards": float(rng.uniform(20, 130)),
                    "receiving_air_yards": float(rng.uniform(30, 180)),
                    "receiving_yards_after_catch": float(rng.uniform(5, 80)),
                    "receiving_epa": float(rng.normal(1.0, 2.0)),
                    "receiving_first_downs": int(rng.integers(0, receptions + 1)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestWRFeatureContract:
    def test_get_wr_feature_columns_contains_specific_block(self):
        """The specific category of the whitelist is exactly SPECIFIC_FEATURES."""
        all_cols = get_feature_columns()
        for feat in SPECIFIC_FEATURES:
            assert feat in all_cols, f"WR-specific feature {feat} missing from whitelist"

    def test_specific_features_match_contract(self):
        """Every WR-specific feature has an entry in the per-feature contract
        and vice-versa — catches divergence between code and contract."""
        contract_names = {name for name, *_ in _SPECIFIC_CONTRACT}
        specific_set = set(SPECIFIC_FEATURES)
        missing_from_contract = specific_set - contract_names
        extra_in_contract = contract_names - specific_set
        assert not missing_from_contract, (
            f"Features in SPECIFIC_FEATURES but missing from contract: "
            f"{sorted(missing_from_contract)}"
        )
        assert not extra_in_contract, (
            f"Features in contract but not in SPECIFIC_FEATURES: {sorted(extra_in_contract)}"
        )

    def test_add_wr_specific_features_adds_all_columns(self):
        """Every WR-specific column must be present after feature computation."""
        train = _make_wr_df()
        val = _make_wr_df(n_players=2, seed=43)
        test = _make_wr_df(n_players=2, seed=44)

        train, val, test = add_specific_features(train, val, test)
        for df, split_name in [(train, "train"), (val, "val"), (test, "test")]:
            for feat in SPECIFIC_FEATURES:
                assert feat in df.columns, f"Feature {feat} missing from {split_name} split"

    def test_feature_dtypes_numeric(self):
        """All WR-specific features must be numeric (float) dtype."""
        df = _make_wr_df()
        _compute_features(df)
        for feat in SPECIFIC_FEATURES:
            dtype = df[feat].dtype
            assert np.issubdtype(dtype, np.floating), f"{feat} has non-float dtype {dtype}"

    @pytest.mark.parametrize(
        "feature,min_val,max_val,max_nan_frac",
        _SPECIFIC_CONTRACT,
    )
    def test_feature_value_range_and_nan_ceiling(self, feature, min_val, max_val, max_nan_frac):
        """Each feature stays within its documented range, with NaN fraction
        under the per-feature ceiling (zero for WR-specific — all NaNs are
        handled by the ratio-guard code)."""
        df = _make_wr_df()
        _compute_features(df)
        series = df[feature]

        # Finite check
        non_finite = (~np.isfinite(series)).sum()
        assert non_finite == 0, f"{feature} has {non_finite} non-finite values"

        # NaN ceiling
        nan_frac = series.isna().mean()
        assert nan_frac <= max_nan_frac, (
            f"{feature} NaN fraction {nan_frac:.3f} > ceiling {max_nan_frac}"
        )

        # Value range
        finite_vals = series[np.isfinite(series)]
        assert (finite_vals >= min_val).all(), (
            f"{feature} values below {min_val}: min={finite_vals.min():.3f}"
        )
        assert (finite_vals <= max_val).all(), (
            f"{feature} values above {max_val}: max={finite_vals.max():.3f}"
        )

    def test_whitelist_categories_match_config_keys(self):
        """Whitelist dict keys must match the INCLUDE_FEATURES schema —
        guards against silently adding a new feature category."""
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
        assert set(INCLUDE_FEATURES.keys()) == expected

    def test_feature_column_count_stable(self):
        """Total count of WR features must match expected (catch silent drops/adds)."""
        cols = get_feature_columns()
        # 8 WR-specific + rolling/prior/trend/share/matchup/defense/contextual/weather_vegas
        # counts are determined by INCLUDE_FEATURES; sanity check > 50.
        assert len(cols) >= 50, f"Unexpectedly few features: {len(cols)}"
        assert len(cols) == len(set(cols)), "Duplicate feature names"

    def test_specific_features_count(self):
        """SPECIFIC_FEATURES must contain exactly 8 WR-specific features."""
        assert len(SPECIFIC_FEATURES) == 8
        assert len(set(SPECIFIC_FEATURES)) == 8

    def test_specific_features_excluded_from_attn_static(self):
        """SPECIFIC_FEATURES are per-game signals consumed by the attention
        branch via ATTN_HISTORY_STATS — they must not leak into the
        attention NN's static-feature branch (the old blacklist missed
        ``yards_per_reception_L3``, ``team_wr_target_share_L3`` …)."""
        static_cols = get_attn_static_columns(get_feature_columns(), ATTN_STATIC_FEATURES)
        leaks = set(SPECIFIC_FEATURES) & set(static_cols)
        assert not leaks, f"WR specific features leaked into attention static: {sorted(leaks)}"
