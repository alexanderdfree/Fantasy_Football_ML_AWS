"""Tests that central feature engineering (src/features/engineer.py) is free of data leakage."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import build_features, fill_nans_safe, get_feature_columns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_games(
    player_id="P1",
    season=2023,
    n_weeks=6,
    position="RB",
    recent_team="KC",
    opponent_team="SF",
    fantasy_points=10.0,
    targets=5,
    receptions=3,
    carries=12,
    rushing_yards=60,
    receiving_yards=40,
    passing_yards=0,
    attempts=0,
    snap_pct=0.7,
    rushing_tds=0,
    receiving_tds=0,
    passing_tds=0,
    interceptions=0,
    fumbles_lost=0,
    receiving_air_yards=20.0,
    sacks=1,
):
    """Create a synthetic multi-week DataFrame for one player."""
    return pd.DataFrame(
        {
            "player_id": [player_id] * n_weeks,
            "season": [season] * n_weeks,
            "week": list(range(1, n_weeks + 1)),
            "position": [position] * n_weeks,
            "recent_team": [recent_team] * n_weeks,
            "opponent_team": [opponent_team] * n_weeks,
            "fantasy_points": [fantasy_points] * n_weeks,
            "fantasy_points_floor": [fantasy_points * 0.8] * n_weeks,
            "targets": [targets] * n_weeks,
            "receptions": [receptions] * n_weeks,
            "carries": [carries] * n_weeks,
            "rushing_yards": [rushing_yards] * n_weeks,
            "receiving_yards": [receiving_yards] * n_weeks,
            "passing_yards": [passing_yards] * n_weeks,
            "attempts": [attempts] * n_weeks,
            "snap_pct": [snap_pct] * n_weeks,
            "rushing_tds": [rushing_tds] * n_weeks,
            "receiving_tds": [receiving_tds] * n_weeks,
            "passing_tds": [passing_tds] * n_weeks,
            "interceptions": [interceptions] * n_weeks,
            "fumbles_lost": [fumbles_lost] * n_weeks,
            "receiving_air_yards": [receiving_air_yards] * n_weeks,
            "sacks": [sacks] * n_weeks,
            "is_home": [1, 0] * (n_weeks // 2) + [1] * (n_weeks % 2),
        }
    )


def _fake_schedules():
    """Create minimal fake schedule data covering teams/seasons used in tests."""
    rows = []
    for season in [2022, 2023]:
        for week in range(1, 19):
            rows.append(
                {
                    "game_type": "REG",
                    "season": season,
                    "week": week,
                    "away_team": "SF",
                    "home_team": "KC",
                    "home_score": 24,
                    "away_score": 17,
                    "spread_line": -3.0,
                    "total_line": 47.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _mock_schedule_parquet():
    """Prevent tests from reading real parquet files."""
    fake = _fake_schedules()
    with patch("src.features.engineer.pd.read_parquet", return_value=fake):
        yield


def _two_player_df(n_weeks=8):
    """Two RBs on opposing teams so matchup features populate."""
    p1 = _make_games(
        "P1",
        season=2023,
        n_weeks=n_weeks,
        position="RB",
        recent_team="KC",
        opponent_team="SF",
        fantasy_points=15.0,
        targets=6,
        carries=14,
        rushing_yards=70,
        receiving_yards=40,
    )
    p2 = _make_games(
        "P2",
        season=2023,
        n_weeks=n_weeks,
        position="RB",
        recent_team="SF",
        opponent_team="KC",
        fantasy_points=10.0,
        targets=4,
        carries=10,
        rushing_yards=50,
        receiving_yards=30,
    )
    return pd.concat([p1, p2], ignore_index=True)


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


class TestRollingLeakage:
    def test_week1_rolling_features_are_nan_or_zero(self):
        """Week 1 has no prior data after shift(1), so rolling features must be NaN or 0."""
        df = build_features(_make_games(n_weeks=6))
        week1 = df[df["week"] == 1]
        rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
        for col in rolling_cols:
            val = week1[col].iloc[0]
            assert val == 0.0 or np.isnan(val), f"{col} = {val} on week 1 (expected NaN or 0)"

    def test_rolling_features_dont_see_future(self):
        """Mutating a future week's raw stat must not change earlier weeks' features."""
        base = _make_games(n_weeks=6, fantasy_points=10.0)
        df_original = build_features(base.copy())

        # Spike week 6 fantasy_points
        mutated = base.copy()
        mutated.loc[mutated["week"] == 6, "fantasy_points"] = 999.0
        df_mutated = build_features(mutated)

        # Weeks 1-5 rolling features should be identical
        rolling_cols = [c for c in df_original.columns if c.startswith("rolling_")]
        for week in range(1, 6):
            orig_row = df_original[df_original["week"] == week][rolling_cols].iloc[0]
            mut_row = df_mutated[df_mutated["week"] == week][rolling_cols].iloc[0]
            pd.testing.assert_series_equal(
                orig_row, mut_row, check_names=False, obj=f"week {week} rolling features"
            )


# ---------------------------------------------------------------------------
# EWMA features
# ---------------------------------------------------------------------------


class TestEWMALeakage:
    def test_ewma_features_dont_see_future(self):
        """Mutating a future week must not change earlier weeks' EWMA features."""
        base = _make_games(n_weeks=6, targets=5)
        df_original = build_features(base.copy())

        mutated = base.copy()
        mutated.loc[mutated["week"] == 6, "targets"] = 999
        df_mutated = build_features(mutated)

        ewma_cols = [c for c in df_original.columns if c.startswith("ewma_")]
        for week in range(1, 6):
            orig_row = df_original[df_original["week"] == week][ewma_cols].iloc[0]
            mut_row = df_mutated[df_mutated["week"] == week][ewma_cols].iloc[0]
            pd.testing.assert_series_equal(
                orig_row, mut_row, check_names=False, obj=f"week {week} EWMA features"
            )


# ---------------------------------------------------------------------------
# Trend / momentum features
# ---------------------------------------------------------------------------


class TestTrendLeakage:
    def test_trend_features_dont_see_future(self):
        """Mutating a future week must not change earlier weeks' trend features."""
        base = _make_games(n_weeks=8, carries=12)
        df_original = build_features(base.copy())

        mutated = base.copy()
        mutated.loc[mutated["week"] == 8, "carries"] = 999
        df_mutated = build_features(mutated)

        trend_cols = [c for c in df_original.columns if c.startswith("trend_")]
        for week in range(1, 8):
            orig_row = df_original[df_original["week"] == week][trend_cols].iloc[0]
            mut_row = df_mutated[df_mutated["week"] == week][trend_cols].iloc[0]
            pd.testing.assert_series_equal(
                orig_row, mut_row, check_names=False, obj=f"week {week} trend features"
            )


# ---------------------------------------------------------------------------
# Share / usage features
# ---------------------------------------------------------------------------


class TestShareLeakage:
    def test_share_features_use_shift(self):
        """Current week's targets must not appear in this week's share features.

        Two players on the SAME team: a spike in P1's week-5 targets should
        not affect P1's week-5 target share (only prior weeks feed the share).
        """
        p1 = _make_games(
            "P1",
            season=2023,
            n_weeks=6,
            position="RB",
            recent_team="KC",
            opponent_team="SF",
            targets=6,
        )
        p2 = _make_games(
            "P2",
            season=2023,
            n_weeks=6,
            position="RB",
            recent_team="KC",
            opponent_team="SF",
            targets=4,
        )
        df = pd.concat([p1, p2], ignore_index=True)
        # Spike P1 targets in week 5
        df.loc[(df["player_id"] == "P1") & (df["week"] == 5), "targets"] = 100
        result = build_features(df)

        # P1's week 5 share should match week 4 share (both see only prior weeks)
        p1_w5 = result[(result["player_id"] == "P1") & (result["week"] == 5)]
        p1_w4 = result[(result["player_id"] == "P1") & (result["week"] == 4)]
        share_cols = [c for c in result.columns if "target_share" in c]
        for col in share_cols:
            w5_val = p1_w5[col].iloc[0]
            w4_val = p1_w4[col].iloc[0]
            assert w5_val == pytest.approx(w4_val, abs=0.01), (
                f"{col}: week 5 = {w5_val}, week 4 = {w4_val} — spike leaked into current week"
            )


# ---------------------------------------------------------------------------
# snap_pct lag
# ---------------------------------------------------------------------------


class TestSnapPctLeakage:
    def test_snap_pct_is_lagged(self):
        """snap_pct feature should be the PRIOR week's value, not current week's."""
        base = _make_games(n_weeks=4, snap_pct=0.5)
        # Set week 3 snap_pct to a distinct value
        base.loc[base["week"] == 3, "snap_pct"] = 0.99
        result = build_features(base)

        # Week 3 should still show ~0.5 (prior week's snap_pct), not 0.99
        w3_snap = result[result["week"] == 3]["snap_pct"].iloc[0]
        assert w3_snap != pytest.approx(0.99, abs=0.01), (
            f"snap_pct at week 3 = {w3_snap}, current week value leaked"
        )

        # Week 4 should reflect the 0.99 from week 3
        w4_snap = result[result["week"] == 4]["snap_pct"].iloc[0]
        assert w4_snap == pytest.approx(0.99, abs=0.01), (
            f"snap_pct at week 4 = {w4_snap}, expected 0.99 (lagged from week 3)"
        )


# ---------------------------------------------------------------------------
# Opponent / matchup features
# ---------------------------------------------------------------------------


class TestOpponentLeakage:
    def test_opponent_features_lagged(self):
        """Matchup features must use prior weeks' opponent stats, not current week."""
        df = _two_player_df(n_weeks=8)
        # Spike opponent allowed points in week 7
        df.loc[(df["recent_team"] == "SF") & (df["week"] == 7), "fantasy_points"] = 200.0
        result = build_features(df)

        opp_cols = [c for c in result.columns if c.startswith("opp_")]
        if not opp_cols:
            pytest.skip("No opponent feature columns generated")

        # P1 plays against SF. Week 7's opp features should NOT include the week 7 spike.
        p1_w7 = result[(result["player_id"] == "P1") & (result["week"] == 7)]
        p1_w6 = result[(result["player_id"] == "P1") & (result["week"] == 6)]
        for col in opp_cols:
            w7_val = p1_w7[col].iloc[0]
            w6_val = p1_w6[col].iloc[0]
            if np.isnan(w6_val) and np.isnan(w7_val):
                continue
            # The week-7 spike should not appear until week 8
            assert abs(w7_val - w6_val) < 50, (
                f"{col}: week 7 = {w7_val}, week 6 = {w6_val} — possible current-week leakage"
            )


# ---------------------------------------------------------------------------
# Cross-season isolation
# ---------------------------------------------------------------------------


class TestCrossSeasonLeakage:
    def test_seasons_dont_leak_across(self):
        """Season 2 week 1 features should NOT reflect season 1 data."""
        s1 = _make_games(season=2022, n_weeks=4, fantasy_points=50.0, carries=30)
        s2 = _make_games(season=2023, n_weeks=4, fantasy_points=5.0, carries=3)
        df = build_features(pd.concat([s1, s2], ignore_index=True))

        w1_s2 = df[(df["season"] == 2023) & (df["week"] == 1)]
        rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
        for col in rolling_cols:
            val = w1_s2[col].iloc[0]
            assert val == 0.0 or np.isnan(val), (
                f"{col} = {val} for season 2023 week 1 (season 2022 data leaked)"
            )

    def test_prior_season_features_alignment(self):
        """prior_season_* features for season S should come from season S-1."""
        s1 = _make_games(season=2022, n_weeks=4, fantasy_points=50.0)
        s2 = _make_games(season=2023, n_weeks=4, fantasy_points=5.0)
        df = build_features(pd.concat([s1, s2], ignore_index=True))

        prior_cols = [c for c in df.columns if c.startswith("prior_season_")]
        if not prior_cols:
            pytest.skip("No prior_season columns generated")

        # Season 2023 rows should have prior_season stats reflecting 2022 (~50 pts)
        s2_row = df[df["season"] == 2023].iloc[0]
        prior_mean_col = [c for c in prior_cols if "mean" in c and "fantasy_points" in c]
        if prior_mean_col:
            val = s2_row[prior_mean_col[0]]
            assert val > 20, f"Prior season mean = {val}, expected ~50 from 2022 season"

        # Season 2022 rows should have NaN prior_season stats (no 2021 data)
        s1_row = df[df["season"] == 2022].iloc[0]
        if prior_mean_col:
            val = s1_row[prior_mean_col[0]]
            assert np.isnan(val), f"Prior season for 2022 = {val}, expected NaN (no 2021 data)"


# ---------------------------------------------------------------------------
# NaN imputation leakage
# ---------------------------------------------------------------------------


class TestFillNansSafe:
    def test_uses_only_train_stats(self):
        """fill_nans_safe must impute using training set means, not val/test."""
        feature_cols = ["feat_a", "feat_b"]
        train = pd.DataFrame(
            {
                "player_id": ["P1", "P1", "P2"],
                "position": ["RB", "RB", "RB"],
                "feat_a": [10.0, 20.0, 30.0],  # mean = 20.0
                "feat_b": [2.0, 4.0, 6.0],  # mean = 4.0
            }
        )
        val = pd.DataFrame(
            {
                "player_id": ["P3"],
                "position": ["RB"],
                "feat_a": [np.nan],
                "feat_b": [np.nan],
            }
        )
        test = pd.DataFrame(
            {
                "player_id": ["P4"],
                "position": ["RB"],
                "feat_a": [np.nan],
                "feat_b": [np.nan],
            }
        )
        train_out, val_out, test_out = fill_nans_safe(train, val, test, feature_cols)

        # Val/test NaNs should be filled with position-level training means
        assert val_out["feat_a"].iloc[0] == pytest.approx(20.0)
        assert val_out["feat_b"].iloc[0] == pytest.approx(4.0)
        assert test_out["feat_a"].iloc[0] == pytest.approx(20.0)
        assert test_out["feat_b"].iloc[0] == pytest.approx(4.0)

    def test_val_test_stats_dont_influence_imputation(self):
        """Even if val/test have extreme values, imputation must come from train only."""
        feature_cols = ["feat"]
        train = pd.DataFrame(
            {
                "player_id": ["P1", "P2"],
                "position": ["RB", "RB"],
                "feat": [10.0, 20.0],  # mean = 15.0
            }
        )
        val = pd.DataFrame(
            {
                "player_id": ["P3"],
                "position": ["RB"],
                "feat": [np.nan],
            }
        )
        test = pd.DataFrame(
            {
                "player_id": ["P4"],
                "position": ["RB"],
                "feat": [np.nan],
            }
        )
        _, val_out, test_out = fill_nans_safe(train, val, test, feature_cols)

        # Should be 15.0 (train mean), not influenced by val/test
        assert val_out["feat"].iloc[0] == pytest.approx(15.0)
        assert test_out["feat"].iloc[0] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Feature column contract (P2.1)
# ---------------------------------------------------------------------------
#
# `build_features` is the single source of truth for the model-input feature
# set. A silent column rename, deletion, or dtype change would cause every
# downstream model (ridge, NN, LGBM) to either error or — worse — silently
# fill zeros and degrade quality. This class pins the contract: stable count,
# full coverage of documented columns, numeric dtypes, bounded NaN rate.

# Weather/Vegas columns are added by shared.weather_features.merge_schedule_features,
# not by build_features. Excluded from build_features()-only assertions.
_WEATHER_VEGAS_FEATURES = frozenset(
    {
        "implied_opp_total",
        "total_line",
        "is_dome",
        "is_grass",
        "temp_adjusted",
        "wind_adjusted",
        "is_divisional",
        "days_rest_improved",
        "rest_advantage",
        "implied_total_x_wind",
    }
)


def _build_features_owned_cols() -> list[str]:
    """Documented columns that build_features (not the schedule merge) owns."""
    return [c for c in get_feature_columns() if c not in _WEATHER_VEGAS_FEATURES]


@pytest.fixture(scope="module")
def contract_features_df():
    """Run build_features once per module — tests below are read-only.

    Two RBs across two seasons (2022 + 2023), 8 weeks each. Two seasons
    are required so prior_season_* features have real values; two players
    on opposing teams ensure matchup features populate.
    """

    def _two_seasons(player_id, team, opponent, pts, targets, carries, ry, recy):
        return pd.concat(
            [
                _make_games(
                    player_id,
                    season=2022,
                    n_weeks=8,
                    position="RB",
                    recent_team=team,
                    opponent_team=opponent,
                    fantasy_points=pts,
                    targets=targets,
                    carries=carries,
                    rushing_yards=ry,
                    receiving_yards=recy,
                ),
                _make_games(
                    player_id,
                    season=2023,
                    n_weeks=8,
                    position="RB",
                    recent_team=team,
                    opponent_team=opponent,
                    fantasy_points=pts,
                    targets=targets,
                    carries=carries,
                    rushing_yards=ry,
                    receiving_yards=recy,
                ),
            ],
            ignore_index=True,
        )

    df = pd.concat(
        [
            _two_seasons("P1", "KC", "SF", 15.0, 6, 14, 70, 40),
            _two_seasons("P2", "SF", "KC", 10.0, 4, 10, 50, 30),
        ],
        ignore_index=True,
    )
    with patch("src.features.engineer.pd.read_parquet", return_value=_fake_schedules()):
        return build_features(df)


@pytest.mark.unit
class TestFeatureColumnContract:
    """Pin the build_features() output schema — catches silent regressions."""

    EXPECTED_FEATURE_COUNT = 179

    def test_get_feature_columns_count_is_stable(self):
        """If this fails, update EXPECTED_FEATURE_COUNT intentionally."""
        cols = get_feature_columns()
        assert len(cols) == self.EXPECTED_FEATURE_COUNT, (
            f"get_feature_columns() returned {len(cols)} columns, expected "
            f"{self.EXPECTED_FEATURE_COUNT}. If this change is intentional, "
            "bump EXPECTED_FEATURE_COUNT in this test and update any "
            "downstream column-count consumers."
        )

    def test_get_feature_columns_has_no_duplicates(self):
        cols = get_feature_columns()
        dupes = [c for c in set(cols) if cols.count(c) > 1]
        assert not dupes, f"Duplicate feature names in get_feature_columns(): {dupes}"

    def test_build_features_produces_all_core_columns(self, contract_features_df):
        """Every column build_features owns must be present in the output.

        Weather/Vegas columns are added by a separate merge step
        (shared.weather_features.merge_schedule_features) and are not part of
        the build_features contract.
        """
        missing = [c for c in _build_features_owned_cols() if c not in contract_features_df.columns]
        assert not missing, (
            f"build_features() dropped {len(missing)} documented columns: {missing[:10]}"
        )

    def test_engineered_features_are_numeric(self, contract_features_df):
        """All build_features outputs must be numeric (float/int), not object."""
        present = [c for c in _build_features_owned_cols() if c in contract_features_df.columns]
        non_numeric = [
            c for c in present if not pd.api.types.is_numeric_dtype(contract_features_df[c])
        ]
        assert not non_numeric, (
            f"Non-numeric feature columns (will break NN/Ridge training): "
            f"{[(c, str(contract_features_df[c].dtype)) for c in non_numeric]}"
        )

    def test_per_feature_nan_rate_ceiling_post_warmup(self, contract_features_df):
        """After the warmup window, no feature should be majority-NaN.

        Rolling / EWMA / trend features are NaN for the first few weeks of
        season 1 (no prior data). Restrict to season 2, week >= 5 — a
        steady-state window where every feature should be populated from
        player history and season-1 priors.

        Catches bugs where a groupby / merge produces all-NaN for an entire
        feature column (a whole-population dropout the NN can't recover from).
        """
        steady = contract_features_df[
            (contract_features_df["season"] == 2023) & (contract_features_df["week"] >= 5)
        ]
        assert len(steady) > 0, "Fixture change broke the steady-state sample"

        present = [c for c in _build_features_owned_cols() if c in contract_features_df.columns]
        nan_rates = steady[present].isna().mean()
        # 15% ceiling per-feature is generous enough for sparse features
        # (e.g. QB receiving stats) but tight enough to catch columns that
        # accidentally drop out entirely.
        over_ceiling = nan_rates[nan_rates > 0.15]
        assert over_ceiling.empty, (
            f"{len(over_ceiling)} features exceed 15% NaN in steady state "
            f"(post-warmup), suggesting silent column dropout:\n"
            f"{over_ceiling.sort_values(ascending=False).to_string()}"
        )

    def test_no_inf_values_in_engineered_features(self, contract_features_df):
        """Inf/-Inf would silently poison the StandardScaler. Division-by-zero
        guards in build_features must catch every case — this pins that."""
        present = [c for c in _build_features_owned_cols() if c in contract_features_df.columns]
        inf_cols = [
            c
            for c in present
            if np.isinf(contract_features_df[c].to_numpy(dtype=float, na_value=0.0)).any()
        ]
        assert not inf_cols, f"Features contain +/-Inf (will break StandardScaler): {inf_cols}"
