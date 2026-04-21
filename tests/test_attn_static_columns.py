"""Contract tests for the attention-NN static-feature whitelist.

Locks down the per-position ``{POS}_ATTN_STATIC_FEATURES`` lists (QB, RB, WR,
TE, DST) as the single source of truth for what flows into the attention NN's
static branch. The attention branch learns its own temporal representation
from ``{POS}_ATTN_HISTORY_STATS``, so any ``rolling_`` / ``ewma_`` / ``trend_``
/ ``_share`` / specific ``_L3``/``_L5`` column appearing in the static set
would re-introduce the collinearity leak that the per-position whitelist was
added to fix.

The historical regression: ``get_attn_static_columns`` used to be a
prefix/suffix blacklist that missed ``{POS}_SPECIFIC_FEATURES`` entries
(``yards_per_carry_L3``, ``team_rb_carry_share_L3``, ``opportunity_index_L3``,
``career_carries``, ``completion_pct_L3``, ``yards_per_reception_L3`` …)
because those columns don't match the blacklist shape. The whitelist fixes
this at the config layer.
"""

from __future__ import annotations

import pytest

from DST.dst_config import DST_ATTN_STATIC_FEATURES
from DST.dst_features import get_dst_feature_columns
from K.k_config import (
    K_ALL_FEATURES,
    K_ATTN_L1_FEATURES,
    K_ATTN_STATIC_FEATURES,
    K_CONTEXTUAL_FEATURES,
)
from QB.qb_config import QB_ATTN_STATIC_CATEGORIES, QB_ATTN_STATIC_FEATURES
from QB.qb_features import get_qb_feature_columns
from RB.rb_config import RB_ATTN_STATIC_CATEGORIES, RB_ATTN_STATIC_FEATURES
from RB.rb_features import get_rb_feature_columns
from src.features.engineer import get_attn_static_columns
from TE.te_config import TE_ATTN_STATIC_CATEGORIES, TE_ATTN_STATIC_FEATURES
from TE.te_features import get_te_feature_columns
from WR.wr_config import WR_ATTN_STATIC_CATEGORIES, WR_ATTN_STATIC_FEATURES
from WR.wr_features import get_wr_feature_columns

# Columns that must NEVER appear in the attention static set — they're
# either temporal aggregates (leaking what the attention branch already
# learns from raw history) or single-lag per-game signals the attention
# branch is designed to consume via its history stats.
_FORBIDDEN_RB_SPECIFIC = {
    "yards_per_carry_L3",
    "team_rb_carry_share_L3",
    "opportunity_index_L3",
    "career_carries",
    "team_rb_carry_hhi_L3",
    "team_rb_target_share_L3",
    "receiving_epa_per_target_L3",
}
_FORBIDDEN_QB_SPECIFIC = {
    "completion_pct_L3",
    "td_rate_L3",
    "int_rate_L3",
    "sack_rate_L3",
    "passing_epa_per_dropback_L3",
}
_FORBIDDEN_WR_SPECIFIC = {
    "yards_per_reception_L3",
    "yards_per_target_L3",
    "team_wr_target_share_L3",
    "air_yards_per_target_L3",
}
_FORBIDDEN_TE_SPECIFIC = {
    "yards_per_reception_L3",
    "team_te_target_share_L3",
    "td_rate_per_target_L3",
}
_FORBIDDEN_SHARE_COLS = {
    "target_share_L3",
    "target_share_L5",
    "carry_share_L3",
    "carry_share_L5",
    "snap_pct",
    "air_yards_share",
}

# Columns that SHOULD appear in the attention static set across the skill
# positions — these are the non-temporal, pre-game signals (matchup,
# defense, contextual, weather/vegas, prior-season).
_EXPECTED_SKILL_STATIC = {
    "opp_def_rank_vs_pos",
    "is_home",
    "implied_team_total",
}

# Expected column count per position after the whitelist filter. Drift
# guard — if the count changes unexpectedly (feature added / renamed), the
# test fails loudly and you re-check on purpose instead of silently
# retraining on a different feature set.
_EXPECTED_COUNTS = {
    "QB": len(QB_ATTN_STATIC_FEATURES),
    "RB": len(RB_ATTN_STATIC_FEATURES),
    "WR": len(WR_ATTN_STATIC_FEATURES),
    "TE": len(TE_ATTN_STATIC_FEATURES),
    "DST": len(DST_ATTN_STATIC_FEATURES),
}


def _static_cols(pos: str) -> list[str]:
    """Resolve the attention static column set for ``pos`` from the config."""
    if pos == "QB":
        return get_attn_static_columns(get_qb_feature_columns(), QB_ATTN_STATIC_FEATURES)
    if pos == "RB":
        return get_attn_static_columns(get_rb_feature_columns(), RB_ATTN_STATIC_FEATURES)
    if pos == "WR":
        return get_attn_static_columns(get_wr_feature_columns(), WR_ATTN_STATIC_FEATURES)
    if pos == "TE":
        return get_attn_static_columns(get_te_feature_columns(), TE_ATTN_STATIC_FEATURES)
    if pos == "DST":
        return get_attn_static_columns(get_dst_feature_columns(), DST_ATTN_STATIC_FEATURES)
    raise ValueError(pos)


@pytest.mark.unit
class TestAttnStaticWhitelistExcludesTemporal:
    """The attention static set must not contain any temporal aggregation."""

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "DST"])
    def test_no_rolling_prefix(self, pos):
        leaks = [c for c in _static_cols(pos) if c.startswith("rolling_")]
        assert not leaks, f"{pos} attention static set contains rolling_* columns: {leaks}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "DST"])
    def test_no_ewma_prefix(self, pos):
        leaks = [c for c in _static_cols(pos) if c.startswith("ewma_")]
        assert not leaks, f"{pos} attention static set contains ewma_* columns: {leaks}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "DST"])
    def test_no_trend_prefix(self, pos):
        leaks = [c for c in _static_cols(pos) if c.startswith("trend_")]
        assert not leaks, f"{pos} attention static set contains trend_* columns: {leaks}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_no_share_columns(self, pos):
        """snap_pct / air_yards_share / *_share_L{3,5} are per-game signals
        the attention branch sees via its history stats, not pre-game state."""
        leaks = set(_static_cols(pos)) & _FORBIDDEN_SHARE_COLS
        assert not leaks, f"{pos} attention static set contains share columns: {sorted(leaks)}"


@pytest.mark.unit
class TestAttnStaticWhitelistExcludesSpecific:
    """The regression the whitelist was written to fix: specific features
    (``yards_per_carry_L3`` etc.) used to leak into the attention static set
    because their names didn't match the old prefix/suffix blacklist."""

    def test_rb_specific_leaks_fixed(self):
        leaks = set(_static_cols("RB")) & _FORBIDDEN_RB_SPECIFIC
        assert not leaks, f"RB-specific features leaked into attention static: {sorted(leaks)}"

    def test_qb_specific_leaks_fixed(self):
        leaks = set(_static_cols("QB")) & _FORBIDDEN_QB_SPECIFIC
        assert not leaks, f"QB-specific features leaked into attention static: {sorted(leaks)}"

    def test_wr_specific_leaks_fixed(self):
        leaks = set(_static_cols("WR")) & _FORBIDDEN_WR_SPECIFIC
        assert not leaks, f"WR-specific features leaked into attention static: {sorted(leaks)}"

    def test_te_specific_leaks_fixed(self):
        leaks = set(_static_cols("TE")) & _FORBIDDEN_TE_SPECIFIC
        assert not leaks, f"TE-specific features leaked into attention static: {sorted(leaks)}"

    def test_dst_specific_excluded(self):
        """Every DST_SPECIFIC feature is a rolling/ewma/trend aggregate; none
        should appear in the attention static set."""
        from DST.dst_config import DST_SPECIFIC_FEATURES

        leaks = set(_static_cols("DST")) & set(DST_SPECIFIC_FEATURES)
        assert not leaks, f"DST_SPECIFIC features leaked into attention static: {sorted(leaks)}"


@pytest.mark.unit
class TestAttnStaticWhitelistIncludesStatic:
    """The attention static set must carry the non-temporal signals that are
    known at game-planning time (home/away, Vegas line, opponent defense)."""

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_matchup_defense_contextual_weather_present(self, pos):
        got = set(_static_cols(pos))
        missing = _EXPECTED_SKILL_STATIC - got
        assert not missing, f"{pos} attention static set missing expected columns: {missing}"

    def test_qb_prior_season_present(self):
        assert "prior_season_mean_fantasy_points" in _static_cols("QB")

    def test_dst_contextual_present(self):
        """DST static set keeps is_home, is_dome, spread/total, prior-season means."""
        got = set(_static_cols("DST"))
        for col in ("is_home", "is_dome", "spread_line", "total_line"):
            assert col in got, f"DST attention static missing {col}"


@pytest.mark.unit
class TestAttnStaticWhitelistShape:
    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "DST"])
    def test_column_count_matches_config(self, pos):
        """Drift guard: the returned count must match the config whitelist
        size (after intersection with the full feature list). A sudden count
        shift means a feature was added/removed without updating the config.
        """
        assert len(_static_cols(pos)) == _EXPECTED_COUNTS[pos]

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE", "DST"])
    def test_no_duplicates(self, pos):
        cols = _static_cols(pos)
        assert len(cols) == len(set(cols)), f"{pos} attention static has duplicates"

    @pytest.mark.parametrize(
        "pos,categories",
        [
            ("QB", QB_ATTN_STATIC_CATEGORIES),
            ("RB", RB_ATTN_STATIC_CATEGORIES),
            ("WR", WR_ATTN_STATIC_CATEGORIES),
            ("TE", TE_ATTN_STATIC_CATEGORIES),
        ],
    )
    def test_categories_exclude_temporal_buckets(self, pos, categories):
        """Config-level invariant: the category whitelist must not include
        ``rolling`` / ``ewma`` / ``trend`` / ``share`` / ``specific``."""
        forbidden = {"rolling", "ewma", "trend", "share", "specific"}
        bad = set(categories) & forbidden
        assert not bad, f"{pos}_ATTN_STATIC_CATEGORIES includes forbidden bucket: {sorted(bad)}"


@pytest.mark.unit
class TestGetAttnStaticColumnsFunction:
    """Contract of the pure function itself."""

    def test_preserves_input_order(self):
        all_cols = ["a", "b", "c", "d", "e"]
        whitelist = ["c", "a", "e"]  # whitelist order is irrelevant
        got = get_attn_static_columns(all_cols, whitelist)
        assert got == ["a", "c", "e"]

    def test_ignores_whitelist_entries_missing_from_input(self):
        """Entries in the whitelist that don't appear in ``all_feature_cols``
        are silently dropped — the function filters, it doesn't assert."""
        got = get_attn_static_columns(["a", "b"], ["a", "z"])
        assert got == ["a"]

    def test_empty_whitelist_returns_empty(self):
        assert get_attn_static_columns(["a", "b", "c"], []) == []


# ---------------------------------------------------------------------------
# K — different contract. K's attention static list is the complete source of
# truth (attn_static_from_df=True), so the filter is a no-op at runtime. The
# invariants are (a) no temporal features in the static list, (b) the new L1
# engineered columns stay OUT of K_ALL_FEATURES so Ridge and the base NN
# never see them.
# ---------------------------------------------------------------------------

_K_FORBIDDEN_STATIC_COLS = {
    # L3/L5/L8 rolling features engineered by compute_k_features
    "fg_attempts_L3",
    "fg_accuracy_L5",
    "pat_volume_L3",
    "total_k_pts_L3",
    "long_fg_rate_L3",
    "k_pts_trend",
    "k_pts_std_L3",
    "avg_fg_distance_L3",
    "avg_fg_prob_L3",
    "fg_pct_40plus_L5",
    "q4_fg_rate_L5",
    "xp_accuracy_L5",
}


@pytest.mark.unit
class TestKAttentionStaticFeatures:
    def test_no_l3_l5_features_in_static_set(self):
        leaks = set(K_ATTN_STATIC_FEATURES) & _K_FORBIDDEN_STATIC_COLS
        assert not leaks, f"K_ATTN_STATIC_FEATURES contains L3/L5 rolling features: {sorted(leaks)}"

    def test_no_rolling_or_share_prefixes(self):
        leaks = [
            c
            for c in K_ATTN_STATIC_FEATURES
            if c.startswith("rolling_") or c.startswith("ewma_") or c.startswith("trend_")
        ]
        assert not leaks, f"K_ATTN_STATIC_FEATURES has temporal prefixes: {leaks}"

    def test_l1_features_excluded_from_k_all_features(self):
        """Critical: the engineered L1 columns must NOT live in K_ALL_FEATURES,
        or Ridge and the base NN would train on them — the exact leakage the
        attn_static_from_df path was designed to prevent."""
        all_features = set(K_ALL_FEATURES)
        leaks = set(K_ATTN_L1_FEATURES) & all_features
        assert not leaks, f"L1 attention features leaked into K_ALL_FEATURES: {sorted(leaks)}"

    def test_column_count_matches_config(self):
        """Drift guard: unexpected count shift => feature added/removed silently."""
        expected = len(K_ATTN_L1_FEATURES) + len(K_CONTEXTUAL_FEATURES)
        assert len(K_ATTN_STATIC_FEATURES) == expected

    def test_no_duplicates(self):
        assert len(K_ATTN_STATIC_FEATURES) == len(set(K_ATTN_STATIC_FEATURES))

    def test_contextual_features_present(self):
        got = set(K_ATTN_STATIC_FEATURES)
        for col in ("is_home", "implied_team_total", "total_line", "game_wind"):
            assert col in got, f"K_ATTN_STATIC_FEATURES missing {col}"
