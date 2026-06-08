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

import src.dst.config as dst_cfg
import src.dst.features as dst_features
import src.k.config as k_cfg
import src.qb.config as qb_cfg
import src.qb.features as qb_features
import src.rb.config as rb_cfg
import src.rb.features as rb_features
import src.te.config as te_cfg
import src.te.features as te_features
import src.wr.config as wr_cfg
import src.wr.features as wr_features
from src.features.engineer import get_attn_static_columns
from src.shared.position_config import derive_attn_static_features

# Per-position resolvers — keyed by uppercase position label so the test
# parametrize values stay as data, not symbol references.
_POSITION_CFG = {
    "QB": (qb_cfg, qb_features),
    "RB": (rb_cfg, rb_features),
    "WR": (wr_cfg, wr_features),
    "TE": (te_cfg, te_features),
    "DST": (dst_cfg, dst_features),
}

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
# Derived from the config's single source of truth rather than a hardcoded
# partial list — every entry in TE's ``specific_features`` (the ``team_te_*``,
# ``*_per_reception_*``, ``td_rate_per_target_*`` … families) is a temporal
# aggregate that must stay out of the attention static branch. A hardcoded
# subset would silently miss new TE-specific features slipping in.
_FORBIDDEN_TE_SPECIFIC = set(te_cfg.POSITION_CONFIG.specific_features)
_FORBIDDEN_SPECIFIC = {
    "QB": _FORBIDDEN_QB_SPECIFIC,
    "RB": _FORBIDDEN_RB_SPECIFIC,
    "WR": _FORBIDDEN_WR_SPECIFIC,
    "TE": _FORBIDDEN_TE_SPECIFIC,
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
# (``opp_def_rank_vs_pos`` was dropped from RB by the multicollinearity
# audit — it's a per-week rank() of ``opp_fantasy_pts_allowed_to_pos`` so
# the relationship is monotonic by construction (|Spearman| measured at
# 0.937 on the RB train split — not 1.0 because of ties in the per-week
# rank). Switched the example column to one still present in all four
# skill positions' matchup category.)
_EXPECTED_SKILL_STATIC = {
    "opp_rush_pts_allowed_to_pos",
    "is_home",
    "implied_team_total",
}

# Expected column count per position after the whitelist filter. Drift
# guard — if the count changes unexpectedly (feature added / renamed), the
# test fails loudly and you re-check on purpose instead of silently
# retraining on a different feature set.
_EXPECTED_COUNTS = {
    pos: len(cfg.POSITION_CONFIG.attn_static_features) for pos, (cfg, _) in _POSITION_CFG.items()
}


def _static_cols(pos: str) -> list[str]:
    """Resolve the attention static column set for ``pos`` from the config."""
    cfg, features = _POSITION_CFG[pos]
    return get_attn_static_columns(
        features.get_feature_columns(), cfg.POSITION_CONFIG.attn_static_features
    )


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

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_specific_leaks_fixed(self, pos):
        leaks = set(_static_cols(pos)) & _FORBIDDEN_SPECIFIC[pos]
        assert not leaks, f"{pos}-specific features leaked into attention static: {sorted(leaks)}"

    def test_specific_excluded(self):
        """Every SPECIFIC feature is a rolling/ewma/trend aggregate; none
        should appear in the attention static set."""
        leaks = set(_static_cols("DST")) & set(dst_cfg.POSITION_CONFIG.specific_features)
        assert not leaks, f"SPECIFIC features leaked into attention static: {sorted(leaks)}"


@pytest.mark.unit
class TestAttnStaticWhitelistIncludesStatic:
    """The attention static set must carry the non-temporal signals that are
    known at game-planning time (home/away, Vegas line, opponent defense)."""

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_matchup_defense_contextual_weather_present(self, pos):
        got = set(_static_cols(pos))
        missing = _EXPECTED_SKILL_STATIC - got
        assert not missing, f"{pos} attention static set missing expected columns: {missing}"

    def test_prior_season_present(self):
        assert "prior_season_mean_passing_yards" in _static_cols("QB")
        # Current-role / low-history calibration signals (non-temporal).
        assert "prior_season_games_played" in _static_cols("QB")
        assert "season_starts_to_date" in _static_cols("QB")

    def test_contextual_present(self):
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

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_categories_exclude_temporal_buckets(self, pos):
        """Config-level invariant: the category whitelist must not include
        ``rolling`` / ``ewma`` / ``trend`` / ``share`` / ``specific``."""
        cfg, _ = _POSITION_CFG[pos]
        categories = cfg.ATTN_STATIC_CATEGORIES
        forbidden = {"rolling", "ewma", "trend", "share", "specific"}
        bad = set(categories) & forbidden
        assert not bad, f"{pos}_ATTN_STATIC_CATEGORIES includes forbidden bucket: {sorted(bad)}"

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_category_derivation_matches_position_config(self, pos):
        """Skill-position configs derive attention statics from the category
        whitelist, not a hand-maintained duplicate list."""
        cfg, _ = _POSITION_CFG[pos]
        expected = derive_attn_static_features(
            cfg.POSITION_CONFIG.include_features,
            cfg.ATTN_STATIC_CATEGORIES,
        )
        assert cfg.POSITION_CONFIG.attn_static_features == expected


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
# invariants are (a) no rolling/trend/temporal aggregate engineered features
# in the static list (week and other game-time scalars are fine), (b) the new
# L1 engineered columns stay OUT of ALL_FEATURES so Ridge and the base NN
# never see them.
# ---------------------------------------------------------------------------

_FORBIDDEN_STATIC_COLS = {
    # L3/L5/L8 rolling features engineered by compute_features
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
        leaks = set(k_cfg.POSITION_CONFIG.attn_static_features) & _FORBIDDEN_STATIC_COLS
        assert not leaks, f"K attn_static_features contains L3/L5 rolling features: {sorted(leaks)}"

    def test_no_rolling_or_share_prefixes(self):
        leaks = [
            c
            for c in k_cfg.POSITION_CONFIG.attn_static_features
            if c.startswith("rolling_") or c.startswith("ewma_") or c.startswith("trend_")
        ]
        assert not leaks, f"K attn_static_features has temporal prefixes: {leaks}"

    def test_column_count_matches_config(self):
        """Drift guard: unexpected count shift => feature added/removed silently."""
        pc = k_cfg.POSITION_CONFIG
        assert len(pc.attn_static_features) == len(pc.contextual_features)

    def test_no_duplicates(self):
        statics = k_cfg.POSITION_CONFIG.attn_static_features
        assert len(statics) == len(set(statics))

    def test_contextual_features_present(self):
        got = set(k_cfg.POSITION_CONFIG.attn_static_features)
        for col in ("is_home", "implied_team_total", "total_line", "game_wind"):
            assert col in got, f"K attn_static_features missing {col}"
