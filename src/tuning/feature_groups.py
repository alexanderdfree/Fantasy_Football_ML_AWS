"""Shared feature-GROUPING + DoE main-effects for the feature-selection screens.

The single source of truth that the feature-selection family/sub-family screens
([ab_feature_screen_extended.py](ab_feature_screen_extended.py),
[ab_feature_screen_k.py](ab_feature_screen_k.py),
[ab_feature_screen_dst.py](ab_feature_screen_dst.py),
[ab_feature_subscreen.py](ab_feature_subscreen.py)) and the
[feature_selection.py](feature_selection.py) driver all import. The validated
core-8 family screen ([ab_feature_screen.py](ab_feature_screen.py)) is left
untouched (its #1172 regression tests pin its internals); this module
generalises the same pattern to: the position-defining ``specific`` family, the
flat K/DST lists, within-family sub-groups, and a **metric-agnostic** effects
estimator (MAE *and* RMSE).

Grouping is by column NAME, sourced from each position's ``PositionConfig`` —
NEVER from the runtime cfg. ``build_pipeline_config`` flattens
``include_features`` into ``get_feature_columns_fn`` and drops the
``include_features`` key, so a mutator that reads ``cfg["include_features"]``
silently no-ops (the #1172 null-result bug). The drop mutator therefore filters
the live ``get_feature_columns_fn()`` output (and ``attn_static_features``) by a
precomputed column set.

A *drop* is always declared ``expect_ridge_identical=False`` on the harness
Variant: removing real columns MUST move the deterministic Ridge fit, so a
``Δ=0`` is the "drop didn't take" smell and the harness fails it loudly.
"""

from __future__ import annotations

import importlib
import statistics
from collections.abc import Callable, Sequence

from src.tuning.ab_harness import Variant
from src.tuning.attn_knob_experiments import plackett_burman_design

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SPECIAL_POSITIONS = ("K", "DST")

# The validated core-8 (mirrors ab_feature_screen.SCREENED_FAMILIES; a test
# asserts they stay equal). ``specific`` (position-defining; dropping the whole
# block can starve the model) and ``ewma`` (usually empty) are screened only by
# the EXTENDED set, deliberately separate from the validated core screen.
CORE_FAMILIES: tuple[str, ...] = (
    "rolling",
    "prior_season",
    "trend",
    "share",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
)
EXTENDED_FAMILIES: tuple[str, ...] = (*CORE_FAMILIES, "specific", "ewma")

_PB_MAX_FACTORS = 11  # the built-in 12-run Plackett-Burman design caps here


# --------------------------------------------------------------------------- #
# Skill positions (QB/RB/WR/TE): families = include_features categories
# --------------------------------------------------------------------------- #
def skill_family_columns(
    families: Sequence[str],
    positions: Sequence[str] = SKILL_POSITIONS,
    *,
    drop_empty: bool = False,
) -> dict[str, frozenset[str]]:
    """Union of each family's columns across ``positions``' ``include_features``.

    Family membership is by column name, which is position-consistent, so the
    union intersected with a given position's live ``get_feature_columns_fn()``
    yields exactly that position's family columns (the ab_feature_screen
    pattern). ``drop_empty`` omits families whose union is empty (e.g. ``ewma``
    on positions that don't populate it) — a screen variant that drops an empty
    family is a no-op and would false-trip the Ridge sentinel.
    """
    cols: dict[str, set[str]] = {fam: set() for fam in families}
    for pos in positions:
        inc = importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG.include_features
        for fam in families:
            cols[fam].update(inc.get(fam, ()))
    out = {fam: frozenset(c) for fam, c in cols.items()}
    if drop_empty:
        out = {fam: c for fam, c in out.items() if c}
    return out


def screenable_skill_families(
    families: Sequence[str], positions: Sequence[str] = SKILL_POSITIONS
) -> list[str]:
    """Families non-empty for EVERY position in ``positions`` (order preserved).

    A family empty on even one screened position makes that position's drop
    variant a no-op → a false Ridge-sentinel trip (the drop "didn't take"). So a
    multi-position screen may only include families populated everywhere. ``ewma``
    is populated only for QB, so it drops out of any multi-skill screen and is
    screened position-locally via the sub-screen instead.
    """
    return [
        fam for fam in families if all(skill_family_columns([fam], [p]).get(fam) for p in positions)
    ]


# --------------------------------------------------------------------------- #
# K / DST: flat all_features lists, partitioned by hand into named groups.
# The partition is VALIDATED exhaustive against POSITION_CONFIG.all_features at
# call time, so a config edit that adds/removes a column fails loudly here
# instead of silently leaving a column unscreened.
# --------------------------------------------------------------------------- #
_K_GROUPS: dict[str, tuple[str, ...]] = {
    "fg_volume": ("fg_attempts_L3", "pat_volume_L3", "total_k_pts_L3"),
    "fg_accuracy": (
        "fg_accuracy_L5",
        "long_fg_rate_L3",
        "fg_pct_40plus_L5",
        "q4_fg_rate_L5",
        "xp_accuracy_L5",
        "avg_fg_prob_L3",
    ),
    "fg_distance": ("avg_fg_distance_L3",),
    "k_trend": ("k_pts_trend", "k_pts_std_L3"),
    "game_context": ("is_home", "week", "implied_team_total", "total_line"),
    "weather": ("is_dome", "game_wind", "game_temp"),
}

_DST_GROUPS: dict[str, tuple[str, ...]] = {
    "dst_production": (
        "sacks_L3",
        "sacks_L5",
        "ints_L3",
        "fumble_rec_L3",
        "forced_fumbles_L3",
        "blocked_kicks_L5",
    ),
    "dst_points": ("dst_pts_L3", "dst_pts_L5", "dst_pts_L8", "dst_pts_ewma"),
    "pts_yds_allowed": (
        "pts_allowed_L3",
        "pts_allowed_L5",
        "pts_allowed_ewma",
        "yards_allowed_L3",
        "yards_allowed_L5",
        "yards_allowed_ewma",
    ),
    "dst_trend": (
        "sack_trend",
        "turnover_trend",
        "pts_allowed_trend",
        "pts_allowed_std_L3",
        "dst_scoring_std_L3",
    ),
    "opp_offense": (
        "opp_scoring_L3",
        "opp_scoring_L5",
        "opp_turnovers_L5",
        "opp_sacks_allowed_L5",
    ),
    "opp_qb": (
        "opp_qb_epa_L5",
        "opp_qb_int_rate_L5",
        "opp_qb_sack_rate_L5",
        "opp_qb_rush_yds_L5",
    ),
    "game_context": ("is_home", "week", "spread_line", "total_line", "rest_days", "div_game"),
    "weather": ("is_dome",),
    "prior_season": ("prior_season_dst_pts_avg", "prior_season_pts_allowed_avg"),
}

_SPECIAL_GROUPS = {"K": _K_GROUPS, "DST": _DST_GROUPS}


def special_family_columns(position: str) -> dict[str, frozenset[str]]:
    """Named feature groups for a flat-list position (K/DST), validated to
    exactly partition ``POSITION_CONFIG.all_features`` (every column in exactly
    one group, no group naming a non-existent column)."""
    pos = position.upper()
    if pos not in _SPECIAL_GROUPS:
        raise ValueError(f"special_family_columns is for K/DST, not {pos!r}")
    groups = _SPECIAL_GROUPS[pos]
    all_feats = set(
        importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG.all_features
    )
    covered: set[str] = set()
    for members in groups.values():
        covered.update(members)
    missing = all_feats - covered
    extra = covered - all_feats
    if missing or extra:
        raise ValueError(
            f"{pos} group partition is not exhaustive vs all_features: "
            f"unscreened columns={sorted(missing)}, unknown group columns={sorted(extra)}. "
            "Update _K_GROUPS / _DST_GROUPS in feature_groups.py to match the config."
        )
    return {g: frozenset(m) for g, m in groups.items()}


# --------------------------------------------------------------------------- #
# Within-family sub-groups (Stage 2): zoom into one family's columns.
# --------------------------------------------------------------------------- #
def _rolling_stat_root(col: str) -> str:
    """``rolling_mean_rushing_yards_L3`` -> ``rushing_yards`` (drop agg + window)."""
    body = col[len("rolling_") :] if col.startswith("rolling_") else col
    for agg in ("mean_", "std_", "max_", "min_"):
        if body.startswith(agg):
            body = body[len(agg) :]
            break
    # strip a trailing _L<window>
    if "_L" in body:
        head, _, tail = body.rpartition("_L")
        if tail.isdigit():
            body = head
    return body


def _prior_season_stat_root(col: str) -> str:
    """``prior_season_mean_yards_per_carry`` -> ``yards_per_carry`` (drop prefix+agg)."""
    body = col[len("prior_season_") :] if col.startswith("prior_season_") else col
    for agg in ("mean_", "std_", "max_", "total_"):
        if body.startswith(agg):
            return body[len(agg) :]
    return body


def subfamily_groups(position: str, family: str) -> dict[str, frozenset[str]]:
    """Sub-groups inside one family for the Stage-2 zoom.

    ``rolling`` and ``prior_season`` are grouped by stat-root (so the screen
    asks "does this stat's history carry signal" rather than resolving each
    window/agg column individually — single columns sit below the 0.02 FP noise
    floor). Every other family is one sub-group per column (small enough for
    column-level leave-one-out).
    """
    pos = position.upper()
    if pos in SPECIAL_POSITIONS:
        cols = sorted(special_family_columns(pos).get(family, frozenset()))
    else:
        cols = sorted(skill_family_columns([family], [pos]).get(family, frozenset()))
    if not cols:
        return {}
    if family == "rolling":
        rooter: Callable[[str], str] = _rolling_stat_root
    elif family == "prior_season":
        rooter = _prior_season_stat_root
    else:
        rooter = lambda c: c  # noqa: E731 — one sub-group per column
    out: dict[str, set[str]] = {}
    for c in cols:
        out.setdefault(rooter(c), set()).add(c)
    return {g: frozenset(c) for g, c in out.items()}


# --------------------------------------------------------------------------- #
# Designs: Plackett-Burman main-effects (3..11 groups) else leave-one-out
# --------------------------------------------------------------------------- #
_PB_MIN_FACTORS = 3  # below this, PB has an all-kept row that drops nothing


def design_for_groups(group_names: Sequence[str]) -> list[tuple[str, frozenset[str]]]:
    """Map a group list to ``(variant_name, dropped_groups)`` rows (no baseline).

    3..11 groups -> the 12-run Plackett-Burman main-effects design (every group's
    effect is estimated from all 12 rows; tighter than OFAT and interactions
    don't alias onto the wrong group). Outside that range -> leave-one-out (one
    row per group, dropping only that group): exact for vs-baseline main effects,
    at N rows. LOO is also used for n<=2 because the small PB designs contain an
    all-``+1`` row that drops nothing — a baseline-identical variant that would
    false-trip the ``expect_ridge_identical=False`` sentinel.
    """
    names = list(group_names)
    n = len(names)
    if n == 0:
        return []
    if _PB_MIN_FACTORS <= n <= _PB_MAX_FACTORS:
        design = plackett_burman_design(n)
        rows: list[tuple[str, frozenset[str]]] = []
        for idx, signs in enumerate(design, start=1):
            drop = frozenset(g for g, s in zip(names, signs, strict=True) if s < 0)
            rows.append((f"pb{idx:02d}", drop))
        return rows
    return [(f"drop_{g}", frozenset({g})) for g in names]


def drop_columns_mutator(cols: frozenset[str]) -> Callable[[dict], dict]:
    """A cfg mutator that filters ``cols`` out of BOTH model paths.

    Filters ``get_feature_columns_fn`` (Ridge / LightGBM / base-NN, and K/DST
    whose callable reads ``POSITION_CONFIG.all_features``) and
    ``attn_static_features`` (the attention NN's static branch). Mirrors
    ``ab_feature_screen._drop_families`` at the column level.
    """
    frozen = frozenset(cols)

    def _mut(cfg: dict, _cols: frozenset[str] = frozen) -> dict:
        base_get = cfg.get("get_feature_columns_fn")
        if base_get is not None:
            cfg["get_feature_columns_fn"] = lambda _b=base_get: [c for c in _b() if c not in _cols]
        static = cfg.get("attn_static_features")
        if static is not None:
            cfg["attn_static_features"] = [c for c in static if c not in _cols]
        return cfg

    return _mut


def build_drop_variants(
    group_cols: dict[str, frozenset[str]],
) -> tuple[list[Variant], dict[str, frozenset[str]]]:
    """``baseline`` + one drop variant per design row.

    Returns ``(variants, row_drops)`` where ``row_drops`` maps each non-baseline
    variant name to the set of GROUP names it drops — the contrast matrix the
    :func:`main_effects` estimator reconstructs the per-group effect from.
    Empty groups are dropped first (a no-op variant would false-trip the Ridge
    sentinel).
    """
    group_cols = {g: c for g, c in group_cols.items() if c}
    names = list(group_cols)
    variants = [Variant("baseline", label="keep all groups (production)")]
    row_drops: dict[str, frozenset[str]] = {}
    for vname, dropped in design_for_groups(names):
        cols = frozenset(c for g in dropped for c in group_cols[g])
        if not cols:
            # A design row that drops no columns is baseline-identical and would
            # false-trip the expect_ridge_identical=False sentinel — skip it.
            continue
        kept = [g for g in names if g not in dropped]
        variants.append(
            Variant(
                vname,
                cfg_mutator=drop_columns_mutator(cols),
                expect_ridge_identical=False,  # a real drop MUST move Ridge (#1172)
                label=f"drop={sorted(dropped)} keep={kept}",
            )
        )
        row_drops[vname] = dropped
    return variants, row_drops


# --------------------------------------------------------------------------- #
# Metric-agnostic main effects (MAE or RMSE) + extraction from harness results
# --------------------------------------------------------------------------- #
def main_effects(
    variant_seed_value: dict[str, dict[int, float]],
    row_drops: dict[str, frozenset[str]],
    group_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per group, averaged across seeds.

    ``variant_seed_value`` maps each non-baseline variant name to its
    ``{seed: value}`` for ONE (model, metric) — extract via
    :func:`extract_variant_seed_metric`. Per seed and per group: mean(value |
    group DROPPED) - mean(value | group KEPT), then averaged across seeds. A
    POSITIVE effect = dropping the group RAISES the metric = the group carries
    signal for that model. Metric-agnostic (pass MAE or RMSE values). Mirrors
    ``attn_knob_experiments.estimate_doe_effects`` /
    ``ab_feature_screen.feature_main_effects``.
    """
    seeds = sorted({s for m in variant_seed_value.values() for s in m})
    out: dict[str, dict[str, float]] = {}
    for grp in group_names:
        per_seed: list[float] = []
        for seed in seeds:
            dropped = [
                variant_seed_value[v][seed]
                for v, drop in row_drops.items()
                if grp in drop and v in variant_seed_value and seed in variant_seed_value[v]
            ]
            kept = [
                variant_seed_value[v][seed]
                for v, drop in row_drops.items()
                if grp not in drop and v in variant_seed_value and seed in variant_seed_value[v]
            ]
            if dropped and kept:
                per_seed.append(statistics.mean(dropped) - statistics.mean(kept))
        if per_seed:
            out[grp] = {
                "mean_effect": statistics.mean(per_seed),
                "std_effect": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                "n_seeds": len(per_seed),
            }
    return out


def extract_variant_seed_metric(
    results: list[dict],
    position: str,
    model: str,
    metric: str = "mae",
) -> dict[str, dict[int, float]]:
    """Pull ``{variant: {seed: value}}`` for one (position, model, metric) from a
    harness/launch_ab per-cell results list (the shape ``collect_results``
    returns). Skips not-ok cells and missing/NaN values."""
    out: dict[str, dict[int, float]] = {}
    for r in results:
        if not r.get("ok") or r.get("position") != position:
            continue
        m = r.get("metrics", {}).get(model)
        if not m:
            continue
        val = m.get(metric)
        if val is None or val != val:  # None / NaN
            continue
        out.setdefault(r["variant"], {})[int(r["seed"])] = float(val)
    return out
