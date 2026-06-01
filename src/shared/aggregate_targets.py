"""Aggregate per-target predictions into fantasy points.

After the raw-stat target migration, each position's model predicts raw NFL stats
(yards, TDs, receptions, INTs, fumbles). This module is the single source of truth
for converting those predictions to fantasy points under any scoring format.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from src.config import SCORING_HALF_PPR, SCORING_PPR, SCORING_STANDARD
from src.dst.targets import _PTS_ALLOWED_TIERS, _YDS_ALLOWED_TIERS
from src.shared.position import Position

_SCORING_BY_FORMAT = {
    "standard": SCORING_STANDARD,
    "half_ppr": SCORING_HALF_PPR,
    "ppr": SCORING_PPR,
}

# Map each position's raw-stat target names to the scoring-dict key.
# K and DST use separate aggregation paths and are NOT in this map:
# - K is a sign-vector sum (see ``_k_predictions_to_fantasy_points``).
# - DST is linear stats + tier-mapped PA/YA bonuses
#   (see ``_dst_predictions_to_fantasy_points``).
# Their target sets are tracked in ``K_TARGETS`` / ``DST_TARGETS`` below so
# ``infer_position`` can route them.
POSITION_TARGET_MAP = {
    Position.QB.value: {
        "passing_yards": "passing_yards",
        "rushing_yards": "rushing_yards",
        "passing_tds": "passing_tds",
        "rushing_tds": "rushing_tds",
        "interceptions": "interceptions",
        "fumbles_lost": "fumbles_lost",
    },
    Position.RB.value: {
        "rushing_tds": "rushing_tds",
        "receiving_tds": "receiving_tds",
        "rushing_yards": "rushing_yards",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
    Position.WR.value: {
        "receiving_tds": "receiving_tds",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
    Position.TE.value: {
        "receiving_tds": "receiving_tds",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
}

# Import-time guard: every key in POSITION_TARGET_MAP must be a valid Position.
# Catches typos in this map (and in any future position additions) before any
# downstream consumer hits a KeyError mid-inference.
for _pos_key in POSITION_TARGET_MAP:
    Position(_pos_key)  # raises ValueError if _pos_key isn't a Position value
del _pos_key

# K and DST target sets — used by ``infer_position`` to route them to their
# bespoke aggregators instead of the standard linear ``target * weight`` path.
# Must stay in sync with ``src/k/config.py::TARGETS`` and
# ``src/dst/config.py::TARGETS`` (covered by parity tests in
# tests/test_aggregate_targets.py).
K_TARGETS: tuple[str, ...] = ("fg_yard_points", "pat_points", "fg_misses", "xp_misses")
DST_TARGETS: tuple[str, ...] = (
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
    "points_allowed",
    "yards_allowed",
)


# Display units for per-target MAE reporting.
TARGET_UNITS = {
    "passing_yards": "yds",
    "rushing_yards": "yds",
    "receiving_yards": "yds",
    "passing_tds": "TDs",
    "rushing_tds": "TDs",
    "receiving_tds": "TDs",
    "receptions": "rec",
    "interceptions": "INT",
    "fumbles_lost": "fum",
    # DST raw-stat units
    "def_sacks": "sacks",
    "def_ints": "INT",
    "def_fumble_rec": "fum",
    "def_fumbles_forced": "FF",
    "def_safeties": "safety",
    "def_tds": "TDs",
    "def_blocked_kicks": "blk",
    "special_teams_tds": "TDs",
    "points_allowed": "pts",
    "yards_allowed": "yds",
    # Kicker raw-stat units (predictions-tab breakdown drill-down). See
    # ``K_TARGETS`` above / ``src/k/targets.py``: fg_yard_points and pat_points
    # are point values, fg_misses / xp_misses are raw miss counts.
    "fg_yard_points": "pts",
    "pat_points": "pts",
    "fg_misses": "misses",
    "xp_misses": "misses",
}


# Precomputed boundary/bonus tables for the two DST tier lookups.
# boundaries = lo of each tier after the first (used for bucketize/digitize).
# bonuses   = one bonus per tier.
def _tier_tables(tiers):
    boundaries = [lo for (lo, _, _) in tiers[1:]]
    bonuses = [bonus for (_, _, bonus) in tiers]
    return boundaries, bonuses


_PA_BOUNDARIES, _PA_BONUSES = _tier_tables(_PTS_ALLOWED_TIERS)
_YA_BOUNDARIES, _YA_BONUSES = _tier_tables(_YDS_ALLOWED_TIERS)


def _tier_bonuses(values, boundaries: list[int], bonuses: list[int]):
    """Vectorized tier lookup for DST PA/YA fantasy-point bonuses.

    Works on torch Tensors (preserving autograd-compatible dtype/device) and
    numpy arrays. Matches the scalar helpers in ``src/dst/targets.py`` —
    values fall into tiers via half-open intervals ``[boundaries[i-1],
    boundaries[i])``.

    np.digitize default ``right=False`` gives exactly this semantic. torch
    uses the opposite ``right`` convention: ``right=True`` in torch gives
    ``[boundaries[i-1], boundaries[i])`` (matching numpy's ``right=False``).
    We pass ``right=True`` to torch to get consistent bucket assignment for
    tie values (e.g. pts_allowed=35 → tier [35, 999] → bonus -4).

    Negative inputs are clamped to ``0`` before bucketize/digitize. The first
    tier (``[0, 0]`` PA / ``[0, 99]`` YA) carries the *max* bonus (+10 / +5);
    without the clamp a negative value (which is unreachable today because
    PA/YA heads are non-negative-clamped, but would be reachable under a
    future signed head) would land in ``idx=0`` and receive the max bonus,
    inverting the scoring semantics. See audit-318 (W.SHARED-PIPE finding 2).

    Note: this is piecewise-constant, so it has zero gradient w.r.t. the tier
    input — PA/YA head updates come entirely from the per-target Huber loss.
    """
    if isinstance(values, torch.Tensor):
        b = torch.tensor(boundaries, dtype=values.dtype, device=values.device)
        bns = torch.tensor(bonuses, dtype=values.dtype, device=values.device)
        clamped = torch.clamp(values.detach(), min=0)
        idx = torch.bucketize(clamped, b, right=True)
        return bns[idx]
    arr = np.clip(np.asarray(values, dtype=np.float64), 0, None)
    idx = np.digitize(arr, boundaries, right=False)
    return np.asarray(bonuses, dtype=np.float64)[idx]


def _dst_predictions_to_fantasy_points(preds_dict: dict):
    """Aggregate the 10 DST raw-stat predictions into fantasy points.

    Must match ``src.dst.targets.compute_targets``'s ``fantasy_points``
    column exactly. Used at serving time (``app.py:_combine_total``) and for
    benchmark reporting; training itself supervises only the raw-stat heads.

    Works on numpy arrays (inference in ``app.py``) and torch Tensors (in
    case a caller wants gradients through the aggregator). The return type
    mirrors the input type.
    """
    linear = (
        preds_dict["def_sacks"] * 1
        + preds_dict["def_ints"] * 2
        + preds_dict["def_fumble_rec"] * 2
        + preds_dict["def_fumbles_forced"] * 1
        + preds_dict["def_safeties"] * 2
        + preds_dict["def_tds"] * 6
        + preds_dict["special_teams_tds"] * 6
        + preds_dict["def_blocked_kicks"] * 2
    )
    pa_bonus = _tier_bonuses(preds_dict["points_allowed"], _PA_BOUNDARIES, _PA_BONUSES)
    ya_bonus = _tier_bonuses(preds_dict["yards_allowed"], _YA_BOUNDARIES, _YA_BONUSES)
    return linear + pa_bonus + ya_bonus


def _k_predictions_to_fantasy_points(preds_dict: dict):
    """Aggregate K's 4 raw-stat predictions into fantasy points.

    Sign vector ``[+1, +1, -1, -1]`` over
    ``[fg_yard_points, pat_points, fg_misses, xp_misses]`` — fg_yard_points
    and pat_points add, misses subtract. Format-invariant; kicker scoring
    doesn't change between PPR / half / standard.

    Must match ``src.k.targets.compute_targets``'s ``fantasy_points`` column
    exactly. Works on numpy arrays and torch Tensors; the return type
    mirrors the input type.
    """
    return (
        preds_dict["fg_yard_points"]
        + preds_dict["pat_points"]
        - preds_dict["fg_misses"]
        - preds_dict["xp_misses"]
    )


def predictions_to_fantasy_points(
    pos: str,
    preds_dict: dict,
    scoring_format: str = "ppr",
) -> np.ndarray:
    """Aggregate per-target predictions to fantasy points.

    Args:
        pos: Position code (QB/RB/WR/TE/K/DST).
        preds_dict: target_name -> per-sample prediction array (or tensor). A
            ``"total"`` key is ignored if present.
        scoring_format: ``"ppr"``, ``"half_ppr"``, or ``"standard"``. Ignored
            for K and DST (their formulas are scoring-format-invariant).
    """
    if pos == "DST":
        return _dst_predictions_to_fantasy_points(preds_dict)
    if pos == "K":
        return _k_predictions_to_fantasy_points(preds_dict)
    if pos not in POSITION_TARGET_MAP:
        raise ValueError(f"No target map for position: {pos}")
    if scoring_format not in _SCORING_BY_FORMAT:
        raise ValueError(f"Unknown scoring format: {scoring_format}")
    target_map = POSITION_TARGET_MAP[pos]
    scoring = _SCORING_BY_FORMAT[scoring_format]
    total = None
    for target_name, scoring_key in target_map.items():
        if target_name not in preds_dict:
            continue
        arr = np.asarray(preds_dict[target_name], dtype=np.float64)
        contribution = arr * scoring[scoring_key]
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError(f"preds_dict has no recognized targets for {pos}")
    return total


def aggregate_fn_for(pos: str, scoring_format: str = "ppr"):
    """Return a callable `aggregate_fn(preds_dict) -> np.ndarray` bound to one position."""
    return partial(predictions_to_fantasy_points, pos, scoring_format=scoring_format)


def infer_position(target_names) -> str | None:
    """Return the position whose target list fully matches ``target_names``.

    QB / RB / WR / TE are matched via ``POSITION_TARGET_MAP``; K and DST use
    bespoke aggregators and are matched against ``K_TARGETS`` / ``DST_TARGETS``.
    Returns ``None`` for unknown sets (callers fall back to a plain sum).
    """
    name_set = set(target_names)
    for pos, tmap in POSITION_TARGET_MAP.items():
        if set(tmap.keys()) == name_set:
            return pos
    if name_set == set(K_TARGETS):
        return "K"
    if name_set == set(DST_TARGETS):
        return "DST"
    return None
