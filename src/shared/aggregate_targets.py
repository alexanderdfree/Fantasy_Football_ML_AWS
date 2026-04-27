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

_SCORING_BY_FORMAT = {
    "standard": SCORING_STANDARD,
    "half_ppr": SCORING_HALF_PPR,
    "ppr": SCORING_PPR,
}

# Map each position's raw-stat target names to the scoring-dict key.
# DST uses a separate aggregation path (tier-mapped PA/YA + linear stats),
# see ``_dst_predictions_to_fantasy_points``.
POSITION_TARGET_MAP = {
    "QB": {
        "passing_yards": "passing_yards",
        "rushing_yards": "rushing_yards",
        "passing_tds": "passing_tds",
        "rushing_tds": "rushing_tds",
        "interceptions": "interceptions",
        "fumbles_lost": "fumbles_lost",
    },
    "RB": {
        "rushing_tds": "rushing_tds",
        "receiving_tds": "receiving_tds",
        "rushing_yards": "rushing_yards",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
    "WR": {
        "receiving_tds": "receiving_tds",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
    "TE": {
        "receiving_tds": "receiving_tds",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        "fumbles_lost": "fumbles_lost",
    },
}

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
}

# Targets whose MAE should be rendered in both raw units and fantasy-point-equivalent
# (MAE × |scoring weight|) for readability. Applied by the frontend/evaluation
# report layer. DST's tier-mapped PA/YA don't have a single scoring weight, so
# they're omitted here; the report layer falls back to raw units for those.
POINT_EQUIVALENT_MULTIPLIER = {
    "passing_tds": 4.0,
    "rushing_tds": 6.0,
    "receiving_tds": 6.0,
    "interceptions": 2.0,
    "fumbles_lost": 2.0,
    "receptions": 1.0,  # only in PPR
    # DST linear-scoring contributions
    "def_sacks": 1.0,
    "def_ints": 2.0,
    "def_fumble_rec": 2.0,
    "def_fumbles_forced": 1.0,
    "def_safeties": 2.0,
    "def_tds": 6.0,
    "def_blocked_kicks": 2.0,
    "special_teams_tds": 6.0,
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

    Note: this is piecewise-constant, so it has zero gradient w.r.t. the tier
    input — PA/YA head updates come entirely from the per-target Huber loss,
    not the aux-total loss. The aux-total gate only works because the linear
    portion (def_sacks*1 + ...) has a live gradient.
    """
    if isinstance(values, torch.Tensor):
        b = torch.tensor(boundaries, dtype=values.dtype, device=values.device)
        bns = torch.tensor(bonuses, dtype=values.dtype, device=values.device)
        idx = torch.bucketize(values.detach(), b, right=True)
        return bns[idx]
    arr = np.asarray(values, dtype=np.float64)
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


def predictions_to_fantasy_points(
    pos: str,
    preds_dict: dict,
    scoring_format: str = "ppr",
) -> np.ndarray:
    """Aggregate per-target predictions to fantasy points.

    Args:
        pos: Position code (QB/RB/WR/TE/DST).
        preds_dict: target_name -> per-sample prediction array (or tensor). A
            ``"total"`` key is ignored if present.
        scoring_format: ``"ppr"``, ``"half_ppr"``, or ``"standard"``.
    """
    if pos == "DST":
        return _dst_predictions_to_fantasy_points(preds_dict)
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
