"""Aggregate per-target predictions into fantasy points.

After the raw-stat target migration, each position's model predicts raw NFL stats
(yards, TDs, receptions, INTs, fumbles). This module is the single source of truth
for converting those predictions to fantasy points under any scoring format.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from src.config import SCORING_HALF_PPR, SCORING_PPR, SCORING_STANDARD

_SCORING_BY_FORMAT = {
    "standard": SCORING_STANDARD,
    "half_ppr": SCORING_HALF_PPR,
    "ppr": SCORING_PPR,
}

# Map each position's raw-stat target names to the scoring-dict key.
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
}

# Targets whose MAE should be rendered in both raw units and fantasy-point-equivalent
# (MAE × |scoring weight|) for readability. Applied by the frontend/evaluation
# report layer.
POINT_EQUIVALENT_MULTIPLIER = {
    "passing_tds": 4.0,
    "rushing_tds": 6.0,
    "receiving_tds": 6.0,
    "interceptions": 2.0,
    "fumbles_lost": 2.0,
    "receptions": 1.0,  # only in PPR
}


def predictions_to_fantasy_points(
    pos: str,
    preds_dict: dict,
    scoring_format: str = "ppr",
) -> np.ndarray:
    """Aggregate per-target predictions to fantasy points.

    Args:
        pos: Position code (QB/RB/WR/TE).
        preds_dict: target_name -> per-sample prediction array. A ``"total"`` key
            is ignored if present.
        scoring_format: ``"ppr"``, ``"half_ppr"``, or ``"standard"``.
    """
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
