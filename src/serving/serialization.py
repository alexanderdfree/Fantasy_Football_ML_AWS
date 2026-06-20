"""Pure serialization / scoring-column helpers for the serving app.

Extracted from ``app.py`` — no Flask, no ``_cache``, no model state. JSON-safe
number/string coercion, scoring-format validation, prediction/actual column
names, and the player-row record builder. ``app.py`` imports (and re-exports)
these so existing ``from src.serving.app import _safe_num`` style call sites and
the route handlers keep working unchanged.
"""

import numpy as np

_VALID_SCORING = ("ppr", "half_ppr", "standard")
_MODEL_PRED_PREFIXES = ("ridge", "nn", "attn_nn", "lgbm")
_EXPERT_PRED_PREFIXES = ("nflcom", "rotowire")
_ROW_PRED_PREFIXES = (*_MODEL_PRED_PREFIXES, *_EXPERT_PRED_PREFIXES)

# (display_name, column_prefix) pairs — the per-model metrics loop in
# core._compute_metrics_locked iterates these. (comparison.py keys its per-model
# blocks by the bare prefixes in _MODEL_PRED_PREFIXES above.)
_MODEL_PRED_COLUMNS = [
    ("Ridge Regression", "ridge"),
    ("Neural Network", "nn"),
    ("Attention NN", "attn_nn"),
    ("LightGBM", "lgbm"),
]


def _safe_num(v):
    """Convert NaN/inf to None so jsonify produces valid JSON (browsers reject NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _safe_str(v, default=""):
    """Return default for NaN/None/non-string values."""
    if v is None:
        return default
    if isinstance(v, float) and not np.isfinite(v):
        return default
    return str(v)


def _validate_scoring(arg):
    """Return arg if it is a known scoring format, else 'ppr' (default).

    Used at every endpoint that accepts ?scoring=. Silent fallback so a stale
    bookmark doesn't error; bad clients are caught by the unit tests.
    """
    if arg in _VALID_SCORING:
        return arg
    return "ppr"


def _actual_col(fmt):
    """DataFrame column for the actual fantasy-point value in this scoring format.

    PPR is canonical and stored under 'fantasy_points' (no suffix); the other
    two are populated via compute_fantasy_points in _compute_scoring_formats.
    """
    return "fantasy_points" if fmt == "ppr" else f"fantasy_points_{fmt}"


def _pred_col(prefix, fmt):
    """DataFrame column for a model's aggregated prediction in this scoring format."""
    return f"{prefix}_pred_{fmt}"


# Meta + bare-prediction + per-scoring-format prediction columns, derived from the
# canonical prefix/scoring tuples so a new model or scoring format flows through
# automatically (the projection in _records_to_player_rows keys by name, not
# position, so the order here only affects the intermediate column selection).
_PLAYER_ROW_COLS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "week",
    "fantasy_points",
    "fantasy_points_half_ppr",
    "fantasy_points_standard",
    *(f"{prefix}_pred" for prefix in _ROW_PRED_PREFIXES),
    *(_pred_col(prefix, fmt) for prefix in _ROW_PRED_PREFIXES for fmt in _VALID_SCORING),
    "headshot_url",
]


def _round_or_none(v):
    """Round a numeric to 2dp, returning None for NaN / missing / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 2)


def _records_to_player_rows(df, scoring="ppr"):
    cols = [c for c in _PLAYER_ROW_COLS if c in df.columns]
    actual_key = _actual_col(scoring)
    pred_keys = {prefix: _pred_col(prefix, scoring) for prefix in _ROW_PRED_PREFIXES}
    return [
        {
            "player_id": _safe_str(r.get("player_id")),
            "name": _safe_str(r.get("player_display_name")),
            "position": _safe_str(r.get("position")),
            "team": _safe_str(r.get("recent_team")),
            "week": int(r["week"]),
            "actual": _round_or_none(r.get(actual_key)),
            **{
                f"{prefix}_pred": _safe_num(r.get(pred_keys[prefix]))
                for prefix in _ROW_PRED_PREFIXES
            },
            "headshot": _safe_str(r.get("headshot_url", "")),
        }
        for r in df[cols].to_dict(orient="records")
    ]
