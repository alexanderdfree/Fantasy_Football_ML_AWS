"""Comparison tab: our live model vs static expert projection sources.

Pure functions over the cached per-row predictions ``DataFrame`` (passed in by
the ``/api/comparison`` route) plus the committed expert-summary / interval JSON
files. No serving-cache or model-load state — extracted from ``app.py`` during the
serving decomposition. The route calls these via ``comparison.<fn>``.
"""

import json
import os
import traceback

import numpy as np
import pandas as pd

from src.serving.serialization import (
    _MODEL_PRED_PREFIXES,
    _ROW_PRED_PREFIXES,
    _actual_col,
    _pred_col,
)
from src.shared.evaluation import compute_metrics

# ---------------------------------------------------------------------------
# Comparison tab: our model vs expert projection sources
# ---------------------------------------------------------------------------
#
# The expert (NFL.com / RotoWire) numbers are static — generated offline by
# ``src.analysis.build_comparison_summary`` and committed beside this file. Our
# model's column is computed LIVE from the loaded models (same metrics path as
# the Model Performance tab), so it auto-updates on every retrain. The committed
# JSON also carries the top-30-per-position ``player_id`` sets so the live model
# column is sliced on the *same* players the experts were scored on.
_COMPARISON_EXPERTS_PATH = os.path.join(os.path.dirname(__file__), "comparison_experts.json")
# Per-projection prediction intervals (80% floor–ceiling bands) for the expert
# sources, generated offline by ``src.analysis.expert_intervals`` and committed
# beside this file. Same committed-JSON + live-endpoint pattern as the expert
# summary above; surfaced under the Comparison tab's "Prediction intervals" block.
_EXPERT_INTERVALS_PATH = os.path.join(os.path.dirname(__file__), "expert_intervals.json")


def _load_comparison_experts():
    """Read the committed expert-summary JSON. Returns ``None`` on any read error."""
    try:
        with open(_COMPARISON_EXPERTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        traceback.print_exc()
        return None


def _load_expert_intervals():
    """Read the committed expert-intervals JSON. Returns ``None`` on any read error.

    Optional: a missing/unreadable file degrades to ``None`` so the Comparison tab
    still renders its accuracy tables without the intervals block.

    RETAINED, NOT CURRENTLY PUBLISHED: the prediction-intervals block was removed
    from the Comparison tab (the route no longer emits ``intervals``). The fitter
    (``src.analysis.expert_intervals``), the committed JSON, and this loader are
    kept so the methodology stays documented and the block can be re-enabled.
    """
    try:
        with open(_EXPERT_INTERVALS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        traceback.print_exc()
        return None


def _position_actuals(results, scoring, pos, id_filter=None):
    """Sliced ``(sub_df, actual_array)`` for ``pos`` on this scoring, or ``None``.

    Shared row-selection for the per-model block/reliability helpers below.
    ``id_filter`` (a set of ``player_id`` strings) restricts to a subset (e.g.
    top-30); ``None`` = all rows. Returns ``None`` when results are missing, the
    actual column is absent, or the slice is empty.
    """
    if results is None or "position" not in results.columns:
        return None
    actual_col = _actual_col(scoring)
    if actual_col not in results.columns:
        return None
    sub = results[results["position"] == pos]
    if id_filter is not None:
        sub = sub[sub["player_id"].astype(str).isin(id_filter)]
    if sub.empty:
        return None
    return sub, sub[actual_col].to_numpy()


def _model_residuals(sub, actual, prefix, scoring):
    """``(actual_masked, pred_masked)`` for one model ``prefix``, or ``None``.

    Masks to rows where this model has a prediction; ``None`` when the prediction
    column is absent (model not trained for this position) or every value is NaN.
    """
    pred_col = _pred_col(prefix, scoring)
    if pred_col not in sub.columns:
        return None
    pred = sub[pred_col]
    mask = pred.notna().to_numpy()
    if not mask.any():
        return None
    return actual[mask], pred.to_numpy()[mask]


def _per_model_from_results(results, scoring, pos, id_filter, stat_fn):
    """Run ``stat_fn(actual_masked, pred_masked)`` per model prefix for one position.

    Shared scaffold for the block/reliability helpers below: slice the position
    rows (``id_filter`` restricts to a subset, ``None`` = all), then for each model
    prefix mask to rows where it has a prediction and call ``stat_fn``. Returns a
    dict keyed by every model prefix, each value ``stat_fn``'s output or ``None``
    (model untrained for this position / no predictions / empty slice).
    """
    out = {prefix: None for prefix in _MODEL_PRED_PREFIXES}
    sliced = _position_actuals(results, scoring, pos, id_filter)
    if sliced is None:
        return out
    sub, actual = sliced
    for prefix in _MODEL_PRED_PREFIXES:
        arr = _model_residuals(sub, actual, prefix, scoring)
        if arr is None:
            continue
        out[prefix] = stat_fn(*arr)
    return out


def _model_blocks_from_results(results, scoring, pos, id_filter=None):
    """Per-model ``{mae,rmse,r2,n}`` for one position from the cached predictions.

    Returns a dict keyed by every model prefix (``ridge``/``nn``/``attn_nn``/``lgbm``),
    each value a metrics block or ``None`` (model not trained for this position / no
    predictions / empty slice). ``id_filter`` (a set of ``player_id`` strings) restricts
    to the top-30 subset; ``None`` = all rows. The frontend renders one column per model
    and highlights the best cell, so no architecture is preselected here.
    """

    def _block(a, p):
        m = compute_metrics(a, p)
        r2 = m["r2"]
        r2 = (
            None
            if (r2 is None or (isinstance(r2, float) and np.isnan(r2)))
            else round(float(r2), 4)
        )
        return {
            "mae": round(float(m["mae"]), 4),
            "rmse": round(float(m["rmse"]), 4),
            "r2": r2,
            "n": int(len(a)),
        }

    return _per_model_from_results(results, scoring, pos, id_filter, _block)


def _model_reliabilities_from_results(results, scoring, pos):
    """Per-model residual σ + bias for one position on the held-out 2025 test rows.

    Returns a dict keyed by every model prefix, each value ``{n,mae,bias,sigma}`` or
    ``None``. Computed from the same cached per-row predictions as
    ``_model_blocks_from_results`` (so it auto-updates on every retrain). Reports
    ``sigma = std(pred − actual, ddof=1)`` and ``bias = mean(pred − actual)`` — the
    model-side counterpart to the experts' ``expert_reliability``. Residual convention
    matches ``src.analysis.expert_uncertainty``: **bias > 0 ⇒ over-predicts**. The model
    is leakage-free only on its test split, so this is 2025-only by design (the whole
    Comparison tab is the 2025 test season).

    RETAINED, NOT CURRENTLY PUBLISHED: the source-reliability block was removed from
    the Comparison tab (the route no longer emits ``model_reliability``). This live
    helper is kept alongside the static ``expert_reliability`` JSON so the σ/bias
    methodology stays documented and the block can be re-enabled.
    """

    def _rel(a, p):
        resid = p.astype(float) - a.astype(float)
        n = int(len(resid))
        return {
            "n": n,
            "mae": round(float(np.mean(np.abs(resid))), 4),
            "bias": round(float(np.mean(resid)), 4),
            "sigma": round(float(np.std(resid, ddof=1)) if n > 1 else 0.0, 4),
        }

    return _per_model_from_results(results, scoring, pos, None, _rel)


# Quartile labels, lowest actual fantasy points to highest. Q4 is the "boom" tier.
_QUARTILE_LABELS = ("Q1", "Q2", "Q3", "Q4")


def _quartile_bias_from_results(results, scoring, pos, n_q=4):
    """Per-source signed bias across this position's actual-FP quartiles, or ``None``.

    Bins the position's test rows into ``n_q`` quartiles by **actual** fantasy
    points — Q1 = lowest scorers … Q4 = highest / boom weeks — rank-based so tied
    actuals never collapse a bin. For every prediction source (our four models
    ``ridge``/``nn``/``attn_nn``/``lgbm`` plus the two experts ``nflcom``/``rotowire``,
    i.e. ``_ROW_PRED_PREFIXES``) it reports per-quartile ``{n, mae, bias}`` where
    ``bias = mean(pred − actual)`` — **bias > 0 ⇒ over-predicts** (same residual
    convention as ``_model_reliabilities_from_results`` / ``expert_uncertainty``).

    Computed live from the same cached per-row predictions as
    ``_model_blocks_from_results`` (so it auto-updates on every retrain). The
    quartile partition is defined once by the shared actual column, so every
    source is scored on the *same* rows and is directly comparable across the
    quartile axis; a source's per-quartile ``n`` may still differ because experts
    don't project every player — that coverage gap is real and surfaced, not hidden.

    Returns ``{ "Q1": {source: {n,mae,bias}|None, ...}, ... }`` keyed by the same
    source prefixes as the accuracy tables, or ``None`` when the slice is missing or
    too small to form ``n_q`` quartiles (e.g. a thin K/DST slice).
    """
    sliced = _position_actuals(results, scoring, pos, None)
    if sliced is None:
        return None
    sub, actual = sliced
    if len(actual) < n_q:
        return None
    labels = list(_QUARTILE_LABELS[:n_q])
    try:
        # Rank-based qcut: equal-frequency bins that never collapse on tied actuals
        # (mirrors the project-wide "signed bias by quartile" convention, e.g.
        # src/analysis/attn_weekly_accuracy.py / rmse_gap_decomposition._tiers).
        ranks = pd.Series(actual).rank(method="first")
        quartile = pd.qcut(ranks, n_q, labels=labels).to_numpy().astype(object)
    except (ValueError, TypeError):
        return None

    out = {q: {} for q in labels}
    for prefix in _ROW_PRED_PREFIXES:
        pred_col = _pred_col(prefix, scoring)
        present = pred_col in sub.columns
        pred = sub[pred_col].to_numpy() if present else None
        for q in labels:
            if not present:
                out[q][prefix] = None
                continue
            mask = (quartile == q) & ~pd.isna(pred)
            n = int(mask.sum())
            if n == 0:
                out[q][prefix] = None
                continue
            resid = pred[mask].astype(float) - actual[mask].astype(float)
            out[q][prefix] = {
                "n": n,
                "mae": round(float(np.mean(np.abs(resid))), 4),
                "bias": round(float(np.mean(resid)), 4),
            }
    return out
