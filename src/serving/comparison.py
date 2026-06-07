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

from src.serving.serialization import _MODEL_PRED_COLUMNS, _actual_col, _pred_col
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
    """
    try:
        with open(_EXPERT_INTERVALS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        traceback.print_exc()
        return None


def _best_model_arrays(results, scoring, pos, id_filter=None):
    """Best-MAE architecture for ``pos`` from the cached per-row predictions.

    Returns ``(best_arch, actual, pred)`` masked to the rows where that model has a
    prediction, or ``None`` when no model has predictions for the slice. ``id_filter``
    (a set of ``player_id`` strings) restricts to a subset (e.g. top-30); ``None`` = all
    rows. Shared selection for ``_model_block_from_results`` /
    ``_model_reliability_from_results`` so both report the *same* architecture (mirrors
    the per-position routing in ``src.benchmarking.benchmark``). The ranking MAE is
    ``mean(|pred-actual|)`` == ``compute_metrics`` mae.
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
    actual = sub[actual_col].to_numpy()
    best = None  # (mae, name, actual_masked, pred_masked)
    for name, prefix in _MODEL_PRED_COLUMNS:
        pred_col = _pred_col(prefix, scoring)
        if pred_col not in sub.columns:
            continue
        pred = sub[pred_col]
        mask = pred.notna().to_numpy()
        if not mask.any():
            continue
        a = actual[mask]
        p = pred.to_numpy()[mask]
        mae = float(np.mean(np.abs(p.astype(float) - a.astype(float))))
        if best is None or mae < best[0]:
            best = (mae, name, a, p)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _model_block_from_results(results, scoring, pos, id_filter=None):
    """Live model {mae,rmse,r2,n,best_arch} for one position, computed from the
    cached per-row predictions. Picks the best-MAE architecture (mirrors the
    per-position routing in ``src.benchmarking.benchmark``). ``id_filter`` (a set
    of ``player_id`` strings) restricts to the top-30 subset; ``None`` = all rows.
    Returns ``None`` when no model has predictions for this slice.
    """
    picked = _best_model_arrays(results, scoring, pos, id_filter)
    if picked is None:
        return None
    name, actual, pred = picked
    m = compute_metrics(actual, pred)
    r2 = m["r2"]
    r2 = None if (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else round(float(r2), 4)
    return {
        "mae": round(float(m["mae"]), 4),
        "rmse": round(float(m["rmse"]), 4),
        "r2": r2,
        "n": int(len(actual)),
        "best_arch": name,
    }


def _model_reliability_from_results(results, scoring, pos):
    """Live model residual σ + bias for one position on the held-out 2025 test rows.

    Computed from the same cached per-row predictions as ``_model_block_from_results``
    (so it auto-updates on every retrain) and picks the same best-MAE architecture,
    then reports ``sigma = std(pred − actual, ddof=1)`` and ``bias = mean(pred − actual)``
    for it — the model-side counterpart to the experts' ``expert_reliability``. Residual
    convention matches ``src.analysis.expert_uncertainty``: **bias > 0 ⇒ over-predicts**.
    The model is leakage-free only on its test split, so this is 2025-only by design
    (the whole Comparison tab is the 2025 test season). ``None`` when no model has
    predictions for this position.
    """
    picked = _best_model_arrays(results, scoring, pos)
    if picked is None:
        return None
    name, actual, pred = picked
    a = actual.astype(float)
    p = pred.astype(float)
    resid = p - a
    n = int(len(resid))
    return {
        "n": n,
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "bias": round(float(np.mean(resid)), 4),
        "sigma": round(float(np.std(resid, ddof=1)) if n > 1 else 0.0, 4),
        "best_arch": name,
    }
