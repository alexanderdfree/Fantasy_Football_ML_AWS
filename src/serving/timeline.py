"""Changelog & Timeline tab backend.

Two data sources, one payload:

- **Weekly head-to-head log** — computed live from the serving results frame
  (``core._get_data``): per-week MAE for our four models AND the two expert
  baselines, the per-week winning model, and the "edge" of that winner over
  the experts. Auto-updates on every retrain. All displayed sources use one
  regular-season player-week intersection and the canonical projected-component
  actuals, matching the Comparison tab's scoring and coverage contract.

- **Release changelog** — the committed, owner-curated
  ``src/serving/release_changelog.json`` (same committed-JSON idiom as
  ``comparison_experts.json``). One entry per notable model release:
  ``{version, date, family, model, title, summary, mae, r2, prev_mae, pr}``.
  ``family`` keys into the model hues; ``mae``/``r2`` are the release's mean
  Attention-NN (or family) benchmark metrics; ``prev_mae`` drives the
  "vs Prev" delta. Seeded from real ``benchmark_history/`` runs; the owner
  appends entries when a milestone lands.

Edge semantics (documented in the payload as ``edge_basis: "common_rows"``):
all weekly MAEs and the winning model use that same intersection. ``edge`` is
the minimum over both experts of ``mae_e − mae_m``. An absent expert or empty
intersection makes the edge unavailable. Source availability is selected for
the whole season so a missing week cannot silently change the comparison.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd

from src.serving import core
from src.serving.comparison import COMPARISON_POSITIONS, _shared_rows
from src.serving.serialization import _actual_col, _pred_col
from src.shared.comparison_scoring import (
    ACTUAL_BASIS,
    EXCLUDED_SOURCES,
    score_actual_components,
    scoring_components,
)
from src.shared.evaluation_cohorts import regular_season_rows

_TIMELINE_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
_TIMELINE_EXPERTS = ("nflcom", "rotowire")
MODEL_LABELS = {
    "ridge": "Ridge",
    "nn": "Neural Net",
    "attn_nn": "Attention NN",
    "lgbm": "LightGBM",
    "nflcom": "NFL.com",
    "rotowire": "RotoWire",
}

_RELEASE_CHANGELOG_PATH = os.path.join(os.path.dirname(__file__), "release_changelog.json")
_RELEASE_REQUIRED_KEYS = {"version", "date", "family", "model", "title", "summary", "mae"}

_releases_lock = threading.Lock()
_releases_cache: list[dict] | None = None


def load_release_changelog() -> list[dict]:
    """Committed release entries, newest first. Malformed entries are dropped
    (defensive: the file is owner-edited by hand)."""
    global _releases_cache
    if _releases_cache is not None:
        return _releases_cache
    with _releases_lock:
        if _releases_cache is not None:
            return _releases_cache
        try:
            with open(_RELEASE_CHANGELOG_PATH) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            raw = []
        entries = [
            e for e in raw if isinstance(e, dict) and _RELEASE_REQUIRED_KEYS.issubset(e.keys())
        ]
        _releases_cache = sorted(entries, key=lambda e: str(e.get("date", "")), reverse=True)
        return _releases_cache


def reset_release_cache() -> None:
    """Test hook."""
    global _releases_cache
    with _releases_lock:
        _releases_cache = None


def _weekly_source_mae(grp, actual, source, scoring):
    col = _pred_col(source, scoring)
    if col not in grp:
        return None
    err = (grp[col] - actual).abs()
    return round(float(err.mean()), 3) if err.notna().any() else None


def compute_timeline(scoring: str) -> dict:
    """The weekly head-to-head log + season summary for one scoring format."""
    results, _ = core._get_data(scoring)
    results = regular_season_rows(results)
    actual_col = _actual_col(scoring)
    results[actual_col] = np.nan
    for pos in COMPARISON_POSITIONS:
        mask = results["position"].eq(pos)
        results.loc[mask, actual_col] = score_actual_components(
            results.loc[mask], pos, scoring, prefix="actual_"
        )
        for source in EXCLUDED_SOURCES.get(pos, {}):
            results.loc[mask, _pred_col(source, scoring)] = np.nan

    sources = (*_TIMELINE_MODELS, *_TIMELINE_EXPERTS)
    # Restrict the canonical intersection helper to the sources displayed here.
    # Select availability once, before grouping weeks, as comparison_tables does
    # before slicing cohorts. Missing truth cannot establish source availability.
    source_cols = [_pred_col(source, scoring) for source in sources]
    available = results.loc[results[actual_col].notna()].reindex(
        columns=["week", actual_col, *source_cols]
    )
    common, columns = _shared_rows(available, scoring)
    if not columns:
        common = common.iloc[:0]

    weekly: list[dict] = []
    for week, grp in results.groupby("week"):
        shared = common.loc[common["week"].eq(week)]
        actual = shared[actual_col]
        actual_n = int(grp[actual_col].notna().sum())
        entry: dict = {
            "week": int(week),
            "n": int(len(shared)),
            "cohort_n": int(len(grp)),
            "actual_n": actual_n,
            "source_n": {
                source: int(
                    np.isfinite(pd.to_numeric(grp[col], errors="coerce"))
                    .where(grp[actual_col].notna(), False)
                    .sum()
                )
                for source, col in columns.items()
            },
            "status": "available" if len(shared) else "unavailable",
        }
        if shared.empty:
            entry["reason"] = (
                "shared_actual_components_missing"
                if not actual_n
                else "predictions_missing"
                if not columns
                else "no_common_source_rows"
            )
        for src in sources:
            entry[src] = (
                _weekly_source_mae(shared, actual, src, scoring) if src in columns else None
            )

        model_maes = {m: entry[m] for m in _TIMELINE_MODELS if entry[m] is not None}
        winner = min(model_maes, key=model_maes.get) if model_maes else None
        entry["winner"] = winner

        edge = None
        if winner is not None and all(entry[e] is not None for e in _TIMELINE_EXPERTS):
            mcol = _pred_col(winner, scoring)
            # The UI calls a positive edge "Beat Both Experts". One available
            # comparison cannot establish that claim when the other is missing.
            m_mae = float((shared[mcol] - actual).abs().mean())
            edge = round(
                min(
                    float((shared[_pred_col(e, scoring)] - actual).abs().mean()) - m_mae
                    for e in _TIMELINE_EXPERTS
                ),
                3,
            )
        entry["edge"] = edge
        weekly.append(entry)

    weekly.sort(key=lambda w: w["week"])

    total_weeks = len(weekly)
    win_counts: dict[str, int] = {}
    for w in weekly:
        if w["winner"]:
            win_counts[w["winner"]] = win_counts.get(w["winner"], 0) + 1
    champion = max(win_counts, key=win_counts.get) if win_counts else None

    best_week = None
    best_mae = None
    for w in weekly:
        if w["winner"] is None:
            continue
        v = w[w["winner"]]
        if v is not None and (best_mae is None or v < best_mae):
            best_mae = v
            best_week = w["week"]

    season = None
    if "season" in results.columns and len(results):
        try:
            season = int(results["season"].max())
        except (TypeError, ValueError):
            season = None

    return {
        "weekly": weekly,
        "summary": {
            "champion": champion,
            "champion_weeks": win_counts.get(champion, 0) if champion else 0,
            "total_weeks": total_weeks,
            "best_week": best_week,
            "best_mae": best_mae,
            "beat_experts": sum(1 for w in weekly if w["edge"] is not None and w["edge"] > 0),
        },
        "season": season,
        "model_labels": MODEL_LABELS,
        "edge_basis": "common_rows",
        "actual_basis": ACTUAL_BASIS,
        "scoring_components": {pos: list(scoring_components(pos)) for pos in COMPARISON_POSITIONS},
        "excluded_sources": EXCLUDED_SOURCES,
        "sources": list(columns),
    }
