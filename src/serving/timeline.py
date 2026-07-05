"""Changelog & Timeline tab backend.

Two data sources, one payload:

- **Weekly head-to-head log** — computed live from the serving results frame
  (``core._get_data``): per-week MAE for our four models AND the two expert
  baselines, the per-week winning model, and the "edge" of that winner over
  the experts. Auto-updates on every retrain, mirrors ``/api/weekly_accuracy``'s
  NaN-exclusion semantics (a source's weekly MAE averages only rows where that
  source has a prediction).

- **Release changelog** — the committed, owner-curated
  ``src/serving/release_changelog.json`` (same committed-JSON idiom as
  ``comparison_experts.json``). One entry per notable model release:
  ``{version, date, family, model, title, summary, mae, r2, prev_mae, pr}``.
  ``family`` keys into the model hues; ``mae``/``r2`` are the release's mean
  Attention-NN (or family) benchmark metrics; ``prev_mae`` drives the
  "vs Prev" delta. Seeded from real ``benchmark_history/`` runs; the owner
  appends entries when a milestone lands.

Edge semantics (documented in the payload as ``edge_basis: "common_rows"``):
for the week's winning model m and each expert e, MAE of both are computed on
the rows where BOTH m and e have predictions; ``edge`` is the minimum over
experts of ``mae_e − mae_m`` — positive means m beat *both* experts on their
own coverage.
"""

from __future__ import annotations

import json
import os
import threading

from src.serving import core
from src.serving.serialization import _actual_col, _pred_col

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
    err = (grp[_pred_col(source, scoring)] - actual).abs()
    return round(float(err.mean()), 3) if err.notna().any() else None


def compute_timeline(scoring: str) -> dict:
    """The weekly head-to-head log + season summary for one scoring format."""
    results, _ = core._get_data(scoring)
    actual_col = _actual_col(scoring)

    weekly: list[dict] = []
    for week, grp in results.groupby("week"):
        actual = grp[actual_col]
        entry: dict = {"week": int(week), "n": int(actual.notna().sum())}
        for src in (*_TIMELINE_MODELS, *_TIMELINE_EXPERTS):
            entry[src] = _weekly_source_mae(grp, actual, src, scoring)

        model_maes = {m: entry[m] for m in _TIMELINE_MODELS if entry[m] is not None}
        winner = min(model_maes, key=model_maes.get) if model_maes else None
        entry["winner"] = winner

        edge = None
        if winner is not None:
            mcol = _pred_col(winner, scoring)
            edges = []
            for e in _TIMELINE_EXPERTS:
                ecol = _pred_col(e, scoring)
                mask = grp[mcol].notna() & grp[ecol].notna() & actual.notna()
                if mask.any():
                    sub = grp.loc[mask]
                    m_mae = float((sub[mcol] - sub[actual_col]).abs().mean())
                    e_mae = float((sub[ecol] - sub[actual_col]).abs().mean())
                    edges.append(e_mae - m_mae)
            if edges:
                edge = round(min(edges), 3)
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
    }
