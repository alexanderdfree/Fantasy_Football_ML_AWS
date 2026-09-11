"""Retrospective weekly evaluation of cached forecasts and a release changelog.

Each comparison group has a fixed source set. All MAEs and per-model edges
use one regular-season player-week intersection and shared-component actuals.
Models retain their own records throughout the season; no outcome-selected
winner is used to construct a track record. Release entries remain historical.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd

from src.serving import core
from src.serving.serialization import _actual_col, _pred_col
from src.shared.comparison_scoring import (
    ACTUAL_BASIS,
    EXCLUDED_SOURCES,
    score_actual_components,
    scoring_components,
)
from src.shared.evaluation_cohorts import regular_season_rows
from src.shared.expert_eligibility import eligible_forecast_rows

_TIMELINE_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
TIMELINE_GROUPS = {
    "offense": {
        "label": "Offense",
        "positions": ("QB", "RB", "WR", "TE"),
        "experts": ("nflcom", "rotowire"),
        "excluded_sources": {},
    },
    "k": {
        "label": "Kickers",
        "positions": ("K",),
        "experts": ("espn",),
        "excluded_sources": {
            **EXCLUDED_SOURCES["K"],
            "rotowire": "RotoWire has no archived kicker forecasts in this data.",
        },
    },
    "dst": {
        "label": "D/ST",
        "positions": ("DST",),
        "experts": ("rotowire", "espn"),
        "excluded_sources": {"nflcom": "NFL.com has no archived D/ST forecasts in this data."},
    },
}
MODEL_LABELS = {
    "ridge": "Ridge",
    "nn": "Neural Net",
    "attn_nn": "Attention NN",
    "lgbm": "LightGBM",
    "nflcom": "NFL.com",
    "rotowire": "RotoWire",
    "espn": "ESPN",
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


def _matched_metrics(frame, scoring, sources, experts):
    """Reduce a single common row set, retaining unrounded errors for win counts."""
    actual = _actual_col(scoring)
    columns = {source: _pred_col(source, scoring) for source in sources}
    values = (
        frame.reindex(columns=[actual, *columns.values()])
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    common = values.dropna()
    actual_n = int(values[actual].notna().sum())
    source_n = {
        source: int((values[col].notna() & values[actual].notna()).sum())
        for source, col in columns.items()
    }
    maes = {
        source: float((common[col] - common[actual]).abs().mean()) if len(common) else None
        for source, col in columns.items()
    }
    edges = {
        model: min(maes[expert] - maes[model] for expert in experts) if len(common) else None
        for model in _TIMELINE_MODELS
    }
    unavailable = [source for source, n in source_n.items() if not n]
    reason = None
    if frame.empty:
        reason = "no_regular_season_rows"
    elif not actual_n:
        reason = "shared_actual_components_missing"
    elif not len(common):
        reason = "required_forecasts_missing" if unavailable else "no_common_source_rows"
    return {
        "n": int(len(common)),
        "cohort_n": int(len(frame)),
        "actual_n": actual_n,
        "source_n": source_n,
        "unavailable_sources": unavailable,
        "status": "available" if len(common) else "unavailable",
        "reason": reason,
        "mae": maes,
        "edges": edges,
    }


def compute_timeline(scoring: str, group: str = "offense", season: int | None = None) -> dict:
    """One season and compatible position/source group; never mix their records."""
    results, _ = core._get_data(scoring)
    results = regular_season_rows(results)
    seasons = sorted(int(s) for s in results.get("season", pd.Series(dtype=int)).dropna().unique())
    if season is None:
        season = seasons[-1] if seasons else None
    config = TIMELINE_GROUPS[group]
    sources = (*_TIMELINE_MODELS, *config["experts"])
    weeks = []
    if not results.empty:
        results = results.loc[results["season"].eq(season)].copy()
        weeks = sorted(int(w) for w in results["week"].dropna().unique())
        results = results.loc[results["position"].isin(config["positions"])].copy()
    actual = _actual_col(scoring)
    results[actual] = np.nan
    for position in config["positions"]:
        if results.empty:
            break
        mask = results["position"].eq(position)
        results.loc[mask, actual] = score_actual_components(
            results.loc[mask], position, scoring, prefix="actual_"
        )
        for source in sources:
            column = _pred_col(source, scoring)
            if position == "DST":
                results.loc[mask, column] = results.loc[mask].get(
                    _pred_col(source, "comparison"), np.nan
                )
            elif source in config["experts"]:
                results.loc[mask, column] = results.loc[mask].get(
                    _pred_col(f"{source}_comparison", scoring), np.nan
                )
            eligible = eligible_forecast_rows(results.loc[mask], source, position)
            results.loc[eligible.index[~eligible], column] = np.nan

    weekly = []
    wins = dict.fromkeys(_TIMELINE_MODELS, 0)
    for week in weeks:
        frame = results.loc[results["week"].eq(week)]
        metrics = _matched_metrics(frame, scoring, sources, config["experts"])
        for model, edge in metrics["edges"].items():
            wins[model] += int(edge is not None and edge > 0)
        weekly.append({"week": int(week), **metrics})
    summary = _matched_metrics(results, scoring, sources, config["experts"])
    summary["total_weeks"] = len(weekly)
    summary["evaluated_weeks"] = sum(w["status"] == "available" for w in weekly)
    summary["models"] = {
        model: {
            "mae": summary["mae"][model],
            "edge": summary["edges"][model],
            "beat_experts": wins[model],
            "evaluated_weeks": summary["evaluated_weeks"],
        }
        for model in _TIMELINE_MODELS
    }
    return {
        "schema_version": 2,
        "weekly": weekly,
        "summary": summary,
        "scoring": scoring,
        "season": season,
        "seasons": seasons,
        "group": group,
        "groups": [{"id": key, "label": value["label"]} for key, value in TIMELINE_GROUPS.items()],
        "positions": config["positions"],
        "sources": sources,
        "experts": config["experts"],
        "model_labels": MODEL_LABELS,
        "actual_basis": ACTUAL_BASIS,
        "scoring_components": {pos: scoring_components(pos) for pos in config["positions"]},
        "excluded_sources": config["excluded_sources"],
        "sample_basis": "shared_player_weeks",
        "edge_basis": "common_rows_per_model",
    }
