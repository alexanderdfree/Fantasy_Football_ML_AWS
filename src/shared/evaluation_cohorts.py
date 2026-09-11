"""Evaluation-only cohorts shared by serving, local runs, and Batch artifacts.

Reference ranks come from archived expert forecasts, never from actual outcomes
or the model being evaluated. This module does not fit models or fetch data.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.shared.comparison_scoring import (
    ACTUAL_BASIS,
    comparison_model_totals,
    score_actual_components,
    scoring_components,
)

REFERENCE_FILENAME = "weekly_evaluation_reference_v1.parquet"
REFERENCE_VERSION = "shared_components_v3"
KEYS = ["player_id", "season", "week"]
MODEL_COLUMNS = {
    "Ridge": "pred_ridge_total",
    "NN": "pred_nn_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
    "ElasticNet": "pred_enet_total",
    "TabPFN": "pred_tabpfn_total",
}


def regular_season_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain regular-season rows, including when an old cache lacks season_type."""
    out = frame.copy()
    for col in ("season_type", "game_type"):
        if col in out:
            out = out[out[col].eq("REG")]
    if "season" in out and "week" in out:
        season = pd.to_numeric(out["season"], errors="coerce")
        week = pd.to_numeric(out["week"], errors="coerce")
        out = out[week.ge(1) & week.le(np.where(season.ge(2021), 18, 17))]
    return out


def ranked_rows(frame: pd.DataFrame, score: str, groups: list[str], n: int) -> pd.DataFrame:
    """Deterministic top-N, with player ID breaking ties rather than input order."""
    return (
        frame.assign(player_id=frame["player_id"].astype(str))
        .dropna(subset=[score])
        .sort_values([*groups, score, "player_id"], ascending=[True] * len(groups) + [False, True])
        .groupby(groups, sort=False, group_keys=False)
        .head(n)
    )


def seasonal_top_mask(frame: pd.DataFrame, n: int, score: str = "fantasy_points") -> pd.Series:
    totals = frame.groupby(["season", "player_id"], as_index=False)[score].sum(min_count=1)
    top = ranked_rows(totals, score, ["season"], n)
    keys = pd.MultiIndex.from_frame(top[["season", "player_id"]])
    return pd.Series(
        pd.MultiIndex.from_frame(frame[["season", "player_id"]]).isin(keys), index=frame.index
    )


def reference_path(*, cache_dir: str | Path | None = None) -> Path:
    """Resolve this IO boundary against the explicit run's raw-data root.

    Evaluation mathematics remains independent of execution context. Only
    the optional local reference reader uses the legacy IO context bridge;
    standalone callers can supply a directory directly.
    """
    if cache_dir is None:
        from src.training.context import raw_data_dir

        cache_dir = raw_data_dir(CACHE_DIR)
    return Path(cache_dir) / REFERENCE_FILENAME


@lru_cache(maxsize=4)
def _read_reference(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    return pd.read_parquet(path)


def load_reference(*, cache_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Read a locally hydrated reference; missing data is explicit, never fetched here."""
    path = reference_path(cache_dir=cache_dir)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return _read_reference(str(path.resolve()), stat.st_mtime_ns, stat.st_size).copy()


def reference_selection(position: str, frame: pd.DataFrame, reference: pd.DataFrame | None, n: int):
    """Join ranks selected on the FULL pregame slate, before dropping missing actuals."""
    empty = pd.Series(False, index=frame.index)
    if reference is None:
        return empty, {"status": "unavailable", "reason": "reference_artifact_missing"}
    required = {*KEYS, "position", "reference_rank", "reference_version"}
    if not required.issubset(reference):
        return empty, {"status": "unavailable", "reason": "reference_schema_missing"}
    ref = regular_season_rows(reference)
    ref = ref[ref["position"].eq(position) & ref["reference_version"].eq(REFERENCE_VERSION)]
    ref = ref[ref["season"].isin(frame["season"].unique())]
    if ref.empty:
        return empty, {"status": "unavailable", "reason": "no_reference_for_evaluation_seasons"}
    if ref.duplicated(KEYS).any():
        raise ValueError("Duplicate player-weeks in evaluation reference")
    weeks = pd.MultiIndex.from_frame(frame[["season", "week"]].drop_duplicates())
    available_weeks = pd.MultiIndex.from_frame(ref[["season", "week"]].drop_duplicates())
    missing = len(weeks.difference(available_weeks))
    ref = ref[pd.MultiIndex.from_frame(ref[["season", "week"]]).isin(weeks)]
    top = ref[ref["reference_rank"].le(n)]
    identity_cols = [c for c in [*KEYS, "reference_rank", "reference_pred"] if c in ref]
    digest = hashlib.sha256(
        ref[identity_cols].sort_values(KEYS).to_json(orient="values", double_precision=15).encode()
    ).hexdigest()
    selected = pd.MultiIndex.from_frame(top[KEYS])
    mask = pd.Series(pd.MultiIndex.from_frame(frame[KEYS]).isin(selected), index=frame.index)
    return mask, {
        "status": "partial" if missing else "available",
        "reference_version": REFERENCE_VERSION,
        "reference_hash": digest,
        "reference_n": int(len(top)),
        "missing_reference_weeks": missing,
    }


def metric_block(frame: pd.DataFrame, columns: dict[str, str]) -> dict:
    """Compact JSON-safe metrics; unavailable forecasts are not zeros."""
    from src.evaluation.metrics import compute_metrics

    models = {}
    for name, col in columns.items():
        if col not in frame:
            continue
        pair = frame[["fantasy_points", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            models[name] = {"n": 0, "mae": None, "rmse": None, "bias": None}
            continue
        actual, pred = pair["fantasy_points"].to_numpy(), pair[col].to_numpy()
        metrics = compute_metrics(actual, pred)
        models[name] = {
            "n": int(len(pair)),
            "mae": round(float(metrics["mae"]), 4),
            "rmse": round(float(metrics["rmse"]), 4),
            "bias": round(float(np.mean(pred - actual)), 4),
        }
    return {"n": int(len(frame)), "models": models}


def _identity(frame: pd.DataFrame) -> str:
    cols = [c for c in [*KEYS, "fantasy_points"] if c in frame]
    records = frame[cols].sort_values(KEYS).to_json(orient="values", double_precision=15)
    return hashlib.sha256(records.encode()).hexdigest()


def weekly_ranking_metrics(frame: pd.DataFrame, columns: dict[str, str], n=24) -> dict:
    """Score each source's own selections against the same actual weekly leaders."""
    results = {}
    for name, col in columns.items():
        if col not in frame:
            continue
        values = []
        for _, week in frame.groupby(["season", "week"]):
            week = week.dropna(subset=["fantasy_points"])
            forecasts = week.dropna(subset=[col])
            if len(week) < n or len(forecasts) < n:
                continue
            actual = ranked_rows(week, "fantasy_points", ["season", "week"], n)
            predicted = ranked_rows(forecasts, col, ["season", "week"], n)
            captured = float(predicted["fantasy_points"].sum())
            values.append(
                {
                    "hit_rate": len(set(actual["player_id"]) & set(predicted["player_id"])) / n,
                    "points_captured": captured,
                    "lineup_regret": float(actual["fantasy_points"].sum()) - captured,
                }
            )
        results[name] = {
            "n_weeks": len(values),
            **{
                metric: round(float(np.mean([v[metric] for v in values])), 4) if values else None
                for metric in ("hit_rate", "points_captured", "lineup_regret")
            },
        }
    return results


def build_cohorts(
    position: str,
    frame: pd.DataFrame | None,
    *,
    prior_frames=(),
    reference=None,
    reference_dir=None,
) -> dict:
    """Produce every named top-24 result, or a reason it cannot be calculated."""
    definitions = {
        "elite_top24": "prior_season_mean_shared_component_points",
        "weekly_reference_top24": "pregame_archived_expert_reference",
        "seasonal_actual_top24": "regular_season_total_actual_fantasy_points",
        "weekly_actual_top24": "actual_weekly_leaders_ranking",
    }
    if frame is None or not {*KEYS, "fantasy_points"}.issubset(frame):
        return {
            key: {
                "status": "unavailable",
                "reason": "held_out_rows_missing",
                "n": None,
                "models": {},
                "definition": definition,
                "actual_basis": ACTUAL_BASIS,
                "scoring_components": list(scoring_components(position)),
            }
            for key, definition in definitions.items()
        }
    df = comparison_model_totals(regular_season_rows(frame), position)
    df["player_id"] = df["player_id"].astype(str)
    df["fantasy_points"] = score_actual_components(df, position)
    if not df["fantasy_points"].notna().any():
        return {
            key: {
                "status": "unavailable",
                "reason": "shared_actual_components_missing",
                "n": None,
                "models": {},
                "definition": definition,
                "actual_basis": ACTUAL_BASIS,
                "scoring_components": list(scoring_components(position)),
            }
            for key, definition in definitions.items()
        }
    df = df[df["fantasy_points"].notna()]
    columns = {name: col for name, col in MODEL_COLUMNS.items() if col in df}
    masks = {
        "week1": df["week"].eq(1),
        "seasonal_actual_top24": seasonal_top_mask(df, 24),
    }
    for name, column, predicate in (
        ("returning", "is_returning_from_absence", lambda x: x.eq(1)),
        ("questionable", "game_status", lambda x: x.lt(1)),
        ("inheritor", "inherited_opportunity", lambda x: x.gt(0)),
    ):
        if column in df:
            masks[name] = predicate(df[column])
    prior_col = "prior_season_mean_shared_component_points"
    # Rebuild prior importance from the same components. A full-fantasy prior
    # mean is not equivalent for offense, and the generic split is invalid for K/DST.
    if prior_frames:
        prior = pd.concat([regular_season_rows(f) for f in prior_frames if f is not None])
        prior["fantasy_points"] = score_actual_components(prior, position)
        if {*KEYS, "fantasy_points"}.issubset(prior):
            prior = (
                prior.drop_duplicates(KEYS)
                .groupby(["player_id", "season"])["fantasy_points"]
                .mean()
            )
            lookup = pd.MultiIndex.from_arrays([df["player_id"], df["season"] - 1])
            df[prior_col] = prior.reindex(lookup).to_numpy()
    prior_ok = (
        prior_col in df
        and np.isfinite(pd.to_numeric(df[prior_col], errors="coerce")).any()
        and (position not in {"K", "DST"} or bool(prior_frames))
    )
    if prior_ok:
        players = df.drop_duplicates(["season", "player_id"])
        top = ranked_rows(players, prior_col, ["season"], 24)
        selected = pd.MultiIndex.from_frame(top[["season", "player_id"]])
        masks["elite_top24"] = pd.Series(
            pd.MultiIndex.from_frame(df[["season", "player_id"]]).isin(selected), index=df.index
        )
    block = {}
    for name, mask in masks.items():
        sub = df[mask]
        block[name] = metric_block(sub, columns) if len(sub) else {"n": 0, "models": {}}
        if name in definitions:
            block[name].update(
                status="available", definition=definitions[name], cohort_hash=_identity(sub)
            )
    if not prior_ok:
        block["elite_top24"] = {
            "status": "unavailable",
            "reason": "prior_season_scores_missing",
            "definition": definitions["elite_top24"],
            "n": None,
            "models": {},
        }
    if reference is None:
        reference = (
            load_reference() if reference_dir is None else load_reference(cache_dir=reference_dir)
        )
    mask, meta = reference_selection(position, df, reference, 24)
    sub = df[mask]
    block["weekly_reference_top24"] = {
        **metric_block(sub, columns),
        **meta,
        "definition": definitions["weekly_reference_top24"],
        "cohort_hash": _identity(sub),
    }
    if meta["status"] == "unavailable":
        block["weekly_reference_top24"].update(n=None, models={})
    block["weekly_actual_top24"] = {
        "status": "available",
        "definition": definitions["weekly_actual_top24"],
        "n": int(len(df)),
        "cohort_hash": _identity(df),
        "models": weekly_ranking_metrics(df, columns),
    }
    for cohort in block.values():
        cohort["actual_basis"] = ACTUAL_BASIS
        cohort["scoring_components"] = list(scoring_components(position))
    return block


def merge_cohorts(*blocks: dict) -> dict:
    """Merge model-disjoint Batch reports; refuse different cohorts or truth vintages."""
    result = {}
    for block in blocks:
        for name, entry in block.items():
            if name not in result:
                result[name] = json.loads(json.dumps(entry))
                continue
            current = result[name]
            left = {k: v for k, v in current.items() if k != "models"}
            right = {k: v for k, v in entry.items() if k != "models"}
            if left != right:
                raise ValueError(f"Split cohort mismatch: {name}")
            overlap = set(current["models"]) & set(entry["models"])
            if overlap:
                raise ValueError(f"Duplicate cohort model metrics: {name}: {sorted(overlap)}")
            current["models"].update(entry["models"])
    return result
