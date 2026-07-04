"""Top-N model/expert gap diagnostic.

This read-only analysis separates two questions that are easy to conflate:

* cohort error: MAE/RMSE/R2/bias on players who actually finished in a
  season-total top-N cohort;
* selection accuracy: whether a forecaster selected/ranked the same weekly or
  season-total top-N players.

The CLI prefers saved model artifacts, validates them with
``artifact_eval.validate_reconstruction()``, and falls back to a fresh position
pipeline only when artifacts are missing or fail validation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.analysis_expert_comparison import (
    _EXPERT_PRED_COL,
    _KEY_COLS,
    ExpertSource,
    _build_experts,
)
from src.analysis.analysis_nflcom_baseline import _json_default
from src.analysis.cohort_analysis import (
    ASCENSION,
    ROOKIE_EARLY,
    ROOKIE_REST,
    label_ascension_rows,
    label_rookie_rows,
    player_min_season,
)
from src.analysis.significance import diebold_mariano_test, paired_bootstrap_metric_ci
from src.config import TEST_SEASONS
from src.shared.evaluation import compute_metrics

ACTUAL_COL = "fantasy_points"
PRED_COL = "pred_total"
DEFAULT_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
DEFAULT_TOP_NS = (12, 30)
DEFAULT_WEEKLY_NS = (12, 24, 30)
DEFAULT_OUT = Path("analysis_output/topn_expert_gap")

MODEL_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("attn_nn", "Attention NN", "pred_attn_nn_total"),
    ("nn", "NN", "pred_nn_total"),
    ("ridge", "Ridge", "pred_ridge_total"),
    ("lgbm", "LightGBM", "pred_lgbm_total"),
)

_AUTO_LOCAL_PRED_COLS = (
    _EXPERT_PRED_COL,
    PRED_COL,
    "projection",
    "projected_points",
    "fantasy_points_projection",
    "fpts",
    "points",
)

_TD_TARGETS = (
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "def_tds",
    "special_teams_tds",
)
_RESERVED_LOCAL_SOURCE_NAMES = frozenset(
    {name for name, _, _ in MODEL_SOURCES} | {"nflcom", "sleeper", "rotowire"}
)


@dataclass(frozen=True)
class SourceMeta:
    name: str
    label: str
    kind: str
    native_col: str


@dataclass(frozen=True)
class PredictionLoad:
    df: pd.DataFrame
    mode: str
    artifact_validation: dict[str, Any] | None = None
    artifact_error: str | None = None


@dataclass(frozen=True)
class LocalExpertSpec:
    """Local benchmark source declaration.

    Syntax on the CLI is ``name=path`` or ``name:pred_col=path``. Local sources
    are deliberately file-based so this diagnostic can score approved snapshots
    without baking scraper or vendor behavior into the repo.
    """

    name: str
    path: Path
    pred_col: str | None = None
    label: str | None = None


def _nan() -> float:
    return float("nan")


def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(str)
    if "season" in out.columns:
        out["season"] = out["season"].astype(int)
    if "week" in out.columns:
        out["week"] = out["week"].astype(int)
    return out


def parse_local_expert_spec(arg: str) -> LocalExpertSpec:
    """Parse ``--local-expert name=path`` or ``name:pred_col=path``."""
    if "=" not in arg:
        raise ValueError("local expert must be NAME=PATH or NAME:PRED_COL=PATH")
    lhs, raw_path = arg.split("=", 1)
    if not lhs.strip() or not raw_path.strip():
        raise ValueError("local expert must include a non-empty name and path")
    if ":" in lhs:
        name, pred_col = lhs.split(":", 1)
        pred_col = pred_col.strip() or None
    else:
        name, pred_col = lhs, None
    name = name.strip()
    if not name:
        raise ValueError("local expert name cannot be empty")
    label = name.replace("_", " ").replace("-", " ").title()
    return LocalExpertSpec(
        name=name, path=Path(raw_path).expanduser(), pred_col=pred_col, label=label
    )


def _read_projection_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _local_projection_col(df: pd.DataFrame, pred_col: str | None) -> str:
    if pred_col:
        if pred_col not in df.columns:
            raise KeyError(f"local expert projection column {pred_col!r} missing")
        return pred_col
    for col in _AUTO_LOCAL_PRED_COLS:
        if col in df.columns:
            return col
    raise KeyError(
        "local expert projection file must include one of "
        f"{list(_AUTO_LOCAL_PRED_COLS)} or pass NAME:PRED_COL=PATH"
    )


def _local_raw_td_projection_cols(df: pd.DataFrame, total_col: str) -> dict[str, str]:
    """Map raw TD projection columns in a local snapshot to expert-prefixed names."""
    mapping: dict[str, str] = {}
    for target in _TD_TARGETS:
        candidates = (
            f"pred_{target}",
            f"projected_{target}",
            f"{target}_projection",
        )
        for col in candidates:
            if col in df.columns and col != total_col:
                mapping[col] = f"expert_pred_{target}"
                break
    return mapping


def _validate_local_projection_frame(df: pd.DataFrame, spec: LocalExpertSpec) -> None:
    missing = [c for c in _KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"local expert projection file missing key columns {missing}")
    _local_projection_col(df, spec.pred_col)


def local_expert_source(spec: LocalExpertSpec) -> ExpertSource:
    """Build an ``ExpertSource`` around an approved local snapshot file."""

    def _load(seasons: Sequence[int]) -> pd.DataFrame:
        df = _read_projection_file(spec.path)
        _validate_local_projection_frame(df, spec)
        if "season" in df.columns:
            df = df[df["season"].astype(int).isin({int(s) for s in seasons})].copy()
        return df

    def _project(raw_df: pd.DataFrame, pos: str, scoring_format: str) -> pd.DataFrame:
        del scoring_format
        df = raw_df.copy()
        pos_col = "position" if "position" in df.columns else "pos" if "pos" in df.columns else None
        if pos_col is not None:
            df = df[df[pos_col].astype(str).str.upper().eq(pos.upper())].copy()
        missing = [c for c in _KEY_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"local expert projection file missing key columns {missing}")
        pred_col = _local_projection_col(df, spec.pred_col)
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        valid_pred = pred.gt(0.0) & np.isfinite(pred)
        df = df.loc[valid_pred].copy()
        if df.empty:
            return pd.DataFrame(columns=[*_KEY_COLS, _EXPERT_PRED_COL])
        df[pred_col] = pred.loc[df.index].astype(float)
        raw_td_cols = _local_raw_td_projection_cols(df, pred_col)
        cols = [*_KEY_COLS, pred_col, *raw_td_cols]
        out = df[cols].rename(columns={pred_col: _EXPERT_PRED_COL, **raw_td_cols})
        for col in raw_td_cols.values():
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    return ExpertSource(
        name=spec.name,
        label=spec.label or spec.name,
        load=_load,
        project=_project,
        note=f"Local snapshot file: {spec.path}",
    )


def build_expert_sources(
    nflcom_loader=None,
    sleeper_loader=None,
    local_experts: Sequence[LocalExpertSpec] | None = None,
) -> list[ExpertSource]:
    """Default experts plus optional approved local-file benchmark sources."""
    experts = list(_build_experts(nflcom_loader, sleeper_loader))
    local_specs = tuple(local_experts or ())
    seen = {src.name.lower() for src in experts} | set(_RESERVED_LOCAL_SOURCE_NAMES)
    for spec in local_specs:
        key = spec.name.lower()
        if key in seen:
            raise ValueError(f"local expert source name {spec.name!r} is reserved or duplicated")
        seen.add(key)
    experts.extend(local_expert_source(spec) for spec in local_specs)
    return experts


def _finite_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    needed = [ACTUAL_COL, PRED_COL]
    work = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if work.empty:
        return work
    actual = work[ACTUAL_COL].to_numpy(dtype=float)
    pred = work[PRED_COL].to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(pred)
    return work.loc[finite].copy()


def error_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    """MAE/RMSE/R2/bias/residual sigma for a source frame."""
    work = _finite_metric_frame(df)
    if work.empty:
        return {
            "n_rows": 0,
            "n_players": 0,
            "mae": _nan(),
            "rmse": _nan(),
            "r2": _nan(),
            "bias": _nan(),
            "resid_std": _nan(),
        }
    actual = work[ACTUAL_COL].to_numpy(dtype=float)
    pred = work[PRED_COL].to_numpy(dtype=float)
    residual = pred - actual
    metrics = compute_metrics(actual, pred)
    return {
        "n_rows": int(len(work)),
        "n_players": int(work["player_id"].nunique()) if "player_id" in work else 0,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]) if metrics["r2"] is not None else _nan(),
        "bias": float(residual.mean()),
        "resid_std": float(residual.std(ddof=0)),
    }


def actual_season_ranks(base_df: pd.DataFrame) -> pd.DataFrame:
    """Actual season-total rank table, one row per (season, player_id)."""
    base = _normalise_keys(base_df)
    totals = (
        base.groupby(["season", "player_id"], as_index=False)[ACTUAL_COL]
        .sum()
        .rename(columns={ACTUAL_COL: "actual_season_total"})
    )
    totals["actual_rank"] = totals.groupby("season")["actual_season_total"].rank(
        method="first", ascending=False
    )
    return totals


def _source_with_actual_ranks(source_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    ranks = actual_season_ranks(base_df)
    out = _normalise_keys(source_df).merge(ranks, on=["season", "player_id"], how="left")
    out["pred_week_rank"] = out.groupby(["season", "week"])[PRED_COL].rank(
        method="first", ascending=False
    )
    return out


def cohort_error_rows(
    position: str,
    source: SourceMeta,
    source_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    top_ns: Sequence[int] = DEFAULT_TOP_NS,
    min_season: pd.Series | None = None,
) -> list[dict[str, Any]]:
    """Top-N cohort error rows, where the cohort is actual season-total top-N."""
    ranked = _source_with_actual_ranks(source_df, base_df)
    rows: list[dict[str, Any]] = []
    for slice_family, slice_name, mask in slice_masks(position, ranked, min_season=min_season):
        sliced = ranked.loc[mask]
        for top_n in top_ns:
            sub = sliced[sliced["actual_rank"] <= top_n]
            rows.append(
                {
                    "metric_group": "cohort_error",
                    "position": position,
                    "source": source.name,
                    "source_label": source.label,
                    "source_kind": source.kind,
                    "top_n": int(top_n),
                    "slice_family": slice_family,
                    "slice_name": slice_name,
                    "season": None,
                    "week": None,
                    **error_metrics(sub),
                }
            )
    return rows


def season_selection_rows(
    position: str,
    source: SourceMeta,
    source_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    top_ns: Sequence[int] = DEFAULT_TOP_NS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Season-total top-N overlap metrics plus hit/miss/false-positive player rows."""
    source_df = _normalise_keys(source_df)
    actual = actual_season_ranks(base_df)
    pred = (
        source_df.groupby(["season", "player_id"], as_index=False)[PRED_COL]
        .sum()
        .rename(columns={PRED_COL: "pred_season_total"})
    )
    pred["pred_rank"] = pred.groupby("season")["pred_season_total"].rank(
        method="first", ascending=False
    )
    joined = actual.merge(pred, on=["season", "player_id"], how="outer")

    metric_rows: list[dict[str, Any]] = []
    miss_rows: list[dict[str, Any]] = []
    seasons = sorted(set(actual["season"].dropna().astype(int)))
    for season in seasons:
        season_actual = joined[joined["season"] == season].copy()
        for top_n in top_ns:
            actual_top = season_actual[season_actual["actual_rank"] <= top_n]
            pred_top = season_actual[season_actual["pred_rank"] <= top_n]
            actual_set = set(actual_top["player_id"])
            pred_set = set(pred_top["player_id"])
            hits = actual_set & pred_set
            precision = len(hits) / len(pred_set) if pred_set else _nan()
            recall = len(hits) / len(actual_set) if actual_set else _nan()
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else _nan()
            )
            hit_ranks = season_actual[season_actual["player_id"].isin(hits)].dropna(
                subset=["actual_rank", "pred_rank"]
            )
            rank_error = (
                float((hit_ranks["actual_rank"] - hit_ranks["pred_rank"]).abs().mean())
                if not hit_ranks.empty
                else _nan()
            )
            pred_actual_total = float(pred_top["actual_season_total"].fillna(0.0).sum())
            ideal_total = float(actual_top["actual_season_total"].fillna(0.0).sum())
            metric_rows.append(
                {
                    "metric_group": "season_selection",
                    "position": position,
                    "source": source.name,
                    "source_label": source.label,
                    "source_kind": source.kind,
                    "top_n": int(top_n),
                    "slice_family": "global",
                    "slice_name": "all",
                    "season": int(season),
                    "week": None,
                    "n_rows": int(len(season_actual)),
                    "n_players": int(season_actual["player_id"].nunique()),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "rank_error": rank_error,
                    "regret": ideal_total - pred_actual_total,
                    "hits": int(len(hits)),
                    "misses": int(len(actual_set - pred_set)),
                    "false_positives": int(len(pred_set - actual_set)),
                }
            )
            for pid in sorted(actual_set | pred_set):
                is_hit = pid in hits
                miss_type = "hit" if is_hit else "miss" if pid in actual_set else "false_positive"
                prow = season_actual[season_actual["player_id"] == pid].iloc[0]
                miss_rows.append(
                    {
                        "position": position,
                        "source": source.name,
                        "source_label": source.label,
                        "season": int(season),
                        "top_n": int(top_n),
                        "player_id": str(pid),
                        "miss_type": miss_type,
                        "actual_rank": _maybe_float(prow.get("actual_rank")),
                        "pred_rank": _maybe_float(prow.get("pred_rank")),
                        "actual_total": _maybe_float(prow.get("actual_season_total")),
                        "pred_total": _maybe_float(prow.get("pred_season_total")),
                    }
                )
    return metric_rows, miss_rows


def weekly_selection_rows(
    position: str,
    source: SourceMeta,
    source_df: pd.DataFrame,
    base_df: pd.DataFrame | None = None,
    *,
    top_ns: Sequence[int] = DEFAULT_WEEKLY_NS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Weekly precision/hit-rate/Spearman/regret rows, skipping undersized weeks."""
    source_df = _normalise_keys(source_df)
    actual_df = _normalise_keys(base_df if base_df is not None else source_df)
    metric_rows: list[dict[str, Any]] = []
    miss_rows: list[dict[str, Any]] = []
    for (season, week), base_wk in actual_df.groupby(["season", "week"], sort=True):
        base_wk = base_wk.dropna(subset=[ACTUAL_COL]).copy()
        forecast_wk = source_df[source_df["season"].eq(season) & source_df["week"].eq(week)].dropna(
            subset=[ACTUAL_COL, PRED_COL]
        )
        if base_wk.empty or forecast_wk.empty:
            continue
        for top_n in top_ns:
            if len(base_wk) < top_n or len(forecast_wk) < top_n:
                continue
            actual_top = base_wk.nlargest(top_n, ACTUAL_COL, keep="first")
            pred_top = forecast_wk.nlargest(top_n, PRED_COL, keep="first")
            actual_set = set(actual_top["player_id"])
            pred_set = set(pred_top["player_id"])
            hits = actual_set & pred_set
            precision = len(hits) / top_n
            regret = float(actual_top[ACTUAL_COL].sum() - pred_top[ACTUAL_COL].sum())
            rho = _spearman(
                forecast_wk[PRED_COL].to_numpy(dtype=float),
                forecast_wk[ACTUAL_COL].to_numpy(dtype=float),
            )
            capture = _nan()
            if top_n == 12 and len(base_wk) >= 12:
                top3 = set(base_wk.nlargest(3, ACTUAL_COL, keep="first")["player_id"])
                capture = len(top3 & pred_set) / len(top3) if top3 else _nan()
            metric_rows.append(
                {
                    "metric_group": "weekly_selection",
                    "position": position,
                    "source": source.name,
                    "source_label": source.label,
                    "source_kind": source.kind,
                    "top_n": int(top_n),
                    "slice_family": "week",
                    "slice_name": f"week_{int(week):02d}",
                    "season": int(season),
                    "week": int(week),
                    "n_rows": int(len(forecast_wk)),
                    "n_players": int(forecast_wk["player_id"].nunique()),
                    "actual_universe_rows": int(len(base_wk)),
                    "actual_universe_players": int(base_wk["player_id"].nunique()),
                    "precision": float(precision),
                    "hit_rate": float(precision),
                    "top3_in_top12_capture": float(capture),
                    "spearman": rho,
                    "regret": regret,
                    "hits": int(len(hits)),
                    "misses": int(len(actual_set - pred_set)),
                    "false_positives": int(len(pred_set - actual_set)),
                }
            )
            actual_by_player = base_wk.drop_duplicates("player_id").set_index("player_id")
            pred_by_player = forecast_wk.drop_duplicates("player_id").set_index("player_id")
            for pid in sorted(actual_set | pred_set):
                if pid in hits:
                    miss_type = "hit"
                elif pid in actual_set:
                    miss_type = "miss"
                else:
                    miss_type = "false_positive"
                actual_player = actual_by_player.loc[pid] if pid in actual_by_player.index else None
                pred_player = pred_by_player.loc[pid] if pid in pred_by_player.index else None
                miss_rows.append(
                    {
                        "position": position,
                        "source": source.name,
                        "source_label": source.label,
                        "season": int(season),
                        "week": int(week),
                        "top_n": int(top_n),
                        "player_id": str(pid),
                        "miss_type": miss_type,
                        "actual_fp": (
                            _maybe_float(actual_player.get(ACTUAL_COL))
                            if actual_player is not None
                            else None
                        ),
                        "pred_fp": (
                            _maybe_float(pred_player.get(PRED_COL))
                            if pred_player is not None
                            else None
                        ),
                    }
                )
    return metric_rows, miss_rows


def paired_uncertainty_rows(
    position: str,
    model_sources: list[tuple[SourceMeta, pd.DataFrame]],
    expert_sources: list[tuple[SourceMeta, pd.DataFrame]],
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired model-vs-expert MAE/RMSE bootstrap and DM rows."""
    rows: list[dict[str, Any]] = []
    for model_meta, model_df in model_sources:
        m = model_df[[*_KEY_COLS, ACTUAL_COL, PRED_COL]].rename(columns={PRED_COL: "model_pred"})
        for expert_meta, expert_df in expert_sources:
            e = expert_df[[*_KEY_COLS, PRED_COL]].rename(columns={PRED_COL: "expert_pred"})
            joined = m.merge(e, on=_KEY_COLS, how="inner")
            joined = joined.dropna(subset=[ACTUAL_COL, "model_pred", "expert_pred"])
            if len(joined) < 2:
                continue
            e_model = joined["model_pred"].to_numpy(dtype=float) - joined[ACTUAL_COL].to_numpy(
                dtype=float
            )
            e_expert = joined["expert_pred"].to_numpy(dtype=float) - joined[ACTUAL_COL].to_numpy(
                dtype=float
            )
            groups = joined["week"].to_numpy()
            for metric in ("mae", "rmse"):
                boot = paired_bootstrap_metric_ci(
                    e_model, e_expert, metric=metric, groups=groups, n_boot=n_boot, seed=seed
                )
                dm = diebold_mariano_test(e_model, e_expert, power=1 if metric == "mae" else 2)
                rows.append(
                    {
                        "metric_group": "paired_uncertainty",
                        "position": position,
                        "source": model_meta.name,
                        "source_label": model_meta.label,
                        "source_kind": model_meta.kind,
                        "expert": expert_meta.name,
                        "expert_label": expert_meta.label,
                        "top_n": None,
                        "slice_family": "global",
                        "slice_name": "all",
                        "season": None,
                        "week": None,
                        "n_rows": int(len(joined)),
                        "n_players": int(joined["player_id"].nunique()),
                        "metric": metric,
                        "delta_model_minus_expert": float(boot["delta"]),
                        "ci_lo": float(boot["lo"]),
                        "ci_hi": float(boot["hi"]),
                        "p_value": float(boot["p_value"]),
                        "dm_stat": float(dm["dm_stat"]),
                        "dm_p_value": float(dm["p_value"]),
                        "favored": dm["favored"],
                    }
                )
    return rows


def td_calibration_rows(
    position: str,
    source: SourceMeta,
    source_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Sparse TD-head calibration rows for sources with raw TD predictions.

    Model pipeline outputs carry per-target prediction columns such as
    ``pred_attn_nn_receiving_tds``. Expert point-only sources usually do not, so
    they simply produce no rows here.
    """
    rows: list[dict[str, Any]] = []
    prefix = _prediction_prefix(source.native_col)
    if not prefix:
        return rows
    for target in _TD_TARGETS:
        pred_col = f"{prefix}{target}"
        if target not in source_df.columns or pred_col not in source_df.columns:
            continue
        work = source_df[[target, pred_col]].dropna().copy()
        if work.empty:
            continue
        actual = work[target].to_numpy(dtype=float)
        pred = np.clip(work[pred_col].to_numpy(dtype=float), 0.0, None)
        finite = np.isfinite(actual) & np.isfinite(pred)
        if not finite.any():
            continue
        actual = actual[finite]
        pred = pred[finite]
        positive = actual > 0.0
        prob_positive = np.clip(1.0 - np.exp(-pred), 0.0, 1.0)
        err = pred - actual
        rows.append(
            {
                "metric_group": "td_calibration",
                "position": position,
                "source": source.name,
                "source_label": source.label,
                "source_kind": source.kind,
                "target": target,
                "top_n": None,
                "slice_family": "target",
                "slice_name": target,
                "season": None,
                "week": None,
                "n_rows": int(len(actual)),
                "n_players": (
                    int(source_df.loc[work.index[finite], "player_id"].nunique())
                    if "player_id" in source_df.columns
                    else 0
                ),
                "actual_positive_rate": float(positive.mean()),
                "pred_positive_rate": float(prob_positive.mean()),
                "actual_mean": float(actual.mean()),
                "pred_mean": float(pred.mean()),
                "mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(err.mean()),
                "brier_td_positive": float(np.mean((prob_positive - positive.astype(float)) ** 2)),
                "auc_td_positive": _binary_auc(prob_positive, positive),
            }
        )
    return rows


def slice_masks(
    position: str,
    ranked_df: pd.DataFrame,
    *,
    min_season: pd.Series | None = None,
) -> list[tuple[str, str, pd.Series]]:
    """Diagnostic row-slice masks. Empty masks are omitted."""
    masks: list[tuple[str, str, pd.Series]] = [
        ("global", "all", pd.Series(True, index=ranked_df.index))
    ]
    if ranked_df.empty:
        return masks

    for week in sorted(ranked_df["week"].dropna().unique()):
        mask = ranked_df["week"].eq(week)
        if mask.any():
            masks.append(("week", f"week_{int(week):02d}", mask))

    if ranked_df[ACTUAL_COL].nunique(dropna=True) >= 4:
        try:
            q = pd.qcut(ranked_df[ACTUAL_COL], 4, labels=["q1", "q2", "q3", "q4"])
            for label in ("q1", "q2", "q3", "q4"):
                mask = q.astype(object).eq(label)
                if mask.any():
                    masks.append(("actual_fp_quartile", f"actual_{label}", mask))
        except ValueError:
            pass

    if "pred_week_rank" in ranked_df:
        buckets = [
            (ranked_df["pred_week_rank"].between(1, 12), "proj_1_12"),
            (ranked_df["pred_week_rank"].between(13, 24), "proj_13_24"),
            (ranked_df["pred_week_rank"].between(25, 30), "proj_25_30"),
            (ranked_df["pred_week_rank"] > 30, "proj_outside_30"),
        ]
        for mask, name in buckets:
            if mask.any():
                masks.append(("projected_rank_bucket", name, mask))

    masks.extend(context_slice_masks(ranked_df))

    rank = ranked_df.get("actual_rank")
    if rank is not None:
        elite = rank <= 24
        if elite.any():
            masks.append(("elite_top24", "elite_top24", elite))
        if position == "RB":
            for lo, hi, name in ((1, 12, "rb1"), (13, 24, "rb2"), (25, 36, "flex")):
                mask = rank.between(lo, hi)
                if mask.any():
                    masks.append(("rb_tier", name, mask))
        if position == "WR":
            for lo, hi, name in ((1, 3, "wr_apex_top3"), (4, 12, "wr_slots_4_12")):
                mask = rank.between(lo, hi)
                if mask.any():
                    masks.append(("wr_tier", name, mask))

    if "game_status" in ranked_df:
        status = pd.to_numeric(ranked_df["game_status"], errors="coerce")
        mask = status < 1.0
        if mask.any():
            masks.append(("availability", "questionable", mask))
    returning = pd.Series(False, index=ranked_df.index)
    if "is_returning_from_absence" in ranked_df:
        returning = (
            pd.to_numeric(ranked_df["is_returning_from_absence"], errors="coerce").fillna(0).eq(1)
        )
    if "days_rest" in ranked_df:
        rest = pd.to_numeric(ranked_df["days_rest"], errors="coerce").fillna(7)
        if "is_returning_from_absence" not in ranked_df:
            returning |= rest >= 14
        ret_1wk = returning & rest.eq(14)
        ret_2plus = returning & rest.gt(14)
        if ret_1wk.any():
            masks.append(("availability", "returning_1wk", ret_1wk))
        if ret_2plus.any():
            masks.append(("availability", "returning_2plus", ret_2plus))
    if returning.any():
        masks.append(("availability", "returning", returning))

    if position == "RB":
        labels = label_ascension_rows(ranked_df)
        inheritor = labels.eq(ASCENSION)
        if inheritor.any():
            masks.append(("miss_type", "inheritor", inheritor))

    if min_season is not None and len(min_season) > 0:
        rookies = label_rookie_rows(ranked_df, min_season)
        rookie_mask = rookies.isin([ROOKIE_EARLY, ROOKIE_REST])
        if rookie_mask.any():
            masks.append(("experience", "rookie", rookie_mask))

    max_week = ranked_df.groupby("season")["week"].transform("max")
    non_final = ranked_df["week"] < max_week
    if non_final.any():
        masks.append(("schedule", "final_week_excluded", non_final))
    return masks


def context_slice_masks(ranked_df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """Pregame context slices for market/team-environment diagnostics."""
    masks: list[tuple[str, str, pd.Series]] = []
    for col, family, low_name, high_name in (
        ("implied_team_total", "market", "implied_team_total_low_q1", "implied_team_total_high_q4"),
        ("total_line", "market", "total_line_low_q1", "total_line_high_q4"),
    ):
        if col not in ranked_df:
            continue
        low, high = _extreme_quantile_masks(pd.to_numeric(ranked_df[col], errors="coerce"))
        if low is None or high is None:
            continue
        if low.any():
            masks.append((family, low_name, low))
        if high.any():
            masks.append((family, high_name, high))

    if "implied_team_total" in ranked_df:
        team_total = pd.to_numeric(ranked_df["implied_team_total"], errors="coerce")
        opp_total = None
        if "implied_opp_total" in ranked_df:
            opp_total = pd.to_numeric(ranked_df["implied_opp_total"], errors="coerce")
        elif "total_line" in ranked_df:
            opp_total = pd.to_numeric(ranked_df["total_line"], errors="coerce") - team_total
        if opp_total is None:
            return masks
        edge = team_total - opp_total
        favorite = edge >= 3.0
        underdog = edge <= -3.0
        pickem = edge.abs() < 3.0
        if favorite.any():
            masks.append(("market", "favorite_by_3plus", favorite))
        if underdog.any():
            masks.append(("market", "underdog_by_3plus", underdog))
        if pickem.any():
            masks.append(("market", "pickem_within_3", pickem))
    return masks


def join_source_projection(
    base_df: pd.DataFrame,
    projection: pd.DataFrame,
    *,
    pred_col: str,
) -> pd.DataFrame:
    """Join a projected source to model actual rows on exact player/season/week keys."""
    base = _normalise_keys(base_df)
    proj = _normalise_keys(projection)
    extra_cols = [
        c for c in proj.columns if c not in _KEY_COLS and c != pred_col and c not in base.columns
    ]
    proj = proj[[*_KEY_COLS, pred_col, *extra_cols]].dropna(subset=[pred_col])
    proj = proj.drop_duplicates(_KEY_COLS, keep="last")
    joined = base.merge(proj, on=_KEY_COLS, how="inner")
    return joined.rename(columns={pred_col: PRED_COL})


def model_source_frames(model_df: pd.DataFrame) -> list[tuple[SourceMeta, pd.DataFrame]]:
    base = _normalise_keys(model_df)
    out: list[tuple[SourceMeta, pd.DataFrame]] = []
    for name, label, col in MODEL_SOURCES:
        if col not in base.columns:
            continue
        meta = SourceMeta(name=name, label=label, kind="model", native_col=col)
        out.append((meta, base.rename(columns={col: PRED_COL})))
    return out


def expert_source_frame(
    source,
    raw_df: pd.DataFrame | None,
    position: str,
    base_df: pd.DataFrame,
    scoring_format: str,
) -> tuple[SourceMeta, pd.DataFrame | None, dict[str, Any]]:
    name = "rotowire" if source.name == "sleeper" else source.name
    label = "RotoWire" if source.name == "sleeper" else source.label
    meta = SourceMeta(name=name, label=label, kind="expert", native_col=_EXPERT_PRED_COL)
    if position in source.skipped:
        return meta, None, {"skipped": True, "reason": f"{label} has no {position} projections"}
    if raw_df is None:
        return meta, None, {"skipped": True, "reason": f"{label} projections unavailable"}
    projection = source.project(raw_df, position, scoring_format)
    if projection is None or projection.empty:
        return meta, None, {"skipped": True, "reason": f"no {label} projections for {position}"}
    joined = join_source_projection(base_df, projection, pred_col=_EXPERT_PRED_COL)
    if joined.empty:
        return (
            meta,
            None,
            {
                "skipped": True,
                "reason": f"no exact (player_id, season, week) overlap for {position}",
            },
        )
    return meta, joined, {"skipped": False, "n_rows": int(len(joined))}


def load_position_predictions(
    position: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    eval_seasons: Sequence[int],
    scoring_format: str,
    from_artifacts: bool,
    validate: bool,
    artifact_builder: Callable[..., pd.DataFrame] | None = None,
    artifact_validator: Callable[..., dict[str, Any]] | None = None,
    fresh_loader: Callable[[str, Sequence[int], str], pd.DataFrame] | None = None,
) -> PredictionLoad:
    """Load model rows from artifacts when valid, else from a fresh pipeline run."""
    position = position.upper()
    fresh_loader = fresh_loader or _fresh_model_predictions
    if from_artifacts:
        try:
            if artifact_builder is None:
                from src.analysis.artifact_eval import build_test_df_from_artifacts

                artifact_builder = build_test_df_from_artifacts
            built = artifact_builder(
                position,
                train_df,
                val_df,
                test_df,
                scoring_format=scoring_format,
            )
            validation = None
            if validate:
                if artifact_validator is None:
                    from src.analysis.artifact_eval import validate_reconstruction

                    artifact_validator = validate_reconstruction
                validation = artifact_validator(
                    position, built, scoring_format=scoring_format, verbose=True
                )
                if validation.get("status") in {"warn", "fail"}:
                    raise RuntimeError(f"artifact validation {validation['status']}: {validation}")
            return PredictionLoad(
                df=_filter_eval_seasons(built, eval_seasons),
                mode="artifacts",
                artifact_validation=validation,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic fallback boundary
            print(
                f"[topn_expert_gap] {position}: artifact path unavailable/untrusted "
                f"({exc!r}); falling back to fresh pipeline predictions."
            )
            fresh = fresh_loader(position, eval_seasons, scoring_format)
            return PredictionLoad(
                df=_filter_eval_seasons(fresh, eval_seasons),
                mode="fresh_fallback",
                artifact_error=repr(exc),
            )
    fresh = fresh_loader(position, eval_seasons, scoring_format)
    return PredictionLoad(df=_filter_eval_seasons(fresh, eval_seasons), mode="fresh")


def _fresh_model_predictions(
    position: str, eval_seasons: Sequence[int], scoring_format: str
) -> pd.DataFrame:
    del eval_seasons, scoring_format
    result = importlib.import_module(f"src.{position.lower()}.run_pipeline").run()
    if "test_df" not in result:
        raise KeyError(f"{position} run() result has no 'test_df'")
    return result["test_df"]


def _filter_eval_seasons(df: pd.DataFrame, eval_seasons: Sequence[int]) -> pd.DataFrame:
    if "season" not in df.columns:
        return df.copy()
    eval_set = {int(s) for s in eval_seasons}
    return df[df["season"].astype(int).isin(eval_set)].copy()


def build_position_report(
    position: str,
    model_df: pd.DataFrame,
    *,
    expert_raws: dict[str, pd.DataFrame | None],
    experts,
    scoring_format: str,
    n_boot: int,
    seed: int,
    min_season: pd.Series | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute all report rows for one position from one row-level substrate."""
    base_df = _normalise_keys(model_df)
    if "position" not in base_df.columns:
        base_df["position"] = position
    model_frames = model_source_frames(base_df)
    expert_frames: list[tuple[SourceMeta, pd.DataFrame]] = []
    coverage_rows: list[dict[str, Any]] = []
    for src in experts:
        meta, frame, status = expert_source_frame(
            src, expert_raws.get(src.name), position, base_df, scoring_format
        )
        coverage_rows.append(
            {
                "position": position,
                "source": meta.name,
                "source_label": meta.label,
                "source_kind": meta.kind,
                "model_or_expert": meta.kind,
                **status,
            }
        )
        if frame is not None:
            expert_frames.append((meta, frame))
    for meta, frame in model_frames:
        coverage_rows.append(
            {
                "position": position,
                "source": meta.name,
                "source_label": meta.label,
                "source_kind": meta.kind,
                "model_or_expert": meta.kind,
                "skipped": False,
                "n_rows": int(len(frame)),
            }
        )

    metric_rows: list[dict[str, Any]] = []
    player_misses: list[dict[str, Any]] = []
    weekly_misses: list[dict[str, Any]] = []
    all_frames = [*model_frames, *expert_frames]
    for meta, frame in all_frames:
        metric_rows.extend(cohort_error_rows(position, meta, frame, base_df, min_season=min_season))
        metric_rows.extend(td_calibration_rows(position, meta, frame))
        season_rows, season_misses = season_selection_rows(position, meta, frame, base_df)
        weekly_rows, week_misses = weekly_selection_rows(position, meta, frame, base_df)
        metric_rows.extend(season_rows)
        metric_rows.extend(weekly_rows)
        player_misses.extend(season_misses)
        weekly_misses.extend(week_misses)
    metric_rows.extend(
        paired_uncertainty_rows(position, model_frames, expert_frames, n_boot=n_boot, seed=seed)
    )
    return metric_rows, player_misses, weekly_misses, coverage_rows


def run_analysis(
    *,
    positions: Sequence[str] = DEFAULT_POSITIONS,
    eval_seasons: Sequence[int] | None = None,
    scoring_format: str = "ppr",
    from_artifacts: bool = True,
    validate: bool = True,
    out_dir: str | Path = DEFAULT_OUT,
    n_boot: int = 1000,
    seed: int = 0,
    model_preds_loader: Callable[[str, Sequence[int], str], pd.DataFrame] | None = None,
    nflcom_loader=None,
    sleeper_loader=None,
    local_experts: Sequence[LocalExpertSpec] | None = None,
) -> dict[str, Any]:
    """Run the diagnostic and write summary/report artifacts."""
    eval_seasons = tuple(int(s) for s in (eval_seasons or TEST_SEASONS or (2025,)))
    positions = tuple(p.upper() for p in positions)

    from src.analysis.cohort_analysis import _load_splits

    train_df, val_df, test_df = _load_splits()
    experts = build_expert_sources(nflcom_loader, sleeper_loader, local_experts)
    expert_raws: dict[str, pd.DataFrame | None] = {}
    for src in experts:
        try:
            expert_raws[src.name] = src.load(list(eval_seasons))
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            print(
                f"[topn_expert_gap] {src.label} load failed "
                f"({type(exc).__name__}: {exc}); skipping this expert."
            )
            expert_raws[src.name] = None

    metrics: list[dict[str, Any]] = []
    player_misses: list[dict[str, Any]] = []
    weekly_misses: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    load_meta: dict[str, dict[str, Any]] = {}
    min_season = player_min_season([train_df, val_df, test_df])

    for position in positions:
        load = load_position_predictions(
            position,
            train_df,
            val_df,
            test_df,
            eval_seasons=eval_seasons,
            scoring_format=scoring_format,
            from_artifacts=from_artifacts,
            validate=validate,
            fresh_loader=model_preds_loader,
        )
        load_meta[position] = {
            "mode": load.mode,
            "artifact_validation": load.artifact_validation,
            "artifact_error": load.artifact_error,
        }
        m_rows, p_rows, w_rows, c_rows = build_position_report(
            position,
            load.df,
            expert_raws=expert_raws,
            experts=experts,
            scoring_format=scoring_format,
            n_boot=n_boot,
            seed=seed,
            min_season=min_season,
        )
        metrics.extend(m_rows)
        player_misses.extend(p_rows)
        weekly_misses.extend(w_rows)
        coverage.extend(c_rows)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    player_misses_df = pd.DataFrame(player_misses)
    weekly_misses_df = pd.DataFrame(weekly_misses)
    coverage_df = pd.DataFrame(coverage)
    metrics_df.to_csv(out_path / "metrics.csv", index=False)
    player_misses_df.to_csv(out_path / "player_misses.csv", index=False)
    weekly_misses_df.to_csv(out_path / "weekly_misses.csv", index=False)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "positions": list(positions),
        "eval_seasons": list(eval_seasons),
        "scoring_format": scoring_format,
        "from_artifacts": bool(from_artifacts),
        "validate": bool(validate),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "prediction_loads": load_meta,
        "coverage": coverage,
        "metrics": metrics,
        "player_misses_count": int(len(player_misses)),
        "weekly_misses_count": int(len(weekly_misses)),
    }
    (out_path / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    (out_path / "summary.md").write_text(render_summary(metrics_df, coverage_df, report))
    return report


def render_summary(metrics: pd.DataFrame, coverage: pd.DataFrame, report: dict[str, Any]) -> str:
    lines = [
        "# Top-N Expert Gap",
        "",
        f"Generated: {report['generated_at']}",
        f"Seasons: {', '.join(map(str, report['eval_seasons']))}",
        f"Scoring: {report['scoring_format']}",
        "",
        "## Coverage",
        "",
        _simple_table(
            coverage.fillna("")
            .loc[
                :,
                [
                    c
                    for c in (
                        "position",
                        "source_label",
                        "source_kind",
                        "n_rows",
                        "skipped",
                        "reason",
                    )
                    if c in coverage
                ],
            ]
            .to_dict("records")
        ),
        "",
        "## Top-30 Cohort Error",
        "",
    ]
    if not metrics.empty:
        top30 = metrics[
            (metrics["metric_group"] == "cohort_error")
            & (metrics["top_n"] == 30)
            & (metrics["slice_family"] == "global")
        ]
        cols = ["position", "source_label", "n_rows", "mae", "rmse", "r2", "bias"]
        lines.append(_simple_table(_rounded_records(top30, cols)))
        lines.extend(["", "## Season Top-30 Selection", ""])
        sel = metrics[(metrics["metric_group"] == "season_selection") & (metrics["top_n"] == 30)]
        cols = ["position", "source_label", "season", "precision", "recall", "f1", "regret"]
        lines.append(_simple_table(_rounded_records(sel, cols)))
        lines.extend(["", "## Paired Model-Vs-Expert Uncertainty", ""])
        paired = metrics[metrics["metric_group"] == "paired_uncertainty"]
        cols = [
            "position",
            "source_label",
            "expert_label",
            "metric",
            "delta_model_minus_expert",
            "ci_lo",
            "ci_hi",
            "p_value",
            "favored",
        ]
        lines.append(_simple_table(_rounded_records(paired, cols)))
    else:
        lines.append("No metrics were produced.")
    lines.append("")
    return "\n".join(lines)


def _rounded_records(df: pd.DataFrame, cols: Sequence[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    use = [c for c in cols if c in df.columns]
    records = df[use].to_dict("records")
    for row in records:
        for key, value in list(row.items()):
            if isinstance(value, float) and math.isfinite(value):
                row[key] = round(value, 4)
    return records


def _simple_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "_No rows._"
    cols = list(records[0])
    widths = {c: max(len(str(c)), *(len(_fmt_cell(r.get(c))) for r in records)) for c in cols}
    header = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    body = [
        "| " + " | ".join(_fmt_cell(row.get(c)).ljust(widths[c]) for c in cols) + " |"
        for row in records
    ]
    return "\n".join([header, sep, *body])


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _prediction_prefix(total_col: str) -> str | None:
    if not total_col.endswith("_total"):
        return None
    if total_col == _EXPERT_PRED_COL:
        return "expert_pred_"
    if not total_col.startswith("pred_"):
        return None
    return total_col[: -len("total")]


def _extreme_quantile_masks(values: pd.Series) -> tuple[pd.Series | None, pd.Series | None]:
    finite = values.dropna()
    if len(finite) < 4 or finite.nunique(dropna=True) < 2:
        return None, None
    lo = float(finite.quantile(0.25))
    hi = float(finite.quantile(0.75))
    if lo >= hi:
        lo = float(finite.min())
        hi = float(finite.max())
    low = pd.Series(False, index=values.index)
    high = pd.Series(False, index=values.index)
    low.loc[finite.index] = finite.le(lo)
    high.loc[finite.index] = finite.ge(hi)
    if (low & high).any():
        return None, None
    return low, high


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return _nan()
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[positive].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size < 2 or np.nanstd(pred) < 1e-12 or np.nanstd(actual) < 1e-12:
        return _nan()
    rho, _ = spearmanr(pred, actual)
    return float(rho) if np.isfinite(rho) else _nan()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="+", default=list(DEFAULT_POSITIONS))
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(TEST_SEASONS) if TEST_SEASONS else [2025],
    )
    parser.add_argument("--scoring-format", default="ppr", choices=["ppr", "half_ppr", "standard"])
    parser.add_argument(
        "--from-artifacts",
        dest="from_artifacts",
        action="store_true",
        default=True,
        help="Prefer saved model artifacts before fresh pipeline fallback (default).",
    )
    parser.add_argument(
        "--fresh",
        dest="from_artifacts",
        action="store_false",
        help="Run fresh position pipelines instead of loading saved model artifacts.",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="Validate reconstructed artifacts before trusting them (default).",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip artifact reconstruction validation.",
    )
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local-expert",
        action="append",
        default=[],
        metavar="NAME[:PRED_COL]=PATH",
        help=(
            "Approved local projection snapshot to score as an expert source. "
            "The file must contain player_id, season, week, and a projection column; "
            "optional raw TD projections use pred_*/projected_*/*_projection names."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.sync:
        from src.analysis.artifact_eval import warn_if_sync_noop
        from src.shared.model_sync import sync_models_from_s3

        warn_if_sync_noop()
        sync_models_from_s3()
    run_analysis(
        positions=args.positions,
        eval_seasons=args.seasons,
        scoring_format=args.scoring_format,
        from_artifacts=args.from_artifacts,
        validate=args.validate,
        out_dir=args.out,
        n_boot=args.n_boot,
        seed=args.seed,
        local_experts=[parse_local_expert_spec(x) for x in args.local_expert],
    )
    print(f"Wrote {os.path.join(args.out, 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
