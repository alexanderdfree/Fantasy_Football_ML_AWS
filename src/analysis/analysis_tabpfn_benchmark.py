"""Six-position benchmark: every model vs RotoWire, with the opt-in TabPFN-3 variant.

Operator CLI (coverage-excluded). For each position it runs the pipeline with the
default-off TabPFN-3 variant enabled (``train_tabpfn=True`` injected into a *copy*
of the position CONFIG — no config-file edit, no retrain trigger), then reports,
per model (Ridge / NN / Attention-NN / LightGBM / TabPFN):

  * regular MAE / RMSE / R2 on the full held-out test split,
  * top-N (default 30, ranked by season-total actual FP) R2 / RMSE,
  * Q1-Q4 bands (equal-frequency quartiles of actual FP) RMSE / MAE,

plus a RotoWire-matched block (RotoWire = Sleeper ``company=="rotowire"``;
QB/RB/WR/TE/DST, no K) where every model and RotoWire are scored on the *same*
inner-joined rows so the model-vs-expert comparison is fair.

TabPFN is the opt-in, non-commercial, benchmark-only variant (ADR-0003): the
``tabpfn`` import is lazy inside the pipeline, so this module imports without the
package and the metric/formatting helpers below are pure pandas (unit-smoke-tested
in ``tests/analysis/test_analysis_tabpfn_benchmark.py``). A real run needs
``tabpfn`` installed + a Prior Labs ``TABPFN_TOKEN`` (see ADR-0003).

Memory note: each position is a full pipeline run. On a memory-constrained box (or
with other heavy sessions active), run one ``--positions`` at a time sharing a
``--cache-dir`` — completed positions are skipped on rerun, so a crash resumes.

Usage:
    python -m src.analysis.analysis_tabpfn_benchmark --cache-dir /tmp/tabpfn_bench --out report.md
    python -m src.analysis.analysis_tabpfn_benchmark --positions WR --no-rotowire
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import pandas as pd

from src.config import TEST_SEASONS
from src.shared.evaluation import compute_metrics

# pred_*_total column -> display label, in report order.
MODEL_LABELS: dict[str, str] = {
    "pred_ridge_total": "Ridge",
    "pred_nn_total": "NN",
    "pred_attn_nn_total": "AttnNN",
    "pred_lgbm_total": "LightGBM",
    "pred_tabpfn_total": "TabPFN",
}
ALL_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
ROTOWIRE_POSITIONS = ("QB", "RB", "WR", "TE", "DST")  # Sleeper/RotoWire has no K
_BANDS = ("Q1", "Q2", "Q3", "Q4")
_EXPERT_COL = "expert_pred_total"


# ---------- pure metric helpers (unit-tested; no GPU / tabpfn / network) -----


def _present_pred_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in MODEL_LABELS if c in df.columns]


def _id_col(df: pd.DataFrame) -> str:
    """The row key: ``player_id`` for offense, ``team`` for DST (the pipeline uses
    ``player_id`` = team abbrev for DST, but fall back to ``team`` defensively)."""
    return "player_id" if "player_id" in df.columns else "team"


def _quartile_bands(actual: pd.Series) -> pd.Series:
    """Equal-frequency quartiles of actual FP, rank-based so ties never collapse a bin."""
    return pd.qcut(actual.rank(method="first"), 4, labels=list(_BANDS))


def position_metrics(test_df: pd.DataFrame, top_n: int = 30) -> dict:
    """Per-model regular / top-N / Q1-Q4 metrics from a pipeline ``test_df``.

    ``test_df`` needs ``fantasy_points``, an id column (``player_id`` or ``team``),
    and one or more ``pred_*_total`` columns. Top-N players are ranked by
    season-total actual FP, then *all* their weekly rows are scored. Returns
    ``{label: {"regular": {mae,rmse,r2,n}, "topN": {r2,rmse,n}, "bands": {Q1: {rmse,mae,n}, ...}}}``.
    """
    y = test_df["fantasy_points"].to_numpy(float)
    idc = _id_col(test_df)
    season_total = test_df.groupby(idc)["fantasy_points"].sum().sort_values(ascending=False)
    top_ids = set(season_total.head(top_n).index)
    m_top = test_df[idc].isin(top_ids).to_numpy()
    bands = _quartile_bands(test_df["fantasy_points"])
    out: dict = {}
    for col in _present_pred_cols(test_df):
        p = test_df[col].to_numpy(float)
        reg = compute_metrics(y, p)
        reg["n"] = int(len(y))
        top = compute_metrics(y[m_top], p[m_top])
        band = {}
        for b in _BANDS:
            bm = (bands == b).to_numpy()
            bmet = compute_metrics(y[bm], p[bm])
            band[b] = {"rmse": bmet["rmse"], "mae": bmet["mae"], "n": int(bm.sum())}
        out[MODEL_LABELS[col]] = {
            "regular": reg,
            "topN": {"r2": top["r2"], "rmse": top["rmse"], "n": int(m_top.sum())},
            "bands": band,
        }
    return out


def rotowire_matched_metrics(
    test_df: pd.DataFrame, expert_df: pd.DataFrame, top_n: int = 30
) -> dict:
    """Inner-join ``expert_df`` onto ``test_df`` and score every model + RotoWire on
    the matched rows.

    ``expert_df`` is ``[player_id, season, week, expert_pred_total]`` (the
    ``_project_sleeper_to_ppr`` output; for DST ``player_id`` is the team abbrev,
    matching the pipeline's DST key). Returns
    ``{label_or_'RotoWire': {"regular": {...}, "topN": {...}}, "_n_matched": int}``;
    ``_n_matched == 0`` (and no model keys) when nothing overlaps.
    """
    idc = _id_col(test_df)
    m = test_df.copy()
    m["_k"] = m[idc].astype(str)
    m["season"] = m["season"].astype(int)
    m["week"] = m["week"].astype(int)
    e = expert_df.copy()
    e["_k"] = e["player_id"].astype(str)
    e["season"] = e["season"].astype(int)
    e["week"] = e["week"].astype(int)
    joined = m.merge(
        e[["_k", "season", "week", _EXPERT_COL]], on=["_k", "season", "week"], how="inner"
    )
    if joined.empty:
        return {"_n_matched": 0}
    y = joined["fantasy_points"].to_numpy(float)
    season_total = joined.groupby("_k")["fantasy_points"].sum().sort_values(ascending=False)
    top_ids = set(season_total.head(top_n).index)
    m_top = joined["_k"].isin(top_ids).to_numpy()
    out: dict = {"_n_matched": int(len(joined))}
    for col in _present_pred_cols(joined) + [_EXPERT_COL]:
        label = MODEL_LABELS.get(col, "RotoWire")
        p = joined[col].to_numpy(float)
        reg = compute_metrics(y, p)
        reg["n"] = int(len(y))
        top = compute_metrics(y[m_top], p[m_top])
        out[label] = {
            "regular": reg,
            "topN": {"r2": top["r2"], "rmse": top["rmse"], "n": int(m_top.sum())},
        }
    return out


# ---------- report formatting (pure) ----------------------------------------


def _metric_row(label: str, reg: dict, top: dict) -> str:
    return (
        f"{label:9s} | {reg['mae']:6.3f} {reg['rmse']:6.3f} {reg['r2']:6.3f} "
        f"| {top['r2']:7.3f} {top['rmse']:7.3f}"
    )


def format_report(metrics_a: dict, metrics_b: dict, top_n: int = 30) -> str:
    """Render the Section A (full test) + Section B (RotoWire-matched) tables."""
    head = f"{'model':9s} | {'MAE':>6} {'RMSE':>6} {'R2':>6} | top{top_n} {'R2':>7} {'RMSE':>7}"
    lines = ["# Section A -- full test set (all models)", ""]
    for pos, pm in metrics_a.items():
        if not pm:
            lines += [f"## {pos}: (not captured)", ""]
            continue
        n = next(iter(pm.values()))["regular"]["n"]
        lines += [f"## {pos}  (n={n})", head]
        for label, d in pm.items():
            lines.append(_metric_row(label, d["regular"], d["topN"]))
        lines.append("  Q1-Q4 RMSE/MAE per actual-FP quartile:")
        for label, d in pm.items():
            cells = "  ".join(
                f"{b} {d['bands'][b]['rmse']:5.2f}/{d['bands'][b]['mae']:4.2f}" for b in _BANDS
            )
            lines.append(f"  {label:9s} | {cells}")
        lines.append("")
    lines += [
        "",
        "# Section B -- vs RotoWire (matched subset; models re-scored on same rows; no K)",
        "",
    ]
    for pos, bm in metrics_b.items():
        if not bm or bm.get("_n_matched", 0) == 0:
            lines += [f"## {pos}: (no RotoWire overlap)", ""]
            continue
        lines += [f"## {pos}  (matched n={bm['_n_matched']})", head]
        for label, d in bm.items():
            if label.startswith("_"):
                continue
            lines.append(_metric_row(label, d["regular"], d["topN"]))
        lines.append("")
    return "\n".join(lines)


# ---------- heavy run path (needs tabpfn + GPU; not unit-tested) -------------

_KEEP_BASE = ("player_id", "team", "season", "week", "fantasy_points")


def run_position(pos: str, seed: int = 42, cache_dir: str | None = None) -> pd.DataFrame:
    """Run ``pos``'s pipeline with TabPFN enabled; return the slimmed ``test_df``.

    Idempotent when ``cache_dir`` is set (a cached ``tdf_{pos}.parquet`` is reused),
    so a crash/OOM mid-sweep resumes. ``train_tabpfn=True`` is injected into a copy
    of the position CONFIG, leaving the committed config untouched.
    """
    import importlib

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, f"tdf_{pos}.parquet")
        if os.path.exists(cached):
            return pd.read_parquet(cached)
    runner = importlib.import_module(f"src.{pos.lower()}.run_pipeline")
    cfg = dict(runner.CONFIG)
    cfg["train_tabpfn"] = True
    # K/DST build their own splits (run(seed, config)); skill positions take frames
    # that default to None -> the pipeline self-loads the held-out split.
    result = runner.run(seed=seed, config=cfg) if pos in ("K", "DST") else runner.run(config=cfg)
    df = result["test_df"]
    slim = df[[c for c in _KEEP_BASE if c in df.columns] + _present_pred_cols(df)].copy()
    if cache_dir:
        slim.to_parquet(os.path.join(cache_dir, f"tdf_{pos}.parquet"), index=False)
    return slim


def load_rotowire(seasons: Sequence[int], pos: str, scoring_format: str = "ppr") -> pd.DataFrame:
    """RotoWire projection for ``pos`` -> ``[player_id, season, week, expert_pred_total]``."""
    from src.analysis.analysis_expert_comparison import _project_sleeper_to_ppr
    from src.analysis.sleeper_loader import load_sleeper_with_gsis_id

    raw = load_sleeper_with_gsis_id(list(seasons))
    return _project_sleeper_to_ppr(raw, pos, scoring_format)


def main(argv: Sequence[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description="6-position TabPFN-3 vs models vs RotoWire benchmark")
    ap.add_argument("--positions", nargs="+", default=list(ALL_POSITIONS))
    ap.add_argument("--seasons", nargs="+", type=int, default=list(TEST_SEASONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument(
        "--cache-dir", default=None, help="reuse/save per-position test_df parquets (resumable)"
    )
    ap.add_argument("--out", default=None, help="write the report markdown to this path")
    ap.add_argument("--no-rotowire", action="store_true", help="skip the RotoWire-matched section")
    args = ap.parse_args(argv)

    metrics_a: dict = {}
    metrics_b: dict = {}
    for pos in args.positions:
        print(f"=== running {pos} (train_tabpfn=True) ===", flush=True)
        tdf = run_position(pos, seed=args.seed, cache_dir=args.cache_dir)
        metrics_a[pos] = position_metrics(tdf, top_n=args.top_n)
        if not args.no_rotowire and pos in ROTOWIRE_POSITIONS:
            expert = load_rotowire(args.seasons, pos)
            if expert is not None and len(expert):
                metrics_b[pos] = rotowire_matched_metrics(tdf, expert, top_n=args.top_n)

    report = format_report(metrics_a, metrics_b, top_n=args.top_n)
    print("\n" + report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"\nwrote {args.out}", flush=True)
    return report


if __name__ == "__main__":
    main()
