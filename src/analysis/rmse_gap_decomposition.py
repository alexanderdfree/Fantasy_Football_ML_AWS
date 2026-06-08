"""Decompose our RB/WR fantasy-point error by per-stat head — where does RMSE come from?

Read-only diagnostic (no retrain, no config/model changes; ``src/analysis/`` is not a
retrain-trigger path). It answers the question behind "experts beat my best model on
RMSE/R² for RB/WR on the Comparison tab": **which target head drives our models'
squared fantasy-point error** — the conservative count/hurdle heads (TDs on Poisson-NLL,
receptions on hurdle-NegBin) or the squared-error yards heads (MSE)? — and **which
player-weeks drive it**.

Method — *exact* additive MSE decomposition
-------------------------------------------
The fantasy-point residual is linear in the per-target errors::

    r   = FP_pred - FP_actual = sum_t c_t,   c_t = w_t * (pred_t - actual_t)

where ``w_t`` is the PPR scoring weight (6 / TD, 1 / reception, 0.1 / yard, -2 / fumble).
Squaring and taking the mean gives an **exact** split of the FP mean-squared error into
per-head contributions (each term is the head's covariance with the *total* error)::

    MSE = mean(r^2) = mean(r * sum_t c_t) = sum_t mean(r * c_t)

The per-head shares sum to 100% of MSE (asserted at runtime). A 1-TD miss costs 6 FP
vs a 30-yard miss costing 3 FP, so the ×6 TD weight gives the count heads outsized
leverage on RMSE even when their *raw-stat* error is small — this quantifies that.

Data
----
* Model per-row, per-target preds: the **served** artifacts via
  :func:`src.analysis.artifact_eval.build_test_df_from_artifacts` (no retrain; Ridge/LGBM
  reproduce served numbers exactly, NN/Attn are loaded weights). So this reflects exactly
  the models the Comparison tab shows.
* Top-30 player slice: ``src/serving/comparison_experts.json`` → ``top30_ids`` (the same
  slice the tab scores on), with the committed expert top-30 aggregates for context.
* Raw-stat actuals + PPR weights: the test split + :data:`src.config.SCORING_PPR`.

The core analysis is fully offline. ``--with-experts`` adds a live per-tier
model-vs-expert block (nflverse + NFL.com + Sleeper; best-effort, skipped on any error).

Usage::

    python -m src.analysis.rmse_gap_decomposition --positions RB WR
    python -m src.analysis.rmse_gap_decomposition --positions RB WR --with-experts
"""

from __future__ import annotations

import os

# CPU-considerate: the owner may be gaming. Inference-only is seconds, but cap BLAS/OMP
# anyway (set before numpy/torch import). Override via the env if you want full cores.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Feature-building emits many pandas "highly fragmented DataFrame" PerformanceWarnings —
# noise for this read-only diagnostic.
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

from src.analysis.artifact_eval import build_test_df_from_artifacts
from src.analysis.cohort_analysis import _load_splits
from src.config import SCORING_PPR
from src.shared.aggregate_targets import POSITION_TARGET_MAP, predictions_to_fantasy_points
from src.shared.error_analysis import compute_stratum_metrics
from src.shared.evaluation import compute_metrics

try:  # torch is a hard dep of artifact_eval; cap its threads too.
    import torch

    torch.set_num_threads(2)
except Exception:  # pragma: no cover - defensive only
    pass

_COMPARISON_EXPERTS_PATH = (
    Path(__file__).resolve().parents[1] / "serving" / "comparison_experts.json"
)
_MODELS = ("ridge", "nn", "attn_nn", "lgbm")
# Human-readable loss-family labels (the actual map is read from POSITION_CONFIG).
_LOSS_PRETTY = {
    "mse": "MSE",
    "huber": "Huber",
    "poisson_nll": "Poisson-NLL",
    "hurdle_negbin": "hurdle-NegBin",
    "hurdle_poisson": "hurdle-Poisson",
}


def _targets(pos: str) -> list[str]:
    """Ordered raw-stat target names for a skill position."""
    return list(POSITION_TARGET_MAP[pos])


def _weights(pos: str) -> dict[str, float]:
    """PPR scoring weight per target (the ``w_t`` in the decomposition)."""
    tmap = POSITION_TARGET_MAP[pos]
    return {t: float(SCORING_PPR[tmap[t]]) for t in tmap}


def _loss_families(pos: str) -> dict[str, str]:
    """Per-head loss family from the production ``POSITION_CONFIG`` (not CONFIG_TINY).

    Mirrors ``MultiHeadTrainer``'s default: a target absent from ``head_losses`` falls
    back to ``"huber"`` (src/shared/training.py). Read from the real config so the label
    can't drift from what actually trains.
    """
    import importlib

    cfg = importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG
    head_losses = dict(getattr(cfg, "head_losses", {}) or {})
    return {t: head_losses.get(t, "huber") for t in _targets(pos)}


def _model_loss_labels(
    pos: str, model: str, targets: list[str], nn_families: dict[str, str]
) -> dict[str, str]:
    """Training objective the DECOMPOSED model actually uses per head.

    The per-head loss *families* (Poisson-NLL / hurdle-NegBin / MSE) are the **NN's**
    ``head_losses`` — NN-only. LightGBM fits one regressor per target but all share a
    single ``lgbm_objective`` (``regression`` = L2 here); Ridge is L2, ElasticNet L1+L2.
    So the same target is optimized by very different objectives depending on the model,
    and the NN's family labels must NOT be pinned onto a LightGBM/Ridge decomposition.
    """
    if model in ("nn", "attn_nn"):
        return {t: _LOSS_PRETTY.get(nn_families[t], nn_families[t]) for t in targets}
    if model == "ridge":
        return {t: "L2" for t in targets}
    if model == "enet":
        return {t: "L1+L2" for t in targets}
    if model == "lgbm":
        import importlib

        cfg = importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG
        obj = getattr(cfg, "lgbm_objective", "regression")
        label = "L2 (regression)" if obj == "regression" else obj
        return {t: label for t in targets}
    return {t: "?" for t in targets}


def _load_top30_ids(pos: str) -> set[str]:
    data = json.loads(_COMPARISON_EXPERTS_PATH.read_text())
    return {str(x) for x in data.get("top30_ids", {}).get(pos, [])}


def _expert_top30_aggregates(pos: str) -> dict:
    data = json.loads(_COMPARISON_EXPERTS_PATH.read_text())
    return data.get("subsets", {}).get("top30", {}).get(pos, {}) or {}


def _resolve_model_dir(pos: str) -> str:
    """Artifact dir that actually exists: the registry/served path, else the local run path.

    Serving + ``artifact_eval`` default to ``src/{pos}/outputs/models`` (the ``--sync``
    target); local ``run_pipeline`` / ``benchmark`` write to ``{pos}/outputs/models``.
    Prefer the former when present, fall back to the latter so a plain local run works.
    """
    p = pos.lower()
    for cand in (f"src/{p}/outputs/models", f"{p}/outputs/models"):
        path = Path(cand)
        if path.is_dir() and any(path.iterdir()):
            return cand
    return f"src/{p}/outputs/models"


def _name_col(df: pd.DataFrame) -> str | None:
    for c in ("player_name", "player_display_name", "display_name", "player"):
        if c in df.columns:
            return c
    return None


def _prepare(pos: str, train, val, test):
    """Build the top-30-sliced test frame with per-model FP predictions + actual FP."""
    df = build_test_df_from_artifacts(
        pos, train, val, test, scoring_format="ppr", model_dir=_resolve_model_dir(pos)
    )
    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)

    top30 = _load_top30_ids(pos)
    if top30:
        df = df[df["player_id"].isin(top30)].copy()

    targets = _targets(pos)
    # Actual FP via the SAME aggregation as the served pred totals → guarantees the
    # decomposition identity r == sum_t c_t holds exactly.
    df["actual_fp"] = predictions_to_fantasy_points(
        pos, {t: df[t].to_numpy(dtype=np.float64) for t in targets}, "ppr"
    )

    available = [m for m in _MODELS if f"pred_{m}_total" in df.columns]
    return df, targets, available


def _decompose(df: pd.DataFrame, pos: str, model: str, targets, weights, families) -> pd.DataFrame:
    """Exact per-head split of the FP MSE for one model on the (already sliced) frame."""
    actual_fp = df["actual_fp"].to_numpy(dtype=np.float64)
    pred_fp = df[f"pred_{model}_total"].to_numpy(dtype=np.float64)
    r = pred_fp - actual_fp

    model_loss = _model_loss_labels(pos, model, list(targets), families)
    rows = []
    c_sum = np.zeros_like(r)
    for t in targets:
        err = df[f"pred_{model}_{t}"].to_numpy(dtype=np.float64) - df[t].to_numpy(dtype=np.float64)
        c_t = weights[t] * err
        c_sum = c_sum + c_t
        rows.append(
            {
                "target": t,
                "loss": model_loss[t],  # objective THIS model uses for the head
                "nn_loss": _LOSS_PRETTY.get(families[t], families[t]),  # NN-only reference
                "weight": weights[t],
                "contrib_mse": float(np.mean(r * c_t)),  # exact additive share of MSE
                "bias_fp": float(np.mean(c_t)),  # signed FP bias from this head (<0 = under-call)
                "raw_mae": float(np.mean(np.abs(err))),  # raw-stat MAE
                "raw_bias": float(np.mean(err)),  # raw-stat signed bias
            }
        )

    mse = float(np.mean(r**2))
    # Exactness checks: r reconstructs from the parts, and the parts sum to MSE.
    assert np.allclose(c_sum, r, atol=1e-6, rtol=1e-6), "residual != sum of per-head contributions"
    contrib_total = sum(row["contrib_mse"] for row in rows)
    assert np.isclose(contrib_total, mse, atol=1e-4, rtol=1e-5), (
        f"per-head MSE contributions {contrib_total:.6f} != MSE {mse:.6f}"
    )

    out = pd.DataFrame(rows)
    out["share_pct"] = 100.0 * out["contrib_mse"] / mse
    out = out.sort_values("share_pct", ascending=False).reset_index(drop=True)
    out.attrs["mse"] = mse
    out.attrs["rmse"] = float(np.sqrt(mse))
    return out


def _tiers(df: pd.DataFrame, model: str, n_tiers: int = 4) -> pd.DataFrame:
    """Per actual-FP quartile: model RMSE / MAE / bias (reuses compute_stratum_metrics)."""
    tmp = df.copy()
    try:
        tmp["tier"] = pd.qcut(
            tmp["actual_fp"],
            n_tiers,
            labels=[f"Q{i + 1}" for i in range(n_tiers)],
            duplicates="drop",
        )
    except ValueError:
        tmp["tier"] = pd.cut(tmp["actual_fp"], n_tiers, labels=False)
    out = compute_stratum_metrics(tmp, "actual_fp", f"pred_{model}_total", "tier")
    # Add the mean actual FP in each tier for readability.
    means = tmp.groupby("tier", observed=True)["actual_fp"].mean().rename("mean_actual_fp")
    return out.merge(means.reset_index(), on="tier", how="left")


def _culprits(df: pd.DataFrame, model: str, targets, weights, n: int = 15) -> pd.DataFrame:
    """Top-N worst FP residuals with the dominant head (largest |c_t|) attributed."""
    work = df.copy()
    actual_fp = work["actual_fp"].to_numpy(dtype=np.float64)
    pred_fp = work[f"pred_{model}_total"].to_numpy(dtype=np.float64)
    work["resid_fp"] = pred_fp - actual_fp

    contribs = {}
    for t in targets:
        err = work[f"pred_{model}_{t}"].to_numpy(dtype=np.float64) - work[t].to_numpy(
            dtype=np.float64
        )
        contribs[t] = weights[t] * err
    contrib_mat = np.vstack([contribs[t] for t in targets])  # (n_targets, n_rows)
    dom_idx = np.argmax(np.abs(contrib_mat), axis=0)
    work["dom_head"] = [targets[i] for i in dom_idx]
    work["dom_head_fp"] = contrib_mat[dom_idx, np.arange(contrib_mat.shape[1])]

    work["abs_resid"] = work["resid_fp"].abs()
    work = work.sort_values("abs_resid", ascending=False).head(n)

    name_col = _name_col(work)
    player = (work[name_col] if name_col else work["player_id"]).astype(str).values
    out = pd.DataFrame(
        {
            "player": player,
            "wk": work["week"].values if "week" in work.columns else np.nan,
            "actual_fp": work["actual_fp"].round(1).values,
            "pred_fp": work[f"pred_{model}_total"].round(1).values,
            "resid_fp": work["resid_fp"].round(1).values,
            "dom_head": work["dom_head"].values,
            "dom_head_fp": work["dom_head_fp"].round(1).values,
        }
    )
    return out.reset_index(drop=True)


def _signal_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """corr/slope/rmse + the best RMSE achievable by affine-rescaling ``y_pred``.

    ``recal_rmse = std(y_true) * sqrt(1 - corr^2)`` is the floor any monotone-linear
    recalibration of ``y_pred`` can reach. Comparing one predictor's ``recal_rmse`` to a
    rival's raw ``rmse`` separates a *calibration* gap (closable by a rescale) from an
    *information* gap (only a higher ``corr`` — i.e. genuinely more signal — closes it).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = int(y_true.size)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if n else float("nan")
    if n < 3 or np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return {
            "n": n,
            "corr": float("nan"),
            "slope": float("nan"),
            "rmse": rmse,
            "recal_rmse": rmse,
        }
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    slope = float(np.cov(y_pred, y_true, bias=True)[0, 1] / np.var(y_pred))
    recal_rmse = float(np.std(y_true) * np.sqrt(max(0.0, 1.0 - corr**2)))
    return {"n": n, "corr": corr, "slope": slope, "rmse": rmse, "recal_rmse": recal_rmse}


def _print_position(pos: str, train, val, test) -> None:
    df, targets, available = _prepare(pos, train, val, test)
    n_rows = len(df)
    n_players = df["player_id"].nunique()
    print("\n" + "=" * 78)
    print(f"  {pos}  —  top-30 slice: {n_rows} player-weeks, {n_players} players (2025 test)")
    print("=" * 78)

    if not available:
        print("  No model artifacts loaded — nothing to decompose.")
        return

    # Overall model metrics on the slice (cross-check vs the Comparison tab) + experts.
    print("\n  Overall fantasy-point metrics on this slice")
    print("  " + "-" * 60)
    overall = []
    for m in available:
        met = compute_metrics(df["actual_fp"].to_numpy(), df[f"pred_{m}_total"].to_numpy())
        overall.append(
            {"source": f"model:{m}", "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"]}
        )
    experts = _expert_top30_aggregates(pos)
    for ename, eblock in experts.items():
        overall.append(
            {
                "source": f"expert:{ename}",
                "mae": eblock.get("mae"),
                "rmse": eblock.get("rmse"),
                "r2": eblock.get("r2"),
            }
        )
    ov = pd.DataFrame(overall)
    best_model = ov[ov["source"].str.startswith("model:")].sort_values("rmse").iloc[0]["source"]
    print(ov.round(4).to_string(index=False))
    headline = best_model.split(":", 1)[1]
    print(f"\n  Lowest-RMSE model on this slice: {headline}")

    weights = _weights(pos)
    families = _loss_families(pos)

    # --- Headline: per-head MSE decomposition for the best model ----------------
    dec = _decompose(df, pos, headline, targets, weights, families)
    print(
        f"\n  Per-head decomposition of FP MSE — model: {headline}  "
        f"(RMSE {dec.attrs['rmse']:.3f}, MSE {dec.attrs['mse']:.2f})"
    )
    print(
        f"  loss = the objective {headline} uses for this head;  nn_loss = the NN's per-head family "
        "(NN-only — LGBM/Ridge use one objective for all heads)"
    )
    print(
        "  share_pct = exact share of FP MSE;  bias_fp = mean FP error from this head (<0 = under-call)"
    )
    print("  " + "-" * 72)
    print(
        dec[
            [
                "target",
                "loss",
                "nn_loss",
                "weight",
                "share_pct",
                "contrib_mse",
                "bias_fp",
                "raw_mae",
                "raw_bias",
            ]
        ]
        .round(
            {
                "weight": 2,
                "share_pct": 1,
                "contrib_mse": 2,
                "bias_fp": 2,
                "raw_mae": 2,
                "raw_bias": 3,
            }
        )
        .to_string(index=False)
    )
    # Count/hurdle vs MSE-yards rollup.
    is_yards = dec["target"].str.contains("yards")
    yards_share = dec.loc[is_yards, "share_pct"].sum()
    count_share = dec.loc[~is_yards, "share_pct"].sum()
    print(
        f"\n  → MSE-yards heads: {yards_share:.1f}% of MSE   |   "
        f"count/hurdle heads (TDs+receptions+fumbles): {count_share:.1f}% of MSE"
    )

    # Same decomposition, all models, share_pct only (does the pattern hold across models?).
    if len(available) > 1:
        print("\n  Per-head MSE share (%) across all loaded models")
        print("  " + "-" * 60)
        share_tbl = None
        for m in available:
            d = _decompose(df, pos, m, targets, weights, families)[["target", "share_pct"]]
            d = d.rename(columns={"share_pct": m}).set_index("target")
            share_tbl = d if share_tbl is None else share_tbl.join(d)
        share_tbl = share_tbl.reindex(targets)
        print(share_tbl.round(1).to_string())

    # --- Tiers: where does the error concentrate? ------------------------------
    print(f"\n  Per actual-FP quartile — model: {headline}")
    print("  " + "-" * 60)
    tiers = _tiers(df, headline)
    print(tiers.round({"mae": 3, "rmse": 3, "bias": 3, "mean_actual_fp": 1}).to_string(index=False))

    # --- Culprits: which players/weeks? ----------------------------------------
    print(
        f"\n  Worst 15 FP residuals — model: {headline}  (dom_head = largest single-head FP miss)"
    )
    print("  " + "-" * 60)
    culprits = _culprits(df, headline, targets, weights, n=15)
    print(culprits.to_string(index=False))


def _with_experts(positions, train, val, test) -> None:
    """Best-effort live per-tier model-vs-expert RMSE (network). Skipped on any error."""
    print("\n" + "#" * 78)
    print("  --with-experts: live per-tier model-vs-expert (network)")
    print("#" * 78)
    try:
        from src.analysis.analysis_expert_comparison import _project_sleeper_to_ppr
        from src.analysis.analysis_nflcom_baseline import _project_nflcom_to_ppr
        from src.analysis.sleeper_loader import load_sleeper_with_gsis_id
        from src.config import TEST_SEASONS
        from src.data.nflcom_loader import load_nflcom_with_gsis_id

        nflcom_full = load_nflcom_with_gsis_id(seasons=TEST_SEASONS)
        sleeper_full = load_sleeper_with_gsis_id(TEST_SEASONS)
    except Exception as e:  # noqa: BLE001 - network/import boundary, degrade gracefully
        print(f"  skipped — could not load expert sources: {e!r}")
        return

    for pos in positions:
        try:
            df, targets, available = _prepare(pos, train, val, test)
            if not available:
                continue
            headline = (
                pd.DataFrame(
                    [
                        {
                            "m": m,
                            "rmse": compute_metrics(
                                df["actual_fp"].to_numpy(), df[f"pred_{m}_total"].to_numpy()
                            )["rmse"],
                        }
                        for m in available
                    ]
                )
                .sort_values("rmse")
                .iloc[0]["m"]
            )
            keys = ["player_id", "season", "week"]
            base = df[[*keys, "actual_fp", f"pred_{headline}_total"]].copy()
            base["player_id"] = base["player_id"].astype(str)

            nfl = _project_nflcom_to_ppr(nflcom_full, pos, "ppr")
            rw = _project_sleeper_to_ppr(sleeper_full, pos, "ppr")
            for proj, col, label in (
                (nfl, "nflcom_pred_total", "nflcom"),
                (rw, "expert_pred_total", "rotowire"),
            ):
                if proj is None or col not in getattr(proj, "columns", []):
                    continue
                p = proj[[*keys, col]].copy()
                p["player_id"] = p["player_id"].astype(str)
                merged = base.merge(p, on=keys, how="inner")
                if merged.empty:
                    continue
                merged["tier"] = pd.qcut(
                    merged["actual_fp"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
                )
                m_t = compute_stratum_metrics(merged, "actual_fp", f"pred_{headline}_total", "tier")
                e_t = compute_stratum_metrics(merged, "actual_fp", col, "tier")
                cmp = (
                    m_t[["tier", "n", "rmse"]]
                    .rename(columns={"rmse": f"model_{headline}_rmse"})
                    .merge(
                        e_t[["tier", "rmse"]].rename(columns={"rmse": f"{label}_rmse"}), on="tier"
                    )
                )
                print(
                    f"\n  {pos} per-tier RMSE — model:{headline} vs expert:{label}  (n={len(merged)})"
                )
                print(cmp.round(3).to_string(index=False))

                # Signal/calibration: corr sets the best RMSE any rescale can reach.
                # model recal_rmse <= expert rmse  => gap is calibration (rescale closes it).
                # expert corr > model corr         => gap is information (need more signal).
                q4 = merged[merged["tier"] == "Q4"]
                sig_rows = []
                for scope_name, scope in (("overall", merged), ("Q4", q4)):
                    ms = _signal_stats(
                        scope["actual_fp"].to_numpy(), scope[f"pred_{headline}_total"].to_numpy()
                    )
                    es = _signal_stats(scope["actual_fp"].to_numpy(), scope[col].to_numpy())
                    sig_rows.append({"scope": scope_name, "who": f"model:{headline}", **ms})
                    sig_rows.append({"scope": scope_name, "who": f"expert:{label}", **es})
                sig = pd.DataFrame(sig_rows)[
                    ["scope", "who", "n", "corr", "slope", "rmse", "recal_rmse"]
                ]
                print(
                    f"\n  {pos} signal/calibration vs {label} — corr sets the rescale floor (recal_rmse)"
                )
                print(
                    sig.round({"corr": 3, "slope": 2, "rmse": 2, "recal_rmse": 2}).to_string(
                        index=False
                    )
                )
        except Exception as e:  # noqa: BLE001
            print(f"  {pos}: expert comparison skipped — {e!r}")


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="*", default=["RB", "WR"])
    parser.add_argument(
        "--with-experts",
        action="store_true",
        help="Also run the live per-tier model-vs-expert block (network; best-effort).",
    )
    args = parser.parse_args(argv)

    positions = [p.upper() for p in args.positions]
    train, val, test = _load_splits()

    for pos in positions:
        if pos not in POSITION_TARGET_MAP:
            print(f"\n[skip] {pos}: not a skill position with a per-target FP map.")
            continue
        _print_position(pos, train, val, test)

    if args.with_experts:
        skill = [p for p in positions if p in POSITION_TARGET_MAP]
        _with_experts(skill, train, val, test)


if __name__ == "__main__":
    _main()
