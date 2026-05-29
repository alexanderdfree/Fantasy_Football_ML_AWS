"""Paired (block) bootstrap significance CIs for model comparison on the test season.

The headline benchmark compares Ridge / Neural Net / Attention NN / LightGBM / baselines
on a **single** test season (2025). A 0.02 MAE gap between two models can be pure season
noise. This module resamples the test season's weeks (a *block* bootstrap) to put a 95%
confidence interval and a bootstrap p-value on each pairwise gap, so "Attention NN beats
Ridge by 0.15 MAE" becomes "… by 0.15 MAE, 95% CI [0.04, 0.27], p=0.01" — or honestly
"within noise."

This file also hosts the lower-level **paired-error primitives** —
:func:`diebold_mariano_test` and :func:`paired_bootstrap_metric_ci` — that
``analysis_expert_comparison.py`` uses to compare the model against an expert
baseline (NFL.com). Both layers share one home for the "is this forecast-accuracy
gap real?" question; see the Layer 1 / Layer 2 banners below.

**Scope — within-season sampling noise only.** This cannot estimate season-to-season
variance (there is one test season). For that, use the rolling-origin evaluation
(``src/data/split.py::rolling_origin_folds`` + ``benchmark.py --rolling-origin``). The two
are complementary: this is the cheap down-payment, rolling-origin is the real
generalization variance.

**Resampling unit = WEEK** (block bootstrap over the ~18 weeks of the test season), not
i.i.d. rows. Rows within a week share opponent / weather / game-script, so their model
errors are correlated; resampling rows i.i.d. would understate variance and produce
anti-conservative CIs. Top-k hit rate is a within-week rank statistic, so the week block
is also the only unit that can bootstrap it. With ~18 blocks the CIs are deliberately wide
— that is the point: one season is little evidence about a model-vs-model gap. A secondary
``--unit player`` mode gives an MAE-only cross-check with more blocks (tighter CIs, ignores
within-week correlation, no hit-rate).

Usage::

    python -m src.analysis.significance [POS ...] [--n-boot 2000] [--unit week|player]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOP_K_RANKING  # noqa: E402
from src.shared.position import Position  # noqa: E402
from src.shared.utils import seed_everything  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis_output"

# Canonical model-name -> prediction-column map written onto ``pos_test`` by
# ``src/shared/pipeline.py`` (see ``backtest_pred_columns`` there). Only the
# subset whose column is actually present is used — ElasticNet / Attention NN /
# LightGBM are conditional on the position's config. ``LastWeekBaseline`` is NOT
# wired into the pipeline, so only ``Season Avg`` appears as a baseline.
CANONICAL_PRED_COLUMNS: dict[str, str] = {
    "Season Avg": "pred_baseline",
    "Ridge": "pred_ridge_total",
    "Neural Net": "pred_nn_total",
    "ElasticNet": "pred_enet_total",
    "Attention NN": "pred_attn_nn_total",
    "LightGBM": "pred_lgbm_total",
}
DEFAULT_BASELINES: tuple[str, ...] = ("Season Avg",)


# ===========================================================================
# Layer 1 — low-level paired-error primitives (model-vs-EXPERT).
# Pure NumPy/SciPy, no model/data deps. Consumed by
# ``analysis_expert_comparison.py`` to decide whether a model-vs-NFL.com gap is
# real. ``paired_bootstrap`` below (Layer 2) is the model-vs-MODEL test-season
# tool that fuses an MAE block-bootstrap with the within-week top-k pass; it is
# kept separate because the top-k hit-rate is a rank statistic these row-level
# error primitives can't express, but its MAE arm is the same idea as
# ``paired_bootstrap_metric_ci(..., groups=week)``.
# ===========================================================================
_METRIC_FNS = {
    "mae": lambda e: float(np.mean(np.abs(e))),
    "rmse": lambda e: float(np.sqrt(np.mean(np.square(e)))),
}


def _as_paired_errors(errors_a, errors_b) -> tuple[np.ndarray, np.ndarray]:
    """Coerce two error vectors to 1-D float arrays, validating shape + size."""
    e_a = np.asarray(errors_a, dtype=float).ravel()
    e_b = np.asarray(errors_b, dtype=float).ravel()
    if e_a.shape != e_b.shape:
        raise ValueError(
            f"errors_a and errors_b must have the same length; got {e_a.shape} vs {e_b.shape}"
        )
    if e_a.size < 2:
        raise ValueError("need at least 2 paired observations")
    return e_a, e_b


def _autocov(x: np.ndarray, k: int) -> float:
    """Population (1/n) k-th autocovariance of ``x`` about its own mean."""
    n = x.size
    xbar = x.mean()
    return float(np.sum((x[k:] - xbar) * (x[: n - k] - xbar)) / n)


def diebold_mariano_test(errors_a, errors_b, *, power: int = 1, h: int = 1) -> dict:
    """Paired Diebold-Mariano test of equal predictive accuracy (HLN-corrected).

    Tests H0: forecaster A and forecaster B have equal expected loss, on the
    *paired* forecast errors. The loss is ``L(e) = |e|**power`` — ``power=1``
    matches an MAE headline, ``power=2`` matches RMSE.

    Args:
        errors_a: forecast errors of model A (``pred - actual``).
        errors_b: forecast errors of model B, same observations / order.
        power: loss exponent (1 = absolute-error loss, 2 = squared-error loss).
        h: forecast horizon. For independent observations (player-weeks are not
           a multi-step horizon) leave at 1; ``h>1`` adds autocovariance terms
           to the long-run variance.

    Returns:
        ``{"dm_stat", "p_value", "n", "mean_loss_diff", "favored"}`` where the
        loss differential is ``d = L(e_a) - L(e_b)`` (negative ⇒ A more
        accurate), ``p_value`` is two-sided from a Student-t with ``n-1`` df, and
        ``favored`` is ``"model"`` (A), ``"expert"`` (B), or ``"tie"``.

    Reference: Diebold & Mariano (1995), *JBES* 13(3); Harvey, Leybourne &
    Newbold (1997), *IJF* 13(2) (small-sample correction).
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    e_a, e_b = _as_paired_errors(errors_a, errors_b)
    n = e_a.size

    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    dbar = float(d.mean())

    gamma0 = _autocov(d, 0)
    if gamma0 <= 0.0:
        # Zero-variance loss differential: identical losses (tie) or a constant
        # nonzero gap (degenerate-decisive). Continuous errors never hit this;
        # guard so we don't divide by zero.
        if abs(dbar) < 1e-12:
            return {"dm_stat": 0.0, "p_value": 1.0, "n": n, "mean_loss_diff": 0.0, "favored": "tie"}
        return {
            "dm_stat": float("inf") if dbar > 0 else float("-inf"),
            "p_value": 0.0,
            "n": n,
            "mean_loss_diff": dbar,
            "favored": "model" if dbar < 0 else "expert",
        }

    # Long-run variance of the mean differential (autocovariances up to h-1).
    lrv = gamma0 + 2.0 * sum(_autocov(d, k) for k in range(1, h))
    var_mean = lrv / n
    if var_mean <= 0.0:
        # Negative LRV estimate (possible for h>1 with strong negative
        # autocorrelation). Fall back to the h=1 variance, which is >0 here.
        var_mean = gamma0 / n

    dm = dbar / np.sqrt(var_mean)
    # Harvey-Leybourne-Newbold small-sample correction + Student-t reference.
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = float(dm * hln_factor)
    p_value = float(2.0 * stats.t.sf(abs(dm_star), df=n - 1))

    return {
        "dm_stat": dm_star,
        "p_value": p_value,
        "n": n,
        "mean_loss_diff": dbar,
        "favored": "model" if dbar < 0 else "expert",
    }


def paired_bootstrap_metric_ci(
    errors_a,
    errors_b,
    *,
    metric: str = "mae",
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for ``metric(A) - metric(B)`` on paired errors.

    Resamples the paired observations with replacement ``n_boot`` times and
    recomputes the metric difference each time. When ``groups`` is supplied
    (e.g. ``player_id`` or ``week``), resampling is **clustered**: whole groups
    are drawn with replacement, so correlated observations move together and the
    interval isn't artificially tight.

    Args:
        errors_a: forecast errors of model A (``pred - actual``).
        errors_b: forecast errors of model B, same observations / order.
        metric: ``"mae"`` or ``"rmse"``.
        groups: optional cluster id per observation (len == n). ``None`` ⇒ i.i.d.
            row resampling.
        n_boot: number of bootstrap replicates.
        seed: PRNG seed (``np.random.default_rng``) for reproducibility.
        ci: central interval mass (0.95 ⇒ 2.5/97.5 percentiles).

    Returns:
        ``{"delta", "lo", "hi", "p_value", "metric", "n_boot"}``. ``delta`` is the
        observed metric difference (negative ⇒ A better); ``[lo, hi]`` excluding 0
        ⇒ the difference is significant at the ``1-ci`` level. ``p_value`` is the
        two-sided bootstrap proportion.
    """
    e_a, e_b = _as_paired_errors(errors_a, errors_b)
    n = e_a.size
    metric = metric.lower()
    if metric not in _METRIC_FNS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_FNS)}; got {metric!r}")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    mfn = _METRIC_FNS[metric]
    observed = mfn(e_a) - mfn(e_b)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)

    if groups is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            deltas[b] = mfn(e_a[idx]) - mfn(e_b[idx])
    else:
        groups_arr = np.asarray(groups).ravel()
        if groups_arr.shape[0] != n:
            raise ValueError(f"groups must have length {n}; got {groups_arr.shape[0]}")
        uniq = np.unique(groups_arr)
        idx_by_group = [np.flatnonzero(groups_arr == g) for g in uniq]
        n_groups = len(idx_by_group)
        for b in range(n_boot):
            chosen = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([idx_by_group[c] for c in chosen])
            deltas[b] = mfn(e_a[idx]) - mfn(e_b[idx])

    alpha = 1.0 - ci
    lo = float(np.percentile(deltas, 100.0 * alpha / 2.0))
    hi = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    # Two-sided bootstrap p: twice the smaller tail mass at 0.
    p_value = float(min(1.0, 2.0 * min(np.mean(deltas >= 0.0), np.mean(deltas <= 0.0))))

    return {
        "delta": float(observed),
        "lo": lo,
        "hi": hi,
        "p_value": p_value,
        "metric": metric,
        "n_boot": int(n_boot),
    }


# ===========================================================================
# Layer 2 — high-level model-vs-MODEL comparison on the test season.
# ===========================================================================


def _block_indices(test_df: pd.DataFrame, unit: str) -> list[np.ndarray]:
    """Return one positional-index array per resampling block.

    ``unit="week"`` blocks by ``week`` (the default — clusters correlated
    within-slate rows); ``unit="player"`` blocks by ``player_id`` (MAE-only
    cross-check). The arrays index into ``test_df.reset_index(drop=True)``.
    """
    if unit not in ("week", "player"):
        raise ValueError(f"unit must be 'week' or 'player', got {unit!r}")
    key = "week" if unit == "week" else "player_id"
    if key not in test_df.columns:
        raise ValueError(f"test_df is missing the '{key}' column required for unit={unit!r}")
    df = test_df.reset_index(drop=True)
    # GroupBy.indices maps each group label -> ndarray of positional row indices.
    return list(df.groupby(key).indices.values())


def _per_block_stats(
    test_df: pd.DataFrame,
    blocks: list[np.ndarray],
    pred_col: str,
    true_col: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block (sum_abs_error, n_rows, top_k_hit_rate) for one model.

    ``hit_rate`` is NaN for blocks with fewer than ``top_k`` rows (mirrors
    ``src/shared/backtest.py``, which skips those weeks). Block stats are
    precomputed once so each bootstrap resample is a cheap gather-and-sum.
    """
    df = test_df.reset_index(drop=True)
    true = df[true_col].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(pred - true)

    sae = np.empty(len(blocks))
    n = np.empty(len(blocks))
    hr = np.empty(len(blocks))
    for i, idx in enumerate(blocks):
        sae[i] = abs_err[idx].sum()
        n[i] = idx.size
        if idx.size >= top_k:
            t = true[idx]
            p = pred[idx]
            actual_top = set(np.argsort(-t)[:top_k])
            pred_top = set(np.argsort(-p)[:top_k])
            hr[i] = len(actual_top & pred_top) / top_k
        else:
            hr[i] = np.nan
    return sae, n, hr


def _bootstrap_metric(
    metric: str,
    boot_idx: np.ndarray,
    sae: np.ndarray,
    n: np.ndarray,
    hr: np.ndarray,
) -> np.ndarray:
    """Vectorised per-resample metric value over a ``(n_boot, B)`` block-index matrix."""
    if metric == "mae":
        return sae[boot_idx].sum(axis=1) / n[boot_idx].sum(axis=1)
    # top_k_hit_rate: weight each drawn block by its resample multiplicity,
    # ignoring NaN (too-small) blocks. den==0 -> NaN (resample had no scorable week).
    valid = (~np.isnan(hr)).astype(float)
    hr0 = np.where(np.isnan(hr), 0.0, hr)
    num = hr0[boot_idx].sum(axis=1)
    den = valid[boot_idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _point_metric(metric: str, sae: np.ndarray, n: np.ndarray, hr: np.ndarray) -> float:
    if metric == "mae":
        total_n = n.sum()
        return float(sae.sum() / total_n) if total_n > 0 else float("nan")
    return float(np.nanmean(hr)) if np.any(~np.isnan(hr)) else float("nan")


def paired_bootstrap(
    test_df: pd.DataFrame,
    pred_columns: dict[str, str],
    *,
    reference: str = "Ridge",
    true_col: str = "fantasy_points",
    n_boot: int = 2000,
    seed: int = 42,
    unit: str = "week",
    top_k: int = TOP_K_RANKING,
    metrics: tuple[str, ...] = ("mae", "top_k_hit_rate"),
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
) -> dict:
    """Paired block-bootstrap CIs + p-values for model-vs-model gaps on one test season.

    Compares every model against ``reference`` (default Ridge) and against each present
    baseline. For MAE the gap is ``mae(other) - mae(model)`` (positive = model better); for
    hit rate it is ``hr(model) - hr(other)`` (positive = model better). The p-value is a
    two-sided bootstrap sign-flip probability, floored to ``1/n_boot`` (never exactly 0).

    Returns a JSON-serialisable dict; see ``format_significance_table`` for a printable view.
    """
    if reference not in pred_columns:
        raise ValueError(
            f"reference model {reference!r} not in pred_columns {sorted(pred_columns)}"
        )
    if unit == "player":
        metrics = tuple(m for m in metrics if m == "mae") or ("mae",)

    seed_everything(seed)
    rng = np.random.default_rng(seed)

    blocks = _block_indices(test_df, unit)
    n_blocks = len(blocks)
    low_power = n_blocks < 3
    boot_idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))

    # Precompute per-block stats once per model.
    stats = {
        name: _per_block_stats(test_df, blocks, col, true_col, top_k)
        for name, col in pred_columns.items()
    }
    models = list(pred_columns)

    point_metrics: dict[str, dict[str, float]] = {}
    boot_cache: dict[tuple[str, str], np.ndarray] = {}
    for name in models:
        sae, n, hr = stats[name]
        point_metrics[name] = {m: _point_metric(m, sae, n, hr) for m in metrics}
        for m in metrics:
            boot_cache[(name, m)] = _bootstrap_metric(m, boot_idx, sae, n, hr)

    def _compare(model: str, other: str, metric: str) -> dict:
        pm, po = point_metrics[model][metric], point_metrics[other][metric]
        if metric == "mae":
            delta_hat = po - pm  # positive => model has lower MAE => better
            delta_boot = boot_cache[(other, metric)] - boot_cache[(model, metric)]
        else:
            delta_hat = pm - po  # positive => model has higher hit rate => better
            delta_boot = boot_cache[(model, metric)] - boot_cache[(other, metric)]
        finite = delta_boot[np.isfinite(delta_boot)]
        if finite.size == 0:
            ci_lo = ci_hi = p_value = float("nan")
            significant = False
        else:
            ci_lo, ci_hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
            p_raw = 2.0 * min(np.mean(finite <= 0), np.mean(finite >= 0))
            p_value = float(min(max(p_raw, 1.0 / n_boot), 1.0))
            significant = (ci_lo > 0) or (ci_hi < 0)
        return {
            "model": model,
            "other": other,
            "metric": metric,
            "delta": float(delta_hat),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p_value,
            "significant": bool(significant),
            "model_better": bool(delta_hat > 0),
            "verdict": _verdict(
                model, other, metric, delta_hat, ci_lo, ci_hi, p_value, significant
            ),
        }

    pairs: list[dict] = []
    for metric in metrics:
        for model in models:
            if model != reference and model not in baselines:
                pairs.append(_compare(model, reference, metric))
        for base in baselines:
            if base not in pred_columns:
                continue
            for model in models:
                if model != base and model != reference:
                    pairs.append(_compare(model, base, metric))

    return {
        "scope": "within_season_sampling_noise",
        "unit": unit,
        "n_boot": n_boot,
        "seed": seed,
        "reference": reference,
        "baselines": [b for b in baselines if b in pred_columns],
        "top_k": top_k,
        "n_blocks": n_blocks,
        "n_rows": int(len(test_df)),
        "low_power": bool(low_power),
        "models": models,
        "point_metrics": point_metrics,
        "pairs": pairs,
    }


def _verdict(
    model: str,
    other: str,
    metric: str,
    delta: float,
    lo: float,
    hi: float,
    p: float,
    significant: bool,
) -> str:
    unit_word = "MAE" if metric == "mae" else "top-k hit rate"
    if not np.isfinite(delta):
        return f"{model} vs {other}: undetermined ({unit_word})"
    if significant:
        verb = "beats" if delta > 0 else "loses to"
        return (
            f"{model} {verb} {other} by {abs(delta):.3f} {unit_word}, "
            f"95% CI [{lo:.3f}, {hi:.3f}], p={p:.3f}"
        )
    return (
        f"{model} vs {other}: within noise ({unit_word} Δ={delta:.3f}, "
        f"CI [{lo:.3f}, {hi:.3f}], p={p:.3f})"
    )


def format_significance_table(result: dict) -> str:
    """Render a human-readable summary of a ``paired_bootstrap`` result."""
    lines = [
        f"Significance ({result['unit']} block bootstrap, n_boot={result['n_boot']}, "
        f"seed={result['seed']}) — scope: {result['scope']}",
        f"  blocks={result['n_blocks']}  rows={result['n_rows']}  reference={result['reference']}",
    ]
    if result["low_power"]:
        lines.append("  WARNING: <3 blocks — bootstrap is low-power; treat CIs as indicative only.")
    by_metric: dict[str, list[dict]] = {}
    for pr in result["pairs"]:
        by_metric.setdefault(pr["metric"], []).append(pr)
    for metric, prs in by_metric.items():
        lines.append(f"\n  [{metric}]")
        for pr in prs:
            mark = "*" if pr["significant"] else " "
            lines.append(f"   {mark} {pr['verdict']}")
    return "\n".join(lines)


def compact_significance(result: dict) -> dict | None:
    """Compact best-model-vs-reference + best-vs-baseline MAE gaps for benchmark fold-in.

    Picks the model with the lowest point MAE (excluding baselines) and reports its MAE gap
    vs the reference and vs the first baseline. JSON-native floats only. Returns None if no
    MAE pairs were computed.
    """
    mae_point = {
        m: v["mae"]
        for m, v in result["point_metrics"].items()
        if m not in result["baselines"] and np.isfinite(v.get("mae", float("nan")))
    }
    if not mae_point:
        return None
    best = min(mae_point, key=mae_point.get)

    def _gap(other: str) -> dict | None:
        for pr in result["pairs"]:
            if pr["metric"] == "mae" and pr["model"] == best and pr["other"] == other:
                return {
                    "model": best,
                    "other": other,
                    "mae_delta": float(pr["delta"]),
                    "ci_lo": float(pr["ci_lo"]),
                    "ci_hi": float(pr["ci_hi"]),
                    "p_value": float(pr["p_value"]),
                    "significant": bool(pr["significant"]),
                }
        return None

    out: dict = {"unit": result["unit"], "n_boot": result["n_boot"], "best_model": best}
    vs_ref = _gap(result["reference"])
    if vs_ref is not None:
        out["best_vs_reference"] = vs_ref
    if result["baselines"]:
        vs_base = _gap(result["baselines"][0])
        if vs_base is not None:
            out["best_vs_baseline"] = vs_base
    return out


def pred_columns_from_test_df(test_df: pd.DataFrame) -> dict[str, str]:
    """Reconstruct the model-name -> column map from whichever ``pred_*`` columns landed."""
    return {name: col for name, col in CANONICAL_PRED_COLUMNS.items() if col in test_df.columns}


def run_for_position(
    position: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    unit: str = "week",
    reference: str = "Ridge",
) -> dict:
    """Re-run ``position``'s pipeline in-process and bootstrap its test predictions.

    The pipeline attaches every model's per-row test predictions to ``result["test_df"]``,
    so a single run yields all the columns the bootstrap needs — no separate persistence.
    Requires ``data/splits/*.parquet`` (and trains models, so this is slower than the other
    CLIs).
    """
    from src.shared.registry import get_runner

    seed_everything(seed)
    result = get_runner(position)()
    if "test_df" not in result:
        raise KeyError(
            f"{position} pipeline result has no 'test_df'; cannot bootstrap predictions."
        )
    test_df = result["test_df"]
    pred_columns = pred_columns_from_test_df(test_df)
    if reference not in pred_columns:
        # Fall back to the first non-baseline model if Ridge somehow absent.
        non_base = [m for m in pred_columns if m not in DEFAULT_BASELINES]
        if not non_base:
            raise ValueError(f"{position}: no non-baseline model predictions found.")
        reference = non_base[0]
    out = paired_bootstrap(
        test_df, pred_columns, reference=reference, n_boot=n_boot, seed=seed, unit=unit
    )
    out["position"] = position
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "positions",
        nargs="*",
        default=Position.values(),
        help="Positions to analyse (default: all six)",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unit",
        choices=["week", "player"],
        default="week",
        help="Bootstrap block unit (default: week)",
    )
    parser.add_argument("--reference", default="Ridge", help="Reference model for gaps")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(exist_ok=True)
    for pos in args.positions:
        print(f"\n{'#' * 60}\n# SIGNIFICANCE {pos}\n{'#' * 60}")
        result = run_for_position(
            pos, n_boot=args.n_boot, seed=args.seed, unit=args.unit, reference=args.reference
        )
        print(format_significance_table(result))
        out_path = OUT_DIR / f"significance_{pos}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n✔ wrote {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
