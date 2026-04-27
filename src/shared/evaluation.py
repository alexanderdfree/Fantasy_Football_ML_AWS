"""Generic position evaluation utilities."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.shared.aggregate_targets import (
    POSITION_TARGET_MAP,
    TARGET_UNITS,
    predictions_to_fantasy_points,
)


def _infer_position(target_names: list[str]) -> str | None:
    """Return the first position whose target map fully matches ``target_names``."""
    name_set = set(target_names)
    for pos, tmap in POSITION_TARGET_MAP.items():
        if set(tmap.keys()) == name_set:
            return pos
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid for evaluating gate logits."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _gate_metrics(y_true: np.ndarray, gate_logit: np.ndarray, value_mu: np.ndarray) -> dict:
    """Per-head gate diagnostics for a gated (hurdle) target.

    Returns:
        {
          "gate_auc": float | None,       # None if only one class in y>0
          "gate_brier": float,            # mean squared error of sigmoid(logit) vs y>0
          "positive_rate": float,         # empirical P(y > 0)
          "predicted_positive_rate": float,  # mean sigmoid(logit)
          "cond_mae": float | None,       # MAE of value_mu vs y on y>0 (None if none)
          "cond_rmse": float | None,
        }
    """
    y_pos = (y_true > 0).astype(np.float64)
    prob = _sigmoid(gate_logit)
    positive_rate = float(y_pos.mean())
    predicted_positive_rate = float(prob.mean())
    brier = float(np.mean((prob - y_pos) ** 2))

    # AUC undefined when one class is absent; leave as None rather than NaN so
    # downstream print_comparison_table can short-circuit cleanly.
    if y_pos.min() == y_pos.max():
        gate_auc = None
    else:
        from sklearn.metrics import roc_auc_score

        gate_auc = float(roc_auc_score(y_pos, prob))

    pos_mask = y_true > 0
    if pos_mask.any():
        cond_err = value_mu[pos_mask] - y_true[pos_mask]
        cond_mae = float(np.mean(np.abs(cond_err)))
        cond_rmse = float(np.sqrt(np.mean(cond_err**2)))
    else:
        cond_mae = None
        cond_rmse = None

    return {
        "gate_auc": gate_auc,
        "gate_brier": brier,
        "positive_rate": positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "cond_mae": cond_mae,
        "cond_rmse": cond_rmse,
    }


def compute_target_metrics(
    y_true_dict: dict,
    y_pred_dict: dict,
    target_names: list[str],
    gate_info: dict | None = None,
) -> dict:
    """Compute per-target and total metrics.

    Each entry gains a ``"unit"`` field (``"yds"``, ``"TDs"``, ``"pts"``, etc.)
    pulled from ``TARGET_UNITS``. The ``"total"`` entry is computed on
    fantasy-points aggregation via the per-position aggregator when the
    position is recognized; otherwise it falls back to the plain sum of
    per-target values.

    When ``gate_info`` is supplied, gated targets additionally report:
    ``gate_auc``, ``gate_brier``, ``positive_rate``, ``predicted_positive_rate``,
    ``cond_mae``, ``cond_rmse`` — a separate view into the gate classifier and
    the conditional value head. Only meaningful for attention-NN runs; Ridge
    and LightGBM produce no gate outputs so their callers pass ``None``.

    ``gate_info`` shape: ``{target: {"gate_logit": ndarray, "value_mu": ndarray}}``.

    Returns:
        {"total": {mae, rmse, r2, unit}, "target_1": {mae, rmse, r2, unit, ...}, ...}
    """
    results = {}

    position = _infer_position(target_names)
    if position is not None:
        true_pts = predictions_to_fantasy_points(position, y_true_dict, "ppr")
        pred_pts = predictions_to_fantasy_points(position, y_pred_dict, "ppr")
    else:
        true_pts = np.sum([y_true_dict[t] for t in target_names], axis=0)
        pred_pts = np.sum([y_pred_dict[t] for t in target_names], axis=0)
    results["total"] = compute_metrics(true_pts, pred_pts)
    results["total"]["unit"] = "pts"

    for target in target_names:
        metrics = compute_metrics(y_true_dict[target], y_pred_dict[target])
        metrics["unit"] = TARGET_UNITS.get(target, "pts")
        if gate_info and target in gate_info:
            info = gate_info[target]
            metrics.update(
                _gate_metrics(
                    np.asarray(y_true_dict[target]),
                    np.asarray(info["gate_logit"]),
                    np.asarray(info["value_mu"]),
                )
            )
        results[target] = metrics
    return results


def build_gate_info(preds: dict, gated_targets: list[str]) -> dict | None:
    """Extract gate_logit + value_mu from a preds dict for gated targets.

    Returns ``None`` when no gated targets are supplied (lets callers pass the
    return straight to ``compute_target_metrics(..., gate_info=...)`` without
    wrapping). Missing keys are skipped silently — Ridge / LightGBM / base-NN
    paths emit no gate outputs.
    """
    if not gated_targets:
        return None
    info = {}
    for t in gated_targets:
        gl = preds.get(f"{t}_gate_logit")
        mu = preds.get(f"{t}_value_mu")
        if gl is not None and mu is not None:
            info[t] = {"gate_logit": np.asarray(gl), "value_mu": np.asarray(mu)}
    return info or None


def compute_fantasy_points_mae(
    pos: str,
    y_true_dict: dict,
    y_pred_dict: dict,
    scoring_format: str = "ppr",
) -> float:
    """MAE in fantasy points after aggregating raw-stat predictions via the
    per-position aggregator. Lets callers compare models across positions on the
    same scale even when target sets differ."""
    true_pts = predictions_to_fantasy_points(pos, y_true_dict, scoring_format)
    pred_pts = predictions_to_fantasy_points(pos, y_pred_dict, scoring_format)
    return float(np.mean(np.abs(pred_pts - true_pts)))


def compute_ranking_metrics(
    test_df: pd.DataFrame,
    pred_col: str = "pred_total",
    true_col: str = "fantasy_points",
    top_k: int = 12,
) -> dict:
    """Per-week ranking quality metrics.

    Returns:
        {
            "weekly": [{"week", "top_k_hit_rate", "spearman"}, ...],
            "season_avg_hit_rate": float,
            "season_avg_spearman": float,
        }
    """
    from scipy.stats import spearmanr

    weekly_results = []
    for week in sorted(test_df["week"].unique()):
        week_df = test_df[test_df["week"] == week]
        if len(week_df) < top_k:
            continue

        actual_top_k = set(week_df.nlargest(top_k, true_col)["player_id"])
        pred_top_k = set(week_df.nlargest(top_k, pred_col)["player_id"])
        hit_rate = len(actual_top_k & pred_top_k) / top_k

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(week_df[pred_col], week_df[true_col])

        if np.isnan(corr):
            print(
                f"  WARNING: Spearman correlation is NaN for week {week} "
                f"(pred_col={pred_col}, n={len(week_df)})"
            )

        weekly_results.append(
            {
                "week": week,
                "top_k_hit_rate": hit_rate,
                "spearman": corr,
            }
        )

    if weekly_results:
        avg_hit_rate = np.mean([r["top_k_hit_rate"] for r in weekly_results])
        spearmans = [r["spearman"] for r in weekly_results]
        # nanmean warns on all-NaN input (happens when every week has constant
        # predictions, e.g. tiny e2e fixtures). Short-circuit to NaN instead.
        avg_spearman = (
            float("nan") if all(np.isnan(s) for s in spearmans) else np.nanmean(spearmans)
        )
    else:
        avg_hit_rate = 0.0
        avg_spearman = 0.0

    return {
        "weekly": weekly_results,
        "season_avg_hit_rate": avg_hit_rate,
        "season_avg_spearman": avg_spearman,
    }


def print_comparison_table(results: dict, position: str, target_names: list[str]) -> None:
    """Pretty-print comparison of all models for a position."""
    print("\n" + "=" * 80)
    print(f"{position} Model Comparison -- Total Fantasy Points")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 56)
    for model_name, metrics in results.items():
        m = metrics["total"]
        print(f"{model_name:<30} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

    # Per-target MAE
    labels = {t: t.replace("_", " ").title() for t in target_names}
    header = "".join(f"{labels[t]:>12}" for t in target_names)
    print(f"\n{'=' * 80}")
    print(f"{position} Model Comparison -- Per-Target MAE")
    print("=" * 80)
    print(f"{'Model':<30} {header}")
    print("-" * (30 + 12 * len(target_names)))
    for model_name, metrics in results.items():
        if target_names[0] in metrics:
            vals = "".join(f"{metrics[t]['mae']:>12.3f}" for t in target_names)
            print(f"{model_name:<30} {vals}")

    # Gated-head diagnostics — only printed for models whose per-target metrics
    # contain gate_auc / gate_brier / cond_mae (attention NN with GatedHead).
    gated_entries = []
    for model_name, metrics in results.items():
        for target in target_names:
            tm = metrics.get(target, {})
            if "gate_auc" in tm or "gate_brier" in tm:
                gated_entries.append((model_name, target, tm))

    if gated_entries:
        print(f"\n{'=' * 80}")
        print(f"{position} Gated-Head Diagnostics")
        print("=" * 80)
        header = f"{'Model':<22}{'Target':<18}{'Gate AUC':>10}{'Brier':>10}{'Pos rate':>12}{'Pred pos':>12}{'Cond MAE':>12}{'Cond RMSE':>12}"
        print(header)
        print("-" * len(header))
        for model_name, target, tm in gated_entries:
            auc = tm.get("gate_auc")
            auc_str = f"{auc:>10.3f}" if auc is not None else f"{'n/a':>10}"
            cond_mae = tm.get("cond_mae")
            cond_mae_str = f"{cond_mae:>12.3f}" if cond_mae is not None else f"{'n/a':>12}"
            cond_rmse = tm.get("cond_rmse")
            cond_rmse_str = f"{cond_rmse:>12.3f}" if cond_rmse is not None else f"{'n/a':>12}"
            print(
                f"{model_name:<22}{target:<18}{auc_str}"
                f"{tm.get('gate_brier', float('nan')):>10.3f}"
                f"{tm.get('positive_rate', float('nan')):>12.3f}"
                f"{tm.get('predicted_positive_rate', float('nan')):>12.3f}"
                f"{cond_mae_str}{cond_rmse_str}"
            )


def plot_pred_vs_actual(
    y_true_dict: dict,
    y_pred_dict: dict,
    target_names: list[str],
    model_name: str,
    save_path: str,
) -> None:
    """Scatter plots of predicted vs actual for each target."""
    targets = [(t, t.replace("_", " ").title()) for t in target_names]

    n = len(targets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = [axes] if n == 1 else list(np.asarray(axes).flat)
    for ax, (target, title) in zip(axes, targets, strict=False):
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name}: {title}")
    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
