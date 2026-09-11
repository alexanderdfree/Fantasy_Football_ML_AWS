"""Artifact and figure sinks, separate from fitting and evaluation."""

from __future__ import annotations

from pathlib import Path

import joblib
import torch

from src.prediction.bundle import write_prediction_bundles
from src.shared.artifact_integrity import wrap_state_dict, write_scaler_meta


def save_artifacts(output_dir, position, cfg, prepared, models, scalers, attention_features=()):
    """Publish the fitted families supplied by holdout, CV, or a Batch branch."""
    destination = Path(output_dir) / "models"
    destination.mkdir(parents=True, exist_ok=True)
    columns, targets = list(prepared.feature_columns), cfg["targets"]
    for family in ("ridge", "lgbm"):
        if models.get(family) is not None:
            models[family].save(str(destination))
    if models.get("elasticnet") is not None:
        models["elasticnet"].save(str(destination / "elasticnet"))
    for family, suffix, stem, inputs in (
        ("nn", "multihead", "nn_scaler", columns),
        ("attn_nn", "attention", "attention_nn_scaler", list(attention_features)),
    ):
        if models.get(family) is None:
            continue
        torch.save(
            wrap_state_dict(models[family].state_dict(), inputs, targets),
            destination / f"{position.lower()}_{suffix}_nn.pt",
        )
        joblib.dump(scalers[family], destination / f"{stem}.pkl")
        write_scaler_meta(destination / f"{stem}_meta.json", inputs, targets)
    return write_prediction_bundles(
        destination,
        position,
        cfg,
        columns,
        {name: model for name, model in models.items() if name != "elasticnet"},
        prepared.train,
        attention_features=attention_features,
    )


def save_figures(
    output_dir,
    position,
    targets,
    feature_columns,
    models,
    history,
    simulation,
    truth,
    predictions,
    attention_history=None,
):
    import matplotlib.pyplot as plt

    from src.shared.backtest import plot_weekly_accuracy
    from src.shared.evaluation import plot_pred_vs_actual
    from src.shared.training import plot_training_curves

    destination = Path(output_dir) / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    prefix = position.lower()
    plot_training_curves(history, targets, str(destination / f"{prefix}_training_curves.png"))
    if attention_history is not None:
        plot_training_curves(
            attention_history, targets, str(destination / f"{prefix}_attention_training_curves.png")
        )
    plot_weekly_accuracy(simulation, position, str(destination / f"{prefix}_weekly_mae.png"))
    plot_pred_vs_actual(
        truth,
        predictions,
        targets,
        f"{position} Multi-Head NN",
        str(destination / f"{prefix}_pred_vs_actual_scatter.png"),
    )
    for family, label, axis_label in (
        ("ridge", "Ridge", "Absolute Coefficient"),
        ("lgbm", "LightGBM", "Gain"),
    ):
        model = models.get(family)
        if model is None:
            continue
        importance = model.get_feature_importance(list(feature_columns))
        _, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
        if len(targets) == 1:
            axes = [axes]
        for axis, (target, values) in zip(axes, importance.items(), strict=False):
            values.head(15).plot(kind="barh", ax=axis)
            axis.set_title(f"{label}: {target} Top-15 Features")
            axis.set_xlabel(axis_label)
        plt.tight_layout()
        plt.savefig(destination / f"{prefix}_{family}_feature_importance.png", dpi=150)
        plt.close()
