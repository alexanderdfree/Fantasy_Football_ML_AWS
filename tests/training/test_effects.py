"""Exercise actual report ownership so full runs cannot fail at the save boundary."""

import pytest

from src.training.effects import save_figures


@pytest.mark.unit
def test_default_figure_sink_calls_training_curve_writer(tmp_path, monkeypatch):
    from src.shared import backtest, evaluation

    monkeypatch.setattr(backtest, "plot_weekly_accuracy", lambda *args: None)
    monkeypatch.setattr(evaluation, "plot_pred_vs_actual", lambda *args: None)
    history = {
        "train_loss": [2.0, 1.0],
        "val_loss": [2.0, 1.5],
        "val_loss_yards": [2.0, 1.5],
        "val_mae_yards": [2.0, 1.5],
    }
    save_figures(tmp_path, "QB", ["yards"], ["feature"], {}, history, {}, {}, {})
    assert (tmp_path / "figures" / "qb_training_curves.png").read_bytes().startswith(b"\x89PNG")
