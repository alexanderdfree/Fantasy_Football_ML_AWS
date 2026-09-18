from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.shared.registry import get_config
from src.training.context import RunContext, training_entrypoint
from src.training.contracts import PreparedDataset, TrainingResult
from src.training.reuse_identity import digest, fit_identity, stable

pytestmark = pytest.mark.unit


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_RESULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FF_RESULT_CACHE_BUCKET", "")
    monkeypatch.delenv("FF_FRESH", raising=False)
    monkeypatch.setattr(
        "src.training.reuse_identity.execution_identity",
        lambda device=None: {"device": "cpu", "regime": "fp32"},
    )
    return RunContext(tmp_path / "outputs", tmp_path / "data", reuse_results=True, report_sink=None)


def frames(config):
    frame = pd.DataFrame({target: np.arange(1.0, 5.0) for target in config["targets"]})
    frame["player_id"] = ["a", "b", "a", "b"]
    frame["season"] = [2025] * 4
    frame["week"] = [1, 1, 2, 2]
    frame["fantasy_points"] = config["aggregate_fn"](
        {target: frame[target].to_numpy() for target in config["targets"]}
    )
    return frame


def pipeline(calls):
    @training_entrypoint
    def run(position, cfg, train_df=None, val_df=None, test_df=None, seed=42, *, context=None):
        calls.append(seed)
        truth = {target: test_df[target].to_numpy() for target in cfg["targets"]}
        prepared = PreparedDataset(
            np.ones((4, 1)),
            np.ones((4, 1)),
            np.ones((4, 1)),
            truth,
            truth,
            truth,
            train_df,
            val_df,
            test_df,
            ("f",),
            "data",
        )
        # Exercise preparation's legitimate mutation without invalidating caller inputs.
        test_df["prepared"] = 1
        return TrainingResult(
            {
                "test_df": test_df,
                "per_target_preds": {"ridge": truth},
                "phase_seconds": {"fit": 30.0},
            },
            cfg,
            prepared,
            {},
            context.run_id,
        )

    return run


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_all_positions_reuse_exact_predictions_without_fitting(position, environment):
    cfg = get_config(position)
    # K's provider normally binds its nested builder before this boundary.
    if position == "K":
        cfg = {**cfg, "train_attention_nn": False}
    df = frames(cfg)
    calls = []
    run = pipeline(calls)
    first = run(position, cfg, df, df, df, context=environment)
    second = run(position, cfg, df, df, df, context=replace(environment, run_id="second"))
    assert calls == [42]
    assert second["reuse"]["cache_hit"]
    assert second["reuse"]["reused_from"] == first.run_id
    assert second.run_id == "second"
    assert "fit" not in second.timings
    assert second["reuse"]["source_phase_seconds"] == {"fit": 30.0}
    assert "prepared" not in df
    for target in cfg["targets"]:
        np.testing.assert_array_equal(
            first.predictions["ridge"][target], second.predictions["ridge"][target]
        )
    assert second["ridge_metrics"]["total"]["mae"] == 0


def test_changes_to_data_config_seed_or_regime_force_fitting(environment, monkeypatch):
    cfg = get_config("RB")
    df = frames(cfg)
    calls = []
    run = pipeline(calls)
    run("RB", cfg, df, df, df, context=environment)
    run("RB", cfg, df, df, df, context=replace(environment, seed=7))
    run("RB", {**cfg, "nn_lr": cfg["nn_lr"] * 2}, df, df, df, context=environment)
    changed = df.copy()
    changed.iloc[0, 0] += 1
    run("RB", cfg, changed, df, df, context=environment)
    monkeypatch.setattr(
        "src.training.reuse_identity.execution_identity",
        lambda device=None: {"device": "cpu", "regime": "other"},
    )
    run("RB", cfg, df, df, df, context=environment)
    assert len(calls) == 5


def test_raw_side_input_change_fresh_and_default_training_bypass(environment, monkeypatch):
    cfg = get_config("RB")
    df = frames(cfg)
    environment.raw_root.mkdir(parents=True)
    side = environment.raw_root / "weather.json"
    side.write_text("old")
    calls = []
    run = pipeline(calls)
    run("RB", cfg, df, df, df, context=environment)
    side.write_text("new")
    run("RB", cfg, df, df, df, context=environment)
    monkeypatch.setenv("FF_FRESH", "1")
    run("RB", cfg, df, df, df, context=environment)
    monkeypatch.delenv("FF_FRESH")
    run("RB", cfg, df, df, df, context=replace(environment, reuse_results=False))
    assert calls == [42] * 4


def test_callbacks_are_executed_instead_of_reused(environment):
    cfg = {**get_config("RB"), "epoch_callback": lambda *_: None}
    df = frames(cfg)
    calls = []
    run = pipeline(calls)
    run("RB", cfg, df, df, df, context=environment)
    run("RB", cfg, df, df, df, context=environment)
    assert calls == [42, 42]


def test_frame_order_and_dtype_are_part_of_identity():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    assert digest(stable(frame)) != digest(stable(frame.iloc[::-1]))
    assert digest(stable(frame)) != digest(stable(frame.astype("float32")))


def test_unavailable_cache_does_not_hide_real_fit_failures(environment):
    @training_entrypoint
    def fail(position, cfg, seed=42, *, context=None):
        raise RuntimeError("real fit failure")

    with pytest.raises(RuntimeError, match="real fit failure"):
        fail("RB", get_config("RB"), context=environment)
