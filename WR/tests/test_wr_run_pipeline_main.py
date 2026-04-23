"""Coverage smoke test for ``WR/run_wr_pipeline.py``'s ``__main__`` block.

Mirrors the QB/RB versions — drives argparse + dispatch with mocked
``shared.pipeline.run_pipeline`` / ``run_cv_pipeline`` so the CLI branch
gets exercised without a real training run.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_wr_pipeline.py"


def _patch_shared_pipeline(monkeypatch):
    import shared.pipeline as sp

    calls: list[dict] = []

    def _fake(position, cfg, *args, **kwargs):
        calls.append({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ridge_metrics": {}, "nn_metrics": {}, "test_df": None, "per_target_preds": {}}

    monkeypatch.setattr(sp, "run_pipeline", _fake)
    monkeypatch.setattr(sp, "run_cv_pipeline", _fake)
    return calls


@pytest.mark.unit
def test_main_default_invokes_run_pipeline(monkeypatch):
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_wr_pipeline.py"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "WR"


@pytest.mark.unit
def test_main_tiny_wires_shrunk_config(monkeypatch):
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_wr_pipeline.py", "--tiny"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]["cfg"]
    assert cfg.get("nn_epochs") == 1
    assert cfg.get("train_attention_nn") is False
    assert cfg.get("train_lightgbm") is False


@pytest.mark.unit
def test_main_cv_routes_to_run_cv_pipeline(monkeypatch):
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_wr_pipeline.py", "--cv"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "WR"


@pytest.mark.unit
def test_run_wr_pipeline_function_passes_through(monkeypatch):
    import WR.run_wr_pipeline as wr_pipe

    seen: dict = {}

    def _fake(position, cfg, *args, **kwargs):
        seen.update({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ok": True}

    monkeypatch.setattr(wr_pipe, "run_pipeline", _fake)
    monkeypatch.setattr(wr_pipe, "run_cv_pipeline", _fake)

    result = wr_pipe.run_wr_pipeline("train", "val", "test", seed=7)
    assert result == {"ok": True}
    assert seen["position"] == "WR"
    assert seen["args"][-1] == 7
    assert seen["cfg"] is wr_pipe.WR_CONFIG

    custom = {"custom": True, "targets": ["x"]}
    wr_pipe.run_wr_pipeline(None, None, None, config=custom)
    assert seen["cfg"] == custom

    wr_pipe.run_wr_cv_pipeline("full", "test", seed=11)
    assert seen["args"][-1] == 11
