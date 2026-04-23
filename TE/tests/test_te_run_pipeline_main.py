"""Coverage smoke test for ``TE/run_te_pipeline.py``'s ``__main__`` block.

TE's CLI only has the ``--tiny`` flag (no ``--cv`` split), so this file
only asserts the default and tiny paths plus the module-level wrapper.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_te_pipeline.py"


def _patch_shared_pipeline(monkeypatch):
    import shared.pipeline as sp

    calls: list[dict] = []

    def _fake(position, cfg, *args, **kwargs):
        calls.append({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ridge_metrics": {}, "nn_metrics": {}, "test_df": None, "per_target_preds": {}}

    monkeypatch.setattr(sp, "run_pipeline", _fake)
    return calls


@pytest.mark.unit
def test_main_default_invokes_run_pipeline(monkeypatch):
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_te_pipeline.py"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "TE"


@pytest.mark.unit
def test_main_tiny_wires_shrunk_config(monkeypatch):
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_te_pipeline.py", "--tiny"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]["cfg"]
    assert cfg.get("nn_epochs") == 1
    assert cfg.get("train_attention_nn") is False


@pytest.mark.unit
def test_run_te_pipeline_function_passes_through(monkeypatch):
    import TE.run_te_pipeline as te_pipe

    seen: dict = {}

    def _fake(position, cfg, *args, **kwargs):
        seen.update({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ok": True}

    monkeypatch.setattr(te_pipe, "run_pipeline", _fake)

    result = te_pipe.run_te_pipeline("train", "val", "test", seed=7)
    assert result == {"ok": True}
    assert seen["position"] == "TE"
    assert seen["args"][-1] == 7
    assert seen["cfg"] is te_pipe.TE_CONFIG

    custom = {"custom": True, "targets": ["x"]}
    te_pipe.run_te_pipeline(None, None, None, config=custom)
    assert seen["cfg"] == custom
