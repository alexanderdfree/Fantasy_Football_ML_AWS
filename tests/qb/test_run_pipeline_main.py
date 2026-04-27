"""Coverage smoke test for ``QB/run_qb_pipeline.py``'s ``__main__`` block.

Runs the script via ``runpy`` with mocked ``src.shared.pipeline.run_pipeline`` /
``run_cv_pipeline`` so we exercise the argparse + dispatch logic without
paying for a real training round-trip. Every other QB test either imports
``QB_CONFIG`` directly or drives its own tiny pipeline — none touch the
``if __name__ == "__main__"`` body, leaving that branch permanently at 0%
coverage in the QB flag.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "qb" / "run_pipeline.py"


def _patch_shared_pipeline(monkeypatch):
    """Replace src.shared.pipeline run_* with no-op stubs that log invocations.

    Returns a list that gets populated with (position, cfg, kwargs) per
    call, so tests can assert which dispatch branch fired.
    """
    import src.shared.pipeline as sp

    calls: list[dict] = []

    def _fake(position, cfg, *args, **kwargs):
        calls.append({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ridge_metrics": {}, "nn_metrics": {}, "test_df": None, "per_target_preds": {}}

    monkeypatch.setattr(sp, "run_pipeline", _fake)
    monkeypatch.setattr(sp, "run_cv_pipeline", _fake)
    return calls


@pytest.mark.unit
def test_main_default_invokes_run_pipeline(monkeypatch):
    """Bare ``python run_qb_pipeline.py`` routes through ``run_pipeline``."""
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_qb_pipeline.py"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "QB"
    assert "targets" in calls[0]["cfg"]


@pytest.mark.unit
def test_main_tiny_wires_shrunk_config(monkeypatch):
    """``--tiny`` builds config via ``tests._pipeline_e2e_utils.build_tiny_config``."""
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_qb_pipeline.py", "--tiny"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]["cfg"]
    assert cfg.get("nn_epochs") == 1
    assert cfg.get("train_attention_nn") is False
    assert cfg.get("train_lightgbm") is False


@pytest.mark.unit
def test_main_cv_routes_to_run_cv_pipeline(monkeypatch):
    """``--cv`` dispatches to ``run_cv_pipeline`` (same stub, same call log)."""
    calls = _patch_shared_pipeline(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_qb_pipeline.py", "--cv"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "QB"


@pytest.mark.unit
def test_run_qb_pipeline_function_passes_through(monkeypatch):
    """The module-level wrapper is a thin pass-through — cover it explicitly.

    ``run_qb_pipeline.py`` did ``from src.shared.pipeline import run_pipeline``
    at import time, so by now the name ``run_pipeline`` inside that module
    is bound to the original function. Patch the module attribute directly
    rather than ``src.shared.pipeline.run_pipeline``.
    """
    import src.qb.run_pipeline as qb_pipe

    seen: dict = {}

    def _fake(position, cfg, *args, **kwargs):
        seen.update({"position": position, "cfg": cfg, "args": args, "kwargs": kwargs})
        return {"ok": True}

    monkeypatch.setattr(qb_pipe, "run_pipeline", _fake)
    monkeypatch.setattr(qb_pipe, "run_cv_pipeline", _fake)

    result = qb_pipe.run_qb_pipeline("train", "val", "test", seed=7)
    assert result == {"ok": True}
    assert seen["position"] == "QB"
    assert seen["args"][-1] == 7  # seed is positional to run_pipeline
    assert seen["cfg"] is qb_pipe.QB_CONFIG

    # Override the cfg kwarg to confirm the fallback branch (config or QB_CONFIG)
    custom = {"custom": True, "targets": ["x"]}
    qb_pipe.run_qb_pipeline(None, None, None, config=custom)
    assert seen["cfg"] == custom

    # Same for cv wrapper
    qb_pipe.run_qb_cv_pipeline("full", "test", seed=11)
    assert seen["args"][-1] == 11
