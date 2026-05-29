"""Coverage for TE's ``run_cv`` wrapper + ``--cv`` CLI dispatch (Change 4).

TE routes through the shared ``run_cv_pipeline`` (the same machinery QB/RB/WR
e2e-test), so this file covers the TE-specific wiring — the wrapper forwards
``("TE", CONFIG)`` and the ``--cv`` flag reaches ``run_cv`` — without paying for
real CV training.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "te" / "run_pipeline.py"


@pytest.mark.unit
def test_run_te_cv_wrapper_dispatches_to_run_cv_pipeline(monkeypatch):
    import src.te.run_pipeline as te_pipe

    seen: list[dict] = []

    def _fake_cv(position, cfg, *args, **kwargs):
        seen.append({"position": position, "cfg": cfg, "args": args})
        return {"cv_metrics": {"ridge": {}, "nn": {}}}

    monkeypatch.setattr(te_pipe, "run_cv_pipeline", _fake_cv)
    te_pipe.run_cv(full_df="full", test_df="test", seed=7)
    assert seen[0]["position"] == "TE"
    assert seen[0]["cfg"] is te_pipe.CONFIG
    assert seen[0]["args"][-1] == 7  # seed travels positionally

    custom = {"custom": True, "targets": ["x"]}
    te_pipe.run_cv(seed=11, config=custom)
    assert seen[1]["cfg"] == custom  # explicit config overrides CONFIG


@pytest.mark.unit
def test_main_cv_flag_invokes_run_cv(monkeypatch):
    """``python run.py --cv`` (via cli_main) must dispatch to run_cv → run_cv_pipeline."""
    import src.shared.pipeline as sp
    import src.te.run_pipeline as te_pipe

    seen: list[str] = []

    def _fake_cv(position, cfg, *args, **kwargs):
        seen.append(position)
        return {"cv_metrics": {}}

    monkeypatch.setattr(sp, "run_cv_pipeline", _fake_cv)
    monkeypatch.setattr(te_pipe, "run_cv_pipeline", _fake_cv)
    monkeypatch.setattr(sys, "argv", ["run.py", "--cv"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert seen == ["TE"]
