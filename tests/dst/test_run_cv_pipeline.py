"""Coverage for DST's ``run_cv`` (Change 4).

DST self-loads team-level data, so (like its ``run()`` main-test) we mock
``build_data`` / ``compute_targets`` / ``compute_features`` + ``run_cv_pipeline``
and assert the frame assembly: full_df = train+val seasons, a 2025 holdout, and
``position="DST"`` forwarded. The shared CV machinery itself is e2e-tested via QB.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "dst" / "run_pipeline.py"


def _synthetic_dst_df() -> pd.DataFrame:
    seasons = [2023, 2024, 2025]
    return pd.DataFrame(
        {
            "team": ["KC", "BUF", "SF"],
            "season": seasons,
            "week": [1, 1, 1],
        }
    )


def _patch_cv(monkeypatch):
    import src.dst.data as dst_data
    import src.dst.features as dst_features
    import src.dst.run_pipeline as dst_pipe
    import src.dst.targets as dst_targets
    import src.shared.pipeline as sp

    df = _synthetic_dst_df()
    calls: list[dict] = []

    def _fake_cv(position, cfg, full_df, test_df, seed):
        calls.append(
            {"position": position, "cfg": cfg, "full": full_df, "test": test_df, "seed": seed}
        )
        return {"cv_metrics": {"ridge": {}, "nn": {}}}

    def _build_data(*, impute_context=True):
        assert impute_context is False
        return df.copy()

    for mod in (dst_data, dst_pipe):
        monkeypatch.setattr(mod, "build_data", _build_data, raising=False)
    for mod in (dst_targets, dst_pipe):
        monkeypatch.setattr(mod, "compute_targets", lambda d: d, raising=False)
    for mod in (dst_features, dst_pipe):
        monkeypatch.setattr(mod, "compute_features", lambda d: None, raising=False)
    monkeypatch.setattr(sp, "run_cv_pipeline", _fake_cv)
    monkeypatch.setattr(dst_pipe, "run_cv_pipeline", _fake_cv)
    return calls


@pytest.mark.unit
def test_run_dst_cv_assembles_frames(monkeypatch):
    calls = _patch_cv(monkeypatch)
    import src.dst.run_pipeline as dst_pipe

    dst_pipe.run_cv(seed=21)
    assert len(calls) == 1
    call = calls[0]
    assert call["position"] == "DST"
    assert call["seed"] == 21
    assert set(np.unique(call["full"]["season"])) == {2023, 2024}
    assert set(np.unique(call["test"]["season"])) == {2025}


@pytest.mark.unit
def test_run_dst_cv_respects_custom_config(monkeypatch):
    calls = _patch_cv(monkeypatch)
    import src.dst.run_pipeline as dst_pipe

    custom = {"targets": ["points_allowed"], "fill_nans_fn": dst_pipe.CONFIG["fill_nans_fn"]}
    original = custom.copy()
    dst_pipe.run_cv(config=custom)
    forwarded = calls[0]["cfg"]
    assert forwarded is not custom
    assert forwarded["targets"] == custom["targets"]
    assert forwarded["fill_nans_fn"].keywords["fill_nans_fn"] is custom["fill_nans_fn"]
    assert custom == original


@pytest.mark.unit
def test_main_cv_flag_invokes_run_dst_cv(monkeypatch):
    calls = _patch_cv(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "--cv"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "DST"
