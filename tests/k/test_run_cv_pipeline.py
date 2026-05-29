"""Coverage for K's ``run_cv`` (Change 4).

K self-loads its data + builds the per-kick attention closure, so (like its
``run()`` main-test) we mock the loaders + ``run_cv_pipeline`` and assert the
frame assembly: full_df from the unfiltered ``k_df`` restricted to train+val
seasons, a 2025 holdout, the closure injected into a COPY of CONFIG (no
mutation), and ``position="K"`` forwarded. The shared CV machinery itself is
e2e-tested via QB.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "k" / "run_pipeline.py"


def _synthetic_k_df() -> pd.DataFrame:
    seasons = [2023, 2024, 2025]
    return pd.DataFrame(
        {
            "player_id": [f"K{i:02d}" for i in range(len(seasons))],
            "season": seasons,
            "week": [1, 1, 1],
            "team": ["KC"] * len(seasons),
        }
    )


def _patch_cv(monkeypatch):
    import src.k.data as k_data
    import src.k.features as k_features
    import src.k.run_pipeline as k_pipe
    import src.k.targets as k_targets
    import src.shared.pipeline as sp

    k_df = _synthetic_k_df()
    kicks_df = pd.DataFrame({"player_id": ["K00"], "kick_distance": [30]})
    calls: list[dict] = []

    def _fake_cv(position, cfg, full_df, test_df, seed):
        calls.append(
            {"position": position, "cfg": cfg, "full": full_df, "test": test_df, "seed": seed}
        )
        return {"cv_metrics": {"ridge": {}, "nn": {}}}

    for mod in (k_data, k_pipe):
        monkeypatch.setattr(mod, "load_data", lambda: k_df.copy(), raising=False)
        monkeypatch.setattr(mod, "load_kicks", lambda df: kicks_df, raising=False)
    for mod in (k_targets, k_pipe):
        monkeypatch.setattr(mod, "compute_targets", lambda df: df, raising=False)
    for mod in (k_features, k_pipe):
        monkeypatch.setattr(mod, "compute_features", lambda df: None, raising=False)
    monkeypatch.setattr(sp, "run_cv_pipeline", _fake_cv)
    monkeypatch.setattr(k_pipe, "run_cv_pipeline", _fake_cv)
    return calls


@pytest.mark.unit
def test_run_k_cv_assembles_frames_and_closure(monkeypatch):
    calls = _patch_cv(monkeypatch)
    import src.k.run_pipeline as k_pipe

    k_pipe.run_cv(seed=13)
    assert len(calls) == 1
    call = calls[0]
    assert call["position"] == "K"
    assert call["seed"] == 13
    # full_df = train+val seasons only (no 2025); test_df = 2025 holdout.
    assert set(np.unique(call["full"]["season"])) == {2023, 2024}
    assert set(np.unique(call["test"]["season"])) == {2025}
    # Per-kick closure injected into the cfg copy, CONFIG left untouched.
    assert "attn_history_builder_fn" in call["cfg"]
    assert "attn_history_builder_fn" not in k_pipe.CONFIG


@pytest.mark.unit
def test_run_k_cv_does_not_mutate_passed_config(monkeypatch):
    _patch_cv(monkeypatch)
    import src.k.run_pipeline as k_pipe

    custom = {"targets": ["fg_made"], "attn_kick_stats": [], "attn_max_games": 5}
    k_pipe.run_cv(config=custom)
    assert "attn_history_builder_fn" not in custom  # copied, not mutated


@pytest.mark.unit
def test_main_cv_flag_invokes_run_k_cv(monkeypatch):
    calls = _patch_cv(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "--cv"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "K"
