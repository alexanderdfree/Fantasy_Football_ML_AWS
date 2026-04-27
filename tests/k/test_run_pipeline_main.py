"""Coverage smoke test for ``src/k/run_pipeline.py``'s ``__main__`` block.

K's pipeline does its own data-loading internally (``load_data`` +
``load_kicks``), so unlike QB/RB/WR/TE we have to mock the loaders
and the season split in addition to ``src.shared.pipeline.run_pipeline`` in
order to exercise the function body without touching the PBP cache.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "k" / "run_pipeline.py"


def _synthetic_k_df(n: int = 6) -> pd.DataFrame:
    """Minimal kicker weekly frame: enough columns for print() + logic."""
    return pd.DataFrame(
        {
            "player_id": [f"K{i:02d}" for i in range(n)],
            "season": np.repeat([2023, 2024, 2025], n // 3 + 1)[:n],
            "week": np.tile([1, 2, 3], n // 3 + 1)[:n],
            "team": ["KC"] * n,
        }
    )


def _patch_all(monkeypatch):
    """Stub loaders + pipeline so ``run()`` runs in-memory.

    We patch both the source modules (``src.k.data``, ``src.k.features``,
    ``src.k.targets``, ``src.shared.pipeline``) and the re-bound names inside
    ``src.k.run_pipeline``. Source-module patching matters when ``runpy``
    re-executes the script: the fresh ``from src.k.data import load_data``
    grabs whatever's on ``src.k.data`` at that moment. Re-binding in-place on
    the already-imported ``k_pipe`` handles direct-call tests.
    """
    import src.k.data as k_data
    import src.k.features as k_features
    import src.k.run_pipeline as k_pipe
    import src.k.targets as k_targets
    import src.shared.pipeline as sp

    k_df = _synthetic_k_df()
    kicks_df = pd.DataFrame({"player_id": ["K00"], "kick_distance": [30]})

    calls: list[dict] = []

    def _fake_pipeline(position, cfg, train_df, val_df, test_df, seed):
        calls.append(
            {
                "position": position,
                "cfg": cfg,
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "seed": seed,
            }
        )
        return {"ridge_metrics": {}, "nn_metrics": {}, "test_df": test_df}

    def _fake_split(df):
        return df.iloc[:2], df.iloc[2:4], df.iloc[4:]

    # Source modules — what runpy will re-import from.
    monkeypatch.setattr(k_data, "load_data", lambda: k_df)
    monkeypatch.setattr(k_data, "load_kicks", lambda df: kicks_df)
    monkeypatch.setattr(k_data, "season_split", _fake_split)
    monkeypatch.setattr(k_targets, "compute_targets", lambda df: df)
    monkeypatch.setattr(k_features, "compute_features", lambda df: None)
    monkeypatch.setattr(sp, "run_pipeline", _fake_pipeline)

    # Already-imported names inside k_pipe — for direct-call tests.
    monkeypatch.setattr(k_pipe, "load_data", lambda: k_df)
    monkeypatch.setattr(k_pipe, "load_kicks", lambda df: kicks_df)
    monkeypatch.setattr(k_pipe, "season_split", _fake_split)
    monkeypatch.setattr(k_pipe, "compute_targets", lambda df: df)
    monkeypatch.setattr(k_pipe, "compute_features", lambda df: None)
    monkeypatch.setattr(k_pipe, "run_pipeline", _fake_pipeline)
    return calls


@pytest.mark.unit
def test_run_k_pipeline_wires_everything(monkeypatch):
    """``run()`` must invoke ``run_pipeline`` with the K position,
    the production config (plus attention history builder injected), and
    the three season splits from ``season_split``."""
    calls = _patch_all(monkeypatch)
    import src.k.run_pipeline as k_pipe

    result = k_pipe.run(seed=13)
    assert result["ridge_metrics"] == {}
    assert len(calls) == 1
    call = calls[0]
    assert call["position"] == "K"
    assert call["seed"] == 13
    # Attention history builder closure is injected onto a copy of CONFIG
    assert "attn_history_builder_fn" in call["cfg"]
    # Sanity-check: the wrapping dict() call leaves CONFIG unmodified
    assert "attn_history_builder_fn" not in k_pipe.CONFIG


@pytest.mark.unit
def test_main_block_invokes_run_k_pipeline(monkeypatch):
    """``python run.py`` enters the ``__main__`` guard and fires
    ``run()`` with the default seed."""
    calls = _patch_all(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "K"
    assert calls[0]["seed"] == 42
