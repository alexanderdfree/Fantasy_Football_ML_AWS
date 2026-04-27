"""Coverage smoke test for ``DST/run_dst_pipeline.py``'s ``__main__`` block.

DST's pipeline (like K) does its own data-building internally via
``build_dst_data()``, so we mock the data builder + ``compute_dst_features``
+ ``shared.pipeline.run_pipeline`` to exercise the orchestrator without
hitting the nflverse parquet reads.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_dst_pipeline.py"


def _synthetic_dst_df() -> pd.DataFrame:
    """Minimal team-week frame spanning the canonical train/val/test seasons."""
    seasons = [2022, 2023, 2024, 2025]
    weeks = list(range(1, 4))
    teams = ["BUF", "KC"]
    rows = []
    for s in seasons:
        for w in weeks:
            for t in teams:
                rows.append({"team": t, "season": s, "week": w, "points_allowed": 20.0})
    return pd.DataFrame(rows)


def _patch_all(monkeypatch):
    """Stub data build + feature compute + pipeline dispatch.

    Patches both the source modules (``DST.dst_data`` etc.) and the
    already-imported re-bound names inside ``DST.run_dst_pipeline``, so
    both ``runpy``-re-executions and direct calls see the stubs.
    """
    import src.DST.dst_data as dst_data
    import src.DST.dst_features as dst_features
    import src.DST.dst_targets as dst_targets
    import src.DST.run_dst_pipeline as dst_pipe
    import src.shared.pipeline as sp

    df = _synthetic_dst_df()

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

    # Source modules — runpy picks up from these via fresh `from ... import`.
    monkeypatch.setattr(dst_data, "build_dst_data", lambda: df)
    monkeypatch.setattr(dst_targets, "compute_dst_targets", lambda d: d)
    monkeypatch.setattr(dst_features, "compute_dst_features", lambda d: None)
    monkeypatch.setattr(sp, "run_pipeline", _fake_pipeline)

    # Already-imported re-bound names — for direct-call tests.
    monkeypatch.setattr(dst_pipe, "build_dst_data", lambda: df)
    monkeypatch.setattr(dst_pipe, "compute_dst_targets", lambda d: d)
    monkeypatch.setattr(dst_pipe, "compute_dst_features", lambda d: None)
    monkeypatch.setattr(dst_pipe, "run_pipeline", _fake_pipeline)
    return calls


@pytest.mark.unit
def test_run_dst_pipeline_splits_by_season(monkeypatch):
    """``run_dst_pipeline`` must split the built DataFrame into train/val/test
    using the canonical season buckets and pass them to ``run_pipeline``."""
    calls = _patch_all(monkeypatch)
    import src.DST.run_dst_pipeline as dst_pipe
    from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS

    result = dst_pipe.run_dst_pipeline(seed=99)
    assert result["ridge_metrics"] == {}
    assert len(calls) == 1
    call = calls[0]
    assert call["position"] == "DST"
    assert call["seed"] == 99
    assert call["cfg"] is dst_pipe.DST_CONFIG

    # Every train row must be a TRAIN_SEASONS season, etc.
    assert set(call["train"]["season"].unique()) <= set(TRAIN_SEASONS)
    assert set(call["val"]["season"].unique()) <= set(VAL_SEASONS)
    assert set(call["test"]["season"].unique()) <= set(TEST_SEASONS)
    assert len(call["train"]) > 0
    assert len(call["test"]) > 0


@pytest.mark.unit
def test_main_block_invokes_run_dst_pipeline(monkeypatch):
    calls = _patch_all(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_dst_pipeline.py"])
    runpy.run_path(str(_MODULE_PATH), run_name="__main__")
    assert len(calls) == 1
    assert calls[0]["position"] == "DST"
    assert calls[0]["seed"] == 42
