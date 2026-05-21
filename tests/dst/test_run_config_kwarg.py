"""Verify DST's run(config=...) kwarg threads through to run_pipeline.

DST is simpler than K (no runtime closure injection) but still builds its
own data internally — the PR 3 contract change is the same: callers can
override the static cfg dict per trial.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _stub_dst_df():
    """Minimal DST DataFrame: one row per (train/val/test) season so the
    ``isin`` splits all return non-empty frames."""
    from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS

    seasons = [
        next(iter(TRAIN_SEASONS)),
        next(iter(VAL_SEASONS)),
        next(iter(TEST_SEASONS)),
    ]
    return pd.DataFrame({"season": seasons, "team": ["KC", "BUF", "PHI"]})


def test_run_forwards_caller_config():
    """A caller-supplied config dict must reach run_pipeline unchanged."""
    from src.dst import run_pipeline as dst_runner

    caller_cfg = {"marker": "from-tune-nn", "attn_d_model": 64}

    with (
        patch("src.dst.run_pipeline.build_data", return_value=_stub_dst_df()),
        patch("src.dst.run_pipeline.compute_targets", side_effect=lambda df: df),
        patch("src.dst.run_pipeline.compute_features"),
        patch("src.dst.run_pipeline.run_pipeline") as mock_run_pipeline,
    ):
        mock_run_pipeline.return_value = "ok"
        result = dst_runner.run(seed=7, config=caller_cfg)

    assert result == "ok"
    forwarded_pos, forwarded_cfg = mock_run_pipeline.call_args.args[:2]
    assert forwarded_pos == "DST"
    # Caller's dict was forwarded — no defensive copy needed since DST
    # doesn't inject anything on top.
    assert forwarded_cfg is caller_cfg


def test_run_defaults_to_module_config():
    """``run()`` with no config kwarg falls back to module-level CONFIG."""
    from src.dst import run_pipeline as dst_runner

    with (
        patch("src.dst.run_pipeline.build_data", return_value=_stub_dst_df()),
        patch("src.dst.run_pipeline.compute_targets", side_effect=lambda df: df),
        patch("src.dst.run_pipeline.compute_features"),
        patch("src.dst.run_pipeline.run_pipeline") as mock_run_pipeline,
    ):
        mock_run_pipeline.return_value = "ok"
        dst_runner.run(seed=42)

    forwarded_cfg = mock_run_pipeline.call_args.args[1]
    assert forwarded_cfg is dst_runner.CONFIG
