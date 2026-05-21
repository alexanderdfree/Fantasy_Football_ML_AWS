"""Verify K's run(config=...) kwarg threads through to run_pipeline.

K is special among the six runners: it builds ``kicks_df`` at runtime and
injects an ``attn_history_builder_fn`` closure into the cfg before calling
``run_pipeline``. The PR 3 contract change is that callers can now pass
``config=`` to override the static cfg dict — but the runtime injection
must still happen on top, otherwise the attention NN would crash on a
missing ``attn_history_builder_fn``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_run_forwards_caller_config_with_builder_injection():
    """A caller-supplied config dict must reach run_pipeline with the
    runtime ``attn_history_builder_fn`` added — not replaced."""
    from src.k import run_pipeline as k_runner

    caller_cfg = {
        "marker": "from-tune-nn",  # arbitrary key to assert pass-through
        "attn_d_model": 64,  # a real cfg key the tuner might override
    }

    with (
        patch("src.k.run_pipeline.load_data") as mock_load,
        patch("src.k.run_pipeline.compute_targets", side_effect=lambda df: df),
        patch("src.k.run_pipeline.compute_features"),
        patch("src.k.run_pipeline.load_kicks") as mock_load_kicks,
        patch("src.k.run_pipeline.season_split", return_value=("train", "val", "test")),
        patch("src.k.run_pipeline.run_pipeline") as mock_run_pipeline,
    ):
        import pandas as pd

        # Pre-populate with a player_id column so the load printout doesn't
        # divide-by-zero or KeyError before the patched chain returns.
        mock_load.return_value = pd.DataFrame({"player_id": ["p1"]})
        mock_load_kicks.return_value = pd.DataFrame()
        mock_run_pipeline.return_value = "ok"

        result = k_runner.run(seed=7, config=caller_cfg)

    assert result == "ok"
    mock_run_pipeline.assert_called_once()
    forwarded_pos, forwarded_cfg = mock_run_pipeline.call_args.args[:2]
    assert forwarded_pos == "K"
    # Caller's keys survived.
    assert forwarded_cfg["marker"] == "from-tune-nn"
    assert forwarded_cfg["attn_d_model"] == 64
    # Runtime closure was injected on top.
    assert "attn_history_builder_fn" in forwarded_cfg
    # Caller's dict must not have been mutated (tuner reuses base_cfg across
    # trials and would crash on trial 2 if we mutated in place).
    assert "attn_history_builder_fn" not in caller_cfg


def test_run_defaults_to_module_config():
    """``run()`` with no config kwarg must still use the module-level CONFIG —
    backward compat with existing callers (cli_main, src/batch/train.py, etc.)."""
    from src.k import run_pipeline as k_runner

    with (
        patch("src.k.run_pipeline.load_data") as mock_load,
        patch("src.k.run_pipeline.compute_targets", side_effect=lambda df: df),
        patch("src.k.run_pipeline.compute_features"),
        patch("src.k.run_pipeline.load_kicks") as mock_load_kicks,
        patch("src.k.run_pipeline.season_split", return_value=("train", "val", "test")),
        patch("src.k.run_pipeline.run_pipeline") as mock_run_pipeline,
    ):
        import pandas as pd

        mock_load.return_value = pd.DataFrame({"player_id": ["p1"]})
        mock_load_kicks.return_value = pd.DataFrame()
        mock_run_pipeline.return_value = "ok"

        k_runner.run(seed=42)

    forwarded_cfg = mock_run_pipeline.call_args.args[1]
    # Module CONFIG keys should be present alongside the runtime injection.
    assert "attn_history_builder_fn" in forwarded_cfg
    # And it shouldn't be the marker dict from the other test.
    assert "marker" not in forwarded_cfg
