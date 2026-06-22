"""Unit tests for the warm-start walk-forward prototype
(src/tuning/warmstart_walkforward.py).

No real training here — those are the manual smoke in the PR. This is the
operator-CLI import-smoke (AGENTS.md: operator-only CLIs need a unit-shard
import test so signature drift fails CI, not PR review) plus coverage of the
pure aggregation/validation helpers and the load-bearing ``init_state_dict``
hook contract on the production attention path.
"""

from __future__ import annotations

import inspect

import pytest

from src.shared import pipeline as P
from src.tuning import warmstart_walkforward as W

pytestmark = pytest.mark.unit


def test_module_imports_production_symbols():
    # Importing the harness binds the production helpers it drives; if any is
    # renamed/removed this import (and thus the test) fails loudly.
    assert callable(W.main)
    assert callable(W._train_attention_holdout)
    assert callable(W._prepare_position_data)
    assert W.SUPPORTED_POSITIONS == ("QB", "RB", "WR", "TE")


def test_init_state_dict_hook_present_on_attention_path():
    # The warm-start hook is the contract between the harness and production.
    # Pin it on all three threaded functions so a refactor can't silently drop
    # it (which would make the warm arm a no-op == cold).
    for fn in (
        P._train_attention_holdout,
        P._train_attention_nn,
        P._train_nested_attention_nn,
    ):
        params = inspect.signature(fn).parameters
        assert "init_state_dict" in params, fn.__name__
        assert params["init_state_dict"].default is None, fn.__name__


def test_unsupported_position_errors():
    # K/DST are out of scope (nested trainer / no frame injection); the CLI must
    # reject them before any data load or training.
    with pytest.raises(SystemExit):
        W.main(["--positions", "K", "--seeds", "42"])
    with pytest.raises(SystemExit):
        W.main(["--positions", "DST", "--seeds", "42"])


def test_agg_mean_std():
    assert W._agg([])["n"] == 0
    one = W._agg([1.0])
    assert one == {"mean": 1.0, "std": 0.0, "n": 1}
    two = W._agg([1.0, 3.0])
    assert two["mean"] == 2.0 and two["std"] == 1.0 and two["n"] == 2


def _fake_run(position, seed, arm, season_mae):
    return {
        "position": position,
        "seed": seed,
        "arm": arm,
        "folds": [
            {
                "origin": i,
                "test_season": s,
                "warm_started": arm == "warm" and i > 0,
                "n_test": 100,
                "mae": m,
                "rmse": m + 1.0,
                "r2": 0.5,
            }
            for i, (s, m) in enumerate(season_mae.items())
        ],
    }


def test_summarize_delta_and_walkforward():
    runs = [
        _fake_run("QB", 42, "cold", {2024: 6.0, 2025: 6.0}),
        _fake_run("QB", 42, "warm", {2024: 6.0, 2025: 5.0}),  # warm better in 2025
    ]
    summary = W._summarize(runs)
    qb = summary["QB"]
    # 2024 identical -> delta 0; 2025 warm better -> negative delta.
    assert qb["per_season"]["2024"]["delta_mae_warm_minus_cold"] == 0.0
    assert qb["per_season"]["2025"]["delta_mae_warm_minus_cold"] == -1.0
    # walk-forward mean over origins: cold (6,6)->6.0, warm (6,5)->5.5
    assert qb["walkforward_mae_over_origins"]["cold"]["mean"] == 6.0
    assert qb["walkforward_mae_over_origins"]["warm"]["mean"] == 5.5
    # _print_table must not raise on a well-formed summary.
    W._print_table(summary)
