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
import json

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


def test_partial_seed_pairs_do_not_reverse_the_comparison():
    runs = [
        _fake_run("QB", 42, "cold", {2023: 8.0}),
        _fake_run("QB", 123, "cold", {2023: 4.0}),
        _fake_run("QB", 123, "warm", {2023: 4.5}),
    ]
    qb = W._summarize(runs)["QB"]
    result = qb["per_season"]["2023"]
    assert result["delta_mae_warm_minus_cold"] == 0.5
    assert result["cold"]["mae"]["mean"] == 4.0
    assert result["warm"]["mae"]["mean"] == 4.5
    assert result["comparison"]["paired_seeds"] == [123]
    assert result["comparison"]["unpaired_seeds"] == {"cold": [42], "warm": []}
    assert qb["walkforward_mae_over_origins"]["cold"]["mean"] == 4.0


def test_no_shared_seed_is_explicitly_unavailable():
    runs = [
        _fake_run("QB", 42, "cold", {2023: 8.0}),
        _fake_run("QB", 123, "warm", {2023: 4.5}),
    ]
    qb = W._summarize(runs)["QB"]
    result = qb["per_season"]["2023"]
    assert result["delta_mae_warm_minus_cold"] is None
    assert result["comparison"]["status"] == "unavailable"
    assert result["comparison"]["reason"] == "no_matched_seed_pairs"
    assert result["cold"]["mae"]["n"] == result["warm"]["mae"]["n"] == 0
    assert qb["walkforward_comparison"]["status"] == "unavailable"


def test_walkforward_pairs_only_shared_origins_for_each_seed():
    runs = [
        _fake_run("QB", 42, "cold", {2023: 100.0, 2024: 4.0}),
        _fake_run("QB", 42, "warm", {2024: 4.5}),
    ]
    qb = W._summarize(runs)["QB"]
    assert qb["walkforward_mae_over_origins"]["cold"]["mean"] == 4.0
    assert qb["walkforward_mae_over_origins"]["warm"]["mean"] == 4.5
    assert qb["walkforward_comparison"]["paired_origins"] == {"42": [2024]}


@pytest.fixture
def main_inputs(monkeypatch, tmp_path):
    state = {"failures": set(), "calls": []}
    monkeypatch.setattr(W, "_configure_runtime_env", lambda: None)
    monkeypatch.setattr(W, "_load_full_featured_frame", lambda: None)
    monkeypatch.setattr(W, "rolling_origin_folds", lambda *args, **kwargs: [])
    monkeypatch.setattr(W, "get_config", lambda position: {"targets": ["passing_yards"]})
    monkeypatch.setattr(W, "_git_sha", lambda: "test")

    def run(position, cfg, folds, seed, *, warm, epochs_warm):
        arm = "warm" if warm else "cold"
        state["calls"].append((seed, arm))
        if (seed, arm) in state["failures"]:
            raise RuntimeError(f"failed {seed}/{arm}")
        value = (8.0 if seed == 42 else 4.0) + (0.5 if warm else 0.0)
        return _fake_run(position, seed, arm, {2023: value})["folds"]

    monkeypatch.setattr(W, "_run_arm", run)
    state["args"] = [
        "--positions",
        "QB",
        "--seeds",
        "42",
        "123",
        "--test-seasons",
        "2023",
        "--out-dir",
        str(tmp_path),
    ]
    state["output"] = lambda: json.loads(next(tmp_path.glob("*.json")).read_text())
    return state


def test_main_all_failed_records_every_failure_and_exits_nonzero(main_inputs):
    state = main_inputs
    state["failures"] = {(s, a) for s in (42, 123) for a in ("cold", "warm")}
    with pytest.raises(SystemExit) as exc:
        W.main(state["args"])
    assert exc.value.code == 1
    result = state["output"]()
    assert result["status"] == "failed"
    assert result["runs"] == []
    assert len(result["failures"]) == len(state["calls"]) == 4
    assert {(r["seed"], r["arm"]) for r in result["failures"]} == state["failures"]
    assert all(r["error_type"] == "RuntimeError" for r in result["failures"])
    assert result["summary"]["QB"]["per_season"]["2023"]["comparison"]["status"] == "unavailable"


def test_main_partial_failure_preserves_completed_runs_and_pairs(main_inputs):
    state = main_inputs
    state["failures"] = {(42, "warm")}
    with pytest.raises(SystemExit) as exc:
        W.main(state["args"])
    assert exc.value.code == 1
    result = state["output"]()
    assert len(result["runs"]) == 3
    assert len(result["failures"]) == 1
    assert len(state["calls"]) == 4
    assert result["summary"]["QB"]["per_season"]["2023"]["delta_mae_warm_minus_cold"] == 0.5


def test_main_complete_pairs_return_success(main_inputs):
    output = W.main(main_inputs["args"])
    assert output.endswith(".json")
    result = main_inputs["output"]()
    assert result["status"] == "complete"
    assert result["failures"] == []
    assert len(result["runs"]) == 4
    assert result["summary"]["QB"]["per_season"]["2023"]["delta_mae_warm_minus_cold"] == 0.5


@pytest.mark.parametrize("arm,mean", [("cold", 6.0), ("warm", 6.5)])
def test_intentional_single_arm_still_reports_its_results(main_inputs, arm, mean):
    W.main([*main_inputs["args"], "--arms", arm])
    result = main_inputs["output"]()
    assert result["status"] == "complete"
    assert result["failures"] == []
    assert len(result["runs"]) == 2
    season = result["summary"]["QB"]["per_season"]["2023"]
    assert season[arm]["mae"]["mean"] == mean
    assert season["delta_mae_warm_minus_cold"] is None
    assert season["comparison"]["reason"] == "single_arm_selected"
