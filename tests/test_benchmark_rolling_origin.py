"""Unit tests for the rolling-origin driver in src/benchmarking/benchmark.py.

The per-origin training (``_score_origin``) and fold construction
(``_rolling_origin_inputs``) are monkeypatched, so these cover the driver's
iteration, mean±std aggregation, production-origin selection, and schema-safety
without training a model or touching data splits.
"""

from __future__ import annotations

import json
import statistics

import pandas as pd
import pytest

import src.benchmarking.benchmark as b

pytestmark = pytest.mark.unit


def _summary(pos, ridge_mae, nn_mae, *, with_lgbm=False):
    s = {
        "position": pos,
        "ridge_mae": ridge_mae,
        "ridge_r2": 0.40,
        "ridge_top12": 0.55,
        "nn_mae": nn_mae,
        "nn_r2": 0.42,
        "nn_top12": 0.58,
    }
    if with_lgbm:
        s.update({"lgbm_mae": nn_mae + 0.1, "lgbm_r2": 0.41, "lgbm_top12": 0.56})
    return s


def _rolling_summary(pos="RB"):
    s = _summary(pos, 4.7, 4.6)
    s["rolling_origin"] = b._aggregate_rolling_origin([(2025, dict(s))])
    return s


def _patch_benchmark_io(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "RESULTS_FILE", str(tmp_path / "benchmark_results.json"))
    monkeypatch.setattr(b, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(b, "collect_global_config", lambda: {"test": True})
    monkeypatch.setattr(b, "collect_pos_config", lambda pos: {"position": pos})
    monkeypatch.setattr(b, "get_git_hash", lambda: "abc1234")
    monkeypatch.setattr(b, "utc_now_iso", lambda: "2026-06-01T12:00:00")
    monkeypatch.setattr(b, "_maybe_upload_to_s3", lambda path: pytest.fail("unexpected S3 sync"))


def test_module_exposes_rolling_origin_api():
    for fn in (
        "run_rolling_origin",
        "_aggregate_rolling_origin",
        "_score_origin",
        "_rolling_origin_inputs",
        "score_one_origin",
        "finalize_rolling_origin",
    ):
        assert hasattr(b, fn)


def test_aggregate_rolling_origin_mean_std_math():
    per_origin = [
        (2023, _summary("RB", 4.9, 4.8)),
        (2024, _summary("RB", 4.7, 4.6)),
        (2025, _summary("RB", 4.5, 4.4)),
    ]
    ro = b._aggregate_rolling_origin(per_origin)
    assert ro["n_origins"] == 3
    assert ro["test_seasons"] == [2023, 2024, 2025]
    assert [o["test_season"] for o in ro["per_origin"]] == [2023, 2024, 2025]
    assert ro["aggregate"]["ridge"]["mae_mean"] == round(statistics.mean([4.9, 4.7, 4.5]), 4)
    assert ro["aggregate"]["ridge"]["mae_std"] == round(statistics.stdev([4.9, 4.7, 4.5]), 4)
    assert ro["aggregate"]["nn"]["mae_mean"] == round(statistics.mean([4.8, 4.6, 4.4]), 4)


def test_aggregate_excludes_model_not_in_every_origin():
    """LightGBM present in only one origin must not appear in the aggregate (avoids
    averaging over a non-uniform model set)."""
    per_origin = [
        (2024, _summary("RB", 4.7, 4.6, with_lgbm=True)),
        (2025, _summary("RB", 4.5, 4.4, with_lgbm=False)),
    ]
    ro = b._aggregate_rolling_origin(per_origin)
    assert "lgbm" not in ro["aggregate"]
    assert "ridge" in ro["aggregate"] and "nn" in ro["aggregate"]
    # per_origin still records the lgbm number where it exists.
    assert "lgbm_mae" in ro["per_origin"][0] and "lgbm_mae" not in ro["per_origin"][1]


def test_aggregate_single_origin_std_is_zero():
    ro = b._aggregate_rolling_origin([(2025, _summary("K", 3.0, 2.9))])
    assert ro["n_origins"] == 1
    assert ro["aggregate"]["ridge"]["mae_std"] == 0.0


def test_run_rolling_origin_assembles_block_and_keeps_flat_keys(monkeypatch):
    # te placeholder carries the season int so the mocked scorer returns per-origin data.
    origins = [(ts, None, None, ts) for ts in (2023, 2024, 2025)]
    canned = {
        2023: _summary("RB", 4.9, 4.8),
        2024: _summary("RB", 4.8, 4.7),
        2025: _summary("RB", 4.7, 4.6),  # production origin
    }
    monkeypatch.setattr(b, "_rolling_origin_inputs", lambda pos: (origins, None))
    monkeypatch.setattr(b, "_score_origin", lambda pos, tr, va, te, cfg: dict(canned[te]))

    s = b.run_rolling_origin("RB")
    # Flat keys equal the production origin (test==TEST_SEASONS[0]==2025), so the
    # row stays comparable to a normal single-split run in the History tab.
    assert s["ridge_mae"] == 4.7
    assert s["nn_mae"] == 4.6
    ro = s["rolling_origin"]
    assert ro["n_origins"] == 3
    assert ro["test_seasons"] == [2023, 2024, 2025]
    assert ro["aggregate"]["ridge"]["mae_mean"] == round(statistics.mean([4.9, 4.8, 4.7]), 4)
    # Whole summary (flat keys + nested rolling_origin block) is JSON-serializable.
    json.dumps(s)


def test_score_one_origin_returns_only_the_requested_origin(monkeypatch):
    # The cell entry point: selects the origin whose test season matches and scores
    # only it (loading the position's folds once), returning (test_season, summary).
    origins = [(ts, None, None, ts) for ts in (2023, 2024, 2025)]
    canned = {
        2023: _summary("RB", 4.9, 4.8),
        2024: _summary("RB", 4.8, 4.7),
        2025: _summary("RB", 4.7, 4.6),
    }
    scored = []

    def _fake_score(pos, tr, va, te, cfg, seed=42):
        scored.append((te, seed))
        return dict(canned[te])

    monkeypatch.setattr(b, "_rolling_origin_inputs", lambda pos: (origins, None))
    monkeypatch.setattr(b, "_score_origin", _fake_score)

    ts, summary = b.score_one_origin("RB", 2024)
    assert ts == 2024
    assert summary["ridge_mae"] == 4.8 and summary["nn_mae"] == 4.7
    assert scored == [(2024, 42)]  # only the requested origin scored, default seed


def test_score_one_origin_raises_on_unknown_origin(monkeypatch):
    origins = [(ts, None, None, ts) for ts in (2023, 2024, 2025)]
    monkeypatch.setattr(b, "_rolling_origin_inputs", lambda pos: (origins, None))
    monkeypatch.setattr(b, "_score_origin", lambda *a, **k: pytest.fail("must not score"))
    with pytest.raises(ValueError, match="not a rolling origin"):
        b.score_one_origin("RB", 1999)


def test_finalize_rolling_origin_keeps_production_flat_keys_and_block():
    # Same result as the tail of run_rolling_origin: production-origin (2025) flat keys
    # + a rolling_origin mean±std block over all origins.
    per_origin = [
        (2023, _summary("RB", 4.9, 4.8)),
        (2024, _summary("RB", 4.8, 4.7)),
        (2025, _summary("RB", 4.7, 4.6)),  # production origin == TEST_SEASONS[0]
    ]
    final = b.finalize_rolling_origin("RB", per_origin)
    assert final["ridge_mae"] == 4.7 and final["nn_mae"] == 4.6
    ro = final["rolling_origin"]
    assert ro["test_seasons"] == [2023, 2024, 2025]
    assert ro["n_origins"] == 3
    assert ro["aggregate"]["ridge"]["mae_mean"] == round(statistics.mean([4.9, 4.8, 4.7]), 4)
    json.dumps(final)


def test_run_rolling_origin_delegates_to_finalize(monkeypatch):
    """run_rolling_origin still loads the frame once + loops origins, then defers the
    production-origin selection + aggregation to finalize_rolling_origin (relocated,
    not duplicated)."""
    origins = [(ts, None, None, ts) for ts in (2023, 2024, 2025)]
    canned = {
        ts: _summary("RB", 5.0 - i * 0.1, 4.9 - i * 0.1) for i, ts in enumerate((2023, 2024, 2025))
    }
    monkeypatch.setattr(b, "_rolling_origin_inputs", lambda pos: (origins, None))
    monkeypatch.setattr(b, "_score_origin", lambda pos, tr, va, te, cfg, seed=42: dict(canned[te]))

    seen = {}

    def _fake_finalize(pos, per_origin):
        seen["pos"] = pos
        seen["seasons"] = [ts for ts, _ in per_origin]
        return {"position": pos, "finalized": True}

    monkeypatch.setattr(b, "finalize_rolling_origin", _fake_finalize)
    out = b.run_rolling_origin("RB")
    assert out == {"position": "RB", "finalized": True}
    assert seen["pos"] == "RB"
    assert seen["seasons"] == [2023, 2024, 2025]


def test_main_cv_alias_routes_to_rolling_origin_and_marks_history(tmp_path, monkeypatch, capsys):
    _patch_benchmark_io(monkeypatch, tmp_path)
    calls = []

    def fake_rolling_origin(pos):
        calls.append(pos)
        return _rolling_summary(pos)

    monkeypatch.setattr(b, "run_rolling_origin", fake_rolling_origin)
    monkeypatch.setattr(b, "run_one", lambda *a, **k: pytest.fail("run_one must not handle --cv"))

    # --sequential pins the in-process loop (vs the parallel orchestrator a capable
    # CUDA box would otherwise pick for the 3 origin cells) so we test the --cv ->
    # rolling-origin REPORTING route + history write here, not the dispatch path.
    b.main(["RB", "--cv", "--no-sync", "--sequential"])

    out = capsys.readouterr().out
    assert "DEPRECATED: benchmark --cv now aliases --rolling-origin" in out
    assert "# BENCHMARKING RB (ROLLING-ORIGIN)" in out
    assert calls == ["RB"]

    history_files = list((tmp_path / "history").glob("*.json"))
    assert len(history_files) == 1
    with open(history_files[0]) as f:
        entry = json.load(f)
    assert entry["mode"] == "rolling_origin"
    assert entry["results"][0]["rolling_origin"]["test_seasons"] == [2025]


def test_main_rolling_origin_with_cv_runs_once_without_deprecation(tmp_path, monkeypatch, capsys):
    _patch_benchmark_io(monkeypatch, tmp_path)
    calls = []

    def fake_rolling_origin(pos):
        calls.append(pos)
        return _rolling_summary(pos)

    monkeypatch.setattr(b, "run_rolling_origin", fake_rolling_origin)
    monkeypatch.setattr(b, "run_one", lambda *a, **k: pytest.fail("run_one must not handle --cv"))

    # --sequential pins the in-process loop (see the --cv test above).
    b.main(["RB", "--rolling-origin", "--cv", "--no-sync", "--sequential"])

    out = capsys.readouterr().out
    assert "DEPRECATED" not in out
    assert calls == ["RB"]

    history_files = list((tmp_path / "history").glob("*.json"))
    assert len(history_files) == 1
    with open(history_files[0]) as f:
        entry = json.load(f)
    assert entry["mode"] == "rolling_origin"


def test_k_rolling_origin_reuses_run_pipeline_kick_history_closure(monkeypatch):
    import src.k.data as k_data
    import src.k.features as k_features
    import src.k.run_pipeline as k_run
    import src.k.targets as k_targets

    weekly = pd.DataFrame(
        {
            "player_id": ["K1", "K1", "K1"],
            "season": [2023, 2024, 2025],
            "week": [1, 1, 1],
            "team": ["KC", "KC", "KC"],
        }
    )
    kicks = pd.DataFrame({"player_id": ["K1"], "kick_distance": [33]})
    sentinel = object()
    seen: dict = {}

    monkeypatch.setattr(k_data, "load_data", lambda: weekly.copy())
    monkeypatch.setattr(k_data, "load_kicks", lambda df: kicks)
    monkeypatch.setattr(k_targets, "compute_targets", lambda df: df)
    monkeypatch.setattr(k_features, "compute_features", lambda df: None)

    def _fake_closure(cfg, kicks_df):
        seen["cfg"] = cfg
        seen["kicks_df"] = kicks_df
        return sentinel

    monkeypatch.setattr(k_run, "_build_kick_history_closure", _fake_closure)

    frame, cfg = b._self_load_full_frame_and_cfg("K")
    assert frame.equals(weekly)
    assert cfg["attn_history_builder_fn"] is sentinel
    assert seen["cfg"] is cfg
    assert seen["kicks_df"] is kicks
