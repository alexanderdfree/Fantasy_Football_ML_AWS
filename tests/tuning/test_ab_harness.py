"""Unit tests for the shared A/B harness (src/tuning/ab_harness.py).

No real training runs here (those are the manual smoke in the PR). Coverage:
spec resolution + baseline detection, cell-grid build, device-gated jobs
autodetect (mocked platform), config deep-copy isolation, frame-injector
threading + chdir/symlink artifact isolation (with a stub ``run``), the default
metric, mean±std + Δ aggregation, and the Ridge-invariance sentinel verdicts.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from types import SimpleNamespace

import pandas as pd
import pytest

from src.tuning import ab_harness as H
from src.tuning.ab_harness import Cell, Spec, Variant

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spec(variants, *, baseline="baseline", positions=("K",), seeds=(42,), metric=None) -> Spec:
    return Spec(
        variants=OrderedDict((v.name, v) for v in variants),
        baseline=baseline,
        positions=list(positions),
        seeds=list(seeds),
        metric_fn=metric or H.default_metric_fn,
        dotted=None,
        name="t",
    )


def _fake_result(mae_offset=0.0):
    """A result dict whose ``test_df`` yields a known per-model MAE."""
    y = [10.0, 20.0, 30.0]
    pred = [v + mae_offset for v in y]
    return {
        "test_df": pd.DataFrame(
            {
                "fantasy_points": y,
                "pred_ridge_total": pred,
                "pred_lgbm_total": pred,
                "pred_nn_total": pred,
                "pred_attn_nn_total": pred,
            }
        )
    }


# --------------------------------------------------------------------------- #
# module surface
# --------------------------------------------------------------------------- #
def test_module_imports_without_side_effects():
    assert hasattr(H, "run_ab")
    assert hasattr(H, "Variant")
    assert hasattr(H, "resolve_jobs")


def test_variant_baseline_shape():
    assert Variant("b").is_baseline_shape
    assert not Variant("v", cfg_mutator=lambda c: c).is_baseline_shape
    assert not Variant("v", frame_injector=lambda *f: f).is_baseline_shape


# --------------------------------------------------------------------------- #
# spec resolution
# --------------------------------------------------------------------------- #
def test_coerce_variants_rejects_dupes_and_non_variants():
    with pytest.raises(ValueError, match="duplicate"):
        H._coerce_variants([Variant("a"), Variant("a")])
    with pytest.raises(TypeError):
        H._coerce_variants(["not a variant"])
    with pytest.raises(ValueError, match="no variants"):
        H._coerce_variants([])


def test_pick_baseline_prefers_identity_then_first():
    vs = OrderedDict(a=Variant("a", cfg_mutator=lambda c: c), b=Variant("b"))
    assert H._pick_baseline(vs, None) == "b"  # identity-shaped
    only_mutators = OrderedDict(a=Variant("a", cfg_mutator=lambda c: c))
    assert H._pick_baseline(only_mutators, None) == "a"  # fall back to first
    with pytest.raises(ValueError, match="not among"):
        H._pick_baseline(vs, "nope")


def test_resolve_spec_from_module_object():
    mod = SimpleNamespace(
        VARIANTS=[Variant("baseline"), Variant("v", cfg_mutator=lambda c: c)],
        POSITIONS=["rb", "WR"],
        SEEDS=[1, 2],
    )
    spec = H.resolve_spec(mod)
    assert spec.baseline == "baseline"
    assert spec.positions == ["RB", "WR"]  # upper-cased
    assert spec.seeds == [1, 2]
    assert spec.dotted is None  # an object isn't importable by workers


def test_resolve_spec_only_keeps_baseline():
    mod = SimpleNamespace(
        VARIANTS=[Variant("baseline"), Variant("x", cfg_mutator=lambda c: c), Variant("y")],
        POSITIONS=["K"],
    )
    spec = H.resolve_spec(mod, only=["x"])
    assert list(spec.variants) == ["baseline", "x"]
    with pytest.raises(ValueError, match="not in spec"):
        H.resolve_spec(mod, only=["zzz"])


def test_resolve_spec_rejects_unknown_position_and_empty():
    mod = SimpleNamespace(VARIANTS=[Variant("baseline")], POSITIONS=["XX"])
    with pytest.raises(ValueError, match="unknown positions"):
        H.resolve_spec(mod)
    mod2 = SimpleNamespace(VARIANTS=[Variant("baseline")])
    with pytest.raises(ValueError, match="no positions"):
        H.resolve_spec(mod2)


def test_resolve_spec_default_seeds():
    mod = SimpleNamespace(VARIANTS=[Variant("baseline")], POSITIONS=["K"])
    assert H.resolve_spec(mod).seeds == list(H.DEFAULT_SEEDS)


def test_example_spec_resolves_as_dotted():
    spec = H.resolve_spec("src.tuning.ab_example", positions=["K"], seeds=[42])
    assert spec.dotted == "src.tuning.ab_example"
    assert spec.baseline == "baseline"
    assert "+season_recency" in spec.variants


def test_history_token_spec_resolves_as_dotted():
    """The history-token A/B spec imports + resolves and keeps its design shape:
    both ``+`` arms carry a frame injector and declare ``expect_ridge_identical=
    False`` (a real feature must move Ridge), and the history arm only *adds* the
    NN-history token on top of the static arm's mutations."""
    spec = H.resolve_spec("src.tuning.ab_history_token")
    assert spec.dotted == "src.tuning.ab_history_token"
    assert spec.positions == ["RB", "WR"]
    assert set(spec.variants) == {"baseline", "+static", "+static+history"}
    for name in ("+static", "+static+history"):
        v = spec.variants[name]
        assert v.frame_injector is not None
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is False


# --------------------------------------------------------------------------- #
# cell grid
# --------------------------------------------------------------------------- #
def test_build_cells_grid_order():
    spec = _spec(
        [Variant("baseline"), Variant("v", cfg_mutator=lambda c: c)],
        positions=("RB", "WR"),
        seeds=(1, 2),
    )
    cells = H.build_cells(spec)
    assert len(cells) == 2 * 2 * 2
    assert cells[0] == Cell("RB", "baseline", 1)
    assert cells[-1] == Cell("WR", "v", 2)


# --------------------------------------------------------------------------- #
# jobs autodetect
# --------------------------------------------------------------------------- #
def _patch_platform(monkeypatch, *, cuda, backend="cuda", cpus=16, phys=16):
    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: cuda)
    monkeypatch.setattr(
        "src.shared.platform_detect.detect_platform",
        lambda: SimpleNamespace(backend=backend, cpu_count=cpus),
    )
    monkeypatch.setattr("src.benchmarking.parallel_train.physical_cores", lambda: list(range(phys)))


def test_resolve_jobs_sequential_and_explicit(monkeypatch):
    monkeypatch.delenv("FF_AB_JOBS", raising=False)
    assert H.resolve_jobs(20, sequential=True) == 1
    assert H.resolve_jobs(20, 4) == 4
    assert H.resolve_jobs(2, 4) == 2  # clamped to n_cells


def test_resolve_jobs_env_override(monkeypatch):
    _patch_platform(monkeypatch, cuda=True)
    monkeypatch.setenv("FF_AB_JOBS", "3")
    assert H.resolve_jobs(20) == 3
    monkeypatch.setenv("FF_AB_JOBS", "garbage")
    assert H.resolve_jobs(20) == 6  # falls through to CUDA autodetect


def test_resolve_jobs_cuda_caps_at_six(monkeypatch):
    monkeypatch.delenv("FF_AB_JOBS", raising=False)
    _patch_platform(monkeypatch, cuda=True)
    assert H.resolve_jobs(20) == 6
    assert H.resolve_jobs(4) == 4


def test_resolve_jobs_cpu_pool_uses_physical_cores(monkeypatch):
    monkeypatch.delenv("FF_AB_JOBS", raising=False)
    _patch_platform(monkeypatch, cuda=False, backend="cpu", cpus=16, phys=16)
    assert H.resolve_jobs(40) == 16
    assert H.resolve_jobs(5) == 5


def test_resolve_jobs_mps_and_small_box_sequential(monkeypatch):
    monkeypatch.delenv("FF_AB_JOBS", raising=False)
    _patch_platform(monkeypatch, cuda=False, backend="mps", cpus=10)
    assert H.resolve_jobs(40) == 1
    _patch_platform(monkeypatch, cuda=False, backend="cpu", cpus=4)
    assert H.resolve_jobs(40) == 1


# --------------------------------------------------------------------------- #
# config deep-copy isolation
# --------------------------------------------------------------------------- #
def test_apply_config_does_not_mutate_base():
    base = {"nn_dropout": 0.3, "get_feature_columns_fn": lambda: ["a", "b"]}

    def mutator(cfg):
        cfg["nn_dropout"] = 0.0  # in-place — must hit the copy, not the shared base
        return cfg

    out = H._apply_config(Variant("v", cfg_mutator=mutator), base)
    assert out["nn_dropout"] == 0.0
    assert base["nn_dropout"] == 0.3  # base untouched


def test_apply_config_identity_when_no_mutator():
    base = {"x": 1, "fn": lambda: 1}
    out = H._apply_config(Variant("baseline"), base)
    assert out == base
    assert out is not base  # still a copy


# --------------------------------------------------------------------------- #
# default metric
# --------------------------------------------------------------------------- #
def test_default_metric_fn_per_model():
    metrics = H.default_metric_fn(_fake_result(mae_offset=2.0), "K")
    assert metrics["Ridge"]["mae"] == pytest.approx(2.0)
    assert metrics["Ridge"]["bias"] == pytest.approx(2.0)  # over-prediction
    assert set(metrics) >= {"Ridge", "LightGBM", "NN", "Attention NN"}


# --------------------------------------------------------------------------- #
# run_cell: artifact isolation + frame injection (stub run, no training)
# --------------------------------------------------------------------------- #
def _write_tiny_splits(data_dir):
    splits = data_dir / "splits"
    splits.mkdir(parents=True)
    df = pd.DataFrame(
        {"season": [2020, 2021, 2022], "season_type": ["REG", "REG", "REG"], "x": [1.0, 2.0, 3.0]}
    )
    for name in ("train", "val", "test"):
        df.to_parquet(splits / f"{name}.parquet")


def test_run_cell_isolates_outputs_and_restores_cwd(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    orig_cwd = os.getcwd()
    captured = {}

    def stub_run(train, val, test, seed=42, config=None):
        captured["cwd"] = os.getcwd()
        captured["seed"] = seed
        # The pipeline writes here unconditionally; prove it lands in the tmp cwd.
        os.makedirs("k/outputs/models", exist_ok=True)
        with open("k/outputs/models/sentinel.txt", "w") as f:
            f.write("x")
        assert os.path.isdir("data/splits")  # data symlinked in
        return _fake_result(mae_offset=1.0)

    res = H.run_cell(
        Cell("K", "baseline", 7),
        Variant("baseline"),
        H.default_metric_fn,
        data_dir=str(data_dir),
        run_fn=stub_run,
    )
    assert res["ok"]
    assert captured["seed"] == 7
    assert res["ridge_mae"] == pytest.approx(1.0)
    # ran somewhere else, then restored
    assert captured["cwd"] != orig_cwd
    assert os.getcwd() == orig_cwd
    # nothing leaked into the working dir (the tmp cell dir was cleaned up)
    assert not (work / "k").exists()


def test_run_cell_threads_injected_frames(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def injector(train, val, test):
        for df in (train, val, test):
            df["season_recency"] = df["season"] - 2012
        return train, val, test

    def stub_run(train, val, test, seed=42, config=None):
        seen["cols"] = list(train.columns)
        seen["n"] = len(train)
        return _fake_result()

    H.run_cell(
        Cell("K", "v", 1),
        Variant("v", frame_injector=injector),
        H.default_metric_fn,
        data_dir=str(data_dir),
        run_fn=stub_run,
    )
    assert "season_recency" in seen["cols"]  # injection threaded into run()
    assert seen["n"] == 3  # REG rows loaded via the symlinked splits


def test_run_cell_no_injector_passes_none(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def stub_run(train, val, test, seed=42, config=None):
        seen["frames"] = (train, val, test)
        return _fake_result()

    H.run_cell(
        Cell("K", "baseline", 1),
        Variant("baseline"),
        H.default_metric_fn,
        data_dir=str(data_dir),
        run_fn=stub_run,
    )
    assert seen["frames"] == (None, None, None)  # run() loads internally


def test_run_cell_kdst_signature_takes_no_frames(tmp_path, monkeypatch):
    # K/DST run() is run(seed, config) — no train/val/test. A cfg-only variant
    # must call it without frame args (else "multiple values for 'seed'").
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def kdst_run(seed=42, config=None):  # the K/DST shape
        seen["seed"] = seed
        seen["cfg"] = config
        return _fake_result()

    res = H.run_cell(
        Cell("K", "v", 9),
        Variant("v", cfg_mutator=lambda c: {**c, "nn_dropout": 0.0}),
        H.default_metric_fn,
        data_dir=str(data_dir),
        run_fn=kdst_run,
    )
    assert res["ok"] and seen["seed"] == 9
    assert seen["cfg"]["nn_dropout"] == 0.0


def test_run_cell_frame_injector_on_kdst_raises(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    monkeypatch.chdir(tmp_path)

    def kdst_run(seed=42, config=None):
        return _fake_result()

    with pytest.raises(ValueError, match="can't inject"):
        H.run_cell(
            Cell("DST", "v", 1),
            Variant("v", frame_injector=lambda *f: f),
            H.default_metric_fn,
            data_dir=str(data_dir),
            run_fn=kdst_run,
        )


def test_run_cell_captures_failure(tmp_path, monkeypatch):
    # run_sequential turns an exception into a failed-cell dict; run_cell itself
    # propagates, so exercise the sequential wrapper.
    data_dir = tmp_path / "data"
    _write_tiny_splits(data_dir)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        H, "run_cell", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))
    )
    spec = _spec([Variant("baseline")])
    results = H.run_sequential(spec, [Cell("K", "baseline", 1)], str(data_dir))
    assert results[0]["ok"] is False
    assert "kaboom" in results[0]["error"]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_mean_std_handles_singletons_and_nan():
    assert H._mean_std([2.0, 4.0]) == (3.0, pytest.approx(1.4142, abs=1e-3), 2)
    mean, std, n = H._mean_std([5.0])
    assert (mean, std, n) == (5.0, 0.0, 1)
    mean, std, n = H._mean_std([None, float("nan")])
    assert n == 0 and mean != mean


def _cell_result(pos, variant, seed, ridge_mae, lgbm_mae=None):
    metrics = {"Ridge": {"mae": ridge_mae, "bias": 0.0, "rmse": ridge_mae, "n": 3}}
    if lgbm_mae is not None:
        metrics["LightGBM"] = {"mae": lgbm_mae, "bias": 0.0, "rmse": lgbm_mae, "n": 3}
    return {
        "position": pos,
        "variant": variant,
        "seed": seed,
        "label": variant,
        "ok": True,
        "metrics": metrics,
        "ridge_mae": ridge_mae,
        "error": None,
    }


def test_aggregate_mean_std_and_delta():
    spec = _spec(
        [Variant("baseline"), Variant("v", cfg_mutator=lambda c: c)],
        positions=("K",),
        seeds=(1, 2),
    )
    results = [
        _cell_result("K", "baseline", 1, 5.0, lgbm_mae=4.0),
        _cell_result("K", "baseline", 2, 5.2, lgbm_mae=4.2),
        _cell_result("K", "v", 1, 4.6, lgbm_mae=3.8),
        _cell_result("K", "v", 2, 4.8, lgbm_mae=3.6),
    ]
    agg = H.aggregate(spec, results)
    base = agg["table"]["K"]["baseline"]["Ridge"]["mae"]
    assert base["mean"] == pytest.approx(5.1)
    assert base["n"] == 2
    v = agg["table"]["K"]["v"]["Ridge"]["mae"]
    assert v["mean"] == pytest.approx(4.7)
    assert v["delta"] == pytest.approx(4.7 - 5.1)  # Δ vs baseline
    assert "delta" not in agg["table"]["K"]["baseline"]["Ridge"]["mae"]
    assert agg["n_ok"] == 4 and agg["failed"] == []


def test_aggregate_delta_when_baseline_not_first():
    # Baseline listed *second* — Δ must still resolve (order-independent pass 2).
    spec = _spec(
        [Variant("v", cfg_mutator=lambda c: c), Variant("baseline")],
        baseline="baseline",
        positions=("K",),
        seeds=(1,),
    )
    results = [_cell_result("K", "v", 1, 4.6), _cell_result("K", "baseline", 1, 5.0)]
    agg = H.aggregate(spec, results)
    assert agg["table"]["K"]["v"]["Ridge"]["mae"]["delta"] == pytest.approx(-0.4)


# --------------------------------------------------------------------------- #
# Ridge-invariance sentinel
# --------------------------------------------------------------------------- #
def test_sentinel_feature_variant_must_move_ridge():
    # expect_ridge_identical=False: identical Ridge MAE is a VIOLATION.
    spec = _spec(
        [
            Variant("baseline"),
            Variant("v", frame_injector=lambda *f: f, expect_ridge_identical=False),
        ],
        positions=("K",),
        seeds=(1,),
    )
    same = [_cell_result("K", "baseline", 1, 5.0), _cell_result("K", "v", 1, 5.0)]
    s = H.aggregate(spec, same)["sentinel"][0]
    assert s["identical"] and "VIOLATION" in s["status"]

    moved = [_cell_result("K", "baseline", 1, 5.0), _cell_result("K", "v", 1, 4.6)]
    s2 = H.aggregate(spec, moved)["sentinel"][0]
    assert not s2["identical"] and s2["status"] == "ok"
    assert s2["delta"] == pytest.approx(-0.4)


def test_sentinel_nn_only_variant_must_not_move_ridge():
    spec = _spec(
        [Variant("baseline"), Variant("v", cfg_mutator=lambda c: c, expect_ridge_identical=True)],
        positions=("K",),
        seeds=(1,),
    )
    moved = [_cell_result("K", "baseline", 1, 5.0), _cell_result("K", "v", 1, 5.3)]
    s = H.aggregate(spec, moved)["sentinel"][0]
    assert not s["identical"] and "VIOLATION" in s["status"]

    same = [_cell_result("K", "baseline", 1, 5.0), _cell_result("K", "v", 1, 5.0)]
    s2 = H.aggregate(spec, same)["sentinel"][0]
    assert s2["identical"] and s2["status"] == "ok"


def test_sentinel_report_only_never_violates():
    spec = _spec(
        [Variant("baseline"), Variant("v", frame_injector=lambda *f: f)],  # expect=None
        positions=("K",),
        seeds=(1,),
    )
    for ridge in (5.0, 4.6):
        s = H.aggregate(
            spec, [_cell_result("K", "baseline", 1, 5.0), _cell_result("K", "v", 1, ridge)]
        )["sentinel"][0]
        assert s["status"] == "ok"


# --------------------------------------------------------------------------- #
# reporting + CLI smoke (no training)
# --------------------------------------------------------------------------- #
def test_print_report_runs(capsys):
    spec = _spec(
        [
            Variant("baseline"),
            Variant("v", frame_injector=lambda *f: f, expect_ridge_identical=False),
        ],
        positions=("K",),
        seeds=(1,),
    )
    agg = H.aggregate(
        spec,
        [
            _cell_result("K", "baseline", 1, 5.0, lgbm_mae=4.0),
            _cell_result("K", "v", 1, 5.0, lgbm_mae=3.9),
        ],
    )
    H.print_report(spec, agg, jobs=2)
    out = capsys.readouterr().out
    assert "Ridge-invariance sentinel" in out
    assert "feature cache DISABLED" in out


def test_cli_list_mode(capsys):
    rc = H.main(["--spec", "src.tuning.ab_example", "--positions", "K", "--seeds", "42", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "6 cells" not in out  # 1 pos × 3 variants × 1 seed = 3
    assert "3 cells" in out


def test_cli_requires_spec(capsys):
    assert H.main([]) == 2
    assert "spec" in capsys.readouterr().err


def test_run_ab_sets_and_restores_cache_env(tmp_path, monkeypatch):
    # The feature cache is disabled by default and re-enabled by feature_cache=True,
    # set explicitly during the run (so workers inherit) and restored afterwards.
    captured = {}

    def fake_seq(spec, cells, data_dir):
        captured["during"] = os.environ.get("FF_FEATURE_CACHE_DISABLE")
        return []

    monkeypatch.setattr(H, "run_sequential", fake_seq)
    monkeypatch.delenv("FF_FEATURE_CACHE_DISABLE", raising=False)
    mod = SimpleNamespace(VARIANTS=[Variant("baseline")], POSITIONS=["K"], SEEDS=[1])

    H.run_ab(mod, sequential=True, data_dir=str(tmp_path))
    assert captured["during"] == "1"  # disabled by default
    assert "FF_FEATURE_CACHE_DISABLE" not in os.environ  # restored (was unset)

    H.run_ab(mod, sequential=True, feature_cache=True, data_dir=str(tmp_path))
    assert captured["during"] == "0"  # --feature-cache re-enables
    assert "FF_FEATURE_CACHE_DISABLE" not in os.environ


# --------------------------------------------------------------------------- #
# ab_example template (pure variant fns — lock the copy-me behaviour)
# --------------------------------------------------------------------------- #
def test_example_inject_adds_season_recency():
    from src.config import SEASONS
    from src.tuning import ab_example as ex

    frames = tuple(pd.DataFrame({"season": [SEASONS[0], SEASONS[0] + 3]}) for _ in range(3))
    train, _val, _test = ex._inject_season_recency(*frames)
    assert list(train["season_recency"]) == [0.0, 3.0]


def test_example_whitelist_extends_both_paths_without_mutating_base():
    from src.tuning import ab_example as ex

    base_cols = ["a", "b"]
    cfg = {"get_feature_columns_fn": lambda: list(base_cols), "attn_static_features": ["a"]}
    out = ex._whitelist_season_recency(cfg)
    assert out["get_feature_columns_fn"]()[-1] == "season_recency"
    assert "season_recency" in out["attn_static_features"]
    # a position without attn_static_features (rare) must not crash
    assert ex._whitelist_season_recency({"get_feature_columns_fn": lambda: ["x"]})


def test_example_zero_dropout():
    from src.tuning import ab_example as ex

    assert ex._zero_dropout({"nn_dropout": 0.3})["nn_dropout"] == 0.0


def test_example_variants_declare_sentinel_expectations():
    from src.tuning import ab_example as ex

    by_name = {v.name: v for v in ex.VARIANTS}
    assert by_name["+season_recency"].expect_ridge_identical is False  # feature MUST move Ridge
    assert by_name["nn_dropout=0"].expect_ridge_identical is True  # NN-only MUST NOT


# --------------------------------------------------------------------------- #
# Stacked-seeds mode (E2): groups, group runner contract, orchestration
# --------------------------------------------------------------------------- #
def test_build_stacked_units_split():
    """Flat-history positions become groups; K/DST fall back to eager cells."""
    spec = _spec(
        [Variant("baseline"), Variant("v", cfg_mutator=lambda c: c)],
        positions=("QB", "K"),
        seeds=(42, 123),
    )
    groups, cells = H.build_stacked_units(spec)
    assert [(g.position, g.variant, g.seeds) for g in groups] == [
        ("QB", "baseline", (42, 123)),
        ("QB", "v", (42, 123)),
    ]
    assert {(c.position, c.variant, c.seed) for c in cells} == {
        ("K", "baseline", 42), ("K", "baseline", 123), ("K", "v", 42), ("K", "v", 123),
    }  # fmt: skip


def _rb_member_preds(n, scale):
    """Per-member predict_stacked stub output: all RB raw targets, constant."""
    import numpy as np

    targets = (
        "rushing_yards", "receiving_yards", "rushing_tds",
        "receiving_tds", "receptions", "fumbles_lost",
    )  # fmt: skip
    return {t: np.full(n, scale if t == "rushing_yards" else 0.0) for t in targets}


def _stacked_metric_fn(result, position):
    df = result["test_df"]
    attn = float((df["pred_attn_nn_total"] - df["fantasy_points"]).abs().mean())
    return {"Ridge": {"mae": 1.5}, "Attention NN": {"mae": attn}}


def test_run_group_stacked_contract(tmp_path, monkeypatch):
    """Phase A runs ONCE (attention off), Phase B supplies per-member attention
    preds; per-seed dicts carry recomputed metrics, a constant ridge_mae, and
    the ensemble env is restored afterwards."""
    import numpy as np

    seen = {"runs": [], "captures": []}

    def stub_run(train, val, test, *, seed, config):
        seen["runs"].append((train, seed, config["train_attention_nn"]))
        return {
            "test_df": pd.DataFrame(
                {"fantasy_points": [0.0, 0.0, 0.0], "pred_ridge_total": [1.0, 2.0, 3.0]}
            )
        }

    def stub_capture(position, seeds, base_cfg, *, frames=None, memo=None):
        seen["captures"].append((position, tuple(seeds), base_cfg["train_attention_nn"], frames))
        return ["cap0", "cap1"], {"args": None}

    monkeypatch.setattr("src.tuning.ab_ensemble_seeds.capture_seeds", stub_capture)
    monkeypatch.setattr(
        "src.tuning.ab_ensemble_seeds.train_stacked", lambda *a, **k: (None, None, None)
    )
    monkeypatch.setattr(
        "src.tuning.ab_ensemble_seeds.predict_stacked",
        lambda *a, **k: [_rb_member_preds(3, 0.0), _rb_member_preds(3, 10.0)],
    )
    monkeypatch.delenv("FF_NN_NORM", raising=False)

    group = H.Group("RB", "v", (42, 123))
    cwd = os.getcwd()
    results = H.run_group_stacked(
        group,
        Variant("v"),
        _stacked_metric_fn,
        data_dir=str(tmp_path),
        stacked_epochs=3,
        run_fn=stub_run,
    )
    assert os.getcwd() == cwd
    assert "FF_NN_NORM" not in os.environ  # ensemble env restored

    # Phase A: exactly one real run, seed=seeds[0], attention disabled, no frames.
    assert seen["runs"] == [(None, 42, False)]
    # Capture got the variant cfg with attention still enabled and no frames.
    assert seen["captures"] == [("RB", (42, 123), True, None)]

    assert [r["seed"] for r in results] == [42, 123]
    assert all(r["ok"] and r["stacked"] for r in results)
    assert {r["ridge_mae"] for r in results} == {1.5}
    # Member 0 predicts 0 rushing yards -> FP 0 -> MAE 0; member 1 predicts
    # 10 rushing yards -> 1.0 FP per row -> MAE 1.0 (PPR: yards/10).
    assert results[0]["metrics"]["Attention NN"]["mae"] == pytest.approx(0.0)
    assert results[1]["metrics"]["Attention NN"]["mae"] == pytest.approx(1.0)
    assert isinstance(results[0]["metrics"]["Attention NN"]["mae"], float)
    assert np is not None  # keep the local import honest


def test_run_group_stacked_threads_frames(tmp_path, monkeypatch):
    """A frame-injector variant's frames reach BOTH Phase A and the captures."""
    injected = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(H, "_load_general_splits", lambda: (injected, injected, injected))
    seen = {}

    def stub_run(train, val, test, *, seed, config):
        seen["phase_a_frames"] = (train is not None, val is not None, test is not None)
        return {"test_df": pd.DataFrame({"fantasy_points": [0.0], "pred_ridge_total": [1.0]})}

    def stub_capture(position, seeds, base_cfg, *, frames=None, memo=None):
        seen["capture_frames"] = frames is not None
        return ["c"], {"args": None}

    monkeypatch.setattr("src.tuning.ab_ensemble_seeds.capture_seeds", stub_capture)
    monkeypatch.setattr(
        "src.tuning.ab_ensemble_seeds.train_stacked", lambda *a, **k: (None, None, None)
    )
    monkeypatch.setattr(
        "src.tuning.ab_ensemble_seeds.predict_stacked", lambda *a, **k: [_rb_member_preds(1, 0.0)]
    )

    variant = Variant("inj", frame_injector=lambda tr, va, te: (tr, va, te))
    H.run_group_stacked(
        H.Group("RB", "inj", (42,)),
        variant,
        _stacked_metric_fn,
        data_dir=str(tmp_path),
        stacked_epochs=2,
        run_fn=stub_run,
    )
    assert seen["phase_a_frames"] == (True, True, True)
    assert seen["capture_frames"] is True


def test_run_sequential_stacked_flattens_and_isolates_failures(monkeypatch):
    """Group results flatten per seed; a failing group yields per-seed failure
    dicts without sinking the run; leftover eager cells still execute."""
    spec = _spec(
        [Variant("baseline"), Variant("v", cfg_mutator=lambda c: c)],
        positions=("QB", "K"),
        seeds=(42, 123),
    )
    groups, cells = H.build_stacked_units(spec)

    def fake_group(group, variant, metric_fn, *, data_dir, stacked_epochs):
        if group.variant == "v":
            raise RuntimeError("boom")
        return [
            {
                "position": group.position,
                "variant": group.variant,
                "seed": s,
                "label": group.variant,
                "ok": True,
                "metrics": {},
                "ridge_mae": 1.0,
                "error": None,
                "stacked": True,
            }  # fmt: skip
            for s in group.seeds
        ]

    monkeypatch.setattr(H, "run_group_stacked", fake_group)
    monkeypatch.setattr(
        H,
        "run_cell",
        lambda cell, variant, metric_fn, *, data_dir, run_fn=None: {
            "position": cell.position,
            "variant": cell.variant,
            "seed": cell.seed,
            "label": cell.variant,
            "ok": True,
            "metrics": {},
            "ridge_mae": 2.0,
            "error": None,
        },  # fmt: skip
    )
    results = H.run_sequential_stacked(spec, groups, cells, data_dir="data", stacked_epochs=3)
    assert len(results) == 8  # 2 groups x 2 seeds + 4 eager K cells
    ok_by_variant = {(r["variant"], r["ok"]) for r in results if r["position"] == "QB"}
    assert ok_by_variant == {("baseline", True), ("v", False)}
    assert all(r["ok"] for r in results if r["position"] == "K")
    failed = [r for r in results if not r["ok"]]
    assert {r["seed"] for r in failed} == {42, 123} and all("boom" in r["error"] for r in failed)
