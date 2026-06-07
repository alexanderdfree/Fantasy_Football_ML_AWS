"""Coverage tests for ``app.py::_apply_position_models``.

The serving path in ``_apply_position_models`` loads Ridge + NN (+ attention
+ LightGBM) model artifacts from disk and writes per-position predictions
into the shared results DataFrame. It's the largest single uncovered block
in ``app.py`` (~220 stmts) because real artifacts only exist post-training.

These tests stub the model loaders with lightweight fakes:

- ``RidgeMultiTarget`` / ``LightGBMMultiTarget`` → fake classes whose
  ``.load()`` / ``.predict()`` return per-target zero arrays.
- ``joblib.load`` → tiny ``StandardScaler`` fitted on a throwaway matrix.
- ``torch.load`` → ``{"model_state": {}, "feature_columns_hash": "..."}``.
- ``MultiHeadNet`` / ``MultiHeadNetWithHistory`` / ``MultiHeadNetWithNestedHistory``
  → fake classes whose ``.load_state_dict`` no-ops and ``.predict_numpy``
  returns the target → zeros dict shape.
- ``assert_scaler_matches`` → no-op (integrity check is tested separately
  in ``tests/shared/test_model_sync.py``).

The goal is branch coverage of the function body, not numerical correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.serving.core as core
import src.serving.routes as routes

# Reuse QB as the canonical "flat-history" position + DST for adjustment_fn.
# The function's per-position branches are structurally identical aside from
# the attention-history-structure fork (nested for K, flat for others).


def _make_results_frame(n: int = 12) -> pd.DataFrame:
    """Empty results frame matching the shape app.py builds.

    Includes the per-target ``actual_{t}`` / ``pred_{model}_{t}`` columns the
    production base frame pre-declares (see ``_load_base_data_locked``) so the
    breakdown-persistence assertions have columns to land in.
    """
    import src.serving.app as app_mod

    frame = pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(n)],
            "position": ["QB"] * n,
            "recent_team": ["KC"] * n,
            "season": [2025] * n,
            "week": list(range(1, n + 1)),
            "fantasy_points": np.linspace(10, 30, n),
            "ridge_pred": [np.nan] * n,
            "nn_pred": [np.nan] * n,
            "attn_nn_pred": [np.nan] * n,
            "lgbm_pred": [np.nan] * n,
        }
    )
    per_target_cols = [f"actual_{t}" for t in app_mod._ALL_TARGETS] + [
        f"pred_{prefix}_{t}"
        for t in app_mod._ALL_TARGETS
        for prefix in app_mod._MODEL_PRED_PREFIXES
    ]
    return pd.concat(
        [frame, pd.DataFrame(np.nan, index=frame.index, columns=per_target_cols)],
        axis=1,
    )


def _make_df(n: int = 12) -> pd.DataFrame:
    """Synthetic QB DataFrame that survives the pipeline's feature build."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "player_id": [f"P{i}" for i in range(n)],
            "player_display_name": [f"Player {i}" for i in range(n)],
            "position": ["QB"] * n,
            "recent_team": ["KC"] * n,
            "opponent_team": ["BUF"] * n,
            "season": [2025] * n,
            "week": list(range(1, n + 1)),
            "passing_yards": rng.uniform(150, 400, n),
            "rushing_yards": rng.uniform(0, 60, n),
            "passing_tds": rng.uniform(0, 4, n),
            "rushing_tds": rng.uniform(0, 2, n),
            "interceptions": rng.uniform(0, 3, n),
            "fumbles_lost": rng.uniform(0, 2, n),
            "sack_fumbles_lost": np.zeros(n),
            "rushing_fumbles_lost": np.zeros(n),
            "receiving_fumbles_lost": np.zeros(n),
            "fantasy_points": np.linspace(10, 30, n),
        }
    )


@pytest.fixture()
def _mocked_app(monkeypatch):
    """Stub every model loader in ``app.py`` with lightweight fakes.

    Returns the module under test for direct function invocation.
    """
    import src.serving.app as app_mod

    # Fake Ridge/LGBM: zero predictions for every target.
    class _FakeMultiTarget:
        def __init__(self, target_names, **kwargs):
            self.target_names = target_names

        def load(self, model_dir):
            self.loaded_from = model_dir

        def predict(self, X):
            n = len(X)
            return {t: np.zeros(n, dtype=np.float32) for t in self.target_names}

    monkeypatch.setattr(core, "RidgeMultiTarget", _FakeMultiTarget)
    monkeypatch.setattr(core, "LightGBMMultiTarget", _FakeMultiTarget)

    # Fake scaler loader — returns a StandardScaler fitted on a 1-feature dummy.
    from sklearn.preprocessing import StandardScaler

    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.zeros((2, 1), dtype=np.float32))

    def _fake_joblib_load(path):
        return dummy_scaler

    monkeypatch.setattr(core.joblib, "load", _fake_joblib_load)

    # Fake torch.load — returns an empty state-dict checkpoint.
    monkeypatch.setattr(
        core.torch,
        "load",
        lambda *args, **kwargs: {"model_state": {}, "feature_columns_hash": "dead"},
    )

    # Skip scaler-matches integrity check (covered elsewhere).
    monkeypatch.setattr(core, "assert_scaler_matches", lambda *a, **k: None)
    monkeypatch.setattr(core, "read_scaler_meta", lambda *a, **k: {})
    monkeypatch.setattr(
        core, "unwrap_state_dict", lambda checkpoint: (checkpoint.get("model_state", {}), "hash")
    )
    # scale_and_clip just pads/clips the input — passthrough is fine here.
    monkeypatch.setattr(core, "scale_and_clip", lambda scaler, X: np.asarray(X, dtype=np.float32))

    # Fake NN classes: .predict_numpy returns target → zeros.
    class _FakeNN:
        def __init__(self, *args, **kwargs):
            self.target_names = kwargs.get("target_names", [])

        def to(self, device):
            return self

        def load_state_dict(self, sd):
            pass

        def predict_numpy(self, *args, **kwargs):
            # Flat history NN gets (X, device); attention gets (X, hist, mask, device)
            X = args[0]
            n = len(X)
            return {t: np.zeros(n, dtype=np.float32) for t in self.target_names}

    monkeypatch.setattr(core, "MultiHeadNet", _FakeNN)
    monkeypatch.setattr(core, "MultiHeadNetWithHistory", _FakeNN)
    monkeypatch.setattr(core, "MultiHeadNetWithNestedHistory", _FakeNN)

    # Short-circuit build_position_features — keep the per-position DataFrame
    # as passed in, with a dummy numeric feature column.
    def _fake_build_features(tr, va, te, reg, feature_cols):
        for df in (tr, va, te):
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        return tr, va, te

    monkeypatch.setattr(core, "build_position_features", _fake_build_features)

    # Feature history builders (attention path)
    monkeypatch.setattr(
        core,
        "build_game_history_arrays",
        lambda df, history_stats, max_seq_len: (
            np.zeros((len(df), max_seq_len, max(1, len(history_stats))), dtype=np.float32),
            np.zeros((len(df), max_seq_len), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        core,
        "get_attn_static_columns",
        lambda feature_cols, allow: (
            [c for c in feature_cols if c in set(allow)][:1] or feature_cols[:1]
        ),
    )
    return app_mod


@pytest.fixture()
def _qb_registry(monkeypatch):
    """Swap ``POSITION_REGISTRY['QB']`` for a minimal stub that drives
    the ridge + nn + lgbm branches but not attention (exercised separately)."""
    import src.serving.app as app_mod

    reg = {
        "targets": ["passing_yards", "rushing_yards"],
        "specific_features": [],
        "filter_fn": lambda df: df[df["position"] == "QB"].copy(),
        "compute_targets_fn": lambda df: df.assign(fumbles_lost=0.0),
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["f0"],
        "aggregate_fn": lambda preds: sum(preds[t] for t in ("passing_yards", "rushing_yards")),
        "model_dir": "src/qb/outputs/models",
        "nn_file": "qb_multihead_nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": False,
        "attn_nn_file": "qb_attention_nn.pt",
        "attn_nn_kwargs_static": {},
        "attn_history_stats": [],
        "attn_static_features": [],
        "attn_max_seq_len": 17,
        "train_lightgbm": True,
    }

    # core.POSITION_REGISTRY is a _LazyInferenceRegistry — stub the __getitem__.
    class _StubRegistry:
        def __getitem__(self, pos):
            return reg if pos == "QB" else POSITION_REGISTRY[pos]

        def __contains__(self, pos):
            return True

    from src.shared.registry import INFERENCE_REGISTRY as POSITION_REGISTRY

    monkeypatch.setattr(core, "POSITION_REGISTRY", _StubRegistry())
    return reg


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_position_models_qb_flat_path(_mocked_app, _qb_registry):
    """QB path: Ridge + flat NN + LightGBM, no attention, no adjustment_fn.

    Verifies results frame gets ridge_pred/nn_pred/lgbm_pred written, and
    per-target metrics land in _cache['position_details']['QB'].
    """
    results = _make_results_frame(n=12)
    df = _make_df(n=12)
    _mocked_app._cache.clear()

    core._apply_position_models(df, df, df, "QB", results)

    # Every row's ridge/nn/lgbm_pred must be populated (zeros, per our stubs).
    assert results["ridge_pred"].notna().all()
    assert results["nn_pred"].notna().all()
    assert results["lgbm_pred"].notna().all()
    # Attention wasn't enabled → attn_nn_pred stays NaN.
    assert results["attn_nn_pred"].isna().all()

    # Per-target raw-stat columns written for the QB stub's targets (the
    # breakdown drill-down reads these). The stub registry's targets are
    # passing_yards + rushing_yards.
    for t in ("passing_yards", "rushing_yards"):
        assert results[f"actual_{t}"].notna().all()
        assert results[f"pred_ridge_{t}"].notna().all()
        assert results[f"pred_nn_{t}"].notna().all()
        assert results[f"pred_lgbm_{t}"].notna().all()
    # attn_nn disabled here → its per-target column stays NaN.
    assert results["pred_attn_nn_passing_yards"].isna().all()
    # A target NOT in the QB stub registry is untouched (stays NaN).
    assert results["actual_receptions"].isna().all()

    details = _mocked_app._cache["position_details"]["QB"]
    assert details["n_features"] == 1
    assert details["n_samples_test"] == 12
    assert "total" in details["target_metrics"]


@pytest.mark.integration
def test_apply_position_models_applies_min_games_filter(monkeypatch):
    """Serving must replicate training's min-games filter on ``pos_train`` so
    the ``fill_nans`` train-means + the StandardScaler match what the loaded
    models were trained on (audit #569). The filter drops low-volume
    player-seasons from the TRAIN frame ONLY; val/test stay unfiltered.

    Capture point is ``build_position_features`` — the frame it receives is
    exactly the post-filter ``pos_train``. Pre-fix this frame still contains
    the low-volume player (test fails); post-fix it is dropped.
    """
    import src.serving.app as app_mod

    captured = {}

    class _Stop(Exception):
        pass

    def _capture_build(tr, va, te, reg, feature_cols):
        captured["train_ids"] = list(tr["player_id"])
        captured["val_ids"] = list(va["player_id"])
        raise _Stop

    monkeypatch.setattr(core, "build_position_features", _capture_build)

    reg = {
        "targets": ["passing_yards"],
        "specific_features": [],
        "min_games_per_season": 2,
        "filter_fn": lambda df: df[df["position"] == "QB"].copy(),
        "compute_targets_fn": lambda df: df,
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["f0"],
        "model_dir": "src/qb/outputs/models",
        "nn_file": "qb_multihead_nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": False,
        "train_lightgbm": False,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(core, "POSITION_REGISTRY", _Stub())

    # HIGH: 3 games in 2025 (>= min_games=2). LOW: 1 game (< 2, must be dropped
    # from train). LOW appears in val with 1 game too — val must NOT be filtered.
    train = pd.DataFrame(
        {
            "player_id": ["HIGH", "HIGH", "HIGH", "LOW"],
            "position": ["QB"] * 4,
            "recent_team": ["KC"] * 4,
            "season": [2025] * 4,
            "week": [1, 2, 3, 1],
            "passing_yards": [300.0, 310.0, 290.0, 250.0],
        }
    )
    val = pd.DataFrame(
        {
            "player_id": ["LOW"],
            "position": ["QB"],
            "recent_team": ["KC"],
            "season": [2025],
            "week": [1],
            "passing_yards": [200.0],
        }
    )
    app_mod._cache.clear()

    with pytest.raises(_Stop):
        core._apply_position_models(train, val, train, "QB", pd.DataFrame())

    # LOW's single-game 2025 season is filtered out of TRAIN.
    assert captured["train_ids"] == ["HIGH", "HIGH", "HIGH"]
    # val is NOT min-games-filtered (mirrors training: filter applies to train only).
    assert captured["val_ids"] == ["LOW"]


@pytest.mark.integration
def test_inference_spec_min_games_and_k_all_features():
    """``get_inference_spec`` exposes the per-position ``min_games_per_season``
    (audit #569) and routes K through ``all_features`` so serving train-mean-
    fills the PBP-derived weather columns exactly as training does (audit #672).
    """
    from src.k.config import POSITION_CONFIG as K_CFG
    from src.rb.config import POSITION_CONFIG as RB_CFG
    from src.shared.registry import get_inference_spec

    # K: specific_features must equal all_features (specific + contextual).
    k_spec = get_inference_spec("K")
    assert list(k_spec["specific_features"]) == list(K_CFG.all_features)
    assert k_spec["min_games_per_season"] == K_CFG.min_games_per_season

    # Non-K (RB): specific_features stays the position-specific block; the
    # per-position min-games override (1) is surfaced, not the global default.
    rb_spec = get_inference_spec("RB")
    assert list(rb_spec["specific_features"]) == list(RB_CFG.specific_features)
    assert rb_spec["min_games_per_season"] == RB_CFG.min_games_per_season == 1


@pytest.mark.integration
def test_apply_position_models_with_attention(_mocked_app, monkeypatch):
    """When reg['train_attention_nn'] is True, the attention branch fires."""
    import src.serving.app as app_mod

    reg = {
        "targets": ["passing_yards"],
        "specific_features": [],
        "filter_fn": lambda df: df[df["position"] == "QB"].copy(),
        "compute_targets_fn": lambda df: df,
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["f0"],
        "aggregate_fn": lambda preds: preds["passing_yards"],
        "model_dir": "src/qb/outputs/models",
        "nn_file": "qb_multihead_nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": True,
        "attn_nn_file": "qb_attention_nn.pt",
        "attn_nn_kwargs_static": {},
        "attn_history_stats": ["passing_yards"],
        "attn_static_features": ["f0"],
        "attn_max_seq_len": 17,
        "train_lightgbm": False,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(core, "POSITION_REGISTRY", _Stub())

    results = _make_results_frame(n=6)
    df = _make_df(n=6)
    _mocked_app._cache.clear()

    core._apply_position_models(df, df, df, "QB", results)

    assert results["attn_nn_pred"].notna().all()
    assert results["lgbm_pred"].isna().all()  # lgbm disabled


@pytest.mark.integration
def test_apply_position_models_with_adjustment_fn(_mocked_app, monkeypatch):
    """``compute_adjustment_fn`` (used by DST) must get applied to totals."""
    import src.serving.app as app_mod

    reg = {
        "targets": ["points_allowed"],
        "specific_features": [],
        "filter_fn": lambda df: df[df["position"] == "QB"].copy(),
        "compute_targets_fn": lambda df: df.assign(points_allowed=20.0),
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["f0"],
        "compute_adjustment_fn": lambda df: pd.Series(np.ones(len(df)) * 5.0, index=df.index),
        "model_dir": "src/qb/outputs/models",
        "nn_file": "fake.pt",
        "nn_kwargs": {},
        "train_attention_nn": False,
        "train_lightgbm": False,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(core, "POSITION_REGISTRY", _Stub())

    results = _make_results_frame(n=6)
    df = _make_df(n=6)
    _mocked_app._cache.clear()

    core._apply_position_models(df, df, df, "QB", results)
    # With adjustment_fn adding 5 to predictions that would otherwise be 0.
    assert (results["ridge_pred"] == 5.0).all()


@pytest.mark.integration
def test_compute_metrics_locked_populates_cache(monkeypatch):
    """``_compute_metrics_locked`` computes overall + per-position metrics
    for every model whose prediction column has any non-NaN row, and caches
    them per scoring format under ``_cache['metrics_by_format']`` (with
    ``_cache['metrics']`` mirroring the PPR slot)."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    n = 30
    rng = np.random.default_rng(7)
    # Real fantasy_points + ridge/nn predictions; attn_nn only partial; lgbm all NaN.
    actual = rng.uniform(5, 25, n)
    ridge = rng.uniform(5, 25, n)
    nn = rng.uniform(5, 25, n)
    attn = np.array(list(rng.uniform(5, 25, 20)) + [np.nan] * 10)
    lgbm = np.full(n, np.nan)
    results = pd.DataFrame(
        {
            "position": (["QB"] * 10 + ["RB"] * 10 + ["WR"] * 10),
            # `week` isn't used by metric math, but _persist_cache_to_disk now
            # also writes snapshot.json via _records_to_player_rows, which
            # requires it — match the production results shape.
            "week": ([1] * 10 + [2] * 10 + [3] * 10),
            "fantasy_points": actual,
            "fantasy_points_half_ppr": actual * 0.95,
            "fantasy_points_standard": actual * 0.9,
        }
    )
    for fmt, m in (("ppr", 1.0), ("half_ppr", 0.95), ("standard", 0.9)):
        results[f"ridge_pred_{fmt}"] = ridge * m
        results[f"nn_pred_{fmt}"] = nn * m
        results[f"attn_nn_pred_{fmt}"] = attn * m
        results[f"lgbm_pred_{fmt}"] = lgbm
    app_mod._cache["results"] = results

    core._compute_metrics_locked()
    metrics = app_mod._cache["metrics"]
    metrics_by_format = app_mod._cache["metrics_by_format"]
    # Per-format cache populated for every scoring format.
    assert set(metrics_by_format) == {"ppr", "half_ppr", "standard"}
    # Legacy 'metrics' key mirrors the PPR slot.
    assert metrics is metrics_by_format["ppr"]
    assert "Ridge Regression" in metrics
    assert "Neural Network" in metrics
    # Ridge + NN have overall metrics; LGBM (all NaN) has None overall + [] by_position.
    assert metrics["Ridge Regression"]["overall"] is not None
    assert metrics["LightGBM"]["overall"] is None
    assert metrics["LightGBM"]["by_position"] == []
    # By-position breakdown lists each position present.
    ridge_positions = {r["position"] for r in metrics["Ridge Regression"]["by_position"]}
    assert ridge_positions == {"QB", "RB", "WR"}


@pytest.mark.integration
def test_safe_num_handles_nan_inf_and_none():
    """_safe_num converts NaN/inf/None/non-numeric to None; finite floats pass
    through unchanged."""
    from src.serving.app import _safe_num

    assert _safe_num(None) is None
    assert _safe_num(float("nan")) is None
    assert _safe_num(float("inf")) is None
    assert _safe_num(float("-inf")) is None
    assert _safe_num("not a number") is None
    assert _safe_num(3.14) == 3.14
    assert _safe_num(0) == 0.0


@pytest.mark.integration
def test_safe_str_handles_nan_and_none():
    """_safe_str falls back to the default on None/NaN and str-converts numbers."""
    from src.serving.app import _safe_str

    assert _safe_str(None, default="X") == "X"
    assert _safe_str(float("nan"), default="Y") == "Y"
    assert _safe_str("hello") == "hello"
    assert _safe_str(42) == "42"


@pytest.mark.integration
def test_compute_scoring_formats_adds_both_columns():
    """``_compute_scoring_formats`` adds standard + half-PPR columns when
    they're missing. Already-present columns must be preserved."""
    import src.serving.app as app_mod

    df = pd.DataFrame(
        {
            "passing_yards": [300.0, 250.0],
            "passing_tds": [2, 1],
            "interceptions": [1, 0],
            "rushing_yards": [20.0, 10.0],
            "rushing_tds": [0, 0],
            "receiving_yards": [0.0, 0.0],
            "receiving_tds": [0, 0],
            "receptions": [0, 0],
            "sack_fumbles_lost": [1, 0],
            "rushing_fumbles_lost": [0, 0],
            "receiving_fumbles_lost": [0, 0],
        }
    )
    core._compute_scoring_formats(df)
    assert "fantasy_points_standard" in df.columns
    assert "fantasy_points_half_ppr" in df.columns
    # Standard scoring should differ from half_ppr only if reception counts > 0
    # — here receptions are 0, so the two are equal.
    assert (df["fantasy_points_standard"] == df["fantasy_points_half_ppr"]).all()


@pytest.mark.integration
def test_categorize_features_buckets_known_prefixes():
    """``_categorize_features`` sorts feature names into buckets by prefix."""
    import src.serving.app as app_mod

    feats = [
        "rolling_mean_yards",
        "prior_season_max_yards",
        "ewma_3_yards",
        "trend_slope_yards",
        "yards_trend",
        "target_share_L5",
        "hhi_targets",
        "opp_def_rank",
        "opp_pass_rate",
        "is_home",  # contextual
        "something_random",
    ]
    cats = routes._categorize_features(feats)
    assert "rolling_mean_yards" in cats.get("rolling", [])
    assert "prior_season_max_yards" in cats.get("prior_season", [])
    assert "ewma_3_yards" in cats.get("ewma", [])
    assert "trend_slope_yards" in cats.get("trend", [])
    assert "yards_trend" in cats.get("trend", [])
    assert "target_share_L5" in cats.get("share", [])
    assert "hhi_targets" in cats.get("share", [])
    # opp_def and opp_pass both fall into "defense" or "matchup" depending on
    # which branch fires first — we just assert the feature landed somewhere.
    assert any("opp_def_rank" in v for v in cats.values())
    assert any("opp_pass_rate" in v for v in cats.values())
    # is_home is in _CONTEXTUAL_FEATURES so it goes to contextual
    assert any("is_home" in v for v in cats.values())
    assert any("something_random" in v for v in cats.values())


@pytest.mark.integration
def test_health_route_degraded_returns_200_when_some_positions_loaded(monkeypatch):
    """/health returns 200 + ``status="degraded"`` when at least one position
    is loaded, even if ``position_load_errors`` has per-model entries.

    Returning 503 here would deregister the ALB target (interval=10s,
    unhealthy_threshold=3 ⇒ ~30s) and ECS would replace the task — but the
    task is still serving five of six positions cleanly. ``/api/predictions``
    already returns 200 + ``degraded_positions`` banner for this exact state;
    ``/health`` must agree. The alexfree.me 2026-05-21 12:16 UTC outage was
    caused by the previous over-eager 503-on-any-error contract.
    """
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["positions_loaded"] = {"QB", "RB", "WR", "TE", "K"}
    app_mod._cache["position_load_errors"] = {"DST_ridge": "missing artifact"}
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "degraded"
        assert "DST_ridge" in body["position_load_errors"]
        assert set(body["positions_loaded"]) == {"QB", "RB", "WR", "TE", "K"}


@pytest.mark.integration
def test_health_route_503_when_no_positions_loaded(monkeypatch):
    """/health returns 503 only when ``positions_loaded`` is empty — the genuine
    "cannot serve anything" state. ALB rotates us out and ECS replaces the
    task; that's the right response when the task truly has nothing to offer.
    """
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["positions_loaded"] = set()
    app_mod._cache["position_load_errors"] = {
        "QB": "boom",
        "RB": "boom",
        "WR": "boom",
        "TE": "boom",
        "K": "boom",
        "DST": "boom",
    }
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.get_json()
        assert body["status"] == "unhealthy"
        assert "QB" in body["position_load_errors"]


@pytest.mark.integration
def test_health_route_ok_when_no_errors(monkeypatch):
    """/health returns 200 + ``status="ok"`` (no degraded body) when there are
    no recorded errors and at least one position is loaded — the steady-state
    happy path."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["positions_loaded"] = {"QB", "RB", "WR", "TE", "K", "DST"}
    # Explicitly no position_load_errors key.
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body == {"status": "ok"}


@pytest.mark.integration
def test_apply_position_models_ridge_load_failure_records_and_nan_fills(_mocked_app, monkeypatch):
    """A RidgeMultiTarget.load exception is recorded in
    ``_cache['position_load_errors']`` under ``{pos}_ridge`` and the position's
    ``ridge_pred`` rows are NaN'd — but the function does NOT raise.

    This is the Part B graceful-degradation contract: one model's failure must
    not take down the position. ``_ensure_position_loaded`` still marks the
    position loaded (with degraded preds); ``_degraded_positions`` surfaces it
    for the frontend banner.
    """
    import src.serving.app as app_mod

    class _BadRidge:
        def __init__(self, **kwargs):
            pass

        def load(self, path):
            raise RuntimeError("ridge artifact missing")

    monkeypatch.setattr(core, "RidgeMultiTarget", _BadRidge)

    # Minimal registry stub for QB. Attention + LGBM disabled so only Ridge
    # fails and NN still runs — confirms the function presses on after
    # recording Ridge's error.
    monkeypatch.setattr(
        core,
        "POSITION_REGISTRY",
        type(
            "_R",
            (),
            {
                "__getitem__": lambda self, k: {
                    "targets": ["passing_yards"],
                    "specific_features": [],
                    "filter_fn": lambda df: df,
                    "compute_targets_fn": lambda df: df,
                    "add_features_fn": lambda tr, va, te: (tr, va, te),
                    "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
                    "get_feature_columns_fn": lambda: ["f0"],
                    "model_dir": "missing",
                    "nn_file": "missing.pt",
                    "nn_kwargs": {},
                    "train_attention_nn": False,
                    "train_lightgbm": False,
                    "aggregate_fn": lambda preds: preds["passing_yards"],
                },
                "__contains__": lambda self, k: True,
            },
        )(),
    )
    results = _make_results_frame(n=4)
    df = _make_df(n=4)
    _mocked_app._cache.clear()

    # MUST NOT raise — the failure is absorbed per the Part B contract.
    core._apply_position_models(df, df, df, "QB", results)

    # Error recorded under the per-model key used by _degraded_positions().
    assert "QB_ridge" in _mocked_app._cache["position_load_errors"]
    assert _mocked_app._cache["position_load_errors"]["QB_ridge"] == (
        "RuntimeError('ridge artifact missing')"
    )
    # Ridge column NaN'd for this position's rows so the frontend renders "--"
    # instead of misleading 0.0 (the DataFrame's init value for ridge_pred).
    assert results["ridge_pred"].isna().all()
    # NN still ran and wrote its predictions — Ridge's failure isn't allowed to
    # cascade and take the other models with it.
    assert results["nn_pred"].notna().all()


@pytest.mark.integration
def test_apply_position_models_clears_stale_errors_for_this_position(_mocked_app, monkeypatch):
    """A successful ``_apply_position_models`` call must clear any pre-existing
    ``position_load_errors`` entries for this position — both the bare ``{pos}``
    key (from the outer ``_ensure_position_loaded`` catch) and the per-model
    ``{pos}_{model_type}`` keys (from the inner per-model try/excepts).

    Without this clear, a transient S3 model-load failure poisons the error
    map permanently. The bug latched ``/health`` to "degraded" forever, which
    on the old 503-on-any-error contract recycled the ECS task every ~30s.
    Other positions' error entries MUST be left alone — we're only touching
    this position's slate.
    """
    import src.serving.app as app_mod

    # Minimal registry stub: same pattern as the ridge-failure test above,
    # keeps the registry-driven feature build cheap and deterministic.
    monkeypatch.setattr(
        core,
        "POSITION_REGISTRY",
        type(
            "_R",
            (),
            {
                "__getitem__": lambda self, k: {
                    "targets": ["passing_yards"],
                    "specific_features": [],
                    "filter_fn": lambda df: df,
                    "compute_targets_fn": lambda df: df,
                    "add_features_fn": lambda tr, va, te: (tr, va, te),
                    "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
                    "get_feature_columns_fn": lambda: ["f0"],
                    "model_dir": "missing",
                    "nn_file": "missing.pt",
                    "nn_kwargs": {},
                    "train_attention_nn": False,
                    "train_lightgbm": False,
                    "aggregate_fn": lambda preds: preds["passing_yards"],
                },
                "__contains__": lambda self, k: True,
            },
        )(),
    )

    # Pre-populate with stale entries for QB AND an unrelated position.
    _mocked_app._cache.clear()
    _mocked_app._cache["position_load_errors"] = {
        "QB": "previous full-failure (cleared)",
        "QB_ridge": "previous ridge failure (cleared)",
        "QB_nn": "previous nn failure (cleared)",
        "RB_lgbm": "unrelated — must survive",
    }

    results = _make_results_frame(n=4)
    df = _make_df(n=4)
    core._apply_position_models(df, df, df, "QB", results)

    errors_after = _mocked_app._cache.get("position_load_errors", {})
    # All QB entries are gone — both bare key and per-model keys.
    assert "QB" not in errors_after
    assert "QB_ridge" not in errors_after
    assert "QB_nn" not in errors_after
    # Other positions' entries untouched.
    assert errors_after.get("RB_lgbm") == "unrelated — must survive"
