"""Additional coverage for ``app.py`` gap branches.

Existing app tests + ``test_app_apply_position_models.py`` cover the Flask
routes and the QB/attention-flat path through ``_apply_position_models``.
This file fills in:

- Nested-history (K) attention branch in ``_apply_position_models``
- Degrade-on-model-load-failure branches (ridge / nn / attn / lgbm all
  individually NaN'd rather than raising the whole request)
- ``_ensure_position_loaded`` cache-miss idempotence + failed-cache short-
  circuit
- ``_position_arch_payload`` scheduler-string branches and include_features
  list-vs-dict branch
- ``/api/top_players`` position filter + ALL paths (via ``client_with_data``)
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# _apply_position_models — K nested-attention + per-model degrade branches
# --------------------------------------------------------------------------


def _make_results_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"K{i}" for i in range(n)],
            "position": ["K"] * n,
            "recent_team": ["KC"] * n,
            "season": [2025] * n,
            "week": list(range(1, n + 1)),
            "fantasy_points": np.linspace(5, 15, n),
            "ridge_pred": [0.0] * n,
            "nn_pred": [0.0] * n,
            "attn_nn_pred": [np.nan] * n,
            "lgbm_pred": [np.nan] * n,
        }
    )


def _make_k_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [f"K{i}" for i in range(n)],
            "player_display_name": [f"Kicker {i}" for i in range(n)],
            "position": ["K"] * n,
            "recent_team": ["KC"] * n,
            "season": [2025] * n,
            "week": list(range(1, n + 1)),
            "fg_yard_points": np.linspace(1, 9, n),
            "pat_points": np.linspace(0, 3, n),
            "fg_misses": np.zeros(n),
            "xp_misses": np.zeros(n),
            "fantasy_points": np.linspace(5, 15, n),
            "static_feat": np.ones(n),
        }
    )


@pytest.fixture()
def _stub_app(monkeypatch):
    """Share the lightweight-model stubs with test_app_apply_position_models."""
    import src.serving.app as app_mod

    class _FakeMultiTarget:
        def __init__(self, target_names, **kwargs):
            self.target_names = target_names

        def load(self, model_dir):
            pass

        def predict(self, X):
            return {t: np.zeros(len(X), dtype=np.float32) for t in self.target_names}

    from sklearn.preprocessing import StandardScaler

    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.zeros((2, 1), dtype=np.float32))

    monkeypatch.setattr(app_mod, "RidgeMultiTarget", _FakeMultiTarget)
    monkeypatch.setattr(app_mod, "LightGBMMultiTarget", _FakeMultiTarget)
    monkeypatch.setattr(app_mod.joblib, "load", lambda path: dummy_scaler)
    monkeypatch.setattr(
        app_mod.torch,
        "load",
        lambda *a, **k: {"model_state": {}, "feature_columns_hash": "h"},
    )
    monkeypatch.setattr(app_mod, "assert_scaler_matches", lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "read_scaler_meta", lambda *a, **k: {})
    monkeypatch.setattr(
        app_mod, "unwrap_state_dict", lambda checkpoint: (checkpoint.get("model_state", {}), "h")
    )
    monkeypatch.setattr(
        app_mod, "scale_and_clip", lambda scaler, X: np.asarray(X, dtype=np.float32)
    )

    class _FakeNN:
        def __init__(self, *args, **kwargs):
            self.target_names = kwargs.get("target_names", [])

        def to(self, device):
            return self

        def load_state_dict(self, sd):
            pass

        def predict_numpy(self, *args, **kwargs):
            X = args[0]
            n = len(X)
            return {t: np.zeros(n, dtype=np.float32) for t in self.target_names}

    monkeypatch.setattr(app_mod, "MultiHeadNet", _FakeNN)
    monkeypatch.setattr(app_mod, "MultiHeadNetWithHistory", _FakeNN)
    monkeypatch.setattr(app_mod, "MultiHeadNetWithNestedHistory", _FakeNN)

    monkeypatch.setattr(
        app_mod,
        "build_position_features",
        lambda tr, va, te, reg, fc: (tr, va, te),
    )
    # K nested path calls k_features.build_nested_kick_history — stub to tiny
    # tensors. Patch on the imported module alias so the call site resolves
    # to our stub (the bare name is no longer an attribute on app_mod after
    # the cross-position-collision cleanup in PR2).
    monkeypatch.setattr(
        app_mod.k_features,
        "build_nested_kick_history",
        lambda df, **kw: (
            np.zeros((len(df), 2, 3, 4), dtype=np.float32),
            np.zeros((len(df), 2, 3), dtype=np.float32),
            np.zeros((len(df), 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "build_game_history_arrays",
        lambda df, history_stats, max_seq_len: (
            np.zeros((len(df), max_seq_len, max(1, len(history_stats))), dtype=np.float32),
            np.zeros((len(df), max_seq_len), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        app_mod,
        "get_attn_static_columns",
        lambda feature_cols, allow: feature_cols[:1] if feature_cols else [],
    )
    return app_mod


@pytest.mark.integration
def test_apply_position_models_k_nested_attention_branch(_stub_app, monkeypatch):
    """K's ``attn_history_structure == "nested"`` branch (lines 486-508) must
    fire end-to-end, producing attn_nn_pred values on the results frame."""
    reg = {
        "targets": ["fg_yard_points", "pat_points", "fg_misses", "xp_misses"],
        "specific_features": [],
        "filter_fn": lambda df: df[df["position"] == "K"].copy(),
        "compute_targets_fn": lambda df: df,
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["static_feat"],
        "target_signs": {
            "fg_yard_points": 1.0,
            "pat_points": 1.0,
            "fg_misses": -1.0,
            "xp_misses": -1.0,
        },
        "model_dir": "src/k/outputs/models",
        "nn_file": "k_multihead_nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": True,
        "attn_nn_file": "k_attention_nn.pt",
        "attn_nn_kwargs_static": {},
        "attn_history_structure": "nested",
        "attn_static_from_df": True,
        "attn_static_features": ["static_feat"],
        "attn_kick_stats": ["kick_distance"],
        "attn_max_games": 2,
        "attn_max_kicks_per_game": 3,
        "train_lightgbm": False,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(_stub_app, "POSITION_REGISTRY", _Stub())
    # K nested attention reads k_kicks_df out of _cache; stub with an empty DF
    # so build_nested_kick_history has something to close over (our helper
    # stub doesn't actually use the contents).
    _stub_app._cache.clear()
    _stub_app._cache["k_kicks_df"] = pd.DataFrame()

    results = _make_results_frame(n=6)
    df = _make_k_df(n=6)
    _stub_app._apply_position_models(df, df, df, "K", results)

    # K's attention must populate attn_nn_pred for every row.
    assert results["attn_nn_pred"].notna().all()


@pytest.mark.integration
def test_apply_position_models_k_nested_attention_missing_kicks_df_raises(_stub_app, monkeypatch):
    """When k_kicks_df is absent from _cache, the nested-attention branch
    raises RuntimeError — the outer except records the failure + NaN's preds."""
    reg = {
        "targets": ["fg_yard_points"],
        "specific_features": [],
        "filter_fn": lambda df: df[df["position"] == "K"].copy(),
        "compute_targets_fn": lambda df: df,
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["static_feat"],
        "target_signs": {"fg_yard_points": 1.0},
        "model_dir": "src/k/outputs/models",
        "nn_file": "k_multihead_nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": True,
        "attn_nn_file": "k_attention_nn.pt",
        "attn_nn_kwargs_static": {},
        "attn_history_structure": "nested",
        "attn_static_from_df": True,
        "attn_static_features": ["static_feat"],
        "attn_kick_stats": ["kick_distance"],
        "attn_max_games": 2,
        "attn_max_kicks_per_game": 3,
        "train_lightgbm": False,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(_stub_app, "POSITION_REGISTRY", _Stub())
    _stub_app._cache.clear()
    # Intentionally NOT setting k_kicks_df.

    results = _make_results_frame(n=4)
    df = _make_k_df(n=4)
    _stub_app._apply_position_models(df, df, df, "K", results)

    # Ridge + NN still populated (their paths succeeded); attn NaN'd.
    assert results["attn_nn_pred"].isna().all()


@pytest.mark.integration
def test_apply_position_models_lgbm_load_failure_leaves_lgbm_pred_nan(monkeypatch, _stub_app):
    """LGBM load exception → ``lgbm_pred`` stays NaN, ridge/nn unaffected."""
    import src.serving.app as app_mod

    class _BadLGBM:
        def __init__(self, **kwargs):
            pass

        def load(self, path):
            raise RuntimeError("lgbm missing")

    monkeypatch.setattr(app_mod, "LightGBMMultiTarget", _BadLGBM)

    reg = {
        "targets": ["passing_yards"],
        "specific_features": [],
        "filter_fn": lambda df: df,
        "compute_targets_fn": lambda df: df,
        "add_features_fn": lambda tr, va, te: (tr, va, te),
        "fill_nans_fn": lambda tr, va, te, specs: (tr, va, te),
        "get_feature_columns_fn": lambda: ["f0"],
        "aggregate_fn": lambda preds: preds["passing_yards"],
        "model_dir": "dir",
        "nn_file": "nn.pt",
        "nn_kwargs": {},
        "train_attention_nn": False,
        "train_lightgbm": True,
    }

    class _Stub:
        def __getitem__(self, pos):
            return reg

        def __contains__(self, pos):
            return True

    monkeypatch.setattr(app_mod, "POSITION_REGISTRY", _Stub())

    results = pd.DataFrame(
        {
            "position": ["QB"] * 4,
            "fantasy_points": [10.0] * 4,
            "ridge_pred": [0.0] * 4,
            "nn_pred": [0.0] * 4,
            "attn_nn_pred": [np.nan] * 4,
            "lgbm_pred": [np.nan] * 4,
        }
    )
    df = pd.DataFrame(
        {
            "player_id": ["P"] * 4,
            "position": ["QB"] * 4,
            "fantasy_points": [10.0] * 4,
            "passing_yards": [250.0] * 4,
            "f0": [1.0] * 4,
        }
    )
    app_mod._cache.clear()
    app_mod._apply_position_models(df, df, df, "QB", results)
    # Ridge + NN succeeded (fake returns zeros); lgbm NaN due to load failure.
    assert results["ridge_pred"].notna().all()
    assert results["nn_pred"].notna().all()
    assert results["lgbm_pred"].isna().all()


# --------------------------------------------------------------------------
# _ensure_position_loaded — short-circuits + failed-cache
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_ensure_position_loaded_noop_if_already_loaded(monkeypatch):
    """When pos is already in ``positions_loaded``, function returns immediately."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["positions_loaded"] = {"QB"}

    # apply would raise if called — assert it isn't.
    monkeypatch.setattr(
        app_mod,
        "_apply_position_models",
        lambda *a, **k: pytest.fail("_apply_position_models fired despite cached load"),
    )
    app_mod._ensure_position_loaded("QB")


@pytest.mark.integration
def test_ensure_position_loaded_noop_if_in_failed_set(monkeypatch):
    """Positions in ``positions_failed`` do not get retried."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["positions_loaded"] = set()
    app_mod._cache["positions_failed"] = {"QB"}

    monkeypatch.setattr(
        app_mod,
        "_apply_position_models",
        lambda *a, **k: pytest.fail("retried a cached-failed position"),
    )
    app_mod._ensure_position_loaded("QB")


# --------------------------------------------------------------------------
# _position_arch_payload — scheduler + feature-grouping branches
# --------------------------------------------------------------------------


class _CfgModule:
    """Fixture config module with the attributes _position_arch_payload reads."""

    NN_EPOCHS = 10
    NN_BATCH_SIZE = 32
    NN_LR = 1e-3
    NN_WEIGHT_DECAY = 0.0
    NN_DROPOUT = 0.1
    NN_PATIENCE = 5
    HUBER_DELTAS = {"passing_yards": 25.0}
    LOSS_WEIGHTS = {"passing_yards": 1.0}
    RIDGE_ALPHA_GRIDS = {"passing_yards": [1.0]}
    NN_BACKBONE_LAYERS = [32, 16]
    NN_HEAD_HIDDEN = 16


@pytest.mark.integration
def test_position_arch_payload_cosine_warm_restarts_scheduler():
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "cosine_warm_restarts"
    cfg.COSINE_T0 = 10
    cfg.COSINE_T_MULT = 2
    cfg.COSINE_ETA_MIN = 1e-5
    payload = app_mod._position_arch_payload(
        "QB", cfg, specific=["f_spec"], targets=["passing_yards"], include_features=["a", "b"]
    )
    assert "CosineAnnealingWarmRestarts" in payload["scheduler"]


@pytest.mark.integration
def test_position_arch_payload_onecycle_scheduler():
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "onecycle"
    cfg.ONECYCLE_MAX_LR = 0.01
    cfg.ONECYCLE_PCT_START = 0.3
    payload = app_mod._position_arch_payload(
        "QB", cfg, specific=["f_spec"], targets=["passing_yards"], include_features=["a"]
    )
    assert "OneCycleLR" in payload["scheduler"]


@pytest.mark.integration
def test_position_arch_payload_plateau_scheduler():
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "plateau"
    payload = app_mod._position_arch_payload(
        "QB", cfg, specific=["f_spec"], targets=["passing_yards"], include_features=["a"]
    )
    assert payload["scheduler"] == "ReduceLROnPlateau"


@pytest.mark.integration
def test_position_arch_payload_include_features_as_dict():
    """When include_features is a {category: [...]} dict, the grouped layout
    is preserved + 'specific' is injected if missing."""
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "plateau"
    payload = app_mod._position_arch_payload(
        "QB",
        cfg,
        specific=["pos_specific"],
        targets=["passing_yards"],
        include_features={"rolling": ["r1", "r2"], "ewma": ["e1"]},
    )
    features = payload["features"]
    assert "rolling" in features
    assert "ewma" in features
    # 'specific' is injected into the grouped dict since it wasn't a key.
    assert "specific" in features
    assert features["specific"] == ["pos_specific"]


# --------------------------------------------------------------------------
# /health degraded branch (already covered in test_app_apply_position_models)
# and /api/top_players ALL vs position path
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_top_players_position_filter_path(client_with_data):
    """Explicit ``position=QB`` → only positions_loaded branch, no _get_data call."""
    resp = client_with_data.get("/api/top_players?position=QB&week=1")
    assert resp.status_code == 200
    body = resp.get_json()
    # Response is a dict with a list of players
    assert isinstance(body, dict)


# --------------------------------------------------------------------------
# _ensure_position_loaded — full-failure path (sets positions_failed)
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_ensure_position_loaded_records_hard_failure(monkeypatch):
    """When _apply_position_models raises outside the inner try/excepts
    (e.g. feature-build blows up), the position is added to ``positions_failed``
    and position_load_errors carries the repr."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["splits"] = {"QB": (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())}
    app_mod._cache["positions_loaded"] = set()
    app_mod._cache["results"] = pd.DataFrame()

    def _boom(*a, **k):
        raise RuntimeError("feature build exploded")

    monkeypatch.setattr(app_mod, "_apply_position_models", _boom)

    app_mod._ensure_position_loaded("QB")
    assert "QB" in app_mod._cache["positions_failed"]
    assert "feature build exploded" in app_mod._cache["position_load_errors"]["QB"]


@pytest.mark.integration
def test_ensure_position_loaded_returns_when_splits_missing(monkeypatch):
    """When base-load ran but ``splits`` was never populated, the function
    returns without calling _apply_position_models."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["positions_loaded"] = set()
    # No 'splits' key.
    monkeypatch.setattr(
        app_mod,
        "_apply_position_models",
        lambda *a, **k: pytest.fail("shouldn't run without splits"),
    )
    app_mod._ensure_position_loaded("QB")


# --------------------------------------------------------------------------
# _ensure_all_positions_loaded + _degraded_positions
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_ensure_all_positions_raises_when_every_position_fails(monkeypatch):
    """If every position ends up in ``positions_failed``, the wrapper raises
    (gunicorn --preload contract)."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["splits"] = {
        p: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) for p in app_mod._ALL_POSITIONS
    }
    app_mod._cache["positions_loaded"] = set()
    app_mod._cache["results"] = pd.DataFrame()

    def _boom(*a, **k):
        raise RuntimeError("everything broken")

    monkeypatch.setattr(app_mod, "_apply_position_models", _boom)

    with pytest.raises(RuntimeError, match="All positions failed"):
        app_mod._ensure_all_positions_loaded()


@pytest.mark.integration
def test_ensure_all_positions_tolerates_partial_failure(monkeypatch):
    """If only one position fails, the rest still get loaded and no exception."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["base_loaded"] = True
    app_mod._cache["splits"] = {
        p: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) for p in app_mod._ALL_POSITIONS
    }
    app_mod._cache["positions_loaded"] = set()
    app_mod._cache["results"] = pd.DataFrame()

    def _only_qb_fails(train, val, test, pos, results):
        if pos == "QB":
            raise RuntimeError("QB only")
        # success: populate nothing

    monkeypatch.setattr(app_mod, "_apply_position_models", _only_qb_fails)
    # Why: must not raise. Partial-failure contract — failed positions are
    # recorded and the wrapper still returns so /health can flag degradation
    # without taking the request down.
    app_mod._ensure_all_positions_loaded()
    assert app_mod._cache["positions_failed"] == {"QB"}
    assert "QB only" in app_mod._cache["position_load_errors"]["QB"]
    expected_loaded = set(app_mod._ALL_POSITIONS) - {"QB"}
    assert app_mod._cache["positions_loaded"] == expected_loaded
    assert "QB" in app_mod._degraded_positions()


@pytest.mark.integration
def test_degraded_positions_empty_when_no_errors():
    import src.serving.app as app_mod

    app_mod._cache.clear()
    assert app_mod._degraded_positions() == []


@pytest.mark.integration
def test_degraded_positions_dedupes_across_per_model_and_pos_keys():
    """Errors keyed as ``{pos}_{model}`` AND bare ``{pos}`` both feed into the
    set; result is sorted unique positions."""
    import src.serving.app as app_mod

    app_mod._cache.clear()
    app_mod._cache["position_load_errors"] = {
        "QB_ridge": "missing",
        "QB_nn": "missing",
        "RB": "all broken",
        "WR_attn_nn": "missing",
    }
    assert app_mod._degraded_positions() == ["QB", "RB", "WR"]


# --------------------------------------------------------------------------
# _load_k_splits / _load_dst_splits — tiny wrappers exercised via monkeypatch
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_load_k_splits_delegates_to_k_data_helpers(monkeypatch):
    import src.serving.app as app_mod

    k_df = pd.DataFrame({"player_id": ["K1"], "season": [2024], "week": [1]})
    kicks_df = pd.DataFrame({"player_id": ["K1"], "kick_distance": [40.0]})

    # K data + features live on the module aliases after the PR2 collision
    # cleanup (load_data is no longer a bare attribute on app_mod).
    monkeypatch.setattr(app_mod.k_data, "load_data", lambda: k_df)
    monkeypatch.setattr(app_mod.k_data, "load_kicks", lambda df: kicks_df)
    monkeypatch.setattr(
        app_mod.k_data,
        "season_split",
        lambda df: (df.iloc[:0], df.iloc[:0], df),
    )
    monkeypatch.setattr(app_mod.k_features, "compute_features", lambda df: None)

    # POSITION_REGISTRY['K']['compute_targets_fn'] — stub to pass-through.
    class _StubReg:
        def __getitem__(self, k):
            return {"compute_targets_fn": lambda df: df}

    monkeypatch.setattr(app_mod, "POSITION_REGISTRY", _StubReg())

    train, val, test, out_kicks = app_mod._load_k_splits()
    assert out_kicks is kicks_df
    assert len(test) == 1


@pytest.mark.integration
def test_load_dst_splits_filters_by_season(monkeypatch):
    import src.serving.app as app_mod

    dst_df = pd.DataFrame(
        {
            "team": ["KC", "KC", "KC", "KC"],
            "season": [2022, 2023, 2024, 2025],
            "week": [1, 1, 1, 1],
        }
    )
    # DST data + features live on module aliases after PR2's collision cleanup.
    monkeypatch.setattr(app_mod.dst_data, "build_data", lambda: dst_df)
    monkeypatch.setattr(app_mod.dst_features, "compute_features", lambda df: None)

    class _StubReg:
        def __getitem__(self, k):
            return {"compute_targets_fn": lambda df: df}

    monkeypatch.setattr(app_mod, "POSITION_REGISTRY", _StubReg())

    train, val, test = app_mod._load_dst_splits()
    # 2025 is TEST_SEASONS
    assert len(test) == 1
    assert test.iloc[0]["season"] == 2025


# --------------------------------------------------------------------------
# Wiki endpoints + helpers
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_wiki_index_lists_every_doc(client):
    """``/api/wiki/index`` enumerates every WIKI_DOCS entry with slug/name/group."""
    import src.serving.app as app_mod

    resp = client.get("/api/wiki/index")
    # Why: this endpoint never depends on _cache so a 200 is the only acceptable
    # status — anything else means WIKI_DOCS got into an inconsistent state.
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, list)
    assert len(body) == len(app_mod.WIKI_DOCS)
    for entry in body:
        assert set(entry) == {"slug", "name", "group"}
        assert entry["slug"] in app_mod.WIKI_DOCS


@pytest.mark.integration
def test_wiki_page_unknown_slug_returns_404(client):
    """Unknown slug → 404 JSON. Why: slug typos in the SPA must not surface as 5xx."""
    resp = client.get("/api/wiki/does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Unknown doc"}


@pytest.mark.integration
def test_wiki_page_returns_rendered_html(client):
    """Known slug → JSON {slug, name, group, html} with markdown rendered to HTML."""
    import src.serving.app as app_mod

    slug = next(iter(app_mod.WIKI_DOCS))
    resp = client.get(f"/api/wiki/{slug}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"slug", "name", "group", "html"}
    assert body["slug"] == slug
    # markdown.markdown emits HTML tags — the rendered string must contain at
    # least one. Why: a regression in _render_wiki_doc that returns raw markdown
    # would silently break the wiki UI without breaking any other test.
    assert "<" in body["html"] and ">" in body["html"]


@pytest.mark.integration
def test_wiki_render_doc_caches_html(app_module, monkeypatch):
    """Second call hits the cached HTML rather than re-rendering markdown."""
    slug = next(iter(app_module.WIKI_DOCS))

    calls = {"count": 0}
    real_md = app_module.markdown.markdown

    def _counting_md(text, **kwargs):
        calls["count"] += 1
        return real_md(text, **kwargs)

    monkeypatch.setattr(app_module.markdown, "markdown", _counting_md)

    app_module._render_wiki_doc(slug)
    app_module._render_wiki_doc(slug)
    # Why: the second call must reuse the cached HTML — repeated markdown
    # parsing on every wiki page request would dominate p95 latency.
    assert calls["count"] == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "href",
    ["", "#section", "https://example.com/page", "http://x", "mailto:a@b"],
)
def test_wiki_rewrite_href_passes_through_external_and_anchor(href):
    """External URLs, mailto, anchor-only, and empty hrefs must NOT be rewritten."""
    import src.serving.app as app_mod

    assert app_mod._wiki_rewrite_href(href, "docs/ARCHITECTURE.md") == href


@pytest.mark.integration
def test_wiki_rewrite_href_unknown_target_passes_through():
    """A relative path that doesn't resolve to a registered wiki doc is returned unchanged."""
    import src.serving.app as app_mod

    assert app_mod._wiki_rewrite_href("nonexistent.md", "docs/ARCHITECTURE.md") == "nonexistent.md"


@pytest.mark.integration
def test_wiki_rewrite_href_resolves_known_target_with_anchor():
    """A relative path pointing to a registered doc → ``#wiki:slug:anchor``."""
    import src.serving.app as app_mod

    slugs = list(app_mod.WIKI_DOCS)
    # Why: this test depends on the cross-doc rewrite path which only fires
    # when WIKI_DOCS holds at least two entries. A future regression that
    # collapses the registry should fail loudly here, not silently skip.
    assert len(slugs) >= 2
    src_slug, target_slug = slugs[0], slugs[1]
    src_path = app_mod.WIKI_DOCS[src_slug]["path"]
    target_path = app_mod.WIKI_DOCS[target_slug]["path"]
    rel = os.path.relpath(target_path, os.path.dirname(src_path) or ".")
    rewritten = app_mod._wiki_rewrite_href(rel + "#section", src_path)
    assert rewritten == f"#wiki:{target_slug}:section"
    rewritten_no_anchor = app_mod._wiki_rewrite_href(rel, src_path)
    assert rewritten_no_anchor == f"#wiki:{target_slug}"


# --------------------------------------------------------------------------
# /api/model_architecture — error handler + unknown-scheduler fallback
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_model_architecture_error_handler_returns_json_500(client, monkeypatch):
    """If _position_arch_payload raises, the route returns a JSON 500 (not HTML)."""
    import src.serving.app as app_mod

    def _boom(*a, **k):
        raise RuntimeError("arch payload exploded")

    monkeypatch.setattr(app_mod, "_position_arch_payload", _boom)
    resp = client.get("/api/model_architecture")
    # Why: the strategy doc requires structured JSON for /api/* errors so the
    # SPA can surface the message instead of rendering Flask's HTML 500 page.
    assert resp.status_code == 500
    assert resp.is_json
    assert "arch payload exploded" in resp.get_json()["error"]


@pytest.mark.integration
def test_position_arch_payload_unknown_scheduler_falls_back_to_str():
    """Unknown SCHEDULER_TYPE → str(value) so the UI shows whatever was set."""
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "constant"
    payload = app_mod._position_arch_payload(
        "QB", cfg, specific=["f_spec"], targets=["passing_yards"], include_features=["a"]
    )
    assert payload["scheduler"] == "constant"


@pytest.mark.integration
def test_position_arch_payload_attn_history_appended_when_provided():
    """`attn_history` arg surfaces under features.attention_history for the UI."""
    import src.serving.app as app_mod

    cfg = _CfgModule()
    cfg.SCHEDULER_TYPE = "plateau"
    payload = app_mod._position_arch_payload(
        "QB",
        cfg,
        specific=["f_spec"],
        targets=["passing_yards"],
        include_features=["a"],
        attn_history=["passing_yards", "rushing_yards"],
    )
    assert payload["features"]["attention_history"] == ["passing_yards", "rushing_yards"]
