"""Tests for src.shared.smoke_test — post-upload load+predict gate.

Strategy: build a complete-but-tiny model_dir on disk (Ridge + NN + scaler +
meta) for a synthetic position config, then patch ``src.shared.registry.
INFERENCE_REGISTRY`` so ``run_smoke_test`` reads our fake config. This
exercises the real load + predict path (including ``assert_scaler_matches``,
``unwrap_state_dict``, scaler.transform, NN forward) without requiring an
actual training run or a position-specific feature pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.shared.artifact_integrity import wrap_state_dict, write_scaler_meta
from src.shared.models import RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)
from src.shared.smoke_test import SmokeTestFailed, run_smoke_test

_FAKE_TARGETS = ["t_yards", "t_tds"]
_FAKE_FEATURE_COLS = ["f_a", "f_b", "f_c", "f_d", "f_e"]


def _build_fake_artifact_dir(tmp_path: Path, *, override_meta: dict | None = None) -> Path:
    """Synthesize a complete artifact dir matching the fake registry below.

    ``override_meta`` lets callers corrupt the scaler meta in-place after the
    canonical write, used by the feature-hash-mismatch test.
    """
    targets = list(_FAKE_TARGETS)
    feature_cols = list(_FAKE_FEATURE_COLS)
    n_features = len(feature_cols)

    model_dir = tmp_path / "models"
    model_dir.mkdir()

    rng = np.random.default_rng(42)
    X = rng.normal(size=(64, n_features)).astype(np.float64)
    y = {t: rng.normal(size=64).astype(np.float64) for t in targets}

    ridge = RidgeMultiTarget(target_names=targets)
    ridge.fit(X, y)
    ridge.save(str(model_dir))

    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, model_dir / "nn_scaler.pkl")
    write_scaler_meta(model_dir / "nn_scaler_meta.json", feature_cols, targets)

    if override_meta is not None:
        meta_path = model_dir / "nn_scaler_meta.json"
        existing = json.loads(meta_path.read_text())
        existing.update(override_meta)
        meta_path.write_text(json.dumps(existing))

    nn_kwargs = dict(backbone_layers=[16], head_hidden=8, dropout=0.0)
    nn = MultiHeadNet(input_dim=n_features, target_names=targets, **nn_kwargs)
    checkpoint = wrap_state_dict(nn.state_dict(), feature_cols, targets)
    torch.save(checkpoint, model_dir / "test_multihead_nn.pt")

    return model_dir


def _fake_reg(model_dir: Path) -> dict:
    """Registry entry mirroring the bare-minimum keys ``run_smoke_test`` reads."""
    return {
        "targets": list(_FAKE_TARGETS),
        "model_dir": str(model_dir),
        "nn_file": "test_multihead_nn.pt",
        "nn_kwargs": dict(backbone_layers=[16], head_hidden=8, dropout=0.0),
        "train_attention_nn": False,
        "train_lightgbm": False,
        "get_feature_columns_fn": lambda: list(_FAKE_FEATURE_COLS),
    }


@pytest.fixture
def patch_registry(monkeypatch):
    """Returns a callable: ``register("TST", fake_reg_dict)``.

    Patches ``src.shared.registry.INFERENCE_REGISTRY`` with a plain dict so
    ``src.shared.smoke_test.run_smoke_test``'s lazy import resolves to it.
    """
    registry: dict[str, dict] = {}

    def _register(pos: str, reg: dict) -> None:
        registry[pos] = reg
        monkeypatch.setattr("src.shared.registry.INFERENCE_REGISTRY", registry, raising=True)

    return _register


@pytest.mark.unit
def test_run_smoke_test_passes_with_valid_artifacts(tmp_path, patch_registry):
    """Happy path: a freshly-trained tiny artifact loads and produces finite
    predictions on a zero input. No exception raised."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    patch_registry("TST", _fake_reg(model_dir))

    # Should return None (no exception).
    assert run_smoke_test("TST", model_dir) is None


@pytest.mark.unit
def test_run_smoke_test_raises_on_missing_model_dir(tmp_path, patch_registry):
    """Non-existent model_dir → SmokeTestFailed, not a vanilla
    FileNotFoundError, so the producer's catch sees the canonical type."""
    patch_registry("TST", _fake_reg(tmp_path / "does_not_exist"))
    with pytest.raises(SmokeTestFailed, match="does not exist"):
        run_smoke_test("TST", tmp_path / "does_not_exist")


@pytest.mark.unit
def test_run_smoke_test_raises_on_missing_nn_file(tmp_path, patch_registry):
    """Tarball passed _validate_remote_tarball (NN file present) but the
    file is removed locally before smoke test runs. Should fail in the NN
    block with a SmokeTestFailed wrapping the FileNotFoundError."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    (model_dir / "test_multihead_nn.pt").unlink()
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_feature_hash_mismatch(tmp_path, patch_registry):
    """Scaler meta lies about its feature_cols_hash (e.g. retrained on a
    different feature set without re-emitting the meta). The smoke test
    must catch this via assert_scaler_matches — this is the primary value
    over the structural _validate_remote_tarball check."""
    model_dir = _build_fake_artifact_dir(
        tmp_path,
        override_meta={"feature_cols_hash": "0" * 64},  # plausible-shape lie
    )
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_n_features_mismatch(tmp_path, patch_registry):
    """The scaler meta says n_features=4 but the registry's feature column
    list has 5. assert_scaler_matches detects this dimension mismatch."""
    model_dir = _build_fake_artifact_dir(
        tmp_path, override_meta={"n_features": len(_FAKE_FEATURE_COLS) - 1}
    )
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_nan_prediction(tmp_path, patch_registry, monkeypatch):
    """Model loads cleanly but produces NaN output (e.g. a head that
    collapsed to nan during training and got serialized). The smoke test
    must catch this — the existing _validate_remote_tarball can't, since
    the file passes structural checks. This is the second meaningful value
    over file-presence validation."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    patch_registry("TST", _fake_reg(model_dir))

    real_predict = MultiHeadNet.predict_numpy

    def _predict_with_nan(self, X, device):
        out = real_predict(self, X, device)
        first = next(iter(out))
        out[first] = np.full_like(out[first], np.nan)
        return out

    monkeypatch.setattr(MultiHeadNet, "predict_numpy", _predict_with_nan)

    with pytest.raises(SmokeTestFailed, match="NaN/Inf"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_corrupt_nn_checkpoint(tmp_path, patch_registry):
    """torch.load fails on a non-torch file. The smoke test wraps the
    UnpicklingError / RuntimeError in SmokeTestFailed."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    (model_dir / "test_multihead_nn.pt").write_bytes(b"NOT A TORCH CHECKPOINT")
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_missing_ridge(tmp_path, patch_registry):
    """Ridge is loaded first; if its files are missing it fails before NN
    is even touched. The error label must identify the model that failed
    so triage can find the right component."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    # Wipe the Ridge target subdirs so RidgeMultiTarget.load raises.
    for t in _FAKE_TARGETS:
        target_dir = model_dir / t
        for p in target_dir.iterdir():
            p.unlink()
        target_dir.rmdir()
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST ridge"):
        run_smoke_test("TST", model_dir)


# ---------------------------------------------------------------------------
# Attention NN smoke-path coverage.
#
# Strategy: layer an attention scaler + attention checkpoint onto the base
# artifact dir produced by ``_build_fake_artifact_dir`` so the Ridge + NN base
# paths still pass (they're checked first inside ``run_smoke_test``). The
# attention scaler is fit only on the static-feature subset, mirroring what
# the training pipeline emits — ``assert_scaler_matches`` would otherwise
# trip on a feature-count mismatch before the forward pass is exercised.
# ---------------------------------------------------------------------------

# Flat-history (skill-position style) attention spec. Static features must
# be a subset of ``_FAKE_FEATURE_COLS`` so the skill-path
# ``get_attn_static_columns(feature_cols, whitelist)`` filter resolves them.
_FAKE_ATTN_STATIC_FEATURES = ["f_a", "f_b", "f_c"]
_FAKE_ATTN_HISTORY_STATS = ["h_pass_yards", "h_rush_yards"]
_FAKE_ATTN_MAX_SEQ_LEN = 4
_FAKE_OPP_ATTN_HISTORY_STATS = ["opp_def_yards_allowed"]
_FAKE_OPP_ATTN_MAX_SEQ_LEN = 4

# Nested-history (K-style) attention spec.
_FAKE_ATTN_KICK_STATS = ["k_yards", "k_made"]
_FAKE_ATTN_MAX_GAMES = 3
_FAKE_ATTN_MAX_KICKS_PER_GAME = 4

# Compact NN dims keep each attention-path test well under 1s.
_FAKE_ATTN_KWARGS = dict(backbone_layers=[8], d_model=8, n_attn_heads=2, head_hidden=8, dropout=0.0)
_FAKE_ATTN_NESTED_KWARGS = dict(
    backbone_layers=[8],
    d_kick=8,
    d_model=8,
    n_attn_heads=2,
    head_hidden=8,
    dropout=0.0,
)


def _write_attn_artifact(
    model_dir: Path,
    *,
    static_cols: list[str],
    targets: list[str],
    model: torch.nn.Module,
    checkpoint_name: str = "test_attention_nn.pt",
    override_meta: dict | None = None,
) -> None:
    """Drop an attention scaler + meta + checkpoint into ``model_dir``.

    The scaler is fit on synthetic data of shape ``(64, len(static_cols))``
    so its ``n_features_in_`` matches the resolved attention static columns
    that ``_smoke_attention`` will derive at smoke-test time.
    """
    rng = np.random.default_rng(7)
    X_static = rng.normal(size=(64, len(static_cols))).astype(np.float64)
    scaler = StandardScaler().fit(X_static)
    joblib.dump(scaler, model_dir / "attention_nn_scaler.pkl")
    write_scaler_meta(model_dir / "attention_nn_scaler_meta.json", static_cols, targets)
    if override_meta is not None:
        meta_path = model_dir / "attention_nn_scaler_meta.json"
        existing = json.loads(meta_path.read_text())
        existing.update(override_meta)
        meta_path.write_text(json.dumps(existing))

    checkpoint = wrap_state_dict(model.state_dict(), static_cols, targets)
    torch.save(checkpoint, model_dir / checkpoint_name)


def _build_flat_attn_artifact_dir(
    tmp_path: Path,
    *,
    with_opp: bool = False,
    override_attn_meta: dict | None = None,
    nan_weights: bool = False,
) -> Path:
    """Base artifact dir + attention scaler/checkpoint for the flat-history
    (MultiHeadNetWithHistory) path. Optionally bolts on the opponent-history
    branch to exercise the ``opp_game_dim is not None`` branch."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    static_cols = list(_FAKE_ATTN_STATIC_FEATURES)
    targets = list(_FAKE_TARGETS)
    opp_game_dim = len(_FAKE_OPP_ATTN_HISTORY_STATS) if with_opp else None
    model = MultiHeadNetWithHistory(
        static_dim=len(static_cols),
        game_dim=len(_FAKE_ATTN_HISTORY_STATS),
        target_names=targets,
        opp_game_dim=opp_game_dim,
        **_FAKE_ATTN_KWARGS,
    )
    if nan_weights:
        # Force the head's final Linear to emit NaN: a non-finite weight on
        # the second linear of every target head is enough to NaN the output.
        with torch.no_grad():
            for head in model.heads.values():
                head[-1].weight.fill_(float("nan"))
    _write_attn_artifact(
        model_dir,
        static_cols=static_cols,
        targets=targets,
        model=model,
        override_meta=override_attn_meta,
    )
    return model_dir


def _build_nested_attn_artifact_dir(tmp_path: Path) -> Path:
    """Base artifact dir + attention scaler/checkpoint for the nested-history
    (MultiHeadNetWithNestedHistory) path, mirroring the K spec."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    static_cols = list(_FAKE_ATTN_STATIC_FEATURES)
    targets = list(_FAKE_TARGETS)
    model = MultiHeadNetWithNestedHistory(
        static_dim=len(static_cols),
        kick_dim=len(_FAKE_ATTN_KICK_STATS),
        target_names=targets,
        **_FAKE_ATTN_NESTED_KWARGS,
    )
    _write_attn_artifact(
        model_dir,
        static_cols=static_cols,
        targets=targets,
        model=model,
    )
    return model_dir


def _fake_flat_attn_reg(model_dir: Path, *, with_opp: bool = False) -> dict:
    """Registry entry for the flat-history attention path.

    Mirrors a skill-position config: ``attn_static_from_df`` is unset (False)
    so ``_resolve_attn_static_cols`` exercises the
    ``get_attn_static_columns(feature_cols, whitelist)`` branch.
    """
    reg = _fake_reg(model_dir)
    reg.update(
        {
            "train_attention_nn": True,
            "attn_nn_file": "test_attention_nn.pt",
            "attn_history_stats": list(_FAKE_ATTN_HISTORY_STATS),
            "attn_static_features": list(_FAKE_ATTN_STATIC_FEATURES),
            "attn_max_seq_len": _FAKE_ATTN_MAX_SEQ_LEN,
            "attn_nn_kwargs_static": dict(_FAKE_ATTN_KWARGS),
        }
    )
    if with_opp:
        reg["opp_attn_history_stats"] = list(_FAKE_OPP_ATTN_HISTORY_STATS)
        reg["opp_attn_max_seq_len"] = _FAKE_OPP_ATTN_MAX_SEQ_LEN
    return reg


def _fake_nested_attn_reg(model_dir: Path) -> dict:
    """Registry entry for the nested-history (K-style) attention path.

    ``attn_static_from_df=True`` so ``_resolve_attn_static_cols`` exercises the
    short-circuit branch that reads ``attn_static_features`` straight off
    the registry instead of intersecting with the feature column list.
    """
    reg = _fake_reg(model_dir)
    reg.update(
        {
            "train_attention_nn": True,
            "attn_nn_file": "test_attention_nn.pt",
            "attn_history_structure": "nested",
            "attn_static_from_df": True,
            "attn_static_features": list(_FAKE_ATTN_STATIC_FEATURES),
            "attn_kick_stats": list(_FAKE_ATTN_KICK_STATS),
            "attn_max_games": _FAKE_ATTN_MAX_GAMES,
            "attn_max_kicks_per_game": _FAKE_ATTN_MAX_KICKS_PER_GAME,
            "attn_nn_kwargs_static": dict(_FAKE_ATTN_NESTED_KWARGS),
        }
    )
    return reg


@pytest.mark.unit
def test_run_smoke_test_passes_with_flat_attention_artifact(tmp_path, patch_registry):
    """Happy path: flat-history MultiHeadNetWithHistory artifact loads and
    produces finite predictions on zero static + zero history input. Exercises
    ``_smoke_attention`` flat branch and the skill-position
    ``_resolve_attn_static_cols`` path."""
    model_dir = _build_flat_attn_artifact_dir(tmp_path)
    patch_registry("TST", _fake_flat_attn_reg(model_dir))

    assert run_smoke_test("TST", model_dir) is None


@pytest.mark.unit
def test_run_smoke_test_passes_with_nested_attention_artifact(tmp_path, patch_registry):
    """Happy path: K-style nested-history MultiHeadNetWithNestedHistory
    artifact loads and produces finite predictions. Exercises
    ``_smoke_attention``'s nested branch and the ``attn_static_from_df=True``
    short-circuit in ``_resolve_attn_static_cols``."""
    model_dir = _build_nested_attn_artifact_dir(tmp_path)
    patch_registry("TST", _fake_nested_attn_reg(model_dir))

    assert run_smoke_test("TST", model_dir) is None


@pytest.mark.unit
def test_run_smoke_test_passes_with_opp_attention_artifact(tmp_path, patch_registry):
    """Happy path: flat-history attention NN with the opponent-history branch
    enabled (``opp_attn_history_stats`` non-empty). Exercises the
    ``opp_game_dim is not None`` arm of ``_smoke_attention`` and feeds the
    opp_hist / opp_mask zeros through ``predict_numpy``."""
    model_dir = _build_flat_attn_artifact_dir(tmp_path, with_opp=True)
    patch_registry("TST", _fake_flat_attn_reg(model_dir, with_opp=True))

    assert run_smoke_test("TST", model_dir) is None


@pytest.mark.unit
def test_run_smoke_test_raises_on_attention_nan_prediction(tmp_path, patch_registry):
    """Attention NN weights serialize cleanly but produce NaN at inference
    (final head Linear weight is NaN). ``_smoke_attention`` must surface this
    via ``_assert_finite_dict`` with the ``attn_nn`` label so triage knows
    which model went bad."""
    model_dir = _build_flat_attn_artifact_dir(tmp_path, nan_weights=True)
    patch_registry("TST", _fake_flat_attn_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="attn_nn"):
        run_smoke_test("TST", model_dir)


@pytest.mark.unit
def test_run_smoke_test_raises_on_attention_scaler_hash_mismatch(tmp_path, patch_registry):
    """The attention scaler meta lies about its ``feature_cols_hash``. The
    smoke test must fail with the ``attention_nn_scaler`` label so triage
    can distinguish this from a base-NN scaler drift."""
    model_dir = _build_flat_attn_artifact_dir(
        tmp_path,
        override_attn_meta={"feature_cols_hash": "0" * 64},
    )
    patch_registry("TST", _fake_flat_attn_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="attention_nn_scaler") as exc_info:
        run_smoke_test("TST", model_dir)
    # Confirm the failure surfaced as the attn block's chain, not the base NN.
    assert "attn_nn" in str(exc_info.value)


@pytest.mark.unit
def test_run_smoke_test_raises_on_attention_missing_target_head(
    tmp_path, patch_registry, monkeypatch
):
    """Attention model returns predictions for only some target heads (e.g.
    a head got dropped via a checkpoint-load typo). ``_assert_finite_dict``
    must raise with ``missing target heads`` to identify the broken
    contract."""
    model_dir = _build_flat_attn_artifact_dir(tmp_path)
    patch_registry("TST", _fake_flat_attn_reg(model_dir))

    real_predict = MultiHeadNetWithHistory.predict_numpy

    def _predict_dropping_one_head(self, X_static, X_history, history_mask, device, **kw):
        out = real_predict(self, X_static, X_history, history_mask, device, **kw)
        # Drop the first target so ``missing target heads`` fires.
        out.pop(self.target_names[0])
        return out

    monkeypatch.setattr(MultiHeadNetWithHistory, "predict_numpy", _predict_dropping_one_head)

    with pytest.raises(SmokeTestFailed, match="missing target heads"):
        run_smoke_test("TST", model_dir)
