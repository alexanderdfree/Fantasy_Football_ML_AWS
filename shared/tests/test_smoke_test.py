"""Tests for shared.smoke_test — post-upload load+predict gate.

Strategy: build a complete-but-tiny model_dir on disk (Ridge + NN + scaler +
meta) for a synthetic position config, then patch ``shared.registry.
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

from shared.artifact_integrity import wrap_state_dict, write_scaler_meta
from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.smoke_test import SmokeTestFailed, run_smoke_test

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

    Patches ``shared.registry.INFERENCE_REGISTRY`` with a plain dict so
    ``shared.smoke_test.run_smoke_test``'s lazy import resolves to it.
    """
    registry: dict[str, dict] = {}

    def _register(pos: str, reg: dict) -> None:
        registry[pos] = reg
        monkeypatch.setattr("shared.registry.INFERENCE_REGISTRY", registry, raising=True)

    return _register


def test_run_smoke_test_passes_with_valid_artifacts(tmp_path, patch_registry):
    """Happy path: a freshly-trained tiny artifact loads and produces finite
    predictions on a zero input. No exception raised."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    patch_registry("TST", _fake_reg(model_dir))

    # Should return None (no exception).
    assert run_smoke_test("TST", model_dir) is None


def test_run_smoke_test_raises_on_missing_model_dir(tmp_path, patch_registry):
    """Non-existent model_dir → SmokeTestFailed, not a vanilla
    FileNotFoundError, so the producer's catch sees the canonical type."""
    patch_registry("TST", _fake_reg(tmp_path / "does_not_exist"))
    with pytest.raises(SmokeTestFailed, match="does not exist"):
        run_smoke_test("TST", tmp_path / "does_not_exist")


def test_run_smoke_test_raises_on_missing_nn_file(tmp_path, patch_registry):
    """Tarball passed _validate_remote_tarball (NN file present) but the
    file is removed locally before smoke test runs. Should fail in the NN
    block with a SmokeTestFailed wrapping the FileNotFoundError."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    (model_dir / "test_multihead_nn.pt").unlink()
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


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


def test_run_smoke_test_raises_on_n_features_mismatch(tmp_path, patch_registry):
    """The scaler meta says n_features=4 but the registry's feature column
    list has 5. assert_scaler_matches detects this dimension mismatch."""
    model_dir = _build_fake_artifact_dir(
        tmp_path, override_meta={"n_features": len(_FAKE_FEATURE_COLS) - 1}
    )
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


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


def test_run_smoke_test_raises_on_corrupt_nn_checkpoint(tmp_path, patch_registry):
    """torch.load fails on a non-torch file. The smoke test wraps the
    UnpicklingError / RuntimeError in SmokeTestFailed."""
    model_dir = _build_fake_artifact_dir(tmp_path)
    (model_dir / "test_multihead_nn.pt").write_bytes(b"NOT A TORCH CHECKPOINT")
    patch_registry("TST", _fake_reg(model_dir))

    with pytest.raises(SmokeTestFailed, match="TST nn"):
        run_smoke_test("TST", model_dir)


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
