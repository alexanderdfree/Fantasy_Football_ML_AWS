"""Unit tests for shared.artifact_integrity.

These guard the scaler/NN-weights integrity check that prevents the
"NN MAE = 25" failure mode: inference silently using a scaler that was
re-fit after the NN was trained, producing garbage predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.artifact_integrity import (
    assert_scaler_matches,
    compute_feature_cols_hash,
    read_scaler_meta,
    unwrap_state_dict,
    wrap_state_dict,
    write_scaler_meta,
)


FEATURES_A = ["feat_a", "feat_b", "feat_c"]
FEATURES_B = ["feat_a", "feat_b", "feat_d"]  # one column differs
TARGETS = ["passing_floor", "rushing_floor", "td_points"]


def _fake_scaler(n_features: int):
    """Minimal stand-in for sklearn.StandardScaler for these checks."""
    return SimpleNamespace(n_features_in_=n_features)


def test_feature_cols_hash_is_stable():
    assert compute_feature_cols_hash(FEATURES_A) == compute_feature_cols_hash(FEATURES_A)


def test_feature_cols_hash_differs_on_column_change():
    assert compute_feature_cols_hash(FEATURES_A) != compute_feature_cols_hash(FEATURES_B)


def test_feature_cols_hash_is_order_sensitive():
    assert compute_feature_cols_hash(FEATURES_A) != compute_feature_cols_hash(list(reversed(FEATURES_A)))


def test_wrap_unwrap_roundtrip():
    sd = {"layer.weight": "tensor-like", "layer.bias": "tensor-like"}
    wrapped = wrap_state_dict(sd, FEATURES_A, TARGETS)
    unwrapped, h = unwrap_state_dict(wrapped)
    assert unwrapped == sd
    assert h == compute_feature_cols_hash(FEATURES_A)


def test_unwrap_passes_through_legacy_raw_state_dict():
    # Legacy artifacts have no wrapper — just a flat {layer: tensor} dict.
    legacy = {"layer.weight": "tensor-like"}
    unwrapped, h = unwrap_state_dict(legacy)
    assert unwrapped is legacy
    assert h is None


def test_write_and_read_scaler_meta_roundtrip(tmp_path: Path):
    meta_path = tmp_path / "nn_scaler_meta.json"
    written = write_scaler_meta(meta_path, FEATURES_A, TARGETS)
    assert meta_path.exists()
    loaded = read_scaler_meta(meta_path)
    assert loaded["n_features"] == len(FEATURES_A)
    assert loaded["feature_cols_hash"] == compute_feature_cols_hash(FEATURES_A)
    assert loaded["target_names"] == TARGETS
    assert loaded == written


def test_read_scaler_meta_returns_none_when_absent(tmp_path: Path):
    assert read_scaler_meta(tmp_path / "missing.json") is None


def test_assert_passes_when_everything_matches():
    scaler = _fake_scaler(len(FEATURES_A))
    nn_hash = compute_feature_cols_hash(FEATURES_A)
    meta = {
        "n_features": len(FEATURES_A),
        "feature_cols_hash": nn_hash,
        "target_names": TARGETS,
    }
    # Must not raise
    assert_scaler_matches("QB", scaler, nn_hash, meta, FEATURES_A, TARGETS)


def test_assert_passes_for_legacy_artifacts():
    # No metadata at all — only shape check runs.
    scaler = _fake_scaler(len(FEATURES_A))
    assert_scaler_matches("QB", scaler, None, None, FEATURES_A, TARGETS)


def test_assert_fails_on_shape_mismatch():
    scaler = _fake_scaler(len(FEATURES_A))
    with pytest.raises(RuntimeError, match=r"fit on \d+ features but inference"):
        # Caller claims a different feature count than the scaler knows about.
        assert_scaler_matches("QB", scaler, None, None, FEATURES_A + ["extra"], TARGETS)


def test_assert_fails_on_scaler_hash_mismatch():
    scaler = _fake_scaler(len(FEATURES_A))
    stale_meta = {
        "n_features": len(FEATURES_A),
        "feature_cols_hash": compute_feature_cols_hash(FEATURES_B),  # different
        "target_names": TARGETS,
    }
    nn_hash = compute_feature_cols_hash(FEATURES_A)
    with pytest.raises(RuntimeError, match="feature_cols_hash mismatch"):
        assert_scaler_matches("QB", scaler, nn_hash, stale_meta, FEATURES_A, TARGETS)


def test_assert_fails_on_nn_hash_mismatch():
    # The exact "NN MAE = 25" failure mode: scaler is fresh (correct hash) but
    # NN weights were saved from a run against a different feature set.
    scaler = _fake_scaler(len(FEATURES_A))
    meta = {
        "n_features": len(FEATURES_A),
        "feature_cols_hash": compute_feature_cols_hash(FEATURES_A),
        "target_names": TARGETS,
    }
    stale_nn_hash = compute_feature_cols_hash(FEATURES_B)
    with pytest.raises(RuntimeError, match="NN feature_cols_hash mismatch"):
        assert_scaler_matches("QB", scaler, stale_nn_hash, meta, FEATURES_A, TARGETS)


def test_assert_fails_on_target_name_mismatch():
    scaler = _fake_scaler(len(FEATURES_A))
    meta = {
        "n_features": len(FEATURES_A),
        "feature_cols_hash": compute_feature_cols_hash(FEATURES_A),
        "target_names": ["different", "targets"],
    }
    with pytest.raises(RuntimeError, match="target_names"):
        assert_scaler_matches("QB", scaler, None, meta, FEATURES_A, TARGETS)


def test_position_name_surfaces_in_error():
    # The error message has to name the position so a /health 500 is actionable.
    scaler = _fake_scaler(99)
    with pytest.raises(RuntimeError, match=r"^QB:"):
        assert_scaler_matches("QB", scaler, None, None, FEATURES_A, TARGETS)
