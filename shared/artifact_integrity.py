"""Scaler/weights integrity helpers for NN artifacts.

The pipeline can silently drift if ``nn_scaler.pkl`` is re-fit (e.g. by a
partial benchmark run after a feature-engineering change) without retraining
the NN weights: the NN then sees features normalized against a distribution
it was never trained on, producing garbage predictions. This module provides
the stable fingerprint (``feature_cols_hash``) that lets inference reject a
mismatched scaler+weights pair at load time rather than returning nonsense.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def compute_feature_cols_hash(feature_cols: Iterable[str]) -> str:
    """Stable sha256 of the ordered feature-column list.

    Stable across machines, joblib versions, and pickle protocols — we only
    need the column identities + order to detect "were these trained against
    the same feature set?".
    """
    joined = "\n".join(feature_cols).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def write_scaler_meta(
    meta_path: str | Path,
    feature_cols: Iterable[str],
    target_names: Iterable[str],
) -> dict:
    """Write a sidecar JSON next to a ``.pkl`` scaler. Returns the dict written."""
    feature_cols = list(feature_cols)
    target_names = list(target_names)
    meta = {
        "n_features": len(feature_cols),
        "feature_cols_hash": compute_feature_cols_hash(feature_cols),
        "target_names": target_names,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    return meta


def read_scaler_meta(meta_path: str | Path) -> dict | None:
    """Return the sidecar dict, or None when the sidecar is absent (legacy artifacts)."""
    path = Path(meta_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def wrap_state_dict(
    state_dict: dict,
    feature_cols: Iterable[str],
    target_names: Iterable[str],
) -> dict:
    """Bundle a torch state_dict with integrity metadata for load-time checks."""
    return {
        "state_dict": state_dict,
        "feature_cols_hash": compute_feature_cols_hash(feature_cols),
        "target_names": list(target_names),
    }


def unwrap_state_dict(checkpoint) -> tuple[dict, str | None]:
    """Unwrap a wrapped state_dict, or pass through a legacy raw state_dict.

    Returns ``(state_dict, feature_cols_hash_or_None)``. Legacy artifacts that
    pre-date the wrapper return ``(checkpoint, None)`` so callers can fall
    back to shape-only checks.
    """
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        return checkpoint["state_dict"], checkpoint.get("feature_cols_hash")
    return checkpoint, None


def assert_scaler_matches(
    position: str,
    scaler,
    nn_feature_cols_hash: str | None,
    meta: dict | None,
    feature_cols: Iterable[str],
    target_names: Iterable[str],
    *,
    scaler_label: str = "nn_scaler",
) -> None:
    """Fail loud if scaler and/or NN weights disagree with the inference feature set.

    ``meta`` is the sidecar dict (may be None for legacy artifacts).
    ``nn_feature_cols_hash`` is from :func:`unwrap_state_dict` (may be None).
    ``feature_cols`` and ``target_names`` are the current inference-time values.

    The shape check always applies. Hash/target checks apply only when the
    corresponding metadata is present, so legacy-formatted artifacts still
    load (with weaker guarantees) until the next retrain re-emits them.
    """
    feature_cols = list(feature_cols)
    target_names = list(target_names)
    n_features_expected = len(feature_cols)
    expected_hash = compute_feature_cols_hash(feature_cols)

    scaler_n = getattr(scaler, "n_features_in_", None)
    if scaler_n is not None and scaler_n != n_features_expected:
        raise RuntimeError(
            f"{position}: {scaler_label} was fit on {scaler_n} features but inference "
            f"expects {n_features_expected}. Retrain the pipeline for {position}."
        )

    if meta is not None:
        if meta.get("n_features") != n_features_expected:
            raise RuntimeError(
                f"{position}: {scaler_label}_meta.n_features={meta.get('n_features')} "
                f"but inference expects {n_features_expected}. Retrain the pipeline."
            )
        if meta.get("feature_cols_hash") != expected_hash:
            raise RuntimeError(
                f"{position}: {scaler_label} feature_cols_hash mismatch — scaler was "
                "fit on a different feature set than inference uses. Retrain the pipeline."
            )
        meta_targets = meta.get("target_names")
        if meta_targets is not None and list(meta_targets) != target_names:
            raise RuntimeError(
                f"{position}: {scaler_label}_meta.target_names={meta_targets} but "
                f"inference expects {target_names}. Retrain the pipeline."
            )

    if nn_feature_cols_hash is not None and nn_feature_cols_hash != expected_hash:
        raise RuntimeError(
            f"{position}: NN feature_cols_hash mismatch — NN weights were trained "
            "against a different feature set than inference uses. Retrain the pipeline."
        )

    if (
        meta is not None
        and nn_feature_cols_hash is not None
        and meta.get("feature_cols_hash") != nn_feature_cols_hash
    ):
        raise RuntimeError(
            f"{position}: {scaler_label} and NN weights have different "
            "feature_cols_hash — they came from different training runs. Retrain the "
            "pipeline so both artifacts are re-emitted together."
        )
