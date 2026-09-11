"""Post-upload smoke test: load every artifact for a position and run a
minimal predict on each. Raises ``SmokeTestFailed`` on any failure.

Called by ``src/batch/train.py::upload_artifacts`` after ``_validate_remote_tarball``
and before the manifest write. A pass advances the manifest's ``stable``
pointer; a failure leaves ``stable`` pinned to the previous good artifact
so the frontend keeps serving last-known-good. The new artifact still lands
in ``current`` and ``history/`` for forensics either way.

Why the load+predict path mirrors ``src/serving/core.py::_apply_position_models`` rather
than re-using it: the Flask layer's path requires a fully-built feature
DataFrame (filter → compute_targets → build_position_features → attention
history arrays), which couples the smoke test to data availability and
position-specific quirks. We instead call ``predict`` on synthetic zero
inputs of the right shape — sufficient to catch the failure modes that
matter at promotion time:

1. Pickle / torch.load deserialization errors (e.g. class import path drift).
2. Shape-mismatch state-dict assignment (NN trained on a different feature
   count than the live registry exposes).
3. Scaler/weights ``feature_cols_hash`` drift (caught by
   ``assert_scaler_matches`` — the canonical training/inference skew check).
4. NaN/Inf predictions on a benign input (rare but real — e.g. a head
   collapsed to nan during training and got serialized).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class SmokeTestFailed(RuntimeError):
    """Raised when the post-upload smoke test detects a broken artifact.

    The producer (``src/batch/train.py``) catches this and refuses to advance
    the manifest's ``stable`` pointer. The frontend keeps serving last-known-
    good in the meantime.
    """


def _assert_finite_dict(pos: str, model_label: str, preds: dict, targets: list[str]) -> None:
    """All target heads present, finite, and the right batch length."""
    missing = set(targets) - set(preds)
    if missing:
        raise SmokeTestFailed(
            f"{pos} {model_label}: missing target heads {sorted(missing)}; got {sorted(preds)}"
        )
    for t in targets:
        arr = np.asarray(preds[t])
        if arr.size == 0:
            raise SmokeTestFailed(f"{pos} {model_label}: target {t!r} produced empty array")
        if not np.all(np.isfinite(arr)):
            raise SmokeTestFailed(
                f"{pos} {model_label}: target {t!r} contains NaN/Inf — model is broken"
            )


def _resolve_attn_static_cols(reg: dict, feature_cols: list[str]) -> list[str]:
    """Mirror the dispatch in ``src/serving/core.py::_apply_position_models``: K reads
    its attention static cols straight off the DataFrame; everyone else
    filters the live ``feature_cols`` through the position's whitelist.
    """
    if reg.get("attn_static_from_df", False):
        return list(reg.get("attn_static_features", []))
    # Lazy import to avoid pulling src/features at smoke-test top level.
    from src.features.engineer import get_attn_static_columns

    return list(get_attn_static_columns(feature_cols, reg.get("attn_static_features", [])))


def _smoke_attention(pos, reg, model_dir, feature_cols, targets, device):
    from src.prediction.predictor import Predictor

    try:
        predictor = Predictor.from_bundle(
            model_dir, "attn_nn", position=pos, legacy_spec=reg, device=device
        )
        predictions = predictor.predict_raw(predictor.zero_inputs())
    except Exception as exc:
        raise SmokeTestFailed(f"{pos} attn_nn: {exc!r}") from exc
    _assert_finite_dict(pos, "attn_nn", predictions, targets)


def run_smoke_test(pos: str, model_dir: str | os.PathLike) -> None:
    """Validate every required family with the production prediction adapter."""
    from src.prediction.predictor import Predictor
    from src.shared.registry import INFERENCE_REGISTRY

    if not Path(model_dir).is_dir():
        raise SmokeTestFailed(f"{pos}: model_dir {str(model_dir)!r} does not exist")
    reg = INFERENCE_REGISTRY[pos]
    families = ["ridge", "nn"]
    if reg.get("train_attention_nn", False) and reg.get("attn_nn_file"):
        families.append("attn_nn")
    if reg.get("train_lightgbm", False):
        families.append("lgbm")
    for family in families:
        try:
            predictor = Predictor.from_bundle(
                model_dir, family, position=pos, legacy_spec=reg, device=torch.device("cpu")
            )
            predictions = predictor.predict_raw(predictor.zero_inputs())
        except Exception as exc:
            raise SmokeTestFailed(f"{pos} {family}: {exc!r}") from exc
        _assert_finite_dict(pos, family, predictions, list(predictor.schema.targets))
