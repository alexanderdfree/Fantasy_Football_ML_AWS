"""Post-upload smoke test: load every artifact for a position and run a
minimal predict on each. Raises ``SmokeTestFailed`` on any failure.

Called by ``batch/train.py::upload_artifacts`` after ``_validate_remote_tarball``
and before the manifest write. A pass advances the manifest's ``stable``
pointer; a failure leaves ``stable`` pinned to the previous good artifact
so the frontend keeps serving last-known-good. The new artifact still lands
in ``current`` and ``history/`` for forensics either way.

Why the load+predict path mirrors ``app.py::_apply_position_models`` rather
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

import joblib
import numpy as np
import torch

from shared.artifact_integrity import assert_scaler_matches, read_scaler_meta, unwrap_state_dict
from shared.feature_build import scale_and_clip
from shared.models import LightGBMMultiTarget, RidgeMultiTarget
from shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)


class SmokeTestFailed(RuntimeError):
    """Raised when the post-upload smoke test detects a broken artifact.

    The producer (``batch/train.py``) catches this and refuses to advance
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
    """Mirror the dispatch in ``app.py::_apply_position_models``: K reads
    its attention static cols straight off the DataFrame; everyone else
    filters the live ``feature_cols`` through the position's whitelist.
    """
    if reg.get("attn_static_from_df", False):
        return list(reg.get("attn_static_features", []))
    # Lazy import to avoid pulling src/features at smoke-test top level.
    from src.features.engineer import get_attn_static_columns

    return list(get_attn_static_columns(feature_cols, reg.get("attn_static_features", [])))


def _smoke_attention(
    pos: str,
    reg: dict,
    model_dir: str,
    feature_cols: list[str],
    targets: list[str],
    device: torch.device,
) -> None:
    """Load attention NN, integrity-check scaler+weights, instantiate the
    correct model class, load state_dict, and run a forward pass on synthetic
    zero history. Raises ``SmokeTestFailed`` (chained) on any error.
    """
    attn_scaler = joblib.load(f"{model_dir}/attention_nn_scaler.pkl")
    attn_meta = read_scaler_meta(f"{model_dir}/attention_nn_scaler_meta.json")
    attn_checkpoint = torch.load(
        f"{model_dir}/{reg['attn_nn_file']}",
        map_location=device,
        weights_only=True,
    )
    attn_state_dict, attn_hash = unwrap_state_dict(attn_checkpoint)

    attn_static_cols = _resolve_attn_static_cols(reg, feature_cols)
    assert_scaler_matches(
        pos,
        attn_scaler,
        attn_hash,
        attn_meta,
        attn_static_cols,
        targets,
        scaler_label="attention_nn_scaler",
    )

    n_static = len(attn_static_cols)
    X_static = np.zeros((1, n_static), dtype=np.float32)
    X_static_scaled = scale_and_clip(attn_scaler, X_static)

    structure = reg.get("attn_history_structure", "flat")
    if structure == "nested":
        kick_dim = len(reg.get("attn_kick_stats", []))
        max_games = reg.get("attn_max_games", 17)
        max_kicks = reg.get("attn_max_kicks_per_game", 10)
        attn_model = MultiHeadNetWithNestedHistory(
            static_dim=n_static,
            kick_dim=kick_dim,
            target_names=targets,
            **reg["attn_nn_kwargs_static"],
        ).to(device)
        attn_model.load_state_dict(attn_state_dict)
        x_kicks = np.zeros((1, max_games, max_kicks, kick_dim), dtype=np.float32)
        outer_mask = np.ones((1, max_games), dtype=bool)
        inner_mask = np.ones((1, max_games, max_kicks), dtype=bool)
        preds = attn_model.predict_numpy(X_static_scaled, x_kicks, outer_mask, inner_mask, device)
    else:
        history_stats = list(reg.get("attn_history_stats", []) or [])
        max_seq_len = reg.get("attn_max_seq_len", 17)
        opp_history_stats = list(reg.get("opp_attn_history_stats", []) or [])
        opp_game_dim = len(opp_history_stats) if opp_history_stats else None
        attn_model = MultiHeadNetWithHistory(
            static_dim=n_static,
            game_dim=len(history_stats),
            target_names=targets,
            opp_game_dim=opp_game_dim,
            **reg["attn_nn_kwargs_static"],
        ).to(device)
        attn_model.load_state_dict(attn_state_dict)
        hist = np.zeros((1, max_seq_len, len(history_stats)), dtype=np.float32)
        mask = np.ones((1, max_seq_len), dtype=bool)
        opp_hist = opp_mask = None
        if opp_game_dim is not None:
            opp_max_seq_len = reg.get("opp_attn_max_seq_len", max_seq_len)
            opp_hist = np.zeros((1, opp_max_seq_len, opp_game_dim), dtype=np.float32)
            opp_mask = np.ones((1, opp_max_seq_len), dtype=bool)
        preds = attn_model.predict_numpy(
            X_static_scaled,
            hist,
            mask,
            device,
            X_opp_history=opp_hist,
            opp_history_mask=opp_mask,
        )
    _assert_finite_dict(pos, "attn_nn", preds, targets)


def run_smoke_test(pos: str, model_dir: str | os.PathLike) -> None:
    """Load + minimally predict every model artifact for ``pos``.

    Raises :class:`SmokeTestFailed` on any failure. The test runs entirely
    on CPU regardless of GPU availability — the cost is dominated by torch
    deserialization, which is the same on either device.
    """
    from shared.registry import INFERENCE_REGISTRY

    model_dir = str(model_dir)
    if not Path(model_dir).is_dir():
        raise SmokeTestFailed(f"{pos}: model_dir {model_dir!r} does not exist")

    reg = INFERENCE_REGISTRY[pos]
    targets = list(reg["targets"])
    feature_cols = list(reg["get_feature_columns_fn"]())
    n_features = len(feature_cols)
    device = torch.device("cpu")

    X_zero = np.zeros((1, n_features), dtype=np.float32)

    # Ridge
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        ridge_preds = ridge.predict(X_zero)
    except Exception as e:
        raise SmokeTestFailed(f"{pos} ridge: {e!r}") from e
    _assert_finite_dict(pos, "ridge", ridge_preds, targets)

    # NN base
    try:
        nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        nn_meta = read_scaler_meta(f"{model_dir}/nn_scaler_meta.json")
        nn_checkpoint = torch.load(
            f"{model_dir}/{reg['nn_file']}", map_location=device, weights_only=True
        )
        nn_state_dict, nn_hash = unwrap_state_dict(nn_checkpoint)
        assert_scaler_matches(
            pos,
            nn_scaler,
            nn_hash,
            nn_meta,
            feature_cols,
            targets,
            scaler_label="nn_scaler",
        )
        X_scaled = scale_and_clip(nn_scaler, X_zero)
        nn_model = MultiHeadNet(input_dim=n_features, target_names=targets, **reg["nn_kwargs"]).to(
            device
        )
        nn_model.load_state_dict(nn_state_dict)
        nn_preds = nn_model.predict_numpy(X_scaled, device)
    except Exception as e:
        raise SmokeTestFailed(f"{pos} nn: {e!r}") from e
    _assert_finite_dict(pos, "nn", nn_preds, targets)

    # Attention NN
    if reg.get("train_attention_nn", False) and reg.get("attn_nn_file"):
        try:
            _smoke_attention(pos, reg, model_dir, feature_cols, targets, device)
        except SmokeTestFailed:
            raise
        except Exception as e:
            raise SmokeTestFailed(f"{pos} attn_nn: {e!r}") from e

    # LightGBM
    if reg.get("train_lightgbm", False):
        try:
            lgbm = LightGBMMultiTarget(target_names=targets)
            lgbm.load(model_dir)
            lgbm_preds = lgbm.predict(X_zero)
        except Exception as e:
            raise SmokeTestFailed(f"{pos} lgbm: {e!r}") from e
        _assert_finite_dict(pos, "lgbm", lgbm_preds, targets)
