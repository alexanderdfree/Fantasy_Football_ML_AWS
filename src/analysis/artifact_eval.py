"""Artifact-eval mode: score the most-recent SAVED model artifacts on the
held-out TEST split, WITHOUT retraining.

The read-only diagnostics (``src.analysis.tier_expert_comparison``,
``src.analysis.cohort_analysis``) historically call ``src.{pos}.run_pipeline.run()``
to obtain a ``test_df`` with per-row predictions — but ``run()`` *retrains* each
pipeline from scratch (slow on CPU, and it measures freshly-fit models rather than
the *served* artifacts; NN/Attn differ run-to-run by seed/device). This module
instead loads the persisted artifacts (the ones serving uses) and runs inference
on the test split, producing a ``test_df`` whose ``pred_{model}_total`` /
``pred_{model}_{target}`` columns match ``src.shared.pipeline`` exactly, so the
diagnostics consume it unchanged.

Design: it reuses the **same** primitives serving uses — ``INFERENCE_REGISTRY``
(``reg``), ``build_position_features`` (so the feature path can't drift from
training/serving), the ``artifact_integrity`` scaler checks, and the model
``.load()`` / ``predict_numpy`` calls — so there is one inference contract, not a
copy. The orchestration here mirrors ``src.serving.app._apply_position_models``;
the intended follow-up is to extract that shared inner loop into one helper both
serving and this module call (see PR description).

v1 scope: Ridge + base NN + **flat** Attention NN (including the opponent-history
side branch the skill positions use) + LightGBM. Only the nested per-kick (K)
attention variant is skipped (its attn column is omitted; the diagnostics' own
``available_models`` then degrades gracefully). Deterministic models (Ridge
always; LightGBM with fixed seed) reproduce the served numbers exactly; NN/Attn
match the served weights since they are loaded, not refit.

Usage (library):
    from src.analysis.artifact_eval import build_test_df_from_artifacts
    test_df = build_test_df_from_artifacts("RB", train_df, val_df, test_df)

CLI smoke (loads splits, optionally syncs latest artifacts from S3, prints the
prediction columns produced per position):
    python -m src.analysis.artifact_eval --positions QB RB WR TE --sync
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import MIN_GAMES_PER_SEASON
from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    build_game_history_arrays,
    build_opp_defense_history_arrays,
    get_attn_static_columns,
)
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import (
    assert_scaler_matches,
    read_scaler_meta,
    unwrap_state_dict,
)
from src.shared.feature_build import build_position_features, scale_and_clip
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import MultiHeadNet, MultiHeadNetWithHistory
from src.shared.registry import INFERENCE_REGISTRY


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_total_fn(pos: str, targets: list[str], reg: dict, scoring_format: str):
    """Return ``preds_dict -> total`` matching pipeline/serving aggregation.

    K uses a sign-vectored raw sum (``target_signs``); QB/RB/WR/TE/DST route
    through ``predictions_to_fantasy_points`` so totals are on the fantasy-point
    scale that ranking metrics compare like-for-like.
    """
    signs = reg.get("target_signs")
    if signs is not None:
        return lambda preds: sum(preds[t] * signs.get(t, 1.0) for t in targets)
    return lambda preds: predictions_to_fantasy_points(pos, preds, scoring_format=scoring_format)


def attach_predictions(
    pos_test: pd.DataFrame, name: str, preds: dict, targets: list[str], total_fn
) -> None:
    """Attach ``pred_{name}_total`` + ``pred_{name}_{target}`` columns in place.

    Mirrors the column shape ``src.shared.pipeline`` writes onto ``test_df`` so
    the cohort/tier diagnostics consume the frame unchanged.
    """
    pos_test[f"pred_{name}_total"] = total_fn(preds)
    for t in targets:
        pos_test[f"pred_{name}_{t}"] = preds[t]


def _attn_is_supported(reg: dict) -> bool:
    """v1 supports the flat attention branch (incl. the opponent-history side
    branch used by the skill positions); the nested per-kick variant (K) is not
    yet handled."""
    return reg.get("attn_history_structure", "flat") != "nested"


def build_test_df_from_artifacts(
    pos: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    scoring_format: str = "ppr",
    device: torch.device | None = None,
    model_dir: str | None = None,
) -> pd.DataFrame:
    """Load ``pos``'s saved artifacts and return its test_df with per-row preds.

    No retraining: each model is loaded from ``model_dir`` (default
    ``src/{pos}/outputs/models``) and run over the test split. Per-model load
    failures are warned and skipped (the column is omitted), matching serving's
    graceful degradation; the diagnostics' ``available_models`` keys off column
    presence.
    """
    reg = INFERENCE_REGISTRY[pos]
    device = device or _device()
    targets = list(reg["targets"])
    model_dir = model_dir or reg["model_dir"]

    pos_train = reg["filter_fn"](train_df)
    pos_val = reg["filter_fn"](val_df)
    pos_test = reg["filter_fn"](test_df)
    if pos not in ("K", "DST"):
        pos_train = reg["compute_targets_fn"](pos_train)
        pos_val = reg["compute_targets_fn"](pos_val)
        pos_test = reg["compute_targets_fn"](pos_test)

    # Mirror the train-only min-games filter that training + serving apply to
    # pos_train BEFORE feature-building (src.serving.app._apply_position_models /
    # src.shared.pipeline._prepare_position_data): the fill-means + StandardScaler
    # are fit on pos_train, so an unfiltered train frame drifts those stats from
    # the served artifacts and these preds stop reproducing the served numbers —
    # this module's whole contract. val/test stay unfiltered, as in training. (#977)
    min_games = reg.get("min_games_per_season")
    if min_games is None:
        min_games = MIN_GAMES_PER_SEASON
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= min_games].copy()

    feature_cols = reg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, reg, feature_cols
    )
    pos_test = pos_test.copy()
    X_test = pos_test[feature_cols].values.astype(np.float32)
    total_fn = _make_total_fn(pos, targets, reg, scoring_format)

    def _warn(model: str, exc: Exception) -> None:
        print(f"[artifact_eval] {pos} {model} load/predict failed: {exc!r} — column omitted")

    # --- Ridge (deterministic; reproduces the served numbers exactly) ---------
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        attach_predictions(pos_test, "ridge", ridge.predict(X_test), targets, total_fn)
    except Exception as e:  # noqa: BLE001 - graceful per-model degradation (mirrors serving)
        _warn("ridge", e)

    # --- Base NN --------------------------------------------------------------
    try:
        nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        nn_meta = read_scaler_meta(f"{model_dir}/nn_scaler_meta.json")
        ckpt = torch.load(f"{model_dir}/{reg['nn_file']}", map_location=device, weights_only=True)
        state_dict, scaler_hash = unwrap_state_dict(ckpt)
        assert_scaler_matches(
            pos, nn_scaler, scaler_hash, nn_meta, feature_cols, targets, scaler_label="nn_scaler"
        )
        x_scaled = scale_and_clip(nn_scaler, X_test)
        model = MultiHeadNet(
            input_dim=len(feature_cols), target_names=targets, **reg["nn_kwargs"]
        ).to(device)
        model.load_state_dict(state_dict)
        attach_predictions(pos_test, "nn", model.predict_numpy(x_scaled, device), targets, total_fn)
    except Exception as e:  # noqa: BLE001
        _warn("nn", e)

    # --- Attention NN (flat variant only in v1) -------------------------------
    if reg.get("train_attention_nn", False) and reg.get("attn_nn_file"):
        if not _attn_is_supported(reg):
            print(
                f"[artifact_eval] {pos}: nested/opp-history attention not supported in v1; "
                "attn_nn column omitted."
            )
        else:
            try:
                if reg.get("attn_static_from_df", False):
                    attn_static_cols = list(reg.get("attn_static_features", []))
                    x_attn = pos_test[attn_static_cols].to_numpy(dtype=np.float32)
                else:
                    attn_static_cols = get_attn_static_columns(
                        feature_cols, reg.get("attn_static_features", [])
                    )
                    col_set = set(attn_static_cols)
                    idx = [i for i, c in enumerate(feature_cols) if c in col_set]
                    x_attn = X_test[:, idx]

                attn_scaler = joblib.load(f"{model_dir}/attention_nn_scaler.pkl")
                attn_meta = read_scaler_meta(f"{model_dir}/attention_nn_scaler_meta.json")
                ckpt = torch.load(
                    f"{model_dir}/{reg['attn_nn_file']}", map_location=device, weights_only=True
                )
                state_dict, scaler_hash = unwrap_state_dict(ckpt)
                assert_scaler_matches(
                    pos,
                    attn_scaler,
                    scaler_hash,
                    attn_meta,
                    attn_static_cols,
                    targets,
                    scaler_label="attention_nn_scaler",
                )
                x_attn_scaled = scale_and_clip(attn_scaler, x_attn)
                max_seq_len = reg.get("attn_max_seq_len", 17)
                hist, mask = build_game_history_arrays(
                    pos_test,
                    history_stats=list(reg.get("attn_history_stats", [])),
                    max_seq_len=max_seq_len,
                )

                # Opponent-history side branch — live for the skill positions
                # (kind "defense", aggregated over the all-position concat). DST's
                # "offense" kind needs the weekly cache parquet; if it's absent the
                # enclosing except omits the attn column (graceful).
                opp_history_stats = reg.get("opp_attn_history_stats") or []
                opp_hist = opp_mask = None
                opp_game_dim = None
                if opp_history_stats:
                    opp_kind = reg.get("opp_attn_kind", "defense")
                    builder = OPP_ATTN_PER_GAME_BUILDERS[opp_kind]
                    if opp_kind == "offense":
                        from src.config import CACHE_DIR, SEASONS

                        opp_source_df = pd.read_parquet(
                            f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
                        )
                    else:
                        opp_source_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
                    opp_per_game = builder(opp_source_df)
                    opp_hist, opp_mask = build_opp_defense_history_arrays(
                        pos_test,
                        opp_per_game,
                        opp_history_stats,
                        reg.get("opp_attn_max_seq_len", max_seq_len),
                    )
                    opp_game_dim = opp_hist.shape[2]

                model = MultiHeadNetWithHistory(
                    static_dim=len(attn_static_cols),
                    game_dim=hist.shape[2],
                    target_names=targets,
                    opp_game_dim=opp_game_dim,
                    **reg["attn_nn_kwargs_static"],
                ).to(device)
                model.load_state_dict(state_dict)
                if opp_game_dim is not None:
                    preds = model.predict_numpy(
                        x_attn_scaled,
                        hist,
                        mask,
                        device,
                        X_opp_history=opp_hist,
                        opp_history_mask=opp_mask,
                    )
                else:
                    preds = model.predict_numpy(x_attn_scaled, hist, mask, device)
                attach_predictions(pos_test, "attn_nn", preds, targets, total_fn)
            except Exception as e:  # noqa: BLE001
                _warn("attn_nn", e)

    # --- LightGBM (QB/RB/WR/TE only) ------------------------------------------
    if reg.get("train_lightgbm", False):
        try:
            lgbm = LightGBMMultiTarget(target_names=targets)
            lgbm.load(model_dir)
            attach_predictions(pos_test, "lgbm", lgbm.predict(X_test), targets, total_fn)
        except Exception as e:  # noqa: BLE001
            _warn("lgbm", e)

    return pos_test


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="*", default=["QB", "RB", "WR", "TE", "K", "DST"])
    parser.add_argument(
        "--sync", action="store_true", help="Pull the latest artifacts from S3 before evaluating."
    )
    parser.add_argument("--scoring-format", default="ppr")
    args = parser.parse_args(argv)

    if args.sync:
        from src.shared.model_sync import sync_models_from_s3

        print("Syncing latest model artifacts from S3 ...")
        sync_models_from_s3()

    from src.analysis.cohort_analysis import _load_splits

    train_df, val_df, test_df = _load_splits()
    for pos in (p.upper() for p in args.positions):
        df = build_test_df_from_artifacts(
            pos, train_df, val_df, test_df, scoring_format=args.scoring_format
        )
        pred_cols = sorted(c for c in df.columns if c.startswith("pred_") and c.endswith("_total"))
        print(f"{pos}: {len(df)} test rows | model totals present: {pred_cols}")


if __name__ == "__main__":
    _main()
