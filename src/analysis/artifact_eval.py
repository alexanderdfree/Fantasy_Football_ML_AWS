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
import json
import os

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


def _producer_model_dir(pos: str) -> str:
    """The staging path a LOCAL pipeline run / the Batch container writes to.

    ``src/batch/train.py::upload_artifacts`` (``src_model_dir``) and
    ``src/batch/launch.py`` (``local_model_dir``) both save to
    ``{pos.lower()}/outputs/models`` BEFORE the tarball is uploaded to S3. This is
    deliberately NOT the served/deploy path (``reg["model_dir"] =
    src/{pos}/outputs/models``), where ``model_sync`` extracts the S3 tarball and
    serving reads — the artifact travels producer → S3 → served. Keep this in sync
    with those two producers if the staging path ever moves.
    """
    return os.path.join(pos.lower(), "outputs", "models")


def _is_populated_dir(path: str) -> bool:
    """True iff ``path`` is a directory with at least one entry.

    An *empty* dir is treated as absent: a failed/partial ``--sync`` leaves an
    empty ``src/{pos}/outputs/models`` (``model_sync`` ``mkdir``s the dest before
    extracting), and preferring it would shadow a freshly-populated producer dir
    and resurrect the very per-model ``FileNotFoundError`` cascade this resolver
    exists to kill. Cheap entry-count, not a full artifact-manifest check.
    """
    try:
        return os.path.isdir(path) and bool(os.listdir(path))
    except OSError:
        return False


def resolve_model_dir(pos: str, reg: dict, override: str | None = None) -> str:
    """Locate ``pos``'s artifacts on disk, preferring the served path.

    Resolution order: explicit ``override`` → served/deploy path
    (``reg["model_dir"]``, where ``--sync`` + serving read) → producer/staging path
    (:func:`_producer_model_dir`, where a LOCAL ``run()`` / the Batch container
    writes them). The fallback is what lets this module score a *local* pipeline
    run's artifacts without an S3 ``--sync``. A served/producer dir that exists but
    is EMPTY counts as absent (:func:`_is_populated_dir`) so a failed sync can't
    shadow a populated producer dir.

    Raise loudly when NEITHER is populated. A missing artifact dir must not
    masquerade as "no models found": that silent degradation hid a served-vs-
    producer path mismatch (``reg["model_dir"]`` = ``src/{pos}/outputs/models`` vs.
    the local producer path ``{pos}/outputs/models``) behind four cryptic per-model
    ``FileNotFoundError`` warnings — the whole reason this resolver exists.
    """
    if override:
        return override
    served = reg["model_dir"]
    if _is_populated_dir(served):
        return served
    producer = _producer_model_dir(pos)
    if _is_populated_dir(producer):
        print(
            f"[artifact_eval] {pos}: served path {served!r} absent/empty; scoring LOCAL "
            f"producer-path artifacts {producer!r} (a local run's outputs, not "
            "necessarily the S3-deployed set — pass --sync for the served artifacts)."
        )
        return producer
    raise FileNotFoundError(
        f"{pos}: no model artifacts at served path {served!r} or producer path "
        f"{producer!r}. Run the position pipeline locally (it writes {producer!r}) "
        "or pass --sync to pull the served set from S3."
    )


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

    No retraining: each model is loaded from ``model_dir`` (resolved by
    :func:`resolve_model_dir` — served path ``src/{pos}/outputs/models``, falling
    back to the local producer path ``{pos}/outputs/models``) and run over the test
    split. A non-existent ``model_dir`` raises loudly; *per-model* load failures
    (a single model's file missing / scaler mismatch) are warned and skipped (that
    column is omitted), matching serving's graceful degradation, and the
    diagnostics' ``available_models`` keys off column presence.
    """
    reg = INFERENCE_REGISTRY[pos]
    device = device or _device()
    targets = list(reg["targets"])
    model_dir = resolve_model_dir(pos, reg, model_dir)

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
    # Capture the pre-filter train so RB/WR's team-total / share / HHI / career
    # features see the full player set, then return only the filtered rows —
    # exactly as training + serving do (#574/#531). fill_nans + the scaler still
    # fit on the filtered train (#569), so this stays faithful to the served
    # artifacts (this module's contract).
    full_train = pos_train
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= min_games].copy()

    feature_cols = reg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, reg, feature_cols, full_train=full_train
    )
    pos_test = pos_test.copy()
    X_test = pos_test[feature_cols].values.astype(np.float32)
    total_fn = _make_total_fn(pos, targets, reg, scoring_format)

    def _warn(model: str, exc: Exception) -> None:
        print(
            f"[artifact_eval] {pos} {model} load/predict failed (model_dir={model_dir!r}): "
            f"{exc!r} — column omitted"
        )

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
                        # Match training/serving: the raw weekly cache carries
                        # postseason rows; drop them so the opp-offense aggregates
                        # reproduce the REG-only path the model trained on (#424).
                        if "season_type" in opp_source_df.columns:
                            opp_source_df = opp_source_df[
                                opp_source_df["season_type"] == "REG"
                            ].copy()
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

    # Self-check: does the loaded deterministic Ridge still reproduce its recorded
    # training MAE on these splits? A large divergence means the saved artifact and
    # the current data/splits are different vintages — the artifact is STALE and its
    # reconstructed predictions will silently corrupt downstream diagnostics (the QB
    # incident: Ridge reconstruct 6.74 vs recorded 5.88 → a spurious −3.46 "elite
    # under-leveling"). Warn-only here so it never breaks an operator's run.
    try:
        validate_reconstruction(pos, pos_test, model_dir=model_dir, scoring_format=scoring_format)
    except Exception as e:  # noqa: BLE001 - a self-check must never break reconstruction
        print(f"[artifact_eval] {pos}: reconstruction self-check errored (non-fatal): {e}")

    return pos_test


# --- Stale-artifact (Ridge-tell) drift check ---------------------------------- #
# Healthy cross-run reconstruction Δ is ~0.02-0.03 FP MAE; the QB drift incident was
# Δ=0.86. warn=0.10 gives ~3x margin over the healthy band; fail=0.30. (Idiom from
# ``ab_harness._RIDGE_TOL`` but sized for cross-run/cross-vintage drift, not its 1e-9
# same-process tolerance.) PPR-only: the recorded ``ridge_metrics.total.mae`` is PPR,
# and only the skill positions carry a fantasy-point total in the split data.
_RECON_WARN_TOL = 0.10
_RECON_FAIL_TOL = 0.30
_RECON_VALIDATABLE_POS = ("QB", "RB", "WR", "TE")


def _reconstruction_verdict(delta: float, warn_tol: float, fail_tol: float) -> str:
    """Map an absolute Ridge-MAE drift to ``ok`` / ``warn`` / ``fail`` (pure, unit-tested)."""
    if delta <= warn_tol:
        return "ok"
    if delta <= fail_tol:
        return "warn"
    return "fail"


def validate_reconstruction(
    pos: str,
    test_df: pd.DataFrame,
    *,
    model_dir: str | None = None,
    scoring_format: str = "ppr",
    warn_tol: float = _RECON_WARN_TOL,
    fail_tol: float = _RECON_FAIL_TOL,
    strict: bool = False,
    verbose: bool = False,
) -> dict:
    """Flag a STALE artifact via the deterministic-Ridge data-identity tell.

    Compares the reconstructed Ridge total MAE against the model's own recorded training
    MAE (``ridge_metrics.total.mae`` in ``{model_dir}/benchmark_metrics.json``). To be a
    clean apples-to-apples staleness signal it rebuilds the actual total the SAME way the
    recorded MAE did — aggregating the *true target stats* via ``predictions_to_fantasy_points``
    (the canonical ``compute_fantasy_points_mae`` truth, mirroring ``rmse_gap_decomposition``),
    NOT the split ``fantasy_points`` column (which can carry scoring components outside a
    position's target set, e.g. a WR's rushing FP, adding a structural offset). Falls back
    to the ``fantasy_points`` column only when the target columns are absent. A large
    divergence means the saved artifact does not reproduce its recorded performance on the
    current splits. WARNs loudly on divergence (raises if ``strict`` and the verdict is not
    ``ok``); returns a verdict dict. ``skipped`` (no warning, no raise) when not applicable:
    non-skill position, non-PPR scoring, missing ``benchmark_metrics.json``/``ridge_metrics``,
    or missing ``pred_ridge_total`` and target columns (e.g. a Ridge load failure upstream).
    """

    def _skip(reason: str) -> dict:
        if verbose:
            print(f"[artifact_eval] {pos}: reconstruction self-check skipped ({reason}).")
        return {"status": "skipped", "reason": reason, "pos": pos}

    if pos.upper() not in _RECON_VALIDATABLE_POS:
        return _skip(f"{pos} carries no fantasy-point total in the split data")
    if scoring_format != "ppr":
        return _skip(f"recorded MAE is PPR; scoring_format={scoring_format!r}")
    if "pred_ridge_total" not in test_df.columns:
        return _skip("pred_ridge_total absent (Ridge load failed?)")
    targets = list(INFERENCE_REGISTRY[pos.upper()].get("targets", []))
    have_targets = bool(targets) and all(t in test_df.columns for t in targets)
    if not have_targets and "fantasy_points" not in test_df.columns:
        return _skip("neither target columns nor fantasy_points present")

    if model_dir is None:
        model_dir = resolve_model_dir(pos, INFERENCE_REGISTRY[pos.upper()])
    metrics_path = os.path.join(model_dir, "benchmark_metrics.json")
    if not os.path.exists(metrics_path):
        return _skip(
            f"no benchmark_metrics.json at {metrics_path} (no reference to validate against)"
        )
    try:
        with open(metrics_path) as f:
            recorded_blob = json.load(f)
        recorded = float(recorded_blob["ridge_metrics"]["total"]["mae"])
    except (OSError, KeyError, ValueError, TypeError) as e:
        return _skip(f"unreadable ridge_metrics.total.mae in benchmark_metrics.json ({e})")

    cols = ["pred_ridge_total", *(targets if have_targets else ["fantasy_points"])]
    sub = test_df[cols].dropna()
    if sub.empty:
        return _skip("no rows with pred_ridge_total + actual total")
    if have_targets:
        # Match the recorded MAE's ground truth (agg of true target stats), not the
        # fantasy_points column — else WR/TE carry a structural rushing-FP offset.
        true_fp = predictions_to_fantasy_points(
            pos, {t: sub[t].to_numpy(dtype=float) for t in targets}, scoring_format
        )
    else:
        true_fp = sub["fantasy_points"].to_numpy(dtype=float)  # fallback (targets absent)
    reconstructed = float(np.abs(sub["pred_ridge_total"].to_numpy(dtype=float) - true_fp).mean())
    delta = abs(reconstructed - recorded)
    verdict = _reconstruction_verdict(delta, warn_tol, fail_tol)
    result = {
        "status": verdict,
        "pos": pos,
        "reconstructed_ridge_mae": round(reconstructed, 4),
        "recorded_ridge_mae": round(recorded, 4),
        "delta": round(delta, 4),
        "git_sha": str(recorded_blob.get("git_sha", "?"))[:12],
        "split_run_id": str(recorded_blob.get("split_run_id", "?"))[:40],
    }
    if verdict != "ok":
        print(
            f"[artifact_eval] STALE ARTIFACT / SPLITS-VINTAGE MISMATCH: {pos} deterministic Ridge "
            f"reconstructs MAE={reconstructed:.4f} vs recorded {recorded:.4f} (Δ={delta:.4f}, "
            f"verdict={verdict}). The saved artifact (git {result['git_sha']}, split "
            f"{result['split_run_id']}) does not reproduce its recorded MAE on the CURRENT local "
            f"splits — EITHER the artifact is stale OR the local data/splits are a different vintage "
            f"than the artifact was trained on. Reconstructed predictions for {pos} are UNRELIABLE "
            f"until the two are realigned: regenerate the artifact on the current splits "
            f"(`python -m src.scripts.regen_served_artifacts --positions {pos}`), or refresh the "
            f"splits to the artifact's vintage."
        )
        if strict:
            raise RuntimeError(
                f"validate_reconstruction({pos}): STALE artifact (Δ={delta:.4f} > warn_tol={warn_tol})"
            )
    elif verbose:
        print(
            f"[artifact_eval] {pos}: reconstruction OK — Ridge MAE {reconstructed:.4f} "
            f"vs recorded {recorded:.4f} (Δ={delta:.4f})."
        )
    return result


def warn_if_sync_noop() -> bool:
    """Warn loudly when an ``--sync`` will silently no-op (S3 bucket unconfigured).

    ``src.shared.model_sync.sync_models_from_s3`` is opt-in via ``FF_MODEL_S3_BUCKET``
    and skips with only a terse ``[model_sync]`` line when it's unset — so ``--sync``
    then scores whatever (often stale) on-disk artifacts exist instead of the served
    set. That is the same silent failure this module's loud-failure contract kills,
    so surface it prominently. Mirror ``model_sync``'s ``.strip()`` gate exactly so a
    whitespace-only value (``FF_MODEL_S3_BUCKET=' '``) is flagged, not silently
    skipped. Returns ``True`` when the bucket is configured (sync will attempt to
    run); a configured bucket can still fetch nothing on bad creds/empty prefix —
    ``model_sync`` owns that failure path.
    """
    if os.environ.get("FF_MODEL_S3_BUCKET", "").strip():
        return True
    print(
        "[artifact_eval] WARNING: --sync requested but FF_MODEL_S3_BUCKET is unset — the S3 "
        "sync will NO-OP and you'll score whatever on-disk artifacts exist (likely stale). "
        "Set FF_MODEL_S3_BUCKET=<bucket> (and FF_MODEL_S3_PREFIX, default 'models') to sync."
    )
    return False


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="*", default=["QB", "RB", "WR", "TE", "K", "DST"])
    parser.add_argument(
        "--sync", action="store_true", help="Pull the latest artifacts from S3 before evaluating."
    )
    parser.add_argument("--scoring-format", default="ppr")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the Ridge-tell stale-artifact self-check per position and print its verdict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --validate, raise (non-zero exit) if any position's artifact is stale.",
    )
    args = parser.parse_args(argv)

    if args.sync:
        warn_if_sync_noop()
        from src.shared.model_sync import sync_models_from_s3

        print("Syncing latest model artifacts from S3 ...")
        sync_models_from_s3()

    from src.analysis.cohort_analysis import _load_splits

    train_df, val_df, test_df = _load_splits()
    for pos in (p.upper() for p in args.positions):
        # resolve_model_dir raises loudly when a position has NO artifacts; in a
        # multi-position sweep that should skip that position, not abort the rest
        # (e.g. a local run produced only QB/RB/WR/TE). The raise stays the loud
        # contract for library callers; the CLI logs it and continues.
        try:
            df = build_test_df_from_artifacts(
                pos, train_df, val_df, test_df, scoring_format=args.scoring_format
            )
        except FileNotFoundError as e:
            print(f"[artifact_eval] {pos}: {e} — skipping this position.")
            continue
        pred_cols = sorted(c for c in df.columns if c.startswith("pred_") and c.endswith("_total"))
        print(f"{pos}: {len(df)} test rows | model totals present: {pred_cols}")
        if args.validate:
            validate_reconstruction(
                pos, df, scoring_format=args.scoring_format, strict=args.strict, verbose=True
            )


if __name__ == "__main__":
    _main()
