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

Preparation and loading delegate to ``src.prediction.frames.predict_position``,
the same adapter used by the serving artifact builder. Versioned bundles bind
ordered inputs, fitted preprocessing and model construction; older artifacts
use the explicit legacy registry adapter. Ridge, base NN, flat/opponent-history
attention, nested K attention and LightGBM are supported. Predictions stay at
their original precision here; the HTTP serializer rounds display values.

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

import numpy as np
import pandas as pd
import torch

from src.shared.aggregate_targets import predictions_to_fantasy_points
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
    pos,
    train_df,
    val_df,
    test_df,
    *,
    scoring_format="ppr",
    device=None,
    model_dir=None,
    kick_history=None,
    opponent_weekly=None,
):
    """Use the same prediction adapter as serving, including nested K history."""
    from src.prediction.frames import predict_position

    reg = dict(INFERENCE_REGISTRY[pos])
    reg["model_dir"] = resolve_model_dir(pos, reg, model_dir)
    if pos == "K" and kick_history is None:
        from src.k.data import load_kicks

        kick_history = load_kicks(pd.concat([train_df, val_df, test_df], ignore_index=True))
    if (
        reg.get("opp_attn_kind") == "offense"
        and reg.get("opp_attn_history_stats")
        and opponent_weekly is None
    ):
        from src.config import CACHE_DIR, SEASONS

        opponent_weekly = pd.read_parquet(f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet")
        if "season_type" in opponent_weekly:
            opponent_weekly = opponent_weekly[opponent_weekly["season_type"].eq("REG")]
    prediction = predict_position(
        pos,
        train_df,
        val_df,
        test_df,
        reg,
        kicks=kick_history,
        opponent_weekly=opponent_weekly,
        device=device or _device(),
    )
    result = prediction.frame.copy()
    for family, predictions in prediction.raw.items():
        result[f"pred_{family}_total"] = prediction.totals[family][scoring_format]
        for target in reg["targets"]:
            result[f"pred_{family}_{target}"] = predictions[target]
    result.attrs["prediction_errors"] = prediction.errors
    result.attrs["model_bundle_ids"] = prediction.bundle_ids
    for family, error in prediction.errors.items():
        print(f"[artifact_eval] {family}: {error}")
    try:
        validate_reconstruction(
            pos, result, model_dir=reg["model_dir"], scoring_format=scoring_format
        )
    except Exception as exc:
        print(f"[artifact_eval] {pos}: reconstruction self-check errored: {exc}")
    return result


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
