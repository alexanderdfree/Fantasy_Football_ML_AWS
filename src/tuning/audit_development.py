"""Isolated historical development cells; all fitting is restricted to AWS Batch.

Reuse the September repair campaign's production providers and content-addressed
observer. This campaign measures previously untested single fixes on fresh main;
it cannot admit a candidate to confirmation or change production defaults.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from functools import wraps

import numpy as np
import pandas as pd

from src.prediction.bundle import canonical_json
from src.shared.comparison_scoring import comparison_actuals, comparison_model_totals
from src.shared.evaluation_cohorts import regular_season_rows
from src.training.context import current_context

KEYS = ["player_id", "season", "week"]
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")
BASELINE_SHA = "ecddeca88d8011cc18866b8842815bd7bc7999e5"
CANDIDATE_PINS = {
    "count_precision": "00873f2b9dbddfc338b53604555ccc9ae9bac97f",
    "bagging": "d690b417e6d521d379843b924707db8ab16efff1",
    "stint_reset": "d27a876cc80dde55a1d33f65b04664171fb3380f",
}
STATE = {}


def install_native_origin(position):
    """Preserve native K/DST construction and K kick-history bindings.

    Defer native context imputation until the existing fold preparation boundary;
    slicing already-imputed production data would leak later seasons.
    """
    from src.training.contracts import DatasetSplits

    module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
    original = getattr(module.provide_dataset, "_audit_original", module.provide_dataset)

    @wraps(original)
    def provider(cfg, *, cross_validation=False):
        if cross_validation:
            raise ValueError("Development cells run one chronological fold, not nested CV")
        dataset = original(cfg, cross_validation=True)
        frames = split_origin(dataset.frames, STATE["origin"], position=position)
        STATE["frames"] = frames
        return DatasetSplits(*frames, dataset.bindings)

    provider._audit_original = original
    module.provide_dataset = provider


def select_changes(arm):
    """Reset every switch on every cell, including baseline and repeat controls."""
    from src.shared import training
    from src.shared.models import LightGBMMultiTarget
    from src.tuning import audit_count_candidate, audit_stint_candidate

    for position in ("wr", "te"):
        module = importlib.import_module(f"src.{position}.features")
        current = module._compute_features
        original = getattr(current, "_audit_original", current)
        candidate = getattr(audit_stint_candidate, f"{position}_compute_features")
        candidate._audit_original = original
        module._compute_features = candidate if arm == "stint_reset" else original

    for name in ("negbin2_log_prob", "ztnb2_log_prob", "ztp_log_prob"):
        current = getattr(training, name)
        original = getattr(current, "_audit_original", current)
        candidate = getattr(audit_count_candidate, name)
        candidate._audit_original = original
        setattr(training, name, candidate if arm == "count_precision" else original)

    original_init = getattr(
        LightGBMMultiTarget.__init__, "_audit_original", LightGBMMultiTarget.__init__
    )

    @wraps(original_init)
    def initialize(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        if arm == "bagging":
            frequency = 1 if model._params["subsample"] < 1.0 else 0
            model._params["subsample_freq"] = frequency
            for estimator in model._models.values():
                estimator.set_params(subsample_freq=frequency)

    initialize._audit_original = original_init
    LightGBMMultiTarget.__init__ = initialize


def install_observer():
    """Observe restored native checkpoints without changing training trajectories."""
    from src.shared.training import MultiHeadTrainer

    original = getattr(MultiHeadTrainer.train, "_audit_original", MultiHeadTrainer.train)

    @wraps(original)
    def train(trainer, train_loader, val_loader, n_epochs):
        import torch

        from src.analysis.repair_count_diagnostics import error_summary, observed_likelihood_check
        from src.tuning.audit_input_observer import observe_loaders

        observe_loaders(train_loader, val_loader)
        history = original(trainer, train_loader, val_loader, n_epochs)
        predictions, truth = _validation_predictions(trainer, val_loader)
        selection = history["checkpoint_selection"]
        rescored = _checkpoint_metrics(predictions, truth, STATE["position"])
        selected_epoch = selection["epoch"]
        for metric in ("mae", "rmse"):
            expected = history[f"val_fantasy_{metric}_ppr"][selected_epoch - 1]
            if not np.isclose(rescored[metric], expected, rtol=2e-5, atol=2e-6):
                raise ValueError(f"Restored checkpoint does not independently reproduce {metric}")
        report = {
            "family": "nn" if type(trainer) is MultiHeadTrainer else "attn_nn",
            "device": str(trainer.device),
            "amp": bool(trainer._use_amp),
            "graph": trainer._graphed_step is not None,
            "checkpoint_selection": selection,
            "epochs_executed": len(history["val_selection_metric"]),
            "stop_reason": "budget"
            if len(history["val_selection_metric"]) == n_epochs
            else "patience",
            "restored_scores": rescored,
            "validation_errors": error_summary(predictions, truth),
        }
        if "receptions_value_mu" in predictions:
            mu = torch.tensor(predictions["receptions_value_mu"], dtype=torch.float64)
            alpha = torch.tensor(
                predictions["receptions_value_log_alpha"], dtype=torch.float64
            ).exp()
            conditional_mean = mu / -torch.expm1(-torch.log1p(alpha * mu) / alpha)
            predictions["receptions_corrected_conditional_mean"] = conditional_mean.numpy()
            predictions["receptions_gate_probability"] = torch.sigmoid(
                torch.tensor(predictions["receptions_gate_logit"])
            ).numpy()
            report["count_numerics"] = observed_likelihood_check(
                truth["receptions"],
                predictions["receptions_value_mu"],
                predictions["receptions_value_log_alpha"],
            )
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            **{f"prediction__{k}": v for k, v in predictions.items()},
            **{f"truth__{k}": v for k, v in truth.items()},
        )
        family = report["family"]
        report["validation_raw"] = evidence(f"{family}-validation.npz", payload.getvalue())
        checkpoint = io.BytesIO()
        torch.save(
            {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}, checkpoint
        )
        report["checkpoint"] = evidence(f"{family}-selected.pt", checkpoint.getvalue())
        STATE["trainers"].append(report)
        return history

    train._audit_original = original
    MultiHeadTrainer.train = train


def configured_position(config):
    """Use the native loader identity; WR and TE deliberately share targets."""
    module = getattr(config.get("filter_fn"), "__module__", "")
    expected = {
        f"src.{position.lower()}.data": position
        for position in ("QB", "RB", "WR", "TE", "K", "DST")
    }
    if module not in expected:
        raise ValueError(f"Unknown production position filter identity: {module!r}")
    return expected[module]


def configure(config, *, arm):

    if not os.environ.get("AWS_BATCH_JOB_ID"):
        raise RuntimeError("All model fitting, including smoke cells, must run on AWS Batch")
    if os.environ.get("FF_AB_STACKED", "").lower() in {"1", "true", "yes"} or int(
        os.environ.get("FF_NN_FIXED_EPOCHS", "0") or 0
    ):
        raise ValueError("Development requires nonstacked production stopping")
    if os.environ.get("FF_AMP_DTYPE", "fp32") != "fp32":
        raise ValueError("Development requires production FP32 execution")
    if arm not in {"baseline", "baseline_rep", *CANDIDATE_PINS}:
        raise ValueError(f"Unknown isolated arm: {arm}")
    origin = int(os.environ.get("FF_AUDIT_ORIGIN", "2022"))
    if origin not in (2022, 2023):
        raise ValueError(
            "Freeze a qualifying candidate before any confirmation; this spec refuses 2024/25"
        )
    position = configured_position(config)
    if arm == "count_precision" and position not in {"RB", "WR", "TE"}:
        raise ValueError("Count candidate has no production hurdle head on this position")
    if arm == "stint_reset" and position not in {"WR", "TE"}:
        raise ValueError("Stint reset changes WR/TE feature building only")
    STATE.clear()
    STATE.update(arm=arm, origin=origin, position=position, trainers=[])
    select_changes(arm)
    install_observer()
    if position in {"WR", "TE"}:
        from src.tuning.audit_input_observer import install

        install(STATE)
    if position in {"K", "DST"}:
        module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
        config = module.with_fold_imputation(config)
        install_native_origin(position)
    return config


def variants(candidate, *, native=False):
    from functools import partial

    from src.tuning.ab_harness import Variant

    return [
        Variant(
            arm,
            cfg_mutator=partial(configure, arm=arm),
            frame_injector=None if native else origin_frames,
            expect_ridge_identical=None if arm == "baseline" else arm != "stint_reset",
        )
        for arm in ("baseline", "baseline_rep", candidate)
    ]


def split_origin(frames, origin, *, position):
    from src.data.split import rolling_origin_folds

    if origin not in (2022, 2023, 2024, 2025):
        raise ValueError("Origin is outside the frozen repair protocol")
    full = pd.concat([f for f in frames if f is not None], ignore_index=True)
    if position == "DST":
        # The native provider selects REG schedules before constructing rows;
        # its team-level schema does not carry generic-player season_type.
        train = full[full.season.between(2013, origin - 2)].copy()
        val = full[full.season.eq(origin - 1)].copy()
        test = full[full.season.eq(origin)].copy()
    else:
        folds = rolling_origin_folds(
            full, test_seasons=[origin], min_train_season=2015 if position == "K" else 2013
        )
        _, train, val, test = folds[0]
    if any(f.empty for f in (train, val, test)):
        raise ValueError(f"Empty historical split for {position}/{origin}")
    return train, val, test


def origin_frames(train, val, test):
    frames = split_origin((train, val, test), STATE["origin"], position="skill")
    STATE["frames"] = frames
    return frames


def evidence(name, payload):
    """Content-addressed evidence cannot overwrite production artifacts."""
    import boto3
    from botocore.exceptions import ClientError

    context = current_context()
    if context is None or not os.environ.get("AWS_BATCH_JOB_ID"):
        raise RuntimeError("Repair evidence must come from an AWS Batch cell")
    prefix = os.environ.get("FF_AB_S3_PREFIX", "").strip("/")
    run = os.environ.get("FF_AB_RUN_ID", "")
    if not prefix.startswith("ab_runs/") or any(p in {"", ".", ".."} for p in prefix.split("/")):
        raise ValueError("Repair evidence requires an isolated ab_runs subdirectory")
    if not run or "/" in run or run in {".", ".."}:
        raise ValueError("Invalid repair run ID")
    digest = hashlib.sha256(payload).hexdigest()
    cell = f"{STATE['position']}-{STATE['arm']}-{context.seed}"
    key = f"{prefix}/{run}/evidence/{cell}/{digest}-{name}"
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=payload, IfNoneMatch="*")
    except ClientError as error:
        if error.response["Error"]["Code"] != "PreconditionFailed":
            raise
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise ValueError("Existing content-addressed evidence is corrupt") from error
    return {"uri": f"s3://{bucket}/{key}", "sha256": digest, "bytes": len(payload)}


def _validation_predictions(trainer, loader):
    import torch

    predictions, truth = {}, {}
    # Iterating an ordinary DataLoader may consume its base RNG seed. Preserve
    # all RNG state so a diagnostic cannot change the next model's fit.
    devices = [torch.cuda.current_device()] if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        trainer.model.eval()
        for batch in loader:
            predicted, actual = trainer._forward_batch(batch)
            for name, values in predicted.items():
                predictions.setdefault(name, []).append(values.detach().float().cpu().numpy())
            for name, values in actual.items():
                truth.setdefault(name, []).append(values.detach().float().cpu().numpy())
    return (
        {k: np.concatenate(v) for k, v in predictions.items()},
        {k: np.concatenate(v) for k, v in truth.items()},
    )


def _checkpoint_metrics(predictions, truth, position):
    from src.shared.aggregate_targets import predictions_to_fantasy_points

    pred = np.asarray(predictions_to_fantasy_points(position, predictions, "ppr"), dtype=float)
    actual = np.asarray(predictions_to_fantasy_points(position, truth, "ppr"), dtype=float)
    error = pred - actual
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def prepared_hashes(prepared):
    hashes = {}
    for split in ("train", "val", "test"):
        frame = getattr(prepared, split)
        frame = frame[[c for c in frame if not c.startswith("pred_")]]
        schema = repr(list(zip(frame.columns, map(str, frame.dtypes), strict=True))).encode()
        hashes[split] = hashlib.sha256(
            schema + pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest()
        values = np.ascontiguousarray(getattr(prepared, f"X_{split}"))
        hashes[f"X_{split}"] = hashlib.sha256(
            str((values.shape, values.dtype)).encode() + values.tobytes()
        ).hexdigest()
        for target, values in getattr(prepared, f"y_{split}").items():
            values = np.ascontiguousarray(values)
            hashes[f"y_{split}/{target}"] = hashlib.sha256(
                str((values.shape, values.dtype)).encode() + values.tobytes()
            ).hexdigest()
    return hashes


def precise_cohorts(result, frame, position):
    """Recompute full precision on the canonical cohort memberships, verifying hashes."""
    from copy import deepcopy

    from src.shared.evaluation_cohorts import (
        _identity,
        build_cohorts,
        load_reference,
        ranked_rows,
        reference_selection,
    )

    context = current_context()
    prior_frames = (result.prepared.train, result.prepared.val)
    blocks = build_cohorts(
        position, frame, prior_frames=prior_frames, reference_dir=context.raw_root
    )
    blocks = deepcopy(blocks)
    df = comparison_model_totals(regular_season_rows(frame), position)
    df["player_id"] = df.player_id.astype(str)
    df["fantasy_points"] = comparison_actuals(df, position)
    df = df[df.fantasy_points.notna()].copy()
    prior = pd.concat([regular_season_rows(f) for f in prior_frames])
    prior["fantasy_points"] = comparison_actuals(prior, position)
    prior = prior.drop_duplicates(KEYS).groupby(["player_id", "season"]).fantasy_points.mean()
    lookup = pd.MultiIndex.from_arrays([df.player_id, df.season - 1])
    df["prior_points"] = prior.reindex(lookup).to_numpy()
    players = ranked_rows(
        df.drop_duplicates(["season", "player_id"]), "prior_points", ["season"], 24
    )
    selected = pd.MultiIndex.from_frame(players[["season", "player_id"]])
    masks = {"elite_top24": pd.MultiIndex.from_frame(df[["season", "player_id"]]).isin(selected)}
    masks["weekly_reference_top24"], _ = reference_selection(
        position, df, load_reference(cache_dir=context.raw_root), 24
    )
    for name, mask in masks.items():
        block = blocks[name]
        if block["status"] != "available":
            continue
        subset = df[mask]
        if len(subset) != block["n"] or _identity(subset) != block["cohort_hash"]:
            raise ValueError(f"Full-precision evaluation changed canonical {name} membership")
        for family in FAMILIES:
            error = subset[f"pred_{family}_total"] - subset.fantasy_points
            error = error[np.isfinite(error)].to_numpy(dtype=float)
            if not len(error):
                raise ValueError(f"No finite protected-cohort errors for {family}")
            block["models"][family] = {
                "n": len(error),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
            }
    return blocks


def inference_parity(result, position):
    from src.analysis.artifact_eval import build_test_df_from_artifacts

    context = current_context()
    frames = STATE["frames"]
    if position in {"K", "DST"}:
        module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
        # The native historical provider deliberately deferred these fills.
        # Replay the same fold-local values, never the production 2023 median.
        frames = tuple(
            module.impute_context_from_train(f, fit_on=result.prepared.train) for f in frames
        )
    replay = build_test_df_from_artifacts(
        position, *frames, model_dir=str(context.output_dir(position) / "models")
    )
    if replay.attrs.get("prediction_errors"):
        raise ValueError(f"Saved inference errors: {replay.attrs['prediction_errors']}")
    actual = result["test_df"].copy()
    for family, predictions in result["per_target_preds"].items():
        if predictions is not None:
            for target, values in predictions.items():
                actual[f"pred_{family}_{target}"] = values
    actual = actual.sort_values(KEYS).reset_index(drop=True)
    replay = replay.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[KEYS], replay[KEYS], check_dtype=False)
    errors = {}
    for family in FAMILIES:
        for target in (*result.recipe["targets"], "total"):
            column = f"pred_{family}_{target}"
            a, b = actual[column].to_numpy(), replay[column].to_numpy()
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5, err_msg=column)
            errors[column] = float(np.max(np.abs(a - b)))
    return errors


def metric_fn(result, position):
    import torch

    from src.shared import training
    from src.tuning.ab_harness import default_metric_fn

    context = current_context()
    if context is None or position != STATE["position"]:
        raise ValueError("Repair result does not match its execution context")
    trainers = STATE["trainers"]
    if {t["family"] for t in trainers} != {"nn", "attn_nn"}:
        raise ValueError("Both production neural trainers must be observed")
    if any(
        t["amp"]
        or not t["device"].startswith("cuda")
        or (not t["graph"] and not (position == "K" and t["family"] == "attn_nn"))
        for t in trainers
    ):
        raise ValueError("Repair evidence did not execute production FP32 CUDA graph policies")
    frame = regular_season_rows(result["test_df"]).copy()
    if set(frame.season.unique()) != {STATE["origin"]} or frame.duplicated(KEYS).any():
        raise ValueError("Wrong origin or duplicate player-weeks")
    for family, predictions in result["per_target_preds"].items():
        if predictions is not None:
            for target, values in predictions.items():
                frame[f"pred_{family}_{target}"] = values
    frame = comparison_model_totals(frame, position)
    frame["comparison_actual"] = comparison_actuals(frame, position)
    metrics = default_metric_fn(result, position)
    for family in FAMILIES:
        if not np.isfinite(frame[f"pred_{family}_total"]).all():
            raise ValueError(
                f"Nonfinite {family} predictions cannot be excluded to improve metrics"
            )
        error = frame[f"pred_{family}_total"] - frame.comparison_actual
        valid = np.isfinite(error)
        if not valid.any():
            raise ValueError("No shared-component observations")
        metrics[f"all:{family}"] = {
            "mae": float(error[valid].abs().mean()),
            "rmse": float(np.sqrt(np.square(error[valid]).mean())),
            "n": int(valid.sum()),
        }
    cohorts = precise_cohorts(result, result["test_df"], position)
    if cohorts["elite_top24"]["status"] != "available":
        raise ValueError("Prior-season important-player cohort is unavailable")
    for name in ("elite_top24", "weekly_reference_top24", "week1", "inheritor"):
        block = cohorts.get(name, {})
        for family, values in block.get("models", {}).items():
            metrics[f"{name}:{family}"] = values
    rows = frame.sort_values(KEYS).reset_index(drop=True)
    metadata = {
        "schema": "isolated-audit-development/v1",
        "baseline_sha": BASELINE_SHA,
        "candidate_pins": CANDIDATE_PINS,
        "promotion_eligible": False,
        "probability_mean": "unchanged main inference mean",
        "position": position,
        "origin": STATE["origin"],
        "variant": STATE["arm"],
        "mode": "isolated_development",
        "count_likelihood_implementation": f"{training.ztnb2_log_prob.__module__}.{training.ztnb2_log_prob.__name__}",
        "seed": context.seed,
        "source_sha": os.environ.get("FF_TRAIN_GIT_SHA"),
        "data_release": os.environ.get("FF_DATA_RELEASE"),
        "batch_job_id": os.environ["AWS_BATCH_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "prepared_hashes": prepared_hashes(result.prepared),
        "execution": result.get("execution"),
        "cohorts": cohorts,
        "metrics": metrics,
        "trainers": STATE["trainers"],
        "rows": evidence("predictions.parquet", rows.to_parquet(index=False)),
        "inference_parity_passed": True,
        "inference_parity_max_absolute_error": inference_parity(result, position),
    }
    metadata["row_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(rows[KEYS], index=False).values.tobytes()
    ).hexdigest()
    metadata["truth_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(rows[[*KEYS, "comparison_actual"]], index=False).values.tobytes()
    ).hexdigest()
    if position in {"WR", "TE"}:
        from src.tuning.audit_input_observer import result_proof

        metadata["input_proof"] = result_proof(result.prepared, STATE)
    receipt = evidence("manifest.json", canonical_json(metadata).encode())
    print("[repair-evidence] " + json.dumps(receipt), flush=True)
    metrics["repair"] = {
        "origin": STATE["origin"],
        "trainer_count": len(STATE["trainers"]),
        "weekly_reference_available": int(
            cohorts["weekly_reference_top24"]["status"] == "available"
        ),
    }
    return metrics
