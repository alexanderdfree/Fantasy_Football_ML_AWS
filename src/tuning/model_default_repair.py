"""AWS-only historical repair experiments using the existing A/B harness.

The production defaults are never mutated by importing this module. Observers
are installed only by an explicitly selected experiment's config mutator.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import tempfile
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.bundle import canonical_json
from src.shared.comparison_scoring import comparison_actuals, comparison_model_totals
from src.shared.evaluation_cohorts import regular_season_rows
from src.training.context import current_context
from src.tuning.repair_selection import SelectionTrace

KEYS = ["player_id", "season", "week"]
FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")
STATE = {}


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


def install_native_origins():
    """Re-slice native provider outputs, retaining K's bound kick histories."""
    from src.training.contracts import DatasetSplits

    for position in ("K", "DST"):
        module = importlib.import_module(f"src.{position.lower()}.run_pipeline")
        current = module.provide_dataset
        original = getattr(current, "_repair_original", current)

        @wraps(original)
        def provider(cfg, *, cross_validation=False, _original=original, _position=position):
            if cross_validation:
                raise ValueError(
                    "Repair origins require chronological train/validation/test splits"
                )
            if _position == "K":
                from unittest.mock import patch

                from src.k import data as kicker_data

                # load_data fits Vegas fill values before season_split. Set its
                # training ceiling before invoking the native provider, rather
                # than re-slicing values already imputed using later seasons.
                with patch.object(kicker_data, "_TRAIN_MAX_SEASON", STATE["origin"] - 2):
                    dataset = _original(cfg)
            else:
                dataset = _original(cfg)
            frames = split_origin(dataset.frames, STATE["origin"], position=_position)
            STATE["frames"] = frames
            return DatasetSplits(*frames, dataset.bindings)

        provider._repair_original = original
        module.provide_dataset = provider


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


def install_observer():
    from src.shared.training import MultiHeadTrainer

    original = getattr(MultiHeadTrainer.train, "_repair_original", MultiHeadTrainer.train)

    @wraps(original)
    def train(trainer, train_loader, val_loader, n_epochs):
        import torch

        from src.analysis.repair_count_diagnostics import error_summary, observed_likelihood_check

        family = "nn" if type(trainer) is MultiHeadTrainer else "attn_nn"
        selection = STATE["mode"] == "selection" and family == "attn_nn"
        trace = SelectionTrace(trainer.patience)
        saved_patience = trainer.patience
        saved_observer = getattr(trainer, "epoch_observer", None)
        if saved_observer is not None or trainer.epoch_callback is not None:
            raise ValueError("Repair diagnostics cannot compose with another trainer callback")
        with tempfile.TemporaryDirectory(prefix="repair-checkpoints-") as directory:
            checkpoint_dir = Path(directory)

            def observe(epoch, history, predictions, truth):
                trace.observe(
                    epoch + 1,
                    history["val_mae_weighted"][-1],
                    history["val_fantasy_mae_ppr"][-1],
                    history["val_fantasy_rmse_ppr"][-1],
                )
                torch.save(
                    {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
                    checkpoint_dir / f"{epoch + 1}.pt",
                )

            if selection:
                trainer.epoch_observer = observe
                trainer.patience = n_epochs + 1
            try:
                history = original(trainer, train_loader, val_loader, n_epochs)
            finally:
                trainer.patience = saved_patience
                trainer.epoch_observer = saved_observer
            report = {
                "family": family,
                "device": str(trainer.device),
                "amp": bool(trainer._use_amp),
                "graph": trainer._graphed_step is not None,
            }
            if selection:
                policy = trace.finish()
                policy["restored"] = {}
                states = {}
                for name in ("legacy", "rmse", "guarded"):
                    selected = policy[name]
                    if selected is None:
                        continue
                    path = checkpoint_dir / f"{selected['epoch']}.pt"
                    trainer.model.load_state_dict(
                        torch.load(path, map_location=trainer.device, weights_only=True)
                    )
                    states[name] = torch.load(path, map_location="cpu", weights_only=True)
                    predictions, truth = _validation_predictions(trainer, val_loader)
                    rescored = _checkpoint_metrics(predictions, truth, STATE["position"])
                    for metric in ("mae", "rmse"):
                        if not np.isclose(rescored[metric], selected[metric], rtol=2e-5, atol=2e-6):
                            raise ValueError(
                                f"Restored {name} checkpoint does not reproduce {metric}"
                            )
                    policy["restored"][name] = {
                        **rescored,
                        "checkpoint": evidence(f"{family}-{name}.pt", path.read_bytes()),
                    }
                # The diagnostic returns the true original legacy checkpoint.
                anchor = policy["legacy"]
                trainer.model.load_state_dict(
                    torch.load(
                        checkpoint_dir / f"{anchor['epoch']}.pt",
                        map_location=trainer.device,
                        weights_only=True,
                    )
                )
                trainer.best_epoch = anchor["epoch"]
                trainer.best_val_metric = anchor["weighted_mae"]
                trainer.best_model_state = {
                    k: v.clone() for k, v in trainer.model.state_dict().items()
                }
                history["checkpoint_selection"].update(
                    metric="weighted_mae",
                    epoch=anchor["epoch"],
                    score=anchor["weighted_mae"],
                    validation_metrics={
                        key: values[anchor["epoch"] - 1]
                        for key, values in history.items()
                        if key.startswith("val_") and isinstance(values, list)
                    },
                )
                report["policies"] = policy
                original_predict = trainer.model.predict_numpy

                @wraps(original_predict)
                def predict_alternatives(*args, **kwargs):
                    devices = [torch.cuda.current_device()] if trainer.device.type == "cuda" else []
                    with torch.random.fork_rng(devices=devices):
                        predictions = {}
                        try:
                            for name, state in states.items():
                                trainer.model.load_state_dict(state)
                                predictions[name] = original_predict(*args, **kwargs)
                        finally:
                            trainer.model.load_state_dict(states["legacy"])
                        STATE["alternative_predictions"].append(predictions)
                        return predictions["legacy"]

                trainer.model.predict_numpy = predict_alternatives
            predictions, truth = _validation_predictions(trainer, val_loader)
            report["validation_errors"] = error_summary(predictions, truth)
            if "receptions_value_mu" in predictions:
                from src.shared.neural_net import ztnb2_conditional_mean

                predictions["receptions_conditional_mean"] = ztnb2_conditional_mean(
                    torch.tensor(predictions["receptions_value_mu"]),
                    torch.tensor(predictions["receptions_value_log_alpha"]),
                ).numpy()
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
            report["validation_raw"] = evidence(f"{family}-validation.npz", payload.getvalue())
            STATE["trainers"].append(report)
        return history

    train._repair_original = original
    MultiHeadTrainer.train = train


def configure(config, *, arm, mode):
    from src.shared.aggregate_targets import infer_position
    from src.tuning.count_likelihood_repair import select_numerical_candidate

    if not os.environ.get("AWS_BATCH_JOB_ID"):
        raise RuntimeError("All repair training must run on AWS Batch")
    if os.environ.get("FF_AB_STACKED", "").lower() in {"1", "true", "yes"} or int(
        os.environ.get("FF_NN_FIXED_EPOCHS", "0") or 0
    ):
        raise ValueError("Repair evaluation requires production nonstacked early stopping")
    if config.get("scheduler_type") == "plateau" or config.get("attn_scheduler_type") == "plateau":
        raise ValueError("Identical policy trajectories require a selection-independent scheduler")
    origin = int(os.environ.get("FF_REPAIR_ORIGIN", "2022"))
    if origin not in (2022, 2023):
        raise ValueError(
            "Development spec cannot inspect confirmation seasons; freeze a qualifying candidate first"
        )
    if (
        arm == "stable_numeric"
        and os.environ.get("FF_REPAIR_NUMERICAL_SCREEN") != "observed_discrepancy_verified"
    ):
        raise ValueError("Numerical repair requires an observed development-range discrepancy")
    select_numerical_candidate(arm == "stable_numeric")
    position = infer_position(config["targets"])
    STATE.clear()
    STATE.update(
        arm=arm,
        mode=mode,
        origin=origin,
        position=position,
        trainers=[],
        alternative_predictions=[],
    )
    corrected = mode == "wr" and arm != "baseline"
    config.update(
        nn_correct_ztnb_mean=corrected,
        nn_poisson_log_rate=corrected,
        nn_magnitude_features=("inherited_opportunity",)
        if corrected and position in {"QB", "WR"}
        else (),
    )
    config.update(
        nn_selection_metric="weighted_mae",
        ridge_selection_metric="raw_mae",
        lgbm_selection_metric="per_target",
    )
    if arm in {"gate_half", "gate_double", "reception_half", "reception_double"}:
        if os.environ.get("FF_REPAIR_WEIGHT_SCREEN") != "observed_numerics_passed":
            raise ValueError(
                "Weight candidates require completed observed-range numerical diagnostics"
            )
        if arm.startswith("gate_"):
            config["attn_gate_weight"] = 0.5 if arm.endswith("half") else 2.0
        else:
            config["loss_weights"] = {
                **config["loss_weights"],
                "receptions": 0.5 if arm.endswith("half") else 2.0,
            }
    install_native_origins()
    install_observer()
    return config


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
    replay = build_test_df_from_artifacts(
        position, *STATE["frames"], model_dir=str(context.output_dir(position) / "models")
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
        "schema": "model-default-repair-evidence/v1",
        "position": position,
        "origin": STATE["origin"],
        "variant": STATE["arm"],
        "mode": STATE["mode"],
        "seed": context.seed,
        "source_sha": os.environ.get("FF_TRAIN_GIT_SHA"),
        "data_release": os.environ.get("FF_DATA_RELEASE"),
        "batch_job_id": os.environ["AWS_BATCH_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "prepared_hashes": prepared_hashes(result.prepared),
        "execution": result.get("execution"),
        "cohorts": cohorts,
        "metrics": metrics,
        "trainers": STATE["trainers"],
        "rows": evidence("predictions.parquet", rows.to_parquet(index=False)),
        "inference_parity_passed": True,
        "inference_parity_max_absolute_error": inference_parity(result, position),
    }
    if STATE["mode"] == "selection":
        from src.shared.aggregate_targets import predictions_to_fantasy_points

        expected = result["per_target_preds"]["attn_nn"]
        matches = [
            call
            for call in STATE["alternative_predictions"]
            if all(
                target in call["legacy"] and np.array_equal(values, call["legacy"][target])
                for target, values in expected.items()
            )
        ]
        if not matches:
            raise ValueError("Alternative checkpoints were not evaluated on the actual test inputs")
        metadata["policy_candidates"] = {}
        for name, predictions in matches[-1].items():
            candidate = result["test_df"].copy()
            for target, values in predictions.items():
                candidate[f"pred_attn_nn_{target}"] = values
            candidate["pred_attn_nn_total"] = predictions_to_fantasy_points(
                position, predictions, "ppr"
            )
            cohorts = precise_cohorts(result, candidate, position)
            shared = comparison_model_totals(candidate, position)
            error = shared.pred_attn_nn_total - comparison_actuals(shared, position)
            error = error[np.isfinite(error)]
            metadata["policy_candidates"][name] = {
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "n": len(error),
                "cohorts": cohorts,
                "rows": evidence(f"{name}-predictions.parquet", candidate.to_parquet(index=False)),
            }
    metadata["row_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(rows[KEYS], index=False).values.tobytes()
    ).hexdigest()
    metadata["truth_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(rows[[*KEYS, "comparison_actual"]], index=False).values.tobytes()
    ).hexdigest()
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
