"""Position frame preparation and prediction, independent of HTTP state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import MIN_GAMES_PER_SEASON
from src.prediction.bundle import bundled_families, canonical_json
from src.prediction.predictor import Predictor
from src.shared.aggregate_targets import TARGET_UNITS
from src.shared.feature_build import build_position_features

SCORING_FORMATS = ("ppr", "half_ppr", "standard")


@dataclass(frozen=True)
class PositionPredictions:
    frame: pd.DataFrame
    raw: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, np.ndarray]]
    errors: dict[str, str]
    details: dict
    bundle_ids: dict[str, str]


def prepare_position_frame(position, train, val, test, spec, *, fitted_state=None):
    train = spec["filter_fn"](train)
    val = spec["filter_fn"](val)
    test = spec["filter_fn"](test)
    if position not in {"K", "DST"}:
        train = spec["compute_targets_fn"](train)
        val = spec["compute_targets_fn"](val)
        test = spec["compute_targets_fn"](test)
    full_train = train
    minimum = spec.get("min_games_per_season")
    if minimum is None:
        minimum = MIN_GAMES_PER_SEASON
    count = train.groupby(["player_id", "season"])["week"].transform("count")
    train = train[count >= minimum].copy()
    if "_practice_status_missing" in test:
        missing = test["_practice_status_missing"].eq(True)
        if missing.any():
            mean = (
                fitted_state.get("practice_status_mean")
                if fitted_state is not None
                else train["practice_status"].mean()
            )
            if pd.isna(mean):
                raise ValueError("Training data has no practice status for live imputation")
            test.loc[missing, "practice_status"] = mean
    columns = list(spec["get_feature_columns_fn"]())
    replay = {"fitted_state": fitted_state} if fitted_state is not None else {}
    train, val, test = build_position_features(
        train, val, test, spec, columns, full_train=full_train, **replay
    )
    return train, val, test, columns


def _details(frame, raw, totals, feature_count, targets):
    target_metrics = {}
    for target in targets:
        if target in frame:
            actual = frame[target].to_numpy()
            target_metrics[target] = {
                **{
                    f"{family}_mae": round(float(np.mean(np.abs(predictions[target] - actual))), 3)
                    for family, predictions in raw.items()
                    if target in predictions
                },
                "unit": TARGET_UNITS.get(target, ""),
            }
    total_metrics = {}
    for scoring in SCORING_FORMATS:
        column = "fantasy_points" if scoring == "ppr" else f"fantasy_points_{scoring}"
        if column not in frame:
            column = "fantasy_points"
        if column in frame:
            actual = frame[column].to_numpy()
            total_metrics[scoring] = {
                f"{family}_mae": round(float(np.mean(np.abs(predictions[scoring] - actual))), 3)
                for family, predictions in totals.items()
            }
    target_metrics["total"] = total_metrics.get("ppr", {})
    target_metrics["total_by_format"] = total_metrics
    return {
        "n_features": feature_count,
        "n_samples_test": len(frame),
        "target_metrics": target_metrics,
    }


def predict_position(
    position, train, val, test, spec, *, kicks=None, opponent_weekly=None, device=None
):
    """Apply all supported families, preserving explicit per-family failures."""
    loaded = {}
    errors = {}
    inventory = bundled_families(spec["model_dir"])
    for family in inventory if inventory is not None else ("ridge", "nn", "attn_nn", "lgbm"):
        if (
            family == "attn_nn"
            and not (spec.get("train_attention_nn") and spec.get("attn_nn_file"))
            and inventory is None
        ):
            continue
        if family == "lgbm" and not spec.get("train_lightgbm") and inventory is None:
            continue
        try:
            loaded[family] = Predictor.from_bundle(
                spec["model_dir"], family, position=position, legacy_spec=spec, device=device
            )
        except Exception as exc:
            errors[f"{position}_{family}"] = repr(exc)
    # Bundled models share fitted preprocessing; reject an accidental mixture.
    bundles = {
        family: predictor.bundle
        for family, predictor in loaded.items()
        if predictor.bundle is not None
    }
    fitted_state = None
    preparation = dict(spec)
    if bundles:
        reference = next(iter(bundles.values())).to_dict()
        fitted_state = reference["preprocessing"]
        prep = reference["preparation"]
        preparation.update({k: v for k, v in prep.items() if k != "features"})
        preparation["targets"] = list(reference["inputs"]["targets"])
        preparation["get_feature_columns_fn"] = lambda: list(prep["features"])
        for family, bundle in bundles.items():
            document = bundle.to_dict()
            provenance_matches = all(
                document["provenance"].get(key) == reference["provenance"].get(key)
                for key in ("data_id", "dataset_id", "image_sha", "code_id")
            )
            if (
                not provenance_matches
                or document["inputs"]["targets"] != reference["inputs"]["targets"]
                or canonical_json(document["preprocessing"]) != canonical_json(fitted_state)
                or canonical_json(document["preparation"]) != canonical_json(prep)
            ):
                errors[f"{position}_{family}"] = (
                    "Model families have incompatible fitted preprocessing"
                )
                loaded.pop(family)
    _, _, prepared, features = prepare_position_frame(
        position, train, val, test, preparation, fitted_state=fitted_state
    )
    if opponent_weekly is None and spec.get("opp_attn_kind", "defense") == "defense":
        opponent_weekly = pd.concat([train, val, test], ignore_index=True)
    raw = {}
    totals = {}
    identities = {}
    for family, predictor in loaded.items():
        try:
            inputs = predictor.inputs_from_frame(
                prepared, kicks=kicks, opponent_weekly=opponent_weekly
            )
            predictions = predictor.predict_raw(inputs)
            adjustment_fn = spec.get("compute_adjustment_fn") if predictor.bundle is None else None
            adjustment = adjustment_fn(prepared).to_numpy() if adjustment_fn is not None else None
            signs = spec.get("target_signs") if predictor.bundle is None else None
            if signs is not None or adjustment is not None:
                value = sum(
                    predictions[t] * (signs or {}).get(t, 1.0) for t in predictor.schema.targets
                )
                if adjustment is not None:
                    value = value + adjustment
                scored = {fmt: value for fmt in SCORING_FORMATS}
            else:
                scored = {fmt: predictor.score(predictions, fmt) for fmt in SCORING_FORMATS}
            raw[family] = predictions
            totals[family] = scored
            if predictor.bundle is not None:
                identities[family] = predictor.bundle.bundle_id
        except Exception as exc:
            errors[f"{position}_{family}"] = repr(exc)
    return PositionPredictions(
        prepared,
        raw,
        totals,
        errors,
        _details(prepared, raw, totals, len(features), preparation["targets"]),
        identities,
    )
