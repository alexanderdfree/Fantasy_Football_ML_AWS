"""Replay synthetic history cohorts against saved checkpoints; record the responses.

A response is a model output on a constructed history. No observed outcome
exists for it, so it is never forecast accuracy and never a training label
(ADR-0029). The attention NN is the only family whose inputs a cohort fully
determines: its static branch is the real forecast game's non-temporal context
and its history branch is the synthetic tensor. Ridge, the base NN and
LightGBM read windowed features the generator does not reconstruct, so they
replay exact identity cohorts only and are otherwise recorded as excluded.
LightGBM must be exercised on Linux/Batch: loading it beside torch and
scikit-learn crashes on the maintainers' macOS libomp stack, so ``all`` skips
it there unless named explicitly.

Replay shares production's code path, not its batch: serving predicts the
whole prepared frame in one batch while a replay predicts the cohort, so the
identity control compares predictions within a float tolerance and records the
largest difference; the inputs themselves must match exactly.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.artifact_eval import resolve_model_dir, warn_if_sync_noop
from src.analysis.synthetic_history import (
    SCHEMA_VERSION,
    code_hashes,
    consume_source,
    model_input_readiness,
    publish_artifact_dir,
    runtime_versions,
)
from src.analysis.synthetic_history_schema import POSITION_HISTORY_SCHEMAS, position_schema
from src.prediction.bundle import MODEL_FAMILIES, bundled_families, digest, file_digest
from src.prediction.frames import SCORING_FORMATS
from src.prediction.predictor import PredictionInputs, Predictor
from src.shared.artifact_integrity import compute_feature_cols_hash
from src.shared.registry import INFERENCE_REGISTRY

REPLAY_SCHEMA_VERSION = 1
RESPONSE_SEMANTICS = (
    "recorded model responses to synthetic histories; no observed outcome exists; never "
    "report as forecast accuracy or use as training labels (ADR-0029)"
)
PREDICTION_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
REQUIRED_MANIFEST_KEYS = (
    "files",
    "recipe",
    "recipe_sha256",
    "sampling_identity_sha256",
    "source_values_sha256",
    "history_columns",
    "history_kind",
    "model_input_readiness",
)
REQUIRED_RECIPE_KEYS = ("name", "position", "mode", "history_games", "donor_seasons")
COHORT_FILES = ("cases.parquet", "context.parquet", "history.npz")
PROVENANCE_KEYS = ("data_id", "dataset_id", "image_sha", "code_id")
LEGACY_OPTION_KEYS = ("attn_max_seq_len", "opp_attn_max_seq_len", "opp_attn_kind")
MACOS_LGBM_REASON = "LightGBM is not loaded beside torch on macOS (libomp); name it explicitly"
CODE_PATHS = (
    "analysis/synthetic_replay.py",
    "analysis/synthetic_history.py",
    "analysis/synthetic_history_schema.py",
    "prediction/predictor.py",
    "prediction/bundle.py",
    "features/engineer.py",
    "shared/feature_build.py",
    "shared/aggregate_targets.py",
)


@dataclass(frozen=True)
class LoadedCohort:
    directory: Path
    manifest: dict
    manifest_sha256: str
    recipe: dict
    cases: pd.DataFrame
    context: pd.DataFrame
    arrays: dict[str, np.ndarray]
    transformed: bool


def load_cohort(directory: Path) -> LoadedCohort:
    """Read a schema-3 cohort after verifying its manifest shape and every file hash."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cohort schema_version {SCHEMA_VERSION} required; regenerate the cohort")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"cohort manifest is missing {missing}")
    recipe = manifest["recipe"]
    missing = [key for key in REQUIRED_RECIPE_KEYS if key not in recipe]
    if missing:
        raise ValueError(f"cohort recipe is missing {missing}")
    if recipe["position"] not in POSITION_HISTORY_SCHEMAS:
        raise ValueError(f"unsupported cohort position {recipe['position']!r}")
    unlisted = sorted(set(COHORT_FILES) - set(manifest["files"]))
    if unlisted:
        raise ValueError(f"cohort manifest does not list {unlisted}")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"cohort artifact mismatch: {name}")
    cases = pd.read_parquet(directory / "cases.parquet")
    context = pd.read_parquet(directory / "context.parquet")
    with np.load(directory / "history.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if not {"history", "mask"} <= set(arrays) or "exact_window" not in cases:
        raise ValueError("cohort arrays or cases predate schema 3; regenerate the cohort")
    if not cases["case_id"].equals(context["case_id"]) or len(cases) != len(arrays["history"]):
        raise ValueError("cohort cases, context and history disagree")
    # Readiness and the history kind follow from the recipe and the cases, never a
    # trusted manifest field.
    transformed = bool(recipe.get("transforms"))
    if manifest["history_kind"] != ("transformed" if transformed else "donor"):
        raise ValueError("cohort manifest history_kind disagrees with its recipe")
    declared = {f: e["ready"] for f, e in manifest["model_input_readiness"].items()}
    derived = model_input_readiness(recipe["mode"], cases, transformed=transformed)
    if declared != {f: e["ready"] for f, e in derived.items()}:
        raise ValueError("cohort manifest readiness disagrees with its recipe and cases")
    return LoadedCohort(
        directory,
        manifest,
        file_digest(manifest_path),
        recipe,
        cases,
        context,
        arrays,
        transformed,
    )


def requested_families(names: list[str], model_dir: str) -> tuple[list[str], dict[str, str]]:
    """Resolve ``--families``; ``all`` expands to the bundled families of the directory."""
    names = list(names)
    excluded: dict[str, str] = {}
    if names == ["all"]:
        inventory = bundled_families(model_dir)
        if inventory is None:
            raise ValueError("name families explicitly for a legacy artifact directory")
        names = list(inventory)
        if "lgbm" in names and platform.system() == "Darwin":
            names.remove("lgbm")
            excluded["lgbm"] = MACOS_LGBM_REASON
    unknown = sorted(set(names) - set(MODEL_FAMILIES))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    return [family for family in MODEL_FAMILIES if family in names], excluded


def prediction_inputs(predictor: Predictor, cohort: LoadedCohort) -> PredictionInputs:
    """Hand-build the checkpoint's ordered inputs from context.parquet and history.npz."""
    schema = predictor.schema
    if schema.structure != "flat" or schema.opponent_history:
        raise ValueError(
            f"{predictor.family} checkpoint needs a {schema.structure} history"
            + (" with an opponent stream" if schema.opponent_history else "")
            + "; schema 3 cohorts carry a flat player history only"
        )
    missing = [column for column in schema.features if column not in cohort.context.columns]
    if missing:
        raise ValueError(
            f"context.parquet is missing features required by {predictor.family}: {missing}"
        )
    values = cohort.context[list(schema.features)].to_numpy(dtype=np.float32)
    if predictor.family != "attn_nn":
        return PredictionInputs(schema, values)
    if list(cohort.manifest["history_columns"]) != list(schema.history):
        raise ValueError("cohort history columns differ from the checkpoint's ordered history")
    history, mask = cohort.arrays["history"], cohort.arrays["mask"]
    window = predictor.zero_inputs().history.shape[1]
    if history.shape[1] != window:
        raise ValueError(f"cohort history length {history.shape[1]} differs from window {window}")
    return PredictionInputs(schema, values, history, mask)


def _loaded_files(root: Path, predictor: Predictor) -> list[Path]:
    """The files the loader actually read for a legacy (bundle-less) family."""
    family, position = predictor.family, predictor.position.lower()
    if family == "ridge":
        paths = [p for t in predictor.schema.targets for p in (root / t).rglob("*") if p.is_file()]
        optional = ("non_negative_targets.json", "ridge_selection.json")
        paths += [root / name for name in optional if (root / name).is_file()]
    elif family == "lgbm":
        paths = [p for p in (root / "lightgbm").rglob("*") if p.is_file()]
    else:
        stem = "attention_nn" if family == "attn_nn" else "nn"
        weights = "attention_nn" if family == "attn_nn" else "multihead_nn"
        paths = [root / f"{position}_{weights}.pt", root / f"{stem}_scaler.pkl"]
        if (root / f"{stem}_scaler_meta.json").is_file():
            paths.append(root / f"{stem}_scaler_meta.json")
    return sorted(paths)


def family_identity(predictor: Predictor, model_dir: str) -> dict:
    schema = predictor.schema
    identity = {
        "feature_cols_hash": compute_feature_cols_hash(list(schema.features)),
        "features": list(schema.features),
        "history": list(schema.history),
        "targets": list(schema.targets),
    }
    if predictor.bundle is not None:
        document = predictor.bundle.to_dict()
        identity.update(
            bundle_id=predictor.bundle.bundle_id,
            provenance=document["provenance"],
            preprocessing_sha256=digest(document["preprocessing"]),
            preparation_sha256=digest(document["preparation"]),
            history_options=document["history_options"],
            files=document["files"],
        )
        return identity
    root = Path(model_dir)
    identity.update(
        bundle_id=None,
        provenance=None,
        preprocessing_sha256=None,
        preparation_sha256=None,
        history_options={
            k: predictor.options[k] for k in LEGACY_OPTION_KEYS if k in predictor.options
        },
        files={str(p.relative_to(root)): file_digest(p) for p in _loaded_files(root, predictor)},
    )
    return identity


def assert_coherent_families(identities: dict[str, dict]) -> None:
    """Bundled families must share one training generation, as serving requires."""
    bundled = {f: i for f, i in identities.items() if i["bundle_id"] is not None}
    if len(bundled) < 2:
        return
    reference_family, reference = next(iter(bundled.items()))
    for family, identity in bundled.items():
        mismatched = [
            key
            for key in PROVENANCE_KEYS
            if identity["provenance"].get(key) != reference["provenance"].get(key)
        ]
        for key in ("targets", "preprocessing_sha256", "preparation_sha256"):
            if identity[key] != reference[key]:
                mismatched.append(key)
        if mismatched:
            raise ValueError(
                f"model families {reference_family} and {family} come from different "
                f"training generations ({', '.join(mismatched)} differ)"
            )


def identity_control(
    cohort: LoadedCohort,
    predictors: dict[str, Predictor],
    inputs: dict[str, PredictionInputs],
    raw: dict[str, dict[str, np.ndarray]],
    source: pd.DataFrame,
    *,
    source_file_sha256: str | None = None,
) -> dict:
    """Prove the hand-built inputs reproduce production on the real calendar.

    Static values must equal the production tensor builder's for every case in
    every mode. For untransformed replay cohorts the newest-first history
    prefix and the mask must match too, and predictions on exact windows (the
    forecast game is the (N+1)th of the season) must match production's
    whole-frame predictions within ``PREDICTION_TOLERANCE``; truncated windows
    only count. Transformed histories are fixtures, so only their context is
    compared.
    """
    recipe = cohort.recipe
    schema = position_schema(recipe["position"])
    consumed = consume_source(
        source, position=recipe["position"], donor_seasons=recipe["donor_seasons"], schema=schema
    )
    if consumed.values_sha256 != cohort.manifest["source_values_sha256"]:
        message = "source values differ from the cohort's consumed source"
        if source_file_sha256 is not None and source_file_sha256 == cohort.manifest.get(
            "source_file_sha256"
        ):
            message += "; the file is the same, so the consuming code or configuration changed"
        raise ValueError(message)
    frame = consumed.frame
    reference_frame = consumed.source.loc[frame.index]
    keyed = {
        key: row
        for row, key in enumerate(
            zip(frame["player_id"], frame["season"], frame["week"], strict=True)
        )
    }
    cases = cohort.cases
    rows = np.array(
        [
            keyed[(player, int(season), int(week))]
            for player, season, week in zip(
                cases["donor_player_id"], cases["donor_season"], cases["forecast_week"], strict=True
            )
        ]
    )
    n = int(recipe["history_games"])
    identity_mode = recipe["mode"] == "replay" and not cohort.transformed
    exact_cases = cases["exact_window"].to_numpy(dtype=bool)
    exact = exact_cases if identity_mode else np.zeros(len(rows), bool)
    result = {"status": None, "prediction_tolerance": PREDICTION_TOLERANCE, "families": {}}
    for family, predictor in predictors.items():
        reference = predictor.inputs_from_frame(reference_frame)
        built = inputs[family]
        if not np.array_equal(built.values, reference.values[rows]):
            raise ValueError(f"identity control failed for {family}: static_values")
        checks = ["static_values"]
        compared = np.ones(len(rows), dtype=bool) if identity_mode else exact
        if family == "attn_nn":
            compared = exact
            if identity_mode:
                if not np.array_equal(built.history[:, :n], reference.history[rows, :n]):
                    raise ValueError(f"identity control failed for {family}: history_prefix")
                mask_ok = (
                    built.history_mask[:, :n].all()
                    and not built.history_mask[:, n:].any()
                    and reference.history_mask[rows, :n].all()
                )
                if not mask_ok:
                    raise ValueError(f"identity control failed for {family}: mask")
                checks += ["history_prefix", "mask"]
        delta = None
        if compared.any():
            production = predictor.predict_raw(reference)
            delta = 0.0
            for target in predictor.schema.targets:
                replayed = raw[family][target][compared]
                expected = production[target][rows][compared]
                delta = max(delta, float(np.max(np.abs(replayed - expected))))
                if not np.allclose(replayed, expected, **PREDICTION_TOLERANCE):
                    raise ValueError(
                        f"identity control failed for {family}: predictions differ from "
                        f"production by up to {delta:.3g} on {target}"
                    )
            checks.append("predictions_on_exact_windows")
        result["families"][family] = {
            "cases": int(len(rows)),
            "exact_window_cases": int(exact_cases.sum()),
            "compared_predictions": int(compared.sum()),
            "max_abs_prediction_delta": delta,
            "checks": checks,
        }
    if not identity_mode:
        result["status"] = "context_only"
    elif all(entry["compared_predictions"] > 0 for entry in result["families"].values()):
        result["status"] = "passed"
    else:
        result["status"] = "inputs_only"
    result["source_file_sha256"] = source_file_sha256
    return result


def replay_cohort(
    cohort: LoadedCohort,
    model_dir: str,
    families: list[str],
    *,
    source: pd.DataFrame | None = None,
    source_file_sha256: str | None = None,
    families_requested: list[str] | None = None,
    excluded: dict[str, str] | None = None,
    sync: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    position = cohort.recipe["position"]
    spec = INFERENCE_REGISTRY[position]
    readiness = cohort.manifest["model_input_readiness"]
    predictors, inputs, raw, totals, identities = {}, {}, {}, {}, {}
    excluded = dict(excluded or {})
    for family in families:
        if not readiness[family]["ready"]:
            excluded[family] = readiness[family]["reason"]
            continue
        predictor = Predictor.from_bundle(
            model_dir, family, position=position, legacy_spec=spec, device=torch.device("cpu")
        )
        identities[family] = family_identity(predictor, model_dir)
        family_inputs = prediction_inputs(predictor, cohort)
        predictions = predictor.predict_raw(family_inputs)
        for target in predictor.schema.targets:
            if not np.isfinite(predictions[target]).all():
                raise ValueError(f"non-finite responses from {family}")
        predictors[family], inputs[family], raw[family] = predictor, family_inputs, predictions
        totals[family] = {fmt: predictor.score(predictions, fmt) for fmt in SCORING_FORMATS}
    if not predictors:
        raise ValueError(f"no requested family is replayable for this cohort: {excluded}")
    assert_coherent_families(identities)
    if source is not None:
        control = identity_control(
            cohort, predictors, inputs, raw, source, source_file_sha256=source_file_sha256
        )
    else:
        control = {"status": "skipped", "reason": "no --source", "source_file_sha256": None}
    frame = cohort.cases.copy()
    for family, predictor in predictors.items():
        for target in predictor.schema.targets:
            frame[f"pred_{family}_{target}"] = raw[family][target]
        frame[f"pred_{family}_total"] = totals[family]["ppr"]
        for fmt in SCORING_FORMATS:
            if fmt != "ppr":
                frame[f"pred_{family}_total_{fmt}"] = totals[family][fmt]
    manifest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "cohort_dir": str(cohort.directory),
        "cohort_name": cohort.recipe["name"],
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "cohort_files": dict(cohort.manifest["files"]),
        "recipe": dict(cohort.recipe),
        "recipe_sha256": cohort.manifest["recipe_sha256"],
        "sampling_identity_sha256": cohort.manifest["sampling_identity_sha256"],
        "source_values_sha256": cohort.manifest["source_values_sha256"],
        "history_kind": cohort.manifest["history_kind"],
        "fixture": bool(cohort.manifest.get("fixture", False)),
        "position": position,
        "model_dir": str(model_dir),
        "sync": sync,
        "families_requested": list(families_requested or families),
        "families": identities,
        "bundle_ids": {family: identity["bundle_id"] for family, identity in identities.items()},
        "families_excluded": excluded,
        "identity_control": control,
        "response_semantics": RESPONSE_SEMANTICS,
        "scoring": {"total_column_format": "ppr", "formats": list(SCORING_FORMATS)},
        "prediction_columns": [c for c in frame.columns if c.startswith("pred_")],
        "device": "cpu",
        "versions": runtime_versions("torch", "scikit-learn"),
        "code_sha256": code_hashes(CODE_PATHS),
    }
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, required=True, help="Cohort artifact directory (schema_version 3)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New replay directory (never overwritten)"
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["attn_nn"],
        help="Model families to replay, or 'all' for every bundled family (default: attn_nn)",
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="Artifact directory override")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Pull served artifacts from S3 first (FF_MODEL_S3_BUCKET)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="The consumed source parquet; runs the identity control",
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        cohort = load_cohort(args.cohort)
        position = cohort.recipe["position"]
        sync = None
        if args.sync:
            warn_if_sync_noop()
            from src.artifacts.model_sync import sync_models_from_s3

            summary = sync_models_from_s3()
            sync = {"requested": True, "summary": json.loads(json.dumps(summary, default=str))}
        model_dir = resolve_model_dir(
            position, INFERENCE_REGISTRY[position], str(args.model_dir) if args.model_dir else None
        )
        families, excluded = requested_families(args.families, model_dir)
        source = pd.read_parquet(args.source) if args.source is not None else None
        predictions, manifest = replay_cohort(
            cohort,
            model_dir,
            families,
            source=source,
            source_file_sha256=file_digest(args.source) if args.source is not None else None,
            families_requested=list(args.families),
            excluded=excluded,
            sync=sync,
        )
        output = publish_artifact_dir(
            args.output,
            lambda directory: predictions.to_parquet(
                directory / "predictions.parquet", index=False
            ),
            manifest,
            manifest_name="replay_manifest.json",
        )
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"synthetic-replay: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "position": position,
                "cases": len(predictions),
                "history_kind": manifest["history_kind"],
                "families": sorted(manifest["families"]),
                "families_excluded": manifest["families_excluded"],
                "identity_control": manifest["identity_control"]["status"],
                "model_dir": manifest["model_dir"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
