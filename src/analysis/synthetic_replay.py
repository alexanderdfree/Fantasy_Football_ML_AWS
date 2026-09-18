"""Replay synthetic history cohorts against saved checkpoints; record the responses.

A response is a model output on a constructed history. No observed outcome
exists for it, so it is never forecast accuracy and never a training label
(ADR-0029). The attention NN is the only family whose inputs a cohort fully
determines: its static branch is the real forecast game's non-temporal context
and its history branch is the synthetic tensor. Ridge, the base NN and
LightGBM read windowed features the generator does not reconstruct, so they
replay identity cohorts only and are otherwise recorded as excluded. LightGBM
must be exercised on Linux/Batch: loading it beside torch and scikit-learn
crashes on the maintainers' macOS libomp stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.artifact_eval import resolve_model_dir, warn_if_sync_noop
from src.analysis.synthetic_history import (
    MODEL_FAMILIES,
    SCHEMA_VERSION,
    HistoryRecipe,
    _context_rows,
    _source_frame,
    consumed_values_hash,
    publish_artifact_dir,
)
from src.analysis.synthetic_history_schema import position_schema
from src.prediction import bundle as bundle_module
from src.prediction.bundle import bundled_families, file_digest
from src.prediction.frames import SCORING_FORMATS
from src.prediction.predictor import PredictionInputs, Predictor
from src.shared.artifact_integrity import compute_feature_cols_hash
from src.shared.registry import INFERENCE_REGISTRY

REPLAY_SCHEMA_VERSION = 1
RESPONSE_SEMANTICS = (
    "recorded model responses to synthetic histories; no observed outcome exists; never "
    "report as forecast accuracy or use as training labels (ADR-0029)"
)
HISTORY_OPTION_KEYS = (
    "attn_max_seq_len",
    "attn_max_games",
    "attn_max_kicks_per_game",
    "opp_attn_max_seq_len",
    "opp_attn_kind",
)
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
    cases: pd.DataFrame
    context: pd.DataFrame
    arrays: dict[str, np.ndarray]


def load_cohort(directory: Path) -> LoadedCohort:
    """Read a schema-2 cohort after verifying every published file hash."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cohort schema_version {SCHEMA_VERSION} required; regenerate the cohort")
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"cohort artifact mismatch: {name}")
    cases = pd.read_parquet(directory / "cases.parquet")
    context = pd.read_parquet(directory / "context.parquet")
    with np.load(directory / "history.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if not cases["case_id"].equals(context["case_id"]) or len(cases) != len(arrays["history"]):
        raise ValueError("cohort cases, context and history disagree")
    return LoadedCohort(directory, manifest, file_digest(manifest_path), cases, context, arrays)


def requested_families(names: list[str], model_dir: str) -> list[str]:
    names = list(names)
    if names == ["all"]:
        inventory = bundled_families(model_dir)
        if inventory is None:
            raise ValueError("name families explicitly for a legacy artifact directory")
        names = list(inventory)
    unknown = sorted(set(names) - set(MODEL_FAMILIES))
    if unknown:
        raise ValueError(f"unknown model families: {unknown}")
    return [family for family in MODEL_FAMILIES if family in names]


def prediction_inputs(predictor: Predictor, cohort: LoadedCohort) -> PredictionInputs:
    """Hand-build the checkpoint's ordered inputs from context.parquet and history.npz."""
    schema = predictor.schema
    missing = [column for column in schema.features if column not in cohort.context.columns]
    if missing:
        raise ValueError(
            f"context.parquet is missing features required by {predictor.family}: {missing}"
        )
    # Row-major like the production builder's fancy-indexed batches; a column-major
    # frame view takes a different BLAS path and differs in the last ulp.
    values = np.ascontiguousarray(cohort.context[list(schema.features)].to_numpy(dtype=np.float32))
    if predictor.family != "attn_nn":
        return PredictionInputs(schema, values)
    if list(cohort.manifest["history_columns"]) != list(schema.history):
        raise ValueError("cohort history columns differ from the checkpoint's ordered history")
    history, mask = cohort.arrays["history"], cohort.arrays["mask"]
    window = predictor.options.get("attn_max_seq_len") or 17
    if history.shape[1] != window:
        raise ValueError(f"cohort history length {history.shape[1]} differs from window {window}")
    return PredictionInputs(schema, values, history, mask)


def family_identity(predictor: Predictor, model_dir: str) -> dict:
    schema = predictor.schema
    if predictor.bundle is not None:
        files = predictor.bundle.to_dict()["files"]
        bundle_id = predictor.bundle.bundle_id
    else:
        # The bundle module owns the one per-family file rule; legacy dirs have no bundle.
        root = Path(model_dir)
        paths = bundle_module._model_files(
            root, predictor.position, predictor.family, list(schema.targets)
        )
        files = {str(path.relative_to(root)): file_digest(path) for path in paths}
        bundle_id = None
    return {
        "bundle_id": bundle_id,
        "feature_cols_hash": compute_feature_cols_hash(list(schema.features)),
        "features": list(schema.features),
        "history": list(schema.history),
        "targets": list(schema.targets),
        "history_options": {
            key: predictor.options[key] for key in HISTORY_OPTION_KEYS if key in predictor.options
        },
        "files": files,
    }


def identity_control(
    cohort: LoadedCohort,
    predictors: dict[str, Predictor],
    inputs: dict[str, PredictionInputs],
    raw: dict[str, dict[str, np.ndarray]],
    source: pd.DataFrame,
) -> dict:
    """Prove the hand-built inputs reproduce production on the real calendar.

    Static values, the newest-first history prefix and the mask must match the
    production tensor builder for every case; predictions must match on the
    cases whose forecast game is exactly the (N+1)th of the season. Cases with
    more real history than N are truncated windows: their inputs are checked,
    their predictions legitimately differ and are only counted.
    """
    recipe = HistoryRecipe.from_dict(cohort.manifest["recipe"])
    if recipe.mode != "replay":
        raise ValueError("identity control requires a replay cohort")
    schema = position_schema(recipe.position)
    source = source.reset_index(drop=True)
    frame = _source_frame(source, recipe, schema)
    context_rows = _context_rows(source, frame.index, schema)
    if consumed_values_hash(frame, context_rows) != cohort.manifest["source_values_sha256"]:
        raise ValueError("source values differ from the cohort's consumed source")
    reference_frame = source.loc[frame.index]
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
    n = recipe.history_games
    result = {"status": "passed", "reason": None, "families": {}}
    for family, predictor in predictors.items():
        reference = predictor.inputs_from_frame(reference_frame)
        built = inputs[family]
        checks = {"static_values": np.array_equal(built.values, reference.values[rows])}
        if family == "attn_nn":
            checks["history_prefix"] = np.array_equal(
                built.history[:, :n], reference.history[rows, :n]
            )
            checks["mask"] = bool(
                built.history_mask[:, :n].all()
                and not built.history_mask[:, n:].any()
                and reference.history_mask[rows, :n].all()
            )
            exact = reference.history_mask[rows].sum(axis=1) == n
            reference_inputs = PredictionInputs(
                predictor.schema,
                reference.values[rows],
                reference.history[rows],
                reference.history_mask[rows],
            )
        else:
            exact = np.ones(len(rows), dtype=bool)
            reference_inputs = PredictionInputs(predictor.schema, reference.values[rows])
        reference_raw = predictor.predict_raw(reference_inputs)
        checks["predictions_on_exact_windows"] = all(
            np.array_equal(raw[family][target][exact], reference_raw[target][exact])
            for target in predictor.schema.targets
        )
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"identity control failed for {family}: {', '.join(failed)}")
        result["families"][family] = {
            "cases": int(len(rows)),
            "exact_window_cases": int(exact.sum()),
            "checks": list(checks),
        }
    return result


def _code_hashes() -> dict[str, str]:
    src_root = Path(__file__).resolve().parents[1]
    return {
        f"src/{relative}": hashlib.sha256((src_root / relative).read_bytes()).hexdigest()
        for relative in CODE_PATHS
    }


def replay_cohort(
    cohort: LoadedCohort,
    model_dir: str,
    families: list[str],
    *,
    source: pd.DataFrame | None = None,
    source_file_sha256: str | None = None,
    synced: bool = False,
) -> tuple[pd.DataFrame, dict]:
    recipe = HistoryRecipe.from_dict(cohort.manifest["recipe"])
    position = recipe.position
    spec = INFERENCE_REGISTRY[position]
    readiness = cohort.manifest["model_input_readiness"]
    predictors, inputs, raw, totals, excluded = {}, {}, {}, {}, {}
    for family in families:
        entry = readiness.get(family)
        if entry is None or not entry["ready"]:
            excluded[family] = (entry or {}).get("reason") or "cohort does not declare this family"
            continue
        predictor = Predictor.from_bundle(
            model_dir, family, position=position, legacy_spec=spec, device=torch.device("cpu")
        )
        family_inputs = prediction_inputs(predictor, cohort)
        predictions = predictor.predict_raw(family_inputs)
        for target in predictor.schema.targets:
            if not np.isfinite(predictions[target]).all():
                raise ValueError(f"non-finite responses from {family}")
        predictors[family], inputs[family], raw[family] = predictor, family_inputs, predictions
        totals[family] = {fmt: predictor.score(predictions, fmt) for fmt in SCORING_FORMATS}
    if not predictors:
        raise ValueError(f"no requested family is replayable for this cohort: {excluded}")
    if source is not None:
        control = identity_control(cohort, predictors, inputs, raw, source)
        control["source_file_sha256"] = source_file_sha256
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
    identities = {family: family_identity(p, model_dir) for family, p in predictors.items()}
    manifest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "cohort_dir": str(cohort.directory),
        "cohort_manifest_sha256": cohort.manifest_sha256,
        "cohort_files": dict(cohort.manifest["files"]),
        "cohort_schema_version": cohort.manifest["schema_version"],
        "recipe": dict(cohort.manifest["recipe"]),
        "recipe_sha256": cohort.manifest["recipe_sha256"],
        "source_values_sha256": cohort.manifest["source_values_sha256"],
        "position": position,
        "model_dir": str(model_dir),
        "synced": bool(synced),
        "families": identities,
        "bundle_ids": {family: identity["bundle_id"] for family, identity in identities.items()},
        "families_excluded": excluded,
        "identity_control": control,
        "response_semantics": RESPONSE_SEMANTICS,
        "scoring": {"total_column_format": "ppr", "formats": list(SCORING_FORMATS)},
        "prediction_columns": [c for c in frame.columns if c.startswith("pred_")],
        "device": "cpu",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": version("pyarrow"),
            "torch": torch.__version__,
            "scikit-learn": version("scikit-learn"),
        },
        "code_sha256": _code_hashes(),
    }
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, required=True, help="Cohort artifact directory (schema_version 2)"
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
        help="The consumed source parquet; runs the identity control for replay cohorts",
    )
    args = parser.parse_args(argv)
    try:
        cohort = load_cohort(args.cohort)
        position = cohort.manifest["recipe"]["position"]
        if args.sync:
            warn_if_sync_noop()
            from src.artifacts.model_sync import sync_models_from_s3

            sync_models_from_s3()
        model_dir = resolve_model_dir(
            position, INFERENCE_REGISTRY[position], str(args.model_dir) if args.model_dir else None
        )
        families = requested_families(args.families, model_dir)
        source = pd.read_parquet(args.source) if args.source is not None else None
        predictions, manifest = replay_cohort(
            cohort,
            model_dir,
            families,
            source=source,
            source_file_sha256=file_digest(args.source) if args.source is not None else None,
            synced=args.sync,
        )
        output = publish_artifact_dir(
            args.output,
            lambda directory: predictions.to_parquet(
                directory / "predictions.parquet", index=False
            ),
            manifest,
            manifest_name="replay_manifest.json",
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"synthetic-replay: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "position": position,
                "cases": len(predictions),
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
