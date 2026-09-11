"""Immutable, self-describing model artifacts; no training or Flask imports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

BUNDLE_VERSION = 1
SCORING_VERSION = 1
MODEL_FAMILIES = ("ridge", "nn", "attn_nn", "lgbm")


def record_constructor(init):
    """Record the actual resolved constructor, without adding tensor state."""
    signature = inspect.signature(init)

    @wraps(init)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init(self, *args, **kwargs)
        if "backbone_norm" in bound.arguments:
            bound.arguments["backbone_norm"] = self.backbone_norm
        self._constructor_spec = {
            "class": type(self).__name__,
            "kwargs": json_value({k: v for k, v in bound.arguments.items() if k != "self"}),
        }

    return initialize


def json_value(value):
    """Detach configuration containers into JSON values with stable ordering."""
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(json_value(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(position: str) -> str:
    root = Path(__file__).resolve().parents[2]
    files = [
        path
        for name in ("shared", "data", "features", "prediction", "training", position.lower())
        for path in (root / "src" / name).rglob("*.py")
    ]
    return digest({str(path.relative_to(root)): file_digest(path) for path in sorted(set(files))})


def bundled_families(directory) -> tuple[str, ...] | None:
    directory = Path(directory)
    families = set()
    indexed = False
    for group in ("cpu", "nn"):
        path = directory / f"{group}.bundle-index.json"
        if path.exists():
            indexed = True
            index = json.loads(path.read_text())
            if index.get("schema_version") != BUNDLE_VERSION:
                raise ValueError("Unsupported bundle index version")
            members = {"ridge", "lgbm"} if group == "cpu" else {"nn", "attn_nn"}
            if not isinstance(index.get("families"), dict) or not set(index["families"]).issubset(
                members
            ):
                raise ValueError("Bundle index contains an invalid model family")
            families.update(index["families"])
    if not indexed:
        families = {
            family for family in MODEL_FAMILIES if (directory / f"{family}.bundle.json").exists()
        }
    return tuple(f for f in MODEL_FAMILIES if f in families) if indexed or families else None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical_json(payload))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class InputSchema:
    features: tuple[str, ...]
    targets: tuple[str, ...]
    history: tuple[str, ...] = ()
    opponent_history: tuple[str, ...] = ()
    kicks: tuple[str, ...] = ()
    structure: str = "flat"

    def __post_init__(self):
        if self.structure not in {"flat", "nested"}:
            raise ValueError(f"Unknown history structure: {self.structure}")
        for name in ("features", "targets", "history", "opponent_history", "kicks"):
            values = getattr(self, name)
            if not all(isinstance(v, str) and v for v in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid ordered {name} schema")
        if not self.targets:
            raise ValueError("A model must name its targets")

    @classmethod
    def from_config(cls, cfg: Mapping, feature_cols, *, attention=False):
        return cls(
            features=tuple(feature_cols),
            targets=tuple(cfg["targets"]),
            history=tuple(cfg.get("attn_history_stats") or ()) if attention else (),
            opponent_history=tuple(cfg.get("opp_attn_history_stats") or ()) if attention else (),
            kicks=tuple(cfg.get("attn_kick_stats") or ()) if attention else (),
            structure=cfg.get("attn_history_structure", "flat") if attention else "flat",
        )

    def to_dict(self):
        return {key: json_value(getattr(self, key)) for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        return cls(
            **{
                key: tuple(value.get(key, ()))
                for key in ("features", "targets", "history", "opponent_history", "kicks")
            },
            structure=value.get("structure", "flat"),
        )


@dataclass(frozen=True)
class ModelBundle:
    """Canonical JSON is immutable; properties return detached containers."""

    document: str

    def __post_init__(self):
        data = self.to_dict()
        if data.get("schema_version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported model bundle version: {data.get('schema_version')}")
        if data.get("family") not in MODEL_FAMILIES:
            raise ValueError("Unknown model family")
        if data.get("scoring_version") != SCORING_VERSION:
            raise ValueError("Unsupported model bundle scoring version")
        InputSchema.from_dict(data["inputs"])
        expected = digest({k: v for k, v in data.items() if k != "bundle_id"})
        if data.get("bundle_id") != expected:
            raise ValueError("Model bundle identity mismatch")

    def to_dict(self) -> dict:
        return json.loads(self.document)

    @property
    def bundle_id(self) -> str:
        return self.to_dict()["bundle_id"]

    @property
    def inputs(self) -> InputSchema:
        return InputSchema.from_dict(self.to_dict()["inputs"])

    def verify_files(self, directory: Path) -> None:
        root = directory.resolve()
        for name, expected in self.to_dict()["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
                raise ValueError(f"Model bundle artifact mismatch: {name}")

    def assert_inputs(self, schema: InputSchema) -> None:
        if schema != self.inputs:
            raise ValueError("Model bundle ordered input schema mismatch")

    @contextmanager
    def pinned_directory(self, directory):
        """Deserialize only bytes verified against this descriptor's generation.

        Checking mutable filenames and opening them later has a refresh race.
        Private verified copies also cover multi-file sklearn/LightGBM models.
        """
        root = Path(directory).resolve()
        with tempfile.TemporaryDirectory(prefix="prediction-bundle-") as temporary:
            target = Path(temporary)
            for name, expected in self.to_dict()["files"].items():
                source = (root / name).resolve()
                if not source.is_relative_to(root):
                    raise ValueError(f"Model bundle artifact escapes its directory: {name}")
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Model bundle artifact mismatch: {name}")
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            yield target


def read_bundle(directory: str | Path, family: str, *, verify=True) -> ModelBundle | None:
    directory = Path(directory)
    path = directory / f"{family}.bundle.json"
    if not path.exists():
        if any(directory.glob("*.bundle.json")) or any(directory.glob("*.bundle-index.json")):
            raise ValueError(f"Model family {family} is absent from this bundle generation")
        return None  # explicitly supported pre-bundle artifact generation
    bundle = ModelBundle(path.read_text())
    if bundle.to_dict()["family"] != family:
        raise ValueError(f"Model family mismatch in {path.name}")
    group = "cpu" if family in {"ridge", "lgbm"} else "nn"
    index_path = directory / f"{group}.bundle-index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if (
            index.get("schema_version") != BUNDLE_VERSION
            or index.get("families", {}).get(family) != bundle.bundle_id
        ):
            raise ValueError(f"Model bundle index mismatch: {family}")
    if verify:
        bundle.verify_files(directory)
    return bundle


def _model_files(directory: Path, position: str, family: str, targets) -> list[Path]:
    if family == "ridge":
        files = [p for target in targets for p in (directory / target).rglob("*") if p.is_file()]
        files.append(directory / "non_negative_targets.json")
    elif family == "lgbm":
        files = [p for p in (directory / "lightgbm").rglob("*") if p.is_file()]
    else:
        stem = "attention_nn" if family == "attn_nn" else "nn"
        weights = "attention_nn" if family == "attn_nn" else "multihead_nn"
        files = [
            directory / f"{position.lower()}_{weights}.pt",
            directory / f"{stem}_scaler.pkl",
            directory / f"{stem}_scaler_meta.json",
        ]
    if not files or any(not p.is_file() for p in files):
        raise ValueError(f"Incomplete {position}/{family} artifacts")
    return sorted(files)


def write_bundle(
    directory, position, family, cfg, feature_cols, model, *, preprocessing=None, data_id=None
):
    """Seal one model family after its weights and fitted scaler are saved."""
    directory = Path(directory)
    schema = InputSchema.from_config(cfg, feature_cols, attention=family == "attn_nn")
    versions = {}
    for package in ("torch", "numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    architecture = getattr(model, "_constructor_spec", None)
    if family in {"nn", "attn_nn"} and architecture is None:
        raise ValueError(f"{position}/{family} model has no constructor specification")
    from src.shared.aggregate_targets import TARGET_UNITS

    options = {}
    for key, value in cfg.items():
        if callable(value) or not (
            key.startswith(("nn_", "attn_", "ridge_", "lgbm_", "train_"))
            or key in {"loss_weights", "huber_deltas", "head_losses", "scheduler_type"}
        ):
            continue
        try:
            options[key] = json.loads(canonical_json(value))
        except (TypeError, ValueError):
            continue  # runtime tensors/callback state are represented by saved weights
    payload = {
        "schema_version": BUNDLE_VERSION,
        "position": position,
        "family": family,
        "inputs": schema.to_dict(),
        "architecture": architecture,
        "preprocessing": preprocessing or {},
        "scoring_version": SCORING_VERSION,
        "target_units": {name: TARGET_UNITS.get(name, "") for name in schema.targets},
        "training_options": options,
        "provenance": {
            "data_id": data_id,
            "dataset_id": os.environ.get("FF_DATASET_ID"),
            "image_sha": os.environ.get("FF_TRAIN_GIT_SHA"),
            "build_plan_id": os.environ.get("FF_BUILD_PLAN_ID"),
            "code_id": source_identity(position),
            "dependencies": versions,
        },
        "preparation": {
            "specific_features": list(cfg.get("specific_features", [])),
            "features": list(cfg["get_feature_columns_fn"]()),
            "min_games_per_season": cfg.get("min_games_per_season"),
        },
        "history_options": {
            key: cfg.get(key)
            for key in (
                "attn_max_seq_len",
                "attn_max_games",
                "attn_max_kicks_per_game",
                "opp_attn_max_seq_len",
                "opp_attn_kind",
            )
            if key in cfg
        },
        "files": {
            str(p.relative_to(directory)): file_digest(p)
            for p in _model_files(directory, position, family, schema.targets)
        },
    }
    payload = json_value(payload)
    payload["bundle_id"] = digest(payload)
    bundle = ModelBundle(canonical_json(payload))
    atomic_json(directory / f"{family}.bundle.json", bundle.to_dict())
    return bundle


def write_prediction_bundles(
    directory, position, cfg, feature_cols, models, train_frame, *, attention_features=()
):
    """Publish independent descriptors so CPU/NN branch outputs merge cleanly."""
    state = train_frame.attrs.get("preprocessing_state", {})
    identity = train_frame.attrs.get("prepared_data_id")
    written = {}
    branch = cfg.get("_artifact_branch")
    owned = (
        {"ridge", "lgbm"}
        if branch == "cpu"
        else {"nn", "attn_nn"}
        if branch == "nn"
        else set(MODEL_FAMILIES)
    )
    for family in owned:
        if models.get(family) is None:
            (Path(directory) / f"{family}.bundle.json").unlink(missing_ok=True)
    for family, model in models.items():
        if model is None:
            continue
        features = attention_features if family == "attn_nn" else feature_cols
        written[family] = write_bundle(
            directory, position, family, cfg, features, model, preprocessing=state, data_id=identity
        ).bundle_id
    for group, members in (("cpu", {"ridge", "lgbm"}), ("nn", {"nn", "attn_nn"})):
        if members & owned:
            atomic_json(
                Path(directory) / f"{group}.bundle-index.json",
                {
                    "schema_version": BUNDLE_VERSION,
                    "families": {
                        family: written[family]
                        for family in MODEL_FAMILIES
                        if family in members and family in written
                    },
                },
            )
    return written
