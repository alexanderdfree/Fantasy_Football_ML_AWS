"""Resolved recipes and named data/results at the training boundary.

The numerical working frames/arrays remain pandas/numpy objects. Recipe
containers are detached and frozen; the Mapping interface returns copies of
lists/dicts for existing consumers that require those concrete types.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True)
class _FrozenList:
    values: tuple


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value.values]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return set(value)
    return value


@dataclass(frozen=True)
class FeatureSpec:
    columns: tuple[str, ...]
    specific: tuple[str, ...]
    static: tuple[str, ...]
    history: tuple[str, ...]
    opponent_history: tuple[str, ...]


@dataclass(frozen=True)
class FlatModelSpec:
    targets: tuple[str, ...]
    history_length: int
    dimensions: tuple[tuple[str, int], ...]
    structure: Literal["flat"] = "flat"


@dataclass(frozen=True)
class NestedModelSpec:
    targets: tuple[str, ...]
    kick_columns: tuple[str, ...]
    max_games: int
    max_kicks: int
    structure: Literal["nested"] = "nested"


@dataclass(frozen=True)
class TrainingOptions:
    values: Mapping[str, Any]


# Current extension controls which are intentionally not PositionConfig fields.
# New experimental controls belong in experimental_options until promoted.
_EXTRA_OPTIONS = frozenset(
    [
        "filter_fn",
        "compute_targets_fn",
        "add_features_fn",
        "fill_nans_fn",
        "get_feature_columns_fn",
        "aggregate_fn",
        "attn_history_builder_fn",
        "attn_history_structure",
        "attn_static_from_df",
        "attn_kick_stats",
        "train_ridge",
        "train_base_nn",
        "trial_data_memo",
        "epoch_callback",
        "nn_log_every",
        "_artifact_branch",
        "classification_targets",
        "attn_history_dropout",
        "attn_learn_temperature",
        "attn_use_alibi_bias",
        "attn_use_swiglu_encoder",
        "experimental_options",
        "compute_adjustment_fn",
    ]
)


@dataclass(frozen=True)
class ResolvedRecipe(Mapping[str, Any]):
    position: str
    features: FeatureSpec
    model: FlatModelSpec | NestedModelSpec
    training: TrainingOptions
    _values: Mapping[str, Any]

    def __getitem__(self, key):
        return _thaw(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def with_overrides(self, **overrides) -> ResolvedRecipe:
        return resolve_recipe(self.position, {**dict(self), **overrides})

    def __reduce__(self):
        return resolve_recipe, (self.position, dict(self))


def resolve_recipe(
    position: str, cfg: Mapping[str, Any], *, require_runtime: bool = True
) -> ResolvedRecipe:
    from src.shared.position_config import PositionConfig
    from src.shared.position_pipeline import PipelineConfigError, validate_pipeline_config

    if isinstance(cfg, ResolvedRecipe):
        if cfg.position != position:
            raise PipelineConfigError("Resolved recipe belongs to a different position")
        if (
            require_runtime
            and cfg.model.structure == "nested"
            and cfg.get("train_attention_nn")
            and not callable(cfg.get("attn_history_builder_fn"))
        ):
            raise PipelineConfigError("Nested attention requires attn_history_builder_fn")
        return cfg
    validate_pipeline_config(cfg, context=position)
    known = {item.name for item in fields(PositionConfig)} | _EXTRA_OPTIONS
    unknown = set(cfg) - known - set(cfg.get("experimental_options") or {})
    if unknown:
        raise PipelineConfigError(
            f"Unknown pipeline options: {sorted(unknown)}; put deliberate "
            "experimental extensions in experimental_options"
        )
    values = dict(cfg)
    if values.get("compute_adjustment_fn") is not None:
        raise PipelineConfigError(
            "compute_adjustment_fn is obsolete; targets must remain raw stats"
        )
    values.pop("trial_data_memo", None)
    extensions = values.get("experimental_options", {})
    if not isinstance(extensions, Mapping):
        raise PipelineConfigError("experimental_options must be a mapping")
    if set(extensions) & known:
        raise PipelineConfigError("experimental_options cannot shadow supported recipe fields")
    values.update(extensions)
    targets = tuple(values["targets"])
    columns = tuple(values["get_feature_columns_fn"]())
    for name, names in (("targets", targets), ("features", columns)):
        if not names or len(set(names)) != len(names):
            raise PipelineConfigError(f"{name} must be nonempty and unique")
    for key in ("loss_weights", "huber_deltas"):
        required = set(targets)
        if key == "huber_deltas":
            required = {
                target
                for target in targets
                if values.get("head_losses", {}).get(target, "huber") in {"huber", "mse"}
            }
        missing = required - set(values[key])
        if missing:
            raise PipelineConfigError(f"{key} missing target coverage: {sorted(missing)}")
    for key in ("nn_non_negative_targets", "gated_targets", "head_losses"):
        extra = set(values.get(key) or ()) - set(targets)
        if extra:
            raise PipelineConfigError(f"{key} contains unknown targets: {sorted(extra)}")
    feature = FeatureSpec(
        columns,
        tuple(values["specific_features"]),
        tuple(values.get("attn_static_features") or ()),
        tuple(values.get("attn_history_stats") or ()),
        tuple(values.get("opp_attn_history_stats") or ()),
    )
    for name in ("static", "history", "opponent_history"):
        names = getattr(feature, name)
        if len(names) != len(set(names)):
            raise PipelineConfigError(f"Duplicate ordered {name} input columns")
    structure = values.get("attn_history_structure", "flat")
    if structure == "nested":
        if (
            require_runtime
            and values.get("train_attention_nn")
            and not callable(values.get("attn_history_builder_fn"))
        ):
            raise PipelineConfigError("Nested attention requires attn_history_builder_fn")
        model = NestedModelSpec(
            targets,
            tuple(values.get("attn_kick_stats") or ()),
            values.get("attn_max_games", 17),
            values.get("attn_max_kicks_per_game", 10),
        )
        if model.max_games < 1 or model.max_kicks < 1:
            raise PipelineConfigError("Nested history dimensions must be positive")
    elif structure == "flat":
        dimensions = tuple(
            (key, values[key]) for key in ("attn_d_model", "attn_n_heads") if key in values
        )
        model = FlatModelSpec(targets, values.get("attn_max_seq_len", 17), dimensions)
        if model.history_length < 1 or any(value < 1 for _, value in dimensions):
            raise PipelineConfigError("Attention dimensions must be positive")
        if values.get("attn_self_layers", 0) and values.get("attn_d_model", 1) % values.get(
            "attn_self_heads", 1
        ):
            raise PipelineConfigError("Self-attention d_model must be divisible by attn_self_heads")
    else:
        raise PipelineConfigError(f"Unknown history structure: {structure}")
    # Bind the resolved ordered projection so later global config mutation
    # cannot change which columns this recipe uses midway through a run.
    values["get_feature_columns_fn"] = _ColumnGetter(columns)
    frozen = _freeze(values)
    training_keys = {
        key: value
        for key, value in values.items()
        if key.startswith(("nn_", "attn_", "ridge_", "lgbm_", "enet_", "train_", "tabpfn_"))
        or key in {"loss_weights", "huber_deltas", "head_losses", "scheduler_type"}
    }
    return ResolvedRecipe(position, feature, model, TrainingOptions(_freeze(training_keys)), frozen)


@dataclass(frozen=True)
class _ColumnGetter:
    columns: tuple[str, ...]

    def __call__(self):
        return list(self.columns)


@dataclass(frozen=True)
class PreparedDataset:
    X_train: Any
    X_val: Any
    X_test: Any
    y_train: Mapping
    y_val: Mapping
    y_test: Mapping | None
    train: Any
    val: Any
    test: Any
    feature_columns: tuple[str, ...]
    data_id: str

    @property
    def preprocessing(self):
        return self.train.attrs.get("preprocessing_state", {})

    def __iter__(self):
        return iter(
            (
                self.X_train,
                self.X_val,
                self.X_test,
                self.y_train,
                self.y_val,
                self.y_test,
                self.train,
                self.val,
                self.test,
                list(self.feature_columns),
            )
        )

    def __len__(self):
        return 10

    def __getitem__(self, key):
        return tuple(self)[key]


@dataclass(frozen=True)
class DatasetSplits:
    """Common provider result; nested kick context remains an explicit binding."""

    train: Any
    val: Any
    test: Any
    bindings: Mapping[str, Any]

    @property
    def frames(self):
        return self.train, self.val, self.test


@dataclass(frozen=True)
class TrainingResult(Mapping[str, Any]):
    _values: Mapping[str, Any]
    recipe: ResolvedRecipe
    prepared: PreparedDataset
    models: Mapping[str, Any]
    run_id: str

    def __post_init__(self):
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    @property
    def predictions(self):
        return self._values["per_target_preds"]

    @property
    def metrics(self):
        return {key: value for key, value in self._values.items() if key.endswith("_metrics")}

    @property
    def timings(self):
        return self._values.get("phase_seconds", {})

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def copy(self):
        return dict(self._values)

    def __reduce__(self):
        return type(self), (
            dict(self._values),
            self.recipe,
            self.prepared,
            dict(self.models),
            self.run_id,
        )


def training_result(values, recipe, prepared, models, context, bundle_ids=None):
    device = None
    for family in ("nn", "attn_nn"):
        model = models.get(family)
        if model is not None and hasattr(model, "parameters"):
            parameter = next(model.parameters(), None)
            if parameter is not None:
                device = str(parameter.device)
                break
    metadata = {
        **values,
        "data_id": prepared.data_id,
        "model_bundle_ids": dict(bundle_ids or {}),
        "run_id": context.run_id,
        "execution": context.metadata(device=device),
    }
    return TrainingResult(metadata, recipe, prepared, models, context.run_id)
