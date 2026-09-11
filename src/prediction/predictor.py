"""One model loading and raw prediction adapter for offline and HTTP consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    build_game_history_arrays,
    build_opp_defense_history_arrays,
)
from src.prediction.bundle import InputSchema, ModelBundle, read_bundle
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import assert_scaler_matches, read_scaler_meta, unwrap_state_dict
from src.shared.feature_build import scale_and_clip
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)


@dataclass(frozen=True)
class PredictionInputs:
    schema: InputSchema
    values: np.ndarray
    history: np.ndarray | None = None
    history_mask: np.ndarray | None = None
    inner_mask: np.ndarray | None = None
    game_history: np.ndarray | None = None
    opponent_history: np.ndarray | None = None
    opponent_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[1] != len(self.schema.features):
            raise ValueError("Prediction input dimensions do not match feature schema")
        n = len(self.values)
        for array in (
            self.history,
            self.history_mask,
            self.inner_mask,
            self.game_history,
            self.opponent_history,
            self.opponent_mask,
        ):
            if array is not None and len(array) != n:
                raise ValueError("Prediction input row counts differ")
        if self.history is not None:
            expected = len(
                self.schema.kicks if self.schema.structure == "nested" else self.schema.history
            )
            if self.history.shape[-1] != expected:
                raise ValueError("Prediction history dimensions do not match ordered schema")
        if self.opponent_history is not None and self.opponent_history.shape[-1] != len(
            self.schema.opponent_history
        ):
            raise ValueError("Opponent history dimensions do not match ordered schema")


def legacy_schema(spec: dict, family: str) -> InputSchema:
    features = list(spec["get_feature_columns_fn"]())
    if family == "attn_nn":
        static = spec.get("attn_static_features", [])
        features = (
            list(static)
            if spec.get("attn_static_from_df")
            else [c for c in features if c in static]
        )
    return InputSchema.from_config(spec, features, attention=family == "attn_nn")


class Predictor:
    """A loaded model owns its input schema, constructor and fitted scaler."""

    def __init__(
        self, position, family, schema, model, *, device, scaler=None, bundle=None, options=None
    ):
        self.position = position
        self.family = family
        self.schema = schema
        self.model = model
        self.device = device
        self.scaler = scaler
        self.bundle: ModelBundle | None = bundle
        self.options = dict(options or {})

    @classmethod
    def from_bundle(cls, directory, family, *, position, legacy_spec=None, device=None):
        bundle = read_bundle(directory, family, verify=False)
        if bundle is not None:
            with bundle.pinned_directory(directory) as pinned:
                return cls._load(
                    pinned,
                    family,
                    position=position,
                    legacy_spec=legacy_spec,
                    device=device,
                    bundle=bundle,
                )
        return cls._load(
            directory, family, position=position, legacy_spec=legacy_spec, device=device
        )

    @classmethod
    def _load(cls, directory, family, *, position, legacy_spec=None, device=None, bundle=None):
        directory = Path(directory)
        device = device or torch.device("cpu")
        if bundle is not None:
            document = bundle.to_dict()
            if document["position"] != position:
                raise ValueError("Model bundle position mismatch")
            schema = bundle.inputs
            options = document["history_options"]
            architecture = document["architecture"]
        else:
            if legacy_spec is None:
                raise ValueError(f"Legacy {position}/{family} model requires an explicit recipe")
            schema = legacy_schema(legacy_spec, family)
            options = legacy_spec
            architecture = None
        scaler = None
        if family == "ridge":
            model = RidgeMultiTarget(target_names=list(schema.targets))
            model.load(str(directory))
        elif family == "lgbm":
            model = LightGBMMultiTarget(target_names=list(schema.targets))
            model.load(str(directory))
        else:
            stem = "attention_nn" if family == "attn_nn" else "nn"
            file_stem = "attention_nn" if family == "attn_nn" else "multihead_nn"
            scaler = joblib.load(directory / f"{stem}_scaler.pkl")
            meta = read_scaler_meta(directory / f"{stem}_scaler_meta.json")
            filename = (
                f"{position.lower()}_{file_stem}.pt"
                if bundle is not None
                else legacy_spec["attn_nn_file" if family == "attn_nn" else "nn_file"]
            )
            checkpoint = torch.load(directory / filename, map_location=device, weights_only=True)
            state, feature_hash = unwrap_state_dict(checkpoint)
            assert_scaler_matches(
                position,
                scaler,
                feature_hash,
                meta,
                schema.features,
                schema.targets,
                scaler_label=f"{stem}_scaler",
            )
            if architecture is None:
                if family == "nn":
                    architecture = {
                        "class": "MultiHeadNet",
                        "kwargs": {
                            "input_dim": len(schema.features),
                            "target_names": list(schema.targets),
                            **legacy_spec["nn_kwargs"],
                        },
                    }
                else:
                    kwargs = {
                        "static_dim": len(schema.features),
                        "target_names": list(schema.targets),
                        **legacy_spec["attn_nn_kwargs_static"],
                    }
                    if schema.structure == "nested":
                        kwargs["kick_dim"] = len(schema.kicks)
                        name = "MultiHeadNetWithNestedHistory"
                    else:
                        kwargs["game_dim"] = len(schema.history)
                        kwargs["opp_game_dim"] = len(schema.opponent_history) or None
                        name = "MultiHeadNetWithHistory"
                    architecture = {"class": name, "kwargs": kwargs}
            constructors = {
                "MultiHeadNet": MultiHeadNet,
                "MultiHeadNetWithHistory": MultiHeadNetWithHistory,
                "MultiHeadNetWithNestedHistory": MultiHeadNetWithNestedHistory,
            }
            if architecture["class"] not in constructors:
                raise ValueError("Unsupported model bundle constructor")
            model = constructors[architecture["class"]](**architecture["kwargs"]).to(device)
            model.load_state_dict(state)
        return cls(
            position,
            family,
            schema,
            model,
            device=device,
            scaler=scaler,
            bundle=bundle,
            options=options,
        )

    def predict_raw(self, inputs: PredictionInputs) -> dict[str, np.ndarray]:
        if inputs.schema != self.schema:
            raise ValueError("Prediction ordered input schema differs from the trained model")
        x = inputs.values
        if self.family in {"ridge", "lgbm"}:
            return self.model.predict(x)
        if self.bundle is None:
            x = scale_and_clip(self.scaler, x)
        else:
            clip = self.bundle.to_dict()["preprocessing"].get("clip", [-4.0, 4.0])
            x = np.clip(self.scaler.transform(x), *clip)
        if self.family == "nn":
            return self.model.predict_numpy(x, self.device)
        if self.schema.structure == "nested":
            return self.model.predict_numpy(
                x,
                inputs.history,
                inputs.history_mask,
                inputs.inner_mask,
                self.device,
                X_game_history=inputs.game_history,
            )
        if self.schema.opponent_history:
            return self.model.predict_numpy(
                x,
                inputs.history,
                inputs.history_mask,
                self.device,
                X_opp_history=inputs.opponent_history,
                opp_history_mask=inputs.opponent_mask,
            )
        return self.model.predict_numpy(x, inputs.history, inputs.history_mask, self.device)

    def score(self, predictions, scoring="ppr"):
        return predictions_to_fantasy_points(self.position, predictions, scoring_format=scoring)

    def inputs_from_frame(self, frame, *, kicks=None, opponent_weekly=None) -> PredictionInputs:
        values = frame[list(self.schema.features)].to_numpy(dtype=np.float32)
        if self.family != "attn_nn":
            return PredictionInputs(self.schema, values)
        options = self.options
        if self.schema.structure == "nested":
            from src.k.features import build_nested_kick_history

            if kicks is None:
                raise ValueError("Nested kicker predictions require explicit kick history")
            games = options.get("attn_max_games") or 17
            history, outer, inner = build_nested_kick_history(
                frame,
                kicks_df=kicks,
                kick_stats=list(self.schema.kicks),
                max_games=games,
                max_kicks_per_game=options.get("attn_max_kicks_per_game") or 10,
            )
            game_history = None
            if self.schema.history:
                game_history, _ = build_game_history_arrays(
                    frame, history_stats=list(self.schema.history), max_seq_len=games
                )
            return PredictionInputs(self.schema, values, history, outer, inner, game_history)
        length = options.get("attn_max_seq_len") or 17
        history, mask = build_game_history_arrays(
            frame, history_stats=list(self.schema.history), max_seq_len=length
        )
        opponent = opponent_mask = None
        if self.schema.opponent_history:
            if opponent_weekly is None:
                raise ValueError("Opponent attention requires explicit weekly context")
            builder = OPP_ATTN_PER_GAME_BUILDERS[options.get("opp_attn_kind", "defense")]
            per_game = builder(opponent_weekly)
            opponent, opponent_mask = build_opp_defense_history_arrays(
                frame,
                per_game,
                list(self.schema.opponent_history),
                options.get("opp_attn_max_seq_len") or length,
            )
        return PredictionInputs(
            self.schema,
            values,
            history,
            mask,
            opponent_history=opponent,
            opponent_mask=opponent_mask,
        )

    def zero_inputs(self) -> PredictionInputs:
        values = np.zeros((1, len(self.schema.features)), dtype=np.float32)
        if self.family != "attn_nn":
            return PredictionInputs(self.schema, values)
        if self.schema.structure == "nested":
            games = self.options.get("attn_max_games") or 17
            kicks = self.options.get("attn_max_kicks_per_game") or 10
            history = np.zeros((1, games, kicks, len(self.schema.kicks)), dtype=np.float32)
            outer = np.ones((1, games), dtype=bool)
            inner = np.ones((1, games, kicks), dtype=bool)
            game_history = (
                np.zeros((1, games, len(self.schema.history)), dtype=np.float32)
                if self.schema.history
                else None
            )
            return PredictionInputs(self.schema, values, history, outer, inner, game_history)
        length = self.options.get("attn_max_seq_len") or 17
        history = np.zeros((1, length, len(self.schema.history)), dtype=np.float32)
        mask = np.ones((1, length), dtype=bool)
        opponent = opponent_mask = None
        if self.schema.opponent_history:
            opp_length = self.options.get("opp_attn_max_seq_len") or length
            opponent = np.zeros(
                (1, opp_length, len(self.schema.opponent_history)), dtype=np.float32
            )
            opponent_mask = np.ones((1, opp_length), dtype=bool)
        return PredictionInputs(
            self.schema,
            values,
            history,
            mask,
            opponent_history=opponent,
            opponent_mask=opponent_mask,
        )
