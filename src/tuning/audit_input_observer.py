"""Read-only input evidence for WR/TE development; never iterate or fit data.

Installed only by the isolated audit spec. Context-local trainer wrappers keep
saved-inference replay and unrelated pipeline calls out of the training proof.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from contextvars import ContextVar
from functools import wraps

import numpy as np
import pandas as pd

from src.prediction.bundle import canonical_json

_ACTIVE = ContextVar("audit_input_observer", default=None)
_SPLITS = ("train", "val", "test")


def array_proof(values, columns=None):
    """Hash shape, dtype and ordered bytes; columns name the final axis."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    result = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(
            str((values.shape, values.dtype)).encode() + values.tobytes()
        ).hexdigest(),
    }
    if columns is not None:
        if values.ndim == 0 or values.shape[-1] != len(columns):
            raise ValueError("Input columns do not match the observed final axis")
        result["columns"] = {
            column: array_proof(values[..., index])["sha256"]
            for index, column in enumerate(columns)
        }
    return result


def _json_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def prepared_proof(prepared):
    features = list(prepared.feature_columns)
    targets = list(prepared.y_train)
    proof = {
        "schema": "audit-input-proof/v1",
        "feature_columns": features,
        "target_columns": targets,
        "prepared": {},
    }
    for split in _SPLITS:
        frame = getattr(prepared, split)
        columns = [c for c in frame if not c.startswith("pred_")]
        keys = frame[["player_id", "season", "week"]]
        proof["prepared"][split] = {
            "row_keys_hash": hashlib.sha256(
                pd.util.hash_pandas_object(keys, index=True).values.tobytes()
            ).hexdigest(),
            "frame_columns": columns,
            "frame_column_hashes": {
                c: hashlib.sha256(
                    str(frame[c].dtype).encode()
                    + pd.util.hash_pandas_object(frame[c], index=True).values.tobytes()
                ).hexdigest()
                for c in columns
            },
            "X": array_proof(getattr(prepared, f"X_{split}"), features),
            "y": {t: array_proof(getattr(prepared, f"y_{split}")[t]) for t in targets},
        }
    state = json.loads(canonical_json(prepared.preprocessing))
    proof["preprocessing"] = {
        "state": state,
        "sha256": _json_hash(state),
        "fields": {k: _json_hash(v) for k, v in state.items() if k != "fill_values"},
        "fill_values": {k: _json_hash(v) for k, v in state.get("fill_values", {}).items()},
    }
    return proof


def _input_proof(features, record):
    names = ["static"]
    columns = [record["feature_columns"]]
    if record["family"] == "attn_nn":
        names += ["history", "mask"]
        columns += [record["history_columns"], None]
        if record["opp_history_columns"]:
            names += ["opp_history", "opp_mask"]
            columns += [record["opp_history_columns"], None]
    if len(features) != len(names):
        raise ValueError("Observed trainer input tuple does not match its feature schema")
    return {
        name: array_proof(values, cols)
        for name, values, cols in zip(names, features, columns, strict=True)
    }


def observe_loaders(train_loader, val_loader):
    """Read stored tensors in source-row order, without consuming sampler RNG."""
    record = _ACTIVE.get()
    if record is None:
        return
    for split, loader in (("train", train_loader), ("val", val_loader)):
        if hasattr(loader, "_features"):
            features, targets = loader._features, loader._y_dict
            source = "GPUResidentBatcher._features"
        else:
            dataset = loader.dataset
            if record["family"] == "nn":
                features = (dataset.X,)
            else:
                features = (dataset.X_static, dataset.X_history, dataset.history_mask)
                if record["opp_history_columns"]:
                    features += (dataset.X_opp_history, dataset.opp_history_mask)
            targets = dataset.targets
            source = "DataLoader.dataset tensors"
        record["actual"][split] = {
            "source": source,
            "inputs": _input_proof(features, record),
            "targets": {t: array_proof(targets[t]) for t in record["target_columns"]},
        }


def _wrap_train(original, family, state):
    signature = inspect.signature(original)

    @wraps(original)
    def train(*args, **kwargs):
        if state.get("position") not in {"WR", "TE"}:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        cfg = values["cfg"]
        features = values["feature_cols"]
        if features is None:
            features = cfg["get_feature_columns_fn"]()
        if family == "attn_nn":
            from src.shared.pipeline import get_attn_static_columns

            features = get_attn_static_columns(features, cfg["attn_static_features"])
        record = {
            "family": family,
            "feature_columns": list(features),
            "target_columns": list(values["targets"]),
            "history_columns": list(cfg.get("attn_history_stats") or [])
            if family == "attn_nn"
            else [],
            "opp_history_columns": list(cfg.get("opp_attn_history_stats") or [])
            if family == "attn_nn"
            else [],
            "actual": {},
        }
        token = _ACTIVE.set(record)
        try:
            result = original(*args, **kwargs)
            if set(record["actual"]) != set(_SPLITS) or "scaled" not in record:
                raise ValueError("Missing actual neural input observations")
            record["actual"]["test"]["targets"] = {
                t: array_proof(values["y_test_dict"][t]) for t in record["target_columns"]
            }
            state.setdefault("neural_inputs", {})[family] = record
            return result
        finally:
            _ACTIVE.reset(token)

    return train


def _wrap_scale(original):
    @wraps(original)
    def scale(*args, **kwargs):
        result = original(*args, **kwargs)
        record = _ACTIVE.get()
        if record is not None:
            scaler, scaled = result
            if len(scaled) != 3:
                raise ValueError("Expected train, validation and test scaler outputs")
            columns = record["feature_columns"]
            if list(kwargs["feature_cols"]) != columns:
                raise ValueError("Scaler feature order differs from trainer feature order")
            record["scaled"] = {
                split: array_proof(values, columns)
                for split, values in zip(_SPLITS, scaled, strict=True)
            }
            record["scaler"] = {
                field: array_proof(getattr(scaler, field), columns)
                for field in ("mean_", "var_", "scale_")
            }
            record["scaler"]["n_samples_seen_"] = array_proof(scaler.n_samples_seen_)
        return result

    return scale


def _wrap_predict(original, family):
    signature = inspect.signature(original)

    @wraps(original)
    def predict(*args, **kwargs):
        record = _ACTIVE.get()
        if record is not None and record["family"] == family:
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            values = values.arguments
            features = (
                [values["X"]]
                if family == "nn"
                else [values["X_static"], values["X_history"], values["history_mask"]]
            )
            if record["opp_history_columns"]:
                features += [values["X_opp_history"], values["opp_history_mask"]]
            if "test" in record["actual"]:
                raise ValueError("Multiple test prediction calls in a single trainer")
            record["actual"]["test"] = {
                "source": "predict_numpy arguments",
                "inputs": _input_proof(features, record),
            }
        return original(*args, **kwargs)

    return predict


def install(state):
    """Idempotently observe the existing pipeline calls, preserving all returns."""
    from src.shared import neural_net, pipeline

    targets = [
        (pipeline, "_train_nn", lambda fn: _wrap_train(fn, "nn", state)),
        (pipeline, "_train_attention_nn", lambda fn: _wrap_train(fn, "attn_nn", state)),
        (pipeline, "_scale_xs", _wrap_scale),
        (neural_net.MultiHeadNet, "predict_numpy", lambda fn: _wrap_predict(fn, "nn")),
        (
            neural_net.MultiHeadNetWithHistory,
            "predict_numpy",
            lambda fn: _wrap_predict(fn, "attn_nn"),
        ),
    ]
    for owner, name, wrapper in targets:
        current = getattr(owner, name)
        original = getattr(current, "_audit_input_original", current)
        observed = wrapper(original)
        observed._audit_input_original = original
        setattr(owner, name, observed)


def result_proof(prepared, state):
    proof = prepared_proof(prepared)
    proof["neural"] = state.get("neural_inputs", {})
    if set(proof["neural"]) != {"nn", "attn_nn"}:
        raise ValueError("Both WR/TE neural input paths must be observed")
    return proof
