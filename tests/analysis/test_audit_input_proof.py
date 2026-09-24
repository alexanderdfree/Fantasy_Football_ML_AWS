"""Synthetic measured-input receipts; no fitting or source data access."""

import hashlib
import importlib
from copy import deepcopy

import numpy as np
import pytest

from src.analysis import audit_input_proof as proof
from src.features.engineer import get_attn_static_columns

pytestmark = pytest.mark.unit


def array(values, columns=None):
    values = np.ascontiguousarray(values)
    result = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(
            str((values.shape, values.dtype)).encode() + values.tobytes()
        ).hexdigest(),
    }
    if columns is not None:
        result["columns"] = {
            name: array(values[..., i])["sha256"] for i, name in enumerate(columns)
        }
    return result


def record(position="WR", *, changed=False):
    cfg = importlib.import_module(f"src.{position.lower()}.run_pipeline").CONFIG
    features, targets = cfg["get_feature_columns_fn"](), list(cfg["targets"])
    history = list(cfg["attn_history_stats"])
    columns = list(dict.fromkeys(["player_id", "season", "week", *features, *targets]))
    x = np.zeros((2, len(features)))
    if changed:
        x[:, features.index("opportunity_index_L3")] = 1
    prepared, hashes = {}, {}
    for split in proof.SPLITS:
        part = {
            "row_keys_hash": "a" * 64,
            "frame_columns": columns,
            "frame_column_hashes": {
                name: hashlib.sha256(name.encode()).hexdigest() for name in columns
            },
            "X": array(x, features),
            "y": {target: array(np.zeros(2)) for target in targets},
        }
        if changed:
            part["frame_column_hashes"]["opportunity_index_L3"] = "b" * 64
        prepared[split] = part
        hashes[f"X_{split}"] = part["X"]["sha256"]
        hashes.update({f"y_{split}/{target}": part["y"][target]["sha256"] for target in targets})
    state = {"fill_values": {name: 0 for name in features}, "fixed_schema": "same"}
    if changed:
        state["fill_values"]["opportunity_index_L3"] = 1
    p = {
        "schema": "audit-input-proof/v1",
        "feature_columns": features,
        "target_columns": targets,
        "prepared": prepared,
        "preprocessing": {
            "state": state,
            "sha256": proof._hash(state),
            "fields": {"fixed_schema": proof._hash("same")},
            "fill_values": {name: proof._hash(v) for name, v in state["fill_values"].items()},
        },
        "neural": {},
    }
    for family in ("nn", "attn_nn"):
        names = (
            features
            if family == "nn"
            else get_attn_static_columns(features, cfg["attn_static_features"])
        )
        mean = np.zeros(len(names))
        if changed and family == "nn":
            mean[names.index("opportunity_index_L3")] = 1
        actual = {}
        for split in proof.SPLITS:
            inputs = {"static": array(np.zeros((2, len(names)), dtype=np.float32), names)}
            if family == "attn_nn":
                inputs.update(
                    history=array(np.zeros((2, 3, len(history)), dtype=np.float32), history),
                    mask=array(np.ones((2, 3), dtype=bool)),
                )
            actual[split] = {
                "source": "predict_numpy arguments"
                if split == "test"
                else "GPUResidentBatcher._features",
                "inputs": inputs,
                "targets": {target: array(np.zeros(2, dtype=np.float32)) for target in targets},
            }
        p["neural"][family] = {
            "family": family,
            "feature_columns": names,
            "target_columns": targets,
            "history_columns": history if family == "attn_nn" else [],
            "opp_history_columns": [],
            "scaler": {
                "mean_": array(mean, names),
                "var_": array(np.ones(len(names)), names),
                "scale_": array(np.ones(len(names)), names),
                "n_samples_seen_": array(np.asarray(2)),
            },
            "scaled": {s: array(np.zeros((2, len(names))), names) for s in proof.SPLITS},
            "actual": actual,
        }
    return {"position": position, "prepared_hashes": hashes, "input_proof": p}


@pytest.mark.parametrize("position", ["WR", "TE"])
def test_intended_columns_and_derived_scaler_changes_are_permitted(position):
    base, changed = record(position), record(position, changed=True)
    proof.verify_stint_pair(base, deepcopy(base), repeat=True)
    proof.verify_stint_pair(base, changed)


def test_hidden_training_column_change_is_rejected():
    base, changed = record(), record(changed=True)
    changed["input_proof"]["prepared"]["train"]["frame_column_hashes"]["player_id"] = "c" * 64
    with pytest.raises(ValueError, match="Unexpected prepared train"):
        proof.verify_stint_pair(base, changed)


def test_attention_history_change_invalidates_control_family():
    base, changed = record(), record(changed=True)
    changed["input_proof"]["neural"]["attn_nn"]["actual"]["val"]["inputs"]["history"]["sha256"] = (
        "c" * 64
    )
    with pytest.raises(ValueError, match="Attention inputs/scaler changed"):
        proof.verify_stint_pair(base, changed)


def test_missing_split_or_column_proof_never_weakens_the_gate():
    base, changed = record(), record(changed=True)
    del changed["input_proof"]["prepared"]["train"]["X"]["columns"]["opportunity_index_L3"]
    with pytest.raises(ValueError, match="Incomplete per-column"):
        proof.verify_stint_pair(base, changed)
    changed = record(changed=True)
    del changed["input_proof"]["neural"]["attn_nn"]["actual"]["test"]
    with pytest.raises(ValueError, match="Missing actual neural input split"):
        proof.verify_stint_pair(base, changed)


def test_noop_stint_is_not_an_active_positive_control():
    base = record()
    with pytest.raises(ValueError, match="did not change"):
        proof.verify_stint_pair(base, deepcopy(base))
