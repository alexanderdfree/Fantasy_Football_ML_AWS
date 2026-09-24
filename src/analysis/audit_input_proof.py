"""Enforce the declared WR/TE stint-feature boundary on actual fitted inputs."""

from __future__ import annotations

import hashlib
import importlib
import re

from src.features.engineer import get_attn_static_columns
from src.prediction.bundle import canonical_json

CHANGED = {"opportunity_index_L3", "redzone_target_share_L3"}
SPLITS = ("train", "val", "test")


def require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _array(proof, columns=None):
    require(
        _digest(proof.get("sha256")) and isinstance(proof.get("dtype"), str),
        "Missing array fingerprint",
    )
    require(
        isinstance(proof.get("shape"), list)
        and all(type(n) is int and n > 0 for n in proof["shape"]),
        "Invalid input shape",
    )
    if columns is not None:
        require(proof["shape"] and proof["shape"][-1] == len(columns), "Wrong input width")
        require(
            set(proof.get("columns", {})) == set(columns)
            and all(_digest(v) for v in proof["columns"].values()),
            "Incomplete per-column input fingerprints",
        )


def _columns(left, right, allowed, name):
    require(set(left) == set(right), f"Changed {name} column schema")
    drift = {column for column in left if left[column] != right[column]}
    require(drift <= allowed, f"Unexpected {name} column drift: {sorted(drift - allowed)}")
    return drift


def _paired_array(left, right, columns, allowed, name):
    _array(left, columns)
    _array(right, columns)
    require(
        (left["shape"], left["dtype"]) == (right["shape"], right["dtype"]),
        f"Changed {name} shape/dtype",
    )
    drift = _columns(left["columns"], right["columns"], allowed, name)
    require(
        bool(drift) == (left["sha256"] != right["sha256"]),
        f"Inconsistent {name} whole/per-column hashes",
    )
    return drift


def validate_input_proof(record):
    """Require actual train/validation/test evidence before permitting feature drift."""
    position = record["position"]
    require(position in {"WR", "TE"}, "Stint input proof supports WR/TE only")
    proof = record.get("input_proof", {})
    require(proof.get("schema") == "audit-input-proof/v1", "Missing stint input proof")
    cfg = importlib.import_module(f"src.{position.lower()}.run_pipeline").CONFIG
    features, targets = cfg["get_feature_columns_fn"](), list(cfg["targets"])
    attention = get_attn_static_columns(features, cfg["attn_static_features"])
    history, opponent = (
        list(cfg["attn_history_stats"]),
        list(cfg.get("opp_attn_history_stats") or []),
    )
    require(
        set(features) >= CHANGED and not CHANGED.intersection([*attention, *history, *opponent]),
        "Declared family reach does not match current whitelists",
    )
    require(
        proof["feature_columns"] == features and proof["target_columns"] == targets,
        "Wrong production feature/target lists",
    )
    require(set(proof["prepared"]) == set(SPLITS), "Missing prepared split proofs")
    for split in SPLITS:
        part = proof["prepared"][split]
        require(_digest(part["row_keys_hash"]), "Missing prepared row identity")
        require(
            len(part["frame_columns"]) == len(set(part["frame_columns"]))
            and set(part["frame_columns"]) == set(part["frame_column_hashes"]),
            "Incomplete prepared frame schema",
        )
        require(
            set(features) <= set(part["frame_columns"])
            and all(_digest(v) for v in part["frame_column_hashes"].values()),
            "Missing prepared column fingerprints",
        )
        _array(part["X"], features)
        require(
            part["X"]["sha256"] == record["prepared_hashes"][f"X_{split}"],
            "Prepared X proof differs from original fingerprint",
        )
        require(set(part["y"]) == set(targets), "Incomplete prepared targets")
        for target in targets:
            _array(part["y"][target])
            require(
                part["y"][target]["sha256"] == record["prepared_hashes"][f"y_{split}/{target}"],
                "Prepared target proof differs from original fingerprint",
            )
    preprocessing = proof["preprocessing"]
    state = preprocessing["state"]
    require(preprocessing["sha256"] == _hash(state), "Invalid fitted preprocessing fingerprint")
    require(
        preprocessing["fields"]
        == {key: _hash(value) for key, value in state.items() if key != "fill_values"},
        "Incomplete preprocessing fields",
    )
    require(
        preprocessing["fill_values"]
        == {key: _hash(value) for key, value in state.get("fill_values", {}).items()},
        "Incomplete fitted imputation values",
    )
    require(set(proof["neural"]) == {"nn", "attn_nn"}, "Missing actual neural input evidence")
    for family in ("nn", "attn_nn"):
        neural = proof["neural"][family]
        columns = features if family == "nn" else attention
        require(
            neural["family"] == family
            and neural["feature_columns"] == columns
            and neural["target_columns"] == targets,
            "Wrong neural input feature order",
        )
        require(
            neural["history_columns"] == ([] if family == "nn" else history)
            and neural["opp_history_columns"] == ([] if family == "nn" else opponent),
            "Wrong neural history columns",
        )
        require(
            set(neural["scaled"]) == set(neural["actual"]) == set(SPLITS),
            "Missing actual neural input split",
        )
        for field in ("mean_", "var_", "scale_"):
            _array(neural["scaler"][field], columns)
        _array(neural["scaler"]["n_samples_seen_"])
        for split in SPLITS:
            _array(neural["scaled"][split], columns)
            actual = neural["actual"][split]
            require(
                actual["source"]
                == (
                    "predict_numpy arguments" if split == "test" else "GPUResidentBatcher._features"
                ),
                "Inputs were not observed on the actual production CUDA path",
            )
            expected = {"static"} if family == "nn" else {"static", "history", "mask"}
            if family == "attn_nn" and opponent:
                expected |= {"opp_history", "opp_mask"}
            require(
                set(actual["inputs"]) == expected and set(actual["targets"]) == set(targets),
                "Incomplete actual inputs/targets",
            )
            for name, array in actual["inputs"].items():
                names = (
                    columns
                    if name == "static"
                    else history
                    if name == "history"
                    else opponent
                    if name == "opp_history"
                    else None
                )
                _array(array, names)
                require(
                    array["shape"][0] == proof["prepared"][split]["X"]["shape"][0],
                    "Actual input row count differs",
                )
            for array in actual["targets"].values():
                _array(array)
    return proof


def verify_stint_pair(base, proposed, *, repeat=False):
    left, right = validate_input_proof(base), validate_input_proof(proposed)
    if repeat:
        require(left == right, "Repeated baseline input proof changed")
        return
    changed = set()
    for split in SPLITS:
        a, b = left["prepared"][split], right["prepared"][split]
        require(
            a["row_keys_hash"] == b["row_keys_hash"] and a["y"] == b["y"],
            "Changed prepared rows/targets",
        )
        changed |= _columns(
            a["frame_column_hashes"], b["frame_column_hashes"], CHANGED, f"prepared {split}"
        )
        _paired_array(a["X"], b["X"], left["feature_columns"], CHANGED, f"prepared X {split}")
    require(changed, "Stint candidate did not change its intended prepared features")
    a, b = left["preprocessing"], right["preprocessing"]
    require(a["fields"] == b["fields"], "Changed unrelated preprocessing")
    _columns(a["fill_values"], b["fill_values"], CHANGED, "fitted imputation")
    require(
        left["neural"]["attn_nn"] == right["neural"]["attn_nn"],
        "Attention inputs/scaler changed; control-family assumption is invalid",
    )
    a, b = left["neural"]["nn"], right["neural"]["nn"]
    require(
        a["scaler"]["n_samples_seen_"] == b["scaler"]["n_samples_seen_"],
        "Changed scaler sample count",
    )
    for field in ("mean_", "var_", "scale_"):
        _paired_array(
            a["scaler"][field],
            b["scaler"][field],
            a["feature_columns"],
            CHANGED,
            f"plain scaler {field}",
        )
    for split in SPLITS:
        _paired_array(
            a["scaled"][split],
            b["scaled"][split],
            a["feature_columns"],
            CHANGED,
            f"plain scaled {split}",
        )
        x, y = a["actual"][split], b["actual"][split]
        require(
            x["source"] == y["source"] and x["targets"] == y["targets"],
            "Changed actual target observations",
        )
        _paired_array(
            x["inputs"]["static"],
            y["inputs"]["static"],
            a["feature_columns"],
            CHANGED,
            f"plain actual {split}",
        )
