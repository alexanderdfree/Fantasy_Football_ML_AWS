"""Fake checkpoints whose input schemas are a position's real production whitelists."""

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.prediction.bundle import write_bundle
from src.prediction.predictor import legacy_schema
from src.shared.artifact_integrity import wrap_state_dict, write_scaler_meta
from src.shared.models import RidgeMultiTarget
from src.shared.neural_net import MultiHeadNet, MultiHeadNetWithHistory
from src.shared.registry import get_inference_spec


def fake_artifacts(
    directory,
    position,
    families=("attn_nn",),
    *,
    seed=33,
    extra_static=(),
    data_ids=None,
    opp_stats=(),
):
    """Fake checkpoints whose input schemas are the position's real production whitelists."""
    directory.mkdir(parents=True, exist_ok=True)
    cfg = get_inference_spec(position)
    cfg["opp_attn_history_stats"] = list(opp_stats)
    features = list(cfg["get_feature_columns_fn"]())
    targets = list(cfg["targets"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for family in families:
        data_id = (data_ids or {}).get(family, "fake")
        if family == "ridge":
            model = RidgeMultiTarget(targets, alpha=1.0)
            X = rng.normal(size=(24, len(features))).astype(np.float32)
            model.fit(X, {t: rng.normal(size=24).astype(np.float32) for t in targets})
            model.save(str(directory))
            write_bundle(
                directory,
                position,
                family,
                cfg,
                features,
                model,
                preprocessing={"clip": [-4.0, 4.0]},
                data_id=data_id,
            )
            continue
        columns = list(legacy_schema(cfg, family).features) + list(extra_static)
        if family == "attn_nn":
            model = MultiHeadNetWithHistory(
                len(columns),
                len(cfg["attn_history_stats"]),
                targets,
                opp_game_dim=len(opp_stats) or None,
                **cfg["attn_nn_kwargs_static"],
            )
        else:
            model = MultiHeadNet(len(columns), targets, **cfg["nn_kwargs"])
        scaler = StandardScaler().fit(rng.normal(size=(8, len(columns))).astype(np.float32))
        stem = "attention_nn" if family == "attn_nn" else "nn"
        weight_stem = "attention_nn" if family == "attn_nn" else "multihead_nn"
        joblib.dump(scaler, directory / f"{stem}_scaler.pkl")
        write_scaler_meta(directory / f"{stem}_scaler_meta.json", columns, targets)
        torch.save(
            wrap_state_dict(model.state_dict(), columns, targets),
            directory / f"{position.lower()}_{weight_stem}.pt",
        )
        write_bundle(
            directory,
            position,
            family,
            cfg,
            columns,
            model,
            preprocessing={"clip": [-4.0, 4.0]},
            data_id=data_id,
        )
    return cfg
