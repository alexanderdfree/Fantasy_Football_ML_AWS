"""Single source of truth for per-position dispatch.

Consumers: app.py (inference), batch/train.py (training), benchmark.py
(local benchmark), tune_lgbm.py (LightGBM tuning), tests/_pipeline_e2e_utils.py.

Everything is lazily imported so loading this module is cheap.
"""

import importlib
from functools import cache

from shared.aggregate_targets import aggregate_fn_for

_POSITION_META = {
    "QB": {
        "runner_module": "QB.run_qb_pipeline",
        "runner_fn": "run_qb_pipeline",
        "cv_runner_fn": "run_qb_cv_pipeline",
        "config_var": "QB_CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "RB": {
        "runner_module": "RB.run_rb_pipeline",
        "runner_fn": "run_rb_pipeline",
        "cv_runner_fn": "run_rb_cv_pipeline",
        "config_var": "RB_CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "WR": {
        "runner_module": "WR.run_wr_pipeline",
        "runner_fn": "run_wr_pipeline",
        "cv_runner_fn": "run_wr_cv_pipeline",
        "config_var": "WR_CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "TE": {
        "runner_module": "TE.run_te_pipeline",
        "runner_fn": "run_te_pipeline",
        "cv_runner_fn": None,
        "config_var": "TE_CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "K": {
        "runner_module": "K.run_k_pipeline",
        "runner_fn": "run_k_pipeline",
        "cv_runner_fn": None,
        # K_CONFIG is built inside run_k_pipeline() (local var), not exported.
        "config_var": None,
        "accepts_dataframes": False,
        "cpu_only": True,
    },
    "DST": {
        "runner_module": "DST.run_dst_pipeline",
        "runner_fn": "run_dst_pipeline",
        "cv_runner_fn": None,
        # DST_CONFIG is built inside run_dst_pipeline() (local var), not exported.
        "config_var": None,
        "accepts_dataframes": False,
        "cpu_only": True,
    },
}

ALL_POSITIONS = list(_POSITION_META.keys())
CPU_ONLY_POSITIONS = {p for p, m in _POSITION_META.items() if m["cpu_only"]}


def _meta(pos: str) -> dict:
    if pos not in _POSITION_META:
        raise ValueError(f"Unknown position: {pos}")
    return _POSITION_META[pos]


@cache
def _import_runner_module(pos: str):
    return importlib.import_module(_meta(pos)["runner_module"])


def get_runner(pos: str):
    return getattr(_import_runner_module(pos), _meta(pos)["runner_fn"])


def get_cv_runner(pos: str):
    fn_name = _meta(pos)["cv_runner_fn"]
    if fn_name is None:
        raise ValueError(f"CV pipeline not implemented for position: {pos}")
    return getattr(_import_runner_module(pos), fn_name)


def get_config(pos: str) -> dict:
    var_name = _meta(pos)["config_var"]
    if var_name is None:
        raise ValueError(f"Module-level config dict not exported for position: {pos}")
    return getattr(_import_runner_module(pos), var_name)


def accepts_dataframes(pos: str) -> bool:
    return _meta(pos)["accepts_dataframes"]


def is_cpu_only(pos: str) -> bool:
    return _meta(pos)["cpu_only"]


# ---------------------------------------------------------------------------
# Inference spec — used by app.py to apply position-specific models.
# Lazy-loaded once per position; per-position modules are only imported when
# the position is first requested.
# ---------------------------------------------------------------------------


def _attn_kwargs_static(cfg, prefix: str) -> dict:
    """Pull all MultiHeadNetWithHistory kwargs from a cfg module EXCEPT the
    runtime-dependent ``static_dim`` / ``game_dim`` / ``target_names`` — those are
    filled in by the caller at inference time once the feature list is known.

    Missing config attributes fall back to the MultiHeadNetWithHistory defaults
    so positions that don't set every knob still work.
    """

    def g(name, default=None):
        return getattr(cfg, f"{prefix}_{name}", default)

    # Gated TD target names. New configs set ``{POS}_GATED_TD_TARGETS`` (list,
    # possibly multi-head for RB). Legacy configs set ``{POS}_GATED_TD_TARGET``
    # (single string). Thread both through; MultiHeadNetWithHistory normalizes
    # the legacy string into a one-element list.
    gated_td_target = g("GATED_TD_TARGET", None)
    gated_td_targets = g("GATED_TD_TARGETS", None)
    # Default for legacy callers: keep "td_points" so pre-migration configs
    # still build a gate on the old combined-TD head.
    if gated_td_targets is None and gated_td_target is None:
        gated_td_target = "td_points"

    kwargs = dict(
        backbone_layers=list(g("NN_BACKBONE_LAYERS", [])),
        d_model=g("ATTN_D_MODEL", 32),
        n_attn_heads=g("ATTN_N_HEADS", 2),
        head_hidden=g("NN_HEAD_HIDDEN", 32),
        dropout=g("NN_DROPOUT", 0.3),
        project_kv=g("ATTN_PROJECT_KV", False),
        use_positional_encoding=g("ATTN_POSITIONAL_ENCODING", False),
        max_seq_len=g("ATTN_MAX_SEQ_LEN", 17),
        use_gated_fusion=g("ATTN_GATED_FUSION", False),
        attn_dropout=g("ATTN_DROPOUT", 0.0),
        encoder_hidden_dim=g("ATTN_ENCODER_HIDDEN_DIM", 0),
        gated_td=g("ATTN_GATED_TD", False),
        td_gate_hidden=g("ATTN_TD_GATE_HIDDEN", 16),
        gated_td_target=gated_td_target,
        gated_td_targets=list(gated_td_targets) if gated_td_targets is not None else None,
    )
    head_hidden_overrides = g("NN_HEAD_HIDDEN_OVERRIDES", None)
    if head_hidden_overrides:
        kwargs["head_hidden_overrides"] = dict(head_hidden_overrides)
    non_negative = g("NN_NON_NEGATIVE_TARGETS", None)
    if non_negative is not None:
        kwargs["non_negative_targets"] = set(non_negative)
    return kwargs


@cache
def get_inference_spec(pos: str) -> dict:
    if pos == "QB":
        import QB.qb_config as qb_cfg
        from QB.qb_config import (
            QB_NN_BACKBONE_LAYERS,
            QB_NN_DROPOUT,
            QB_NN_HEAD_HIDDEN,
            QB_SPECIFIC_FEATURES,
            QB_TARGETS,
        )
        from QB.qb_data import filter_to_qb
        from QB.qb_features import (
            add_qb_specific_features,
            fill_qb_nans,
            get_qb_feature_columns,
        )
        from QB.qb_targets import compute_qb_targets

        return {
            "targets": QB_TARGETS,
            "specific_features": QB_SPECIFIC_FEATURES,
            "filter_fn": filter_to_qb,
            "compute_targets_fn": compute_qb_targets,
            "add_features_fn": add_qb_specific_features,
            "fill_nans_fn": fill_qb_nans,
            "get_feature_columns_fn": get_qb_feature_columns,
            "aggregate_fn": aggregate_fn_for("QB"),
            "model_dir": "QB/outputs/models",
            "nn_file": "qb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=QB_NN_BACKBONE_LAYERS,
                head_hidden=QB_NN_HEAD_HIDDEN,
                dropout=QB_NN_DROPOUT,
            ),
            "train_attention_nn": bool(getattr(qb_cfg, "QB_TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "qb_attention_nn.pt",
            "attn_history_stats": list(getattr(qb_cfg, "QB_ATTN_HISTORY_STATS", []) or []),
            "attn_max_seq_len": getattr(qb_cfg, "QB_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(qb_cfg, "QB"),
            "train_lightgbm": bool(getattr(qb_cfg, "QB_TRAIN_LIGHTGBM", False)),
        }
    if pos == "RB":
        import RB.rb_config as rb_cfg
        from RB.rb_config import (
            RB_NN_BACKBONE_LAYERS,
            RB_NN_DROPOUT,
            RB_NN_HEAD_HIDDEN,
            RB_NN_HEAD_HIDDEN_OVERRIDES,
            RB_SPECIFIC_FEATURES,
            RB_TARGETS,
        )
        from RB.rb_data import filter_to_rb
        from RB.rb_features import (
            add_rb_specific_features,
            fill_rb_nans,
            get_rb_feature_columns,
        )
        from RB.rb_targets import compute_rb_targets

        return {
            "targets": RB_TARGETS,
            "specific_features": RB_SPECIFIC_FEATURES,
            "filter_fn": filter_to_rb,
            "compute_targets_fn": compute_rb_targets,
            "add_features_fn": add_rb_specific_features,
            "fill_nans_fn": fill_rb_nans,
            "get_feature_columns_fn": get_rb_feature_columns,
            "aggregate_fn": aggregate_fn_for("RB"),
            "model_dir": "RB/outputs/models",
            "nn_file": "rb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=RB_NN_BACKBONE_LAYERS,
                head_hidden=RB_NN_HEAD_HIDDEN,
                dropout=RB_NN_DROPOUT,
                head_hidden_overrides=RB_NN_HEAD_HIDDEN_OVERRIDES,
            ),
            "train_attention_nn": bool(getattr(rb_cfg, "RB_TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "rb_attention_nn.pt",
            "attn_history_stats": list(getattr(rb_cfg, "RB_ATTN_HISTORY_STATS", []) or []),
            "attn_max_seq_len": getattr(rb_cfg, "RB_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(rb_cfg, "RB"),
            "train_lightgbm": bool(getattr(rb_cfg, "RB_TRAIN_LIGHTGBM", False)),
        }
    if pos == "WR":
        import WR.wr_config as wr_cfg
        from WR.wr_config import (
            WR_NN_BACKBONE_LAYERS,
            WR_NN_DROPOUT,
            WR_NN_HEAD_HIDDEN,
            WR_SPECIFIC_FEATURES,
            WR_TARGETS,
        )
        from WR.wr_data import filter_to_wr
        from WR.wr_features import (
            add_wr_specific_features,
            fill_wr_nans,
            get_wr_feature_columns,
        )
        from WR.wr_targets import compute_wr_targets

        return {
            "targets": WR_TARGETS,
            "specific_features": WR_SPECIFIC_FEATURES,
            "filter_fn": filter_to_wr,
            "compute_targets_fn": compute_wr_targets,
            "add_features_fn": add_wr_specific_features,
            "fill_nans_fn": fill_wr_nans,
            "get_feature_columns_fn": get_wr_feature_columns,
            "aggregate_fn": aggregate_fn_for("WR"),
            "model_dir": "WR/outputs/models",
            "nn_file": "wr_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=WR_NN_BACKBONE_LAYERS,
                head_hidden=WR_NN_HEAD_HIDDEN,
                dropout=WR_NN_DROPOUT,
            ),
            "train_attention_nn": bool(getattr(wr_cfg, "WR_TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "wr_attention_nn.pt",
            "attn_history_stats": list(getattr(wr_cfg, "WR_ATTN_HISTORY_STATS", []) or []),
            "attn_max_seq_len": getattr(wr_cfg, "WR_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(wr_cfg, "WR"),
            "train_lightgbm": bool(getattr(wr_cfg, "WR_TRAIN_LIGHTGBM", False)),
        }
    if pos == "TE":
        import TE.te_config as te_cfg
        from TE.te_config import (
            TE_NN_BACKBONE_LAYERS,
            TE_NN_DROPOUT,
            TE_NN_HEAD_HIDDEN,
            TE_NN_HEAD_HIDDEN_OVERRIDES,
            TE_SPECIFIC_FEATURES,
            TE_TARGETS,
        )
        from TE.te_data import filter_to_te
        from TE.te_features import (
            add_te_specific_features,
            fill_te_nans,
            get_te_feature_columns,
        )
        from TE.te_targets import compute_te_targets

        return {
            "targets": TE_TARGETS,
            "specific_features": TE_SPECIFIC_FEATURES,
            "filter_fn": filter_to_te,
            "compute_targets_fn": compute_te_targets,
            "add_features_fn": add_te_specific_features,
            "fill_nans_fn": fill_te_nans,
            "get_feature_columns_fn": get_te_feature_columns,
            "aggregate_fn": aggregate_fn_for("TE"),
            "model_dir": "TE/outputs/models",
            "nn_file": "te_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=TE_NN_BACKBONE_LAYERS,
                head_hidden=TE_NN_HEAD_HIDDEN,
                dropout=TE_NN_DROPOUT,
                head_hidden_overrides=TE_NN_HEAD_HIDDEN_OVERRIDES,
            ),
            "train_attention_nn": bool(getattr(te_cfg, "TE_TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "te_attention_nn.pt",
            "attn_history_stats": list(getattr(te_cfg, "TE_ATTN_HISTORY_STATS", []) or []),
            "attn_max_seq_len": getattr(te_cfg, "TE_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(te_cfg, "TE"),
            "train_lightgbm": bool(getattr(te_cfg, "TE_TRAIN_LIGHTGBM", False)),
        }
    if pos == "K":
        import K.k_config as k_cfg
        from K.k_config import (
            K_NN_BACKBONE_LAYERS,
            K_NN_DROPOUT,
            K_NN_HEAD_HIDDEN,
            K_NN_NON_NEGATIVE_TARGETS,
            K_SPECIFIC_FEATURES,
            K_TARGETS,
        )
        from K.k_data import filter_to_k
        from K.k_features import (
            add_k_specific_features,
            fill_k_nans,
            get_k_feature_columns,
        )
        from K.k_targets import compute_k_targets

        return {
            "targets": K_TARGETS,
            "specific_features": K_SPECIFIC_FEATURES,
            "filter_fn": filter_to_k,
            "compute_targets_fn": compute_k_targets,
            "add_features_fn": add_k_specific_features,
            "fill_nans_fn": fill_k_nans,
            "get_feature_columns_fn": get_k_feature_columns,
            "compute_adjustment_fn": None,
            # Sign vector for the 4 K heads when aggregating to fantasy points:
            # fg_yard_points and pat_points add, fg_misses and xp_misses subtract.
            "target_signs": {
                "fg_yard_points": 1.0,
                "pat_points": 1.0,
                "fg_misses": -1.0,
                "xp_misses": -1.0,
            },
            "model_dir": "K/outputs/models",
            "nn_file": "k_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=K_NN_BACKBONE_LAYERS,
                head_hidden=K_NN_HEAD_HIDDEN,
                dropout=K_NN_DROPOUT,
                non_negative_targets=K_NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(k_cfg, "K_TRAIN_ATTENTION_NN", False)),
            "train_lightgbm": bool(getattr(k_cfg, "K_TRAIN_LIGHTGBM", False)),
        }
    if pos == "DST":
        import DST.dst_config as dst_cfg
        from DST.dst_config import (
            DST_NN_BACKBONE_LAYERS,
            DST_NN_DROPOUT,
            DST_NN_HEAD_HIDDEN,
            DST_NN_HEAD_HIDDEN_OVERRIDES,
            DST_NN_NON_NEGATIVE_TARGETS,
            DST_SPECIFIC_FEATURES,
            DST_TARGETS,
        )
        from DST.dst_data import filter_to_dst
        from DST.dst_features import (
            add_dst_specific_features,
            fill_dst_nans,
            get_dst_feature_columns,
        )
        from DST.dst_targets import compute_dst_targets

        return {
            "targets": DST_TARGETS,
            "specific_features": DST_SPECIFIC_FEATURES,
            "filter_fn": filter_to_dst,
            "compute_targets_fn": compute_dst_targets,
            "add_features_fn": add_dst_specific_features,
            "fill_nans_fn": fill_dst_nans,
            "get_feature_columns_fn": get_dst_feature_columns,
            "compute_adjustment_fn": None,
            "model_dir": "DST/outputs/models",
            "nn_file": "dst_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=DST_NN_BACKBONE_LAYERS,
                head_hidden=DST_NN_HEAD_HIDDEN,
                dropout=DST_NN_DROPOUT,
                head_hidden_overrides=DST_NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=DST_NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(dst_cfg, "DST_TRAIN_ATTENTION_NN", False)),
            "train_lightgbm": bool(getattr(dst_cfg, "DST_TRAIN_LIGHTGBM", False)),
        }
    raise ValueError(f"Unknown position: {pos}")


class _LazyInferenceRegistry:
    """Dict-like view over `get_inference_spec` — `registry[pos]` triggers lazy import."""

    def __getitem__(self, pos):
        return get_inference_spec(pos)

    def __contains__(self, pos):
        return pos in _POSITION_META


INFERENCE_REGISTRY = _LazyInferenceRegistry()
