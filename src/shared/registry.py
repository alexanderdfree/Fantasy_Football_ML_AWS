"""Single source of truth for per-position dispatch.

Consumers: app.py (inference), src/batch/train.py (training), benchmark.py
(local benchmark), tune_lgbm.py (LightGBM tuning), tests/_pipeline_e2e_utils.py.

Everything is lazily imported so loading this module is cheap.
"""

import importlib
from functools import cache

from src.shared.aggregate_targets import aggregate_fn_for

_POSITION_META = {
    "QB": {
        "runner_module": "src.qb.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": "run_cv",
        "config_var": "CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "RB": {
        "runner_module": "src.rb.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": "run_cv",
        "config_var": "CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "WR": {
        "runner_module": "src.wr.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": "run_cv",
        "config_var": "CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "TE": {
        "runner_module": "src.te.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": None,
        "config_var": "CONFIG",
        "accepts_dataframes": True,
        "cpu_only": False,
    },
    "K": {
        "runner_module": "src.k.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": None,
        "config_var": "CONFIG",
        "accepts_dataframes": False,
        "cpu_only": True,
    },
    "DST": {
        "runner_module": "src.dst.run_pipeline",
        "runner_fn": "run",
        "cv_runner_fn": None,
        "config_var": "CONFIG",
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


def _attn_kwargs_static(cfg) -> dict:
    """Pull all MultiHeadNetWithHistory kwargs from a cfg module EXCEPT the
    runtime-dependent ``static_dim`` / ``game_dim`` / ``target_names`` — those are
    filled in by the caller at inference time once the feature list is known.

    Missing config attributes fall back to the MultiHeadNetWithHistory defaults
    so positions that don't set every knob still work. Pre-#154 this looked
    up ``f"{POS}_NAME"``; the rename dropped that prefix from every per-position
    config module, so we now read bare attribute names.
    """

    def g(name, default=None):
        return getattr(cfg, name, default)

    gated_targets = g("GATED_TARGETS", None)

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
        gated=g("ATTN_GATED", False),
        gate_hidden=g("ATTN_GATE_HIDDEN", 16),
        gated_targets=list(gated_targets) if gated_targets is not None else None,
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
        import src.qb.config as qb_cfg
        from src.qb.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.qb.data import filter_to_position
        from src.qb.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.qb.targets import compute_targets

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "aggregate_fn": aggregate_fn_for("QB"),
            "model_dir": "src/qb/outputs/models",
            "nn_file": "qb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(qb_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "qb_attention_nn.pt",
            "attn_history_stats": list(getattr(qb_cfg, "ATTN_HISTORY_STATS", []) or []),
            "attn_static_features": list(getattr(qb_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_max_seq_len": getattr(qb_cfg, "ATTN_MAX_SEQ_LEN", 17),
            "opp_attn_history_stats": list(getattr(qb_cfg, "OPP_ATTN_HISTORY_STATS", []) or []),
            "opp_attn_max_seq_len": getattr(qb_cfg, "OPP_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(qb_cfg),
            "train_lightgbm": bool(getattr(qb_cfg, "TRAIN_LIGHTGBM", False)),
        }
    if pos == "RB":
        import src.rb.config as rb_cfg
        from src.rb.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_HEAD_HIDDEN_OVERRIDES,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.rb.data import filter_to_position
        from src.rb.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.rb.targets import compute_targets

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "aggregate_fn": aggregate_fn_for("RB"),
            "model_dir": "src/rb/outputs/models",
            "nn_file": "rb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                head_hidden_overrides=NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(rb_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "rb_attention_nn.pt",
            "attn_history_stats": list(getattr(rb_cfg, "ATTN_HISTORY_STATS", []) or []),
            "attn_static_features": list(getattr(rb_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_max_seq_len": getattr(rb_cfg, "ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(rb_cfg),
            "train_lightgbm": bool(getattr(rb_cfg, "TRAIN_LIGHTGBM", False)),
        }
    if pos == "WR":
        import src.wr.config as wr_cfg
        from src.wr.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_HEAD_HIDDEN_OVERRIDES,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.wr.data import filter_to_position
        from src.wr.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.wr.targets import compute_targets

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "aggregate_fn": aggregate_fn_for("WR"),
            "model_dir": "src/wr/outputs/models",
            "nn_file": "wr_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                head_hidden_overrides=NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(wr_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "wr_attention_nn.pt",
            "attn_history_stats": list(getattr(wr_cfg, "ATTN_HISTORY_STATS", []) or []),
            "attn_static_features": list(getattr(wr_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_max_seq_len": getattr(wr_cfg, "ATTN_MAX_SEQ_LEN", 17),
            "opp_attn_history_stats": list(getattr(wr_cfg, "OPP_ATTN_HISTORY_STATS", []) or []),
            "opp_attn_max_seq_len": getattr(wr_cfg, "OPP_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(wr_cfg),
            "train_lightgbm": bool(getattr(wr_cfg, "TRAIN_LIGHTGBM", False)),
        }
    if pos == "TE":
        import src.te.config as te_cfg
        from src.te.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_HEAD_HIDDEN_OVERRIDES,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.te.data import filter_to_position
        from src.te.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.te.targets import compute_targets

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "aggregate_fn": aggregate_fn_for("TE"),
            "model_dir": "src/te/outputs/models",
            "nn_file": "te_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                head_hidden_overrides=NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(te_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "te_attention_nn.pt",
            "attn_history_stats": list(getattr(te_cfg, "ATTN_HISTORY_STATS", []) or []),
            "attn_static_features": list(getattr(te_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_max_seq_len": getattr(te_cfg, "ATTN_MAX_SEQ_LEN", 17),
            "opp_attn_history_stats": list(getattr(te_cfg, "OPP_ATTN_HISTORY_STATS", []) or []),
            "opp_attn_max_seq_len": getattr(te_cfg, "OPP_ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(te_cfg),
            "train_lightgbm": bool(getattr(te_cfg, "TRAIN_LIGHTGBM", False)),
        }
    if pos == "K":
        import src.k.config as k_cfg
        from src.k.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.k.data import filter_to_position
        from src.k.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.k.targets import compute_targets

        # K attention NN uses nested kick history (inner: per-game kicks,
        # outer: per-target across prior games). Kwargs mirror those passed
        # into MultiHeadNetWithNestedHistory by src/k/run_pipeline.py.
        k_max_games = getattr(k_cfg, "ATTN_MAX_GAMES", 17)
        k_attn_nn_kwargs_static = dict(
            backbone_layers=list(NN_BACKBONE_LAYERS),
            d_kick=getattr(k_cfg, "ATTN_KICK_DIM", 16),
            d_model=getattr(k_cfg, "ATTN_D_MODEL", 32),
            n_attn_heads=getattr(k_cfg, "ATTN_N_HEADS", 2),
            head_hidden=NN_HEAD_HIDDEN,
            dropout=NN_DROPOUT,
            non_negative_targets=set(NN_NON_NEGATIVE_TARGETS),
            project_kv=getattr(k_cfg, "ATTN_PROJECT_KV", False),
            use_positional_encoding=getattr(k_cfg, "ATTN_POSITIONAL_ENCODING", False),
            max_games=k_max_games,
            attn_dropout=getattr(k_cfg, "ATTN_DROPOUT", 0.0),
            encoder_hidden_dim=getattr(k_cfg, "ATTN_ENCODER_HIDDEN_DIM", 0),
        )

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "compute_adjustment_fn": None,
            # Sign vector for the 4 K heads when aggregating to fantasy points:
            # fg_yard_points and pat_points add, fg_misses and xp_misses subtract.
            "target_signs": {
                "fg_yard_points": 1.0,
                "pat_points": 1.0,
                "fg_misses": -1.0,
                "xp_misses": -1.0,
            },
            "model_dir": "src/k/outputs/models",
            "nn_file": "k_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(k_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "k_attention_nn.pt",
            "attn_history_structure": "nested",
            "attn_static_from_df": True,
            "attn_static_features": list(getattr(k_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_kick_stats": list(getattr(k_cfg, "ATTN_KICK_STATS", []) or []),
            "attn_max_games": k_max_games,
            "attn_max_kicks_per_game": getattr(k_cfg, "ATTN_MAX_KICKS_PER_GAME", 10),
            "attn_nn_kwargs_static": k_attn_nn_kwargs_static,
            "train_lightgbm": bool(getattr(k_cfg, "TRAIN_LIGHTGBM", False)),
        }
    if pos == "DST":
        import src.dst.config as dst_cfg
        from src.dst.config import (
            NN_BACKBONE_LAYERS,
            NN_DROPOUT,
            NN_HEAD_HIDDEN,
            NN_HEAD_HIDDEN_OVERRIDES,
            NN_NON_NEGATIVE_TARGETS,
            SPECIFIC_FEATURES,
            TARGETS,
        )
        from src.dst.data import filter_to_position
        from src.dst.features import (
            add_specific_features,
            fill_nans,
            get_feature_columns,
        )
        from src.dst.targets import compute_targets

        return {
            "targets": TARGETS,
            "specific_features": SPECIFIC_FEATURES,
            "filter_fn": filter_to_position,
            "compute_targets_fn": compute_targets,
            "add_features_fn": add_specific_features,
            "fill_nans_fn": fill_nans,
            "get_feature_columns_fn": get_feature_columns,
            "compute_adjustment_fn": None,
            "aggregate_fn": aggregate_fn_for("DST"),
            "model_dir": "src/dst/outputs/models",
            "nn_file": "dst_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=NN_BACKBONE_LAYERS,
                head_hidden=NN_HEAD_HIDDEN,
                dropout=NN_DROPOUT,
                head_hidden_overrides=NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=NN_NON_NEGATIVE_TARGETS,
            ),
            "train_attention_nn": bool(getattr(dst_cfg, "TRAIN_ATTENTION_NN", False)),
            "attn_nn_file": "dst_attention_nn.pt",
            "attn_history_stats": list(getattr(dst_cfg, "ATTN_HISTORY_STATS", []) or []),
            "attn_static_features": list(getattr(dst_cfg, "ATTN_STATIC_FEATURES", []) or []),
            "attn_max_seq_len": getattr(dst_cfg, "ATTN_MAX_SEQ_LEN", 17),
            "attn_nn_kwargs_static": _attn_kwargs_static(dst_cfg),
            "train_lightgbm": bool(getattr(dst_cfg, "TRAIN_LIGHTGBM", False)),
        }
    raise ValueError(f"Unknown position: {pos}")


class _LazyInferenceRegistry:
    """Dict-like view over `get_inference_spec` — `registry[pos]` triggers lazy import."""

    def __getitem__(self, pos):
        return get_inference_spec(pos)

    def __contains__(self, pos):
        return pos in _POSITION_META


INFERENCE_REGISTRY = _LazyInferenceRegistry()
