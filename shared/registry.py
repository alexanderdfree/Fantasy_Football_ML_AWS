"""Single source of truth for per-position dispatch.

Consumers: app.py (inference), batch/train.py (training), benchmark_nn.py
(local benchmark), tune_lgbm.py (LightGBM tuning), tests/_pipeline_e2e_utils.py.

Everything is lazily imported so loading this module is cheap.
"""
import importlib
from functools import lru_cache


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
        "config_var": "K_CONFIG",
        "accepts_dataframes": False,
        "cpu_only": True,
    },
    "DST": {
        "runner_module": "DST.run_dst_pipeline",
        "runner_fn": "run_dst_pipeline",
        "cv_runner_fn": None,
        "config_var": "DST_CONFIG",
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


@lru_cache(maxsize=None)
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
    return getattr(_import_runner_module(pos), _meta(pos)["config_var"])


def accepts_dataframes(pos: str) -> bool:
    return _meta(pos)["accepts_dataframes"]


def is_cpu_only(pos: str) -> bool:
    return _meta(pos)["cpu_only"]


# ---------------------------------------------------------------------------
# Inference spec — used by app.py to apply position-specific models.
# Lazy-loaded once per position; per-position modules are only imported when
# the position is first requested.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_inference_spec(pos: str) -> dict:
    if pos == "QB":
        from QB.qb_data import filter_to_qb
        from QB.qb_targets import compute_qb_targets, compute_qb_adjustment
        from QB.qb_features import (
            add_qb_specific_features, get_qb_feature_columns, fill_qb_nans,
        )
        from QB.qb_config import (
            QB_TARGETS, QB_SPECIFIC_FEATURES,
            QB_NN_BACKBONE_LAYERS, QB_NN_HEAD_HIDDEN, QB_NN_DROPOUT,
        )
        return {
            "targets": QB_TARGETS,
            "specific_features": QB_SPECIFIC_FEATURES,
            "filter_fn": filter_to_qb,
            "compute_targets_fn": compute_qb_targets,
            "add_features_fn": add_qb_specific_features,
            "fill_nans_fn": fill_qb_nans,
            "get_feature_columns_fn": get_qb_feature_columns,
            "compute_adjustment_fn": compute_qb_adjustment,
            "model_dir": "QB/outputs/models",
            "nn_file": "qb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=QB_NN_BACKBONE_LAYERS,
                head_hidden=QB_NN_HEAD_HIDDEN,
                dropout=QB_NN_DROPOUT,
            ),
        }
    if pos == "RB":
        from RB.rb_data import filter_to_rb
        from RB.rb_targets import compute_rb_targets, compute_fumble_adjustment
        from RB.rb_features import (
            add_rb_specific_features, get_rb_feature_columns, fill_rb_nans,
        )
        from RB.rb_config import (
            RB_TARGETS, RB_SPECIFIC_FEATURES,
            RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_HEAD_HIDDEN_OVERRIDES,
            RB_NN_DROPOUT,
        )
        return {
            "targets": RB_TARGETS,
            "specific_features": RB_SPECIFIC_FEATURES,
            "filter_fn": filter_to_rb,
            "compute_targets_fn": compute_rb_targets,
            "add_features_fn": add_rb_specific_features,
            "fill_nans_fn": fill_rb_nans,
            "get_feature_columns_fn": get_rb_feature_columns,
            "compute_adjustment_fn": compute_fumble_adjustment,
            "model_dir": "RB/outputs/models",
            "nn_file": "rb_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=RB_NN_BACKBONE_LAYERS,
                head_hidden=RB_NN_HEAD_HIDDEN,
                dropout=RB_NN_DROPOUT,
                head_hidden_overrides=RB_NN_HEAD_HIDDEN_OVERRIDES,
            ),
        }
    if pos == "WR":
        from WR.wr_data import filter_to_wr
        from WR.wr_targets import compute_wr_targets, compute_wr_fumble_adjustment
        from WR.wr_features import (
            add_wr_specific_features, get_wr_feature_columns, fill_wr_nans,
        )
        from WR.wr_config import (
            WR_TARGETS, WR_SPECIFIC_FEATURES,
            WR_NN_BACKBONE_LAYERS, WR_NN_HEAD_HIDDEN, WR_NN_DROPOUT,
        )
        return {
            "targets": WR_TARGETS,
            "specific_features": WR_SPECIFIC_FEATURES,
            "filter_fn": filter_to_wr,
            "compute_targets_fn": compute_wr_targets,
            "add_features_fn": add_wr_specific_features,
            "fill_nans_fn": fill_wr_nans,
            "get_feature_columns_fn": get_wr_feature_columns,
            "compute_adjustment_fn": compute_wr_fumble_adjustment,
            "model_dir": "WR/outputs/models",
            "nn_file": "wr_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=WR_NN_BACKBONE_LAYERS,
                head_hidden=WR_NN_HEAD_HIDDEN,
                dropout=WR_NN_DROPOUT,
            ),
        }
    if pos == "TE":
        from TE.te_data import filter_to_te
        from TE.te_targets import compute_te_targets, compute_te_fumble_adjustment
        from TE.te_features import (
            add_te_specific_features, get_te_feature_columns, fill_te_nans,
        )
        from TE.te_config import (
            TE_TARGETS, TE_SPECIFIC_FEATURES,
            TE_NN_BACKBONE_LAYERS, TE_NN_HEAD_HIDDEN, TE_NN_HEAD_HIDDEN_OVERRIDES,
            TE_NN_DROPOUT,
        )
        return {
            "targets": TE_TARGETS,
            "specific_features": TE_SPECIFIC_FEATURES,
            "filter_fn": filter_to_te,
            "compute_targets_fn": compute_te_targets,
            "add_features_fn": add_te_specific_features,
            "fill_nans_fn": fill_te_nans,
            "get_feature_columns_fn": get_te_feature_columns,
            "compute_adjustment_fn": compute_te_fumble_adjustment,
            "model_dir": "TE/outputs/models",
            "nn_file": "te_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=TE_NN_BACKBONE_LAYERS,
                head_hidden=TE_NN_HEAD_HIDDEN,
                dropout=TE_NN_DROPOUT,
                head_hidden_overrides=TE_NN_HEAD_HIDDEN_OVERRIDES,
            ),
        }
    if pos == "K":
        from K.k_data import filter_to_k
        from K.k_targets import compute_k_targets, compute_k_miss_adjustment
        from K.k_features import (
            add_k_specific_features, get_k_feature_columns, fill_k_nans,
        )
        from K.k_config import (
            K_TARGETS, K_SPECIFIC_FEATURES,
            K_NN_BACKBONE_LAYERS, K_NN_HEAD_HIDDEN, K_NN_DROPOUT,
        )
        return {
            "targets": K_TARGETS,
            "specific_features": K_SPECIFIC_FEATURES,
            "filter_fn": filter_to_k,
            "compute_targets_fn": compute_k_targets,
            "add_features_fn": add_k_specific_features,
            "fill_nans_fn": fill_k_nans,
            "get_feature_columns_fn": get_k_feature_columns,
            "compute_adjustment_fn": compute_k_miss_adjustment,
            "model_dir": "K/outputs/models",
            "nn_file": "k_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=K_NN_BACKBONE_LAYERS,
                head_hidden=K_NN_HEAD_HIDDEN,
                dropout=K_NN_DROPOUT,
            ),
        }
    if pos == "DST":
        from DST.dst_data import filter_to_dst
        from DST.dst_targets import compute_dst_targets, compute_dst_adjustment
        from DST.dst_features import (
            add_dst_specific_features, get_dst_feature_columns, fill_dst_nans,
        )
        from DST.dst_config import (
            DST_TARGETS, DST_SPECIFIC_FEATURES,
            DST_NN_BACKBONE_LAYERS, DST_NN_HEAD_HIDDEN, DST_NN_HEAD_HIDDEN_OVERRIDES,
            DST_NN_DROPOUT, DST_NN_NON_NEGATIVE_TARGETS,
        )
        return {
            "targets": DST_TARGETS,
            "specific_features": DST_SPECIFIC_FEATURES,
            "filter_fn": filter_to_dst,
            "compute_targets_fn": compute_dst_targets,
            "add_features_fn": add_dst_specific_features,
            "fill_nans_fn": fill_dst_nans,
            "get_feature_columns_fn": get_dst_feature_columns,
            "compute_adjustment_fn": compute_dst_adjustment,
            "model_dir": "DST/outputs/models",
            "nn_file": "dst_multihead_nn.pt",
            "nn_kwargs": dict(
                backbone_layers=DST_NN_BACKBONE_LAYERS,
                head_hidden=DST_NN_HEAD_HIDDEN,
                dropout=DST_NN_DROPOUT,
                head_hidden_overrides=DST_NN_HEAD_HIDDEN_OVERRIDES,
                non_negative_targets=DST_NN_NON_NEGATIVE_TARGETS,
            ),
        }
    raise ValueError(f"Unknown position: {pos}")


class _LazyInferenceRegistry:
    """Dict-like view over `get_inference_spec` — `registry[pos]` triggers lazy import."""

    def __getitem__(self, pos):
        return get_inference_spec(pos)

    def __contains__(self, pos):
        return pos in _POSITION_META


INFERENCE_REGISTRY = _LazyInferenceRegistry()
