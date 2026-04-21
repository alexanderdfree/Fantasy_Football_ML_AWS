"""Shared helpers for E2E and reproducibility tests at ``tests/``.

Consolidates three pieces of glue that both test files need:

1. ``build_tiny_config(position)`` — assemble a shrunk pipeline config
   (1 epoch, 2-layer x 8-unit NN, no attention/LightGBM) from the
   position's real config module. If the position exposes a
   ``{POS}_CONFIG_TINY`` symbol we splice it onto the callables; otherwise
   we shrink the full production config inline.

2. ``load_tiny_splits(position)`` — return ``(train, val, test)`` frames
   sized for a <20s pipeline round-trip. For player-level positions (QB,
   RB, WR, TE) this slices the pre-engineered parquets to the top-N
   players by game count. For K and DST it rebuilds the per-position
   dataset via their loaders and takes a recent slice.

3. ``run_pipeline_in_tmp(position, cfg, splits, tmp_path, seed)`` —
   chdir into ``tmp_path`` with a symlink to ``data/`` so the pipeline
   finds schedule parquets, runs ``run_pipeline``, and restores cwd.
   Required because the pipeline hard-codes ``{POS}/outputs`` for model
   saves and would otherwise clobber the checked-in outputs tree.

Ensures project root is on ``sys.path`` so tests run from any cwd.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Make the project root importable before any pipeline imports.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


ALL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")


# ---------------------------------------------------------------------------
# Shrunk-config assembly
# ---------------------------------------------------------------------------

# Keys that production-config callers overwrite with shrunken values. These
# knobs dominate the 20s budget (NN epochs/width, LightGBM/attention toggles).
_TINY_OVERRIDES: dict[str, Any] = {
    "nn_backbone_layers": [8, 8],
    "nn_head_hidden": 4,
    "nn_dropout": 0.0,
    "nn_head_hidden_overrides": None,
    "nn_lr": 1e-3,
    "nn_weight_decay": 0.0,
    "nn_epochs": 1,
    "nn_batch_size": 64,
    "nn_patience": 1,
    "nn_log_every": 100,
    "scheduler_type": "cosine_warm_restarts",
    "cosine_t0": 1,
    "cosine_t_mult": 2,
    "cosine_eta_min": 1e-5,
    "ridge_cv_folds": 2,
    "ridge_refine_points": 0,
    "ridge_pca_components": None,
    # Kill the heavy side-models — they each add several seconds
    "train_attention_nn": False,
    "train_lightgbm": False,
}


def _qb_tiny() -> dict:
    from QB.run_qb_pipeline import QB_CONFIG

    return _shrink(QB_CONFIG)


def _rb_tiny() -> dict:
    from RB.run_rb_pipeline import RB_CONFIG

    return _shrink(RB_CONFIG)


def _wr_tiny() -> dict:
    """WR config — prefer WR_CONFIG_TINY if present, else shrink WR_CONFIG."""
    from WR.wr_data import filter_to_wr
    from WR.wr_features import (
        add_wr_specific_features,
        fill_wr_nans,
        get_wr_feature_columns,
    )
    from WR.wr_targets import compute_wr_fumble_adjustment, compute_wr_targets

    callables = {
        "filter_fn": filter_to_wr,
        "compute_targets_fn": compute_wr_targets,
        "add_features_fn": add_wr_specific_features,
        "fill_nans_fn": fill_wr_nans,
        "get_feature_columns_fn": get_wr_feature_columns,
        "compute_adjustment_fn": compute_wr_fumble_adjustment,
    }
    try:
        from WR.wr_config import WR_CONFIG_TINY

        cfg = dict(WR_CONFIG_TINY)
    except ImportError:
        from WR.run_wr_pipeline import WR_CONFIG

        cfg = _shrink(WR_CONFIG)
    cfg.update(callables)
    return cfg


def _te_tiny() -> dict:
    """TE config — use TE_CONFIG_TINY if exposed, else shrink TE_CONFIG inline."""
    from TE.te_data import filter_to_te
    from TE.te_features import (
        add_te_specific_features,
        fill_te_nans,
        get_te_feature_columns,
    )
    from TE.te_targets import compute_te_targets

    callables = {
        "filter_fn": filter_to_te,
        "compute_targets_fn": compute_te_targets,
        "add_features_fn": add_te_specific_features,
        "fill_nans_fn": fill_te_nans,
        "get_feature_columns_fn": get_te_feature_columns,
    }
    try:
        from TE.te_config import TE_CONFIG_TINY

        cfg = dict(TE_CONFIG_TINY)
    except ImportError:
        from TE.run_te_pipeline import TE_CONFIG

        cfg = _shrink(TE_CONFIG)
    cfg.update(callables)
    return cfg


def _k_tiny() -> dict:
    """K config — use K_CONFIG_TINY if exposed, else build a shrunk cfg from k_config.py."""
    from K.k_data import filter_to_k
    from K.k_features import (
        add_k_specific_features,
        fill_k_nans,
        get_k_feature_columns,
    )
    from K.k_targets import compute_k_targets

    try:
        from K.k_config import K_CONFIG_TINY

        cfg = dict(K_CONFIG_TINY)
    except ImportError:
        # Build a fresh tiny config from the exported config module constants.
        from K.k_config import (
            K_CV_SPLIT_COLUMN,
            K_HUBER_DELTAS,
            K_LOSS_W_TOTAL,
            K_LOSS_WEIGHTS,
            K_SPECIFIC_FEATURES,
            K_TARGETS,
        )

        cfg = {
            "targets": K_TARGETS,
            "specific_features": K_SPECIFIC_FEATURES,
            "loss_weights": K_LOSS_WEIGHTS,
            "loss_w_total": K_LOSS_W_TOTAL,
            "huber_deltas": K_HUBER_DELTAS,
            "cv_split_column": K_CV_SPLIT_COLUMN,
            "ridge_alpha_grids": {t: [1.0, 10.0] for t in K_TARGETS},
        }
        cfg.update(_TINY_OVERRIDES)

    cfg.update(
        {
            "filter_fn": filter_to_k,
            "compute_targets_fn": compute_k_targets,
            "add_features_fn": add_k_specific_features,
            "fill_nans_fn": fill_k_nans,
            "get_feature_columns_fn": get_k_feature_columns,
            "compute_adjustment_fn": None,
        }
    )
    return cfg


def _dst_tiny() -> dict:
    """DST config — use DST_CONFIG_TINY if exposed, else build a shrunk cfg inline."""
    from DST.dst_config import (
        DST_HUBER_DELTAS,
        DST_LOSS_W_TOTAL,
        DST_LOSS_WEIGHTS,
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

    cfg = {
        "targets": DST_TARGETS,
        "ridge_alpha_grids": {t: [1.0, 10.0] for t in DST_TARGETS},
        "specific_features": DST_SPECIFIC_FEATURES,
        "loss_weights": DST_LOSS_WEIGHTS,
        "loss_w_total": DST_LOSS_W_TOTAL,
        "huber_deltas": DST_HUBER_DELTAS,
        "filter_fn": filter_to_dst,
        "compute_targets_fn": compute_dst_targets,
        "add_features_fn": add_dst_specific_features,
        "fill_nans_fn": fill_dst_nans,
        "get_feature_columns_fn": get_dst_feature_columns,
        "compute_adjustment_fn": None,
    }
    cfg.update(_TINY_OVERRIDES)

    try:
        from DST.dst_config import DST_CONFIG_TINY

        cfg.update(DST_CONFIG_TINY)
    except ImportError:
        # DST_CONFIG_TINY not exposed; the generic tiny overrides above suffice.
        pass
    return cfg


_TINY_BUILDERS = {
    "QB": _qb_tiny,
    "RB": _rb_tiny,
    "WR": _wr_tiny,
    "TE": _te_tiny,
    "K": _k_tiny,
    "DST": _dst_tiny,
}


def _shrink(base_cfg: dict) -> dict:
    """Return a copy of ``base_cfg`` with tiny NN/scheduler/ridge overrides.

    Preserves the position-specific callables and feature schema; only the
    training hyperparameters get squashed. Used for positions whose config
    module does not (yet) expose a ``_CONFIG_TINY`` variant.
    """
    cfg = dict(base_cfg)
    cfg.update(_TINY_OVERRIDES)
    # Force a compact, deterministic ridge grid so per-target CV tuning is fast.
    if "ridge_alpha_grids" in cfg:
        cfg["ridge_alpha_grids"] = {
            t: [1.0, 10.0] for t in cfg.get("targets", cfg["ridge_alpha_grids"])
        }
    return cfg


def build_tiny_config(position: str) -> dict:
    """Return a shrunk run_pipeline config for the given position."""
    position = position.upper()
    if position not in _TINY_BUILDERS:
        raise ValueError(f"Unknown position {position!r}")
    return _TINY_BUILDERS[position]()


# ---------------------------------------------------------------------------
# Tiny splits (real data, sliced to tiny)
# ---------------------------------------------------------------------------

_SPLITS_DIR = Path(_PROJECT_ROOT) / "data" / "splits"


def _top_n_players(df: pd.DataFrame, n: int) -> pd.Index:
    """Return the ``n`` player_ids with the most rows in ``df`` (stable order)."""
    return (
        df.groupby("player_id").size().sort_values(ascending=False, kind="mergesort").head(n).index
    )


def _load_player_splits(
    position: str, n_players: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and slice pre-engineered splits for a player-level position."""
    train = pd.read_parquet(_SPLITS_DIR / "train.parquet")
    val_full = pd.read_parquet(_SPLITS_DIR / "val.parquet")
    test_full = pd.read_parquet(_SPLITS_DIR / "test.parquet")

    pos_train_all = train[train["position"] == position]
    top_players = _top_n_players(pos_train_all, n_players)

    # Train: take the most recent 2 seasons to bound runtime
    recent_seasons = sorted(pos_train_all["season"].unique())[-2:]
    pos_train = pos_train_all[
        pos_train_all["season"].isin(recent_seasons) & pos_train_all["player_id"].isin(top_players)
    ].copy()

    pos_val = val_full[
        (val_full["position"] == position) & val_full["player_id"].isin(top_players)
    ].copy()
    pos_test = test_full[
        (test_full["position"] == position) & test_full["player_id"].isin(top_players)
    ].copy()

    # Fallback: if val or test are too small (test set has different players),
    # reuse the last weeks of train so each split has rows.
    if len(pos_val) < 20:
        pos_val = pos_train_all[
            pos_train_all["season"].isin(recent_seasons)
            & pos_train_all["player_id"].isin(top_players)
            & (pos_train_all["week"] >= 15)
        ].copy()
    if len(pos_test) < 20:
        pos_test = pos_train_all[
            pos_train_all["season"].isin(recent_seasons)
            & pos_train_all["player_id"].isin(top_players)
            & (pos_train_all["week"] < 10)
        ].copy()

    return pos_train, pos_val, pos_test


def _build_k_splits(
    n_players: int = 30, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tiny K splits using the K loader (cached PBP parquet).

    Mirrors the logic in ``K/run_k_pipeline.py``: loads the reconstructed
    kicker weekly data, computes targets + features on the full frame, then
    splits by season (train<=2023 / val=2024 / test=2025) and subsets to
    the top-N most active kickers.
    """
    from K.k_data import load_kicker_data
    from K.k_features import compute_k_features
    from K.k_targets import compute_k_targets

    full = load_kicker_data()
    full = compute_k_targets(full)
    compute_k_features(full)

    top = _top_n_players(full, n_players)
    full = full[full["player_id"].isin(top)].copy()

    train = full[full["season"] <= 2023].copy()
    val = full[full["season"] == 2024].copy()
    test = full[full["season"] == 2025].copy()
    return train, val, test


def _build_dst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tiny D/ST splits using the DST team-level builder."""
    from DST.dst_data import build_dst_data
    from DST.dst_features import compute_dst_features
    from DST.dst_targets import compute_dst_targets

    full = build_dst_data()
    full = compute_dst_targets(full)
    compute_dst_features(full)

    # Restrict to the last 4 seasons so rolling features stabilise while
    # keeping the frame tiny enough for a <20s pipeline round-trip.
    full = full[full["season"] >= 2022].copy()

    train = full[full["season"] <= 2023].copy()
    val = full[full["season"] == 2024].copy()
    test = full[full["season"] == 2025].copy()
    return train, val, test


def load_tiny_splits(position: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a tiny (train, val, test) triple for the given position."""
    position = position.upper()
    if position in ("QB", "RB", "WR", "TE"):
        return _load_player_splits(position)
    if position == "K":
        return _build_k_splits()
    if position == "DST":
        return _build_dst_splits()
    raise ValueError(f"Unknown position {position!r}")


# ---------------------------------------------------------------------------
# Pipeline invocation with isolated cwd
# ---------------------------------------------------------------------------


def run_pipeline_in_tmp(
    position: str,
    cfg: dict,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
    seed: int = 42,
) -> dict:
    """Run ``shared.pipeline.run_pipeline`` inside ``tmp_path``.

    The pipeline hard-codes ``{POS}/outputs`` for artifact saves. We chdir
    into a tmp workspace and symlink ``data/`` so schedule parquet reads
    keep working without polluting the checked-in outputs directory.
    """
    from shared.pipeline import run_pipeline

    train_df, val_df, test_df = splits
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        data_link = tmp_path / "data"
        if not data_link.exists():
            data_link.symlink_to(Path(cwd) / "data", target_is_directory=True)
        return run_pipeline(
            position,
            cfg,
            train_df.copy(),
            val_df.copy(),
            test_df.copy(),
            seed=seed,
        )
    finally:
        os.chdir(cwd)
