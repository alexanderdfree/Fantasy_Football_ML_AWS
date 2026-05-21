"""Shared helpers for E2E and reproducibility tests at ``tests/``.

Consolidates three pieces of glue that both test files need:

1. ``build_tiny_config(position)`` — assemble a shrunk pipeline config
   (1 epoch, 2-layer x 8-unit NN, no attention/LightGBM) by feeding the
   position's ``POSITION_CONFIG`` through ``build_pipeline_config``, then
   layering ``_TINY_OVERRIDES`` and the per-position ``CONFIG_TINY``
   (if exposed in ``src/{pos}/config.py``) on top. Resolves the
   data/features/targets callables via :func:`build_position_callables`
   so a new position only needs to expose ``POSITION_CONFIG`` plus the
   standard symbols to get a tiny config for free.

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

import importlib
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


def build_tiny_config(position: str) -> dict:
    """Return a shrunk run_pipeline config for the given position.

    Pipeline:

    1. Start from the position's full CONFIG via
       :func:`src.shared.position_pipeline.build_pipeline_config` — this is
       the same code path ``src/{pos}/run_pipeline.py::CONFIG`` exercises,
       so callables and POSITION_CONFIG-derived fields are all populated.
    2. Apply ``_TINY_OVERRIDES`` to shrink the heavy NN/scheduler/ridge knobs.
    3. If the position's config module exposes a ``CONFIG_TINY`` dict, merge
       it last so its values override the generic overrides (e.g. WR/TE/K's
       single-alpha ridge grids); otherwise compact the ridge grid to two
       alphas via the legacy ``_shrink`` behaviour.
    4. K and DST add an explicit ``compute_adjustment_fn=None`` (historical
       contract — both positions never wired an adjustment function and the
       slot is read unconditionally).
    """
    from src.shared.position_pipeline import build_pipeline_config

    position = position.upper()
    if position not in ALL_POSITIONS:
        raise ValueError(f"Unknown position {position!r}")

    config_mod = importlib.import_module(f"src.{position.lower()}.config")
    cfg = build_pipeline_config(position, config_mod.POSITION_CONFIG)

    cfg.update(_TINY_OVERRIDES)

    # Compact ridge grid for speed — CONFIG_TINY can re-override below.
    if "ridge_alpha_grids" in cfg:
        cfg["ridge_alpha_grids"] = {t: [1.0, 10.0] for t in cfg.get("targets", [])}

    cfg_tiny = getattr(config_mod, "CONFIG_TINY", None)
    if cfg_tiny is not None:
        cfg.update(cfg_tiny)

    if position in ("K", "DST"):
        cfg.setdefault("compute_adjustment_fn", None)

    return cfg


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


def _build_k_splits(n_players: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tiny K splits using the K loader (cached PBP parquet).

    Mirrors the logic in ``src/k/run_pipeline.py``: loads the reconstructed
    kicker weekly data, computes targets + features on the full frame, then
    splits by season (train<=2023 / val=2024 / test=2025) and subsets to
    the top-N most active kickers.
    """
    from src.k.data import load_data
    from src.k.features import compute_features
    from src.k.targets import compute_targets

    full = load_data()
    full = compute_targets(full)
    compute_features(full)

    top = _top_n_players(full, n_players)
    full = full[full["player_id"].isin(top)].copy()

    train = full[full["season"] <= 2023].copy()
    val = full[full["season"] == 2024].copy()
    test = full[full["season"] == 2025].copy()
    return train, val, test


def _build_dst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build tiny D/ST splits using the DST team-level builder."""
    from src.dst.data import build_data
    from src.dst.features import compute_features
    from src.dst.targets import compute_targets

    full = build_data()
    full = compute_targets(full)
    compute_features(full)

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
    """Run ``src.shared.pipeline.run_pipeline`` inside ``tmp_path``.

    The pipeline hard-codes ``{POS}/outputs`` for artifact saves. We chdir
    into a tmp workspace and symlink ``data/`` so schedule parquet reads
    keep working without polluting the checked-in outputs directory.
    """
    from src.shared.pipeline import run_pipeline

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
