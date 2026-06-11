"""Generic position model pipeline.

Each position calls run_pipeline() with a config dict that bundles all
position-specific callables and hyperparameters.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CACHE_DIR,
    MIN_GAMES_PER_SEASON,
    SEASONS,
    SPLITS_DIR,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.data.split import expanding_window_folds
from src.features.engineer import (
    OPP_ATTN_PER_GAME_BUILDERS,
    build_game_history_arrays,
    build_opp_defense_history_arrays,
    get_attn_static_columns,
)
from src.shared import feature_cache
from src.shared.artifact_integrity import (
    wrap_state_dict,
    write_scaler_meta,
)
from src.shared.backtest import plot_weekly_accuracy, run_weekly_simulation
from src.shared.core_pool import lease_cores
from src.shared.evaluation import (
    build_gate_info,
    compute_metrics,
    compute_ranking_metrics,
    compute_target_metrics,
    plot_pred_vs_actual,
    print_comparison_table,
)
from src.shared.feature_build import build_position_features, scale_and_clip
from src.shared.models import (
    ElasticNetModel,
    ElasticNetMultiTarget,
    LightGBMMultiTarget,
    RidgeModel,
    RidgeMultiTarget,
    SeasonAverageBaseline,
    TabPFNMultiTarget,
)
from src.shared.neural_net import (
    build_multihead_net,
    build_multihead_net_with_history,
    build_multihead_net_with_nested_history,
)
from src.shared.training import (
    MultiHeadHistoryTrainer,
    MultiHeadHistoryWithOppTrainer,
    MultiHeadNestedHistoryTrainer,
    MultiHeadTrainer,
    MultiTargetLoss,
    make_dataloaders,
    make_history_dataloaders,
    make_history_with_opp_dataloaders,
    make_nested_kick_dataloaders,
    plot_training_curves,
)
from src.shared.utils import cuda_enabled, mps_enabled, seed_everything, timed


def _read_split(path: str) -> pd.DataFrame:
    """Read a split parquet, dropping playoff rows.

    Fantasy leagues end with the regular season, and the schedule lookup used
    for Vegas/weather features only covers REG games.
    """
    df = pd.read_parquet(path)
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()
    return df


def _resolve_nn_log_every(cfg):
    """Resolve nn_log_every: cfg wins, else LOG_EVERY env var, else 10.

    Batch jobs set LOG_EVERY=1 via containerOverrides so training prints every
    epoch without requiring a code edit. Position runners that don't know about
    that env var still get sensible behavior via the cfg key.
    """
    if "nn_log_every" in cfg and cfg["nn_log_every"] is not None:
        return int(cfg["nn_log_every"])
    env_val = os.environ.get("LOG_EVERY")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 10


# ---------------------------------------------------------------------------
# Shared NN training scaffolding
#
# Every neural-net training path in this module (the flat ``_train_nn``, the
# two attention variants, and the per-fold training inside
# ``run_cv_pipeline``) repeats the same structure: fit a StandardScaler,
# clip, build an optimizer + scheduler + loss + Trainer, run
# ``trainer.train``. Centralising that scaffolding prevents the class of
# drift documented in TODO.md archive "``run_cv_pipeline`` missing
# ``non_negative_targets`` on MultiHeadNet" — adding a new training-loop
# concern (scheduler warmup, loss kwarg, etc.) requires one edit instead
# of four.
# ---------------------------------------------------------------------------


def _nn_device() -> torch.device:
    if cuda_enabled():
        # Batch shapes are stable across epochs (DataLoader uses fixed
        # batch_size with drop_last=True on train), so the autotuner's
        # selected kernels stay valid; the one-time search cost amortises
        # across the epoch loop. Idempotent — safe to set on every call.
        #
        # cuDNN benchmark can introduce per-run kernel-selection variation
        # under unstable shapes or memory pressure. Disable it whenever the
        # caller has opted into deterministic algorithms (so a single global
        # toggle drives the whole reproducibility story), and expose
        # ``FF_DETERMINISTIC=1`` as an out-of-band override for debugging
        # numerical drift without having to touch ``use_deterministic_algorithms``.
        force_deterministic = os.environ.get("FF_DETERMINISTIC", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        deterministic = force_deterministic or torch.are_deterministic_algorithms_enabled()
        torch.backends.cudnn.benchmark = not deterministic
        # TF32 for FP32 matmuls on Ampere+ (sm_80+). This is a hardware no-op on
        # Turing (T4, sm_75) and on CPU, so the T4/Mac/CI numerics are unchanged;
        # it only speeds up residual FP32 GEMMs on Blackwell (RTX 5080). Skipped
        # under deterministic mode, where full-precision FP32 is the contract.
        if not deterministic:
            torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    # MPS is opt-in (FF_DEVICE=mps) and never reached on the auto/cpu/cuda paths,
    # so the default CUDA-or-CPU selection above stays byte-identical. The
    # GPU-resident batcher (_gpu_resident_device) and FP16 AMP (_autocast) are
    # CUDA-only by design, so an MPS run uses the DataLoader + FP32 path.
    if mps_enabled():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Opt-in ``torch.compile`` for sm_80+ GPUs; a no-op everywhere by default.

    D12 measured ``torch.compile(model, dynamic=True)`` at **+32%** wall-clock on
    the T4 (cb3c960 train-ec2 run): the T4 is sm_75 with too few SMs for
    ``max_autotune_gemm`` (Inductor logged "Not enough SMs"), and the attention
    path's variable padded-sequence lengths trigger dynamic-shape guard re-checks
    that never amortize. **That decision stands for the T4** — so this stays a
    no-op on T4 and, crucially, a no-op *everywhere unless explicitly opted in*,
    keeping AWS T4 production and CPU/Mac/CI byte-identical to today.

    It is re-enabled ONLY when ``FF_COMPILE`` is truthy AND the GPU is sm_80+
    (e.g. RTX 5080 ``sm_120``, which has many more SMs than the T4), as a local
    experiment. The +32% was a *T4* result; sm_80+ may win — but the dynamic-
    shape caveat above is hardware-independent, so **benchmark before trusting
    it** (diff ``benchmark_history/`` with vs without ``FF_COMPILE=1``).
    """
    if os.environ.get("FF_COMPILE", "").lower() not in {"1", "true", "yes", "on"}:
        return model
    if not cuda_enabled():
        return model
    # sm_80 (Ampere) is the first generation with enough SMs for the GEMM
    # autotune path; T4 (sm_75) stays on the proven no-compile path (D12).
    if torch.cuda.get_device_capability()[0] < 8:
        return model
    return torch.compile(model, dynamic=True)


def _maybe_force_dropout_zero(cfg: dict) -> dict:
    """Test-only (``FF_FORCE_DROPOUT_ZERO``): return a cfg copy with every NN
    dropout rate zeroed; a no-op (returns ``cfg`` unchanged) by default.

    Sole purpose is the CUDA-graph Part-1 inertness gate: with dropout off the
    forward draws no random masks, so an ``FF_CUDA_GRAPH=0`` vs ``=1`` A/B can
    assert *exact* (per-target MAE Δ=0.0000) kernel inertness instead of the
    seed-noise band live dropout forces — graph capture's warmup iters and RNG
    offset stepping perturb *which* masks are drawn, not the kernel math. Never
    set in production; it changes the model (regularisation), not just the
    launch mechanics.
    """
    if os.environ.get("FF_FORCE_DROPOUT_ZERO", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return cfg
    out = dict(cfg)
    for key in ("nn_dropout", "attn_dropout", "attn_history_dropout", "attn_self_dropout"):
        out[key] = 0.0
    return out


def _scale_xs(*X_arrays: np.ndarray) -> tuple[StandardScaler, list[np.ndarray]]:
    """Fit a StandardScaler on the first array, transform + clip all arrays.

    Returns ``(scaler, [X_train_s, X_val_s, ...])``. Callers unpack the list
    with as many positional targets as they passed in. See
    ``src/shared/feature_build.py::scale_and_clip`` for the clip rationale.
    """
    scaler = StandardScaler()
    scaled = [scale_and_clip(scaler, X_arrays[0], fit=True)]
    scaled.extend(scale_and_clip(scaler, X) for X in X_arrays[1:])
    return scaler, scaled


def _run_nn_training(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: dict,
    targets: list[str],
    trainer_cls,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_kwargs: dict | None = None,
    scheduler_prefix: str = "",
) -> dict:
    """Build optimizer / scheduler / loss / trainer, run ``trainer.train``.

    Centralises the four bodies that previously rebuilt this stack inline
    (``_train_nn``, ``_train_attention_nn``, ``_train_nested_attention_nn``,
    and ``run_cv_pipeline``'s per-fold loop). Callers construct the model,
    loaders, and trainer *class* — everything between that and the trained
    weights lives here.
    """
    # ``fused=True`` does the whole parameter update in one CUDA kernel instead
    # of the default ``foreach`` multi-tensor path's several launches — a
    # launch-overhead win for this small, host-bound model (~20% GPU util).
    # CUDA-only; ``fused=False`` is the valid no-op on CPU/MPS.
    _first_param = next(model.parameters(), None)
    _fused = _first_param is not None and _first_param.is_cuda
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=_fused
    )
    scheduler, scheduler_per_batch = _build_scheduler(
        optimizer, cfg, train_loader, scheduler_prefix=scheduler_prefix
    )
    # If the model has no GatedHead, hurdle families are not supported (no
    # value_mu / value_log_alpha emissions). Transparently downgrade those
    # entries to huber so base NN / nested-attention paths run cleanly without
    # requiring callers to maintain a parallel head_losses config.
    from src.shared.neural_net import GatedHead  # local import avoids circular deps

    head_losses = cfg.get("head_losses")
    if head_losses and not any(isinstance(m, GatedHead) for m in model.modules()):
        head_losses = {
            n: ("huber" if lt in ("hurdle_negbin", "hurdle_poisson") else lt)
            for n, lt in head_losses.items()
        }

    criterion = MultiTargetLoss(
        target_names=targets,
        loss_weights=cfg["loss_weights"],
        huber_deltas=cfg["huber_deltas"],
        poisson_targets=cfg.get("poisson_targets"),
        head_losses=head_losses,
        **(loss_kwargs or {}),
    )
    # Optuna pruning hook from src/tuning/tune_nn.py. Sourced from cfg so the
    # run() / run_pipeline() signatures stay untouched. Gated on attention
    # trainer kinds because the tuner targets attention NN — without this gate
    # the callback would also fire for the regular MultiHeadNet phase that
    # runs first in the pipeline, mingling two unrelated loss trajectories
    # and breaking the pruner's monotonicity assumption.
    _ATTENTION_TRAINERS = (
        "MultiHeadHistoryTrainer",
        "MultiHeadHistoryWithOppTrainer",
        "MultiHeadNestedHistoryTrainer",
    )
    cfg_epoch_cb = (
        cfg.get("epoch_callback") if trainer_cls.__name__ in _ATTENTION_TRAINERS else None
    )

    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=next(model.parameters()).device,
        target_names=targets,
        patience=patience,
        scheduler_per_batch=scheduler_per_batch,
        log_every=_resolve_nn_log_every(cfg),
        epoch_callback=cfg_epoch_cb,
        use_amp=cfg.get("nn_use_amp", False),
    )
    return trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])


# ---------------------------------------------------------------------------
# Ridge hyperparameter tuning helpers
# ---------------------------------------------------------------------------


def _build_expanding_cv_folds(split_values, n_folds):
    """Build expanding-window cross-validation fold indices.

    Args:
        split_values: Array of season or week labels for each row.
        n_folds: Number of CV folds.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    unique_periods = sorted(np.unique(split_values))
    val_periods = unique_periods[-n_folds:]

    folds = []
    for val_period in val_periods:
        train_mask = split_values < val_period
        val_mask = split_values == val_period
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


def _eval_alpha_cv(X, y, folds, alpha, pca_n_components=None):
    """Evaluate a single Ridge alpha across CV folds, returning mean MAE."""
    maes = []
    for train_idx, val_idx in folds:
        model = RidgeModel(alpha=alpha, pca_n_components=pca_n_components)
        model.fit(X[train_idx], y[train_idx])
        # Unconditional >=0 clamp is safe: every position's raw-stat targets are
        # non-negative (all six set nn_non_negative_targets=set(_TARGETS)), so no
        # signed head is wrongly truncated here today. Gate on the non-negative
        # set only if a position ever adds a signed target (#351 F13).
        preds = np.maximum(model.predict(X[val_idx]), 0)
        maes.append(np.mean(np.abs(preds - y[val_idx])))
    return np.mean(maes)


def _tune_ridge_alphas_cv(
    X_train,
    y_train_dict,
    split_values,
    targets,
    alpha_grids,
    n_cv_folds=4,
    refine_points=5,
    pca_n_components=None,
    n_jobs=-1,
):
    """Per-target Ridge alpha tuning with expanding-window CV.

    Pass 1: coarse grid search across CV folds.
    Pass 2: fine refinement around the best coarse alpha.

    Each pass fans its alpha grid out over ``joblib.Parallel(n_jobs=-1,
    prefer="threads")``. Threads (not processes) because sklearn's Ridge
    delegates the normal-equation solve to BLAS, which releases the GIL —
    actual parallelism without process-spawn / pickle / nested-pool costs that
    bite on macOS. Each ``_eval_alpha_cv`` call is pure and the final
    best-alpha selection uses ``argmin``, so execution order is immaterial and
    output is numerically identical to the serial version.

    Returns dict mapping each target name to its optimal alpha.
    """
    folds = _build_expanding_cv_folds(split_values, n_cv_folds)
    best_alphas = {}

    for target in targets:
        y = y_train_dict[target]
        grid = list(alpha_grids[target])

        # --- Pass 1: coarse search ---
        coarse_maes = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_eval_alpha_cv)(X_train, y, folds, alpha, pca_n_components) for alpha in grid
        )
        best_idx = int(np.argmin(coarse_maes))
        best_mae = float(coarse_maes[best_idx])
        best_alpha = grid[best_idx]

        # --- Pass 2: fine refinement ---
        if refine_points > 0 and len(grid) >= 2:
            log_step = np.log10(grid[1]) - np.log10(grid[0])
            center = np.log10(best_alpha)
            fine_grid = list(np.logspace(center - log_step, center + log_step, refine_points))
            fine_maes = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_eval_alpha_cv)(X_train, y, folds, alpha, pca_n_components)
                for alpha in fine_grid
            )
            fine_best = int(np.argmin(fine_maes))
            if fine_maes[fine_best] < best_mae:
                best_mae = float(fine_maes[fine_best])
                best_alpha = fine_grid[fine_best]

        best_alphas[target] = round(best_alpha, 6)
        print(f"  {target}: best alpha={best_alphas[target]:.4f} (CV MAE={best_mae:.3f})")

    return best_alphas


# ---------------------------------------------------------------------------
# ElasticNet hyperparameter tuning helpers
#
# Parallel to Ridge's single-alpha-per-target tuning. ElasticNet adds a
# second axis (l1_ratio) for L1/L2 mixing. The search is nested — for each
# l1_ratio we run the same coarse-then-fine alpha routine Ridge uses, and
# pick the (l1_ratio, alpha) pair at the joint minimum. No PCA branch: L1
# on a rotated basis doesn't zero original features, which defeats the
# purpose of choosing ElasticNet over Ridge in the first place.
# ---------------------------------------------------------------------------


def _eval_enet_cv(X, y, folds, alpha, l1_ratio):
    """Evaluate a single ElasticNet (alpha, l1_ratio) across CV folds.

    Returns mean post-clip MAE, matching Ridge's ``_eval_alpha_cv`` so the
    two linear baselines are tuned on the same objective.
    """
    maes = []
    for train_idx, val_idx in folds:
        model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X[train_idx], y[train_idx])
        # Unconditional >=0 clamp — safe because all targets are non-negative raw
        # stats; see _ridge_cv_mae for the precondition (#351 F13).
        preds = np.maximum(model.predict(X[val_idx]), 0)
        maes.append(np.mean(np.abs(preds - y[val_idx])))
    return np.mean(maes)


def _tune_enet_cv(
    X_train,
    y_train_dict,
    split_values,
    targets,
    alpha_grids,
    l1_ratios,
    n_cv_folds=4,
    refine_points=5,
    n_jobs=-1,
):
    """Per-target ElasticNet tuning over (alpha, l1_ratio).

    For each target, for each l1_ratio candidate, runs Ridge's two-pass
    coarse/fine alpha search. The best (alpha, l1_ratio) pair is the joint
    minimum across all passes. Per-ratio alpha refinement is independent —
    optimal alpha scales with l1_ratio because L1 and L2 penalize differently.

    Returns ``dict[target, {"alpha": float, "l1_ratio": float}]``.
    """
    folds = _build_expanding_cv_folds(split_values, n_cv_folds)
    best = {}

    for target in targets:
        y = y_train_dict[target]
        grid = list(alpha_grids[target])
        best_mae = float("inf")
        best_alpha = grid[0]
        best_l1_ratio = l1_ratios[0]

        for l1_ratio in l1_ratios:
            # --- Pass 1: coarse alpha search at this l1_ratio ---
            coarse_maes = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_eval_enet_cv)(X_train, y, folds, alpha, l1_ratio) for alpha in grid
            )
            idx = int(np.argmin(coarse_maes))
            mae = float(coarse_maes[idx])
            alpha_here = grid[idx]

            # --- Pass 2: fine refinement around the coarse winner ---
            if refine_points > 0 and len(grid) >= 2:
                log_step = np.log10(grid[1]) - np.log10(grid[0])
                center = np.log10(alpha_here)
                fine_grid = list(np.logspace(center - log_step, center + log_step, refine_points))
                fine_maes = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_eval_enet_cv)(X_train, y, folds, alpha, l1_ratio)
                    for alpha in fine_grid
                )
                fine_idx = int(np.argmin(fine_maes))
                if fine_maes[fine_idx] < mae:
                    mae = float(fine_maes[fine_idx])
                    alpha_here = fine_grid[fine_idx]

            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha_here
                best_l1_ratio = l1_ratio

        best[target] = {
            "alpha": round(float(best_alpha), 6),
            "l1_ratio": round(float(best_l1_ratio), 4),
        }
        print(
            f"  {target}: best alpha={best[target]['alpha']:.4f}, "
            f"l1_ratio={best[target]['l1_ratio']:.2f} (CV MAE={best_mae:.3f})"
        )

    return best


def _scheduler_value(cfg: dict, key: str, scheduler_prefix: str):
    """Return a scheduler value, preferring an optional prefixed override."""
    if scheduler_prefix:
        override_key = f"{scheduler_prefix}{key}"
        if cfg.get(override_key) is not None:
            return cfg[override_key]
    return cfg[key]


def _build_scheduler(optimizer, cfg, train_loader, *, scheduler_prefix: str = ""):
    """Create the LR scheduler from config."""
    # Prefix-aware so the attention path (scheduler_prefix="attn_") can override
    # the scheduler TYPE too, not just the LR-scale values — falls back to the
    # shared ``scheduler_type`` when ``attn_scheduler_type`` is unset (#792).
    sched_type = _scheduler_value(cfg, "scheduler_type", scheduler_prefix)
    if sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_scheduler_value(cfg, "onecycle_max_lr", scheduler_prefix),
            epochs=cfg["nn_epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=_scheduler_value(cfg, "onecycle_pct_start", scheduler_prefix),
        ), True  # scheduler_per_batch=True
    elif sched_type == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=_scheduler_value(cfg, "cosine_t0", scheduler_prefix),
            T_mult=_scheduler_value(cfg, "cosine_t_mult", scheduler_prefix),
            eta_min=_scheduler_value(cfg, "cosine_eta_min", scheduler_prefix),
        ), False  # scheduler_per_batch=False
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=_scheduler_value(cfg, "plateau_factor", scheduler_prefix),
            patience=_scheduler_value(cfg, "plateau_patience", scheduler_prefix),
        ), False  # scheduler_per_batch=False
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def _prepare_position_data_uncached(position, cfg, train_df, val_df, test_df=None):
    """Filter to position, compute targets, add features, build arrays.

    Returns:
        (X_train, X_val, X_test_or_None,
         y_train_dict, y_val_dict, y_test_dict_or_None,
         pos_train, pos_val, pos_test_or_None, feature_cols)
    """
    pos = position

    # Filter to position
    pos_train = cfg["filter_fn"](train_df)
    pos_val = cfg["filter_fn"](val_df)
    pos_test = cfg["filter_fn"](test_df) if test_df is not None else None

    # Compute targets — raw-stat only, does not depend on schedule features, so it
    # runs before the schedule merge inside build_position_features. Done BEFORE the
    # min-games filter (it's per-row, so it commutes with the row filter → Δ0 for
    # the surviving rows) so the UNFILTERED train captured below carries targets —
    # RB/WR compute per-game share/HHI/career features over that full frame and
    # select the filtered rows back, so the dropped low-volume players don't
    # undercount those denominators/sums (#574/#531).
    targets = cfg["targets"]
    pos_train = cfg["compute_targets_fn"](pos_train)
    pos_val = cfg["compute_targets_fn"](pos_val)
    if pos_test is not None:
        pos_test = cfg["compute_targets_fn"](pos_test)

    # Min-games filter: training only. Per-position via cfg["min_games_per_season"]
    # (None → the global MIN_GAMES_PER_SEASON). Relaxing it adds low-volume
    # player-seasons back to TRAIN, which helps the cold-start test subgroup the
    # model would otherwise never see (TODO.md min-games entry).
    # ``full_train`` is the pre-filter (targeted) train: RB/WR's add_specific_features
    # computes its per-game team-total / share / HHI / career features over it and
    # returns only the filtered rows, so those aggregates see the full player set.
    # fill_nans + the StandardScaler still fit on the FILTERED train below (#569).
    full_train = pos_train
    min_games = cfg.get("min_games_per_season")
    if min_games is None:
        min_games = MIN_GAMES_PER_SEASON
    games_per_season = pos_train.groupby(["player_id", "season"])["week"].transform("count")
    pos_train = pos_train[games_per_season >= min_games].copy()

    dfs_for_features = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    sizes = ", ".join(
        f"{name}={len(df)}"
        for name, df in zip(["train", "val", "test"], dfs_for_features, strict=False)
    )
    print(f"  {pos} splits: {sizes}")

    feature_cols = cfg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, cfg, feature_cols, full_train=full_train
    )

    X_train = pos_train[feature_cols].values.astype(np.float32)
    X_val = pos_val[feature_cols].values.astype(np.float32)
    X_test = pos_test[feature_cols].values.astype(np.float32) if pos_test is not None else None

    # .astype(np.float32) gives a writable float32 copy — matches X above and
    # avoids PyTorch's non-writable-tensor warning when these feed into
    # MultiTargetDataset (torch.FloatTensor refuses to borrow read-only numpy
    # buffers that pandas arithmetic produces on computed columns).
    y_train_dict = {t: pos_train[t].values.astype(np.float32) for t in targets}
    y_val_dict = {t: pos_val[t].values.astype(np.float32) for t in targets}
    y_test_dict = (
        {t: pos_test[t].values.astype(np.float32) for t in targets}
        if pos_test is not None
        else None
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train_dict,
        y_val_dict,
        y_test_dict,
        pos_train,
        pos_val,
        pos_test,
        feature_cols,
    )


def _prepare_position_data(position, cfg, train_df, val_df, test_df=None):
    """Cached wrapper around ``_prepare_position_data_uncached``.

    The cache key is content-hashed on (position, train_df, val_df, test_df,
    relevant cfg keys). Each CV fold's (train, val) is unique within a single
    process, so the within-run benefit is from re-runs (Optuna trials, CLI
    iteration) — but the disk cache means the second pipeline run on the same
    data skips feature engineering entirely.

    Bypass with ``FF_FEATURE_CACHE_DISABLE=1``.
    """
    return feature_cache.load_or_compute(
        position,
        train_df,
        val_df,
        test_df,
        cfg,
        lambda: _prepare_position_data_uncached(position, cfg, train_df, val_df, test_df),
    )


def _prepare_train_val(position, cfg, train_df, val_df):
    """Train+val variant — same prep as _prepare_position_data without test."""
    (X_train, X_val, _, y_train_dict, y_val_dict, _, pos_train, pos_val, _, feature_cols) = (
        _prepare_position_data(
            position,
            cfg,
            train_df,
            val_df,
            test_df=None,
        )
    )
    return X_train, X_val, y_train_dict, y_val_dict, pos_train, pos_val, feature_cols


def build_train_matrix(position: str, cfg: dict) -> tuple[np.ndarray, dict, list[str]]:
    """Rebuild the training feature matrix + target dict for a position.

    Entry point for diagnostics that need the exact ``X_train`` the pipeline
    sees (SHAP, permutation importance, ablation scripts). Delegates to
    ``_prepare_train_val`` so there's no parallel feature-building logic to
    drift — diagnostic scripts that inline their own pipeline setup miss
    target renames and silently use stale data prep.

    Val is built and discarded; training-side feature engineering is agnostic
    to whether val is present, so the work is cheap and the interface stays
    single-purpose.
    """
    train_df = _read_split(f"{SPLITS_DIR}/train.parquet")
    val_df = _read_split(f"{SPLITS_DIR}/val.parquet")
    X_train, _, y_train_dict, _, _, _, feature_cols = _prepare_train_val(
        position, cfg, train_df, val_df
    )
    return X_train, y_train_dict, feature_cols


def _train_nn(
    X_train,
    X_val,
    X_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    cfg,
    targets,
    seed,
):
    """Train a MultiHeadNet and return (model, scaler, test_preds, metrics, history)."""
    seed_everything(seed)
    cfg = _maybe_force_dropout_zero(cfg)
    nn_scaler, (X_train_s, X_val_s, X_test_s) = _scale_xs(X_train, X_val, X_test)

    device = _nn_device()
    train_loader, val_loader = make_dataloaders(
        X_train_s,
        y_train_dict,
        X_val_s,
        y_val_dict,
        batch_size=cfg["nn_batch_size"],
        device=device,
    )

    model = build_multihead_net(cfg, input_dim=X_train_s.shape[1], targets=targets).to(device)

    history = _run_nn_training(
        model=_maybe_compile(model),
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        targets=targets,
        trainer_cls=MultiHeadTrainer,
        lr=cfg["nn_lr"],
        weight_decay=cfg["nn_weight_decay"],
        patience=cfg["nn_patience"],
    )

    test_preds = model.predict_numpy(X_test_s, device)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)

    return model, nn_scaler, test_preds, metrics, history


def _train_attention_nn(
    X_train,
    X_val,
    X_test,
    hist_train,
    mask_train,
    hist_val,
    mask_val,
    hist_test,
    mask_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    cfg,
    targets,
    seed,
    feature_cols=None,
    opp_hist_train=None,
    opp_mask_train=None,
    opp_hist_val=None,
    opp_mask_val=None,
    opp_hist_test=None,
    opp_mask_test=None,
):
    """Train a MultiHeadNetWithHistory and return (model, scaler, test_preds, metrics, history).

    Like _train_nn but feeds both static features and game history sequences.
    When ``opp_hist_*`` tensors are provided (all six must be set together),
    a parallel attention branch attends over the opponent defense's per-game
    history; otherwise the model runs as a single-history net (back-compat).
    """
    seed_everything(seed)
    cfg = _maybe_force_dropout_zero(cfg)
    use_opp = opp_hist_train is not None
    if use_opp and (
        opp_mask_train is None
        or opp_hist_val is None
        or opp_mask_val is None
        or opp_hist_test is None
        or opp_mask_test is None
    ):
        raise ValueError("All opp history + mask tensors must be provided together.")

    # Filter to the per-position whitelist of static-branch features — the
    # attention branch learns its own temporal representation from raw game
    # stats, so rolling / EWMA / trend / share / specific categories are
    # excluded by config (``POSITION_CONFIG.attn_static_features``).
    if feature_cols is not None:
        static_whitelist = cfg["attn_static_features"]
        static_cols = get_attn_static_columns(feature_cols, static_whitelist)
        static_set = set(static_cols)
        col_idx = [i for i, c in enumerate(feature_cols) if c in static_set]
        X_train = X_train[:, col_idx]
        X_val = X_val[:, col_idx]
        X_test = X_test[:, col_idx]
        suffix = " (filtered)" if len(col_idx) != len(feature_cols) else ""
        print(f"  Attention static features: {len(col_idx)}/{len(feature_cols)}{suffix}")

    nn_scaler, (X_train_s, X_val_s, X_test_s) = _scale_xs(X_train, X_val, X_test)

    attn_batch_size = cfg.get("attn_batch_size", cfg["nn_batch_size"])
    device = _nn_device()
    if use_opp:
        train_loader, val_loader = make_history_with_opp_dataloaders(
            X_train_s,
            hist_train,
            mask_train,
            opp_hist_train,
            opp_mask_train,
            y_train_dict,
            X_val_s,
            hist_val,
            mask_val,
            opp_hist_val,
            opp_mask_val,
            y_val_dict,
            batch_size=attn_batch_size,
            device=device,
        )
        trainer_cls = MultiHeadHistoryWithOppTrainer
    else:
        train_loader, val_loader = make_history_dataloaders(
            X_train_s,
            hist_train,
            mask_train,
            y_train_dict,
            X_val_s,
            hist_val,
            mask_val,
            y_val_dict,
            batch_size=attn_batch_size,
            device=device,
        )
        trainer_cls = MultiHeadHistoryTrainer

    model = build_multihead_net_with_history(
        cfg,
        static_dim=X_train_s.shape[1],
        game_dim=hist_train.shape[2],
        targets=targets,
        opp_game_dim=(opp_hist_train.shape[2] if use_opp else None),
    ).to(device)

    history = _run_nn_training(
        model=_maybe_compile(model),
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        targets=targets,
        trainer_cls=trainer_cls,
        lr=cfg.get("attn_lr", cfg["nn_lr"]),
        weight_decay=cfg.get("attn_weight_decay", cfg["nn_weight_decay"]),
        patience=cfg.get("attn_patience", cfg["nn_patience"]),
        scheduler_prefix="attn_",
        loss_kwargs={
            "gate_weight": cfg.get("attn_gate_weight", 1.0),
            "gated_targets": cfg.get("gated_targets"),
        },
    )

    if use_opp:
        test_preds = model.predict_numpy(
            X_test_s,
            hist_test,
            mask_test,
            device,
            X_opp_history=opp_hist_test,
            opp_history_mask=opp_mask_test,
        )
    else:
        test_preds = model.predict_numpy(X_test_s, hist_test, mask_test, device)
    gate_info = build_gate_info(test_preds, cfg.get("gated_targets") or [])
    metrics = compute_target_metrics(y_test_dict, test_preds, targets, gate_info=gate_info)

    return model, nn_scaler, test_preds, metrics, history


def _train_nested_attention_nn(
    X_train,
    X_val,
    X_test,
    hist_train,
    outer_train,
    inner_train,
    hist_val,
    outer_val,
    inner_val,
    hist_test,
    outer_test,
    inner_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    cfg,
    targets,
    seed,
    game_hist_train=None,
    game_hist_val=None,
    game_hist_test=None,
):
    """Train a MultiHeadNetWithNestedHistory on pre-padded nested history.

    Parallel to _train_attention_nn but consumes a 4-D kick tensor plus outer
    and inner masks. No whitelist filter here: when attn_static_from_df=True
    the caller has already rebuilt X from the DataFrame using
    cfg['attn_static_features'].

    The optional ``game_hist_*`` arrays carry pre-computed per-game
    aggregates (``ATTN_HISTORY_STATS``) aligned to the same outer game order
    as the nested kick tensor; when provided they are concatenated with the
    inner-pool output before the outer attention.
    """
    seed_everything(seed)
    cfg = _maybe_force_dropout_zero(cfg)
    nn_scaler, (X_train_s, X_val_s, X_test_s) = _scale_xs(X_train, X_val, X_test)

    attn_batch_size = cfg.get("attn_batch_size", cfg["nn_batch_size"])
    device = _nn_device()
    train_loader, val_loader = make_nested_kick_dataloaders(
        X_train_s,
        hist_train,
        outer_train,
        inner_train,
        y_train_dict,
        X_val_s,
        hist_val,
        outer_val,
        inner_val,
        y_val_dict,
        batch_size=attn_batch_size,
        X_train_history=game_hist_train,
        X_val_history=game_hist_val,
        device=device,
    )

    game_dim = 0 if game_hist_train is None else game_hist_train.shape[-1]

    model = build_multihead_net_with_nested_history(
        cfg,
        static_dim=X_train_s.shape[1],
        kick_dim=hist_train.shape[-1],
        max_games=hist_train.shape[1],
        targets=targets,
        game_dim=game_dim,
    ).to(device)

    history = _run_nn_training(
        model=_maybe_compile(model),
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        targets=targets,
        trainer_cls=MultiHeadNestedHistoryTrainer,
        lr=cfg.get("attn_lr", cfg["nn_lr"]),
        weight_decay=cfg.get("attn_weight_decay", cfg["nn_weight_decay"]),
        patience=cfg.get("attn_patience", cfg["nn_patience"]),
        scheduler_prefix="attn_",
        # Thread the gate/gated-target loss config like the flat attention path
        # (_train_attention_nn) so a nested+gated config trains its gate head
        # instead of silently dropping it (#392/#422). Inert for today's K
        # config (no gated_targets → empty gate loop), but it removes the
        # flat/nested asymmetry that would silently break a future gated
        # nested head.
        loss_kwargs={
            "gate_weight": cfg.get("attn_gate_weight", 1.0),
            "gated_targets": cfg.get("gated_targets"),
        },
    )

    test_preds = model.predict_numpy(
        X_test_s, hist_test, outer_test, inner_test, device, X_game_history=game_hist_test
    )
    # Mirror the flat attention path: surface gate diagnostics for any gated
    # targets so a nested+gated config doesn't silently lose gate AUC/Brier/
    # conditional-MAE. ``build_gate_info`` returns ``None`` when no targets are
    # gated (today's K config), leaving metrics unchanged.
    gate_info = build_gate_info(test_preds, cfg.get("gated_targets") or [])
    metrics = compute_target_metrics(y_test_dict, test_preds, targets, gate_info=gate_info)

    return model, nn_scaler, test_preds, metrics, history


def _splits_stat_key() -> tuple:
    """(name, mtime_ns, size) of the three split parquets.

    Cheap staleness guard for the per-worker trial-data memo: catches a
    splits rebuild mid-process without paying the per-trial parquet read +
    content hash the memo exists to skip.
    """
    return tuple(
        (name, st.st_mtime_ns, st.st_size)
        for name in ("train.parquet", "val.parquet", "test.parquet")
        for st in (os.stat(f"{SPLITS_DIR}/{name}"),)
    )


def _train_attention_holdout(
    position,
    cfg,
    targets,
    seed,
    X_train,
    X_val,
    X_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    pos_train,
    pos_val,
    pos_test,
    feature_cols,
    opp_source_frames,
):
    """Build attention static/history/opp tensors and train the attention NN.

    Shared by both ``run_pipeline`` (flat path) and ``run_cv_pipeline``'s final
    holdout so the two entry points can't drift on static-feature selection,
    nested-vs-flat dispatch, or the opponent-history branch. Returns
    ``(attn_model, attn_nn_scaler, attn_nn_test_preds, attn_nn_metrics,
    attn_history, attn_feature_cols)``.

    ``opp_source_frames`` is the ``(train_df, val_df, test_df)`` triple of
    *all-position* frames the "defense" opponent builder concatenates; the
    "offense" builder ignores it and reads the weekly cache instead.
    """
    # Static features: either filter the base matrix (RB-style whitelist)
    # or rebuild directly from the DataFrame (K-style — its L1 attention
    # features live outside ALL_FEATURES to stay out of Ridge/base NN).
    if cfg.get("attn_static_from_df", False):
        attn_static_cols = cfg["attn_static_features"]
        X_attn_train = pos_train[attn_static_cols].to_numpy(dtype=np.float32)
        X_attn_val = pos_val[attn_static_cols].to_numpy(dtype=np.float32)
        X_attn_test = pos_test[attn_static_cols].to_numpy(dtype=np.float32)
        attn_feature_cols = list(attn_static_cols)
    else:
        X_attn_train, X_attn_val, X_attn_test = X_train, X_val, X_test
        attn_feature_cols = feature_cols

    structure = cfg.get("attn_history_structure", "flat")
    if structure == "nested":
        builder_fn = cfg["attn_history_builder_fn"]
        hist_train, outer_train, inner_train = builder_fn(pos_train)
        hist_val, outer_val, inner_val = builder_fn(pos_val)
        hist_test, outer_test, inner_test = builder_fn(pos_test)
        print(
            f"  Nested history shape: {hist_train.shape} "
            f"(max_games={hist_train.shape[1]}, "
            f"max_kicks={hist_train.shape[2]}, "
            f"kick_dim={hist_train.shape[3]})"
        )

        # Optional per-game aggregate branch ("ATTN_HISTORY_STATS" for the
        # nested path). Built with the same build_game_history_arrays helper the
        # flat positions use, but capped to the nested model's max_games so the
        # outer sequence length matches the kick tensor.
        game_history_stats = cfg.get("attn_history_stats")
        game_hist_train = game_hist_val = game_hist_test = None
        if game_history_stats:
            nested_max_games = hist_train.shape[1]
            game_hist_train, _ = build_game_history_arrays(
                pos_train, history_stats=game_history_stats, max_seq_len=nested_max_games
            )
            game_hist_val, _ = build_game_history_arrays(
                pos_val, history_stats=game_history_stats, max_seq_len=nested_max_games
            )
            game_hist_test, _ = build_game_history_arrays(
                pos_test, history_stats=game_history_stats, max_seq_len=nested_max_games
            )
            print(
                f"  Per-game history shape: {game_hist_train.shape} "
                f"(game_dim={game_hist_train.shape[-1]})"
            )

        return (
            *_train_nested_attention_nn(
                X_attn_train,
                X_attn_val,
                X_attn_test,
                hist_train,
                outer_train,
                inner_train,
                hist_val,
                outer_val,
                inner_val,
                hist_test,
                outer_test,
                inner_test,
                y_train_dict,
                y_val_dict,
                y_test_dict,
                cfg,
                targets,
                seed,
                game_hist_train=game_hist_train,
                game_hist_val=game_hist_val,
                game_hist_test=game_hist_test,
            ),
            attn_feature_cols,
        )

    history_stats = cfg.get("attn_history_stats", None)
    max_seq_len = cfg.get("attn_max_seq_len", 17)

    # Per-worker trial-data memo (see run_pipeline): the history/opp arrays
    # depend only on the prepared frames + the attn cfg below — never on
    # sampled trial hyperparams — so trials 2+ of a tune worker reuse them.
    # The fingerprint (attn cfg + row counts) guards both cfg drift and a
    # same-position different-frames caller; consumers are read-only
    # (tensors are index_select'd / narrow'd, never written).
    memo = cfg.get("trial_data_memo")
    attn_fp = None
    cached_arrays = None
    if memo is not None:
        attn_fp = {
            "history_stats": list(history_stats or []),
            "max_seq_len": max_seq_len,
            "opp_history_stats": list(cfg.get("opp_attn_history_stats") or []),
            "opp_max_seq_len": cfg.get("opp_attn_max_seq_len"),
            "opp_kind": cfg.get("opp_attn_kind", "defense"),
            "n_rows": (len(pos_train), len(pos_val), len(pos_test)),
        }
        entry = memo.get(("attn_arrays", position))
        if entry is not None and entry["fp"] == attn_fp:
            cached_arrays = entry
    if cached_arrays is not None:
        hist_train, mask_train, hist_val, mask_val, hist_test, mask_test = cached_arrays["hist"]
        (
            opp_hist_train,
            opp_mask_train,
            opp_hist_val,
            opp_mask_val,
            opp_hist_test,
            opp_mask_test,
        ) = cached_arrays["opp"]
        print(f"  History shape: {hist_train.shape} (game_dim={hist_train.shape[2]}) [trial_memo]")
        return (
            *_train_attention_nn(
                X_attn_train,
                X_attn_val,
                X_attn_test,
                hist_train,
                mask_train,
                hist_val,
                mask_val,
                hist_test,
                mask_test,
                y_train_dict,
                y_val_dict,
                y_test_dict,
                cfg,
                targets,
                seed,
                feature_cols=attn_feature_cols,
                opp_hist_train=opp_hist_train,
                opp_mask_train=opp_mask_train,
                opp_hist_val=opp_hist_val,
                opp_mask_val=opp_mask_val,
                opp_hist_test=opp_hist_test,
                opp_mask_test=opp_mask_test,
            ),
            attn_feature_cols,
        )

    hist_train, mask_train = build_game_history_arrays(
        pos_train, history_stats=history_stats, max_seq_len=max_seq_len
    )
    hist_val, mask_val = build_game_history_arrays(
        pos_val, history_stats=history_stats, max_seq_len=max_seq_len
    )
    hist_test, mask_test = build_game_history_arrays(
        pos_test, history_stats=history_stats, max_seq_len=max_seq_len
    )
    print(f"  History shape: {hist_train.shape} (game_dim={hist_train.shape[2]})")

    # Optional second attention branch: opponent-side game log. Kind-based
    # dispatch via OPP_ATTN_PER_GAME_BUILDERS — "defense" (QB/RB/WR/TE)
    # aggregates over the all-position concat of train/val/test; "offense"
    # (DST) loads the raw player-week cache because DST's train/val/test frames
    # are team-level and lack the offensive columns the offense aggregation
    # needs. Either way the downstream `build_opp_defense_history_arrays` call
    # is reused unchanged — it's generic over the per-game frame and stat list.
    opp_history_stats = cfg.get("opp_attn_history_stats")
    opp_hist_train = opp_mask_train = None
    opp_hist_val = opp_mask_val = None
    opp_hist_test = opp_mask_test = None
    if opp_history_stats:
        opp_max_seq_len = cfg.get("opp_attn_max_seq_len", max_seq_len)
        opp_attn_kind = cfg.get("opp_attn_kind", "defense")
        builder = OPP_ATTN_PER_GAME_BUILDERS[opp_attn_kind]
        if opp_attn_kind == "offense":
            weekly_cache_path = f"{CACHE_DIR}/weekly_{SEASONS[0]}_{SEASONS[-1]}.parquet"
            opp_source_df = pd.read_parquet(weekly_cache_path)
            # The raw weekly cache is written unfiltered (src/data/loader.py
            # ``_fetch_weekly``), so it carries postseason rows. Every other
            # signal in this pipeline is REG-only — the "defense" branch below
            # concats the already-``_read_split``-filtered frames. Drop playoff
            # rows here too so DST's opp-offense per-game aggregates align with
            # the rest of the REG-only worldview instead of mixing in playoff
            # games (#424). Guarded like ``_read_split`` for frames/tests that
            # lack the column.
            if "season_type" in opp_source_df.columns:
                opp_source_df = opp_source_df[opp_source_df["season_type"] == "REG"].copy()
        else:
            opp_source_df = pd.concat(list(opp_source_frames), ignore_index=True)
        opp_per_game = builder(opp_source_df)
        opp_hist_train, opp_mask_train = build_opp_defense_history_arrays(
            pos_train, opp_per_game, opp_history_stats, opp_max_seq_len
        )
        opp_hist_val, opp_mask_val = build_opp_defense_history_arrays(
            pos_val, opp_per_game, opp_history_stats, opp_max_seq_len
        )
        opp_hist_test, opp_mask_test = build_opp_defense_history_arrays(
            pos_test, opp_per_game, opp_history_stats, opp_max_seq_len
        )
        print(
            f"  Opp-{opp_attn_kind} history shape: {opp_hist_train.shape} "
            f"(opp_dim={opp_hist_train.shape[2]})"
        )

    if memo is not None:
        memo[("attn_arrays", position)] = {
            "fp": attn_fp,
            "hist": (hist_train, mask_train, hist_val, mask_val, hist_test, mask_test),
            "opp": (
                opp_hist_train,
                opp_mask_train,
                opp_hist_val,
                opp_mask_val,
                opp_hist_test,
                opp_mask_test,
            ),
        }

    return (
        *_train_attention_nn(
            X_attn_train,
            X_attn_val,
            X_attn_test,
            hist_train,
            mask_train,
            hist_val,
            mask_val,
            hist_test,
            mask_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            cfg,
            targets,
            seed,
            feature_cols=attn_feature_cols,
            opp_hist_train=opp_hist_train,
            opp_mask_train=opp_mask_train,
            opp_hist_val=opp_hist_val,
            opp_mask_val=opp_mask_val,
            opp_hist_test=opp_hist_test,
            opp_mask_test=opp_mask_test,
        ),
        attn_feature_cols,
    )


def _build_lgbm(targets, cfg, seed, n_jobs):
    """Construct a ``LightGBMMultiTarget`` from cfg hyperparameters.

    Single source of the 14-kwarg constructor shared by the holdout
    (``_train_lightgbm``) and the per-fold CV path, so the two paths cannot
    drift apart. ``n_jobs`` is threaded by the caller (holdout ``n_jobs`` param
    vs the CV fold's ``lease_cores`` lease).
    """
    return LightGBMMultiTarget(
        target_names=targets,
        n_estimators=cfg.get("lgbm_n_estimators", 500),
        learning_rate=cfg.get("lgbm_learning_rate", 0.05),
        num_leaves=cfg.get("lgbm_num_leaves", 31),
        max_depth=cfg.get("lgbm_max_depth", -1),
        subsample=cfg.get("lgbm_subsample", 0.8),
        colsample_bytree=cfg.get("lgbm_colsample_bytree", 0.8),
        reg_lambda=cfg.get("lgbm_reg_lambda", 1.0),
        reg_alpha=cfg.get("lgbm_reg_alpha", 0.0),
        min_child_samples=cfg.get("lgbm_min_child_samples", 20),
        min_split_gain=cfg.get("lgbm_min_split_gain", 0.0),
        objective=cfg.get("lgbm_objective", "huber"),
        seed=seed,
        n_jobs=n_jobs,
        # Carry the per-head clamp set so the saved model persists it (predict()
        # below still passes it explicitly, so this changes no training metric;
        # it makes the serving reload, which calls predict() without the kwarg,
        # honor the position's choice). ``None`` -> clamp every head (default).
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )


def _train_lightgbm(
    X_train,
    X_val,
    X_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    cfg,
    targets,
    feature_cols,
    seed,
    n_jobs=None,
):
    """Train a LightGBM multi-target model. Returns (model, test_preds, metrics)."""
    model = _build_lgbm(targets, cfg, seed, n_jobs)
    model.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)

    # Mirror the NN/Ridge/ElasticNet paths: honor the position's per-head
    # non-negative set so LGBM heads clamp like every other model instead of
    # leaking negative raw-stat predictions into fantasy-point aggregation.
    # ``None`` (a position that never set the knob) preserves the prior
    # clamp-every-head default inside ``LightGBMMultiTarget.predict``.
    test_preds = model.predict(X_test, non_negative_targets=cfg.get("nn_non_negative_targets"))
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)
    return model, test_preds, metrics


def _build_tabpfn(targets, cfg, seed):
    """Construct a ``TabPFNMultiTarget`` from cfg hyperparameters.

    Device follows the same FF_DEVICE-aware resolution as the NN (``cuda_enabled``):
    TabPFN runs its forward pass on CUDA when available and the operator hasn't
    pinned CPU, else CPU (TabPFN has no MPS path, so non-CUDA -> CPU).
    """
    return TabPFNMultiTarget(
        target_names=targets,
        device="cuda" if cuda_enabled() else "cpu",
        n_estimators=cfg.get("tabpfn_n_estimators", 8),
        ignore_pretraining_limits=cfg.get("tabpfn_ignore_pretraining_limits", False),
        pca_n_components=cfg.get("tabpfn_pca_components"),
        softmax_temperature=cfg.get("tabpfn_softmax_temperature", 0.9),
        auto_scale_n_estimators=cfg.get("tabpfn_auto_scale_n_estimators", True),
        inference_config=cfg.get("tabpfn_inference_config"),
        seed=seed,
        # Carry the per-head clamp set so heads clamp like every other model
        # (predict() below also passes it explicitly); ``None`` -> clamp every head.
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )


def _train_tabpfn(
    X_train,
    X_val,
    X_test,
    y_train_dict,
    y_val_dict,
    y_test_dict,
    cfg,
    targets,
    feature_cols,
    seed,
):
    """Train a TabPFN multi-target model. Returns (model, test_preds, metrics).

    ``X_val`` / ``y_val_dict`` are unused (TabPFN is in-context — no gradient
    training or early stopping) but kept in the signature to mirror
    ``_train_lightgbm`` so the holdout call site reads identically.
    """
    model = _build_tabpfn(targets, cfg, seed)
    model.fit(X_train, y_train_dict, feature_names=feature_cols)
    test_preds = model.predict(X_test, non_negative_targets=cfg.get("nn_non_negative_targets"))
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)
    return model, test_preds, metrics


def _train_elasticnet(
    X_train,
    X_test,
    y_train_dict,
    y_test_dict,
    cfg,
    targets,
    best_hparams,
):
    """Train an ElasticNet multi-target model.

    ``best_hparams`` is the dict returned by ``_tune_enet_cv`` — per-target
    ``{"alpha": ..., "l1_ratio": ...}``. Two-stage / classification heads fall
    through to their existing domain-specific classes; only vanilla targets
    use ElasticNet. Returns ``(model, test_preds, metrics)``.
    """
    alphas = {t: best_hparams[t]["alpha"] for t in best_hparams}
    l1_ratios = {t: best_hparams[t]["l1_ratio"] for t in best_hparams}
    model = ElasticNetMultiTarget(
        target_names=targets,
        alpha=alphas,
        l1_ratio=l1_ratios,
        two_stage_targets=cfg.get("two_stage_targets", {}),
        classification_targets=cfg.get("classification_targets", {}),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )
    model.fit(X_train, y_train_dict)
    test_preds = model.predict(X_test)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)
    return model, test_preds, metrics


def run_pipeline(position, cfg, train_df=None, val_df=None, test_df=None, seed=42):
    """Run the full position model pipeline.

    Args:
        position: Position abbreviation (e.g. "QB", "RB", "WR", "TE", "K", "DST")
        cfg: Dict with keys:
            # Targets & features
            targets: list[str]
            ridge_alpha_grids: dict[str, list[float]]
            specific_features: list[str]
            # Position-specific callables
            filter_fn: callable(df) -> df
            compute_targets_fn: callable(df) -> df
            add_features_fn: callable(train, val, test) -> (train, val, test)
            fill_nans_fn: callable(train, val, test, features) -> (train, val, test)
            get_feature_columns_fn: callable() -> list[str]
            # Neural net architecture
            nn_backbone_layers: list[int]
            nn_head_hidden: int
            nn_dropout: float
            nn_head_hidden_overrides: dict | None
            nn_lr: float
            nn_weight_decay: float
            nn_epochs: int
            nn_batch_size: int
            nn_patience: int
            # Loss
            loss_weights: dict[str, float]
            huber_deltas: dict[str, float]
            # Scheduler
            scheduler_type: str  ("onecycle" | "cosine_warm_restarts" | "plateau")
            + scheduler-specific keys (onecycle_max_lr, etc.)
    """
    seed_everything(seed)

    pos = position
    pos_lower = pos.lower()
    targets = cfg["targets"]
    output_dir = f"{pos_lower}/outputs"

    # Per-phase wall-clock breakdown returned in the result dict so the EC2
    # entrypoint (src/batch/train.py) can fold it into benchmark_metrics.json.
    # Without this, the `[timing]` log lines from inside run_pipeline are only
    # observable via CloudWatch scrape and not persisted alongside metrics.
    phase_seconds: dict[str, float] = {}

    # --- Load data ---
    # Per-worker trial-data memo (tune-only: src/tuning/tune_nn.py injects the
    # dict post-deepcopy; absent everywhere else). Even an in-process
    # feature-cache LRU hit still pays the per-trial parquet re-read and frame
    # content-hashing — the memo skips both on trials 2+ of a worker. Guarded
    # by a (mtime, size) stat of the split files so a splits rebuild
    # mid-process invalidates. Disk-read path only: caller-passed frames are
    # caller-owned (K/DST rebuild their own; the feature-cache LRU covers
    # their prepared tuple).
    memo = cfg.get("trial_data_memo")
    prepared = None
    frames_from_disk = train_df is None
    if frames_from_disk and memo is not None:
        entry = memo.get(("prepared", position))
        if entry is not None and entry["splits_stat"] == _splits_stat_key():
            train_df, val_df, test_df = entry["frames"]
            prepared = entry["prepared"]
            print(f"  [trial_memo] reusing prepared {position} data")
    if train_df is None:
        print("Loading general splits from disk...")
        train_df = _read_split(f"{SPLITS_DIR}/train.parquet")
        val_df = _read_split(f"{SPLITS_DIR}/val.parquet")
        test_df = _read_split(f"{SPLITS_DIR}/test.parquet")

    # --- Prepare position data ---
    print(f"Preparing {pos} data...")
    with timed("prepare_data", store=phase_seconds):
        if prepared is None:
            prepared = _prepare_position_data(position, cfg, train_df, val_df, test_df)
            if frames_from_disk and memo is not None:
                memo[("prepared", position)] = {
                    "splits_stat": _splits_stat_key(),
                    "frames": (train_df, val_df, test_df),
                    "prepared": prepared,
                }
        (
            X_train,
            X_val,
            X_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            pos_train,
            pos_val,
            pos_test,
            feature_cols,
        ) = prepared

    print(f"  Feature matrix shape: {X_train.shape}")

    # --- Baseline ---
    print(f"\n=== {pos} Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(pos_test)
    fp_truth = pos_test["fantasy_points"].to_numpy()
    baseline_metrics = {"total": compute_metrics(fp_truth, baseline_preds)}
    print(f"  Season Avg Baseline MAE: {baseline_metrics['total']['mae']:.3f}")

    # --- Trained-model branches (CPU and GPU run concurrently) ---
    # Ridge / ElasticNet / LightGBM are CPU-bound (BLAS, scikit-learn, LightGBM's
    # C++ trainer — all release the GIL); base NN and Attention NN are GPU-bound
    # (PyTorch CUDA kernels release the GIL too). Their outputs are independent
    # and only meet at the ``comparison`` dict below, so we overlap the two
    # branches via a single worker thread. Wall-clock collapses from
    # ``cpu + gpu`` to roughly ``max(cpu, gpu)``. ``phase_seconds`` is mutated
    # from both branches; CPython's GIL makes the per-key writes atomic and the
    # keys don't overlap.

    cv_col = cfg.get("cv_split_column", "season")
    two_stage_targets = cfg.get("two_stage_targets", {})
    classification_targets = cfg.get("classification_targets", {})
    pca_n = cfg.get("ridge_pca_components")
    special_targets = set(two_stage_targets) | set(classification_targets)
    ridge_tune_targets = [t for t in targets if t not in special_targets]
    ridge_tune_grids = {t: cfg["ridge_alpha_grids"][t] for t in ridge_tune_targets}

    def _cpu_branch():
        # --- Ridge multi-target with per-target alpha tuning ---
        ridge_model = None
        ridge_test_preds = None
        ridge_metrics = None
        if cfg.get("train_ridge", True):
            print(f"\n=== {pos} Ridge Multi-Target (Per-Target CV Tuning) ===")
            with lease_cores("ridge_cv") as _nj, timed("ridge_tune", store=phase_seconds):
                best_alphas = (
                    _tune_ridge_alphas_cv(
                        X_train,
                        y_train_dict,
                        pos_train[cv_col].values,
                        targets=ridge_tune_targets,
                        alpha_grids=ridge_tune_grids,
                        n_cv_folds=cfg.get("ridge_cv_folds", 4),
                        refine_points=cfg.get("ridge_refine_points", 5),
                        pca_n_components=pca_n,
                        n_jobs=_nj,
                    )
                    if ridge_tune_targets
                    else {}
                )
            if two_stage_targets:
                print(f"  Two-stage targets: {list(two_stage_targets.keys())}")
            if classification_targets:
                print(f"  Ordinal classification targets: {list(classification_targets.keys())}")
            if pca_n:
                print(f"  PCR: {pca_n} components")
            ridge_non_neg = cfg.get("nn_non_negative_targets")
            with timed("ridge_fit", store=phase_seconds):
                ridge_model = RidgeMultiTarget(
                    target_names=targets,
                    alpha=best_alphas,
                    two_stage_targets=two_stage_targets,
                    classification_targets=classification_targets,
                    pca_n_components=pca_n,
                    non_negative_targets=ridge_non_neg,
                )
                ridge_model.fit(X_train, y_train_dict)
                ridge_test_preds = ridge_model.predict(X_test)
                ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

        # --- ElasticNet (optional parallel linear baseline with L1+L2) ---
        enet_model = None
        enet_test_preds = None
        enet_metrics = None
        if cfg.get("train_elasticnet", False):
            print(f"\n=== {pos} ElasticNet Multi-Target (CV alpha + l1_ratio) ===")
            # ElasticNet shares the ridge alpha grid — no position currently
            # declares a separate ``enet_alpha_grids`` (the field isn't on
            # ``PositionConfig``). The previous silent fallback masked typos in
            # any future grid override; require the ridge grid explicitly.
            enet_tune_grids = {t: cfg["ridge_alpha_grids"][t] for t in ridge_tune_targets}
            enet_l1_ratios = cfg.get("enet_l1_ratios", [0.3, 0.5, 0.7])
            with lease_cores("enet_cv") as _nj, timed("elasticnet_tune", store=phase_seconds):
                enet_best = (
                    _tune_enet_cv(
                        X_train,
                        y_train_dict,
                        pos_train[cv_col].values,
                        targets=ridge_tune_targets,
                        alpha_grids=enet_tune_grids,
                        l1_ratios=enet_l1_ratios,
                        n_cv_folds=cfg.get("ridge_cv_folds", 4),
                        refine_points=cfg.get("ridge_refine_points", 5),
                        n_jobs=_nj,
                    )
                    if ridge_tune_targets
                    else {}
                )
            with timed("elasticnet_fit", store=phase_seconds):
                enet_model, enet_test_preds, enet_metrics = _train_elasticnet(
                    X_train,
                    X_test,
                    y_train_dict,
                    y_test_dict,
                    cfg,
                    targets,
                    enet_best,
                )
            non_converged = [
                t for t, info in enet_model.convergence_report().items() if not info["converged"]
            ]
            if non_converged:
                # Surface silent non-convergence — CV-optimal coefficients lose
                # meaning if the solver didn't actually reach tol.
                print(f"  WARNING: ElasticNet did not converge for: {non_converged}")

        # --- LightGBM Multi-Target (conditional) ---
        lgbm_test_preds = None
        lgbm_metrics = None
        lgbm_model = None
        if cfg.get("train_lightgbm", False):
            print(f"\n=== {pos} LightGBM Multi-Target ===")
            with lease_cores("lgbm", default=None) as _nj, timed("lgbm_train", store=phase_seconds):
                lgbm_model, lgbm_test_preds, lgbm_metrics = _train_lightgbm(
                    X_train,
                    X_val,
                    X_test,
                    y_train_dict,
                    y_val_dict,
                    y_test_dict,
                    cfg,
                    targets,
                    feature_cols,
                    seed,
                    n_jobs=_nj,
                )

        # --- TabPFN Multi-Target (conditional; pretrained tabular transformer) ---
        tabpfn_test_preds = None
        tabpfn_metrics = None
        tabpfn_model = None
        if cfg.get("train_tabpfn", False):
            print(f"\n=== {pos} TabPFN Multi-Target ===")
            with timed("tabpfn_train", store=phase_seconds):
                tabpfn_model, tabpfn_test_preds, tabpfn_metrics = _train_tabpfn(
                    X_train,
                    X_val,
                    X_test,
                    y_train_dict,
                    y_val_dict,
                    y_test_dict,
                    cfg,
                    targets,
                    feature_cols,
                    seed,
                )

        return {
            "ridge_model": ridge_model,
            "ridge_test_preds": ridge_test_preds,
            "ridge_metrics": ridge_metrics,
            "enet_model": enet_model,
            "enet_test_preds": enet_test_preds,
            "enet_metrics": enet_metrics,
            "lgbm_model": lgbm_model,
            "lgbm_test_preds": lgbm_test_preds,
            "lgbm_metrics": lgbm_metrics,
            "tabpfn_model": tabpfn_model,
            "tabpfn_test_preds": tabpfn_test_preds,
            "tabpfn_metrics": tabpfn_metrics,
        }

    def _gpu_branch():
        # --- Multi-head NN ---
        model = None
        nn_scaler = None
        nn_test_preds = None
        nn_metrics = None
        history = None
        if cfg.get("train_base_nn", True):
            print(f"\n=== {pos} Multi-Head Neural Net ===")
            with timed("nn_train", store=phase_seconds):
                model, nn_scaler, nn_test_preds, nn_metrics, history = _train_nn(
                    X_train,
                    X_val,
                    X_test,
                    y_train_dict,
                    y_val_dict,
                    y_test_dict,
                    cfg,
                    targets,
                    seed,
                )

        # --- Attention NN (game history as variable-length sequences) ---
        attn_nn_test_preds = None
        attn_nn_metrics = None
        attn_model = None
        attn_nn_scaler = None
        attn_history = None
        attn_feature_cols = None
        if cfg.get("train_attention_nn", False):
            print(f"\n=== {pos} Attention Multi-Head Neural Net ===")
            with timed("attn_nn_train", store=phase_seconds):
                (
                    attn_model,
                    attn_nn_scaler,
                    attn_nn_test_preds,
                    attn_nn_metrics,
                    attn_history,
                    attn_feature_cols,
                ) = _train_attention_holdout(
                    position,
                    cfg,
                    targets,
                    seed,
                    X_train,
                    X_val,
                    X_test,
                    y_train_dict,
                    y_val_dict,
                    y_test_dict,
                    pos_train,
                    pos_val,
                    pos_test,
                    feature_cols,
                    opp_source_frames=(train_df, val_df, test_df),
                )

        return {
            "model": model,
            "nn_scaler": nn_scaler,
            "nn_test_preds": nn_test_preds,
            "nn_metrics": nn_metrics,
            "history": history,
            "attn_model": attn_model,
            "attn_nn_scaler": attn_nn_scaler,
            "attn_nn_test_preds": attn_nn_test_preds,
            "attn_nn_metrics": attn_nn_metrics,
            "attn_history": attn_history,
            "attn_feature_cols": attn_feature_cols,
        }

    with (
        timed("train_models_total", store=phase_seconds),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-branch") as ex,
    ):
        cpu_future = ex.submit(_cpu_branch)
        gpu_results = _gpu_branch()
        cpu_results = cpu_future.result()  # propagates exceptions from worker

    ridge_model = cpu_results["ridge_model"]
    ridge_test_preds = cpu_results["ridge_test_preds"]
    ridge_metrics = cpu_results["ridge_metrics"]
    enet_model = cpu_results["enet_model"]
    enet_test_preds = cpu_results["enet_test_preds"]
    enet_metrics = cpu_results["enet_metrics"]
    lgbm_model = cpu_results["lgbm_model"]
    lgbm_test_preds = cpu_results["lgbm_test_preds"]
    lgbm_metrics = cpu_results["lgbm_metrics"]
    # tabpfn_model intentionally not unpacked — the pilot reads only the metrics /
    # preds and skips persisting the (large, train-set-carrying) TabPFN artifact.
    tabpfn_test_preds = cpu_results["tabpfn_test_preds"]
    tabpfn_metrics = cpu_results["tabpfn_metrics"]
    model = gpu_results["model"]
    nn_scaler = gpu_results["nn_scaler"]
    nn_test_preds = gpu_results["nn_test_preds"]
    nn_metrics = gpu_results["nn_metrics"]
    history = gpu_results["history"]
    attn_model = gpu_results["attn_model"]
    attn_nn_scaler = gpu_results["attn_nn_scaler"]
    attn_nn_test_preds = gpu_results["attn_nn_test_preds"]
    attn_nn_metrics = gpu_results["attn_nn_metrics"]
    attn_history = gpu_results["attn_history"]
    attn_feature_cols = gpu_results["attn_feature_cols"]

    # Tuning / split-branch short-circuit: Ridge or base NN may be skipped.
    # Everything below — comparison table, prediction attachment, ranking,
    # backtest, full-artifact save, figures — requires both ridge_test_preds
    # and nn_test_preds. Split Batch branch runs opt into saving the partial
    # model artifacts they did train before returning the minimal result.
    if ridge_test_preds is None or nn_test_preds is None:
        artifact_branch = cfg.get("_artifact_branch")
        if artifact_branch in {"cpu", "nn"}:
            with timed("save_artifacts", store=phase_seconds):
                os.makedirs(f"{output_dir}/models", exist_ok=True)
                if artifact_branch == "cpu":
                    if ridge_model is None:
                        raise RuntimeError("CPU split branch did not train Ridge artifacts")
                    ridge_model.save(f"{output_dir}/models")
                    if lgbm_model is not None:
                        lgbm_model.save(f"{output_dir}/models")
                else:
                    if model is None or nn_scaler is None:
                        raise RuntimeError("NN split branch did not train base NN artifacts")
                    torch.save(
                        wrap_state_dict(model.state_dict(), feature_cols, targets),
                        f"{output_dir}/models/{pos_lower}_multihead_nn.pt",
                    )
                    joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")
                    write_scaler_meta(
                        f"{output_dir}/models/nn_scaler_meta.json", feature_cols, targets
                    )
                    if attn_model is not None:
                        if cfg.get("attn_static_from_df", False):
                            attn_static_cols = attn_feature_cols
                        else:
                            attn_static_cols = get_attn_static_columns(
                                attn_feature_cols, cfg["attn_static_features"]
                            )
                        torch.save(
                            wrap_state_dict(attn_model.state_dict(), attn_static_cols, targets),
                            f"{output_dir}/models/{pos_lower}_attention_nn.pt",
                        )
                        joblib.dump(attn_nn_scaler, f"{output_dir}/models/attention_nn_scaler.pkl")
                        write_scaler_meta(
                            f"{output_dir}/models/attention_nn_scaler_meta.json",
                            attn_static_cols,
                            targets,
                        )
        result = {
            "ridge_metrics": ridge_metrics,
            "nn_metrics": nn_metrics,
            "phase_seconds": phase_seconds,
        }
        if attn_nn_metrics is not None:
            result["attn_nn_metrics"] = attn_nn_metrics
            result["attn_history"] = attn_history
        if enet_metrics is not None:
            result["elasticnet_metrics"] = enet_metrics
        if lgbm_metrics is not None:
            result["lgbm_metrics"] = lgbm_metrics
        if tabpfn_metrics is not None:
            result["tabpfn_metrics"] = tabpfn_metrics
        return result

    # --- Comparison ---
    comparison = {
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge Multi-Target": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
    }
    if enet_metrics is not None:
        comparison[f"{pos} ElasticNet"] = enet_metrics
    if attn_nn_metrics is not None:
        comparison[f"{pos} Attention NN"] = attn_nn_metrics
    if lgbm_metrics is not None:
        comparison[f"{pos} LightGBM"] = lgbm_metrics
    if tabpfn_metrics is not None:
        comparison[f"{pos} TabPFN"] = tabpfn_metrics
    print_comparison_table(comparison, position=pos, target_names=targets)

    # --- Attach predictions to test DataFrame ---
    # Totals use ``cfg["aggregate_fn"]`` so ranking metrics compare like-for-like
    # against the frontend's fantasy-point outputs; falls back to sum(heads) when
    # no aggregator is registered. The fallback is wrong-sign for K (miss heads
    # would add instead of subtract) and DST (yards_allowed would add as if
    # positive); ``validate_pipeline_config`` (src/shared/position_pipeline.py)
    # now rejects K/DST cfgs that omit ``aggregate_fn`` to surface the mistake
    # at build time. See audit-318 (W.SHARED-PIPE finding 3).
    agg = cfg.get("aggregate_fn")

    def _total(preds):
        if agg is None:
            raise ValueError(
                "Pipeline config is missing 'aggregate_fn'; a raw-stat sum is "
                "wrong-sign for negative-weighted heads (fumbles/INTs, K, DST). "
                "build_pipeline_config sets it for every position — fail loud "
                "rather than report a silently-wrong total (#369 F18)."
            )
        return agg(preds)

    pos_test = pos_test.copy()
    pos_test["pred_ridge_total"] = _total(ridge_test_preds)
    pos_test["pred_nn_total"] = _total(nn_test_preds)
    pos_test["pred_baseline"] = baseline_preds
    for t in targets:
        pos_test[f"pred_ridge_{t}"] = ridge_test_preds[t]
        pos_test[f"pred_nn_{t}"] = nn_test_preds[t]

    backtest_pred_columns = {
        "Season Avg": "pred_baseline",
        "Ridge": "pred_ridge_total",
        "Neural Net": "pred_nn_total",
    }

    ridge_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ridge_total")
    nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate:    {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:       {nn_ranking['season_avg_hit_rate']:.3f}")

    enet_ranking = None
    if enet_test_preds is not None:
        pos_test["pred_enet_total"] = _total(enet_test_preds)
        for t in targets:
            pos_test[f"pred_enet_{t}"] = enet_test_preds[t]
        backtest_pred_columns["ElasticNet"] = "pred_enet_total"
        enet_ranking = compute_ranking_metrics(pos_test, pred_col="pred_enet_total")
        print(f"ElasticNet Top-12 Hit Rate: {enet_ranking['season_avg_hit_rate']:.3f}")

    attn_nn_ranking = None
    if attn_nn_test_preds is not None:
        pos_test["pred_attn_nn_total"] = _total(attn_nn_test_preds)
        for t in targets:
            pos_test[f"pred_attn_nn_{t}"] = attn_nn_test_preds[t]
        backtest_pred_columns["Attention NN"] = "pred_attn_nn_total"
        attn_nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_attn_nn_total")
        print(f"Attention NN Top-12 Hit Rate: {attn_nn_ranking['season_avg_hit_rate']:.3f}")

    lgbm_ranking = None
    if lgbm_test_preds is not None:
        pos_test["pred_lgbm_total"] = _total(lgbm_test_preds)
        for t in targets:
            pos_test[f"pred_lgbm_{t}"] = lgbm_test_preds[t]
        backtest_pred_columns["LightGBM"] = "pred_lgbm_total"
        lgbm_ranking = compute_ranking_metrics(pos_test, pred_col="pred_lgbm_total")
        print(f"LightGBM Top-12 Hit Rate: {lgbm_ranking['season_avg_hit_rate']:.3f}")

    tabpfn_ranking = None
    if tabpfn_test_preds is not None:
        pos_test["pred_tabpfn_total"] = _total(tabpfn_test_preds)
        for t in targets:
            pos_test[f"pred_tabpfn_{t}"] = tabpfn_test_preds[t]
        backtest_pred_columns["TabPFN"] = "pred_tabpfn_total"
        tabpfn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_tabpfn_total")
        print(f"TabPFN Top-12 Hit Rate: {tabpfn_ranking['season_avg_hit_rate']:.3f}")

    # --- Weekly backtest ---
    print("\n=== Weekly Backtest ===")
    with timed("backtest", store=phase_seconds):
        sim_results = run_weekly_simulation(pos_test, pred_columns=backtest_pred_columns)
        for model_name, summary in sim_results["season_summary"].items():
            print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # --- Save outputs ---
    with timed("save_artifacts", store=phase_seconds):
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

        ridge_model.save(f"{output_dir}/models")
        if enet_model is not None:
            enet_model.save(f"{output_dir}/models/elasticnet")
        torch.save(
            wrap_state_dict(model.state_dict(), feature_cols, targets),
            f"{output_dir}/models/{pos_lower}_multihead_nn.pt",
        )
        joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")
        write_scaler_meta(f"{output_dir}/models/nn_scaler_meta.json", feature_cols, targets)

        if attn_model is not None:
            # Persist the exact column list the attention scaler was fit on.
            # Mirror the two trainer paths:
            #   * attn_static_from_df=True  (K): _train_nested_attention_nn fits
            #     on attn_feature_cols as-is (already cfg["attn_static_features"]).
            #   * attn_static_from_df=False (QB/RB/WR/TE): _train_attention_nn
            #     filters X internally via get_attn_static_columns(feature_cols,
            #     attn_static_features) before fitting, so we mirror that here.
            # Reusing attn_feature_cols unconditionally would over-report
            # n_features for the latter path and trip assert_scaler_matches.
            if cfg.get("attn_static_from_df", False):
                attn_static_cols = attn_feature_cols
            else:
                attn_static_cols = get_attn_static_columns(
                    attn_feature_cols, cfg["attn_static_features"]
                )
            torch.save(
                wrap_state_dict(attn_model.state_dict(), attn_static_cols, targets),
                f"{output_dir}/models/{pos_lower}_attention_nn.pt",
            )
            joblib.dump(attn_nn_scaler, f"{output_dir}/models/attention_nn_scaler.pkl")
            write_scaler_meta(
                f"{output_dir}/models/attention_nn_scaler_meta.json",
                attn_static_cols,
                targets,
            )

        if lgbm_model is not None:
            lgbm_model.save(f"{output_dir}/models")

    with timed("figures", store=phase_seconds):
        plot_training_curves(
            history, targets, f"{output_dir}/figures/{pos_lower}_training_curves.png"
        )
        if attn_history is not None:
            plot_training_curves(
                attn_history,
                targets,
                f"{output_dir}/figures/{pos_lower}_attention_training_curves.png",
            )
        plot_weekly_accuracy(sim_results, pos, f"{output_dir}/figures/{pos_lower}_weekly_mae.png")
        plot_pred_vs_actual(
            y_test_dict,
            nn_test_preds,
            targets,
            f"{pos} Multi-Head NN",
            f"{output_dir}/figures/{pos_lower}_pred_vs_actual_scatter.png",
        )

        feature_importance = ridge_model.get_feature_importance(feature_cols)
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
        if len(targets) == 1:
            axes = [axes]
        for ax, (target, importance) in zip(axes, feature_importance.items(), strict=False):
            importance.head(15).plot(kind="barh", ax=ax)
            ax.set_title(f"Ridge: {target} Top-15 Features")
            ax.set_xlabel("Absolute Coefficient")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{pos_lower}_ridge_feature_importance.png", dpi=150)
        plt.close()

        if lgbm_model is not None:
            lgbm_importance = lgbm_model.get_feature_importance(feature_cols)
            fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
            if len(targets) == 1:
                axes = [axes]
            for ax, (target, importance) in zip(axes, lgbm_importance.items(), strict=False):
                importance.head(15).plot(kind="barh", ax=ax)
                ax.set_title(f"LightGBM: {target} Top-15 Features")
                ax.set_xlabel("Gain")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/figures/{pos_lower}_lgbm_feature_importance.png", dpi=150)
            plt.close()

    print(f"\n{pos} pipeline complete. Outputs saved to {output_dir}/")
    per_target_preds = {
        "ridge": ridge_test_preds,
        "nn": nn_test_preds,
    }
    if enet_test_preds is not None:
        per_target_preds["elasticnet"] = enet_test_preds
    if attn_nn_test_preds is not None:
        per_target_preds["attn_nn"] = attn_nn_test_preds
    if lgbm_test_preds is not None:
        per_target_preds["lgbm"] = lgbm_test_preds
    if tabpfn_test_preds is not None:
        per_target_preds["tabpfn"] = tabpfn_test_preds

    result = {
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
        "test_df": pos_test,
        "per_target_preds": per_target_preds,
        "phase_seconds": phase_seconds,
    }
    if enet_metrics is not None:
        result["elasticnet_metrics"] = enet_metrics
        result["elasticnet_ranking"] = enet_ranking
    if attn_nn_metrics is not None:
        result["attn_nn_metrics"] = attn_nn_metrics
        result["attn_nn_ranking"] = attn_nn_ranking
        # Per-epoch attention training curves (dict of "val_loss" / "train_loss"
        # / "val_mae_{t}" / "val_loss_{t}" lists). Exposed for src/tuning/
        # tune_nn.py — the tuner uses min(history["val_loss"]) as the Optuna
        # trial objective so the search optimizes against val, not the leakage-
        # prone test metrics above.
        result["attn_history"] = attn_history
    if lgbm_metrics is not None:
        result["lgbm_metrics"] = lgbm_metrics
        result["lgbm_ranking"] = lgbm_ranking
    if tabpfn_metrics is not None:
        result["tabpfn_metrics"] = tabpfn_metrics
        result["tabpfn_ranking"] = tabpfn_ranking
    return result


def run_cv_pipeline(position, cfg, full_df=None, test_df=None, seed=42):
    """Run expanding-window cross-validation, then final holdout evaluation.

    CV folds determine best Ridge alpha and report multi-season metrics for
    both Ridge and NN.  After CV, retrains on all pre-test data and evaluates
    on the holdout test set.

    Args:
        position: Position abbreviation (e.g. "QB")
        cfg: Position config dict (same as run_pipeline)
        full_df: Combined DataFrame containing all seasons (train + val).
                 If None, loads train + val from disk and concatenates.
        test_df: Holdout test DataFrame. If None, loads from disk.
        seed: Random seed
    """
    seed_everything(seed)

    pos = position
    pos_lower = pos.lower()
    targets = cfg["targets"]
    output_dir = f"{pos_lower}/outputs"

    # --- Load data ---
    if full_df is None:
        print("Loading splits from disk and combining for CV...")
        train_df = _read_split(f"{SPLITS_DIR}/train.parquet")
        val_df = _read_split(f"{SPLITS_DIR}/val.parquet")
        full_df = pd.concat([train_df, val_df], ignore_index=True)
    if test_df is None:
        test_df = _read_split(f"{SPLITS_DIR}/test.parquet")

    # --- Generate CV folds ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Expanding-Window Cross-Validation")
    print(f"{'=' * 60}")
    folds = expanding_window_folds(full_df)

    # --- Tune Ridge alphas once on full training data ---
    print("\nTuning Ridge alphas on full training data...")
    alpha_train_df = full_df[full_df["season"].isin(TRAIN_SEASONS)].copy()
    alpha_val_df = full_df[full_df["season"].isin(VAL_SEASONS)].copy()
    X_alpha, _, y_alpha_dict, _, pos_alpha, _, _ = _prepare_train_val(
        position,
        cfg,
        alpha_train_df,
        alpha_val_df,
    )
    cv_col = cfg.get("cv_split_column", "season")
    cv_special = set(cfg.get("two_stage_targets", {})) | set(cfg.get("classification_targets", {}))
    cv_ridge_targets = [t for t in targets if t not in cv_special]
    cv_ridge_grids = {t: cfg["ridge_alpha_grids"][t] for t in cv_ridge_targets}
    # Lease cores around the parallel alpha CV (mirrors run_pipeline's
    # ``lease_cores("ridge_cv")``) and tune on the SAME PCA basis the fold +
    # final Ridge models are fit on — without ``pca_n_components`` the alphas
    # were selected on the raw-feature basis while the models use PCA, so for
    # PCA positions (RB/WR/DST) CV picked alphas for the wrong basis (#386).
    with lease_cores("ridge_cv") as _nj:
        best_alphas = _tune_ridge_alphas_cv(
            X_alpha,
            y_alpha_dict,
            pos_alpha[cv_col].values,
            targets=cv_ridge_targets,
            alpha_grids=cv_ridge_grids,
            n_cv_folds=cfg.get("ridge_cv_folds", 4),
            refine_points=cfg.get("ridge_refine_points", 5),
            pca_n_components=cfg.get("ridge_pca_components"),
            n_jobs=_nj,
        )

    # --- Per-fold training ---
    fold_nn_metrics = []
    fold_ridge_metrics = []
    fold_lgbm_metrics = []

    for fold_idx, fold_train_df, fold_val_df in folds:
        print(f"\n--- Fold {fold_idx + 1} ---")
        seed_everything(seed)

        X_train, X_val, y_train_dict, y_val_dict, pos_train, pos_val, feature_cols = (
            _prepare_train_val(
                position,
                cfg,
                fold_train_df,
                fold_val_df,
            )
        )

        ridge_fold = RidgeMultiTarget(
            target_names=targets,
            alpha=best_alphas,
            two_stage_targets=cfg.get("two_stage_targets", {}),
            classification_targets=cfg.get("classification_targets", {}),
            pca_n_components=cfg.get("ridge_pca_components"),
            non_negative_targets=cfg.get("nn_non_negative_targets"),
        )
        ridge_fold.fit(X_train, y_train_dict)
        ridge_val_preds = ridge_fold.predict(X_val)
        fold_ridge_metrics.append(compute_target_metrics(y_val_dict, ridge_val_preds, targets))

        # NN training for this fold
        _, (X_train_s, X_val_s) = _scale_xs(X_train, X_val)

        device = _nn_device()
        train_loader, val_loader = make_dataloaders(
            X_train_s,
            y_train_dict,
            X_val_s,
            y_val_dict,
            batch_size=cfg["nn_batch_size"],
            device=device,
        )

        model = build_multihead_net(cfg, input_dim=X_train_s.shape[1], targets=targets).to(device)

        _run_nn_training(
            model=_maybe_compile(model),
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            targets=targets,
            trainer_cls=MultiHeadTrainer,
            lr=cfg["nn_lr"],
            weight_decay=cfg["nn_weight_decay"],
            patience=cfg["nn_patience"],
        )

        nn_val_preds = model.predict_numpy(X_val_s, device)
        fold_nn_metrics.append(compute_target_metrics(y_val_dict, nn_val_preds, targets))

        # LightGBM for this fold
        if cfg.get("train_lightgbm", False):
            # Lease cores + thread n_jobs/LGBM_N_JOBS like the holdout path
            # (#818/#819); clamp the val preds with non_negative_targets so the
            # CV-fold metrics match the holdout predict, which supplies it
            # (#479/#787).
            with lease_cores("lgbm", default=None) as _nj:
                lgbm_fold = _build_lgbm(targets, cfg, seed, _nj)
                lgbm_fold.fit(X_train, y_train_dict, X_val, y_val_dict, feature_names=feature_cols)
            lgbm_val_preds = lgbm_fold.predict(
                X_val, non_negative_targets=cfg.get("nn_non_negative_targets")
            )
            fold_lgbm_metrics.append(compute_target_metrics(y_val_dict, lgbm_val_preds, targets))

    # --- Aggregate CV results ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Cross-Validation Results ({len(folds)} folds)")
    print(f"{'=' * 60}")

    # Final per-target alphas: tune on full pre-test training data
    print("\nFinal per-target Ridge alpha tuning (full training data)...")

    # Aggregate per-fold metrics
    cv_metrics = {"ridge": {}, "nn": {}}
    model_fold_pairs = [("ridge", fold_ridge_metrics), ("nn", fold_nn_metrics)]
    if fold_lgbm_metrics:
        cv_metrics["lgbm"] = {}
        model_fold_pairs.append(("lgbm", fold_lgbm_metrics))
    for model_name, fold_metrics_list in model_fold_pairs:
        for key in ["total"] + targets:
            maes = [fm[key]["mae"] for fm in fold_metrics_list]
            r2s = [fm[key]["r2"] for fm in fold_metrics_list]
            cv_metrics[model_name][key] = {
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "r2_mean": np.mean(r2s),
                "r2_std": np.std(r2s),
                "mae_per_fold": maes,
                "r2_per_fold": r2s,
            }

    cv_model_names = ["ridge", "nn"] + (["lgbm"] if fold_lgbm_metrics else [])
    print(f"\n{'Model':<12} {'MAE (mean +/- std)':>22} {'R2 (mean +/- std)':>22}")
    print("-" * 58)
    for model_name in cv_model_names:
        m = cv_metrics[model_name]["total"]
        print(
            f"{model_name.upper():<12} {m['mae_mean']:>8.3f} +/- {m['mae_std']:<8.3f} "
            f"{m['r2_mean']:>8.3f} +/- {m['r2_std']:<8.3f}"
        )

    # --- Final holdout evaluation ---
    print(f"\n{'=' * 60}")
    print(f"  {pos} Final Holdout Evaluation (test on 2025)")
    print(f"{'=' * 60}")

    # Build final train = all pre-test data, val = last CV fold (2024) for NN early stopping
    final_train_seasons = TRAIN_SEASONS
    final_val_seasons = VAL_SEASONS
    final_train_df = full_df[full_df["season"].isin(final_train_seasons)].copy()
    final_val_df = full_df[full_df["season"].isin(final_val_seasons)].copy()

    (
        X_train,
        X_val,
        X_test,
        y_train_dict,
        y_val_dict,
        y_test_dict,
        pos_train,
        pos_val,
        pos_test,
        feature_cols,
    ) = _prepare_position_data(position, cfg, final_train_df, final_val_df, test_df)

    # Baseline
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(pos_test)
    fp_truth = pos_test["fantasy_points"].to_numpy()
    baseline_metrics = {"total": compute_metrics(fp_truth, baseline_preds)}

    # Ridge with per-target CV alphas tuned on full training data
    best_cv_alphas = best_alphas

    ridge_model = RidgeMultiTarget(
        target_names=targets,
        alpha=best_cv_alphas,
        two_stage_targets=cfg.get("two_stage_targets", {}),
        classification_targets=cfg.get("classification_targets", {}),
        pca_n_components=cfg.get("ridge_pca_components"),
        non_negative_targets=cfg.get("nn_non_negative_targets"),
    )
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_metrics = compute_target_metrics(y_test_dict, ridge_test_preds, targets)

    # ElasticNet (optional parallel linear baseline with L1+L2) — mirrors
    # run_pipeline so CV mode reports the same model when train_elasticnet=True.
    enet_metrics = None
    enet_test_preds = None
    if cfg.get("train_elasticnet", False):
        print(f"\n=== {pos} ElasticNet Multi-Target (Final Holdout) ===")
        enet_tune_grids = {t: cfg["ridge_alpha_grids"][t] for t in cv_ridge_targets}
        enet_l1_ratios = cfg.get("enet_l1_ratios", [0.3, 0.5, 0.7])
        if cv_ridge_targets:
            with lease_cores("enet_cv") as _nj:
                enet_best = _tune_enet_cv(
                    X_train,
                    y_train_dict,
                    pos_train[cv_col].values,
                    targets=cv_ridge_targets,
                    alpha_grids=enet_tune_grids,
                    l1_ratios=enet_l1_ratios,
                    n_cv_folds=cfg.get("ridge_cv_folds", 4),
                    refine_points=cfg.get("ridge_refine_points", 5),
                    n_jobs=_nj,
                )
        else:
            enet_best = {}
        enet_model, enet_test_preds, enet_metrics = _train_elasticnet(
            X_train,
            X_test,
            y_train_dict,
            y_test_dict,
            cfg,
            targets,
            enet_best,
        )
        non_converged = [
            t for t, info in enet_model.convergence_report().items() if not info["converged"]
        ]
        if non_converged:
            print(f"  WARNING: ElasticNet did not converge for: {non_converged}")

    # NN
    print(f"\n=== {pos} Multi-Head NN (Final Holdout) ===")
    model, nn_scaler, nn_test_preds, nn_metrics, history = _train_nn(
        X_train,
        X_val,
        X_test,
        y_train_dict,
        y_val_dict,
        y_test_dict,
        cfg,
        targets,
        seed,
    )

    # Attention NN (game history as sequences) — mirrors run_pipeline so CV mode
    # reports attention metrics when train_attention_nn=True (QB/RB/WR in prod).
    # All-position frames for the opponent-defense builder are the season-sliced
    # full frames + holdout test (they retain every position, unlike pos_*).
    attn_nn_metrics = None
    attn_test_preds = None
    attn_model = None
    attn_nn_scaler = None
    attn_static_cols = None
    if cfg.get("train_attention_nn", False):
        print(f"\n=== {pos} Attention Multi-Head Neural Net (Final Holdout) ===")
        (
            attn_model,
            attn_nn_scaler,
            attn_test_preds,
            attn_nn_metrics,
            _attn_history,
            attn_static_cols,
        ) = _train_attention_holdout(
            position,
            cfg,
            targets,
            seed,
            X_train,
            X_val,
            X_test,
            y_train_dict,
            y_val_dict,
            y_test_dict,
            pos_train,
            pos_val,
            pos_test,
            feature_cols,
            opp_source_frames=(final_train_df, final_val_df, test_df),
        )

    # LightGBM
    lgbm_test_preds = None
    lgbm_metrics = None
    lgbm_model = None
    if cfg.get("train_lightgbm", False):
        print(f"\n=== {pos} LightGBM Multi-Target (Final Holdout) ===")
        # Lease cores + thread n_jobs like the holdout path (#818/#819).
        with lease_cores("lgbm", default=None) as _nj:
            lgbm_model, lgbm_test_preds, lgbm_metrics = _train_lightgbm(
                X_train,
                X_val,
                X_test,
                y_train_dict,
                y_val_dict,
                y_test_dict,
                cfg,
                targets,
                feature_cols,
                seed,
                n_jobs=_nj,
            )

    # Comparison
    comparison = {
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge (per-target CV alphas)": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
    }
    if enet_metrics is not None:
        comparison[f"{pos} ElasticNet"] = enet_metrics
    if attn_nn_metrics is not None:
        comparison[f"{pos} Attention NN"] = attn_nn_metrics
    if lgbm_metrics is not None:
        comparison[f"{pos} LightGBM"] = lgbm_metrics
    print_comparison_table(comparison, position=pos, target_names=targets)

    # Ranking metrics (totals via aggregate_fn for fantasy-point-scale comparison)
    agg = cfg.get("aggregate_fn")

    def _total(preds):
        if agg is None:
            raise ValueError(
                "Pipeline config is missing 'aggregate_fn'; a raw-stat sum is "
                "wrong-sign for negative-weighted heads (fumbles/INTs, K, DST). "
                "build_pipeline_config sets it for every position — fail loud "
                "rather than report a silently-wrong total (#369 F18)."
            )
        return agg(preds)

    pos_test = pos_test.copy()
    pos_test["pred_ridge_total"] = _total(ridge_test_preds)
    pos_test["pred_nn_total"] = _total(nn_test_preds)
    pos_test["pred_baseline"] = baseline_preds

    backtest_pred_columns = {
        "Season Avg": "pred_baseline",
        "Ridge": "pred_ridge_total",
        "Neural Net": "pred_nn_total",
    }

    ridge_ranking = compute_ranking_metrics(pos_test, pred_col="pred_ridge_total")
    nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_nn_total")
    print(f"\nRidge Top-12 Hit Rate:    {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:       {nn_ranking['season_avg_hit_rate']:.3f}")

    # ElasticNet + Attention NN rankings — mirror run_pipeline so CV mode reports
    # their Top-12 hit rate (and backtests them) instead of silently dropping
    # both (previously their test preds were discarded → attn_nn_top12 == 0).
    enet_ranking = None
    if enet_test_preds is not None:
        pos_test["pred_enet_total"] = _total(enet_test_preds)
        for t in targets:
            pos_test[f"pred_enet_{t}"] = enet_test_preds[t]
        backtest_pred_columns["ElasticNet"] = "pred_enet_total"
        enet_ranking = compute_ranking_metrics(pos_test, pred_col="pred_enet_total")
        print(f"ElasticNet Top-12 Hit Rate: {enet_ranking['season_avg_hit_rate']:.3f}")

    attn_nn_ranking = None
    if attn_test_preds is not None:
        pos_test["pred_attn_nn_total"] = _total(attn_test_preds)
        for t in targets:
            pos_test[f"pred_attn_nn_{t}"] = attn_test_preds[t]
        backtest_pred_columns["Attention NN"] = "pred_attn_nn_total"
        attn_nn_ranking = compute_ranking_metrics(pos_test, pred_col="pred_attn_nn_total")
        print(f"Attention NN Top-12 Hit Rate: {attn_nn_ranking['season_avg_hit_rate']:.3f}")

    lgbm_ranking = None
    if lgbm_test_preds is not None:
        pos_test["pred_lgbm_total"] = _total(lgbm_test_preds)
        for t in targets:
            pos_test[f"pred_lgbm_{t}"] = lgbm_test_preds[t]
        backtest_pred_columns["LightGBM"] = "pred_lgbm_total"
        lgbm_ranking = compute_ranking_metrics(pos_test, pred_col="pred_lgbm_total")
        print(f"LightGBM Top-12 Hit Rate: {lgbm_ranking['season_avg_hit_rate']:.3f}")

    # Weekly backtest
    print("\n=== Weekly Backtest ===")
    sim_results = run_weekly_simulation(pos_test, pred_columns=backtest_pred_columns)
    for model_name, summary in sim_results["season_summary"].items():
        print(f"  {model_name}: MAE={summary['mae']:.3f}, R2={summary['r2']:.3f}")

    # Save outputs
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    ridge_model.save(f"{output_dir}/models")
    torch.save(
        wrap_state_dict(model.state_dict(), feature_cols, targets),
        f"{output_dir}/models/{pos_lower}_multihead_nn.pt",
    )
    joblib.dump(nn_scaler, f"{output_dir}/models/nn_scaler.pkl")
    write_scaler_meta(f"{output_dir}/models/nn_scaler_meta.json", feature_cols, targets)

    # Persist the attention NN too (mirrors run_pipeline) — without this a
    # CV-built model dir is missing ``{pos}_attention_nn.pt`` and serving/upload
    # for the attention model fails. ``attn_static_cols`` is the exact column
    # list the attention scaler was fit on (returned by _train_attention_holdout).
    if attn_model is not None:
        torch.save(
            wrap_state_dict(attn_model.state_dict(), attn_static_cols, targets),
            f"{output_dir}/models/{pos_lower}_attention_nn.pt",
        )
        joblib.dump(attn_nn_scaler, f"{output_dir}/models/attention_nn_scaler.pkl")
        write_scaler_meta(
            f"{output_dir}/models/attention_nn_scaler_meta.json",
            attn_static_cols,
            targets,
        )

    if lgbm_model is not None:
        lgbm_model.save(f"{output_dir}/models")

    plot_training_curves(history, targets, f"{output_dir}/figures/{pos_lower}_training_curves.png")
    plot_weekly_accuracy(sim_results, pos, f"{output_dir}/figures/{pos_lower}_weekly_mae.png")
    plot_pred_vs_actual(
        y_test_dict,
        nn_test_preds,
        targets,
        f"{pos} Multi-Head NN",
        f"{output_dir}/figures/{pos_lower}_pred_vs_actual_scatter.png",
    )

    feature_importance = ridge_model.get_feature_importance(feature_cols)
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
    if len(targets) == 1:
        axes = [axes]
    for ax, (target, importance) in zip(axes, feature_importance.items(), strict=False):
        importance.head(15).plot(kind="barh", ax=ax)
        ax.set_title(f"Ridge: {target} Top-15 Features")
        ax.set_xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/{pos_lower}_ridge_feature_importance.png", dpi=150)
    plt.close()

    if lgbm_model is not None:
        lgbm_importance = lgbm_model.get_feature_importance(feature_cols)
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 8))
        if len(targets) == 1:
            axes = [axes]
        for ax, (target, importance) in zip(axes, lgbm_importance.items(), strict=False):
            importance.head(15).plot(kind="barh", ax=ax)
            ax.set_title(f"LightGBM: {target} Top-15 Features")
            ax.set_xlabel("Gain")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/{pos_lower}_lgbm_feature_importance.png", dpi=150)
        plt.close()

    print(f"\n{pos} CV pipeline complete. Outputs saved to {output_dir}/")
    per_target_preds = {
        "ridge": ridge_test_preds,
        "nn": nn_test_preds,
    }
    if enet_test_preds is not None:
        per_target_preds["elasticnet"] = enet_test_preds
    if attn_test_preds is not None:
        per_target_preds["attn_nn"] = attn_test_preds
    if lgbm_test_preds is not None:
        per_target_preds["lgbm"] = lgbm_test_preds

    result = {
        "cv_metrics": cv_metrics,
        "best_cv_alphas": best_cv_alphas,
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
        "sim_results": sim_results,
        "test_df": pos_test,
        "per_target_preds": per_target_preds,
    }
    if lgbm_metrics is not None:
        result["lgbm_metrics"] = lgbm_metrics
        result["lgbm_ranking"] = lgbm_ranking
    if enet_metrics is not None:
        result["elasticnet_metrics"] = enet_metrics
        result["elasticnet_ranking"] = enet_ranking
    if attn_nn_metrics is not None:
        result["attn_nn_metrics"] = attn_nn_metrics
        result["attn_nn_ranking"] = attn_nn_ranking
    return result
