"""One tiny QB cell plus actual-CUDA capture rollback/replay verification.

Submit with the existing ``src.tuning.launch_ab`` route after building this
unmerged image. This spec imports no torch and executes no probe during
``--list`` or ``--dry-run``. Use eager seed execution, not stacked seeds.
"""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB"]
SEEDS = [42]
BASELINE = "verify"


def _tiny_config(cfg):
    from src.qb.config import CONFIG_TINY

    cfg.update(CONFIG_TINY)
    cfg.update(
        nn_backbone_layers=[8, 8],
        nn_head_hidden=4,
        nn_head_hidden_overrides=None,
        nn_dropout=0.0,
        attn_dropout=0.0,
        nn_epochs=1,
        nn_patience=1,
        nn_batch_size=64,
        nn_log_every=100,
        ridge_alpha_grids={target: [1.0] for target in cfg["targets"]},
        ridge_cv_folds=2,
        ridge_refine_points=0,
        ridge_pca_components=None,
        train_base_nn=True,
        train_ridge=True,
        train_elasticnet=False,
        train_lightgbm=False,
        train_attention_nn=False,
    )
    return cfg


def metric_fn(result, position):
    from src.analysis.verify_cuda_capture_rollback import verify_cuda_capture_rollback

    if position != "QB":
        raise ValueError("The capture proof spec is scoped to the tiny QB pipeline cell")
    rows = len(result["test_df"])
    if rows == 0 or not result.get("nn_metrics"):
        raise RuntimeError("The normal tiny QB pipeline did not produce test predictions")
    proof = verify_cuda_capture_rollback(seed=SEEDS[0])
    return {
        "cuda_capture": proof,
        "NN": {"mae": float(result["nn_metrics"]["total"]["mae"]), "test_rows": float(rows)},
    }


VARIANTS = [Variant(BASELINE, cfg_mutator=_tiny_config, label="CUDA capture rollback and replay")]


if __name__ == "__main__":
    raise SystemExit(ab_main(__spec__.name))
