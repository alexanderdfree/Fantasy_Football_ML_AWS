"""One tiny RB GPU pipeline plus actual hurdle/validation acceptance probes."""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["RB"]
SEEDS = [42]
BASELINE = "verify"


def _tiny_config(cfg):
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
        train_attention_nn=True,
    )
    return cfg


def metric_fn(result, position):
    from src.analysis.verify_hurdle_expectations import verify_hurdle_expectations
    from src.analysis.verify_validation_reduction import verify_validation_reduction

    if position != "RB" or not result.get("attn_nn_metrics") or result["test_df"].empty:
        raise RuntimeError("The normal RB pipeline must produce attention and test results")
    return {
        "hurdle_expectations": verify_hurdle_expectations(seed=SEEDS[0]),
        "validation_reduction": verify_validation_reduction(seed=SEEDS[0]),
        "pipeline": {
            "attention_mae": float(result["attn_nn_metrics"]["total"]["mae"]),
            "test_rows": float(len(result["test_df"])),
        },
    }


VARIANTS = [Variant(BASELINE, cfg_mutator=_tiny_config, label="Hurdle and validation GPU checks")]


if __name__ == "__main__":
    raise SystemExit(ab_main(__spec__.name))
