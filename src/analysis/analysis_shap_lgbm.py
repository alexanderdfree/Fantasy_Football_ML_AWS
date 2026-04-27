"""SHAP diagnostic for trained LightGBM models.

Loads the already-trained per-position LightGBM booster, rebuilds the train
split via ``src.shared.pipeline.build_train_matrix`` (the same setup the pipeline
used, so there's no drift between training-time and explain-time feature
matrices), and emits per-target SHAP summary plots plus a JSON ranking of
mean absolute SHAP by feature.

Stays out of ``run_pipeline`` on purpose — SHAP on LightGBM with many trees
is slow, and the goal is diagnostic insight (is the whitelist pruned right?),
not production telemetry. Run on demand after training.

Usage:
    python analysis_shap_lgbm.py QB
    python analysis_shap_lgbm.py QB RB WR --background-samples 500
    python analysis_shap_lgbm.py QB --targets passing_yards
"""

import argparse
import datetime
import importlib
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.shared.models import LightGBMMultiTarget  # noqa: E402
from src.shared.pipeline import build_train_matrix  # noqa: E402


def _load_position_config(pos: str) -> dict:
    """Import the position's CONFIG dict from its runner module.

    Position runners expose ``{POS}_CONFIG`` (e.g. ``QB_CONFIG``) by the time
    they're imported; pulling it off the module gives the same cfg the
    pipeline used to fit the saved LightGBM.
    """
    pos_lower = pos.lower()
    mod = importlib.import_module(f"src.{pos_lower}.run_pipeline")
    cfg_name = f"{pos}_CONFIG"
    if not hasattr(mod, cfg_name):
        raise AttributeError(f"{mod.__name__} has no {cfg_name}")
    return getattr(mod, cfg_name)


def _model_trained_at(model_dir: str) -> str | None:
    """mtime of the saved LightGBM meta.json (ISO string, or None)."""
    meta_path = f"{model_dir}/lightgbm/meta.json"
    if not os.path.exists(meta_path):
        return None
    return datetime.datetime.fromtimestamp(os.path.getmtime(meta_path)).isoformat()


def _sample_background(X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Deterministic background subsample.

    SHAP plots need a representative sample of feature space; the same
    (seed, n_samples) must yield the same rows so plot-diffs and JSON-diffs
    between runs reflect model changes, not sampling noise.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    take = min(n_samples, n)
    idx = rng.choice(n, size=take, replace=False)
    return X[idx]


def _run_shap_for_position(
    pos: str,
    *,
    target_filter: list[str] | None,
    background_samples: int,
    seed: int,
    output_dir: str | None,
) -> str:
    """Run SHAP for one position. Returns path to the written ranking JSON."""
    import shap  # local import so the test smoke path doesn't force the dep

    cfg = _load_position_config(pos)
    pos_lower = pos.lower()
    model_dir = output_dir or f"{pos}/outputs"
    figures_dir = f"{model_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n=== SHAP ({pos}) ===")
    print(f"  Loading model from {model_dir}/models/lightgbm/")
    model = LightGBMMultiTarget(target_names=cfg["targets"])
    model.load(f"{model_dir}/models")

    print("  Rebuilding train matrix...")
    X_train, _, feature_cols = build_train_matrix(pos, cfg)
    background = _sample_background(X_train, background_samples, seed)
    X_bg_df = pd.DataFrame(background, columns=feature_cols)
    print(f"  Background: {background.shape[0]} rows x {background.shape[1]} features")

    trained_at = _model_trained_at(f"{model_dir}/models")
    ranking: dict = {
        "_meta": {
            "position": pos,
            "shap_computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_trained_at": trained_at,
            "background_samples": int(background.shape[0]),
            "seed": int(seed),
            "n_features": int(background.shape[1]),
        }
    }

    targets = list(model.target_names)
    if target_filter:
        missing = set(target_filter) - set(targets)
        if missing:
            raise ValueError(f"{pos} has no targets named: {missing}")
        targets = [t for t in targets if t in target_filter]

    for target in targets:
        booster = model._models[target]
        print(f"  [{target}] computing SHAP values...")
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_bg_df)

        mean_abs = np.abs(shap_values).mean(axis=0)
        target_ranking = {
            name: float(val)
            for name, val in sorted(
                zip(feature_cols, mean_abs, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
        }
        ranking[target] = target_ranking

        # Bar + beeswarm summary plot. A single PNG per target keeps artifacts
        # easy to diff visually; the JSON ranking is the source of truth for
        # any automated downstream use.
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_bg_df,
            feature_names=feature_cols,
            show=False,
            plot_type="dot",
            max_display=20,
        )
        plt.title(f"{pos} LightGBM SHAP — {target}")
        plt.tight_layout()
        out_path = f"{figures_dir}/{pos_lower}_shap_summary_{target}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {out_path}")

    ranking_path = f"{model_dir}/{pos_lower}_shap_ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"  Ranking JSON: {ranking_path}")
    return ranking_path


def main():
    parser = argparse.ArgumentParser(
        description="SHAP feature importance diagnostic for trained LightGBM models."
    )
    parser.add_argument(
        "positions",
        nargs="+",
        help="Positions to analyze (QB RB WR TE K DST).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional subset of targets to explain (default: all targets per position).",
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=1000,
        help="Rows sampled from the training matrix for SHAP summary plots (default 1000, min 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling the deterministic background subsample.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the model/figures directory (default: {POS}/outputs).",
    )
    args = parser.parse_args()

    if args.background_samples < 100:
        parser.error("--background-samples must be >= 100")

    for pos in args.positions:
        pos = pos.upper()
        _run_shap_for_position(
            pos,
            target_filter=args.targets,
            background_samples=args.background_samples,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
