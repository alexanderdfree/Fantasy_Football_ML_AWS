"""Regenerate a position's SERVED model artifacts from the current local ``data/splits``.

Use when a position's local artifacts have drifted from the current splits — i.e.
``python -m src.analysis.artifact_eval --positions <POS> --validate`` warns STALE because the
saved model was trained on a different data vintage than the on-disk splits (the QB incident:
deterministic Ridge reconstructed MAE 6.74 vs its recorded 5.88). Retraining the position on the
*current* splits realigns the artifact so ``artifact_eval`` reconstructs faithfully again.

What a bare ``run_pipeline`` does NOT do, and this helper adds:
  1. it saves to the PRODUCER path ``{pos}/outputs/models`` (relative to CWD) — this promotes that
     to the SERVED path ``src/{pos}/outputs/models`` that serving + ``artifact_eval`` read;
  2. it does NOT write ``benchmark_metrics.json`` (that is Batch-only, ``src/batch/train.py``) — this
     writes a fresh one from the run's metrics so the drift check has a CURRENT reference.

Artifacts are gitignored (``**/outputs/models/``) → no commit, no retrain trigger. Local 5080,
~1-3 min/position. Scope is whatever ``--positions`` you pass (default QB — the only drifted one).

    python -m src.scripts.regen_served_artifacts                 # QB
    python -m src.scripts.regen_served_artifacts --positions QB RB
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

from src.analysis.artifact_eval import _producer_model_dir
from src.analysis.cohort_analysis import _load_splits
from src.config import SPLITS_DIR
from src.shared.registry import INFERENCE_REGISTRY, get_runner
from src.shared.utils import seed_everything

_METRIC_KEYS = (
    "ridge_metrics",
    "nn_metrics",
    "attn_nn_metrics",
    "lgbm_metrics",
    "elasticnet_metrics",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _splits_fingerprint() -> str:
    """mtime fingerprint of the splits the artifact is being trained on (vintage tell)."""
    parts = []
    for name in ("train", "val", "test"):
        p = os.path.join(SPLITS_DIR, f"{name}.parquet")
        parts.append(
            f"{name}:{int(os.path.getmtime(p))}" if os.path.exists(p) else f"{name}:missing"
        )
    return ";".join(parts)


def _write_benchmark_metrics(pos: str, result: dict, seed: int, served_dir: str) -> float | None:
    blob: dict = {
        "position": pos,
        "seed": seed,
        "git_sha": _git_sha(),
        "split_run_id": _splits_fingerprint(),
        "split_merged": False,
        "regenerated_by": "src.scripts.regen_served_artifacts",
    }
    for key in _METRIC_KEYS:
        if result.get(key):
            blob[key] = result[key]
    if "ridge_metrics" not in blob:
        raise RuntimeError(
            f"{pos}: run result has no ridge_metrics — cannot write benchmark_metrics.json"
        )
    with open(os.path.join(served_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(blob, f, indent=2)
    return blob["ridge_metrics"].get("total", {}).get("mae")


def regen(pos: str, seed: int = 42) -> None:
    pos = pos.upper()
    if pos not in INFERENCE_REGISTRY:
        raise ValueError(f"unknown position {pos!r}")
    reg = INFERENCE_REGISTRY[pos]
    served = reg["model_dir"]
    producer = _producer_model_dir(pos)

    train_df, val_df, test_df = _load_splits()
    seed_everything(seed)
    print(f"[regen] {pos}: training on current data/splits (producer path {producer}) ...")
    result = get_runner(pos)(train_df=train_df, val_df=val_df, test_df=test_df, seed=seed)

    if not os.path.isdir(producer) or not os.listdir(producer):
        raise RuntimeError(f"{pos}: pipeline did not populate producer dir {producer!r}")

    # Promote producer -> served (mirror batch/train.py::_replace_model_dir_contents).
    os.makedirs(os.path.dirname(served) or ".", exist_ok=True)
    if os.path.isdir(served):
        shutil.rmtree(served)
    shutil.copytree(producer, served)
    ridge_mae = _write_benchmark_metrics(pos, result, seed, served)
    print(
        f"[regen] {pos}: promoted {producer} -> {served}; wrote benchmark_metrics.json "
        f"(ridge total MAE={ridge_mae}). Verify: "
        f"`python -m src.analysis.artifact_eval --positions {pos} --validate`."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", nargs="+", default=["QB"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    for pos in a.positions:
        regen(pos, a.seed)


if __name__ == "__main__":
    main()
