"""Reproducibility tests — same seed => bit-identical predictions.

Parametrized across all 6 positions. For each, runs ``run_pipeline`` twice
with ``seed=42`` and asserts:

  * Ridge predictions identical (``np.testing.assert_array_equal``).
  * NN predictions bit-identical (``atol=0, rtol=0``).
  * NN ``state_dict`` weight tensors identical across runs.

These tests guard a reviewer concern explicitly called out in the plan:
"training is reproducible" had previously gone unverified. A non-determinism
regression (e.g. a forgotten ``shuffle=True`` without a seeded generator, or
a new layer initialised with ``torch.randn`` without seeding) will fail here
the moment it lands.

Marked as ``@pytest.mark.e2e`` and ``@pytest.mark.integration`` so CI can
gate them behind the slower test lane.

All three test functions share a single module-scoped pair of pipeline
invocations per position (indirect parametrization), so each position
trains twice per module — not six times — saving ~3x wall clock.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._pipeline_e2e_utils import (
    ALL_POSITIONS,
    build_tiny_config,
    load_tiny_splits,
    run_pipeline_in_tmp,
)

# Markers (unit / integration / e2e / regression) are registered in
# ``tests/conftest.py`` so no local pytest_configure is needed here.


def _freeze_rngs() -> None:
    """Seed every RNG used by the pipeline.

    Covers ``random``, ``numpy``, and ``torch`` — the three sources of
    stochasticity in ``run_pipeline``. ``PYTHONHASHSEED`` is set defensively:
    it only affects new interpreter subprocesses, but is free to set.
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` asks PyTorch
    to prefer deterministic kernels where available; ``warn_only`` avoids
    hard-failing on ops that have no deterministic implementation.
    """
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError, TypeError):
        # Older torch builds or CUDA-only paths may not accept this; the
        # run_pipeline seeding is what actually drives the tests.
        pass


# ---------------------------------------------------------------------------
# Fixtures — one pair of module-scoped pipeline runs per position.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_runs(request, tmp_path_factory):
    """Run the pipeline twice with the same seed; cache the pair per position.

    Indirect parametrization supplies the position via ``request.param``.
    Returns a dict with:
      - ``r1`` / ``r2``: the two ``run_pipeline`` result dicts
      - ``run1_dir`` / ``run2_dir``: the tmp workdirs the two runs wrote
        artifacts into (state_dict comparisons read them back from disk)
      - ``cfg``: the tiny config used, so tests can iterate over its target list
    """
    position = request.param
    splits_root = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not (splits_root / "train.parquet").exists():
        pytest.skip(f"Real splits not present at {splits_root}; skipping E2E")

    splits = load_tiny_splits(position)
    cfg = build_tiny_config(position)

    _freeze_rngs()
    run1_dir = tmp_path_factory.mktemp(f"repro_{position}_run1")
    r1 = run_pipeline_in_tmp(position, cfg, splits, run1_dir, seed=42)

    _freeze_rngs()
    run2_dir = tmp_path_factory.mktemp(f"repro_{position}_run2")
    r2 = run_pipeline_in_tmp(position, cfg, splits, run2_dir, seed=42)

    return {
        "r1": r1,
        "r2": r2,
        "run1_dir": run1_dir,
        "run2_dir": run2_dir,
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# Helper: collect a fingerprint of model outputs for cross-run comparison
# ---------------------------------------------------------------------------


def _nn_state_dict_from_outputs(pos_outputs: Path, position: str) -> dict:
    """Load the NN state_dict that ``run_pipeline`` just wrote to disk.

    Accepts both the legacy raw-state-dict format and the wrapped format
    ``{"state_dict": ..., "feature_cols_hash": ..., ...}`` introduced alongside
    the scaler/weights integrity guardrail.
    """
    from src.shared.artifact_integrity import unwrap_state_dict

    pos_lower = position.lower()
    nn_path = pos_outputs / "models" / f"{pos_lower}_multihead_nn.pt"
    assert nn_path.exists(), f"NN weights not written at {nn_path}"
    # map_location=cpu so tests pass on CUDA-less machines
    checkpoint = torch.load(nn_path, map_location="cpu", weights_only=True)
    state_dict, _ = unwrap_state_dict(checkpoint)
    return state_dict


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.parametrize(
    "pipeline_runs,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_runs"],
)
def test_ridge_predictions_bit_identical(pipeline_runs, position):
    """Ridge: two seeded runs on identical data produce identical predictions."""
    cfg = pipeline_runs["cfg"]
    targets = list(cfg["targets"])

    p1 = pipeline_runs["r1"]["per_target_preds"]["ridge"]
    p2 = pipeline_runs["r2"]["per_target_preds"]["ridge"]
    for key in targets:
        np.testing.assert_array_equal(
            np.asarray(p1[key]),
            np.asarray(p2[key]),
            err_msg=(
                f"{position} Ridge.{key} differs across identical-seed runs — "
                "reproducibility regression"
            ),
        )


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.parametrize(
    "pipeline_runs,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_runs"],
)
def test_nn_predictions_bit_identical(pipeline_runs, position):
    """NN: atol=0, rtol=0 — any drift surfaces as a test failure."""
    cfg = pipeline_runs["cfg"]
    targets = list(cfg["targets"])

    p1 = pipeline_runs["r1"]["per_target_preds"]["nn"]
    p2 = pipeline_runs["r2"]["per_target_preds"]["nn"]
    for key in targets:
        t1 = torch.as_tensor(np.asarray(p1[key]))
        t2 = torch.as_tensor(np.asarray(p2[key]))
        torch.testing.assert_close(
            t1,
            t2,
            atol=0,
            rtol=0,
            msg=(
                f"{position} NN.{key} differs across identical-seed runs — "
                "reproducibility regression"
            ),
        )


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.parametrize(
    "pipeline_runs,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["pipeline_runs"],
)
def test_nn_state_dict_weights_identical(pipeline_runs, position):
    """The persisted NN state_dict tensors are bit-identical across runs."""
    sd1 = _nn_state_dict_from_outputs(pipeline_runs["run1_dir"] / position / "outputs", position)
    sd2 = _nn_state_dict_from_outputs(pipeline_runs["run2_dir"] / position / "outputs", position)

    assert set(sd1.keys()) == set(sd2.keys()), (
        f"{position}: state_dict key sets diverged: {set(sd1) ^ set(sd2)}"
    )
    for key, tensor1 in sd1.items():
        tensor2 = sd2[key]
        # Use atol=0, rtol=0 — bit-identical is the contract we care about.
        torch.testing.assert_close(
            tensor1,
            tensor2,
            atol=0,
            rtol=0,
            msg=f"{position} state_dict[{key!r}] differs across identical-seed runs",
        )
