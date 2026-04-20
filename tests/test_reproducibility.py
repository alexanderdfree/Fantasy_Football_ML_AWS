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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_rng():
    """Seed every RNG used by the pipeline before a test runs.

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
    yield


@pytest.fixture(scope="module")
def tiny_splits(request):
    """(train, val, test) tuple sized for the given position — module-scoped."""
    position = request.param
    splits_root = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not (splits_root / "train.parquet").exists():
        pytest.skip(f"Real splits not present at {splits_root}; skipping E2E")
    return load_tiny_splits(position)


# ---------------------------------------------------------------------------
# Helper: collect a fingerprint of model outputs for cross-run comparison
# ---------------------------------------------------------------------------


def _nn_state_dict_from_outputs(pos_outputs: Path, position: str) -> dict:
    """Load the NN state_dict that ``run_pipeline`` just wrote to disk.

    Accepts both the legacy raw-state-dict format and the wrapped format
    ``{"state_dict": ..., "feature_cols_hash": ..., ...}`` introduced alongside
    the scaler/weights integrity guardrail.
    """
    from shared.artifact_integrity import unwrap_state_dict

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
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_ridge_predictions_bit_identical(tiny_splits, position, tmp_path, frozen_rng):
    """Ridge: two seeded runs on identical data produce identical predictions."""
    cfg = build_tiny_config(position)
    targets = list(cfg["targets"])

    r1 = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path / "run1", seed=42)
    r2 = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path / "run2", seed=42)

    p1 = r1["per_target_preds"]["ridge"]
    p2 = r2["per_target_preds"]["ridge"]
    for key in targets + ["total"]:
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
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_nn_predictions_bit_identical(tiny_splits, position, tmp_path, frozen_rng):
    """NN: atol=0, rtol=0 — any drift surfaces as a test failure."""
    cfg = build_tiny_config(position)
    targets = list(cfg["targets"])

    r1 = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path / "run1", seed=42)
    r2 = run_pipeline_in_tmp(position, cfg, tiny_splits, tmp_path / "run2", seed=42)

    p1 = r1["per_target_preds"]["nn"]
    p2 = r2["per_target_preds"]["nn"]
    for key in targets + ["total"]:
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
    "tiny_splits,position",
    [(pos, pos) for pos in ALL_POSITIONS],
    indirect=["tiny_splits"],
)
def test_nn_state_dict_weights_identical(tiny_splits, position, tmp_path, frozen_rng):
    """The persisted NN state_dict tensors are bit-identical across runs."""
    cfg = build_tiny_config(position)
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    run_pipeline_in_tmp(position, cfg, tiny_splits, run1_dir, seed=42)
    run_pipeline_in_tmp(position, cfg, tiny_splits, run2_dir, seed=42)

    sd1 = _nn_state_dict_from_outputs(run1_dir / position / "outputs", position)
    sd2 = _nn_state_dict_from_outputs(run2_dir / position / "outputs", position)

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
