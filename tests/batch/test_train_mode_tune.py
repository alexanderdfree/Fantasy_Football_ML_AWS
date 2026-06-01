"""Unit tests for src/batch/train.py's --mode=tune dispatch.

The actual tune execution is mocked — we only verify that train.py builds
the right sys.argv before delegating to tune_nn.main().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.batch import train

pytestmark = pytest.mark.unit


def test_mode_tune_delegates_to_tune_nn_with_forwarded_args(monkeypatch):
    """--mode=tune must call src.tuning.tune_nn.main() after rewriting sys.argv
    to the tune CLI shape with --checkpoint-s3 always enabled."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--position",
            "RB",
            "--mode",
            "tune",
            "--seed",
            "7",
            "--n-trials",
            "5",
            "--timeout",
            "300",
        ],
    )
    # Skip GPU assertion (no CUDA in the test env) and the actual tune call.
    with (
        patch("src.batch.train._assert_gpu") as mock_assert_gpu,
        patch("src.tuning.tune_nn.main") as mock_tune_main,
    ):
        train.main()

    mock_assert_gpu.assert_called_once_with("RB", force=True)
    mock_tune_main.assert_called_once_with()
    # Verify sys.argv was rewritten to the tune CLI shape (tune_nn.main reads
    # it via argparse). --checkpoint-s3 must always be set in Batch dispatch.
    import sys as _sys

    assert _sys.argv == [
        "tune_nn",
        "RB",
        "--checkpoint-s3",
        "--seed",
        "7",
        "--n-trials",
        "5",
        "--timeout",
        "300",
    ]


def test_mode_tune_without_n_trials_or_timeout(monkeypatch):
    """Optional --n-trials and --timeout should be omitted from forwarded
    argv so tune_nn uses its own defaults."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--position", "QB", "--mode", "tune"],
    )
    with patch("src.batch.train._assert_gpu"), patch("src.tuning.tune_nn.main"):
        train.main()
    import sys as _sys

    assert _sys.argv == ["tune_nn", "QB", "--checkpoint-s3", "--seed", "42"]


def test_mode_tune_forwards_parallel_backend_and_n_jobs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--position",
            "K",
            "--mode",
            "tune",
            "--parallel-backend",
            "auto",
            "--n-jobs",
            "3",
        ],
    )
    with patch("src.batch.train._assert_gpu") as mock_assert_gpu, patch("src.tuning.tune_nn.main"):
        train.main()
    import sys as _sys

    mock_assert_gpu.assert_called_once_with("K", force=True)
    assert _sys.argv == [
        "tune_nn",
        "K",
        "--checkpoint-s3",
        "--seed",
        "42",
        "--parallel-backend",
        "auto",
        "--n-jobs",
        "3",
    ]


@pytest.mark.parametrize(
    "conflicting",
    [
        ["--ablation", "rb-gate"],
        ["--sweep"],
        ["--dry-run"],
    ],
)
def test_mode_tune_rejects_conflicting_flags(monkeypatch, conflicting):
    """--mode=tune is mutually exclusive with --ablation / --sweep / --dry-run.
    The dispatcher should error out instead of running a half-pipeline."""
    pos = "RB" if conflicting[0] == "--ablation" else "WR" if conflicting[0] == "--sweep" else "QB"
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--position", pos, "--mode", "tune"] + conflicting,
    )
    with pytest.raises(SystemExit):
        train.main()


@pytest.mark.parametrize("pos", ["K", "DST"])
def test_mode_tune_accepts_k_and_dst(monkeypatch, pos):
    """K and DST gained ``run(config=...)`` in PR 3, so the --mode=tune
    dispatch must accept them and forward through to tune_nn.main()."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--position", pos, "--mode", "tune", "--n-trials", "5"],
    )
    with patch("src.batch.train._assert_gpu"), patch("src.tuning.tune_nn.main") as mock_tune_main:
        train.main()
    mock_tune_main.assert_called_once_with()
    import sys as _sys

    assert _sys.argv == [
        "tune_nn",
        pos,
        "--checkpoint-s3",
        "--seed",
        "42",
        "--n-trials",
        "5",
    ]


def test_mode_train_dry_run_skips_tune_dispatch(monkeypatch):
    """--dry-run on the default --mode=train path must NOT invoke the tune
    dispatch — it should run through _dry_run_artifacts and exit. Asserted
    by ensuring tune_nn.main is never called."""
    monkeypatch.setattr("sys.argv", ["train", "--position", "RB", "--dry-run"])
    with (
        patch("src.batch.train._dry_run_artifacts") as mock_dry,
        patch("src.batch.train.os.makedirs"),
        patch("src.tuning.tune_nn.main") as mock_tune_main,
    ):
        train.main()
    mock_dry.assert_called_once()
    mock_tune_main.assert_not_called()


def test_mode_defaults_to_train(monkeypatch):
    """argparse default for --mode must be 'train' so existing launch.py
    callers (which don't pass --mode) continue to hit the training path."""
    monkeypatch.setattr("sys.argv", ["train", "--position", "QB", "--dry-run"])
    with (
        patch("src.batch.train._dry_run_artifacts"),
        patch("src.batch.train.os.makedirs"),
        patch("src.tuning.tune_nn.main") as mock_tune_main,
    ):
        train.main()
    mock_tune_main.assert_not_called()
