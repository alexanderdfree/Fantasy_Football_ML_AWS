"""Version-controlled history for hyperparameter-tuning runs.

Each ``tune_nn`` / ``tune_lgbm`` run appends one git-tracked JSON file under
``benchmark_history/tuning/``, mirroring the per-run history convention used by
``src/benchmarking/benchmark.py`` and ``src/tuning/ablate_rb_gate.py``.

This module lives under ``src/tuning/`` (not ``src/shared/``) on purpose:
editing ``src/shared/`` trips the global retrain trigger in
``src/scripts/scope_positions.py``, whereas ``src/tuning/`` changes do not. It
only *imports* the shared history helpers; it never modifies them.
"""

import os

from src.shared.benchmark_utils import append_to_history, get_git_hash, utc_now_iso

HISTORY_DIR = os.path.join("benchmark_history", "tuning")


def append_tuning_run(
    kind,
    results,
    *,
    n_trials=None,
    positions=None,
    note=None,
    pr_number=None,
    history_dir=HISTORY_DIR,
):
    """Append one tuning run to the version-controlled history.

    Args:
        kind: Run type, e.g. ``"tune_nn"`` or ``"tune_lgbm"``. Becomes the
            ``run_id`` suffix so the filename self-documents the tuner.
        results: The per-position results dict the tuner already builds
            (``{POS: {best_trial, best_*_mae/loss, best_params, ...}}``),
            stored verbatim under ``"results"``.
        n_trials: Trials per position for this run (recorded for context).
        positions: Positions covered; defaults to ``sorted(results)``.
        note: Optional free-text note (e.g. a backfill marker).
        pr_number: Forwarded to ``append_to_history`` for the PR badge.
        history_dir: Target directory; overridable for tests.

    Returns:
        The path of the written JSON file.
    """
    now = utc_now_iso()
    git_hash = get_git_hash()
    entry = {
        "run_id": f"{now}_{git_hash}_{kind}",
        "timestamp": now,
        "git_hash": git_hash,
        "kind": kind,
        "n_trials": n_trials,
        "positions": list(positions) if positions is not None else sorted(results),
        "results": results,
    }
    if note:
        entry["note"] = note
    return append_to_history(history_dir, entry, pr_number=pr_number)
