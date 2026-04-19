"""Shared pytest fixture factories for position test suites.

Six position packages (QB/RB/WR/TE/K/DST) each need near-identical test
helpers to build simulation DataFrames, ranking DataFrames, tensor dicts,
and (train, val, test) splits.  This module consolidates the shared bits
so each position conftest collapses to ~15-25 lines of position-specific
bindings.

The exported factories (``make_sim_df``, ``make_test_df``,
``make_tensors``, ``make_splits``, ``make_position_df``) are plain
module-level *functions* — importable and callable from any pytest
fixture.  Position conftests wrap them with thin ``@pytest.fixture``
bindings that inject the position's scoring scale, player-id prefix,
and target list.

``register_position_markers(config, extra=None)`` centralises marker
registration (unit / integration / e2e / regression) that every
position was repeating.

RNG choice
----------

The original position conftests used a mix of ``np.random.seed`` /
``np.random.RandomState`` / ``np.random.default_rng``.  Those RNGs emit
different values for the same seed, so a refactor that silently
switched RNG kinds would change test behavior.  The helpers here
accept ``rng_kind="legacy"`` (``np.random.seed`` globals — what QB,
RB, and DST originally used) or ``rng_kind="default"`` (WR/TE/K).
Tests that assert same-seed determinism keep passing either way.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

_STANDARD_MARKERS = (
    ("unit", "fast isolated unit test (<1s, no I/O or training)"),
    ("integration", "multi-component test that exercises shared modules"),
    ("e2e", "full-pipeline end-to-end smoke test"),
    ("regression", "model-quality threshold assertions (MAE/R2)"),
)


def register_position_markers(config, extra: Iterable[tuple[str, str]] | None = None) -> None:
    """Register the standard position test markers on a pytest ``config``.

    Idempotent — pytest tolerates duplicate marker registration, so this
    helper is safe to call multiple times (e.g. from nested conftests).
    Pass ``extra`` to append position-specific markers.
    """
    for name, desc in _STANDARD_MARKERS:
        config.addinivalue_line("markers", f"{name}: {desc}")
    if extra:
        for name, desc in extra:
            config.addinivalue_line("markers", f"{name}: {desc}")


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def _seed_legacy(seed: int):
    """Seed numpy's legacy global RNG and return a shim exposing ``rand``/``randn``."""
    np.random.seed(seed)

    class _Legacy:
        @staticmethod
        def rand(*args, **kwargs):
            return np.random.rand(*args, **kwargs)

        @staticmethod
        def randn(*args, **kwargs):
            return np.random.randn(*args, **kwargs)

    return _Legacy()


def _make_rng(seed: int, rng_kind: str):
    """Return a uniform ``rng`` with ``.rand`` / ``.randn`` methods.

    ``rng_kind`` is ``"legacy"`` (``np.random.seed`` + globals) or
    ``"default"`` (``np.random.default_rng`` with adapter methods).
    """
    if rng_kind == "legacy":
        return _seed_legacy(seed)
    if rng_kind == "default":
        rng = np.random.default_rng(seed)

        class _Default:
            @staticmethod
            def rand():
                return rng.random()

            @staticmethod
            def randn():
                return rng.standard_normal()

        return _Default()
    raise ValueError(f"Unknown rng_kind: {rng_kind!r}")


# ---------------------------------------------------------------------------
# make_sim_df — weekly-simulation DataFrame for backtest tests
# ---------------------------------------------------------------------------


def make_sim_df(
    scoring_scale: float,
    n_weeks: int = 4,
    n_players: int = 15,
    seed: int = 42,
    id_prefix: str = "P",
    rng_kind: str = "legacy",
) -> pd.DataFrame:
    """Build a synthetic weekly-simulation DataFrame.

    Each row is one player-week with a ground-truth ``fantasy_points``
    drawn from ``U(0, scoring_scale)`` plus two noisy predictions:

    - ``pred_ridge`` — Gaussian noise (std=2) added to the truth.
    - ``pred_nn`` — Gaussian noise (std=3) added to the truth.

    Used by ``run_weekly_simulation`` tests to verify structure and
    metrics.  The ``scoring_scale`` determines the typical magnitude
    (QB~25, RB/WR~20, DST~15, TE~15, K~12 in this project).
    """
    rng = _make_rng(seed, rng_kind)
    rows = []
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            fp = float(rng.rand() * scoring_scale)
            rows.append(
                {
                    "week": week,
                    "player_id": f"{id_prefix}{pid}",
                    "fantasy_points": fp,
                    "pred_ridge": fp + float(rng.randn()) * 2,
                    "pred_nn": fp + float(rng.randn()) * 3,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# make_test_df — ranking DataFrame for evaluation tests
# ---------------------------------------------------------------------------


def make_test_df(
    scoring_scale: float,
    n_weeks: int = 3,
    n_players: int = 15,
    seed: int = 42,
    id_prefix: str = "P",
    rng_kind: str = "legacy",
) -> pd.DataFrame:
    """Build a synthetic DataFrame for ``compute_ranking_metrics`` tests.

    Columns: ``week``, ``player_id``, ``pred_total``, ``fantasy_points``.
    Both ``pred_total`` and ``fantasy_points`` are drawn independently
    from ``U(0, scoring_scale)`` — so predictions are uncorrelated with
    truth by construction; tests that assert positive correlation must
    override ``pred_total``.
    """
    rng = _make_rng(seed, rng_kind)
    rows = []
    for week in range(1, n_weeks + 1):
        for pid in range(1, n_players + 1):
            rows.append(
                {
                    "week": week,
                    "player_id": f"{id_prefix}{pid}",
                    "pred_total": float(rng.rand() * scoring_scale),
                    "fantasy_points": float(rng.rand() * scoring_scale),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# make_tensors — (preds, targets) dict pair for MultiTargetLoss tests
# ---------------------------------------------------------------------------


def make_tensors(
    targets: Iterable[str],
    n: int = 10,
    seed: int | None = 42,
    include_total: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build ``(preds, targets)`` dicts of ``torch.randn`` tensors.

    Each dict maps each target name in ``targets`` (plus ``"total"`` if
    ``include_total``) to an independent ``torch.randn(n)`` draw.  Used
    by the position MultiTargetLoss tests.  Pass ``seed=None`` to
    avoid touching torch's global RNG state (DST's original behavior).
    """
    if seed is not None:
        torch.manual_seed(seed)
    targets = list(targets)
    preds = {t: torch.randn(n) for t in targets}
    if include_total:
        preds["total"] = torch.randn(n)
    truth = {t: torch.randn(n) for t in targets}
    if include_total:
        truth["total"] = torch.randn(n)
    return preds, truth


# ---------------------------------------------------------------------------
# make_splits — (train, val, test) single-column DataFrames for NaN-fill
# ---------------------------------------------------------------------------


def make_splits(train_vals, val_vals, test_vals, col: str = "feat1"):
    """Build three single-column DataFrames for NaN-fill tests.

    ``train_vals``, ``val_vals``, ``test_vals`` are iterables of scalar
    values (possibly including ``NaN`` / ``inf``).  Returns a
    ``(train, val, test)`` tuple of single-column DataFrames.
    """
    train = pd.DataFrame({col: train_vals})
    val = pd.DataFrame({col: val_vals})
    test = pd.DataFrame({col: test_vals})
    return train, val, test


# ---------------------------------------------------------------------------
# make_position_df — position-encoded DataFrame for filter_to_{pos} tests
# ---------------------------------------------------------------------------


def make_position_df(
    positions,
    stat_col: str = "passing_yards",
    has_pos_cols: bool = True,
):
    """Build a DataFrame used by ``filter_to_{pos}`` tests.

    ``positions`` is a list of position strings (``"QB"``, ``"RB"``,
    ``"WR"``, ``"TE"``).  The DataFrame carries the raw ``position``
    column plus the ``stat_col`` (filled with ``range(len(positions))``),
    and optional ``pos_QB`` / ``pos_RB`` / ``pos_WR`` / ``pos_TE``
    one-hot encoded columns.  ``stat_col`` lets each position keep its
    original filler column name (``passing_yards`` for QB,
    ``rushing_yards`` for RB, ``receiving_yards`` for WR/TE).
    """
    data = {"position": positions, stat_col: range(len(positions))}
    if has_pos_cols:
        data.update(
            {
                "pos_QB": [1 if p == "QB" else 0 for p in positions],
                "pos_RB": [1 if p == "RB" else 0 for p in positions],
                "pos_WR": [1 if p == "WR" else 0 for p in positions],
                "pos_TE": [1 if p == "TE" else 0 for p in positions],
            }
        )
    return pd.DataFrame(data)
