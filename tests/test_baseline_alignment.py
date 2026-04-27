"""Regression tests for baseline row-order alignment.

Previously both baselines sorted internally but returned `.values` in sorted
order, so predictions misaligned with the caller's row order whenever the
caller did not pre-sort. Callers in src/shared/pipeline.py and
tests/dst/test_regression.py do not pre-sort, so every baseline MAE the
project reports was wrong. The fix makes the baselines self-aligning; these
tests pin that behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import LastWeekBaseline, SeasonAverageBaseline


def _build_shuffled_df() -> pd.DataFrame:
    """Two players, one season, 5 weeks each, passed to predict() shuffled."""
    rows = []
    for pid in ("A", "B"):
        base = 10.0 if pid == "A" else 20.0
        for wk in range(1, 6):
            rows.append({"player_id": pid, "season": 2024, "week": wk, "fantasy_points": base + wk})
    df_sorted = pd.DataFrame(rows)
    rng = np.random.default_rng(seed=17)
    shuffle = rng.permutation(len(df_sorted))
    return df_sorted.iloc[shuffle].reset_index(drop=True)


@pytest.mark.unit
def test_season_average_baseline_preserves_caller_row_order():
    df = _build_shuffled_df()
    preds = SeasonAverageBaseline().predict(df)

    assert preds.shape == (len(df),)
    for i, row in df.reset_index(drop=True).iterrows():
        prior = df[
            (df["player_id"] == row["player_id"])
            & (df["season"] == row["season"])
            & (df["week"] < row["week"])
        ]["fantasy_points"]
        expected = prior.mean() if len(prior) else 0.0
        assert preds[i] == pytest.approx(expected), (
            f"row {i} ({row['player_id']} w{row['week']}): got {preds[i]}, expected {expected}"
        )


@pytest.mark.unit
def test_last_week_baseline_preserves_caller_row_order():
    df = _build_shuffled_df()
    preds = LastWeekBaseline().predict(df)

    assert preds.shape == (len(df),)
    for i, row in df.reset_index(drop=True).iterrows():
        prior = df[
            (df["player_id"] == row["player_id"])
            & (df["season"] == row["season"])
            & (df["week"] < row["week"])
        ].sort_values("week")["fantasy_points"]
        if len(prior) == 0:
            expected = 0.0
        elif len(prior) == 1:
            # shift(1) gives NaN on the second week; fillna(season_avg) fills
            # with the expanding mean of prior weeks (one prior → that week's value).
            expected = prior.iloc[-1]
        else:
            expected = prior.iloc[-1]
        assert preds[i] == pytest.approx(expected), (
            f"row {i} ({row['player_id']} w{row['week']}): got {preds[i]}, expected {expected}"
        )


@pytest.mark.unit
def test_baseline_handles_non_unique_index():
    """`pos_test` in src/shared/pipeline.py retains its source frame's non-unique
    index after filtering. Baselines must not rely on df.index being unique."""
    df = _build_shuffled_df()
    df.index = [0] * len(df)
    preds = SeasonAverageBaseline().predict(df)
    assert preds.shape == (len(df),)
    assert not np.any(np.isnan(preds))

    preds_last = LastWeekBaseline().predict(df)
    assert preds_last.shape == (len(df),)
    assert not np.any(np.isnan(preds_last))
