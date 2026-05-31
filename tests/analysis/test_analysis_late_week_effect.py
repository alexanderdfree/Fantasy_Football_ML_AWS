"""Smoke + unit tests for src/analysis/analysis_late_week_effect.py.

The stage2 / ablation paths train production models (slow, data-dependent) and are
out of scope for unit tests. These cover the pure helpers — the season-aware
final-week bucketing, the eval-week bucketer, the train-cut filter, and the weekly
ranking aggregator — so a regression in the decision-relevant logic (especially the
era-aware "final week" definition) fails CI rather than the PR-review pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_late_week_effect as alwe

pytestmark = pytest.mark.unit


def test_module_imports_cleanly():
    for sym in ("stage1_label_anomaly", "stage2_prediction_degradation", "run_ablation", "main"):
        assert hasattr(alwe, sym)
    # Stage 1 is skill-only by design: the raw split's fantasy_points is skill-scoring
    # (~0 for K) and DST is absent from that split. A change here would be a silent
    # scope regression.
    assert alwe.SKILL_POSITIONS == ["QB", "RB", "WR", "TE"]


def test_eval_week_bucket():
    assert alwe._week_bucket_eval(1) == "wk1-16"
    assert alwe._week_bucket_eval(16) == "wk1-16"
    assert alwe._week_bucket_eval(17) == "wk17"
    assert alwe._week_bucket_eval(18) == "wk18"


def test_final_week_is_season_aware():
    """17-game seasons end at wk17; 18-game (2021+) at wk18 — not a fixed integer."""
    df = pd.DataFrame(
        {
            "season": [2019, 2019, 2019, 2023, 2023, 2023],
            "week": [16, 17, 1, 17, 18, 1],
        }
    )
    out = alwe.assign_final_week_buckets(df)
    assert out.loc[out["season"] == 2019, "final_week"].eq(17).all()
    assert out.loc[out["season"] == 2023, "final_week"].eq(18).all()

    keys = zip(out["season"], out["week"], strict=True)
    bucket = dict(zip(keys, out["week_bucket"], strict=True))
    assert bucket[(2019, 17)] == alwe.FINAL
    assert bucket[(2019, 16)] == alwe.PENULT
    assert bucket[(2023, 18)] == alwe.FINAL
    assert bucket[(2023, 17)] == alwe.PENULT
    assert bucket[(2019, 1)] == alwe.EARLY

    assert out.loc[out["season"] == 2019, "era"].iloc[0].startswith("2012-2020")
    assert out.loc[out["season"] == 2023, "era"].iloc[0].startswith("2021")


def test_drop_final_week_is_season_aware():
    """CUT removes each season's max week (wk17 in 2019, wk18 in 2023), keeps the rest."""
    df = pd.DataFrame({"season": [2019, 2019, 2023, 2023], "week": [17, 5, 18, 5]})
    cut = alwe._drop_final_week(df)
    assert sorted(cut["week"]) == [5, 5]
    assert len(cut) == 2


def test_avg_weekly_ranking_perfect_prediction():
    """Predicted order == true order → top-k hit = 1.0 and Spearman = 1.0."""
    df = pd.DataFrame(
        {
            "week": [1, 1, 1, 1],
            "player_id": ["a", "b", "c", "d"],
            "fantasy_points": [10.0, 8.0, 6.0, 4.0],
            "pred": [9.0, 7.0, 5.0, 3.0],  # same ranking, different scale
        }
    )
    hit, spear = alwe._avg_weekly_ranking(df, "pred", top_k=2)
    assert hit == 1.0
    assert spear == pytest.approx(1.0)


def test_avg_weekly_ranking_skips_thin_weeks():
    """A week with fewer than top_k rows has no defined ranking → NaN, not a crash."""
    df = pd.DataFrame({"week": [1], "player_id": ["a"], "fantasy_points": [5.0], "pred": [5.0]})
    hit, spear = alwe._avg_weekly_ranking(df, "pred", top_k=2)
    assert np.isnan(hit) and np.isnan(spear)
