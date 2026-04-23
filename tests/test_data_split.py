"""Coverage tests for ``src/data/split.py``.

Exercises ``temporal_split`` (with + without season_type filter, custom
bucket overrides, parquet write) and ``expanding_window_folds`` (default
val-season list + custom override).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import expanding_window_folds, temporal_split


@pytest.fixture()
def _sample_df():
    """Tiny multi-season frame used by both temporal_split and CV-fold tests.

    Does NOT carry a ``season_type`` column — that branch is exercised by a
    dedicated fixture below (the filter is `df[df["season_type"] == "REG"]`,
    so mixing REG + NaN kills every row).
    """
    rows = []
    for season in [2020, 2021, 2022, 2023, 2024, 2025]:
        for wk in range(1, 4):
            rows.append({"season": season, "week": wk, "player_id": f"P{wk}"})
    return pd.DataFrame(rows)


@pytest.fixture()
def _sample_df_with_season_type():
    """Same shape as ``_sample_df`` but every row has season_type = REG/POST."""
    rows = []
    for season in [2020, 2021, 2022, 2023, 2024, 2025]:
        for wk in range(1, 4):
            rows.append({"season": season, "week": wk, "player_id": f"P{wk}", "season_type": "REG"})
    for wk in (19, 20, 21):
        rows.append({"season": 2024, "week": wk, "player_id": "PPO", "season_type": "POST"})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_temporal_split_uses_defaults(tmp_path, monkeypatch, _sample_df):
    """No season overrides → uses TRAIN/VAL/TEST_SEASONS defaults."""
    import src.data.split as s

    monkeypatch.setattr(s, "SPLITS_DIR", str(tmp_path))
    # Defaults are TRAIN=range(2012, 2024), VAL=[2024], TEST=[2025].
    train, val, test = temporal_split(_sample_df)
    assert set(train["season"].unique()) <= set(range(2012, 2024))
    assert set(val["season"].unique()) == {2024}
    assert set(test["season"].unique()) == {2025}
    # Parquets land on disk.
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()


@pytest.mark.unit
def test_temporal_split_custom_overrides(tmp_path, monkeypatch, _sample_df):
    """Explicit season-list overrides propagate to the returned frames."""
    import src.data.split as s

    monkeypatch.setattr(s, "SPLITS_DIR", str(tmp_path))
    train, val, test = temporal_split(
        _sample_df,
        train_seasons=[2020, 2021],
        val_seasons=[2022],
        test_seasons=[2023],
    )
    assert set(train["season"].unique()) == {2020, 2021}
    assert set(val["season"].unique()) == {2022}
    assert set(test["season"].unique()) == {2023}


@pytest.mark.unit
def test_temporal_split_drops_postseason_rows(tmp_path, monkeypatch, _sample_df_with_season_type):
    """Rows with ``season_type != 'REG'`` get dropped (playoff exclusion)."""
    import src.data.split as s

    monkeypatch.setattr(s, "SPLITS_DIR", str(tmp_path))
    train, val, test = temporal_split(
        _sample_df_with_season_type,
        train_seasons=[2020, 2021, 2022, 2023],
        val_seasons=[2024],
        test_seasons=[2025],
    )
    # POST rows were in 2024 — val shouldn't include them.
    assert (val["season_type"] != "POST").all()
    assert (val["week"] < 19).all()  # weeks 19-21 were POST


@pytest.mark.unit
def test_temporal_split_raises_on_season_overlap(tmp_path, monkeypatch, _sample_df):
    """Overlapping splits trip the assert."""
    import src.data.split as s

    monkeypatch.setattr(s, "SPLITS_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="Season overlap"):
        temporal_split(
            _sample_df, train_seasons=[2020, 2021], val_seasons=[2021], test_seasons=[2022]
        )


@pytest.mark.unit
def test_expanding_window_folds_default_val_seasons(_sample_df):
    """Default CV_VAL_SEASONS = [2021, 2022, 2023, 2024]; each fold trains on
    all prior seasons from min_train_season (default 2012)."""
    folds = expanding_window_folds(_sample_df)
    assert len(folds) == 4
    # First fold: train covers 2020-2020 (only season >=2012 in our sample), val=2021
    _, train0, val0 = folds[0]
    assert set(val0["season"].unique()) == {2021}
    # Last fold: val=2024
    _, _, val_last = folds[-1]
    assert set(val_last["season"].unique()) == {2024}


@pytest.mark.unit
def test_expanding_window_folds_custom_seasons(_sample_df):
    """Custom val-season list works, and min_train_season is honored."""
    folds = expanding_window_folds(_sample_df, val_seasons=[2022, 2023], min_train_season=2021)
    assert len(folds) == 2
    # Fold 1: train seasons 2021, val = 2022
    _, train0, val0 = folds[0]
    assert set(train0["season"].unique()) == {2021}
    assert set(val0["season"].unique()) == {2022}
    # Fold 2: train seasons 2021-2022, val = 2023
    _, train1, val1 = folds[1]
    assert set(train1["season"].unique()) == {2021, 2022}
    assert set(val1["season"].unique()) == {2023}
