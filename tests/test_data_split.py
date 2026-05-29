"""Coverage tests for ``src/data/split.py``.

Exercises ``temporal_split`` (REG-only filter + assertion on absent
``season_type`` column, custom bucket overrides, parquet write) and
``expanding_window_folds`` (default val-season list + custom override).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.data.split import expanding_window_folds, rolling_origin_folds, temporal_split


@pytest.fixture()
def _sample_df():
    """Tiny multi-season frame used by both temporal_split and CV-fold tests.

    Carries ``season_type='REG'`` on every row — temporal_split now asserts
    the column's presence (nflverse always emits it; absence indicates a
    malformed upstream frame). Tests that need to exercise the missing-column
    branch use ``_sample_df_no_season_type`` below.
    """
    rows = []
    for season in [2020, 2021, 2022, 2023, 2024, 2025]:
        for wk in range(1, 4):
            rows.append({"season": season, "week": wk, "player_id": f"P{wk}", "season_type": "REG"})
    return pd.DataFrame(rows)


@pytest.fixture()
def _sample_df_no_season_type():
    """Frame without ``season_type`` — exercises the AssertionError branch."""
    rows = []
    for season in [2020, 2021]:
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
def test_temporal_split_asserts_when_season_type_column_absent(
    tmp_path, monkeypatch, _sample_df_no_season_type
):
    """Missing ``season_type`` is a malformed-frame signal — temporal_split
    must fail loudly so the upstream loader bug surfaces immediately rather
    than silently including playoff rows in the splits."""
    import src.data.split as s

    monkeypatch.setattr(s, "SPLITS_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="season_type"):
        temporal_split(_sample_df_no_season_type)


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


@pytest.mark.unit
def test_expanding_window_folds_imputes_snap_pct_train_only():
    """audit-320 F106: each fold's val ``snap_pct`` NaN is filled with that
    fold's TRAIN-only (position, week) median, mirroring ``temporal_split``.

    The val season carries a distinct snap_pct distribution; a leaky
    full-frame median would mix it in. Asserting the imputed val value
    equals the train-only median (and is NOT the val-inclusive median)
    pins the train-only contract.
    """
    import numpy as np

    rows = []
    # Train season 2021, (QB, week 1): 0.20, 0.30 -> train-only median 0.25
    rows.append({"season": 2021, "week": 1, "player_id": "T1", "position": "QB", "snap_pct": 0.20})
    rows.append({"season": 2021, "week": 1, "player_id": "T2", "position": "QB", "snap_pct": 0.30})
    # Val season 2022, (QB, week 1): one observed 0.90 + one NaN to impute.
    # Full-frame median over {0.20,0.30,0.90} = 0.30; train-only median = 0.25.
    rows.append({"season": 2022, "week": 1, "player_id": "V1", "position": "QB", "snap_pct": 0.90})
    rows.append(
        {"season": 2022, "week": 1, "player_id": "V2", "position": "QB", "snap_pct": np.nan}
    )
    df = pd.DataFrame(rows)

    folds = expanding_window_folds(df, val_seasons=[2022], min_train_season=2021)
    _, train0, val0 = folds[0]
    # Train rows untouched.
    assert set(train0[train0["position"] == "QB"]["snap_pct"].round(2)) == {0.20, 0.30}
    # Val NaN filled with TRAIN-only median (0.25), not the val-inclusive 0.30.
    v2 = val0[val0["player_id"] == "V2"].iloc[0]
    assert v2["snap_pct"] == pytest.approx(0.25)
    # Observed val value passes through.
    assert val0[val0["player_id"] == "V1"].iloc[0]["snap_pct"] == pytest.approx(0.90)


# --------------------------------------------------------------------------
# rolling_origin_folds
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_rolling_origin_folds_default_boundaries(_sample_df):
    """Default ROLLING_ORIGIN_TEST_SEASONS = [2023, 2024, 2025]; each origin has
    train [..T-2], val T-1, test T, with disjoint season sets."""
    folds = rolling_origin_folds(_sample_df)
    assert len(folds) == 3
    expected_test = [2023, 2024, 2025]
    for (_i, train, val, test), t in zip(folds, expected_test, strict=True):
        assert set(test["season"].unique()) == {t}
        assert set(val["season"].unique()) == {t - 1}
        assert train["season"].max() == t - 2
        # No season appears in more than one split.
        tr, va, te = set(train["season"]), set(val["season"]), set(test["season"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)


@pytest.mark.unit
def test_rolling_origin_final_origin_matches_production_split(_sample_df):
    """The last origin (test=2025) reproduces temporal_split's season slices, so a
    rolling-origin run is directly comparable to the production single split."""
    folds = rolling_origin_folds(_sample_df)
    _, train, val, test = folds[-1]
    present = set(_sample_df["season"].unique())
    assert set(train["season"].unique()) == (set(TRAIN_SEASONS) & present)
    assert set(val["season"].unique()) == (set(VAL_SEASONS) & present)
    assert set(test["season"].unique()) == (set(TEST_SEASONS) & present)


@pytest.mark.unit
def test_rolling_origin_folds_custom_test_seasons(_sample_df):
    folds = rolling_origin_folds(_sample_df, test_seasons=[2024], min_train_season=2021)
    assert len(folds) == 1
    _, train, val, test = folds[0]
    assert set(train["season"].unique()) == {2021, 2022}
    assert set(val["season"].unique()) == {2023}
    assert set(test["season"].unique()) == {2024}


@pytest.mark.unit
def test_rolling_origin_folds_asserts_when_season_type_absent(_sample_df_no_season_type):
    with pytest.raises(AssertionError, match="season_type"):
        rolling_origin_folds(_sample_df_no_season_type, test_seasons=[2021], min_train_season=2020)


@pytest.mark.unit
def test_rolling_origin_folds_drops_postseason(_sample_df_with_season_type):
    folds = rolling_origin_folds(_sample_df_with_season_type, test_seasons=[2025])
    _, _, val, _ = folds[0]  # val season 2024 carried the POST rows
    assert (val["season_type"] != "POST").all()


@pytest.mark.unit
def test_rolling_origin_folds_raises_when_no_train_seasons(_sample_df):
    """A test season leaving no train seasons above min_train_season is a config
    error, not a silent empty-train run."""
    with pytest.raises(ValueError, match="no train seasons"):
        rolling_origin_folds(_sample_df, test_seasons=[2013], min_train_season=2012)


@pytest.mark.unit
def test_rolling_origin_folds_imputes_snap_pct_train_only():
    """Each origin's val/test ``snap_pct`` NaN is filled with that origin's
    TRAIN-only median — same train-only contract as expanding_window_folds."""
    import numpy as np

    rows = [
        # Train (2021, QB, wk1): 0.20, 0.30 -> train-only median 0.25
        {
            "season": 2021,
            "week": 1,
            "player_id": "T1",
            "position": "QB",
            "snap_pct": 0.20,
            "season_type": "REG",
        },
        {
            "season": 2021,
            "week": 1,
            "player_id": "T2",
            "position": "QB",
            "snap_pct": 0.30,
            "season_type": "REG",
        },
        # Val (2022, QB, wk1): observed 0.90 + a NaN to impute.
        {
            "season": 2022,
            "week": 1,
            "player_id": "V1",
            "position": "QB",
            "snap_pct": 0.90,
            "season_type": "REG",
        },
        {
            "season": 2022,
            "week": 1,
            "player_id": "V2",
            "position": "QB",
            "snap_pct": np.nan,
            "season_type": "REG",
        },
        # Test (2023, QB, wk1): a NaN to impute.
        {
            "season": 2023,
            "week": 1,
            "player_id": "X1",
            "position": "QB",
            "snap_pct": np.nan,
            "season_type": "REG",
        },
    ]
    df = pd.DataFrame(rows)
    folds = rolling_origin_folds(df, test_seasons=[2023], min_train_season=2021)
    _, train, val, test = folds[0]
    assert set(train["snap_pct"].round(2)) == {0.20, 0.30}
    assert val[val["player_id"] == "V2"].iloc[0]["snap_pct"] == pytest.approx(0.25)
    assert test[test["player_id"] == "X1"].iloc[0]["snap_pct"] == pytest.approx(0.25)
