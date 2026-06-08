"""Tests for ``src.data.cache_io.atomic_write_parquet`` — the atomic cache-write
helper that fixes the pytest-xdist ``ArrowInvalid`` race (a reader seeing a
half-written cache parquet; #1056/#1057 flake family)."""

from __future__ import annotations

import glob
import os

import pandas as pd
import pytest

from src.data.cache_io import atomic_write_parquet


@pytest.mark.unit
def test_round_trip_and_creates_nested_dir(tmp_path):
    path = str(tmp_path / "sub" / "cache.parquet")  # dir does not exist yet
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    atomic_write_parquet(df, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)


@pytest.mark.unit
def test_forwards_kwargs(tmp_path):
    path = str(tmp_path / "cache.parquet")
    df = pd.DataFrame({"a": [1, 2]}, index=["r0", "r1"])
    atomic_write_parquet(df, path, index=False)
    # index=False → the custom index is dropped (default RangeIndex on read).
    assert list(pd.read_parquet(path).index) == [0, 1]


@pytest.mark.unit
def test_never_exposes_partial_file_mid_write(tmp_path, monkeypatch):
    """The atomicity guarantee: ``os.path.exists(path)`` is False until the file
    is COMPLETE — so a concurrent reader (the xdist race) never sees a partial.
    We spy on ``to_parquet`` and assert it writes to a *temp*, not the path."""
    path = str(tmp_path / "cache.parquet")
    seen = {}
    real = pd.DataFrame.to_parquet

    def spy(self, target, **kw):
        seen["target_is_temp"] = str(target) != path
        seen["path_exists_during_write"] = os.path.exists(path)
        return real(self, target, **kw)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", spy)
    atomic_write_parquet(pd.DataFrame({"a": [1]}), path)

    assert seen["target_is_temp"] is True
    assert seen["path_exists_during_write"] is False
    assert os.path.exists(path)  # complete afterwards


@pytest.mark.unit
def test_failed_write_leaves_target_intact_and_no_temp(tmp_path):
    """A write failure must not corrupt/create the target or strand a temp —
    the reader keeps seeing the previous complete file."""
    path = str(tmp_path / "cache.parquet")
    good = pd.DataFrame({"a": [1, 2, 3]})
    atomic_write_parquet(good, path)  # establish a complete cache

    with pytest.raises(ValueError):
        atomic_write_parquet(good, path, engine="does-not-exist")  # forced failure

    pd.testing.assert_frame_equal(pd.read_parquet(path), good)  # original intact
    assert not glob.glob(str(tmp_path / ".tmp-*"))  # no stray temp
