"""Source export for skill positions with the production loaders replaced by fakes."""

import hashlib
import json

import pandas as pd
import pytest

from src.analysis import synthetic_history_sources as sources
from tests.analysis.conftest import position_rows

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_loaders(monkeypatch, tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "val"):
        pd.DataFrame({"player_id": ["p1"], "season": [2022], "week": [1]}).to_parquet(
            splits / f"{name}.parquet", index=False
        )
    calls = []

    def read_split(path):
        calls.append(("read", path.name))
        return pd.read_parquet(path)

    def prepare(position, config, train, val, test=None):
        calls.append(("prepare", position, len(train), len(val)))
        frame = position_rows(position)
        return (None, None, None, None, None, None, frame, None, None, [])

    monkeypatch.setattr(sources, "_read_split", read_split)
    monkeypatch.setattr(sources, "_prepare_position_data", prepare)
    return splits, calls


def test_export_prepares_the_train_frame_the_production_way(fake_loaders):
    splits, calls = fake_loaders
    frame = sources.export_skill_source("RB", splits_dir=splits)
    assert calls == [("read", "train.parquet"), ("read", "val.parquet"), ("prepare", "RB", 1, 1)]
    assert frame["position"].eq("RB").all()
    with pytest.raises(ValueError, match="not a skill position"):
        sources.export_skill_source("K", splits_dir=splits)


def test_write_sources_publishes_parquet_and_hashes_without_overwriting(fake_loaders, tmp_path):
    splits, _ = fake_loaders
    output = sources.write_sources("WR", tmp_path / "wr", splits_dir=splits)
    manifest = json.loads((output / "sources.json").read_text())
    assert manifest["position"] == "WR" and manifest["source"] == "wr.parquet"
    assert manifest["rows"] == len(pd.read_parquet(output / "wr.parquet"))
    assert (
        manifest["files"]["wr.parquet"]
        == hashlib.sha256((output / "wr.parquet").read_bytes()).hexdigest()
    )
    assert set(manifest["splits"]) == {"train", "val"}
    assert manifest["duplicate_game_keys"] == []
    assert "src/wr/config.py" in manifest["code_sha256"]
    with pytest.raises(FileExistsError):
        sources.write_sources("WR", tmp_path / "wr", splits_dir=splits)


def test_cli_round_trip_and_missing_feature_columns_fail(fake_loaders, tmp_path, capsys):
    splits, _ = fake_loaders
    output = tmp_path / "te"
    argv = ["--position", "TE", "--splits-dir", str(splits), "--output", str(output)]
    assert sources.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "te.parquet"
    with pytest.raises(SystemExit) as exit_info:
        sources.main(argv)
    assert exit_info.value.code == 2


def test_duplicate_game_keys_are_reported_not_repaired(fake_loaders, monkeypatch, tmp_path):
    splits, _ = fake_loaders
    frame = position_rows("TE")
    frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    monkeypatch.setattr(sources, "_prepare_position_data", lambda *a, **k: (None,) * 6 + (frame,))
    output = sources.write_sources("TE", tmp_path / "te", splits_dir=splits)
    manifest = json.loads((output / "sources.json").read_text())
    assert manifest["rows"] == len(frame)
    assert manifest["duplicate_game_keys"] == [
        {"player_id": "p1", "season": 2022, "weeks": [1, 2], "rows": 4}
    ]
    assert len(pd.read_parquet(output / "te.parquet")) == len(frame)


def test_incomplete_prepared_frame_is_rejected(fake_loaders, monkeypatch):
    splits, _ = fake_loaders
    monkeypatch.setattr(
        sources,
        "_prepare_position_data",
        lambda *a, **k: (
            (None,) * 6 + (position_rows("RB").drop(columns="rolling_mean_carries_L3"),)
        ),
    )
    with pytest.raises(ValueError, match="lacks feature columns"):
        sources.export_skill_source("RB", splits_dir=splits)
