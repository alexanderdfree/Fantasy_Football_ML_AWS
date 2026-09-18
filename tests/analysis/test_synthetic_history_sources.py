"""Source export for skill positions with the production loaders replaced by fakes."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis import synthetic_history_sources as sources
from src.analysis.synthetic_history_schema import position_schema
from tests.analysis.conftest import position_rows

pytestmark = pytest.mark.unit


def prepared_dataset(frame, position):
    return SimpleNamespace(
        train=frame,
        feature_columns=position_schema(position).feature_columns,
        data_id="fake-data-id",
    )


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
        return prepared_dataset(position_rows(position), position)

    monkeypatch.setattr(sources, "_read_split", read_split)
    monkeypatch.setattr(sources, "_prepare_position_data", prepare)
    return splits, calls


def test_export_prepares_the_train_frame_the_production_way(fake_loaders):
    splits, calls = fake_loaders
    prepared = sources.export_skill_source("RB", splits_dir=splits)
    assert calls == [("read", "train.parquet"), ("read", "val.parquet"), ("prepare", "RB", 1, 1)]
    assert prepared.train["position"].eq("RB").all()
    with pytest.raises(ValueError, match="not a skill position"):
        sources.export_skill_source("K", splits_dir=splits)


def test_write_sources_publishes_parquet_and_hashes_without_overwriting(fake_loaders, tmp_path):
    splits, calls = fake_loaders
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
    assert manifest["prepared_data_id"] == "fake-data-id"
    assert {"src/wr/config.py", "src/wr/features.py", "src/shared/team_box_score.py"} <= set(
        manifest["code_sha256"]
    )
    calls.clear()
    with pytest.raises(FileExistsError):
        sources.write_sources("WR", tmp_path / "wr", splits_dir=splits)
    assert calls == []  # refused before any preparation ran


def test_cli_accepts_lowercase_positions_and_refuses_overwrites(fake_loaders, tmp_path, capsys):
    splits, _ = fake_loaders
    output = tmp_path / "te"
    argv = ["--position", "te", "--splits-dir", str(splits), "--output", str(output)]
    assert sources.main(argv) == 0
    # The production preparation logs first; the summary is the final stdout line.
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["source"] == "te.parquet" and summary["duplicate_game_keys"] == 0
    with pytest.raises(SystemExit) as exit_info:
        sources.main(argv)
    assert exit_info.value.code == 2


def test_duplicate_game_keys_are_reported_not_repaired_and_hash_ignores_row_order(
    fake_loaders, monkeypatch, tmp_path
):
    splits, _ = fake_loaders
    frame = position_rows("TE")
    twin = frame.iloc[[0, 1]].assign(snap_pct_raw=0.11)
    forward, backward = (
        pd.concat([frame, twin], ignore_index=True),
        pd.concat([twin, frame], ignore_index=True),
    )
    digests = []
    for name, ordered in (("a", forward), ("b", backward)):
        monkeypatch.setattr(
            sources,
            "_prepare_position_data",
            lambda *a, ordered=ordered, **k: prepared_dataset(ordered, "TE"),
        )
        output = sources.write_sources("TE", tmp_path / name, splits_dir=splits)
        manifest = json.loads((output / "sources.json").read_text())
        assert manifest["rows"] == len(frame) + 2
        assert manifest["duplicate_game_keys"] == [
            {"player_id": "p1", "season": 2022, "weeks": [1, 2], "rows": 4}
        ]
        assert "whole prepared frame" in manifest["duplicate_game_keys_scope"]
        assert len(pd.read_parquet(output / "te.parquet")) == len(frame) + 2
        digests.append(manifest["values_sha256"])
    assert digests[0] == digests[1]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda f: f.drop(columns="rolling_mean_carries_L3"), "lacks columns"),
        (lambda f: f.drop(columns="game_carry_share"), "lacks columns"),
        (lambda f: f.assign(week=[np.nan] + list(f["week"][1:])), "missing player/season/week"),
    ],
)
def test_incomplete_prepared_frames_are_rejected(fake_loaders, monkeypatch, mutate, error):
    splits, _ = fake_loaders
    frame = mutate(position_rows("RB"))
    monkeypatch.setattr(
        sources, "_prepare_position_data", lambda *a, **k: prepared_dataset(frame, "RB")
    )
    with pytest.raises(ValueError, match=error):
        sources.export_skill_source("RB", splits_dir=splits)


def test_feature_column_disagreement_with_the_registry_is_rejected(fake_loaders, monkeypatch):
    splits, _ = fake_loaders
    prepared = prepared_dataset(position_rows("WR"), "WR")
    prepared.feature_columns = prepared.feature_columns[::-1]
    monkeypatch.setattr(sources, "_prepare_position_data", lambda *a, **k: prepared)
    with pytest.raises(ValueError, match="differ from the registry"):
        sources.export_skill_source("WR", splits_dir=splits)
