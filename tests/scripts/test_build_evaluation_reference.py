"""Reference artifacts contain a complete pregame slate, never outcome-selected rows."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.scripts import build_evaluation_reference as builder
from src.shared.evaluation_cohorts import REFERENCE_VERSION

pytestmark = pytest.mark.unit


def fake_sources(monkeypatch, nfl_rows, rw_rows):
    skipped = frozenset({"QB", "RB", "TE", "K", "DST"})

    def source(name, rows):
        return SimpleNamespace(
            name=name,
            skipped=skipped,
            load=lambda seasons: rows.copy(),
            project=lambda raw, pos, scoring: raw.copy(),
        )

    monkeypatch.setattr(
        "src.analysis.analysis_expert_comparison._build_experts",
        lambda *args: [source("nflcom", nfl_rows), source("sleeper", rw_rows)],
    )


def rows():
    return pd.DataFrame(
        {
            "player_id": ["b", "a", "c"],
            "season": 2025,
            "week": 1,
            "position": "WR",
            "expert_pred_total": [20.0, 20.0, 10.0],
        }
    )


def test_reference_is_fixed_mean_and_ties_are_deterministic(monkeypatch):
    nfl, rw = rows(), rows()
    rw.loc[2, "expert_pred_total"] = 40
    fake_sources(monkeypatch, nfl, rw)
    ref = builder.build_reference([2025])
    assert ref.player_id.tolist() == ["c", "a", "b"]
    assert ref.reference_rank.tolist() == [1, 2, 3]
    assert ref.reference_pred.tolist() == [25, 20, 20]
    assert set(ref.reference_version) == {REFERENCE_VERSION}
    assert set(ref.reference_source) == {"nflcom+rotowire"}
    assert "fantasy_points" not in ref


def test_missing_provider_does_not_change_reference_recipe(monkeypatch):
    fake_sources(monkeypatch, rows(), rows().iloc[:2])
    ref = builder.build_reference([2025])
    assert set(ref.player_id) == {"a", "b"}


def test_kicker_reference_uses_matching_espn_components_not_nfl_native_totals(monkeypatch):
    def source(name, points):
        return SimpleNamespace(
            name=name,
            skipped=frozenset(),
            load=lambda seasons: rows().assign(expert_pred_total=points),
            project=lambda raw, pos, scoring: raw.copy(),
        )

    monkeypatch.setattr(
        "src.analysis.analysis_expert_comparison._build_experts",
        lambda *args: [source("nflcom", [900, 800, 700]), source("espn", [5, 7, 9])],
    )
    ref = builder.build_reference([2025])
    assert set(ref.position) == {"K"}
    assert set(ref.reference_source) == {"espn"}
    assert ref.player_id.tolist() == ["c", "a", "b"]
    assert ref.reference_pred.tolist() == [9, 7, 5]


def test_backfilled_nflcom_offense_seasons_cannot_enter_reference(monkeypatch):
    data = pd.concat([rows(), rows().assign(season=2023)], ignore_index=True)
    fake_sources(monkeypatch, data, data)
    ref = builder.build_reference([2023, 2025])
    assert set(ref.season) == {2025}


def test_publication_preserves_other_seasons_and_rejects_empty_replacement(tmp_path, monkeypatch):
    path = tmp_path / "reference.parquet"
    old = rows().assign(season=2024, reference_version=REFERENCE_VERSION)
    old.to_parquet(path)
    monkeypatch.setattr(builder, "reference_path", lambda: path)
    monkeypatch.setattr(
        builder,
        "build_reference",
        lambda *a, **k: rows().assign(reference_version=REFERENCE_VERSION),
    )
    builder.write_reference([2025])
    assert set(pd.read_parquet(path).season) == {2024, 2025}
    before = path.read_bytes()
    monkeypatch.setattr(builder, "build_reference", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="existing artifact retained"):
        builder.write_reference([2025])
    assert path.read_bytes() == before


def test_publication_preserves_other_recipe_versions(tmp_path, monkeypatch):
    path = tmp_path / "reference.parquet"
    other = rows().assign(reference_version="another_recipe")
    other.to_parquet(path)
    monkeypatch.setattr(builder, "reference_path", lambda: path)
    monkeypatch.setattr(
        builder,
        "build_reference",
        lambda *a, **k: rows().assign(reference_version=REFERENCE_VERSION),
    )
    builder.write_reference([2025])
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        saved[saved.reference_version.eq("another_recipe")].reset_index(drop=True), other
    )
    assert len(saved[saved.reference_version.eq(REFERENCE_VERSION)]) == 3


def test_refresh_with_superset_loader_cache_does_not_duplicate_other_seasons(tmp_path, monkeypatch):
    cached = pd.concat([rows().assign(season=2024), rows()], ignore_index=True)
    fake_sources(monkeypatch, cached, cached)
    path = tmp_path / "reference.parquet"
    monkeypatch.setattr(builder, "reference_path", lambda: path)
    builder.write_reference([2024, 2025])
    builder.write_reference([2025])
    saved = pd.read_parquet(path)
    assert len(saved) == 6
    assert not saved.duplicated(
        ["position", "player_id", "season", "week", "reference_version"]
    ).any()


def test_partial_refresh_cannot_erase_archived_weeks(tmp_path, monkeypatch):
    complete = pd.concat([rows(), rows().assign(week=2)], ignore_index=True)
    fake_sources(monkeypatch, complete, complete)
    path = tmp_path / "reference.parquet"
    monkeypatch.setattr(builder, "reference_path", lambda: path)
    builder.write_reference([2025])
    before = path.read_bytes()
    fake_sources(monkeypatch, rows(), rows())
    with pytest.raises(ValueError, match="loses archived forecast coverage"):
        builder.write_reference([2025])
    assert path.read_bytes() == before


def test_loaders_receive_only_supported_seasons(monkeypatch):
    calls = {}
    skipped = frozenset({"QB", "RB", "TE", "K", "DST"})

    def source(name):
        def load(seasons):
            calls[name] = seasons
            return rows()

        return SimpleNamespace(
            name=name, skipped=skipped, load=load, project=lambda raw, pos, scoring: raw.copy()
        )

    monkeypatch.setattr(
        "src.analysis.analysis_expert_comparison._build_experts",
        lambda *a: [source("nflcom"), source("sleeper")],
    )
    result = builder.build_reference([2017, 2025])
    assert calls == {"nflcom": [2017, 2025], "sleeper": [2025]}
    assert set(result.season) == {2025}
    calls.clear()
    assert builder.build_reference([2012]).empty
    assert calls == {}
