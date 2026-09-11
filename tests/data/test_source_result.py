"""Source contracts retain observed zero, empty success, and failed retrieval."""

import pandas as pd
import pytest

from src.data import identity, nflcom_loader
from src.data.source_result import SourceResult, SourceStatus, content_identity

pytestmark = pytest.mark.unit


def test_normalization_reexports_preserve_identity_and_join_universes():
    assert nflcom_loader.normalize_team_code is identity.normalize_team_code
    assert nflcom_loader.normalize_player_name is identity.normalize_player_name
    assert (
        nflcom_loader.schedule_team_code_normalization is identity.schedule_team_code_normalization
    )
    assert nflcom_loader.TEAM_CODE_MAP is identity.TEAM_CODE_MAP
    assert identity.normalize_team_code("@STL") == "LAR"
    assert identity.schedule_team_code_normalization()["STL"] == "LA"
    assert identity.normalize_player_name("A.J. Brown Jr.") == "aj brown"


def test_source_outcomes_do_not_turn_unknown_values_into_observed_zero():
    zero = SourceResult.capture(pd.DataFrame({"wind": [0.0]}), provider="weather")
    missing = SourceResult.capture(
        pd.DataFrame({"wind": [float("nan")]}), provider="weather", status="partial"
    )
    empty = SourceResult.capture(pd.DataFrame({"wind": []}), provider="weather")
    failed = SourceResult.capture(
        pd.DataFrame({"wind": []}), provider="weather", status="unavailable", errors=("offline",)
    )
    assert zero.status is SourceStatus.AVAILABLE
    assert zero.content_id != missing.content_id
    assert empty.status is SourceStatus.EMPTY and empty.content_id is not None
    assert failed.status is SourceStatus.UNAVAILABLE and failed.content_id is None
    assert failed.metadata()["errors"] == ["offline"]


def test_content_identity_is_stable_across_retrievals_but_binds_player_assignments():
    data = pd.DataFrame({"value": [1.0, 2.0]}, index=["a", "b"])
    first = SourceResult.capture(data, provider="contracts", retrieved_at="2026-09-10T12:00Z")
    second = SourceResult.capture(
        data.copy(), provider="contracts", retrieved_at="2026-09-10T13:00Z"
    )
    assert first.content_id == second.content_id
    assert first.retrieved_at != second.retrieved_at
    assert content_identity(data.set_axis(["b", "a"])) != first.content_id
    assert content_identity(data[["value"]].assign(value=[2.0, 1.0])) != first.content_id
