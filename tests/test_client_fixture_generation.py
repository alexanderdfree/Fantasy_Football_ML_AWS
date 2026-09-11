"""Keep client fixtures tied to real local server serialization behavior."""

import json

import pytest

from ios.scripts.generate_client_fixtures import OUTPUT, client_fixtures
from ios.scripts.generate_comparison_fixture import FIXTURE, comparison_fixture
from src.contracts.api import API_CONTRACT
from src.contracts.export import OUTPUT as CONTRACT_OUTPUT

pytestmark = pytest.mark.unit


def test_browser_contract_is_current():
    assert json.loads(CONTRACT_OUTPUT.read_text()) == API_CONTRACT


def test_native_comparison_fixture_is_current():
    assert json.loads(FIXTURE.read_text()) == comparison_fixture()


def test_shared_prediction_fixtures_are_current():
    assert json.loads(OUTPUT.read_text()) == client_fixtures()
