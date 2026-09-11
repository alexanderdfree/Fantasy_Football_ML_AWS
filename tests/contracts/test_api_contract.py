"""Wire contracts run without models, data, AWS, or the production app."""

import copy
import json
import unittest
from pathlib import Path

from flask import Flask

from src.contracts.api import API_CONTRACT, ContractError, install_api_contract, validate_response

FIXTURES = Path(__file__).resolve().parents[2] / "ios/Tests/Fixtures"


class APIContractTests(unittest.TestCase):
    def test_contract_installation_is_per_app_and_idempotent(self):
        for app in (Flask("first"), Flask("second")):
            install_api_contract(app)
            install_api_contract(app)
            response = app.test_client().get("/api/contract")
            self.assertEqual(response.get_json(), API_CONTRACT)
            self.assertEqual(response.headers["X-FFP-Contract-Version"], "1.0")

    def test_current_shared_fixtures_and_legacy_comparison(self):
        fixture = json.loads((FIXTURES / "client_contract.json").read_text())
        validate_response("/api/snapshot", fixture["snapshot"])
        for body in fixture["predictions"].values():
            validate_response("/api/predictions", body)
        for name in ("comparison", "comparison_current"):
            validate_response(
                "/api/comparison", json.loads((FIXTURES / f"{name}.json").read_text())
            )
        current = json.loads((FIXTURES / "comparison_current.json").read_text())
        self.assertEqual(current["actual_basis"], API_CONTRACT["comparison"]["actual_basis"])
        self.assertEqual(current["sample_basis"], API_CONTRACT["comparison"]["sample_basis"])
        self.assertEqual(
            set(current["subsets"]["all"]["QB"]),
            set(API_CONTRACT["model_sources"] + API_CONTRACT["expert_sources"]),
        )

    def test_missing_is_not_zero_and_wrong_type_is_rejected(self):
        fixture = json.loads((FIXTURES / "client_contract.json").read_text())["snapshot"]
        for value in (None, 0):
            data = copy.deepcopy(fixture)
            data["scoring"]["ppr"][0]["ridge_pred"] = value
            validate_response("/api/snapshot", data)
        for value in ("0", True, float("nan"), float("inf")):
            data["scoring"]["ppr"][0]["ridge_pred"] = value
            with self.assertRaises(ContractError):
                validate_response("/api/snapshot", data)
        del fixture["scoring"]["standard"]
        with self.assertRaises(ContractError):
            validate_response("/api/snapshot", fixture)

    def test_warming_unavailable_and_error_envelopes(self):
        validate_response("/api/upcoming_week", {"status": "warming"}, 503)
        validate_response("/api/upcoming_week", {"available": False, "reason": "offseason"})
        validate_response("/api/snapshot", {"error": "snapshot not available"}, 404)
        with self.assertRaises(ContractError):
            validate_response("/api/upcoming_week", {"available": False})
