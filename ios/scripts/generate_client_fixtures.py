"""Build shared browser/Swift fixtures using real API serializers and routes."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.contracts.api import POSITIONS, SCORING_FORMATS, validate_response
from src.serving import app as app_module
from src.serving import core, routes, state
from src.serving.serialization import _ROW_PRED_PREFIXES, _actual_col

OUTPUT = Path(__file__).resolve().parents[1] / "Tests/Fixtures/client_contract.json"


def client_fixtures():
    records = []
    for index, pos in enumerate(POSITIONS):
        row = {
            "player_id": f"fixture-{pos}",
            "player_display_name": f"Fixture {pos}",
            "position": pos,
            "recent_team": "KC",
            "season": 2025,
            "week": 1,
        }
        for scoring, value in zip(SCORING_FORMATS, (30.0, 20.0, 10.0), strict=True):
            row[_actual_col(scoring)] = value + index
            for source in _ROW_PRED_PREFIXES:
                row[f"{source}_pred_{scoring}"] = value if pos != "TE" else np.nan
            if pos == "K":
                row[f"ridge_pred_{scoring}"] = 0.0
        records.append(row)
    frame = pd.DataFrame(records)
    fixtures = {"predictions": {}, "player": {}}
    owner = state.ServingState(cache={"results": frame, "metrics": {}})
    owner.cache["model_bundle_ids"] = {"QB": {"nn": "fixture-bundle-nn"}}
    owner.cache["model_metadata"] = {
        "QB": {
            "nn": {
                "bundle_id": "fixture-bundle-nn",
                "status": "available",
                "inputs": {"features": ["fixture_input"], "targets": ["passing_yards"]},
                "architecture": {
                    "class": "MultiHeadNet",
                    "kwargs": {"backbone_layers": [7], "head_hidden": 3, "dropout": 0.125},
                },
                "training_options": {"nn_lr": 0.002, "nn_epochs": 4, "scheduler_type": "plateau"},
                "provenance": {"dependencies": {"torch": "fixture-version"}},
            }
        }
    }
    app = app_module.create_app(serving_state=owner)
    with (
        state.use_state(owner),
        patch.object(core, "_degraded_positions", return_value=["TE"]),
        patch.object(core, "_ensure_base_data", return_value=None),
        patch.object(core, "_ensure_position_loaded", return_value=None),
        patch.object(routes, "_results_for_position", return_value=frame),
        app.test_client() as client,
    ):
        # Route fixtures use a complete app-owned in-memory snapshot. Disk
        # generation integrity is exercised separately by the artifact tests.
        owner.publish()
        response = client.get("/api/snapshot")
        assert response.status_code == 200
        fixtures["snapshot"] = response.get_json()
        fixtures["snapshot"]["generated_at"] = "2026-09-10T00:00:00+00:00"
        validate_response("/api/snapshot", fixtures["snapshot"])
        response = client.get("/api/model_architecture")
        assert response.status_code == 200
        fixtures["architecture"] = response.get_json()
        for scoring in SCORING_FORMATS:
            for key, path in (
                ("predictions", "/api/predictions"),
                ("player", "/api/player/fixture-QB"),
            ):
                response = client.get(f"{path}?scoring={scoring}")
                assert response.status_code == 200
                fixtures[key][scoring] = response.get_json()
                validate_response(path, fixtures[key][scoring])
    return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(client_fixtures(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if OUTPUT.read_text() != rendered:
            raise SystemExit(
                "Client fixtures are stale: python -m ios.scripts.generate_client_fixtures"
            )
    else:
        OUTPUT.write_text(rendered)


if __name__ == "__main__":
    main()
