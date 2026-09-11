"""Generate the native comparison fixture through the local Flask API.

Run from the repository root with ``python -m ios.scripts.generate_comparison_fixture``.
Synthetic inputs exercise all six positions, excluded sources, missing forecasts,
partial/missing references, unavailable actuals, and a real zero-error metric.
No model loading, production requests, or local data artifacts are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.serving import app as app_module
from src.serving import comparison, core, state
from src.serving.serialization import _EXPERT_PRED_PREFIXES, _ROW_PRED_PREFIXES
from src.shared.comparison_scoring import score_actual_components, scoring_components
from src.shared.evaluation_cohorts import REFERENCE_VERSION

FIXTURE = Path(__file__).resolve().parents[1] / "Tests/Fixtures/comparison_current.json"


def comparison_fixture() -> dict:
    """Serialize real route behavior using deterministic, isolated source data."""
    records = []
    references = []
    for pos in comparison.COMPARISON_POSITIONS:
        for week in (1, 2):
            for player in range(3):
                row = {
                    "player_id": f"fixture-{pos}-{player}",
                    "position": pos,
                    "season": 2025,
                    "week": week,
                    "season_type": "REG",
                    **{f"actual_{key}": float(player + week) for key in scoring_components(pos)},
                }
                actual = score_actual_components(pd.DataFrame([row]), pos, prefix="actual_").iloc[0]
                for index, source in enumerate(_ROW_PRED_PREFIXES):
                    row[f"{source}_pred_ppr"] = float(actual + index)
                if pos == "K":
                    row["espn_pred_ppr"] = float(actual)
                    row["rotowire_pred_ppr"] = np.nan
                if pos == "DST":
                    row["nflcom_pred_ppr"] = np.nan
                if pos == "RB" and week == 1 and player == 0:
                    row["espn_pred_ppr"] = np.nan
                if pos == "TE":
                    row["actual_receptions"] = np.nan
                for source in _ROW_PRED_PREFIXES:
                    if pos == "DST":
                        row[f"{source}_pred_comparison"] = row[f"{source}_pred_ppr"]
                        row[f"{source}_pred_ppr"] += 10.0
                    elif source in _EXPERT_PRED_PREFIXES:
                        row[f"{source}_comparison_pred_ppr"] = row[f"{source}_pred_ppr"]
                records.append(row)
                if pos != "DST" and not (pos == "WR" and week == 2):
                    references.append(
                        {
                            **{
                                key: row[key] for key in ("player_id", "position", "season", "week")
                            },
                            "reference_rank": player + 1,
                            "reference_version": REFERENCE_VERSION,
                        }
                    )
    metadata = {"experts_meta": {"espn": {"label": "ESPN", "note": "Synthetic fixture forecasts."}}}
    owner = state.ServingState(cache={"results": pd.DataFrame(records)})
    app = app_module.create_app(serving_state=owner)
    with (
        patch.object(core, "_ensure_metrics", return_value=None),
        patch.object(comparison, "load_reference", return_value=pd.DataFrame(references)),
        patch.object(comparison, "_load_comparison_experts", return_value=metadata),
        app.test_client() as client,
    ):
        response = client.get("/api/comparison")
        if response.status_code != 200:
            raise RuntimeError(f"Fixture API returned HTTP {response.status_code}")
        body = response.get_json()
    body["generated_at"] = "2026-09-10T00:00:00+00:00"
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the fixture differs from the API"
    )
    args = parser.parse_args()
    rendered = json.dumps(comparison_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if FIXTURE.read_text() != rendered:
            raise SystemExit("Native comparison fixture is stale; regenerate it locally.")
    else:
        FIXTURE.write_text(rendered)


if __name__ == "__main__":
    main()
