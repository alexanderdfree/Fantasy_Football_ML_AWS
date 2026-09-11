"""Exercise the production API with ML and offline-builder imports forbidden.

Run with the serving interpreter; --require-absent also checks installed packages.
Uses a temporary, locally published six-position artifact and never accesses AWS.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ML_PACKAGES = {
    "torch",
    "scikit-learn",
    "lightgbm",
    "scipy",
    "matplotlib",
    "joblib",
    "mord",
    "shap",
    "nflreadpy",
    "polars",
}
ML_IMPORTS = (ML_PACKAGES - {"scikit-learn"}) | {"sklearn"}


class RuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ML_IMPORTS or any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in ("src.prediction", "src.training", "src.features")
        ):
            raise ImportError(f"Forbidden serving dependency: {fullname}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-absent", action="store_true")
    args = parser.parse_args()
    if args.require_absent:
        installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
        assert not installed & ML_PACKAGES, sorted(installed & ML_PACKAGES)
    sys.path.insert(0, str(Path.cwd()))
    sys.meta_path.insert(0, RuntimeImports())
    os.environ["FF_ALLOW_RUNTIME_INFERENCE"] = "0"
    for name in ("FF_DATA_RELEASE", "FF_DATASET_ID", "FF_MODEL_S3_BUCKET"):
        os.environ.pop(name, None)

    import pandas as pd

    from src.artifacts import serving_snapshot
    from src.artifacts.position_metadata import POSITION_INFO
    from src.contracts.serialization import (
        _MODEL_PRED_COLUMNS,
        _MODEL_PRED_PREFIXES,
        _ROW_PRED_PREFIXES,
        _VALID_SCORING,
        _actual_col,
        _records_to_player_rows,
    )
    from src.serving import core
    from src.serving.app import create_app

    rows = []
    for position, info in POSITION_INFO.items():
        for week in (1, 2):
            row = {
                "player_id": f"{position}-demo",
                "player_name": position + " Player",
                "player_display_name": position + " Player",
                "position": position,
                "team": "BUF",
                "recent_team": "BUF",
                "season": 2025,
                "week": week,
            }
            for scoring in _VALID_SCORING:
                row[_actual_col(scoring)] = float(10 + week)
                for model in _ROW_PRED_PREFIXES:
                    row[f"{model}_pred_{scoring}"] = float(9 + week)
                    row[f"{model}_comparison_pred_{scoring}"] = float(9 + week)
                    if position == "DST":
                        row[f"{model}_pred_comparison"] = float(9 + week)
            for target in info["targets"]:
                row[f"actual_{target['key']}"] = float(week)
                for model in _MODEL_PRED_PREFIXES:
                    row[f"{model}_{target['key']}"] = float(week)
            rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = {
        name: {"overall": {"mae": 1.0, "rmse": 1.0, "r2": -3.0}, "by_position": []}
        for name, _ in _MODEL_PRED_COLUMNS
    }
    comparison = {"scoring": "ppr", "model_source": "live", "subsets": {}, "coverage": {}}
    snapshot = {
        "weeks": [1, 2],
        "degraded_positions": [],
        "scoring": {fmt: _records_to_player_rows(frame, fmt) for fmt in _VALID_SCORING},
    }
    parquet = io.BytesIO()
    frame.to_parquet(parquet, index=True)
    files = {
        "predictions.parquet": parquet.getvalue(),
        "metrics.json": json.dumps(
            {
                "metrics_by_format": {fmt: metrics for fmt in _VALID_SCORING},
                "comparison_snapshot": comparison,
            }
        ).encode(),
        "fingerprint.json": json.dumps(
            {
                "schema_version": serving_snapshot.CACHE_SCHEMA_VERSION,
                "sha256": "probe",
            }
        ).encode(),
        "snapshot.json": json.dumps(snapshot).encode(),
    }
    with tempfile.TemporaryDirectory(prefix="ff-serving-probe-") as directory:
        core._PREDICTIONS_CACHE_DIR = directory
        application = create_app()
        client = application.test_client()
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        generation = serving_snapshot.publish_local(directory, files)
        assert client.get("/ready").status_code == 200
        paths = [
            "/",
            "/privacy",
            "/support",
            "/health",
            "/ready",
            "/warm",
            "/api/predictions?position=ALL",
            "/api/snapshot",
            "/api/metrics",
            "/api/weeks",
            "/api/teams",
            "/api/player/QB-demo",
            "/api/predictions/breakdown?player_id=QB-demo&week=1",
            "/api/weekly_accuracy",
            "/api/position_details",
            "/api/model_architecture",
            "/api/comparison",
            "/api/benchmark_history",
            "/api/timeline",
            "/api/wiki/index",
            "/api/contract",
        ]
        for path in paths:
            response = client.get(path)
            assert response.status_code == 200, (
                path,
                response.status_code,
                response.get_data(as_text=True)[:500],
            )
        assert client.get("/api/snapshot").get_json() == snapshot
        assert client.get("/api/comparison").get_json() == comparison
        assert client.get("/api/metrics").get_json() == metrics
        for group in ("offense", "k", "dst"):
            for scoring in _VALID_SCORING:
                timeline = client.get(f"/api/timeline?group={group}&scoring={scoring}")
                assert timeline.status_code == 200
                assert timeline.get_json()["summary"]["status"] == "available"
                assert timeline.get_json()["summary"]["n"] > 0
        assert client.get("/api/upcoming_week").status_code == 503
        upcoming = {
            "available": True,
            "season": 2025,
            "week": 1,
            "generated_at": "2000-01-01T00:00:00Z",
            "scoring": snapshot["scoring"],
        }
        (Path(directory) / "upcoming_week.json").write_text(json.dumps(upcoming))
        live = client.get("/api/upcoming_week")
        assert live.status_code == 200
        assert live.get_json()["available"] is True
        assert live.get_json()["scoring"] == snapshot["scoring"]
        # Fresh workers reject damaged generations without importing a builder.
        (generation / "metrics.json").write_bytes(b"damaged")
        cold = create_app().test_client()
        assert cold.get("/ready").status_code == 503
        assert cold.get("/api/predictions").status_code == 503
        assert cold.get("/api/comparison").status_code == 503
    print(
        json.dumps(
            {"runtime_probe": "passed", "endpoints": len(paths), "positions": len(POSITION_INFO)}
        )
    )


if __name__ == "__main__":
    main()
