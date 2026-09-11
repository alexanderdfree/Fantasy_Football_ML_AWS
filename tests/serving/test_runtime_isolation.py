"""Exercise both dependency boundaries in fresh interpreters."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def run_python(arguments):
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-5000:]
    return completed.stdout


def test_api_serves_verified_artifacts_without_ml_imports():
    assert '"runtime_probe": "passed"' in run_python(["scripts/check-serving-runtime.py"])


def test_offline_builders_publish_all_positions_without_http_imports():
    source = r"""
import importlib.abc
import io
import json
import sys
import tempfile

class NoHTTP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.serving" or fullname.startswith("src.serving.") or fullname.split(".")[0] in {"flask", "werkzeug"}:
            raise ImportError("Offline builder imported HTTP: " + fullname)

sys.meta_path.insert(0, NoHTTP())
import pandas as pd
from src.prediction import historical, upcoming, build_snapshot, comparison_snapshot
from src.artifacts import serving_snapshot
from src.artifacts.snapshot_state import ServingState, use_state
from src.contracts.serialization import _VALID_SCORING, _ROW_PRED_PREFIXES, _actual_col

rows = []
for position in ("QB", "RB", "WR", "TE", "K", "DST"):
    for week in (1, 2):
        row = {"position": position, "week": week, "season": 2025, "player_id": position, "recent_team": "BUF", "player_display_name": position}
        for scoring in _VALID_SCORING:
            row[_actual_col(scoring)] = float(week + 10)
            for model in _ROW_PRED_PREFIXES:
                row[f"{model}_pred_{scoring}"] = float(week + 9)
        rows.append(row)
owner = ServingState()
owner.cache.update(results=pd.DataFrame(rows), prediction_inputs_fingerprint="fixed-inputs")
with tempfile.TemporaryDirectory() as directory, use_state(owner):
    historical._PREDICTIONS_CACHE_DIR = directory
    historical._compute_models_fingerprint = lambda: ("fixed-inputs", [])
    historical._any_position_sentinel_advanced = lambda: False
    historical._ensure_all_positions_loaded = lambda: None
    historical._ensure_metrics()
    generation, files = serving_snapshot.read_generation(directory)
    payload = json.loads(files["metrics.json"])
    for scoring in _VALID_SCORING:
        for metrics in payload["metrics_by_format"][scoring].values():
            assert metrics["overall"]["mae"] == 1.0
            assert {row["position"] for row in metrics["by_position"]} == {"QB", "RB", "WR", "TE", "K", "DST"}
    replay = pd.read_parquet(io.BytesIO(files["predictions.parquet"]))
    pd.testing.assert_frame_equal(replay, owner.cache["results"])
    assert owner.snapshots.current().generation == generation.name
assert not any(name == "src.serving" or name.startswith("src.serving.") or name.split(".")[0] in {"flask", "werkzeug"} for name in sys.modules)
print("offline publication passed")
"""
    assert "offline publication passed" in run_python(["-c", source])
