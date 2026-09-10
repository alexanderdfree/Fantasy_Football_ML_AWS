import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit

SPECS = [
    "ab_example",
    "ab_attn_arch",
    "ab_scheduler_type",
    "ab_opp_def",
    "ab_feature_screen",
    "ab_feature_screen_extended",
    "ab_feature_screen_k",
    "ab_feature_screen_dst",
    "ab_feature_subscreen",
    "ab_feature_confirm",
    "ab_knob_doe",
    "ab_air_yards",
    "ab_history_token",
    "ab_inheritance_te",
    "ab_qb_inheritance",
    "ab_boom_signals_wr",
    "ab_boom_signals_te",
    "ab_opp_coverage_wr",
    "ab_rolling_origin_rotowire",
    "ab_verify_cuda_capture",
]


@pytest.mark.parametrize("name", SPECS)
def test_spec_metadata_needs_only_workflow_launch_dependencies(name):
    code = """
import importlib.abc, sys
class InstalledLaunchDeps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {
            'optuna', 'torch', 'sklearn', 'lightgbm', 'scipy', 'matplotlib',
            'joblib', 'nflreadpy', 'polars', 'pyarrow', 'shap', 'mord',
        }:
            raise ModuleNotFoundError(f'{fullname} is not installed by ab-batch.yml')
sys.meta_path.insert(0, InstalledLaunchDeps())
from src.tuning.launch_ab import resolve_spec
spec = resolve_spec(sys.argv[1])
assert spec.variants and spec.positions and spec.seeds
"""
    result = subprocess.run(
        [sys.executable, "-c", code, f"src.tuning.{name}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        env={**os.environ, "FF_CONFIRM_DROP_COLS": "trend_targets", "OPENBLAS_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr
