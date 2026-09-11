"""Cached NumPy scoring remains usable without loading execution/plotting libraries."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_numpy_scoring_before_torch_import_preserves_later_tensor_dispatch():
    script = """
import sys
import numpy as np
import pandas as pd
from src.shared.comparison_scoring import score_actual_components, scoring_components
from src.shared.evaluation_cohorts import regular_season_rows
from src.shared.aggregate_targets import _tier_bonuses
assert not {'torch', 'sklearn', 'matplotlib'} & sys.modules.keys()
frame = pd.DataFrame({name: [0.0] for name in scoring_components('DST')})
assert score_actual_components(frame, 'DST').iloc[0] == 5.0
assert len(regular_season_rows(frame)) == 1
import torch
value = _tier_bonuses(torch.tensor([0.0, 35.0], dtype=torch.float64), [1, 7, 14, 21, 28, 35], [10, 7, 4, 1, 0, -1, -4])
assert isinstance(value, torch.Tensor)
assert value.dtype == torch.float64 and value.tolist() == [10.0, -4.0]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
