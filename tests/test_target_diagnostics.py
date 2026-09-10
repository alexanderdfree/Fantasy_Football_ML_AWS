"""Target diagnostics must compare the production scoring contract."""

import importlib

import pandas as pd
import pytest

from src.data.preprocessing import preprocess


@pytest.mark.unit
@pytest.mark.parametrize("position", ["qb", "rb", "wr", "te"])
@pytest.mark.parametrize(
    "conversion",
    ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"],
)
@pytest.mark.parametrize("corruption", [0.0, 2.0])
def test_conversion_rows_use_canonical_scoring(position, conversion, corruption, capsys):
    raw = pd.DataFrame(
        [
            {
                "player_id": "00-0000001",
                "position": position.upper(),
                "season_type": "REG",
                "snap_pct": 1.0,
                "passing_yards": 10.0,
                "passing_tds": 1.0,
                "rushing_yards": 5.0,
                "rushing_tds": 1.0,
                "receiving_yards": 20.0,
                "receiving_tds": 1.0,
                "receptions": 2.0,
                conversion: 1.0,
            }
        ]
    )
    frame = preprocess(raw)
    compute_targets = importlib.import_module(f"src.{position}.targets").compute_targets
    targets = importlib.import_module(f"src.{position}.config").POSITION_CONFIG.targets
    without_conversion = compute_targets(preprocess(raw.assign(**{conversion: 0.0})))
    capsys.readouterr()

    frame["fantasy_points"] += corruption
    result = compute_targets(frame)
    warning = "target decomposition discrepancy" in capsys.readouterr().out
    assert warning is (corruption != 0.0)
    pd.testing.assert_frame_equal(result[targets], without_conversion[targets])
