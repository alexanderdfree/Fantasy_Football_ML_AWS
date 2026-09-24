"""No-fit contracts for the eager expectation comparison observer."""

import pandas as pd
import pytest

from src.shared.comparison_scoring import scoring_components
from src.tuning import ab_inheritance_reception as spec

pytestmark = pytest.mark.unit


def _certified_frame(total):
    frame = pd.DataFrame({target: [0.0] for target in scoring_components("WR")})
    frame["season"] = 2022
    frame["week"] = 1
    frame["actual_projected_total"] = total
    frame.attrs["actual_projected_total_metadata"] = {
        "basis": "configured_target_aggregation_v1",
        "targets": list(scoring_components("WR")),
        "scoring_format": "ppr",
    }
    return frame


def test_missing_certified_actuals_fail_before_scoring_zero_filled_targets(monkeypatch):
    monkeypatch.setattr(spec, "default_metric_fn", lambda *_: pytest.fail("scored missing truth"))
    with pytest.raises(ValueError, match="Shared actual components unavailable"):
        spec.metric_fn({"test_df": _certified_frame(float("nan"))}, "WR")


def test_certified_actuals_reach_metrics_instead_of_zero_filled_targets(monkeypatch):
    class Observed(Exception):
        pass

    def capture(result, position):
        assert position == "WR"
        assert result["test_df"].fantasy_points.tolist() == [12.5]
        raise Observed

    monkeypatch.setattr(spec, "default_metric_fn", capture)
    with pytest.raises(Observed):
        spec.metric_fn({"test_df": _certified_frame(12.5)}, "WR")


def test_cuda_auto_mode_keeps_artifact_observer_eager(monkeypatch, tmp_path):
    from src.tuning import ab_harness

    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: True)
    calls = []
    monkeypatch.setattr(ab_harness, "run_sequential", lambda *args: calls.append("eager") or [])
    monkeypatch.setattr(ab_harness, "run_sequential_stacked", lambda *args: pytest.fail("stacked"))
    ab_harness.run_ab(spec, positions=["WR"], seeds=[42], data_dir=str(tmp_path), jobs=1)
    assert calls == ["eager"]
    with pytest.raises(ValueError, match="does not support stacked"):
        ab_harness.run_ab(
            spec,
            positions=["WR"],
            seeds=[42],
            data_dir=str(tmp_path),
            jobs=1,
            stacked_seeds=True,
        )
