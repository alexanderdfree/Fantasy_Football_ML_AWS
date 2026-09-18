"""Unit pins for the E2 O-line RotoWire-slate confirm spec (src/tuning/ab_oline_confirm.py).

No training / no rolling-origin retrain here (those run on the fleet): spec
resolution + variant grid, the metric_fn's RotoWire-slate regret on a synthetic
test_df + slate, and the module-level-import launcher-safety guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.tuning.ab_harness as H
import src.tuning.ab_oline_confirm as spec

pytestmark = pytest.mark.unit


def test_spec_resolves_with_origin_toggle_grid():
    resolved = H.resolve_spec("src.tuning.ab_oline_confirm")
    assert resolved.positions == ["QB", "RB", "TE"]
    assert resolved.baseline == "origin2025_base"
    expected = {f"origin{t}_{s}" for t in (2022, 2023, 2024, 2025) for s in ("base", "e2")}
    assert set(resolved.variants) == expected
    # base arms: origin re-slice only (no whitelist). e2 arms: both.
    assert resolved.variants["origin2024_base"].cfg_mutator is None
    assert resolved.variants["origin2024_base"].frame_injector is not None
    assert resolved.variants["origin2024_e2"].cfg_mutator is not None
    assert resolved.variants["origin2024_e2"].frame_injector is not None
    for (
        v
    ) in resolved.variants.values():  # rolling-origin retrains: Ridge not comparable across origins
        assert v.expect_ridge_identical is None


def test_variant_names_are_fleet_only_safe():
    import re

    for v in spec.VARIANTS:
        assert re.fullmatch(r"[A-Za-z0-9_]+", v.name)


def test_module_level_imports_are_launcher_safe():
    """launch_ab imports this spec on the deps-light runner to size the grid; no
    module-level import may pull src.data / nflreadpy (the heavy frame builders
    are deferred into the injector closure / metric_fn). Mirrors the #1479 guard.
    Importing the launcher-safe ab_oline_continuity is fine (it carries no
    module-level src.data import)."""
    import ast
    import pathlib

    src = pathlib.Path(spec.__file__).read_text()
    banned = ("src.data", "nflreadpy")
    for node in ast.parse(src).body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert not any(name == b or name.startswith(b + ".") for b in banned), (
                f"module-level import {name!r} is launcher-hostile; defer it"
            )


def _synthetic_result_and_slate(tmp_path, monkeypatch):
    """A test_df (2025) + a RotoWire slate parquet covering a subset of players."""
    rng = np.random.default_rng(0)
    rows = []
    for w in (1, 2):
        for i in range(20):
            actual = float(5 + i + rng.normal(0, 0.1))
            rows.append(
                {
                    "player_id": f"p{i}",
                    "season": 2025,
                    "week": w,
                    "fantasy_points": actual,
                    "pred_ridge_total": actual + 1.0,
                    "pred_attn_nn_total": actual + 0.3,
                    "pred_lgbm_total": actual - 0.3,
                }
            )
    df = pd.DataFrame(rows)
    # RotoWire covers only players p0..p14 (a curated subset, like the real feed).
    slate_rows = []
    for w in (1, 2):
        for i in range(15):
            slate_rows.append(
                {
                    "player_id": f"p{i}",
                    "season": 2025,
                    "week": w,
                    "position": "TE",
                    "rotowire_pred": float(5 + i),
                }
            )
    slate = pd.DataFrame(slate_rows)
    slate.to_parquet(tmp_path / spec._ROTOWIRE_SLATE)
    monkeypatch.setattr("src.config.CACHE_DIR", str(tmp_path), raising=False)
    return {"test_df": df}


def test_metric_fn_scores_on_rotowire_covered_slate(tmp_path, monkeypatch):
    result = _synthetic_result_and_slate(tmp_path, monkeypatch)
    out = spec.metric_fn(result, "TE")
    assert "Ridge" in out and "attn_nn" in out and "lgbm" in out and "rotowire" in out
    # Slate restricted to the 15 covered players × 2 weeks = 30 rows.
    assert out["attn_nn"]["slate_n"] == 30.0
    assert out["rotowire"]["slate_n"] == 30.0
    for name in ("attn_nn", "lgbm", "rotowire"):
        assert {"regret", "hit12", "spearman"} <= set(out[name])
    # Monotone-offset preds rank the slate perfectly -> zero regret at n<=slate width.
    assert out["attn_nn"]["regret"] == pytest.approx(0.0, abs=1e-9)


def test_lineup_sizes_match_positions():
    assert spec._LINEUP_N["TE"] == 12 and spec._LINEUP_N["RB"] == 24 and spec._LINEUP_N["QB"] == 12
