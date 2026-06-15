"""Unit tests for the air-yards history-token A/B spec (src/tuning/ab_air_yards.py).

No training here (that's the GPU-fleet harness run). Coverage: spec resolution + design
shape (history-only → ``expect_ridge_identical=True`` on every ``+`` arm), the position-
inferring cfg mutators (receiving vs passing tokens, the RB skip-if-present, QB mobility,
dedup + base-order preservation, missing-key safety), and the boom-subgroup metric for both
a receiving (``receiving_tds``) and a passing (``passing_tds``) frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tuning import ab_air_yards as A
from src.tuning import ab_harness as H

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# spec resolution + design shape
# --------------------------------------------------------------------------- #
def test_air_yards_spec_resolves_as_dotted():
    """Imports + resolves: WR-default, four arms, every ``+`` arm is a cfg-mutator-only
    history change (no frame injector) declaring ``expect_ridge_identical=True`` (history
    feeds the Attn NN only — Ridge must stay byte-identical); ``baseline`` is identity."""
    spec = H.resolve_spec("src.tuning.ab_air_yards")
    assert spec.dotted == "src.tuning.ab_air_yards"
    assert spec.positions == ["WR"]
    assert spec.baseline == "baseline"
    assert set(spec.variants) == {"baseline", "+air", "+air_yac", "+air_yac_fd"}
    assert spec.variants["baseline"].is_baseline_shape  # identity → subgroup slice is native
    for name in ("+air", "+air_yac", "+air_yac_fd"):
        v = spec.variants[name]
        assert v.cfg_mutator is not None
        assert v.frame_injector is None  # columns already on the frame → no injection
        assert v.expect_ridge_identical is True


# --------------------------------------------------------------------------- #
# config mutators — position inference + skip-if-present + dedup
# --------------------------------------------------------------------------- #
def _recv_cfg(extra=()) -> dict:
    """A receiving-position history (marker token ``receiving_yards``)."""
    return {"attn_history_stats": ["receiving_yards", "targets", "receptions", *extra]}


def _qb_cfg() -> dict:
    """A QB history (marker token ``passing_yards``; no receiving_yards token)."""
    return {"attn_history_stats": ["passing_yards", "attempts", "completions", "carries"]}


def test_recv_air_yac_fd_progression():
    base = _recv_cfg()["attn_history_stats"]
    assert A._mut_air(_recv_cfg())["attn_history_stats"] == [*base, "receiving_air_yards"]
    assert A._mut_air_yac(_recv_cfg())["attn_history_stats"] == [
        *base,
        "receiving_air_yards",
        "receiving_yards_after_catch",
    ]
    assert A._mut_air_yac_fd(_recv_cfg())["attn_history_stats"] == [
        *base,
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_first_downs",
    ]


def test_rb_skips_already_present_first_downs():
    """RB already carries ``receiving_first_downs`` — the fd arm must NOT re-add it."""
    cfg = A._mut_air_yac_fd(_recv_cfg(extra=["receiving_first_downs"]))
    hist = cfg["attn_history_stats"]
    assert hist.count("receiving_first_downs") == 1  # not duplicated
    added = [t for t in hist if t not in ("receiving_yards", "targets", "receptions")]
    assert added == ["receiving_first_downs", "receiving_air_yards", "receiving_yards_after_catch"]


def test_qb_infers_passing_tokens_and_mobility():
    assert A._mut_air(_qb_cfg())["attn_history_stats"][-1] == "passing_air_yards"
    fd = A._mut_air_yac_fd(_qb_cfg())["attn_history_stats"]
    # QB gets the passing decomposition + the rushing-first-downs mobility token; no recv tokens.
    assert fd[-4:] == [
        "passing_air_yards",
        "passing_yards_after_catch",
        "passing_first_downs",
        "rushing_first_downs",
    ]
    assert not any(t.startswith("receiving_") for t in fd)


def test_mutators_preserve_base_order_and_dedup():
    cfg = A._mut_air_yac_fd(_recv_cfg())
    hist = cfg["attn_history_stats"]
    assert hist[:3] == ["receiving_yards", "targets", "receptions"]  # base order intact
    assert len(hist) == len(set(hist))  # no duplicate tokens
    # Idempotent: applying again adds nothing.
    assert A._mut_air_yac_fd(dict(cfg))["attn_history_stats"] == hist


def test_mutator_missing_history_key_is_safe():
    out = A._mut_air({"get_feature_columns_fn": lambda: ["x"]})
    assert "attn_history_stats" not in out  # no key → unchanged, no crash


# --------------------------------------------------------------------------- #
# metric — boom subgroup (receiving + passing frames)
# --------------------------------------------------------------------------- #
def _boom_df(td_col: str) -> pd.DataFrame:
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 22.0, 30.0])  # Q4 (>=q75) = top quartile
    return pd.DataFrame(
        {
            "fantasy_points": y,
            td_col: [0, 0, 0, 0, 0, 1, 2, 2],
            "pred_ridge_total": y + 1.0,  # uniform +1 over-prediction
            "pred_lgbm_total": y + 1.0,
        }
    )


def test_metric_fn_receiving_boom_subgroup():
    df = _boom_df("receiving_tds")
    out = A.metric_fn({"test_df": df}, "WR")
    assert out["Ridge"]["mae"] == pytest.approx(1.0)  # overall MAE → Ridge sentinel
    y = df["fantasy_points"].to_numpy()
    assert out["Ridge"]["q4_n"] == pytest.approx((y >= np.quantile(y, 0.75)).sum())
    assert out["Ridge"]["q4_bias"] == pytest.approx(1.0)
    assert out["Ridge"]["q4_corr"] == pytest.approx(1.0)  # pred = y + 1 → perfectly correlated
    assert out["LightGBM"]["tdgame_n"] == pytest.approx(3.0)  # receiving_tds >= 1


def test_metric_fn_passing_frame_uses_passing_tds():
    df = _boom_df("passing_tds")
    out = A.metric_fn({"test_df": df}, "QB")
    # td-game cut adapts to the passing column when receiving_tds is absent.
    assert out["Ridge"]["tdgame_n"] == pytest.approx(3.0)
