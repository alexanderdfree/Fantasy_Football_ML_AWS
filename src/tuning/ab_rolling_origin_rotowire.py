"""Multi-season confirmation of P0 (recalibration) + P1 (head selection) vs RotoWire.

Everything in P0/P1 was 2025-only. This re-runs the same analysis across **rolling-origin** test
seasons 2022-2025 (each a fresh deployment-mirror: train ``[2013..T-2]``, val ``T-1``, test ``T``,
NO leakage) and compares to **RotoWire** (clean 2018+; NFL.com pre-2024 is backfill-contaminated so
it's excluded here). Origins are modeled as ab-harness *variants* (a frame-injector re-slices the
full featured frame per origin), so the harness's **artifact isolation** prevents the rolling-origin
retrains from clobbering the served ``{pos}/outputs`` weights, and its GPU-sharing parallelizes the
16 (position × origin) cells.

Per origin × position, ``metric_fn`` reports — for {attn_nn, lgbm, recal(attn), recal(lgbm),
rotowire} — the **elite (top-30-by-actual) RMSE + signed bias** on the model∩rotowire intersection
(the recalibration question) and **optimal-lineup regret** on the shared RotoWire-covered slate —
identical for every arm so the regret ceilings are comparable (the head-selection question).
Recalibration = leave-one-week-out isotonic (``recalibration_eval.lowo_isotonic``).

Confirms P0 if: recal(attn)/recal(lgbm) drive elite bias → ~0 and cut elite RMSE at QB/TE in EVERY
season (not just 2025). Confirms P1 if: lgbm (or recal_lgbm) beats attn_nn on regret every season.

Run::

    python -m src.tuning.ab_rolling_origin_rotowire            # 4 pos × 4 origins, 1 seed, eager
    python -m src.tuning.ab_rolling_origin_rotowire --list
    python -m src.tuning.ab_rolling_origin_rotowire --only origin_2024 --positions QB
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.recalibration_eval import _bias, _rmse, lowo_isotonic
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42]
_ROTOWIRE = (
    Path(__file__).resolve().parents[2] / "scratchpad" / "multiseason" / "rotowire_all.parquet"
)
_LINEUP_N = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
_MIN_TRAIN = 2013


# --------------------------------------------------------------------------- #
# Origin frame injectors (re-slice the full featured frame; harness isolates artifacts)
# --------------------------------------------------------------------------- #
def _make_origin_injector(test_season: int):
    def _inject(train, val, test):
        from src.benchmarking.benchmark import _load_full_featured_frame
        from src.data.split import rolling_origin_folds

        full = _load_full_featured_frame()
        folds = rolling_origin_folds(full, test_seasons=[test_season], min_train_season=_MIN_TRAIN)
        _, tr, va, te = folds[0]
        return tr, va, te

    return _inject


# --------------------------------------------------------------------------- #
# Metric — per origin × position: recalibration (elite) + head selection (regret) vs RotoWire
# --------------------------------------------------------------------------- #
def _regret(df: pd.DataFrame, col: str, n: int) -> float:
    regrets = []
    for _, g in df.groupby("week"):
        gg = g[g[col].notna()]
        if len(gg) < n:
            continue
        opt = gg.nlargest(n, "fantasy_points")["fantasy_points"].sum()
        got = gg.nlargest(n, col)["fantasy_points"].sum()
        regrets.append(opt - got)
    return float(np.mean(regrets)) if regrets else float("nan")


def metric_fn(result, position):
    df = result["test_df"].copy()
    df = df[df["fantasy_points"].notna()].copy()
    df["player_id"] = df["player_id"].astype(str)
    season = int(df["season"].iloc[0])
    n = _LINEUP_N[position]

    # RotoWire join (pre-built; absolute path — metric_fn runs in the isolated cwd)
    rw = pd.read_parquet(_ROTOWIRE)
    rw = rw[(rw["position"] == position) & (rw["season"] == season)][
        ["player_id", "season", "week", "rotowire_pred"]
    ].copy()
    rw["player_id"] = rw["player_id"].astype(str)
    df = df.merge(rw, on=["player_id", "season", "week"], how="left")

    # recalibrated heads (LOWO isotonic on the full position slate)
    df["recal_attn"] = lowo_isotonic(df, "pred_attn_nn_total")
    df["recal_lgbm"] = lowo_isotonic(df, "pred_lgbm_total")

    cands = {
        "attn_nn": "pred_attn_nn_total",
        "lgbm": "pred_lgbm_total",
        "recal_attn": "recal_attn",
        "recal_lgbm": "recal_lgbm",
        "rotowire": "rotowire_pred",
    }

    # elite cohort = top-30 by actual season-total FP; recalibration judged on elite∩rotowire (paired)
    top30 = set(df.groupby("player_id")["fantasy_points"].sum().nlargest(30).index)
    elite = df[df["player_id"].isin(top30) & df["rotowire_pred"].notna()]
    y = elite["fantasy_points"].to_numpy()

    # Regret on a SINGLE shared slate (the RotoWire-covered rows) for EVERY candidate, so the
    # weekly optimum is identical across arms — comparing a full-slate model regret against a
    # subset-slate rotowire regret would be apples-to-oranges (different achievable ceilings).
    slate = df[df["rotowire_pred"].notna()]

    out: dict = {}
    for name, col in cands.items():
        row = {
            "elite_rmse": _rmse(elite[col], y) if len(elite) else float("nan"),
            "elite_bias": _bias(elite[col], y) if len(elite) else float("nan"),
            "regret": _regret(slate, col, n),
            "season": float(season),
            "n_elite": float(len(elite)),
        }
        out[name] = row
    return out


VARIANTS = [
    Variant("baseline", label="origin 2025 (= production split)"),
    Variant(
        "origin_2024",
        frame_injector=_make_origin_injector(2024),
        expect_ridge_identical=None,
        label="rolling origin: test 2024",
    ),
    Variant(
        "origin_2023",
        frame_injector=_make_origin_injector(2023),
        expect_ridge_identical=None,
        label="rolling origin: test 2023",
    ),
    Variant(
        "origin_2022",
        frame_injector=_make_origin_injector(2022),
        expect_ridge_identical=None,
        label="rolling origin: test 2022",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
