"""E2 O-line continuity — RotoWire-slate rolling-origin confirm pass (expert-edge Phase 2b).

The E2 screen (src/tuning/ab_oline_continuity.py) was borderline-positive on 2025:
the ATTENTION head's deep-lineup regret improved consistently, landing on the
served ranker for TE (regret@12 −2.25, 3/3 seeds) and QB (−2.82, 2/3), MAE flat;
the served RB/WR rankers (LightGBM) did not benefit. This confirm pass tests
whether that attention-head regret gain (a) survives the RotoWire-COVERED slate
(a fair expert-comparison population, not the full slate the screen used) and
(b) replicates across ROLLING-ORIGIN test seasons 2022-2025 (each a fresh
deployment-mirror retrain: train [2013..T-2], val T-1, test T) rather than the
single 2025 season — the two limitations the screen's gate flagged.

Structure (ab_rolling_origin_rotowire.py pattern): origins × the E2 toggle are
ab-harness variants. Each variant's frame_injector re-slices the full featured
frame to origin T (rolling_origin_folds) and, for the `_e2` arms, additionally
injects the E2 continuity features (reusing ab_oline_continuity._inject_oline)
with the whitelist mutator (._mut_whitelist). The harness's artifact isolation
keeps the rolling-origin retrains from clobbering served {pos}/outputs. metric_fn
reports attn_nn / lgbm / rotowire optimal-lineup regret + hit@12 + Spearman on the
single shared RotoWire-covered slate (identical rows per arm → comparable regret
ceilings). Aggregate e2-vs-base per origin: E2 confirms if it lowers the attention
head's regret in ≥3/4 origins (toward RotoWire's ceiling).

RotoWire slate: pre-built data/raw/rotowire_slate_ppr_2018_2025.parquet (Sleeper
projections × the rosters sleeper→gsis bridge; on S3, synced into the container),
read in metric_fn — no in-container crosswalk fetch.

Fleet (rolling-origin RETRAINS need GPU; ADR-0020 branch flow):
    python -m src.tuning.launch_ab --spec src.tuning.ab_oline_confirm --dry-run
    # smoke ONE cell first:
    python -m src.tuning.launch_ab --spec src.tuning.ab_oline_confirm \
        --positions TE --only origin2025_e2 --seeds 42
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Launcher-safe module imports only (os/numpy/pandas/ab_harness + the launcher-safe
# ab_oline_continuity helpers). The heavy frame builders (_load_full_featured_frame,
# rolling_origin_folds, _inject_oline) are imported inside the injector closure /
# metric_fn, which run only in the Batch container. Pinned by test_ab_oline_confirm.
from src.tuning.ab_harness import Variant, ab_main
from src.tuning.ab_oline_continuity import _mut_whitelist

POSITIONS = ["QB", "RB", "TE"]  # where the E2 screen moved a head (WR flat, dropped)
SEEDS = [42]  # the 4 rolling origins provide the replication, not seeds
_ORIGINS = (2022, 2023, 2024, 2025)
_LINEUP_N = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
_MIN_TRAIN = 2013
_ROTOWIRE_SLATE = "rotowire_slate_ppr_2018_2025.parquet"  # under CACHE_DIR (data/raw), S3-synced


# --------------------------------------------------------------------------- #
# Composed frame injector: rolling-origin re-slice (+ optional E2 inject)
# --------------------------------------------------------------------------- #
def _make_injector(test_season: int, with_e2: bool):
    def _inject(train, val, test):
        from src.benchmarking.benchmark import _load_full_featured_frame
        from src.data.split import rolling_origin_folds

        full = _load_full_featured_frame()
        _, tr, va, te = rolling_origin_folds(
            full, test_seasons=[test_season], min_train_season=_MIN_TRAIN
        )[0]
        if with_e2:
            from src.tuning.ab_oline_continuity import _inject_oline

            tr, va, te = _inject_oline(tr, va, te)
        return tr, va, te

    return _inject


# --------------------------------------------------------------------------- #
# Metric — attn/lgbm/rotowire regret on the shared RotoWire-covered slate
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


def metric_fn(result: dict, position: str) -> dict[str, dict[str, float]]:
    from src.config import CACHE_DIR
    from src.shared.evaluation import compute_ranking_metrics

    df = result["test_df"].copy()
    df = df[df["fantasy_points"].notna()].copy()
    df["player_id"] = df["player_id"].astype(str)
    season = int(df["season"].iloc[0])
    n = _LINEUP_N[position]

    rw = pd.read_parquet(f"{CACHE_DIR}/{_ROTOWIRE_SLATE}")
    rw = rw[(rw["position"] == position) & (rw["season"] == season)][
        ["player_id", "season", "week", "rotowire_pred"]
    ].copy()
    rw["player_id"] = rw["player_id"].astype(str)
    df = df.merge(rw, on=["player_id", "season", "week"], how="left")
    slate = df[df["rotowire_pred"].notna()]  # shared slate: identical rows for every arm

    # Ridge overall MAE (sentinel feed; not constrained here — origins differ, so
    # every variant's expect_ridge_identical is None).
    out: dict[str, dict[str, float]] = {
        "Ridge": {"mae": float((df["pred_ridge_total"] - df["fantasy_points"]).abs().mean())}
    }
    cands = {
        "attn_nn": "pred_attn_nn_total",
        "lgbm": "pred_lgbm_total",
        "rotowire": "rotowire_pred",
    }
    for name, col in cands.items():
        rank = compute_ranking_metrics(slate, pred_col=col, top_k=12) if len(slate) else {}
        out[name] = {
            "regret": _regret(slate, col, n),
            "hit12": rank.get("season_avg_hit_rate", float("nan")),
            "spearman": rank.get("season_avg_spearman", float("nan")),
            "slate_n": float(len(slate)),
            "season": float(season),
        }
    return out


# --------------------------------------------------------------------------- #
# Variants: 4 origins × {base, e2}
# --------------------------------------------------------------------------- #
def _build_variants() -> list[Variant]:
    variants: list[Variant] = []
    for t in _ORIGINS:
        variants.append(
            Variant(
                f"origin{t}_base",
                frame_injector=_make_injector(t, with_e2=False),
                expect_ridge_identical=None,
                label=f"rolling origin {t}: baseline",
            )
        )
    for t in _ORIGINS:
        variants.append(
            Variant(
                f"origin{t}_e2",
                cfg_mutator=_mut_whitelist,
                frame_injector=_make_injector(t, with_e2=True),
                expect_ridge_identical=None,
                label=f"rolling origin {t}: + O-line continuity",
            )
        )
    return variants


VARIANTS = _build_variants()
BASELINE = "origin2025_base"  # production split; pairwise e2-vs-base done per origin in aggregation


if __name__ == "__main__":
    ab_main(__spec__.name)
