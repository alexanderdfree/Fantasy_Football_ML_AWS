"""A/B: opponent-defense attention branch ON vs OFF (QB/RB/WR/TE).

Settles "did opp-def actually help" cleanly. The opp-defense branch (PRs #123
QB/WR/TE, #214 RB) adds a second ``AttentionPool`` over the opposing defense's
per-game history, replacing the six static ``opp_def_*_L5`` aggregates on the
attention NN's static path. It is **attention-NN-only by construction**: Ridge /
LightGBM / NN-static never read ``opp_attn_history_stats`` (the activation
trigger, ``src/shared/pipeline.py`` — non-empty list ⇒ the pipeline builds opp
tensors ⇒ ``MultiHeadNetWithHistory(opp_game_dim=…)``; empty ⇒ single-branch
model, byte-identical to pre-#123). So every arm is ``expect_ridge_identical``.

Why this is the right harness: the committed benchmark history can't isolate
opp-def — #123 merged inside a ~22-PR train (incl. dropping ``fantasy_points``
as a feature) and the only EC2 before/after that *is* clean (RB #214,
26becd2→6020a18) showed attn-NN −0.164 MAE but on a single seed, and LightGBM
(the best RB model) was unmoved. This A/B runs the **stacked N=24 seed-ensemble**
(#1165) on the **production GPU metric path** (L4/sm_89, FP16, graphs) so the
delta is mean±std over 24 seeds, not one. Stacked mode trains Ridge/LGBM/NN once
(Phase A, attention off — identical across arms ⇒ the Ridge data-identity
sentinel holds) and vmap-trains all 24 attention seeds (Phase B), so the measured
Δ is purely the opp branch. Those three non-attention models therefore report
Δ=0.000 (std 0) — a built-in check that opp-def is attention-only.

Scope: the four opp-**defense** positions. K has no opponent-history branch
(``opp_attn_max_seq_len=None``); DST's branch is opp-**offense**
(``opp_attn_kind="offense"``) — both are out of scope for an "opp-def" question.

Metric: per-model overall MAE/bias **and** the Q4 boom tier (top fantasy-point
quartile, on actuals ⇒ identical across arms) — opp-def is an attention change,
so judge it on the tail where the attention NN is the served model for WR/TE and
the top-30 expert gap concentrates (#1053; the top-30 rebaseline), not just
overall MAE which dilutes a tail effect to noise.

Run (local ``--list`` only — real training SIGSEGVs on the macOS libomp triple)::

    python -m src.tuning.ab_opp_def --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_opp_def \
        --stacked-seeds --max-cells 200          # GPU Batch fleet, 4-position fan-out
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_ensemble_seeds import stacked_default_seed_list
from src.tuning.ab_harness import Variant, ab_main

# The four opp-DEFENSE positions (all flat-history ⇒ all stack). K has no opp
# branch; DST is opp-offense — both excluded from an opp-def A/B.
POSITIONS = ["QB", "RB", "WR", "TE"]
# The canonical 24-seed stacked grid (range(42, 66)). Baked into the spec because
# the Batch entry (ab_batch.py) reads spec.SEEDS — it does not apply the local
# stacked-default fallback, so the wide grid must live here.
SEEDS = stacked_default_seed_list()


def _drop_opp_def(cfg):
    """Disable the opponent-defense attention branch.

    Empty ``opp_attn_history_stats`` ⇒ the pipeline builds no opp tensors ⇒
    ``opp_game_dim=None`` ⇒ single-branch NN (the static path keeps the
    ``opp_def_*_L5`` aggregates it had pre-#123, since this drops only the
    *attention* branch's stats, not the static whitelist). History-branch-only:
    Ridge / LightGBM / NN-static are untouched. The harness deep-copies cfg per
    cell, so the in-place write is safe.
    """
    cfg["opp_attn_history_stats"] = []
    return cfg


def metric_fn(result, position):
    """Per-model overall MAE/bias + Q4 boom-tier MAE/bias on ``result["test_df"]``.

    ``q4`` = top fantasy-point quartile (cut on actuals ⇒ identical across arms;
    the baseline needs no injection). Derives solely from ``test_df`` per the
    stacked-mode contract (other result keys carry Phase-A values). Overall
    ``mae`` feeds the harness Ridge-invariance sentinel and the report's primary
    column; the Q4 keys land in the per-cell JSON / summary for the tail read.
    """
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)

    q4_m: dict = {}
    if len(df):
        q75 = float(np.quantile(df["fantasy_points"].to_numpy(dtype=float), 0.75))
        q4 = df[df["fantasy_points"] >= q75]
        if len(q4):
            q4_m = per_model_metrics(q4, models)

    out: dict = {}
    for name in models:
        row = {"mae": float(overall[name]["mae"]), "bias": float(overall[name]["bias"])}
        if name in q4_m:
            row["q4_mae"] = float(q4_m[name]["mae"])
            row["q4_bias"] = float(q4_m[name]["bias"])
        out[name] = row
    return out


VARIANTS = [
    Variant("baseline", label="opp-def ON (production opp_attn_history_stats)"),
    Variant(
        "-opp_def",
        cfg_mutator=_drop_opp_def,
        expect_ridge_identical=True,  # attention-branch-only ⇒ Ridge/LGBM/NN-static unaffected
        label="opp-def OFF (opp_attn_history_stats=[], single-branch NN)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
