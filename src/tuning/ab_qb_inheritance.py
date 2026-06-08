"""A/B: QB role-inheritance — does flagging an inherited start help, and with which proxy?

QB is the spot-start blind spot from issue #1102: a backup who inherits the start (the QB1 ahead
is Out/Doubtful) is projected as a backup because nothing flags the current-week vacancy. The
attention HISTORY branch can't learn it on the *first* start — it pools the player's OWN past
games, and the teammate vacancy is in no past sequence — so the signal must enter the STATIC
branch (mirrors ab_history_token.py's RB/WR/TE finding; AGENTS.md stop-rule on attn_history_stats).

Arms (QB only; binary {Out,Doubtful} out-set):

* ``baseline``      — production config (inheritance column carried un-whitelisted, for slicing).
* ``+ffopp``        — ``is_top_available`` + ``inherited_opportunity`` whitelisted, role proxy =
                      ff_opportunity expected fantasy points (``total_fantasy_points_exp``). This is
                      the production design (engineer._INHERITANCE_ROLE_COL["QB"]).
* ``+attempts``     — same, but role proxy = pass ``attempts`` (the higher-signal-than-snap%
                      alternative considered before ff-opp; head-to-head proxy comparison).
* ``depth_shift``   — no new column: instead shift ``depth_chart_rank`` → 1 for the inheriting QB
                      (the "just shift the depth chart" alternative). Tests feature-vs-depth-shift.

Judge on the **inheritor subgroup** (``inherited_opportunity > 0``), NOT overall MAE — the feature
fires on ~3-4% of QB rows so overall MAE dilutes it to noise (project rule: subgroup error = bias).
The per-model cohort delta for **attn_nn** specifically answers "did attention already learn it"
(baseline already contains the full attention NN; if the feature still moves attn_nn, the sequence
wasn't capturing the vacancy). The role estimate is **prior-to-W only** (leakage-safe).

Run::

    python -m src.tuning.ab_qb_inheritance                 # QB, 3 seeds, autodetect -j
    python -m src.tuning.ab_qb_inheritance --list          # show the grid, run nothing
    python -m src.tuning.ab_qb_inheritance --only +ffopp --seeds 42   # fast smoke
    python -m src.tuning.ab_qb_inheritance --seeds 42 123 7 13 99 7   # bump seeds (small cohort)
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB"]
SEEDS = [42, 123, 7]

_POSITIONS = ("QB",)
# Role/opportunity proxy for the prior-to-W expanding-mean "who is the established starter" rank
# and the vacated-role magnitude. ff-opp expected FP is the production choice (snap% is ~1.0 for any
# starter → no magnitude; attempts is the runner-up). Scaled per-feature downstream.
_ROLE_FFOPP = "total_fantasy_points_exp"
_ROLE_ATTEMPTS = "attempts"
_STATIC = ["is_top_available", "inherited_opportunity"]  # → Ridge + LGBM + NN-static


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean QB role-inheritance columns (within QB)
# --------------------------------------------------------------------------- #
def _compute_inheritance(df, outmap, role_col):
    """Return (is_top_available, inherited_opportunity) arrays for ``df`` using ``role_col``.

    role(player, W) = prior-to-W expanding mean of ``role_col`` (weeks < W, in-season → no leak).
    ``is_top_available`` = top prior-role among *present* same-team QBs that week.
    ``inherited_opportunity`` = Σ prior-role of OUT/Doubtful same-team QBs ranked above, for the
    top-available QB only. Mirrors src.features.engineer._build_inheritance_features.
    """
    df["player_id"] = df["player_id"].astype(str)
    pref: dict = {}
    for pos in _POSITIONS:
        table: dict = {}
        sub_pos = df[df["position"] == pos].sort_values("week")
        for (p, s), sub in sub_pos.groupby(["player_id", "season"]):
            wks = sub["week"].to_numpy()
            vals = np.nan_to_num(sub[role_col].to_numpy(float), nan=0.0)
            table[(p, s)] = (wks, np.cumsum(vals) / np.arange(1, len(vals) + 1))
        pref[pos] = table

    def role_before(pos, p, s, w):
        e = pref[pos].get((p, s))
        if e is None:
            return 0.0
        wks, cm = e
        i = int(np.searchsorted(wks, w, side="left")) - 1
        return float(cm[i]) if i >= 0 else 0.0

    is_top = np.zeros(len(df))
    inh = np.zeros(len(df))
    for pos in _POSITIONS:
        grp = df[df["position"] == pos]
        for (s, tm, w), idx in grp.groupby(["season", "recent_team", "week"]).groups.items():
            si, wi = int(s), int(w)
            pids = df.loc[idx, "player_id"].to_numpy()
            roles = np.array([role_before(pos, p, si, wi) for p in pids])
            out_set = outmap.get((pos, si, tm, wi), set())
            out_roles = np.array([role_before(pos, g, si, wi) for g in out_set])
            for j, rp in enumerate(roles):
                top = 1.0 if (roles > rp).sum() == 0 else 0.0
                oa = float(out_roles[out_roles > rp].sum()) if out_roles.size else 0.0
                pi = df.index.get_loc(idx[j])
                is_top[pi] = top
                inh[pi] = top * oa
    return is_top, inh


def _build_outmap(train, val, test):
    """(position, season, team, week) -> {OUT/Doubtful gsis_ids} from the injury report."""
    from src.data import nfl_source

    seasons = sorted({int(s) for df in (train, val, test) for s in df["season"].unique()})
    inj = nfl_source.injuries(seasons)
    out = inj[(inj["report_status"].isin(["Out", "Doubtful"])) & (inj["position"].isin(_POSITIONS))]
    outmap: dict = {}
    for pos, s, t, w, g in zip(
        out["position"],
        out["season"].astype(int),
        out["team"],
        out["week"].astype(int),
        out["gsis_id"].astype(str),
        strict=True,
    ):
        outmap.setdefault((pos, s, t, w), set()).add(g)
    return outmap


def _make_injector(role_col, *, shift_depth=False):
    def _inject(train, val, test):
        outmap = _build_outmap(train, val, test)

        def _add(df):
            is_top, inh = _compute_inheritance(df, outmap, role_col)
            df["is_top_available"] = is_top
            df["inherited_opportunity"] = inh
            if shift_depth and "depth_chart_rank" in df.columns:
                # "Just shift the depth chart": the inheriting QB (top-available with a vacated
                # role) becomes the de-facto starter → rank 1. No new column whitelisted.
                promote = (inh > 0) & (is_top > 0)
                df.loc[promote, "depth_chart_rank"] = 1
            return df

        return _add(train), _add(val), _add(test)

    return _inject


# --------------------------------------------------------------------------- #
# Config mutator
# --------------------------------------------------------------------------- #
def _whitelist_static(cfg):
    get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [*get_cols(), *_STATIC]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *_STATIC]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — per-model overall + inheritor-cohort bias/MAE/n
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    sub = (
        df[df["inherited_opportunity"] > 0]
        if "inherited_opportunity" in df.columns
        else df.iloc[0:0]
    )
    sub_m = per_model_metrics(sub, models) if len(sub) else {}
    out: dict = {}
    for m, mv in overall.items():
        row = {"mae": float(mv["mae"]), "bias": float(mv["bias"])}
        if m in sub_m:
            row["inh_mae"] = float(sub_m[m]["mae"])
            row["inh_bias"] = float(sub_m[m]["bias"])
            row["inh_n"] = float(sub_m[m]["n"])
        out[m] = row
    return out


VARIANTS = [
    Variant(
        "baseline",
        frame_injector=_make_injector(_ROLE_FFOPP),  # carry inh col for slicing; model unchanged
        label="baseline (QB; inh col carried un-whitelisted for slicing)",
    ),
    Variant(
        "+ffopp",
        cfg_mutator=_whitelist_static,
        frame_injector=_make_injector(_ROLE_FFOPP),
        expect_ridge_identical=False,  # a real feature MUST move Ridge
        label="+inheritance static, ff-opp exp-FP proxy (production design)",
    ),
    Variant(
        "+attempts",
        cfg_mutator=_whitelist_static,
        frame_injector=_make_injector(_ROLE_ATTEMPTS),
        expect_ridge_identical=False,
        label="+inheritance static, pass-attempts proxy (proxy comparison)",
    ),
    Variant(
        "depth_shift",
        frame_injector=_make_injector(_ROLE_FFOPP, shift_depth=True),
        expect_ridge_identical=False,  # depth_chart_rank changed → Ridge moves
        label="depth_chart_rank→1 for the inheritor (no new column)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
