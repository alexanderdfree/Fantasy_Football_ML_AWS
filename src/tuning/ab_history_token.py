"""A/B: does a per-game role-inheritance token in the attention HISTORY branch help?

Three arms on RB (the validated role-inheritance position, todo/rotowire_gap_remediation.md):

* ``baseline``         — production config.
* ``+static``          — ``is_top_available`` + ``inherited_opportunity`` whitelisted into
                         ``include_features`` (→ Ridge + LightGBM + NN static branch).
* ``+static+history``  — same, PLUS ``inherited_opportunity`` as a per-game token in
                         ``attn_history_stats`` (NN history branch). This is the user's
                         hypothesis: the attention NN should learn a player's past
                         *spot-start* performance, not just the current-week vacancy.

The history token's marginal value is ``(+static+history) − (+static)`` on **Attention NN**
(Ridge/LightGBM don't read ``attn_history_stats``, so they should be identical between the two
``+`` arms — a built-in check). Metric: per-model overall MAE/bias plus **ascension-cohort
bias** (the documented RB blind spot; cohort_analysis).

Run::

    python -m src.tuning.ab_history_token              # RB, 3 seeds, autodetect -j
    python -m src.tuning.ab_history_token --list       # show the grid, run nothing
    python -m src.tuning.ab_history_token --seeds 42 -j 2   # fast smoke

The role estimate is **prior-to-W only** (the season-mean leak inflated the first manual cut
~60%); the Ridge sentinel checks the static feature *took*, not that it's honest — leakage-safety
is owned here.

Result (RB, 3 seeds, 2026-06-07) — **the history token does NOT help; static-only wins.**
On the Attention NN, ``+static+history`` is −0.32 FP / ~3σ *worse* on the ascension-cohort bias
than ``+static`` (−10.72 vs −10.40; baseline −11.53) and the overall-MAE flicker (−0.017) is
inside the seed band. Why: a past spot-start is *already* encoded in the existing history tokens
(``snap_pct_raw``, ``game_carry_share``, carries, production) — the attention pool can already
learn it — so a derived ``inherited_opportunity`` token (an expanding-mean of ``snap_pct_raw``)
re-encodes existing signal *and* averages an already-averaged quantity. The branch gains nothing
and the redundant token slightly hurts. The static arm is the win because the *current-week*
vacancy is genuinely new (it's in no past-game sequence). Built-in invariant confirmed:
LGBM/NN/Ridge byte-identical between the two ``+`` arms (the token is Attention-NN-only). Don't
re-propose role/inheritance in ``attn_history_stats`` — see AGENTS.md stop-rules.
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["RB"]
SEEDS = [42, 123, 7]

_POS = "RB"
_SNAP = "snap_pct_raw"
_STATIC = ["is_top_available", "inherited_opportunity"]  # → Ridge + LGBM + NN-static
_HISTORY = ["inherited_opportunity"]  # → NN history sequence (per-game spot-start token)


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean role-inheritance columns (within-position)
# --------------------------------------------------------------------------- #
def _inject_inheritance(train, val, test):
    """Add ``is_top_available`` + ``inherited_opportunity`` per player-week.

    * role(player, W) = mean snap over that player's weeks < W (in-season) — prior-to-W,
      so no week-W outcome leaks in.
    * ``is_top_available`` = this RB has the top prior-role among *present* RBs that week.
    * ``inherited_opportunity`` = Σ prior-role of same-team OUT/Doubtful RBs ranked above,
      but only for the top-available RB (the next-man-up who actually absorbs the role).

    Computed WITHIN position (the splits are all-position; a cross-position group makes an
    RB never "top" — a QB at snap~1.0 outranks every RB). Same column serves both branches:
    its week-W value is the static feature; its per-game sequence is the history token.
    """
    from src.data import nfl_source

    seasons = sorted({int(s) for df in (train, val, test) for s in df["season"].unique()})
    inj = nfl_source.injuries(seasons)
    out = inj[(inj["report_status"].isin(["Out", "Doubtful"])) & (inj["position"] == _POS)]
    outmap: dict = {}
    for s, t, w, g in zip(
        out["season"].astype(int),
        out["team"],
        out["week"].astype(int),
        out["gsis_id"].astype(str),
        strict=True,  # all four are columns of `out` → equal-length by construction
    ):
        outmap.setdefault((s, t, w), set()).add(g)

    def _add(df):
        df["player_id"] = df["player_id"].astype(str)
        pref: dict = {}  # (player, season) -> (weeks_sorted, cumulative-mean-incl-i)
        for (p, s), sub in df.sort_values("week").groupby(["player_id", "season"]):
            wks = sub["week"].to_numpy()
            snaps = sub[_SNAP].to_numpy(float)
            pref[(p, s)] = (wks, np.cumsum(snaps) / np.arange(1, len(snaps) + 1))

        def role_before(p, s, w):
            e = pref.get((p, s))
            if e is None:
                return 0.0
            wks, cm = e
            i = int(np.searchsorted(wks, w, side="left")) - 1  # largest week < w
            return float(cm[i]) if i >= 0 else 0.0

        is_top = np.zeros(len(df))
        inh = np.zeros(len(df))
        rb = df[df["position"] == _POS]
        for (s, tm, w), idx in rb.groupby(["season", "recent_team", "week"]).groups.items():
            si, wi = int(s), int(w)
            pids = df.loc[idx, "player_id"].to_numpy()
            roles = np.array([role_before(p, si, wi) for p in pids])
            out_roles = np.array([role_before(g, si, wi) for g in outmap.get((si, tm, wi), set())])
            for j, rp in enumerate(roles):
                top = 1.0 if (roles > rp).sum() == 0 else 0.0
                oa = float(out_roles[out_roles > rp].sum()) if out_roles.size else 0.0
                pi = df.index.get_loc(idx[j])
                is_top[pi] = top
                inh[pi] = top * oa
        df["is_top_available"] = is_top
        df["inherited_opportunity"] = inh
        return df

    return _add(train), _add(val), _add(test)


# --------------------------------------------------------------------------- #
# Config mutators
# --------------------------------------------------------------------------- #
def _whitelist_static(cfg):
    get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [*get_cols(), *_STATIC]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *_STATIC]
    return cfg


def _whitelist_static_and_history(cfg):
    cfg = _whitelist_static(cfg)
    if "attn_history_stats" in cfg:  # NN-only; Ridge/LGBM unaffected
        cfg["attn_history_stats"] = [*cfg["attn_history_stats"], *_HISTORY]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — per-model overall + ascension-cohort bias
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    from src.analysis.cohort_analysis import (
        available_models,
        label_ascension_rows,
        per_model_metrics,
    )

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    try:
        lab = label_ascension_rows(df)
        asc = df[lab.to_numpy() == "ascension"]
    except Exception:
        asc = df.iloc[0:0]
    asc_m = per_model_metrics(asc, models) if len(asc) else {}
    out: dict = {}
    for m, mv in overall.items():
        row = {"mae": float(mv["mae"]), "bias": float(mv["bias"])}
        if m in asc_m:
            row["asc_bias"] = float(asc_m[m]["bias"])
            row["asc_n"] = float(asc_m[m]["n"])
        out[m] = row
    return out


VARIANTS = [
    Variant("baseline", label="baseline (production config)"),
    Variant(
        "+static",
        cfg_mutator=_whitelist_static,
        frame_injector=_inject_inheritance,
        expect_ridge_identical=False,  # a real feature MUST move Ridge
        label="+inheritance static (Ridge+LGBM+NN-static)",
    ),
    Variant(
        "+static+history",
        cfg_mutator=_whitelist_static_and_history,
        frame_injector=_inject_inheritance,
        expect_ridge_identical=False,  # static feature still in feature_cols → Ridge moves
        label="+inheritance static + attn-history token (NN-only delta)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
