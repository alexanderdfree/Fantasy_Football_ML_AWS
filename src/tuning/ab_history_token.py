"""A/B: does a per-game role-inheritance token in the attention HISTORY branch help?

Also the productionization-validation harness for the *static* inheritance feature on RB
and WR (run ``--only +static``). The injector computes inheritance within each of
``_POSITIONS`` so one spec covers both.

Three arms (RB is the validated role-inheritance position, todo/rotowire_gap_remediation.md):

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

Static-feature productionization (``--only +static``, 3 seeds, 2026-06-07). **Judge on the
inheritor subgroup (``inherited_opportunity > 0``), NOT overall MAE** — the feature fires on ~1%
of rows so overall MAE dilutes it to noise (RB Ridge overall −0.053 was the shadow of a −1.6 FP
subgroup effect). Earlier "WR not robust" was an artifact of reading overall MAE + a WR-inapplicable
RB ascension labeler (n=2). Proper subgroup: RB n=33, WR n=41.
* **RB — strong, robust win, ship it.** On RB inheritors (+static vs baseline): inh-MAE Ridge
  −1.61 / LGBM −0.71 / NN −0.72 / Attn −0.93; inh-bias (under-call→0) Ridge +3.16 / LGBM +2.17 /
  NN +2.62 / Attn +3.31. Baseline under-predicts inheritors ~4–6 FP; +static cuts it ~half across
  *every* model. Concentrated-workload position — when the lead back sits, one back absorbs the
  carries (clean signal). This is the boom-tier gap, invisible in overall MAE.
* **WR — modest but real on bias; flat on MAE.** On WR inheritors: inh-bias Ridge +0.80 / LGBM
  +0.24 / NN +1.22 / Attn +0.19 (under-call reduced across all four, cleanest on deterministic
  Ridge) but inh-MAE flat (Ridge −0.00 / LGBM +0.07 / NN −0.08 / Attn +0.14) and overall MAE costs
  a trivial +0.006. Distributed position — a WR1 absence redistributes targets diffusely, so the
  feature shifts the center up correctly (bias) but can't pin the noisy individual outcomes (MAE).
  ~4× weaker than RB; ship-WR is a judgment call, not a slam dunk.
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["RB", "WR"]
SEEDS = [42, 123, 7]

# Inheritance is computed WITHIN each of these positions (rank teammates of the same
# position; the splits are all-position). A cell reads only its own position's column,
# so one injector serves both the RB and WR cells. WR was added for the productionization
# validation (the history-token science was RB-only; run WR with `--only +static`).
_POSITIONS = ("RB", "WR")
# Position-appropriate opportunity proxy for the "role" estimate (prior-to-W expanding mean).
# RB: snap-share (snaps ≈ carries ≈ opportunity — validated). WR: per-game targets (a WR's
# value is *targets*, not snaps; a slot WR is high-snap/low-target, so snap-based inheritance
# was mis-specified for WR — it only helped the Attention NN and regressed Ridge). Scaled
# per-feature by the pipeline, so the snap-fraction vs target-count unit mismatch is fine.
_ROLE_COL = {"RB": "snap_pct_raw", "WR": "targets"}
_STATIC = ["is_top_available", "inherited_opportunity"]  # → Ridge + LGBM + NN-static
_HISTORY = ["inherited_opportunity"]  # → NN history sequence (per-game spot-start token)


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean role-inheritance columns (within-position)
# --------------------------------------------------------------------------- #
def _inject_inheritance(train, val, test):
    """Add ``is_top_available`` + ``inherited_opportunity`` per player-week, within position.

    * role(player, W) = mean of the position's opportunity proxy (``_ROLE_COL``: RB snap-share,
      WR per-game targets) over that player's weeks < W (in-season) — prior-to-W, no leak.
    * ``is_top_available`` = this player has the top prior-role among *present* same-position
      teammates that week.
    * ``inherited_opportunity`` = Σ prior-role of same-team, same-position OUT/Doubtful
      players ranked above, but only for the top-available one (the next-man-up who absorbs
      the role).

    Computed WITHIN each position in ``_POSITIONS`` (the splits are all-position; a
    cross-position group makes a back never "top" — a QB at snap~1.0 outranks every RB).
    The injector has no view of the cell's position, so it computes the column for every
    position in ``_POSITIONS`` and each cell reads only its own rows. Same column serves
    both branches: its week-W value is the static feature; its per-game sequence is the
    history token.
    """
    from src.data import nfl_source

    seasons = sorted({int(s) for df in (train, val, test) for s in df["season"].unique()})
    inj = nfl_source.injuries(seasons)
    out = inj[(inj["report_status"].isin(["Out", "Doubtful"])) & (inj["position"].isin(_POSITIONS))]
    outmap: dict = {}  # (position, season, team, week) -> {out player ids}
    for pos, s, t, w, g in zip(
        out["position"],
        out["season"].astype(int),
        out["team"],
        out["week"].astype(int),
        out["gsis_id"].astype(str),
        strict=True,  # all five are columns of `out` → equal-length by construction
    ):
        outmap.setdefault((pos, s, t, w), set()).add(g)

    def _add(df):
        df["player_id"] = df["player_id"].astype(str)
        # Per-position prior-to-W expanding-mean role table, each from its own opportunity
        # proxy (_ROLE_COL). Keyed by position so an OUT player's role reads the right column.
        pref: dict = {}  # position -> {(player, season): (weeks_sorted, cumulative-mean)}
        for pos in _POSITIONS:
            col = _ROLE_COL[pos]
            table: dict = {}
            sub_pos = df[df["position"] == pos].sort_values("week")
            for (p, s), sub in sub_pos.groupby(["player_id", "season"]):
                wks = sub["week"].to_numpy()
                vals = np.nan_to_num(sub[col].to_numpy(float), nan=0.0)
                table[(p, s)] = (wks, np.cumsum(vals) / np.arange(1, len(vals) + 1))
            pref[pos] = table

        def role_before(pos, p, s, w):
            e = pref[pos].get((p, s))
            if e is None:
                return 0.0
            wks, cm = e
            i = int(np.searchsorted(wks, w, side="left")) - 1  # largest week < w
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
    """Per-model overall MAE/bias PLUS the inheritor-subgroup MAE/bias.

    The feature fires on few rows (the next-man-up when a higher same-position teammate is
    Out), so its effect is invisible in overall MAE — judge it on the **targeted subgroup**
    instead (project rule: subgroup error = bias; MAE-delta is fine on a *fixed* ablation
    slice). The subgroup is ``inherited_opportunity > 0`` — feature-active rows — which is
    position-general (works for WR, unlike the RB-specific ascension labeler that matched
    ~0 WR rows). The column is injected into *every* arm (baseline carries it un-whitelisted,
    so the model is unchanged but the slice is identical across arms).
    """
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    if "inherited_opportunity" in df.columns:
        sub = df[df["inherited_opportunity"] > 0]
    else:
        sub = df.iloc[0:0]
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
        # Inject the column (for the inheritor-subgroup slice) but do NOT whitelist it — the
        # baseline model is unchanged (Ridge MAE must match production), the slice is identical
        # to the +static arm's.
        frame_injector=_inject_inheritance,
        label="baseline (model unchanged; inh col carried for slicing)",
    ),
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
