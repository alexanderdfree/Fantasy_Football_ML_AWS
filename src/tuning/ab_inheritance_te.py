"""A/B: does the static role-inheritance feature (validated on RB/WR, #1053) help TE?

Parity probe — TE is the left-behind skill position (RB/WR shipped inheritance via #1053;
``_INHERITANCE_POSITIONS=("RB","WR")`` excludes TE). A starter TE going Out/Doubtful frees
targets for the next TE up, so the same next-man-up signal *could* help TE. Validated here
BEFORE the expensive production merge (engineer.py → 6-position retrain + refresh-splits).

Only the **static** arm is tested: the ``attn_history`` inheritance token was tested-rejected
on RB (#1053, AGENTS.md stop-rule — a past spot-start is already encoded by the usage tokens),
so re-testing it on TE would re-propose a rejected mechanism. TE role proxy = **targets**
(a TE's value is targets, not snaps — same choice as WR).

* ``baseline``  — production config (the inheritance column is injected but NOT whitelisted,
                  so the model is unchanged and the inheritor slice is identical across arms).
* ``+static``   — ``is_top_available`` + ``inherited_opportunity`` whitelisted into
                  ``include_features`` (→ Ridge + LightGBM + NN-static branch).

Judge on the **inheritor subgroup** (``inherited_opportunity > 0``) bias, NOT overall MAE
(the feature fires on a handful of rows). **Watch ``inh_n``** — the TE inheritor cohort may be
small (TEs sit less often than RBs); if n is tiny, treat the result as inconclusive rather than
forcing a call. Leakage-safe: role(player,W) is the prior-to-W expanding mean (weeks < W only);
the Ridge sentinel checks the feature *took*, not that it's honest.

Run::

    python -m src.tuning.ab_inheritance_te                       # TE, 3 seeds
    python -m src.tuning.ab_inheritance_te --seeds 42 123 7 1 99 2024  # 6-seed confirm
    python -m src.tuning.ab_inheritance_te --list
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["TE"]
SEEDS = [42, 123, 7]

# Inheritance computed WITHIN TE (rank same-position teammates; the splits are all-position).
_POSITIONS = ("TE",)
# TE opportunity proxy = per-game targets (mirrors WR; a TE's value is targets, not snaps).
_ROLE_COL = {"TE": "targets"}
_STATIC = ["is_top_available", "inherited_opportunity"]  # → Ridge + LGBM + NN-static


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean role-inheritance columns (within TE)
# --------------------------------------------------------------------------- #
def _inject_inheritance(train, val, test):
    """Add ``is_top_available`` + ``inherited_opportunity`` per TE player-week.

    * role(player, W) = mean of TE targets over that player's weeks < W (prior-to-W, no leak).
    * ``is_top_available`` = top prior-role among *present* same-team TEs that week.
    * ``inherited_opportunity`` = Σ prior-role of same-team TEs Out/Doubtful ranked above,
      only for the top-available one (the next man up). Mirrors src/features/engineer.py
      ``_build_inheritance_features`` so the production build will be byte-identical on GO.
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
        strict=True,
    ):
        outmap.setdefault((pos, s, t, w), set()).add(g)

    def _add(df):
        df["player_id"] = df["player_id"].astype(str)
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
# Config mutator
# --------------------------------------------------------------------------- #
def _whitelist_static(cfg):
    get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [*get_cols(), *_STATIC]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *_STATIC]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — per-model overall + inheritor-subgroup bias
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    """Per-model overall MAE/bias PLUS inheritor-subgroup (``inherited_opportunity > 0``).

    Judge the targeted subgroup, not overall MAE (the feature fires on few rows). The column
    is injected into *every* arm (baseline carries it un-whitelisted), so the slice is identical
    across arms. ``inh_n`` flags whether the TE cohort is large enough to read.
    """
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
]


if __name__ == "__main__":
    ab_main(__spec__.name)
