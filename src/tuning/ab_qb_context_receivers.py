"""A/B: cross-position QB context for pass-catchers — can WR/TE/RB see their QB is out?

The blind spot (todo/expert-gap-other-reasons-2026-08.md §B5): the shipped role-inheritance
features are same-position-only by construction (``engineer._build_inheritance_features``
ranks same-team, SAME-POSITION teammates), so a WR/TE/RB model carries zero signal that its
QB is Out this week — while experts propagate a QB change onto the whole receiving corps
instantly. This screen broadcasts the QB situation to the team's pass-catchers, reusing the
prior-role machinery the QB spot-start work validated (#1102, ab_qb_inheritance.py).

Three team-week columns, computed from the QB rows of the SAME general (position-unfiltered)
splits the harness injects into, merged onto every row by ``(season, recent_team, week)``:

* ``team_qb_out``          — 1.0 when an Out/Doubtful QB out-ranks (by prior role) every QB
                             who plays that week: the expected starter is missing.
* ``team_qb_vacated_role`` — Σ prior-role of those missing higher-ranked QBs (magnitude of
                             what was lost; the receiver-side view of the starter's value).
* ``team_expected_qb_role``— prior-role of the best QB actually present (quality of who IS
                             throwing). Also covers chronic absences the weekly report can't
                             flag (an IR'd starter drops off the report; the replacement's
                             lower role persists) and benchings (the benched QB has no row).

Arms (WR/TE/RB; QB rows drive the computation but the QB pipeline is never run):

* ``baseline``     — injector runs, nothing whitelisted (columns carried for cohort slicing).
* ``+qb_out``      — whitelist the two event columns (``team_qb_out`` + vacated magnitude).
* ``+qb_quality``  — whitelist ``team_expected_qb_role`` alone (continuous QB-quality level).
* ``+both``        — all three.

Judge on the **qbout cohort** (``team_qb_out > 0`` — the receiver's team lost its expected
starter; fires on roughly a few % of rows), NOT overall MAE (project rule: subgroup error =
bias). Overall RMSE is reported too — the surviving expert edge at RB/WR is an RMSE/ordering
edge, and a QB-out week is a classic bust the model currently can't see coming.

Leakage & serving notes: the role estimate is the **prior-to-W in-season expanding mean** of
ff-opportunity expected FP (weeks < W only — the production-validated proxy); the out-set is
the pre-kickoff injury report (Out/Doubtful). "Present" = has a stat row that week, the same
convention the production inheritance features use (pre-kickoff rosters at serving, #1106).
Week-1 roles are 0 for everyone (in-season-only screen simplification); production wiring
would reuse engineer's prior-season fallback (#1106 finding B). These are current-week TEAM
state — non-temporal, static-branch-eligible like ``inherited_opportunity`` — and must NEVER
enter ``attn_history_stats`` (stop-rule: the history branch is the player's own past games).

Run::

    python -m src.tuning.ab_qb_context_receivers                   # WR/TE/RB, 3 seeds
    python -m src.tuning.ab_qb_context_receivers --list            # show the grid, run nothing
    python -m src.tuning.ab_qb_context_receivers --positions WR --only +qb_out --seeds 42  # smoke
    python -m src.tuning.launch_ab --spec src.tuning.ab_qb_context_receivers  # Batch fleet
                                                                   # (ADR-0020; dispatch
                                                                   # batch-image.yml on this
                                                                   # branch first)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR", "TE", "RB"]
SEEDS = [42, 123, 7]

_QB = "QB"
# Prior-role proxy: ff-opportunity expected fantasy points — the proxy the QB inheritance A/B
# selected over snap% (~1.0 for any starter, no magnitude) and pass attempts (#1102).
_ROLE_COL = "total_fantasy_points_exp"
_EVENT = ["team_qb_out", "team_qb_vacated_role"]
_QUALITY = ["team_expected_qb_role"]
_ALL_COLS = _EVENT + _QUALITY


# --------------------------------------------------------------------------- #
# QB prior-role table + team-week context (mirrors ab_qb_inheritance's roles)
# --------------------------------------------------------------------------- #
def _qb_role_table(df: pd.DataFrame) -> dict:
    """(player_id, season) -> (weeks, expanding mean of the role proxy INCLUDING that week).

    Consumers index the entry strictly BEFORE the query week (searchsorted left − 1), so the
    value used for week W is the mean over weeks < W — leakage-safe.
    """
    table: dict = {}
    qb = df[df["position"] == _QB].sort_values("week")
    for (p, s), sub in qb.groupby(["player_id", "season"]):
        wks = sub["week"].to_numpy()
        vals = np.nan_to_num(sub[_ROLE_COL].to_numpy(float), nan=0.0)
        table[(str(p), int(s))] = (wks, np.cumsum(vals) / np.arange(1, len(vals) + 1))
    return table


def _role_before(table: dict, p: str, s: int, w: int) -> float:
    e = table.get((p, s))
    if e is None:
        return 0.0
    wks, cm = e
    i = int(np.searchsorted(wks, w, side="left")) - 1
    return float(cm[i]) if i >= 0 else 0.0


def _team_qb_context(df: pd.DataFrame, outmap: dict) -> pd.DataFrame:
    """One row per (season, recent_team, week) with the three QB-context columns.

    ``team_expected_qb_role`` = max prior-role among QBs present (playing) that week;
    ``team_qb_vacated_role`` = Σ prior-role of Out/Doubtful QBs ranked ABOVE that max
    (an out backup vacates nothing); ``team_qb_out`` = 1.0 iff that sum is positive.
    """
    table = _qb_role_table(df)
    qb = df[df["position"] == _QB]
    recs = []
    for (s, tm, w), idx in qb.groupby(["season", "recent_team", "week"]).groups.items():
        si, wi = int(s), int(w)
        pids = qb.loc[idx, "player_id"].astype(str)
        roles = np.array([_role_before(table, p, si, wi) for p in pids])
        top_role = float(roles.max()) if roles.size else 0.0
        out_ids = outmap.get((si, tm, wi), set())
        out_roles = np.array([_role_before(table, g, si, wi) for g in out_ids])
        vacated = float(out_roles[out_roles > top_role].sum()) if out_roles.size else 0.0
        recs.append((si, tm, wi, 1.0 if vacated > 0 else 0.0, vacated, top_role))
    return pd.DataFrame(recs, columns=["season", "recent_team", "week", *_ALL_COLS])


def _build_outmap(train, val, test) -> dict:
    """(season, team, week) -> {gsis_ids of Out/Doubtful QBs} from the injury report."""
    from src.data import nfl_source

    seasons = sorted({int(s) for df in (train, val, test) for s in df["season"].unique()})
    inj = nfl_source.injuries(seasons)
    out = inj[(inj["report_status"].isin(["Out", "Doubtful"])) & (inj["position"] == _QB)]
    outmap: dict = {}
    for s, t, w, g in zip(
        out["season"].astype(int),
        out["team"],
        out["week"].astype(int),
        out["gsis_id"].astype(str),
        strict=True,
    ):
        outmap.setdefault((s, t, w), set()).add(g)
    return outmap


def _inject_qb_context(train, val, test):
    outmap = _build_outmap(train, val, test)

    def _add(df):
        # Per-frame computation is exact: roles are in-season expanding means and the
        # splits are season-disjoint, so per-frame ≡ global (no cross-frame concat needed —
        # contrast the cross-season games-gap A/B, which DOES need the concat).
        df = df.copy()
        df["player_id"] = df["player_id"].astype(str)
        ctx = _team_qb_context(df, outmap)
        df = df.merge(ctx, on=["season", "recent_team", "week"], how="left")
        for c in _ALL_COLS:
            df[c] = df[c].fillna(0.0)  # bye/no-QB-row team-weeks → neutral
        return df

    return _add(train), _add(val), _add(test)


# --------------------------------------------------------------------------- #
# Config mutators — static branch only (current-week team state, non-temporal)
# --------------------------------------------------------------------------- #
def _make_whitelister(cols):
    def _mut(cfg):
        get_cols = cfg["get_feature_columns_fn"]
        cfg["get_feature_columns_fn"] = lambda: [*get_cols(), *cols]
        if "attn_static_features" in cfg:
            cfg["attn_static_features"] = [*cfg["attn_static_features"], *cols]
        return cfg

    return _mut


# --------------------------------------------------------------------------- #
# Metric — per-model overall MAE/bias/RMSE + qbout-cohort bias/MAE/n
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    sub = df[df["team_qb_out"] > 0] if "team_qb_out" in df.columns else df.iloc[0:0]
    sub_m = per_model_metrics(sub, models) if len(sub) else {}
    out: dict = {}
    for m, mv in overall.items():
        row = {
            "mae": float(mv["mae"]),
            "bias": float(mv["bias"]),
            "rmse": float(mv["rmse"]),
        }
        if m in sub_m:
            row["qbout_mae"] = float(sub_m[m]["mae"])
            row["qbout_bias"] = float(sub_m[m]["bias"])
            row["qbout_n"] = float(sub_m[m]["n"])
        out[m] = row
    return out


VARIANTS = [
    Variant(
        "baseline",
        frame_injector=_inject_qb_context,  # columns carried un-whitelisted for slicing
        label="baseline (QB-context cols carried un-whitelisted for the qbout cohort)",
    ),
    Variant(
        "+qb_out",
        cfg_mutator=_make_whitelister(_EVENT),
        frame_injector=_inject_qb_context,
        expect_ridge_identical=False,  # a real feature MUST move Ridge
        label="+team_qb_out + team_qb_vacated_role (event: expected starter missing)",
    ),
    Variant(
        "+qb_quality",
        cfg_mutator=_make_whitelister(_QUALITY),
        frame_injector=_inject_qb_context,
        expect_ridge_identical=False,
        label="+team_expected_qb_role (continuous quality of the QB actually playing)",
    ),
    Variant(
        "+both",
        cfg_mutator=_make_whitelister(_ALL_COLS),
        frame_injector=_inject_qb_context,
        expect_ridge_identical=False,
        label="+event + quality (all three QB-context columns)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
