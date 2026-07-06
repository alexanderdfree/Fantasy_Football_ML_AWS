"""Week-1 / empty-history games-gap A/B (the TODO.md "[PRIORITY] Week-1
(empty-history) under-projection" calibration + its recorded "Deferred (b)
games-gap A/B" design).

Run it::

    python -m src.tuning.ab_games_gap --list                  # show the grid
    python -m src.tuning.ab_games_gap                         # QB RB WR TE, 3 seeds
    python -m src.tuning.ab_games_gap --positions K --only baseline --seeds 42 123
    python -m src.tuning.ab_games_gap --positions RB --only baseline rb_returning

Variant names are bare identifiers (no ``+`` prefix, unlike older specs):
ab-batch.yml validates each ``only`` entry against ``^[A-Za-z0-9_]+$``, and this
spec is designed to be dispatched with ``--only`` (K baseline-only measurement,
the 1-position smoke, the RB-only re-add arm).

Why: every model under-projects week 1 (QB measured: each model 0.8-1.6 FP more
negative than its rest-of-season bias) because every within-season feature is
zeroed at a season opener by the deliberate per-season window reset
(src/features/engineer.py — see the #1109/#1137 REVERT NOTE there: a blanket
cross-season carry is rejected; the fix must be a week-1-specific signal). The
arms here are *conditioning* signals, not carried form:

* ``baseline`` — production, unchanged. **Its ``week1_bias`` metrics double as
  the outstanding RB/WR/TE/K week-1 measurement** (TODO.md status: QB/DST
  measured, RB/WR/TE/K pending) — read them off this arm before judging the
  feature arms. K is baseline-only reachable (its ``run(seed, config)`` builds
  its own splits, so frame-injector arms raise by harness contract); DST is
  measured flat and skipped.
* ``career_gap`` — ``career_weeks_since_last_game``: cross-season weeks since
  the player's previous NFL game (52-week index ``season*52+week``, per-player
  ``diff()``, career debut = the 104 cap). The production per-season
  ``weeks_since_last_game`` is computed-then-dropped in engineer.py and reads
  week 1 as "played last week"; this scalar instead *tells* the models "this row
  follows an offseason/absence" so they can price the empty-history regime.
  Non-windowed scalar -> attention-static eligible (stop-rule forbids only
  windowed features).
* ``itt_empty`` — ``itt_x_no_history``: Vegas ``implied_team_total`` gated to
  rows with no in-season history (the TODO.md fix direction "weight Vegas
  implied totals more when in-season history is empty"), 0 elsewhere.
* ``rb_returning`` — RB-only re-add of the existing
  ``is_returning_from_absence`` column (dropped from RB's whitelist for
  r=0.934 with ``days_rest``; the TODO.md games-gap entry asks for it as a
  separate arm). Run it ONLY as ``--positions RB --only baseline rb_returning``:
  QB/WR/TE already whitelist the column, so the arm would no-op there — hence
  ``expect_ridge_identical=None`` (report-only) rather than ``False``.

Ship gates (TODO.md "Deferred (b)", judged per position on the harness report):
``week1_bias`` and ``returning_bias`` move toward 0 vs baseline on >=2/3 seeds
and >=3/4 models; QB ``passing_yards_mae`` non-regression on every model (the
#1137 guard — carried cross-season form misled QB passing yards at openers
once already); overall ``mae`` within the noise band; Ridge sentinel reads
"data changed" on every feature-arm cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42, 123, 7]

GAP_COL = "career_weeks_since_last_game"
ITT_COL = "itt_x_no_history"
GAP_CAP = 104.0  # two 52-week years; the career-debut sentinel == the cap
_WEEKS_INDEX = 52  # weeks are <=22, so season*52+week is monotone per player


# --------------------------------------------------------------------------- #
# Arm B — +career_gap (frame injection + whitelist)
# --------------------------------------------------------------------------- #
def _inject_career_gap(train, val, test):
    """Add ``career_weeks_since_last_game`` to all three splits.

    MUST be computed on the concatenated train+val+test: the splits are
    season-disjoint (train 2013-2023 / val 2024 / test 2025), so a per-frame
    ``diff()`` would make every val/test week-1 row read as a career debut.
    Leakage-safe: the gap uses only the *timing* of strictly-prior games (the
    schedule position is known before kickoff). The first data season's (2013)
    openers read as debuts — 2012 is context-only and absent from the splits —
    bounded by the cap, same "first season = unknown" stance as QB's rookie
    features. Alignment relies on each frame having a unique index (the
    harness loads splits fresh via ``pd.read_parquet`` -> RangeIndex).
    """
    frames = {"train": train, "val": val, "test": test}
    all3 = pd.concat(frames, names=["_ab_split", None])
    all3 = all3.sort_values(["player_id", "season", "week"])
    idx52 = all3["season"] * _WEEKS_INDEX + all3["week"]
    gap = idx52.groupby(all3["player_id"]).diff()  # NaN at each career-first row
    gap = gap.fillna(GAP_CAP).clip(1.0, GAP_CAP).astype("float64")
    for name, df in frames.items():
        df[GAP_COL] = gap.loc[name]  # Series assignment aligns on the original index
    return train, val, test


def _mut_career_gap(cfg):
    """Whitelist the gap scalar into BOTH model paths (non-windowed -> static-eligible)."""
    base = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda base=base: [*base(), GAP_COL]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], GAP_COL]
    return cfg


# --------------------------------------------------------------------------- #
# Arm C — +itt_empty (frame injection + whitelist)
# --------------------------------------------------------------------------- #
def _inject_itt_empty(train, val, test):
    """Add ``itt_x_no_history`` = ``implied_team_total`` on a player's first
    in-season row, 0 elsewhere.

    Safe per frame (each split holds whole seasons, so "first in-season row"
    never crosses a split boundary). The indicator counts strictly-prior
    in-season games (``cumcount()==0`` on the week-sorted frame) — the same
    pre-kickoff argument as QB's ``season_starts_to_date``. ``implied_team_total``
    is merged upstream by ``build_features`` with NaN preserved for unmatched
    games; mirror the pipeline's catch-all fill with ``fillna(0)``.
    """
    for df in (train, val, test):
        srt = df.sort_values(["player_id", "season", "week"])
        first = srt.groupby(["player_id", "season"]).cumcount().eq(0).reindex(df.index)
        itt = df["implied_team_total"].fillna(0.0)
        df[ITT_COL] = np.where(first.to_numpy(), itt.to_numpy(dtype=float), 0.0)
    return train, val, test


def _mut_itt_empty(cfg):
    base = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda base=base: [*base(), ITT_COL]
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], ITT_COL]
    return cfg


# --------------------------------------------------------------------------- #
# Arm D — +rb_returning (cfg-only re-add of an existing column; RB-only)
# --------------------------------------------------------------------------- #
def _mut_rb_returning(cfg):
    """Re-add ``is_returning_from_absence`` (add-if-absent, so a position that
    already whitelists it — QB/WR/TE — no-ops rather than double-listing)."""
    col = "is_returning_from_absence"
    base = cfg["get_feature_columns_fn"]

    def _cols(base=base):
        cols = list(base())
        return cols if col in cols else [*cols, col]

    cfg["get_feature_columns_fn"] = _cols
    if "attn_static_features" in cfg and col not in cfg["attn_static_features"]:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], col]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — week-1 / returning / empty-history cohorts + the #1137 guard
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    """Per-model overall + cohort slices off ``result["test_df"]``.

    ``mae`` stays the top-level key (it feeds the harness Ridge sentinel).
    ``week1_*`` vs ``rest_*`` is the measurement the TODO.md priority entry
    asks for; ``npg0_*`` is the adjacent empty-history (first in-season game)
    view; ``returning_*`` is the second gate cohort. ``passing_yards_mae`` (per
    model, from the eager per-target ``pred_{model}_passing_yards`` columns) is
    the #1137 non-regression guard — present on QB cells only.
    """
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)

    cuts: dict = {}
    if len(df) and "week" in df.columns:
        cuts["week1"] = df[df["week"] == 1]
        cuts["rest"] = df[df["week"] > 1]
        srt = df.sort_values(["player_id", "season", "week"])
        first = srt.groupby(["player_id", "season"]).cumcount().eq(0).reindex(df.index)
        cuts["npg0"] = df[first]
    if len(df) and "is_returning_from_absence" in df.columns:
        cuts["returning"] = df[df["is_returning_from_absence"] == 1]
    sub = {k: per_model_metrics(v, models) for k, v in cuts.items()}

    out: dict = {}
    for name, col in models.items():
        row = {"mae": float(overall[name]["mae"]), "bias": float(overall[name]["bias"])}
        for k, cut in cuts.items():
            row[f"{k}_bias"] = float(sub[k][name]["bias"])
            row[f"{k}_mae"] = float(sub[k][name]["mae"])
            row[f"{k}_n"] = float(sub[k][name]["n"])
            row[f"{k}_actual"] = float(cut["fantasy_points"].mean()) if len(cut) else float("nan")
        py_col = col.replace("_total", "_passing_yards")
        if len(df) and "passing_yards" in df.columns and py_col in df.columns:
            y = df["passing_yards"].to_numpy(dtype=float)
            p = df[py_col].to_numpy(dtype=float)
            row["passing_yards_mae"] = float(np.mean(np.abs(p - y)))
        out[name] = row
    return out


VARIANTS = [
    Variant("baseline", label="production (unchanged; week1_bias here = the measurement)"),
    Variant(
        "career_gap",
        cfg_mutator=_mut_career_gap,
        frame_injector=_inject_career_gap,
        expect_ridge_identical=False,  # a real whitelist feature MUST move Ridge
        label="+career_weeks_since_last_game (cross-season gap scalar)",
    ),
    Variant(
        "itt_empty",
        cfg_mutator=_mut_itt_empty,
        frame_injector=_inject_itt_empty,
        expect_ridge_identical=False,
        label="+implied_team_total x no-in-season-history interaction",
    ),
    Variant(
        "rb_returning",
        cfg_mutator=_mut_rb_returning,
        expect_ridge_identical=None,  # no-ops where already whitelisted; RB-only arm
        label="+is_returning_from_absence re-add (RB-only; run with --only)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
