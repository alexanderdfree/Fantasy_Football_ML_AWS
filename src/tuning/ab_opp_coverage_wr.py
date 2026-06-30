"""A/B (P3): does an opponent pass-COVERAGE-quality feature close the WR boom gap?

The expert-gap benchmark (scratchpad/expert_gap_report.md) localized the WR loss to an
**apex-studs ordering** gap whose error is **unpredictable from any pre-kickoff feature we
currently have** (E1: held-out R² ≤ 0). A feature-presence audit confirmed WR is blind to
coverage *quality*: it carries only team-level pass-defense **volume** (``opp_def_pass_yds_allowed_L5``)
and position-DvP (``opp_recv_pts_allowed_to_pos``) — outcome volume, not *how open receivers get*.
``src/wr/config.py:319-328`` documents this exact gap (#1210) and ``condq`` WR is a forward bet
waiting for it.

This adds the canonical coverage-openness signal, **genuinely net-new** (NGS is ingested nowhere):

* ``opp_sep_allowed_L5``     — rolling avg receiver **separation** the upcoming opponent's defense
  has allowed (lower = tighter coverage), NGS ``avg_separation``.
* ``opp_cushion_allowed_L5`` — rolling avg **cushion** allowed, NGS ``avg_cushion``.

Both are **matchup** features (a property of the upcoming opponent, like the existing
``opp_def_*`` block) → whitelist only (``get_feature_columns_fn`` → Ridge + LightGBM, our best WR
model). Leakage-safe: per-defense ``shift(1).rolling(5)`` so week W sees only the defense's prior
games (mirrors ``engineer._build_defense_matchup_features``). Source NGS starts 2016; pre-2016 WR
rows get NaN → the pipeline's train-mean fill. NGS uses standard ``team_abbr`` (100% join to the WR
split, validated in scratchpad/coverage/).

Metric: the WR **boom subgroup** (Q4 / receiving-TD games), per model — judged on bias / RMSE /
**correlation** (the decomposition's closable edge), NOT overall MAE (the #1053 dilution trap).
Headline = LightGBM boom-corr↑ / boom-bias→0 with overall MAE flat, across 3 seeds.

Run::

    python -m src.tuning.ab_opp_coverage_wr            # WR, 3 seeds, autodetect -j
    python -m src.tuning.ab_opp_coverage_wr --list     # show the grid, run nothing
    python -m src.tuning.ab_opp_coverage_wr --only +coverage -j 2   # smoke
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR"]
SEEDS = [42, 123, 7]

_OPP_ROLL = 5  # mirror src/config.OPP_ROLLING_WINDOW
_NGS_MIN = 2016
_COVERAGE_COLS = ["opp_sep_allowed_L5", "opp_cushion_allowed_L5"]
_KEY = ["opponent_team", "season", "week"]


# --------------------------------------------------------------------------- #
# Net-new opponent-coverage table (NGS separation/cushion the defense allowed)
# --------------------------------------------------------------------------- #
def _to_pd(df):
    try:
        return df.to_pandas()
    except Exception:
        return df


def _build_coverage_table(seasons: list[int]) -> pd.DataFrame:
    """(opponent_team, season, week) -> leakage-safe rolling coverage-allowed columns.

    NGS gives the OFFENSE's per-game receiver separation/cushion; a defense's *allowed*
    value is what its week-W opponent's receivers achieved, mapped via the schedule. Then a
    per-defense ``shift(1).rolling(5)`` makes week W depend only on the defense's prior games.
    """
    import nflreadpy as nf

    os.environ.setdefault("NFLREADPY_CACHE", "filesystem")
    yrs = [int(s) for s in seasons]
    ngs_yrs = [s for s in yrs if s >= _NGS_MIN]
    if not ngs_yrs:
        return pd.DataFrame(columns=["opponent_team", "season", "week", *_COVERAGE_COLS])

    sch = _to_pd(nf.load_schedules(seasons=yrs))
    sch = sch[sch["game_type"] == "REG"][["season", "week", "home_team", "away_team"]]
    team_opp = pd.concat(
        [
            sch.rename(columns={"home_team": "team", "away_team": "opponent"}),
            sch.rename(columns={"away_team": "team", "home_team": "opponent"}),
        ],
        ignore_index=True,
    )[["season", "week", "team", "opponent"]]

    ngs = _to_pd(nf.load_nextgen_stats(seasons=ngs_yrs, stat_type="receiving"))
    ngs = ngs[(ngs["week"] > 0) & (ngs["season_type"] == "REG")].copy()
    ngs["tgt"] = ngs["targets"].fillna(0).astype(float)

    def _wmean(d, col):
        w = d["tgt"]
        return np.average(d[col], weights=w) if w.sum() > 0 else d[col].mean()

    g = ngs.groupby(["team_abbr", "season", "week"])
    off = (
        pd.DataFrame(
            {
                "sep": g.apply(lambda d: _wmean(d, "avg_separation")),
                "cushion": g.apply(lambda d: _wmean(d, "avg_cushion")),
            }
        )
        .reset_index()
        .rename(columns={"team_abbr": "team"})
    )

    # offense separation == what its opponent (the defense) allowed that game
    allowed = off.merge(team_opp, on=["team", "season", "week"], how="inner").rename(
        columns={"opponent": "defense_team"}
    )[["defense_team", "season", "week", "sep", "cushion"]]

    # full (defense, season, week) grid so the per-season rolling sees every game the defense
    # played that season
    grid = team_opp.rename(columns={"team": "defense_team"})[
        ["defense_team", "season", "week"]
    ].drop_duplicates()
    base = grid.merge(allowed, on=["defense_team", "season", "week"], how="left").sort_values(
        ["defense_team", "season", "week"]
    )
    # Roll WITHIN season (groupby includes season) so a season opener gets a fresh value, not the
    # prior season's tail — matching production engineer._build_defense_matchup_features (groupby
    # ["opponent_team","season"]); cross-season carry regressed openers (#1137).
    for _col, _out in (("sep", "opp_sep_allowed_L5"), ("cushion", "opp_cushion_allowed_L5")):
        base[_out] = base.groupby(["defense_team", "season"])[_col].transform(
            lambda s: s.shift(1).rolling(_OPP_ROLL, min_periods=1).mean()
        )
    return base.rename(columns={"defense_team": "opponent_team"})[
        ["opponent_team", "season", "week", *_COVERAGE_COLS]
    ]


# --------------------------------------------------------------------------- #
# Frame injector — merge the opponent-coverage columns onto WR rows
# --------------------------------------------------------------------------- #
def _inject_coverage(train, val, test):
    seasons = sorted(set(pd.concat([f[["season"]] for f in (train, val, test)])["season"].unique()))
    tbl = _build_coverage_table(seasons)
    tbl["opponent_team"] = tbl["opponent_team"].astype(str)

    def _add(df):
        df = df.copy()
        for c in _COVERAGE_COLS:
            df[c] = 0.0
        is_wr = df["position"] == "WR"
        wr = df[is_wr].copy()
        wr["opponent_team"] = wr["opponent_team"].astype(str)
        merged = wr.merge(tbl, on=_KEY, how="left", suffixes=("", "_cov"))
        for c in _COVERAGE_COLS:
            vals = merged[f"{c}_cov"].to_numpy()
            # Leave NaN where coverage is absent (pre-2016 / unmatched) so the pipeline's fill_nans
            # imputes the train mean — 0.0 is an out-of-distribution separation/cushion value (yds),
            # and era-correlated 0.0s would leak a spurious signal. Non-WR rows keep 0.0 (filtered
            # out downstream, never read).
            df.loc[wr.index, c] = vals
        return df

    return _add(train), _add(val), _add(test)


def _mut_coverage(cfg):
    """Whitelist the coverage columns (→ Ridge + LightGBM). Matchup feature, not the static
    rolling stop-rule: it is the opponent's signal, not the player's own windowed history."""
    base = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda base=base: [*base(), *_COVERAGE_COLS]
    return cfg


def _mut_coverage_static(cfg):
    """EXPLORATORY arm: whitelist AND route the coverage cols into the attention static branch so
    the attention NN + condq (the documented #1210 landing spot) actually sees them.

    NOTE this DELIBERATELY puts rolling (_L5) features on the static branch, which AGENTS.md's
    stop-rule forbids for production ("Never add rolling / ewma / trend / L3 / L5 / L8 ... to
    ATTN_STATIC_FEATURES"). The existing matchup features that feed attn_static are NON-rolling
    (the rolling ``opp_def_*_L5`` block is excluded from DEFAULT_ATTN_STATIC_CATEGORIES,
    src/shared/position_config.py). This arm exists only to MEASURE whether condq can exploit a
    coverage signal at all; a positive result would NOT license promoting these L5 cols into
    production ATTN_STATIC_FEATURES (that would recreate the windowed double-feed the design
    forbids)."""
    _mut_coverage(cfg)
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *_COVERAGE_COLS]
    return cfg


# --------------------------------------------------------------------------- #
# Metric — WR boom subgroup (mirrors ab_boom_signals_wr)
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)

    cuts: dict = {}
    if len(df):
        q75 = float(np.quantile(df["fantasy_points"].to_numpy(dtype=float), 0.75))
        cuts["q4"] = df[df["fantasy_points"] >= q75]
        if "receiving_tds" in df.columns:
            cuts["rztd"] = df[df["receiving_tds"] >= 1]
    sub_m = {k: per_model_metrics(v, models) for k, v in cuts.items()}

    def _corr(sub, col):
        if len(sub) < 2:
            return float("nan")
        a, b = sub[col].to_numpy(dtype=float), sub["fantasy_points"].to_numpy(dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out: dict = {}
    for name, col in models.items():
        row = {"mae": float(overall[name]["mae"]), "bias": float(overall[name]["bias"])}
        for k, sub in cuts.items():
            row[f"{k}_bias"] = float(sub_m[k][name]["bias"])
            row[f"{k}_rmse"] = float(sub_m[k][name]["rmse"])
            row[f"{k}_corr"] = _corr(sub, col)
            row[f"{k}_n"] = float(sub_m[k][name]["n"])
        out[name] = row
    return out


VARIANTS = [
    Variant("baseline", label="WR production (unchanged)"),
    Variant(
        "+coverage",
        cfg_mutator=_mut_coverage,
        frame_injector=_inject_coverage,
        expect_ridge_identical=False,  # a real whitelist feature MUST move Ridge
        label="+opponent NGS separation/cushion allowed (L5, whitelist)",
    ),
    Variant(
        "+coverage_static",
        cfg_mutator=_mut_coverage_static,
        frame_injector=_inject_coverage,
        expect_ridge_identical=False,
        label="+coverage in whitelist AND attn static branch (condq #1210 test)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
