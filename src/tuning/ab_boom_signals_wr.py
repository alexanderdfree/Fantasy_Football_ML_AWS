"""A/B: close the WR boom-tier gap vs experts with red-zone receiving + opportunity parity.

Experts (RotoWire) beat our best model (LightGBM) on RMSE/R² for WR. The cause is
settled (``src/analysis/rmse_gap_decomposition.py``): the gap is the **Q4 boom
quartile**, it is **TD-dominated** (×6 leverage), ~78% is the shared near-irreducible
boom under-call, and the *only* closable edge is **correlation = more boom signal**.
A signal closes the gap only if it feeds the **whitelist** (``get_feature_columns_fn``
→ Ridge/LightGBM/NN-static — LightGBM is the best model), is **leakage-safe**, and
tracks the TD boom tier. History-branch-only adds help only the attention NN (not the
best model) — upside, not gap-closers (the #1053 inheritance-token lesson).

A two-agent feature-landscape scan (2026-06-08) found **WR is blind to red-zone
receiving in every branch** (RB carries it everywhere) and **thin on priors / per-game
usage vs RB**. The user-selected bundle for this round:

* ``+rz``     — red-zone receiving.   Whitelist: ``redzone_targets_L3`` /
  ``redzone_target_share_L3`` (rolling, ``shift=1``) + ``prior_season_total_redzone_touches``
  / ``..._per_game`` (already in the splits for all positions). History (NN, raw per-game):
  ``redzone_targets`` / ``redzone_target_share`` (already in the splits).
* ``+parity`` — bring WR toward RB parity. Whitelist: ``prior_season_games_played``
  (in splits) + ``prior_season_mean_catch_rate`` (derived S-1) + ``opportunity_index_L3``
  (rolling). History (NN, raw per-game): ``game_target_share`` / ``game_target_hhi`` /
  ``game_opportunity_index``.   The per-game ``game_opportunity_index`` history token is
  the user's explicit addition; the rolling ``opportunity_index_L3`` is the LGBM-relevant
  RB-parity counterpart.
* ``+all``    — ``+rz`` ∪ ``+parity``.

Branch placement (per-model attribution disentangles it: LGBM/Ridge Δ = the *whitelist*
effect = the gap-relevant signal; attn Δ = whitelist + history):

    column                                  whitelist(→Ridge/LGBM)  static(NN)  history(NN)
    redzone_targets_L3 / _share_L3          ✓ (rolling)             —           —
    prior_season_total_redzone_touches/pg   ✓ (prior)               ✓           —
    redzone_targets / redzone_target_share  —                       —           ✓ (raw)
    prior_season_games_played / catch_rate  ✓ (prior)               ✓           —
    opportunity_index_L3                    ✓ (rolling)             —           —
    game_target_share / _hhi / _opp_index   —                       —           ✓ (raw)

**NOT a re-run of the rejected inheritance history token** (#1053, AGENTS.md stop-rule).
That token was a *windowed/expanding-mean role* signal already encoded by the existing
usage tokens — "averaging an average." These are **raw per-game red-zone / usage signals
genuinely absent from WR's sequence** (AGENTS.md history reach #2: "red-zone splits,
share-style measures … a raw per-game signal genuinely absent from the sequence"), and
RB already carries every one of them (red-zone via PR #235; game shares/HHI in
``src/rb/features.py``). The static-branch stop-rule is respected: rolling features go to
``get_feature_columns_fn`` only (never ``attn_static_features``); only the non-windowed
prior-season aggregates touch the static branch.

Metric: judge on the **WR boom subgroup**, NOT overall MAE (the #1053 trap — a feature
that fires on ~1% of rows dilutes to noise overall). Subgroups are defined on *actuals*
(identical across arms, so the baseline needs no injector): **Q4** = top fantasy-point
quartile, and **rztd** = receiving-TD games. Per model we report subgroup bias, RMSE and
**correlation** (the closable edge) + overall MAE (held-flat sanity). Headline = LightGBM
boom-bias→0 / boom-corr↑ with overall MAE flat, across 3 seeds. Leakage-safety is owned
here (the Ridge sentinel checks the feature *took*, not that it is honest): whitelist =
rolling(``shift=1``)/prior-season only; raw per-game values go to history, which
``build_game_history_arrays`` lags.

Run::

    python -m src.tuning.ab_boom_signals_wr               # WR, 3 seeds, autodetect -j
    python -m src.tuning.ab_boom_signals_wr --list        # show the grid, run nothing
    python -m src.tuning.ab_boom_signals_wr --only +rz -j 2  # fast smoke
"""

from __future__ import annotations

import numpy as np

from src.shared.feature_build import rolling_agg, safe_divide
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR"]
SEEDS = [42, 123, 7]

_GRP = ["player_id", "season"]

# Already baked into data/splits for every position (engineer.build_features) — whitelist only.
_RZ_PRIOR = ["prior_season_total_redzone_touches", "prior_season_mean_redzone_touches_per_game"]
# Built by the injector (rolling, leakage-safe shift=1) — whitelist (rolling → NOT static).
_RZ_ROLL = ["redzone_targets_L3", "redzone_target_share_L3"]
# Raw per-game, already in splits — attention HISTORY only (build_game_history_arrays lags them).
_RZ_HIST = ["redzone_targets", "redzone_target_share"]

# Parity whitelist: prior-season (static-eligible, non-windowed). games_played in splits.
_PARITY_PRIOR = ["prior_season_games_played", "prior_season_mean_catch_rate"]
# Parity whitelist: rolling opportunity index (rolling → NOT static).
_PARITY_ROLL = ["opportunity_index_L3"]
# Parity history: raw per-game usage tokens (game_opportunity_index = the user's addition).
_PARITY_HIST = ["game_target_share", "game_target_hhi", "game_opportunity_index"]

# Columns the injector builds (the rest are already in the splits).
_BUILT = [*_RZ_ROLL, "prior_season_mean_catch_rate", *_PARITY_ROLL, *_PARITY_HIST]


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean WR red-zone + opportunity columns (within position)
# --------------------------------------------------------------------------- #
def _inject_wr_signals(train, val, test):
    """Build the WR red-zone / opportunity columns on each split's WR rows.

    The splits are all-position; team WR totals are WR-scoped (a WR's target share is
    of the team's *WR* targets). Computed within position, written onto the WR rows of
    the general frame (the WR pipeline filters to WR downstream); non-WR rows get 0.0
    and are never read. Mirrors the proven ``src/rb/features.py`` game-share / opp-index
    code (carries + 2·targets weighting) with WR-scoped denominators.

    Leakage: rolling features use ``rolling_agg(..., shift=1)`` and prior-season uses the
    S-1 aggregates already in the splits, so the whitelist set is pre-kickoff. The raw
    per-game ``game_*`` / ``redzone_*`` columns are HISTORY-only — ``build_game_history_arrays``
    applies its own prior-games shift — so their current-week value never reaches the model.
    """

    def _add(df):
        df = df.copy()
        is_wr = df["position"] == "WR"
        # Sort so the per-group rolling windows operate on chronological rows; assign back
        # by index (the transform/rolling outputs are wr-indexed, so order-agnostic on write).
        wr = df[is_wr].sort_values(["player_id", "season", "week"]).copy()

        # Team WR totals per game (WR-scoped), via transform to keep wr's index.
        g_team = wr.groupby(["recent_team", "season", "week"])
        team_tgt = g_team["targets"].transform("sum").to_numpy(dtype=float)
        team_car = g_team["carries"].transform("sum").to_numpy(dtype=float)
        tgt = wr["targets"].fillna(0).to_numpy(dtype=float)
        car = wr["carries"].fillna(0).to_numpy(dtype=float)

        # Raw per-game shares / concentration / weighted-opportunity index (HISTORY tokens).
        wr["game_target_share"] = np.divide(
            tgt, team_tgt, out=np.zeros_like(tgt), where=team_tgt > 0
        )
        wr["game_target_hhi"] = wr.groupby(["recent_team", "season", "week"])[
            "game_target_share"
        ].transform(lambda x: (x**2).sum())
        team_w = team_car + 2.0 * team_tgt
        player_w = car + 2.0 * tgt
        wr["game_opportunity_index"] = np.divide(
            player_w, team_w, out=np.zeros_like(player_w), where=team_w > 0
        )

        # Rolling (leakage-safe shift=1) — whitelist features for Ridge/LGBM.
        wr["opportunity_index_L3"] = rolling_agg(wr, "game_opportunity_index", _GRP, 3, agg="mean")
        wr["redzone_targets_L3"] = rolling_agg(wr, "redzone_targets", _GRP, 3, agg="mean")
        wr["redzone_target_share_L3"] = rolling_agg(wr, "redzone_target_share", _GRP, 3, agg="mean")

        # Prior-season catch rate (S-1 → S; mirrors the src/rb/features.py low-volume guard).
        if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(wr.columns):
            cr = safe_divide(wr["prior_season_mean_receptions"], wr["prior_season_mean_targets"])
            wr["prior_season_mean_catch_rate"] = cr.where(wr["prior_season_mean_targets"] >= 0.5)
        else:
            wr["prior_season_mean_catch_rate"] = 0.0

        for c in _BUILT:
            df[c] = 0.0
            df.loc[wr.index, c] = wr[c].to_numpy()
        return df

    return _add(train), _add(val), _add(test)


# --------------------------------------------------------------------------- #
# Config mutators — place each column on its correct branch
# --------------------------------------------------------------------------- #
def _extend_features(cfg, cols):
    """Extend ``get_feature_columns_fn`` (→ Ridge + LightGBM + NN-static input list)."""
    base = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda base=base: [*base(), *cols]


def _extend_static(cfg, cols):
    """Extend ``attn_static_features`` (→ NN static branch). Prior-season only — never rolling."""
    if "attn_static_features" in cfg:
        cfg["attn_static_features"] = [*cfg["attn_static_features"], *cols]


def _extend_history(cfg, cols):
    """Extend ``attn_history_stats`` (→ NN per-game history; Ridge/LGBM unaffected)."""
    if "attn_history_stats" in cfg:
        cfg["attn_history_stats"] = [*cfg["attn_history_stats"], *cols]


def _mut_rz(cfg):
    _extend_features(cfg, [*_RZ_PRIOR, *_RZ_ROLL])
    _extend_static(cfg, _RZ_PRIOR)  # prior-season is static-eligible; rolling is NOT (stop-rule)
    _extend_history(cfg, _RZ_HIST)
    return cfg


def _mut_parity(cfg):
    _extend_features(cfg, [*_PARITY_PRIOR, *_PARITY_ROLL])
    _extend_static(cfg, _PARITY_PRIOR)
    _extend_history(cfg, _PARITY_HIST)
    return cfg


def _mut_all(cfg):
    return _mut_parity(_mut_rz(cfg))


# --------------------------------------------------------------------------- #
# Metric — WR boom subgroup (Q4 / receiving-TD games), per model
# --------------------------------------------------------------------------- #
def metric_fn(result, position):
    """Per-model overall MAE/bias + boom-subgroup bias/RMSE/correlation.

    The gap is the boom tier, so judge there, not on overall MAE (a feature on ~1% of
    rows dilutes to noise). Subgroups are defined on *actuals* — ``q4`` = top fantasy-point
    quartile, ``rztd`` = receiving-TD games — so the slice is identical across arms with no
    baseline injection. ``correlation`` (Pearson pred-vs-actual on the slice) is the
    decomposition's closable edge. ``mae`` (overall) feeds the harness Ridge sentinel.
    """
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
        a = sub[col].to_numpy(dtype=float)
        b = sub["fantasy_points"].to_numpy(dtype=float)
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
        "+rz",
        cfg_mutator=_mut_rz,
        frame_injector=_inject_wr_signals,
        expect_ridge_identical=False,  # a real whitelist feature MUST move Ridge
        label="+red-zone (whitelist rolling+prior, NN-history raw)",
    ),
    Variant(
        "+parity",
        cfg_mutator=_mut_parity,
        frame_injector=_inject_wr_signals,
        expect_ridge_identical=False,
        label="+prior/per-game parity + opp-index",
    ),
    Variant(
        "+all",
        cfg_mutator=_mut_all,
        frame_injector=_inject_wr_signals,
        expect_ridge_identical=False,
        label="+rz +parity (full bundle)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
