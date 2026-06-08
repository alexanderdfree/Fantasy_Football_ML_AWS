"""A/B: propagate the RB/WR red-zone + opportunity boom block to TE (parity).

TE is the left-behind skill position — RB carries red-zone + per-game opportunity
everywhere, WR got them via #1061 (``src/tuning/ab_boom_signals_wr.py``), but TE has
**none** of red-zone / opportunity / per-game-share in any branch. TEs are heavy
red-zone targets, so the same boom-tier signal that helped WR plausibly helps TE. This
is the validation gate before merging the block into ``src/te/{features,config}.py``.

Self-contained (the production TE config does NOT carry these yet): the frame injector
builds the columns and the cfg mutators whitelist / history-place them, so the baseline
is pure production TE and the variant is +block. Mirrors the WR A/B exactly (TE-scoped
team totals; TE role proxy is targets, like WR).

* ``+rz``     — red-zone receiving.   Whitelist: ``redzone_targets_L3`` /
  ``redzone_target_share_L3`` (rolling, ``shift=1``) + ``prior_season_total_redzone_touches``
  / ``..._per_game`` (already in splits). History (NN, raw per-game): ``redzone_targets`` /
  ``redzone_target_share`` (already in splits).
* ``+parity`` — ``prior_season_games_played`` (splits) + ``prior_season_mean_catch_rate``
  (derived S-1) + ``opportunity_index_L3`` (rolling) → whitelist; ``game_target_share`` /
  ``game_target_hhi`` / ``game_opportunity_index`` (raw per-game) → history.
* ``+all``    — ``+rz`` ∪ ``+parity`` (the arm WR shipped).

Branch placement (per-model attribution: LGBM/Ridge Δ = the whitelist effect = the
gap-relevant signal; attn Δ = whitelist + history). Rolling features go to the whitelist
only — NEVER ``attn_static_features`` (AGENTS.md stop-rule); only non-windowed
prior-season aggregates touch the static branch. The raw per-game ``game_*`` /
``redzone_*`` tokens are history-only (``build_game_history_arrays`` lags them), so their
current-week value never reaches the model — NOT the rejected windowed inheritance token
(#1053), but raw per-game signals genuinely absent from TE's sequence (history reach #2).

Metric: judge on the **TE boom subgroup**, NOT overall MAE (the #1053 dilution trap).
Subgroups are on *actuals* (identical across arms): ``q4`` = top fantasy-point quartile,
``rztd`` = receiving-TD games. Per model: subgroup bias / RMSE / correlation + overall MAE
(held-flat sanity + Ridge sentinel). Ship if the boom bias is direction-robust across all
four models and seeds with overall MAE flat.

Run::

    python -m src.tuning.ab_boom_signals_te               # TE, 3 seeds, autodetect -j
    python -m src.tuning.ab_boom_signals_te --list        # show the grid, run nothing
    python -m src.tuning.ab_boom_signals_te --only +rz -j 2  # fast smoke
"""

from __future__ import annotations

import numpy as np

from src.shared.feature_build import rolling_agg, safe_divide
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["TE"]
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
# Parity history: raw per-game usage tokens.
_PARITY_HIST = ["game_target_share", "game_target_hhi", "game_opportunity_index"]

# Columns the injector builds (the rest are already in the splits).
_BUILT = [*_RZ_ROLL, "prior_season_mean_catch_rate", *_PARITY_ROLL, *_PARITY_HIST]


# --------------------------------------------------------------------------- #
# Frame injector — leakage-clean TE red-zone + opportunity columns (within position)
# --------------------------------------------------------------------------- #
def _inject_te_signals(train, val, test):
    """Build the TE red-zone / opportunity columns on each split's TE rows.

    The splits are all-position; team TE totals are TE-scoped (a TE's target share is of
    the team's *TE* targets). Computed within position, written onto the TE rows of the
    general frame (the TE pipeline filters to TE downstream); non-TE rows get 0.0 and are
    never read. Mirrors src/wr/features.py (carries + 2·targets opportunity weighting) with
    TE-scoped denominators — must stay byte-identical to the production src/te/features.py
    block that ships on GO.

    Leakage: rolling uses ``rolling_agg(..., shift=1)`` and prior-season uses the S-1
    aggregates already in the splits, so the whitelist set is pre-kickoff. The raw per-game
    ``game_*`` / ``redzone_*`` columns are HISTORY-only — ``build_game_history_arrays`` applies
    its own prior-games shift — so their current-week value never reaches the model.
    """

    def _add(df):
        df = df.copy()
        is_te = df["position"] == "TE"
        te = df[is_te].sort_values(["player_id", "season", "week"]).copy()

        # Team TE totals per game (TE-scoped), via transform to keep te's index.
        g_team = te.groupby(["recent_team", "season", "week"])
        team_tgt = g_team["targets"].transform("sum").to_numpy(dtype=float)
        team_car = g_team["carries"].transform("sum").to_numpy(dtype=float)
        tgt = te["targets"].fillna(0).to_numpy(dtype=float)
        car = te["carries"].fillna(0).to_numpy(dtype=float)

        # Raw per-game shares / concentration / weighted-opportunity index (HISTORY tokens).
        te["game_target_share"] = np.divide(
            tgt, team_tgt, out=np.zeros_like(tgt), where=team_tgt > 0
        )
        te["game_target_hhi"] = te.groupby(["recent_team", "season", "week"])[
            "game_target_share"
        ].transform(lambda x: (x**2).sum())
        team_w = team_car + 2.0 * team_tgt
        player_w = car + 2.0 * tgt
        te["game_opportunity_index"] = np.divide(
            player_w, team_w, out=np.zeros_like(player_w), where=team_w > 0
        )

        # Rolling (leakage-safe shift=1) — whitelist features for Ridge/LGBM.
        te["opportunity_index_L3"] = rolling_agg(te, "game_opportunity_index", _GRP, 3, agg="mean")
        te["redzone_targets_L3"] = rolling_agg(te, "redzone_targets", _GRP, 3, agg="mean")
        te["redzone_target_share_L3"] = rolling_agg(te, "redzone_target_share", _GRP, 3, agg="mean")

        # Prior-season catch rate (S-1 → S; mirrors the src/wr/features.py low-volume guard).
        if {"prior_season_mean_receptions", "prior_season_mean_targets"} <= set(te.columns):
            cr = safe_divide(te["prior_season_mean_receptions"], te["prior_season_mean_targets"])
            te["prior_season_mean_catch_rate"] = cr.where(te["prior_season_mean_targets"] >= 0.5)
        else:
            te["prior_season_mean_catch_rate"] = 0.0

        for c in _BUILT:
            df[c] = 0.0
            df.loc[te.index, c] = te[c].to_numpy()
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
# Metric — TE boom subgroup (Q4 / receiving-TD games), per model
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
    Variant("baseline", label="TE production (unchanged)"),
    Variant(
        "+rz",
        cfg_mutator=_mut_rz,
        frame_injector=_inject_te_signals,
        expect_ridge_identical=False,  # a real whitelist feature MUST move Ridge
        label="+red-zone (whitelist rolling+prior, NN-history raw)",
    ),
    Variant(
        "+parity",
        cfg_mutator=_mut_parity,
        frame_injector=_inject_te_signals,
        expect_ridge_identical=False,
        label="+prior/per-game parity + opp-index",
    ),
    Variant(
        "+all",
        cfg_mutator=_mut_all,
        frame_injector=_inject_te_signals,
        expect_ridge_identical=False,
        label="+rz +parity (full bundle)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
