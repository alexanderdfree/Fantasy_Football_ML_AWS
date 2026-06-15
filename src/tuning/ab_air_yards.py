"""A/B: raw yards-decomposition tokens in the attention per-game history (4 positions).

The RB/WR top-30 RMSE gap vs RotoWire is 100% the Q4 boom tier, and the diagnosis
(``src/analysis/rmse_gap_decomposition.py``) names exactly one closable edge:
**correlation = more boom signal**. Every position's attention sequence already tokenizes
*total* yards but is **blind to the air-yards-vs-after-catch split** — the downfield
target-depth / role signal that drives boom games and is the most orthogonal axis to the
existing volume / red-zone / efficiency tokens.

This adds the **raw, charted per-game** decomposition, NOT the derived ratios. RACR
(``recv_yds / air_yds``) and WOPR are reconstructable by the NN from the raw components and
are *already* columns in the splits (``racr``, ``wopr``, ``air_yards_share``) — lower
information than their raw inputs. We feed the raw inputs and let the sequence model learn
the rest (AGENTS.md history reach #2: "a raw per-game signal genuinely absent from the
sequence"; the owner's "use the raw inputs, not the ratios" principle).

Tokens (all raw nflverse-weekly columns, already passthrough in ``data/splits/*.parquet`` —
so this is **config-only**: the ``build_game_history_arrays`` KeyError guard won't fire, no
plumbing, per-position retrain):

    family   receiving positions (WR/TE/RB)        QB
    air      receiving_air_yards                   passing_air_yards
    yac      receiving_yards_after_catch           passing_yards_after_catch
    fd       receiving_first_downs*                passing_first_downs + rushing_first_downs

    *RB already carries receiving_first_downs (+ redzone splits); it is skipped there.

Position is inferred from the existing tokens (the pipeline cfg carries no position key):
receiving positions hold ``receiving_yards`` in ``attn_history_stats``, QB holds
``passing_yards``. Tokens already present are skipped, so one spec serves all four.

**Attention-NN-only by construction.** ``attn_history_stats`` feeds the NN history branch
only — Ridge / LightGBM / NN-static never read it — so every arm is
``expect_ridge_identical=True`` (the harness Ridge data-identity sentinel must stay
byte-identical across arms; if it moves, the change leaked into the shared path). Success =
the Attn NN's boom-tier (Q4 / TD-game) correlation up and bias→0 with overall MAE flat.

Metric: judge on the **boom subgroup** (Q4 = top fantasy-point quartile; tdgame = receiving-
or passing-TD games), NOT overall MAE — a token that matters on the tail dilutes to noise
overall (the #1053 lesson). Subgroups are on *actuals* (identical across arms; baseline
needs no injection). Report per-model bias / RMSE / correlation on the slice + overall MAE.

Gated rollout (ship per position, each its own PR + per-position retrain): WR leads (the
measured gap); on a win, mirror to TE, then RB (receiving side), then QB (passing side). A
flat/negative WR result pauses the rollout (the signal may be position-specific).

Run (local authoring smoke only — the real A/B runs on the GPU Batch fleet; local training
SIGSEGVs on the macOS torch+lightgbm+sklearn libomp triple-load)::

    python -m src.tuning.ab_air_yards --list                       # show the grid, run nothing
    python -m src.tuning.launch_ab --spec src.tuning.ab_air_yards   # GPU fleet (WR, default)
    python -m src.tuning.ab_air_yards --positions TE         # next gated phase
    python -m src.tuning.ab_air_yards --positions RB --only baseline +air +air_yac
    python -m src.tuning.ab_air_yards --positions QB         # passing decomposition + mobility
"""

from __future__ import annotations

import numpy as np

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR"]  # lead; run TE/RB/QB via --positions for the gated rollout
SEEDS = [42, 123, 7, 2024, 88, 17]  # 6 seeds — boom deltas are small (AGENTS.md seed band)

# Raw charted per-game tokens, keyed by the base yards token that marks the position as
# relevant. Already passthrough columns in data/splits; tokens already present are skipped.
_AIR = {"receiving_yards": ["receiving_air_yards"], "passing_yards": ["passing_air_yards"]}
_YAC = {
    "receiving_yards": ["receiving_yards_after_catch"],
    "passing_yards": ["passing_yards_after_catch"],
}
_FD = {
    "receiving_yards": ["receiving_first_downs"],
    "passing_yards": ["passing_first_downs", "rushing_first_downs"],
}


def _extend(cfg, families):
    """Append the position-relevant raw tokens to ``attn_history_stats`` (history → Attn-NN only).

    Position is inferred from the existing tokens; tokens already present are skipped;
    order is preserved and duplicates dropped. Mirrors the in-place ``cfg[...] = [...]``
    pattern of ``ab_boom_signals_wr._extend_history`` (the harness deep-copies cfg per cell).
    """
    hist = cfg.get("attn_history_stats")
    if not hist:
        return cfg
    out = list(hist)
    seen = set(out)
    for fam in families:
        for base_tok, new_toks in fam.items():
            if base_tok not in seen:
                continue
            for tok in new_toks:
                if tok not in seen:
                    out.append(tok)
                    seen.add(tok)
    cfg["attn_history_stats"] = out
    return cfg


def _mut_air(cfg):
    return _extend(cfg, [_AIR])


def _mut_air_yac(cfg):
    return _extend(cfg, [_AIR, _YAC])


def _mut_air_yac_fd(cfg):
    return _extend(cfg, [_AIR, _YAC, _FD])


def metric_fn(result, position):
    """Per-model overall MAE/bias + boom-subgroup bias/RMSE/correlation.

    Judge on the boom tier, not overall MAE. ``q4`` = top fantasy-point quartile;
    ``tdgame`` = receiving- (WR/TE/RB) or passing- (QB) TD games. Slices are on actuals, so
    identical across arms. ``corr`` (pred-vs-actual on the slice) is the closable edge;
    overall ``mae`` feeds the harness Ridge sentinel.
    """
    from src.analysis.cohort_analysis import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)

    cuts: dict = {}
    if len(df):
        q75 = float(np.quantile(df["fantasy_points"].to_numpy(dtype=float), 0.75))
        cuts["q4"] = df[df["fantasy_points"] >= q75]
        td_col = next((c for c in ("receiving_tds", "passing_tds") if c in df.columns), None)
        if td_col is not None:
            cuts["tdgame"] = df[df[td_col] >= 1]
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
    Variant("baseline", label="production attn_history_stats (unchanged)"),
    Variant(
        "+air",
        cfg_mutator=_mut_air,
        expect_ridge_identical=True,  # history-only → Ridge/LGBM/NN-static unaffected
        label="+air_yards (recv/pass) raw per-game history token",
    ),
    Variant(
        "+air_yac",
        cfg_mutator=_mut_air_yac,
        expect_ridge_identical=True,
        label="+air_yards +yards_after_catch",
    ),
    Variant(
        "+air_yac_fd",
        cfg_mutator=_mut_air_yac_fd,
        expect_ridge_identical=True,
        label="+air_yards +yac +first_downs (QB also +rushing_first_downs)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)
