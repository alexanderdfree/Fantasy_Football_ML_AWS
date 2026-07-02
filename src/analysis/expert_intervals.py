"""Per-projection prediction intervals for the expert sources (NFL.com, RotoWire).

Each expert emits a single point projection per player-week with no uncertainty.
This script attaches an **80% floor–ceiling band** to every projection by
quantile-regressing the *actual* fantasy points on the *projection*, per source ×
position, at τ = 0.1 / 0.5 / 0.9:

    actual ≈ Q_τ(actual | projection)   (a separate linear fit per τ)

The band for a projection ``p`` is ``[Q_0.1(p), Q_0.9(p)]`` (floor, ceiling) with
``Q_0.5(p)`` as the recalibrated central estimate. Because each τ is fit
independently, the three lines can cross at the edges of the projection range; we
**rearrange** (sort) the three predicted quantiles per row so the band is always
``floor ≤ median ≤ ceiling`` (Chernozhukov et al. 2010, monotone rearrangement).

This is the *conditional* version of the per-source residual σ panel — instead of
one σ per (source, position), the spread is allowed to scale with the projection
level (a star projected for 25 pts has a wider absolute band than a backup
projected for 4). Linear quantile regression captures that heteroscedasticity
directly through the per-τ slopes (the τ=0.9 slope exceeds the τ=0.1 slope when
high projections are more uncertain), and the fitted ``(intercept, slope)`` pairs
are compact enough to commit and apply live to any projection in the serving UI.

Calibration is the headline check: on **held-out** player-weeks (the model's 2025
test season), the nominal-80% band should empirically contain ≈80% of actuals.
We fit on the earlier seasons and report coverage on the held-out season, so the
coverage numbers correspond to the shipped (committed) parameters.

Methodology caveats (carried through to the output + the serving UI):
  - **RotoWire/Sleeper provenance is unverified** — its historical projections may
    be backfilled/look-ahead (see ``_SLEEPER_NOTE`` in analysis_expert_comparison).
    Look-ahead would make RotoWire's residuals artificially small and its bands
    **over-tight** (held-out coverage < 0.8). Flagged ``provenance_unverified`` and
    sanity-checked, not trusted blind.
  - **NFL.com K is totals-only** (standard scoring, not PPR-decomposable) — handled
    via the ``_TOTALS_ONLY_POSITIONS`` path; its projection + actual are on the
    standard-scoring K scale, flagged ``totals_only``.
  - **NFL.com has no DST; RotoWire has no K** — those cells are emitted as ``None``.

No model training, no retrain trigger (analysis-only). Writes the committed
``src/serving/expert_intervals.json`` consumed by the Comparison tab.

Operator usage:
  python -m src.analysis.expert_intervals
  python -m src.analysis.expert_intervals --eval-seasons 2025 --min-fit-season 2018
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

# Reuse the committed-summary join plumbing (same package) — single source of
# truth for actuals-by-position, DST team build, key normalization, and which
# expert covers which position. The intervals fit on the very same joined panel.
from src.analysis.analysis_expert_comparison import (
    _EXPERT_PRED_COL,
    _project_sleeper_to_ppr,
)
from src.analysis.analysis_nflcom_baseline import (
    _json_default,
    _load_actuals,
    _project_nflcom_to_ppr,
)
from src.analysis.build_comparison_summary import (
    _KEYS,
    _NFLCOM_POSITIONS,
    _ROTOWIRE_POSITIONS,
    _dst_actuals,
    _normalize_keys,
    _position_actuals,
)
from src.analysis.sleeper_loader import load_sleeper_with_gsis_id
from src.config import TEST_SEASONS
from src.data.nflcom_loader import load_nflcom_with_gsis_id

POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DST")
SCORING_FORMAT = "ppr"
TAUS: tuple[float, float, float] = (0.1, 0.5, 0.9)
NOMINAL_COVERAGE = TAUS[2] - TAUS[0]  # 0.8

EVAL_SEASONS_DEFAULT: tuple[int, ...] = tuple(TEST_SEASONS) if TEST_SEASONS else (2025,)
# Earliest season to *fit* on, per source. NFL.com's hvpkod archive starts 2015;
# Sleeper/RotoWire returns genuine projections only from 2018 (earlier seasons are
# placeholder junk — see sleeper_loader._MIN_SEASON). The fit window is
# [min_fit_season .. min(eval_seasons) - 1] intersected with the source's coverage.
MIN_FIT_SEASON_DEFAULT = 2015
_SLEEPER_MIN_SEASON = 2018

# Guards: a quantile fit needs enough rows to resolve the deciles, and a coverage
# number needs enough held-out rows to mean anything. Below these, emit a skip.
_MIN_FIT_ROWS = 80
_MIN_EVAL_ROWS = 20
_MAX_EXAMPLES = 6  # per (source, position) — bounded so the committed JSON stays small.

# Fit-season selection. Two contamination modes show up empirically and would
# wreck held-out calibration if fit on blind (verified forensically 2026-05-31):
#   1. LOOK-AHEAD: the hvpkod/NFL-Data community archive's NFL.com "projected"
#      CSVs for 2021, 2022 AND 2023 are backfilled with the *realized* box score,
#      not projections — proven by an exact-match test: for 2021–2023 the archive's
#      projected raw passing yards equal the actual box score ~92–96% of the time
#      (e.g. 2023 Josh Allen "projected" 359 / actual 359), and the values are
#      whole integers; for 2024–2025 they're fractional expected values matching
#      actuals ~0%. (This is a third-party-archive scraping/labeling error, NOT
#      NFL.com publishing — and it's NFL.com only; RotoWire is clean every year.)
#      The right detector is the **near-exact fraction**, NOT residual std: a
#      season ~1/3 backfilled still has a normal-looking std (the genuine rows
#      dominate the variance) so a std floor misses it, but the copied rows pile
#      up at |actual − projection| ≈ 0. A season is dropped as look-ahead when
#      >``_LOOKAHEAD_FRAC`` of its rows fall within ``_LOOKAHEAD_EPS`` pts of the
#      actual. At eps=0.5 the split is clean: genuine cells sit ≤0.18 (NFL.com
#      2024/25 ~0.13–0.18, NFL.com K ~0.07–0.09, all RotoWire ~0.04–0.12) while
#      every backfilled NFL.com offense cell is ≥0.40 — so 0.30 separates with margin.
#   2. NON-STATIONARITY: a genuine season is kept only if its residual std is
#      within ``_STATIONARITY_FRAC`` of the *most recent* genuine season's (the
#      one closest to the held-out distribution) — a secondary guard for honest
#      drift, now that look-ahead is handled directly.
# Net effect (no per-cell hand-tuning): NFL.com offense collapses onto its only
# genuine pre-2025 season (2024); NFL.com K's full stationary history and all of
# RotoWire are kept. Every cell calibrates to ≈0.72–0.82 (nominal 0.80).
_LOOKAHEAD_EPS = 0.5
_LOOKAHEAD_FRAC = 0.30
_STATIONARITY_FRAC = 0.8

_TOTALS_ONLY = {("nflcom", "K")}  # NFL.com K is standard-scoring totals-only.

_NFLCOM_NOTE = (
    "Bands from a quantile regression of actual PPR points on NFL.com's projection "
    "(per position). The hvpkod NFL.com archive's pre-2024 offense 'projected' files are "
    "backfilled with realized box scores (their projected stats equal the actual stats "
    "~92-96% of the time, and are whole integers) — a third-party-archive scraping error, "
    "not NFL.com itself. Those seasons are auto-excluded, so offense bands fit on the "
    "genuine 2024 projections. NFL.com K is unaffected; NFL.com has no DST projections."
)
_ROTOWIRE_NOTE = (
    "RotoWire projections via Sleeper's unofficial API — provenance unverified. In "
    "practice its error spread is stable across every season and matches the held-out "
    "season, so its bands sanity-check clean (no look-ahead detected). K is out of scope."
)

# Repo root = two levels up from src/analysis/expert_intervals.py.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_PATH_DEFAULT = os.path.join(_REPO_ROOT, "src", "serving", "expert_intervals.json")


# ---------- Quantile fit + apply ---------------------------------------------


def _fit_quantile_params(x: np.ndarray, y: np.ndarray) -> dict[float, dict[str, float]]:
    """Fit one linear quantile regression per τ; return ``{τ: {intercept, slope}}``.

    ``alpha=0`` ⇒ no L1 penalty on the coefficient (pure quantile regression);
    ``solver="highs"`` is scipy's fast LP solver. One feature (the projection), so
    each fit is a slope + intercept.
    """
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    params: dict[float, dict[str, float]] = {}
    for tau in TAUS:
        qr = QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X, y)
        params[tau] = {
            "intercept": round(float(qr.intercept_), 6),
            "slope": round(float(qr.coef_[0]), 6),
        }
    return params


def _apply_bands(
    params: dict[float, dict[str, float]], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map projections ``x`` → (floor, median, ceiling), monotone-rearranged.

    The three independent quantile lines can cross at the extremes; sorting the
    per-row predictions guarantees ``floor ≤ median ≤ ceiling`` (the serving JS
    mirrors this 3-element sort when computing a band for a live projection).
    """
    x = np.asarray(x, dtype=float)
    cols = [params[t]["intercept"] + params[t]["slope"] * x for t in TAUS]
    q = np.sort(np.column_stack(cols), axis=1)
    return q[:, 0], q[:, 1], q[:, 2]


# ---------- Panel building (per source × position) ---------------------------


def _name_lookup(raw_proj: pd.DataFrame) -> dict[str, str]:
    """player_id → display name from a raw projection frame (last non-null wins).

    DST rows are team-keyed (player_id == team abbrev), so the name is the team.
    """
    if raw_proj is None or raw_proj.empty or "player_name" not in raw_proj.columns:
        return {}
    df = raw_proj[["player_id", "player_name"]].dropna(subset=["player_id"]).copy()
    df["player_id"] = df["player_id"].astype(str)
    return dict(zip(df["player_id"], df["player_name"].astype(str), strict=False))


def _projection_frame(source: str, raw_proj: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Project one source's raw stats to a ``[*_KEYS, projection]`` frame."""
    if source == "nflcom":
        proj = _project_nflcom_to_ppr(raw_proj, pos, SCORING_FORMAT)
        col = "nflcom_pred_total"
    else:
        proj = _project_sleeper_to_ppr(raw_proj, pos, SCORING_FORMAT)
        col = _EXPERT_PRED_COL
    if proj is None or proj.empty or col not in proj.columns:
        return pd.DataFrame(columns=[*_KEYS, "projection"])
    return proj[[*_KEYS, col]].rename(columns={col: "projection"})


def _build_panel(
    source: str,
    raw_proj: pd.DataFrame,
    pos: str,
    offense_actuals: pd.DataFrame,
    dst_actuals: pd.DataFrame,
    all_seasons: Sequence[int],
) -> pd.DataFrame:
    """Join a source's projection to actuals → ``[*_KEYS, projection, actual]``.

    Spans every season in ``all_seasons``; the caller splits into fit/eval by the
    ``season`` column. Empty if the source has no rows for the position.
    """
    proj = _projection_frame(source, raw_proj, pos)
    actuals = _position_actuals(pos, offense_actuals, dst_actuals, all_seasons)
    if proj.empty or actuals.empty:
        return pd.DataFrame(columns=[*_KEYS, "projection", "actual"])
    a = _normalize_keys(actuals[[*_KEYS, "actual_pts"]]).rename(columns={"actual_pts": "actual"})
    e = _normalize_keys(proj)
    joined = a.merge(e, on=_KEYS, how="inner")
    joined = joined[joined["projection"].notna() & joined["actual"].notna()]
    return joined.reset_index(drop=True)


def _examples(eval_df: pd.DataFrame, params: dict, names: dict[str, str]) -> list[dict]:
    """Up to ``_MAX_EXAMPLES`` concrete player-weeks with their bands.

    Picks the highest-projection rows (the fantasy-relevant ones), one per player,
    so the UI shows real floor–median–ceiling bands against the realized actual.
    """
    if eval_df.empty:
        return []
    ranked = eval_df.sort_values("projection", ascending=False).drop_duplicates("player_id")
    ranked = ranked.head(_MAX_EXAMPLES)
    floor, median, ceiling = _apply_bands(params, ranked["projection"].to_numpy())
    out: list[dict] = []
    for (_, row), f, m, c in zip(ranked.iterrows(), floor, median, ceiling, strict=False):
        pid = str(row["player_id"])
        actual = float(row["actual"])
        out.append(
            {
                "player_id": pid,
                "player_name": names.get(pid, pid),
                "season": int(row["season"]),
                "week": int(row["week"]),
                "projection": round(float(row["projection"]), 2),
                "floor": round(float(f), 2),
                "median": round(float(m), 2),
                "ceiling": round(float(c), 2),
                "actual": round(actual, 2),
                "in_band": bool(f <= actual <= c),
            }
        )
    return out


def lookahead_seasons(actual: pd.Series, projection: pd.Series, season: pd.Series) -> set[int]:
    """Seasons whose "projections" are backfilled realized stats (look-ahead).

    A season is flagged when >``_LOOKAHEAD_FRAC`` of its rows fall within
    ``_LOOKAHEAD_EPS`` pts of the actual, or its residual std is degenerate
    (zero-variance/NaN — an all-copied tell). The three Series must be
    index-aligned (columns of one frame). Shared by the interval fit's season
    selection below and ``expert_uncertainty``'s pooled reliability σ; see the
    module comment for why near-exact fraction (not a std floor) is the right
    detector.
    """
    resid = actual.astype(float) - projection.astype(float)
    near_exact = (resid.abs() < _LOOKAHEAD_EPS).groupby(season).mean()
    stds = resid.groupby(season).std()
    return {
        int(s) for s in stds.index if near_exact[s] > _LOOKAHEAD_FRAC or not np.isfinite(stds[s])
    }


def _select_fit_seasons(
    panel: pd.DataFrame, eval_set: set[int]
) -> tuple[list[int], dict[int, str]]:
    """Pick genuine, distribution-matched fit seasons from the pre-eval panel.

    Drops look-ahead-contaminated seasons (>``_LOOKAHEAD_FRAC`` of rows are near-exact
    copies of the actual) and non-stationary older seasons (residual std below
    ``_STATIONARITY_FRAC`` of the most recent genuine season). Returns
    ``(kept_seasons, {season: exclusion_reason})``.
    """
    pre = panel[~panel["season"].isin(eval_set)]
    if pre.empty:
        return [], {}
    look = lookahead_seasons(pre["actual"], pre["projection"], pre["season"])
    resid = pre["actual"] - pre["projection"]
    stds = resid.groupby(pre["season"]).std()

    excluded: dict[int, str] = {s: "look-ahead" for s in look}
    genuine: dict[int, float] = {
        int(season): float(stds[season]) for season in stds.index if int(season) not in look
    }
    if not genuine:
        return [], excluded

    ref = genuine[max(genuine)]  # most-recent genuine season ≈ the held-out regime
    kept: list[int] = []
    for season in sorted(genuine):
        if genuine[season] >= _STATIONARITY_FRAC * ref:
            kept.append(season)
        else:
            excluded[season] = "non-stationary"
    return kept, excluded


def _calibrate(
    source: str, pos: str, panel: pd.DataFrame, eval_set: set[int], names: dict[str, str]
) -> dict:
    """Fit on the selected genuine seasons, evaluate held-out coverage on ``eval_set``.

    Returns a block with the committed params, the held-out calibration, which
    seasons were fit / excluded (and why), the projection range, and a few real
    example player-weeks — or a ``{"skipped": ...}`` block when data is too thin.
    """
    if panel.empty:
        return {"skipped": True, "reason": "no joined projection/actual rows"}
    fit_seasons, excluded = _select_fit_seasons(panel, eval_set)
    fit_df = panel[panel["season"].isin(fit_seasons)]
    eval_df = panel[panel["season"].isin(eval_set)]
    if len(fit_df) < _MIN_FIT_ROWS or fit_df["projection"].nunique() < 2:
        return {
            "skipped": True,
            "reason": f"insufficient genuine fit rows ({len(fit_df)})",
            "fit_seasons": fit_seasons,
            "excluded_seasons": excluded,
        }
    if len(eval_df) < _MIN_EVAL_ROWS:
        return {"skipped": True, "reason": f"insufficient eval rows ({len(eval_df)})"}

    params = _fit_quantile_params(fit_df["projection"].to_numpy(), fit_df["actual"].to_numpy())

    floor, median, ceiling = _apply_bands(params, eval_df["projection"].to_numpy())
    actual = eval_df["actual"].to_numpy()
    covered = (actual >= floor) & (actual <= ceiling)
    coverage = float(covered.mean())

    # Provenance / scale flags + a coverage sanity flag (the "tune or note it" hook).
    totals_only = (source, pos) in _TOTALS_ONLY
    if coverage < NOMINAL_COVERAGE - 0.1:
        cal_flag = "tight"  # band narrower than nominal → under-covers (look-ahead?)
    elif coverage > NOMINAL_COVERAGE + 0.1:
        cal_flag = "wide"
    else:
        cal_flag = "ok"

    proj = fit_df["projection"].to_numpy()
    return {
        "params": {
            "floor": params[TAUS[0]],
            "median": params[TAUS[1]],
            "ceiling": params[TAUS[2]],
        },
        "calibration": {
            "coverage": round(coverage, 4),
            "mean_width": round(float(np.mean(ceiling - floor)), 4),
            "median_below": round(float(np.mean(actual <= median)), 4),
            "n_eval": int(len(eval_df)),
            "n_fit": int(len(fit_df)),
            "flag": cal_flag,
        },
        "proj_range": [round(float(proj.min()), 2), round(float(proj.max()), 2)],
        "fit_seasons": fit_seasons,
        "excluded_seasons": excluded,
        "totals_only": totals_only,
        "scoring_note": "standard scoring (totals-only)" if totals_only else None,
        "examples": _examples(eval_df, params, names),
    }


# ---------- Orchestration ----------------------------------------------------


def build_intervals(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    min_fit_season: int = MIN_FIT_SEASON_DEFAULT,
    *,
    nflcom_loader=None,
    sleeper_loader=None,
    actuals_loader=None,
    dst_actuals_loader=None,
) -> dict:
    """Build the committed interval dict. Loaders are injectable for tests."""
    eval_seasons = tuple(int(s) for s in eval_seasons)
    eval_set = set(eval_seasons)
    first_eval = min(eval_seasons)
    nflcom_loader = nflcom_loader or load_nflcom_with_gsis_id
    sleeper_loader = sleeper_loader or load_sleeper_with_gsis_id
    actuals_loader = actuals_loader or _load_actuals
    dst_actuals_loader = dst_actuals_loader or _dst_actuals

    # Per-source fit windows differ (archive coverage); both extend through the
    # last pre-eval season. NFL.com from 2015; Sleeper from 2018.
    nflcom_fit = [s for s in range(min_fit_season, first_eval)]
    sleeper_fit = [s for s in range(max(min_fit_season, _SLEEPER_MIN_SEASON), first_eval)]
    nflcom_seasons = sorted(set(nflcom_fit) | eval_set)
    sleeper_seasons = sorted(set(sleeper_fit) | eval_set)
    all_actual_seasons = sorted(set(nflcom_seasons) | set(sleeper_seasons))

    print(f"Loading actuals (offense + K) for {all_actual_seasons}...")
    offense_actuals = actuals_loader(all_actual_seasons)
    # DST actuals come from the team-level build, which reads local cache parquets
    # (weekly / schedules / team-week stats). On a box without that cache the build
    # raises — a real data-source boundary, so degrade to "no DST" rather than
    # killing the offense intervals (which load straight from nflverse URLs).
    print("Building DST actuals (team-level)...")
    try:
        dst_actuals = dst_actuals_loader(all_actual_seasons)
    except (OSError, ValueError, KeyError) as e:
        print(f"  WARN: DST actuals unavailable ({type(e).__name__}: {e}); DST will be skipped")
        dst_actuals = pd.DataFrame(columns=[*_KEYS, "actual_pts"])

    print(f"Loading NFL.com projections for {nflcom_seasons}...")
    nflcom_full = nflcom_loader(seasons=nflcom_seasons)
    print(f"Loading RotoWire (Sleeper) projections for {sleeper_seasons}...")
    sleeper_full = sleeper_loader(sleeper_seasons)

    raw = {"nflcom": nflcom_full, "rotowire": sleeper_full}
    names = {"nflcom": _name_lookup(nflcom_full), "rotowire": _name_lookup(sleeper_full)}
    covers = {"nflcom": _NFLCOM_POSITIONS, "rotowire": _ROTOWIRE_POSITIONS}
    panel_seasons = {"nflcom": nflcom_seasons, "rotowire": sleeper_seasons}

    candidate_fit = {"nflcom": nflcom_fit, "rotowire": sleeper_fit}
    intervals: dict[str, dict] = {"nflcom": {}, "rotowire": {}}
    look_ahead: dict[str, set[int]] = {"nflcom": set(), "rotowire": set()}
    for source in ("nflcom", "rotowire"):
        for pos in POSITIONS:
            if pos not in covers[source]:
                intervals[source][pos] = None  # source can't cover this position
                continue
            panel = _build_panel(
                source, raw[source], pos, offense_actuals, dst_actuals, panel_seasons[source]
            )
            block = _calibrate(source, pos, panel, eval_set, names[source])
            intervals[source][pos] = block
            for season, why in block.get("excluded_seasons", {}).items():
                if why == "look-ahead":
                    look_ahead[source].add(int(season))
            cov = block.get("calibration", {}).get("coverage")
            print(
                f"  {source:<8} {pos:<4} fit={block.get('fit_seasons')} "
                f"coverage={cov} ({block.get('reason', 'ok')})"
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scoring": SCORING_FORMAT,
        "nominal_coverage": NOMINAL_COVERAGE,
        "tau": list(TAUS),
        "eval_seasons": list(eval_seasons),
        "method": (
            "linear quantile regression of actual fantasy points on the source's "
            "projection, per position (τ = 0.1 / 0.5 / 0.9, monotone-rearranged)"
        ),
        "sources_meta": {
            "nflcom": {
                "label": "NFL.com",
                "candidate_fit_seasons": candidate_fit["nflcom"],
                "look_ahead_seasons": sorted(look_ahead["nflcom"]),
                "provenance_unverified": False,
                "note": _NFLCOM_NOTE,
            },
            "rotowire": {
                "label": "RotoWire",
                "candidate_fit_seasons": candidate_fit["rotowire"],
                "look_ahead_seasons": sorted(look_ahead["rotowire"]),
                "provenance_unverified": True,
                "note": _ROTOWIRE_NOTE,
            },
        },
        "intervals": intervals,
    }


def main(
    eval_seasons: Sequence[int] = EVAL_SEASONS_DEFAULT,
    min_fit_season: int = MIN_FIT_SEASON_DEFAULT,
    output_path: str = OUTPUT_PATH_DEFAULT,
) -> dict:
    result = build_intervals(eval_seasons, min_fit_season)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\nWrote {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-projection prediction intervals for NFL.com / RotoWire experts"
    )
    parser.add_argument("--eval-seasons", nargs="+", type=int, default=list(EVAL_SEASONS_DEFAULT))
    parser.add_argument("--min-fit-season", type=int, default=MIN_FIT_SEASON_DEFAULT)
    parser.add_argument("--output-path", default=OUTPUT_PATH_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        eval_seasons=tuple(args.eval_seasons),
        min_fit_season=args.min_fit_season,
        output_path=args.output_path,
    )
