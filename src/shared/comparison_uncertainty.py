"""Paired, player-clustered bootstrap intervals for model-versus-expert gaps.

A highlighted winner needs evidence beyond a point estimate. Every source is
graded on the same player-weeks. Each replicate therefore resamples players
(all of a player's weeks together, which preserves within-player correlation) and
recomputes every source's MAE and RMSE on that one draw. Group minima are taken
inside each replicate, so picking the best of four models or the best expert
after the fact is part of the interval rather than an unmodelled selection.

The draws are seeded, so a snapshot is reproducible. Only numpy and pandas are
used, because the serving image computes Timeline intervals at request time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REPLICATES = 2000
CONFIDENCE = 0.95
SEED = 20260925
METRICS = ("mae", "rmse")
# Cohorts whose rows are selected on outcomes or on a graded expert's own
# forecasts. Their cells are still reported, but no verdict is attached: selection
# on outcomes rewards bullish forecasts, and selection on a source's forecasts
# penalizes that source (winner's curse), so a winner there is not forecasting skill.
NOT_APPLICABLE_COHORTS = {
    "weekly_reference_top24": "selected_by_graded_forecast",
    "top30": "selected_on_outcomes",
    "top12": "selected_on_outcomes",
}
# Disclosed beside every backtest verdict. The graded season's model inputs are
# not all pregame, and the expert archives have no capture timestamp.
INFORMATION_SET_NOTE = (
    "Backtest model inputs for the graded season include realized quarterback "
    "participation, closing betting lines, recorded game-time weather and game-day "
    "roster status (ADR-0004), and the graded population is players who recorded a "
    "snap. The expert archives were fetched after the season and carry no capture "
    "timestamp; both graded experts omit every player ruled Out, so their information "
    "time is the final injury report or later. Live forecasts use pregame inputs only."
)
METHOD = {
    "method": "player_clustered_paired_bootstrap",
    "replicates": REPLICATES,
    "confidence": CONFIDENCE,
    "seed": SEED,
    "metrics": list(METRICS),
    "gap": "best_model_minus_best_expert_minimum_within_replicate",
    "headline_gap": "served_model_minus_best_expert",
    "winner_rule": (
        "A row names a winner only when the 95% interval for the served model minus "
        "the best expert excludes zero in the same direction for both MAE and RMSE; "
        "otherwise the row is a statistical tie. The best-of-four gap is context only: "
        "its minimum is taken inside each replicate, but it still gives the model "
        "family four draws."
    ),
    "no_verdict_cohorts": dict(NOT_APPLICABLE_COHORTS),
}


def _interval(draws: np.ndarray) -> list[float]:
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return [round(float(low), 4), round(float(high), 4)]


def _verdict(interval: list[float]) -> str:
    """Direction for a model-minus-expert error gap (negative favors models)."""
    if interval[1] < 0:
        return "models"
    if interval[0] > 0:
        return "experts"
    return "tie"


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def group_gap_intervals(
    frame: pd.DataFrame,
    actual: str,
    columns: dict[str, str],
    models,
    experts,
    *,
    cluster: str = "player_id",
    replicates: int = REPLICATES,
    seed: int = SEED,
) -> dict:
    """Best model minus best expert, and each model minus best expert, with CIs.

    ``columns`` maps each source to its forecast column on ``frame``. Rows must
    already be the common graded sample; any non-finite row is dropped for all
    sources together so the comparison stays paired. Deltas are model error
    minus expert error, so a negative value favors the models.
    """
    models = [source for source in models if source in columns]
    experts = [source for source in experts if source in columns]
    if not models or not experts:
        return _unavailable("model_or_expert_group_missing")
    if frame is None or frame.empty or cluster not in frame or actual not in frame:
        return _unavailable("no_common_rows")
    sources = [*models, *experts]
    # An absent column reads as missing for every row, never as a zero forecast.
    values = frame.reindex(columns=[actual, *(columns[source] for source in sources)]).apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        return _unavailable("no_common_rows")
    codes, players = pd.factorize(frame.loc[finite, cluster].astype(str), sort=True)
    groups = len(players)
    if groups < 2:
        return _unavailable("too_few_players")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(groups, np.full(groups, 1.0 / groups), size=replicates)
    weights = weights.astype(float)
    counts = weights @ np.bincount(codes, minlength=groups).astype(float)
    truth = values.loc[finite, actual].to_numpy(dtype=float)
    point = {metric: {} for metric in METRICS}
    draws = {metric: {} for metric in METRICS}
    for source in sources:
        error = values.loc[finite, columns[source]].to_numpy(dtype=float) - truth
        absolute = weights @ np.bincount(codes, weights=np.abs(error), minlength=groups)
        squared = weights @ np.bincount(codes, weights=error**2, minlength=groups)
        point["mae"][source] = float(np.mean(np.abs(error)))
        point["rmse"][source] = float(np.sqrt(np.mean(error**2)))
        draws["mae"][source] = absolute / counts
        draws["rmse"][source] = np.sqrt(squared / counts)
    out = {"status": "available", "n": int(finite.sum()), "players": int(groups)}
    for metric in METRICS:
        best_expert = min(experts, key=point[metric].get)
        best_model = min(models, key=point[metric].get)
        expert_draw = np.min([draws[metric][source] for source in experts], axis=0)
        model_draw = np.min([draws[metric][source] for source in models], axis=0)
        interval = _interval(model_draw - expert_draw)
        per_model = {}
        for model in models:
            model_interval = _interval(draws[metric][model] - expert_draw)
            per_model[model] = {
                "minus_best_expert": round(point[metric][model] - point[metric][best_expert], 4),
                "ci": model_interval,
                "verdict": _verdict(model_interval),
            }
        out[metric] = {
            "best_model": best_model,
            "best_expert": best_expert,
            "best_model_minus_best_expert": round(
                point[metric][best_model] - point[metric][best_expert], 4
            ),
            "ci": interval,
            "verdict": _verdict(interval),
            "models": per_model,
        }
    verdicts = {out[metric]["verdict"] for metric in METRICS}
    out["winner"] = verdicts.pop() if len(verdicts) == 1 else "tie"
    return out


def served_gap(result: dict, model: str | None) -> dict:
    """One pre-specified model against the best expert, under the both-metrics rule.

    ``result`` is a ``group_gap_intervals`` report. Its best-of-four gap credits
    the model family; this credits only ``model`` (the model the site serves for
    the position), so the headline does not get four draws. The best expert is
    still the minimum inside each replicate, which is conservative for the model.
    """
    if not model:
        return {**_unavailable("served_model_unknown"), "model": None}
    if not isinstance(result, dict) or result.get("status") != "available":
        reason = (
            result.get("reason", "gap_unavailable")
            if isinstance(result, dict)
            else "gap_unavailable"
        )
        return {**_unavailable(reason), "model": model}
    out = {"status": "available", "model": model}
    for metric in METRICS:
        entry = ((result.get(metric) or {}).get("models") or {}).get(model)
        if entry is None:
            return {**_unavailable("served_model_not_graded"), "model": model}
        out[metric] = {
            "best_expert": result[metric]["best_expert"],
            "minus_best_expert": entry["minus_best_expert"],
            "ci": list(entry["ci"]),
            "verdict": entry["verdict"],
        }
    verdicts = {out[metric]["verdict"] for metric in METRICS}
    out["winner"] = verdicts.pop() if len(verdicts) == 1 else "tie"
    return out
