"""Paired significance tests for comparing two forecasters' accuracy.

Used by ``analysis_expert_comparison.py`` to decide whether a gap between the
model and an expert baseline (e.g. NFL.com) is real or noise. Both forecasters
are scored against the *same* held-out observations, so the errors are paired —
which is exactly what these tests assume.

Two complementary tools:

- :func:`diebold_mariano_test` — the textbook forecast-comparison test
  (Diebold & Mariano 1995) with the Harvey-Leybourne-Newbold (1997) small-sample
  correction. Parametric; loss-power matched to the headline metric (MAE/RMSE).
- :func:`paired_bootstrap_metric_ci` — a confidence interval on the *difference*
  in a metric (ΔMAE / ΔRMSE), with optional **player-clustered** resampling so
  within-player correlation across weeks doesn't understate the interval.

Why both: DM gives a p-value under stationarity assumptions; the clustered
bootstrap is assumption-light and honest about the panel structure (the same
player appears in many player-weeks). Report the bootstrap CI as primary and DM
as the named, citable companion.

Neither test cares what *loss* either model trained on — comparison happens on
held-out errors under a shared metric. That is the whole point: a Huber-trained
model and an MSE-oriented expert are compared on the same footing here.

Pure NumPy/SciPy — no model or data-loading dependencies, so it is fast to unit
test in isolation. Lives under ``src/analysis/`` (not ``src/shared/``) on
purpose: ``src/scripts/scope_positions.py`` treats ``src/shared/`` as a global
trigger that would fire a full 6-position retrain, while ``src/analysis/`` does
not. See the plan/ADR for the rationale.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_METRIC_FNS = {
    "mae": lambda e: float(np.mean(np.abs(e))),
    "rmse": lambda e: float(np.sqrt(np.mean(np.square(e)))),
}


def _as_paired_errors(errors_a, errors_b) -> tuple[np.ndarray, np.ndarray]:
    """Coerce two error vectors to 1-D float arrays, validating shape + size."""
    e_a = np.asarray(errors_a, dtype=float).ravel()
    e_b = np.asarray(errors_b, dtype=float).ravel()
    if e_a.shape != e_b.shape:
        raise ValueError(
            f"errors_a and errors_b must have the same length; got {e_a.shape} vs {e_b.shape}"
        )
    if e_a.size < 2:
        raise ValueError("need at least 2 paired observations")
    return e_a, e_b


def _autocov(x: np.ndarray, k: int) -> float:
    """Population (1/n) k-th autocovariance of ``x`` about its own mean."""
    n = x.size
    xbar = x.mean()
    return float(np.sum((x[k:] - xbar) * (x[: n - k] - xbar)) / n)


def diebold_mariano_test(errors_a, errors_b, *, power: int = 1, h: int = 1) -> dict:
    """Paired Diebold-Mariano test of equal predictive accuracy (HLN-corrected).

    Tests H0: forecaster A and forecaster B have equal expected loss, on the
    *paired* forecast errors. The loss is ``L(e) = |e|**power`` — ``power=1``
    matches an MAE headline, ``power=2`` matches RMSE.

    Args:
        errors_a: forecast errors of model A (``pred - actual``).
        errors_b: forecast errors of model B, same observations / order.
        power: loss exponent (1 = absolute-error loss, 2 = squared-error loss).
        h: forecast horizon. For independent observations (player-weeks are not
           a multi-step horizon) leave at 1; ``h>1`` adds autocovariance terms
           to the long-run variance.

    Returns:
        ``{"dm_stat", "p_value", "n", "mean_loss_diff", "favored"}`` where the
        loss differential is ``d = L(e_a) - L(e_b)`` (negative ⇒ A more
        accurate), ``p_value`` is two-sided from a Student-t with ``n-1`` df, and
        ``favored`` is ``"model"`` (A), ``"expert"`` (B), or ``"tie"``.

    Reference: Diebold & Mariano (1995), *JBES* 13(3); Harvey, Leybourne &
    Newbold (1997), *IJF* 13(2) (small-sample correction).
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    e_a, e_b = _as_paired_errors(errors_a, errors_b)
    n = e_a.size

    d = np.abs(e_a) ** power - np.abs(e_b) ** power
    dbar = float(d.mean())

    gamma0 = _autocov(d, 0)
    if gamma0 <= 0.0:
        # Zero-variance loss differential: identical losses (tie) or a constant
        # nonzero gap (degenerate-decisive). Continuous errors never hit this;
        # guard so we don't divide by zero. (compute_metrics already emits NaN
        # to JSON for r2<2 samples, so an inf here is consistent with the
        # codebase's existing nan/inf-in-JSON convention.)
        if abs(dbar) < 1e-12:
            return {"dm_stat": 0.0, "p_value": 1.0, "n": n, "mean_loss_diff": 0.0, "favored": "tie"}
        return {
            "dm_stat": float("inf") if dbar > 0 else float("-inf"),
            "p_value": 0.0,
            "n": n,
            "mean_loss_diff": dbar,
            "favored": "model" if dbar < 0 else "expert",
        }

    # Long-run variance of the mean differential (autocovariances up to h-1).
    lrv = gamma0 + 2.0 * sum(_autocov(d, k) for k in range(1, h))
    var_mean = lrv / n
    if var_mean <= 0.0:
        # Negative LRV estimate (possible for h>1 with strong negative
        # autocorrelation). Fall back to the h=1 variance, which is >0 here.
        var_mean = gamma0 / n

    dm = dbar / np.sqrt(var_mean)
    # Harvey-Leybourne-Newbold small-sample correction + Student-t reference.
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = float(dm * hln_factor)
    p_value = float(2.0 * stats.t.sf(abs(dm_star), df=n - 1))

    return {
        "dm_stat": dm_star,
        "p_value": p_value,
        "n": n,
        "mean_loss_diff": dbar,
        "favored": "model" if dbar < 0 else "expert",
    }


def paired_bootstrap_metric_ci(
    errors_a,
    errors_b,
    *,
    metric: str = "mae",
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for ``metric(A) - metric(B)`` on paired errors.

    Resamples the paired observations with replacement ``n_boot`` times and
    recomputes the metric difference each time. When ``groups`` is supplied
    (e.g. ``player_id``), resampling is **clustered**: whole groups are drawn
    with replacement, so correlated player-weeks move together and the interval
    isn't artificially tight.

    Args:
        errors_a: forecast errors of model A (``pred - actual``).
        errors_b: forecast errors of model B, same observations / order.
        metric: ``"mae"`` or ``"rmse"``.
        groups: optional cluster id per observation (len == n). ``None`` ⇒ i.i.d.
            row resampling.
        n_boot: number of bootstrap replicates.
        seed: PRNG seed (``np.random.default_rng``) for reproducibility.
        ci: central interval mass (0.95 ⇒ 2.5/97.5 percentiles).

    Returns:
        ``{"delta", "lo", "hi", "p_value", "metric", "n_boot"}``. ``delta`` is the
        observed metric difference (negative ⇒ A better); ``[lo, hi]`` excluding 0
        ⇒ the difference is significant at the ``1-ci`` level. ``p_value`` is the
        two-sided bootstrap proportion.
    """
    e_a, e_b = _as_paired_errors(errors_a, errors_b)
    n = e_a.size
    metric = metric.lower()
    if metric not in _METRIC_FNS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_FNS)}; got {metric!r}")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    mfn = _METRIC_FNS[metric]
    observed = mfn(e_a) - mfn(e_b)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)

    if groups is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            deltas[b] = mfn(e_a[idx]) - mfn(e_b[idx])
    else:
        groups_arr = np.asarray(groups).ravel()
        if groups_arr.shape[0] != n:
            raise ValueError(f"groups must have length {n}; got {groups_arr.shape[0]}")
        uniq = np.unique(groups_arr)
        idx_by_group = [np.flatnonzero(groups_arr == g) for g in uniq]
        n_groups = len(idx_by_group)
        for b in range(n_boot):
            chosen = rng.integers(0, n_groups, size=n_groups)
            idx = np.concatenate([idx_by_group[c] for c in chosen])
            deltas[b] = mfn(e_a[idx]) - mfn(e_b[idx])

    alpha = 1.0 - ci
    lo = float(np.percentile(deltas, 100.0 * alpha / 2.0))
    hi = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    # Two-sided bootstrap p: twice the smaller tail mass at 0.
    p_value = float(min(1.0, 2.0 * min(np.mean(deltas >= 0.0), np.mean(deltas <= 0.0))))

    return {
        "delta": float(observed),
        "lo": lo,
        "hi": hi,
        "p_value": p_value,
        "metric": metric,
        "n_boot": int(n_boot),
    }
