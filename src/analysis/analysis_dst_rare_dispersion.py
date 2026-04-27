"""DST rare-event dispersion diagnostic.

Decides the right loss function for the 4 very-rare DST targets:
    def_safeties, def_tds, def_blocked_kicks, special_teams_tds

Decision rules applied to each target on the TRAIN split only:
  - dispersion = Var/Mean.  ~1.0 ⇒ Poisson; >>1 ⇒ overdispersed (NB / Tweedie);
    <<1 ⇒ underdispersed.
  - zero-excess = P(y=0)_observed - P(y=0)_Poisson(lambda=mean).  Near 0 ⇒ the
    observed zero mass matches Poisson; large positive ⇒ zero-inflation.
  - P(y>=2) tells us how often the event "fires more than once" in a game.
    If this is tiny (<~2% of rows), the target is effectively binary and BCE
    on (y>0) is cleaner than regressing a count.

Prints a per-target table and a recommendation.  No models trained here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np

from src.config import TRAIN_SEASONS
from src.dst.data import build_dst_data

RARE_TARGETS = ["def_safeties", "def_tds", "def_blocked_kicks", "special_teams_tds"]


def describe(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    n = len(y)
    mean = float(y.mean())
    var = float(y.var(ddof=1))
    dispersion = var / mean if mean > 0 else float("nan")
    p_zero_obs = float((y == 0).mean())
    p_zero_poisson = math.exp(-mean) if mean > 0 else 1.0
    zero_excess = p_zero_obs - p_zero_poisson
    p_ge1 = float((y >= 1).mean())
    p_ge2 = float((y >= 2).mean())
    p_ge3 = float((y >= 3).mean())
    max_y = float(y.max())
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "dispersion": dispersion,
        "p_zero_obs": p_zero_obs,
        "p_zero_poisson": p_zero_poisson,
        "zero_excess": zero_excess,
        "p_ge1": p_ge1,
        "p_ge2": p_ge2,
        "p_ge3": p_ge3,
        "max": max_y,
    }


def recommend(stats: dict) -> str:
    d = stats["dispersion"]
    ze = stats["zero_excess"]
    p_ge2 = stats["p_ge2"]
    mean = stats["mean"]

    # If almost never fires >= 2, the count is essentially a Bernoulli outcome.
    if p_ge2 < 0.02:
        return (
            f"BCE on (y>0) -- P(y>=2)={p_ge2:.3%} is negligible so the count is effectively binary; "
            f"regressing the count wastes capacity on a degenerate head."
        )

    # Dispersion ~ 1 and zero-excess ~ 0 -> Poisson fits; NLL is simpler than BCE reframe.
    if 0.8 <= d <= 1.2 and abs(ze) < 0.02:
        return (
            f"Poisson NLL (mean={mean:.4f}) -- dispersion={d:.2f} and zero-excess={ze:+.3f} match "
            f"Poisson; no need for BCE or NB."
        )

    if ze >= 0.05:
        return (
            f"Zero-inflated Poisson or BCE on (y>0) -- excess zeros "
            f"({ze:+.3f} above Poisson) suggest a two-part model beats plain NLL."
        )

    if d > 1.2:
        return (
            f"Negative-Binomial or Tweedie NLL -- dispersion={d:.2f} is overdispersed; "
            f"Poisson will under-penalize large residuals."
        )

    return (
        f"Keep Huber (manual-check case) -- dispersion={d:.2f}, zero-excess={ze:+.3f}, "
        f"P(y>=2)={p_ge2:.3%}."
    )


def main() -> None:
    print(f"Building DST data (train seasons = {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}) ...")
    df = build_dst_data()
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    print(f"  n_team_games (train) = {len(train):,}\n")

    print(
        f"{'target':<20} {'n':>6} {'mean':>8} {'var':>8} {'disp':>6} "
        f"{'P(0)obs':>8} {'P(0)poi':>8} {'zExc':>7} {'P>=1':>7} {'P>=2':>7} {'P>=3':>7} {'max':>5}"
    )
    print("-" * 110)
    rows = []
    for t in RARE_TARGETS:
        s = describe(train[t].values)
        rows.append((t, s))
        print(
            f"{t:<20} {s['n']:>6d} {s['mean']:>8.4f} {s['var']:>8.4f} {s['dispersion']:>6.2f} "
            f"{s['p_zero_obs']:>8.3%} {s['p_zero_poisson']:>8.3%} {s['zero_excess']:>+7.3f} "
            f"{s['p_ge1']:>7.3%} {s['p_ge2']:>7.3%} {s['p_ge3']:>7.3%} {int(s['max']):>5d}"
        )

    print("\n=== Recommendations ===")
    for t, s in rows:
        print(f"- {t}: {recommend(s)}")

    # One-line summary ready for paste into TODO.md / PR description.
    print("\n=== Summary line ===")
    summary = {t: recommend(s).split(" --")[0] for t, s in rows}
    print(" | ".join(f"{t}={v}" for t, v in summary.items()))


if __name__ == "__main__":
    main()
