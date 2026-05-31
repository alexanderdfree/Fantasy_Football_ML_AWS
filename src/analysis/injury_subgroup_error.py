"""Injury / return-from-absence subgroup error analysis (operator-only CLI).

For each position, runs the production pipeline once and slices test-set
fantasy-point MAE by injury/return subgroups, to quantify whether the models are
materially worse on returning-from-absence or Questionable players — and how
large those subgroups even are. **Measurement only; changes no model.**

Why this exists: the injury (``game_status`` / ``practice_status``) and return
(``days_rest`` / ``is_returning_from_absence``) features feed every position
(both the linear/tree models and the attention NN's static branch), but their
marginal value was never measured on the affected subgroup. This produces the
*tracked subgroup metric* the draft-capital probe lacked (TODO.md
``[TESTED, REJECTED]``): a feature can be benchmark-flat overall yet move a
~15%-of-rows subgroup, so an overall-MAE delta alone can't settle "does it help".

Structural caveats baked into the 2025 test split (read the printed n's, not just
the MAEs):

* ``Out`` (game_status 0.0) and ``Doubtful`` (0.1) player-weeks are **absent**
  from played-game rows — preprocessing drops no-play rows, so a player who is
  actually Out has no row. The designation signal therefore collapses to
  ``Questionable`` (0.5), ~3% of rows. Out/Doubtful effects are *unmeasurable*
  here, not merely small.
* ``days_rest`` is ``week.diff()*7`` clipped to [4, 21], so it is discrete:
  7 (normal week), 14 (one week missed), 21 (two-plus weeks missed). The
  ``returning`` subgroup (``is_returning_from_absence == 1``) is exactly
  ``days_rest >= 14`` and is the headline (~15% of rows).

The full pipeline runs on every invocation (``run()`` per position), so this
module is gated behind ``if __name__ == "__main__"`` — importing it must NOT fire
the pipeline. It reuses the pure helpers from
:mod:`src.analysis.analysis_rb_lgbm_disagreement` rather than re-deriving metrics.

Usage:
    python -m src.analysis.injury_subgroup_error                 # all six positions
    python -m src.analysis.injury_subgroup_error QB RB WR TE     # subset
    python -m src.analysis.injury_subgroup_error --no-history    # skip JSON artifact
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.analysis.analysis_rb_lgbm_disagreement import (  # noqa: E402  reuse pure helpers
    ACTUAL,
    available_models,
    per_model_metrics,
)
from src.shared.benchmark_utils import (  # noqa: E402
    append_to_history,
    get_git_hash,
    utc_now_iso,
)

ANALYSIS_NAME = "injury_subgroup_error"
HISTORY_DIR = os.path.join("benchmark_history", "analysis")
DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

# A subgroup with fewer rows than this has an MAE too noisy to act on; flagged in
# the table and the verdict so a small-n slice is never read as a clean signal.
SMALL_N = 40

# Injury/return context columns retained on test_df (merged in loader.py /
# derived in engineer.py). Not every position carries every one — RB drops
# is_returning_from_absence as r=0.934-collinear with days_rest — so each
# subgroup names the column it needs and is skipped when that column is absent.
RETURN_FLAG = "is_returning_from_absence"
DAYS_REST = "days_rest"
GAME_STATUS = "game_status"

# (key, pretty label, required column or None, mask builder). The arrow-indented
# rows are sub-slices of ``returning``; ``settled`` + ``returning`` partition the
# rows, as do ``healthy`` + ``questionable``.
SUBGROUP_SPECS: list[tuple[str, str, str | None, Callable[[pd.DataFrame], pd.Series]]] = [
    ("global", "GLOBAL (all rows)", None, lambda df: pd.Series(True, index=df.index)),
    (
        "settled",
        "settled — no week gap (days_rest=7)",
        RETURN_FLAG,
        lambda df: df[RETURN_FLAG] == 0,
    ),
    ("returning", "returning — any gap >1 wk", RETURN_FLAG, lambda df: df[RETURN_FLAG] == 1),
    (
        "ret_1wk",
        "  ↳ returning, exactly 1 wk missed (days_rest=14)",
        DAYS_REST,
        lambda df: df[DAYS_REST] == 14,
    ),
    (
        "ret_2wk",
        "  ↳ returning, 2+ wk missed (days_rest>14)",
        DAYS_REST,
        lambda df: df[DAYS_REST] > 14,
    ),
    ("healthy", "healthy (game_status=1.0)", GAME_STATUS, lambda df: df[GAME_STATUS] >= 1.0),
    (
        "questionable",
        "Questionable+ (game_status<1.0)",
        GAME_STATUS,
        lambda df: df[GAME_STATUS] < 1.0,
    ),
]


def _flag(n: int) -> str:
    if n == 0:
        return "  [EMPTY — not measurable in test]"
    if n < SMALL_N:
        return f"  [small-n<{SMALL_N}: noisy, do not act on alone]"
    return ""


def analyze_position(pos: str) -> dict:
    """Run ``pos``'s production pipeline, print per-subgroup MAE tables, and
    return a JSON-serializable record of every subgroup's per-model metrics."""
    run = importlib.import_module(f"src.{pos.lower()}.run_pipeline").run
    result = run()
    df = result["test_df"].copy()
    models = available_models(df)
    n_total = len(df)

    print(f"\n{'=' * 80}")
    print(
        f"{pos}: injury / return subgroup error   "
        f"(test player-weeks: {n_total}, mean actual FP: {df[ACTUAL].mean():.2f})"
    )
    print(f"models present: {', '.join(models)}")
    print("=" * 80)

    record: dict[str, dict] = {}
    for key, label, needed, mask_fn in SUBGROUP_SPECS:
        if needed is not None and needed not in df.columns:
            continue
        sub = df[mask_fn(df)]
        n = len(sub)
        pct = 100.0 * n / n_total if n_total else 0.0
        metrics = per_model_metrics(sub, models)
        record[key] = {"label": label.strip(), "n": n, "pct": round(pct, 1), "models": metrics}

        print(f"\n--- {label}   (n={n}, {pct:.1f}% of test){_flag(n)} ---")
        if n == 0:
            continue
        print(f"    {'model':14}{'MAE':>9}{'bias':>9}{'RMSE':>9}")
        for name, m in metrics.items():
            print(f"    {name:14}{m['mae']:9.3f}{m['bias']:9.3f}{m['rmse']:9.3f}")

    _print_verdict(pos, record)
    return {
        "position": pos,
        "n_test": n_total,
        "models_present": list(models),
        "subgroups": record,
    }


def _print_verdict(pos: str, record: dict) -> None:
    """One-line-per-comparison summary: best model's MAE on each affected
    subgroup vs global, so 'is the model worse on these players, and how rare are
    they' is readable without scanning every table."""
    g = record.get("global", {}).get("models", {})
    if not g:
        return
    best = min(g, key=lambda k: g[k]["mae"])
    g_mae = g[best]["mae"]
    print(f"\n  → best model (lowest GLOBAL MAE): {best} = {g_mae:.3f}")
    for key, name in [("returning", "returning"), ("questionable", "Questionable")]:
        sub = record.get(key)
        if not sub or sub["n"] == 0:
            continue
        bm = sub["models"][best]["mae"]
        tag = _flag(sub["n"]).strip()
        tag = f"  {tag}" if tag else ""
        print(
            f"    {best} MAE on {name:12}: {bm:7.3f}  "
            f"(Δ vs global {bm - g_mae:+.3f}, n={sub['n']}, {sub['pct']:.1f}% of test){tag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "positions",
        nargs="*",
        default=None,
        help="positions to analyze (default: all six)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="skip writing the JSON subgroup-metric artifact to benchmark_history/analysis/",
    )
    args = parser.parse_args()

    positions = [p.upper() for p in (args.positions or DEFAULT_POSITIONS)]
    records = [analyze_position(p) for p in positions]

    if not args.no_history:
        now = utc_now_iso()
        git_hash = get_git_hash()
        entry = {
            "run_id": f"{now}_{git_hash}_{ANALYSIS_NAME}",
            "timestamp": now,
            "git_hash": git_hash,
            "kind": "analysis",
            "name": ANALYSIS_NAME,
            "results": records,
        }
        append_to_history(HISTORY_DIR, entry)


if __name__ == "__main__":
    main()
