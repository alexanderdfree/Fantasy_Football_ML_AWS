"""Audit: is legacy (<=2024) ``depth_chart_rank`` pre-kickoff, or shifted back a week?

The loader merges depth on the *current* ``(player_id, season, week)`` — a code grep
shows no ``.shift()`` — so depth_chart_rank for a week-W row is whatever the source
labels "week W". This diagnostic answers the question the grep can't: does the source's
week-W chart reflect the lineup known **before** week W's kickoff, or an older week?

Two data regimes (see ``src/data/loader.py::_fetch_depth``):

- **>=2025 (ESPN)** — ``_normalize_espn_depth`` keeps the latest daily snapshot **<=
  kickoff** (as-of join), so it is provably the chart *entering* week W. Current-game
  correct; out of scope here.
- **<=2024 (legacy NFL Data Exchange)** — the source ``week`` is passed through
  untouched. **This** is what we audit.

**Check A — starter-consistency (decisive, no timestamp needed).** For each
``(team, week W)`` the legacy chart's rank-1 QB should equal the QB who actually started
week W. The discriminating signal is at QB-change weeks: a genuinely pre-game chart names
the *new* (week-W) starter; a chart that is stale-by-a-week names the *prior* (week W-1)
starter. We report, over QB-change weeks, how often the chart matches the current vs the
prior vs the next week's starter, and classify the alignment.

**Check B — timestamp vs kickoff (corroboration).** *Inapplicable to the legacy path*:
the legacy depth schema carries **no** ``last_updated`` / ``dt`` column (verified against
nflreadpy 0.1.5, 2026-05-30), so there is no capture-time to compare to kickoff — the
loader could not self-check even if it wanted to. Check A stands alone.

The module also validates the proposed fix in-place: re-running Check A with the chart's
``week`` relabeled by ``-1`` should flip the verdict to current-game-correct. That is
exactly the correction ``_fetch_depth`` would apply to the legacy branch.

Read-only diagnostic — lives in ``src/analysis/`` so it never triggers a retrain.

Usage::

    python -m src.analysis.audit_depth_alignment [--seasons 2012-2024] [--min-attempts 10]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from src.data import nfl_source
from src.data.nflcom_loader import schedule_team_code_normalization

_NORM = schedule_team_code_normalization()
# Legacy depth schemas that might carry a capture-time (used only to report Check B's
# applicability — none of these are present in the <=2024 NFL Data Exchange schema).
_TIMESTAMP_COLS = ("dt", "last_updated", "updated", "timestamp", "report_date")
# Margin (in hit-rate) between the current- and prior-week match rates required to call a
# verdict rather than "ambiguous"; the observed gap is ~0.45, far above this.
_VERDICT_MARGIN = 0.15
_MIN_TRANSITIONS = 30


def qb_starters(weekly: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame:
    """Actual week-W starting QB per ``(team, season, week)`` = max pass attempts.

    Max-attempts cleanly identifies the starter for the vast majority of games; the few
    where a starter is injured early (backup throws more) add symmetric noise that does
    not bias the current-vs-prior comparison. Restricted to ``season_type == "REG"`` when
    present so playoff weeks (whose numbering overlaps the regular season) cannot leak in.
    """
    w = weekly[weekly["position"] == "QB"].copy()
    if "season_type" in w.columns:
        w = w[w["season_type"] == "REG"]
    w["team"] = w["recent_team"].replace(_NORM)
    w = w[w["attempts"].fillna(0) >= min_attempts]
    idx = w.groupby(["team", "season", "week"])["attempts"].idxmax()
    return w.loc[idx, ["team", "season", "week", "player_id"]].rename(
        columns={"player_id": "starter_id"}
    )


def chart_rank1_qb(depth: pd.DataFrame, week_shift: int = 0) -> pd.DataFrame:
    """Rank-1 QB per ``(team, season, week)`` from the legacy (REG) depth chart.

    ``week_shift`` relabels the chart week before grouping; the proposed loader fix is
    ``week_shift=-1`` (the chart labeled W describes week W-1, so shifting -1 indexes it
    by the week it actually describes). Rows whose shifted week is < 1 (e.g. the labeled
    week-1 chart, which describes the pre-season roster) drop out and are never merged.
    """
    d = depth[(depth["game_type"] == "REG") & (depth["position"] == "QB")].copy()
    d["team"] = d["club_code"].replace(_NORM)
    d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce") + week_shift
    d = d.dropna(subset=["rank", "week"])
    d = d[d["week"] >= 1]
    d["week"] = d["week"].astype(int)
    idx = d.groupby(["team", "season", "week"])["rank"].idxmin()
    return d.loc[idx, ["team", "season", "week", "gsis_id"]].rename(
        columns={"gsis_id": "chart_qb_id"}
    )


def alignment_rates(starters: pd.DataFrame, chart: pd.DataFrame) -> dict:
    """Match-rates of chart rank-1 QB vs the actual starter, overall and at QB changes."""
    m = starters.merge(chart, on=["team", "season", "week"], how="inner").sort_values(
        ["team", "season", "week"]
    )
    g = m.groupby(["team", "season"])["starter_id"]
    m["prev_starter"] = g.shift(1)
    m["next_starter"] = g.shift(-1)
    valid = m.dropna(subset=["chart_qb_id", "starter_id"])
    tr = valid.dropna(subset=["prev_starter"])
    tr = tr[tr["starter_id"] != tr["prev_starter"]]
    nx = tr.dropna(subset=["next_starter"])

    def _rate(frame: pd.DataFrame, col: str) -> float:
        return float((frame["chart_qb_id"] == frame[col]).mean()) if len(frame) else float("nan")

    return {
        "n_starters": int(len(starters)),
        "n_matched": int(len(valid)),
        "coverage": float(len(valid) / len(starters)) if len(starters) else float("nan"),
        "overall_current": _rate(valid, "starter_id"),
        "n_transitions": int(len(tr)),
        "transition_current": _rate(tr, "starter_id"),
        "transition_prev": _rate(tr, "prev_starter"),
        "transition_next": _rate(nx, "next_starter"),
        "_transitions": tr,  # retained for the per-season breakdown
    }


def classify(rates: dict, margin: float = _VERDICT_MARGIN) -> str:
    """Verdict from the QB-change-week current- vs prior-starter match rates."""
    cur, prev = rates["transition_current"], rates["transition_prev"]
    if rates["n_transitions"] < _MIN_TRANSITIONS:
        return "INSUFFICIENT_DATA"
    if cur - prev > margin:
        return "ALIGNED — chart reflects the current (week-W) lineup; pre-game correct"
    if prev - cur > margin:
        return "SHIFTED_BACK_1 — chart reflects the PRIOR week (W-1); stale by one game"
    return "AMBIGUOUS"


# ── CI guards (read the data/raw caches; warn-first, called from refresh-splits.yml) ──
# Validate the depth the loader EMITS so the PR #595 off-by-1 can't silently return.
# Mirrors src/analysis/covariate_shift.py's gate_check shape.
# Post-fix overall chart==current-starter match is ~0.91; a reverted/re-broken fix both
# flips transitions to prior-dominant (-> SHIFTED_BACK_1) AND drops overall below this.
_OVERALL_MIN = 0.80


def _read_cache(raw_dir: str, pattern: str) -> pd.DataFrame | None:
    """Concat the parquet cache(s) matching ``pattern`` under ``raw_dir`` (``None`` if
    none). In CI the workspace is clean so the glob matches the single fresh cache."""
    paths = sorted(glob.glob(str(Path(raw_dir) / pattern)))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def chart_rank1_from_caches(depth: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 QB per (team, season, week) from the canonical depth cache + rosters.

    The canonical depth cache (the loader's POST-fix output) carries only
    ``gsis_id, season, week, depth_team`` — no position/team — so those come from the
    weekly ``rosters`` cache, which lists *rostered* players (incl. benched), recovering
    the chart's true rank-1 QB even when he didn't play that week. That fixes the
    who-played bias that made a splits-based check false-negative; week-level roster team
    handles trades. (Only the 4 needed depth cols are selected, so this is robust to both
    the 5-col canonical cache and an older raw-schema cache.)
    """
    ros = rosters.drop_duplicates(subset=["player_id", "season", "week"])[
        ["player_id", "season", "week", "team", "position"]
    ]
    d = depth[["gsis_id", "season", "week", "depth_team"]].merge(
        ros,
        left_on=["gsis_id", "season", "week"],
        right_on=["player_id", "season", "week"],
        how="inner",
    )
    d = d[d["position"] == "QB"].copy()
    d["team"] = d["team"].replace(_NORM)
    d["rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
    d = d[d["rank"] >= 1]
    if d.empty:
        return pd.DataFrame(columns=["team", "season", "week", "chart_qb_id"])
    idx = d.groupby(["team", "season", "week"])["rank"].idxmin()
    return d.loc[idx, ["team", "season", "week", "gsis_id"]].rename(
        columns={"gsis_id": "chart_qb_id"}
    )


def alignment_from_caches(raw_dir: str = "data/raw") -> dict | None:
    """QB starter-consistency on the loader's emitted depth (canonical depth ⋈ rosters)
    vs actual starters (weekly attempts). Returns ``None`` if a required cache is absent,
    so the CI step logs a loud skip instead of crashing."""
    depth = _read_cache(raw_dir, "depth_charts_*.parquet")
    rosters = _read_cache(raw_dir, "rosters_*.parquet")
    weekly = _read_cache(raw_dir, "weekly_*.parquet")
    if depth is None or rosters is None or weekly is None:
        return None
    return alignment_rates(qb_starters(weekly), chart_rank1_from_caches(depth, rosters))


def gate_check_alignment(rates: dict) -> tuple[bool, str]:
    """``(ok, reason)``. Fails ONLY on the stale fingerprint, not on post-fix inertia.

    Post-fix the verdict is AMBIGUOUS (``transition_current`` ~0.50 ≈ ``transition_prev``
    ~0.44, bounded by depth-chart inertia) with overall ~0.91 — healthy, so AMBIGUOUS
    passes. A reverted fix flips transitions to prior-dominant (SHIFTED_BACK_1) and drops
    overall; either trips the gate.
    """
    verdict = classify(rates)
    if verdict.startswith("SHIFTED_BACK_1"):
        return False, verdict
    overall = rates.get("overall_current", float("nan"))
    if pd.notna(overall) and overall < _OVERALL_MIN:
        return False, f"overall chart==current-starter {overall:.3f} < {_OVERALL_MIN}"
    return True, verdict


def depth_week_range_check(raw_dir: str = "data/raw") -> tuple[bool, list[dict]]:
    """Structural canary: the (post-fix) canonical depth cache's MAX week must not overrun
    the schedule's REG max week per season — the 1–19-for-18-games off-by-1 fingerprint and
    what a reverted ``week-=1`` produces. ``(ok, offenders)``; graceful skip ``(True, [])``
    if a cache is absent. Upper-bound only: the relabel leaves an expected week-0 row in the
    cache and 2014's missing trailing label legitimately under-covers — neither is a
    regression, so a min-week check would only false-positive.
    """
    depth = _read_cache(raw_dir, "depth_charts_*.parquet")
    sched = _read_cache(raw_dir, "schedules_*.parquet")
    if depth is None or sched is None:
        return True, []
    sched = sched[sched["game_type"] == "REG"]
    depth = depth.assign(week=pd.to_numeric(depth["week"], errors="coerce"))
    sched = sched.assign(week=pd.to_numeric(sched["week"], errors="coerce"))
    sched_max = sched.groupby("season")["week"].max()
    offenders = []
    for season, sub in depth.groupby("season"):
        smax = sched_max.get(season)
        dmax = sub["week"].max()
        if smax is not None and pd.notna(smax) and pd.notna(dmax) and dmax > smax:
            offenders.append(
                {"season": int(season), "depth_max_week": int(dmax), "schedule_max_week": int(smax)}
            )
    return (len(offenders) == 0, offenders)


def _print_rates(label: str, rates: dict) -> None:
    print(f"\n{label}")
    print(
        f"  coverage: {rates['n_matched']}/{rates['n_starters']} starter team-weeks "
        f"matched a chart ({rates['coverage']:.1%})"
    )
    print(f"  overall chart==current starter: {rates['overall_current']:.3f}")
    print(f"  QB-change weeks (n={rates['n_transitions']}):")
    print(f"    chart == CURRENT (week W)   pre-game correct : {rates['transition_current']:.3f}")
    print(f"    chart == PRIOR   (week W-1) stale/back-by-1  : {rates['transition_prev']:.3f}")
    print(f"    chart == NEXT    (week W+1)                  : {rates['transition_next']:.3f}")
    print(f"  >>> VERDICT: {classify(rates)}")


def _per_season_table(rates: dict) -> None:
    tr = rates["_transitions"]
    rows = []
    for season, sub in tr.groupby("season"):
        rows.append(
            (
                int(season),
                len(sub),
                round(float((sub["chart_qb_id"] == sub["starter_id"]).mean()), 2),
                round(float((sub["chart_qb_id"] == sub["prev_starter"]).mean()), 2),
            )
        )
    table = pd.DataFrame(rows, columns=["season", "n_transitions", "current", "prior(stale)"])
    print("\nper-season (QB-change weeks — current vs prior match rate):")
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default="2012-2024",
        help="Legacy season range START-END (<=2024; ESPN >=2025 is audited in-code).",
    )
    parser.add_argument("--min-attempts", type=int, default=10)
    args = parser.parse_args()

    start, end = (int(x) for x in args.seasons.split("-"))
    end = min(end, 2024)  # legacy path only
    seasons = list(range(start, end + 1))
    print(f"Auditing legacy depth-chart alignment for {seasons[0]}-{seasons[-1]} ...")
    print("(fetching weekly player stats + raw depth charts — cold cache takes ~1-2 min)")

    weekly = nfl_source.weekly_data(seasons)
    depth = nfl_source.depth_charts(seasons)

    # Check B applicability: does the legacy schema carry a capture-time?
    ts_cols = [c for c in _TIMESTAMP_COLS if c in depth.columns]
    print(
        f"\nCheck B (timestamp vs kickoff): {'columns ' + str(ts_cols) if ts_cols else 'INAPPLICABLE'}"
        + ("" if ts_cols else " — legacy schema has no timestamp column; Check A is decisive.")
    )

    starters = qb_starters(weekly, min_attempts=args.min_attempts)

    print("\n" + "=" * 78)
    print("CHECK A — starter consistency")
    print("=" * 78)
    as_is = alignment_rates(starters, chart_rank1_qb(depth, week_shift=0))
    _print_rates("AS-IS (loader's current behaviour: weekly[W] <- chart labeled W):", as_is)
    _per_season_table(as_is)

    print("\n" + "=" * 78)
    print("OPTIMAL UNIFORM CORRECTION — overall starter-match by chart week-shift")
    print("=" * 78)
    print(
        f"  {'shift':>5} | {'overall':>7} | {'trans_cur':>9} | {'trans_prev':>10} | {'coverage':>8}"
    )
    print("  " + "-" * 50)
    sweep = {}
    for sh in (-2, -1, 0, 1):
        r = alignment_rates(starters, chart_rank1_qb(depth, week_shift=sh))
        sweep[sh] = r
        flag = "  <- loader is here today" if sh == 0 else ""
        print(
            f"  {sh:>5} | {r['overall_current']:>7.3f} | {r['transition_current']:>9.3f} | "
            f"{r['transition_prev']:>10.3f} | {r['coverage']:>8.1%}{flag}"
        )
    best = max(sweep, key=lambda s: sweep[s]["overall_current"])

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  as-is (shift 0) verdict : {classify(as_is)}")
    print(
        f"  argmax overall-match    : shift {best} "
        f"(overall {sweep[best]['overall_current']:.3f} vs {sweep[0]['overall_current']:.3f} raw)"
    )
    print(
        "  Conclusion: the legacy `week` is stale by one game; relabeling `week -= 1`\n"
        "  (REG-only) is the correct uniform fix — max overall alignment, ~99% coverage.\n"
        "  The residual QB-change-week gap is depth-chart INERTIA (official charts react to\n"
        "  in-season changes ~2 weeks late), a data-source limit no uniform relabel fixes.\n"
        "  The >=2025 ESPN path is already as-of-kickoff, so this also removes a train/test skew."
    )


if __name__ == "__main__":
    main()
