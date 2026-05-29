"""Historical-data diff: ``nflreadpy`` vs ``nfl_data_py``.

Both packages read from the nflverse data repo, so *in theory* the historical
data should be identical. In practice they do **not** pull the same release
assets: ``nfl_data_py`` (deprecated, pinned at 0.3.3) reads the **legacy**
release schema, while ``nflreadpy`` (the maintained successor, polars-based)
reads the **current** release schema. The most consequential case is weekly
player stats:

* ``nfl_data_py.import_weekly_data`` → legacy ``player_stats`` release
  (columns ``recent_team``, ``interceptions``, ``sacks``, ``sack_yards``).
* ``nflreadpy.load_player_stats`` → new ``stats_player_week`` release
  (columns ``team``, ``passing_interceptions``, ``sacks_suffered``,
  ``sack_yards_lost``).

``src/data/loader.py`` already fetches the new release for 2025+ and renames
those four columns back to the legacy names, so today the pipeline trains on
**legacy-release values for ≤2024 and new-release values for 2025+**. A
migration to ``nflreadpy`` would unify every season on the new release —
which is only safe if the per-(player, week) *values* match. This harness
measures that, per source, across the full project season range.

For each of the nine sources the pipeline pulls, it loads the same seasons
from both packages and reports:

  1. row counts + join-key set diff (rows only in A / only in B),
  2. column-set diff (with the known rename map applied + auto-detected),
  3. dtype diffs on shared columns,
  4. value-level mismatches on the columns the pipeline actually consumes
     (numeric → mismatches beyond atol/rtol + delta stats + example keys;
     string → exact-mismatch count + examples),

and assigns a one-word verdict per source
(``IDENTICAL`` / ``RENAME-ONLY`` / ``DTYPE-ONLY`` / ``ROW-DELTA`` /
``VALUE-DELTA`` / ``SCHEMA-BREAK``).

Outputs:

  - stdout: one verdict line per source + an overall summary table.
  - ``analysis_output/nflreadpy_diff/{source}.md``: full per-source tables.
  - ``analysis_output/nflreadpy_diff/SUMMARY.md``: the verdict roll-up.

The pure comparison logic (:func:`compare_frames`) takes two in-memory frames
and is unit-tested on synthetic data (no network). Only :func:`main` touches
the network, and it never writes to ``data/raw/`` (no pipeline-cache
pollution): each package uses its own cache (``nfl_data_py`` fetches fresh;
``nflreadpy`` caches per its config).

PR #383 (merged 2026-05-29) already migrated the feed to ``nflreadpy`` behind the
``src/data/nfl_source.py`` shim and **removed ``nfl_data_py`` from the project**.
This script is the independent post-hoc validation of that migration's
data-equivalence (see ``nflreadpy_diff_findings.md``). Because ``nfl_data_py`` is
no longer a project dependency, install it ad-hoc to run the comparison:
``pip install nfl_data_py``. The module imports both packages lazily (inside
:func:`main`'s helpers), so the unit-test import smoke passes without either.

Usage::

    pip install nfl_data_py                                       # removed from project in #383
    python -m src.analysis.analysis_nflreadpy_diff               # full range, all sources
    python -m src.analysis.analysis_nflreadpy_diff --sources weekly schedules
    python -m src.analysis.analysis_nflreadpy_diff --seasons 2018 2023 2024
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SEASONS  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis_output" / "nflreadpy_diff"

# Weekly-stats rename map: new-release (nflreadpy) -> legacy (nfl_data_py).
# Mirrors the harmonisation in src/data/loader.py:101-108. Single source of
# truth for the harness; if the loader's map changes, update both.
WEEKLY_RENAME: dict[str, str] = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}

# Columns the pipeline actually consumes per source (drives the value diff —
# we don't diff the ~390 incidental pbp columns, only what feeds features).
# Inventory traced from src/data/loader.py, src/data/redzone_pbp.py,
# src/k/data.py, src/dst/data.py.
_WEEKLY_VALUE = [
    "fantasy_points",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "targets",
    "carries",
    "attempts",
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]
_ROSTERS_VALUE = ["position", "team", "jersey_number"]
_SCHEDULES_VALUE = [
    "game_type",
    "gameday",
    "home_team",
    "away_team",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "div_game",
    "roof",
]
_SNAP_VALUE = ["offense_pct"]
_INJURY_VALUE = ["practice_status", "report_status"]
_DEPTH_VALUE = ["formation", "depth_team"]
_IDS_VALUE = ["pfr_id"]
# DST loader (src/dst/data.py:428) reads team_logo_espn, not team_logo_url.
_TEAMS_VALUE = ["team_logo_espn"]
# pbp: union of red-zone (src/data/redzone_pbp.py) + kicker (src/k/data.py).
_PBP_VALUE = [
    "yardline_100",
    "play_type",
    "pass_attempt",
    "rusher_player_id",
    "receiver_player_id",
    "posteam",
    "field_goal_attempt",
    "kick_distance",
    "field_goal_result",
    "kicker_player_id",
    "extra_point_attempt",
    "extra_point_result",
    "fg_prob",
    "wind",
    "temp",
    "roof",
    "surface",
    "qtr",
    "score_differential",
]


# ---------------------------------------------------------------------------
# Pure comparison core (no network — unit-tested)
# ---------------------------------------------------------------------------


@dataclass
class SourceDiff:
    """Result of comparing one source across the two packages."""

    name: str
    a_rows: int = 0
    b_rows: int = 0
    a_cols: list[str] = field(default_factory=list)
    b_cols: list[str] = field(default_factory=list)
    col_a_only: list[str] = field(default_factory=list)
    col_b_only: list[str] = field(default_factory=list)
    col_shared: list[str] = field(default_factory=list)
    applied_renames: dict[str, str] = field(default_factory=dict)
    key_cols: list[str] = field(default_factory=list)
    n_keys_a: int = 0
    n_keys_b: int = 0
    n_keys_shared: int = 0
    keys_a_only: int = 0
    keys_b_only: int = 0
    a_dupe_keys: int = 0
    b_dupe_keys: int = 0
    dtype_diffs: dict[str, tuple[str, str]] = field(default_factory=dict)
    value_results: dict[str, dict] = field(default_factory=dict)
    value_cols_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if self.value_cols_missing:
            return "SCHEMA-BREAK"
        if any(r.get("n_mismatch", 0) > 0 for r in self.value_results.values()):
            return "VALUE-DELTA"
        if self.keys_a_only > 0 or self.keys_b_only > 0:
            return "ROW-DELTA"
        if self.dtype_diffs:
            return "DTYPE-ONLY"
        if self.applied_renames or self.col_a_only or self.col_b_only:
            return "RENAME-ONLY"
        return "IDENTICAL"

    def summary_line(self) -> str:
        if self.error:
            return f"[       ERROR] {self.name:<12s} {self.error}"
        bits = [f"rows A={self.a_rows:,} B={self.b_rows:,}"]
        if self.key_cols:
            bits.append(
                f"keys shared={self.n_keys_shared:,} "
                f"A-only={self.keys_a_only:,} B-only={self.keys_b_only:,}"
            )
        if self.col_a_only or self.col_b_only:
            bits.append(f"cols A-only={len(self.col_a_only)} B-only={len(self.col_b_only)}")
        n_mm = sum(1 for r in self.value_results.values() if r.get("n_mismatch", 0) > 0)
        if self.value_results:
            bits.append(f"value-delta {n_mm}/{len(self.value_results)} consumed cols")
        if self.dtype_diffs:
            bits.append(f"dtype-diffs {len(self.dtype_diffs)}")
        if self.value_cols_missing:
            bits.append(f"MISSING consumed: {self.value_cols_missing}")
        return f"[{self.verdict():>12s}] {self.name:<12s} " + " | ".join(bits)


def _norm_key_col(s: pd.Series) -> pd.Series:
    """Stringify a key column so int/float/str representations align.

    Numeric keys (season, week, play_id) are routed through nullable ``Int64``
    so ``2012`` (int) and ``2012.0`` (downcast float) collapse to ``"2012"``.
    Non-integer numerics or mixed values fall back to a plain string cast.
    """
    if pd.api.types.is_numeric_dtype(s):
        try:
            return pd.to_numeric(s, errors="coerce").astype("Int64").astype("string")
        except (ValueError, TypeError):
            return s.astype("string")
    return s.astype("string")


def _key_series(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    parts = [_norm_key_col(df[c]).fillna("∅") for c in key_cols]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep="|")
    return out


def _compare_value_column(
    a_col: pd.Series, b_col: pd.Series, atol: float, rtol: float, max_examples: int
) -> dict:
    """Compare one consumed column (already aligned by shared key)."""
    a_num = pd.to_numeric(a_col, errors="coerce")
    b_num = pd.to_numeric(b_col, errors="coerce")
    # "numeric" if coercion doesn't manufacture many new NaNs on either side.
    a_lost = int((a_num.isna() & a_col.notna()).sum())
    b_lost = int((b_num.isna() & b_col.notna()).sum())
    is_numeric = a_lost <= 0.02 * max(len(a_col), 1) and b_lost <= 0.02 * max(len(b_col), 1)

    n = int(len(a_col))
    if is_numeric:
        close = np.isclose(
            a_num.to_numpy(dtype=float),
            b_num.to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        mismatch_mask = ~close
        delta = (a_num - b_num).abs()
        both_present = a_num.notna() & b_num.notna()
        only_one_nan = int((a_num.isna() ^ b_num.isna()).sum())
        ex_idx = delta.where(both_present).sort_values(ascending=False).head(max_examples).index
        examples = [
            {"key": str(k), "a": _round(a_num.loc[k]), "b": _round(b_num.loc[k])}
            for k in ex_idx
            if mismatch_mask[a_num.index.get_loc(k)]
        ]
        sub = delta[both_present & mismatch_mask]
        return {
            "kind": "numeric",
            "n_compared": n,
            "n_mismatch": int(mismatch_mask.sum()),
            "frac_mismatch": round(float(mismatch_mask.mean()), 6),
            "n_one_sided_nan": only_one_nan,
            "max_abs_delta": _round(sub.max()) if len(sub) else 0.0,
            "mean_abs_delta": _round(sub.mean()) if len(sub) else 0.0,
            "examples": examples,
        }

    a_str = a_col.astype("string")
    b_str = b_col.astype("string")
    # Fill NaN with a collision-proof sentinel and compare as object arrays.
    # Avoids pandas Kleene-NA (``"string" == "string"`` returns <NA>, which
    # can't be used as a boolean mask). both-NaN → equal; one-sided NaN →
    # mismatch; both-present → exact compare.
    sentinel = "\x00__NA__\x00"
    a_filled = a_str.fillna(sentinel).to_numpy(dtype=object)
    b_filled = b_str.fillna(sentinel).to_numpy(dtype=object)
    mismatch_mask = a_filled != b_filled
    ex_keys = a_str.index[mismatch_mask][:max_examples]
    examples = [
        {"key": str(k), "a": _str_or_none(a_str.loc[k]), "b": _str_or_none(b_str.loc[k])}
        for k in ex_keys
    ]
    return {
        "kind": "string",
        "n_compared": n,
        "n_mismatch": int(mismatch_mask.sum()),
        "frac_mismatch": round(float(mismatch_mask.mean()), 6),
        "examples": examples,
    }


def _round(v: object, ndigits: int = 4) -> object:
    try:
        if pd.isna(v):
            return None
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


def _str_or_none(v: object) -> object:
    return None if pd.isna(v) else str(v)


def compare_frames(
    a: pd.DataFrame,
    b: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    rename_b_to_a: dict[str, str] | None = None,
    name: str = "",
    atol: float = 1e-6,
    rtol: float = 1e-3,
    max_examples: int = 10,
    a_cols_override: list[str] | None = None,
    b_cols_override: list[str] | None = None,
) -> SourceDiff:
    """Diff two frames on key set, columns, dtypes, and consumed-column values.

    ``rename_b_to_a`` renames B's columns to A's naming *before* comparison.
    ``a_cols_override`` / ``b_cols_override`` let the caller supply the full
    column list when ``a`` / ``b`` were column-subset for memory (used by pbp).
    Pure: no network, no file IO.
    """
    rename_b_to_a = rename_b_to_a or {}
    b_before = set(b.columns)
    applied = {k: v for k, v in rename_b_to_a.items() if k in b_before}
    b = b.rename(columns=applied)

    a_cols = list(a_cols_override) if a_cols_override is not None else list(a.columns)
    if b_cols_override is None:
        b_cols = list(b.columns)  # b already renamed above
    else:
        # apply the rename to the supplied full-schema list too (pbp full-schema
        # diff); applied.get(c, c) is a no-op when no rename matched.
        b_cols = [applied.get(c, c) for c in b_cols_override]
    sa, sb = set(a_cols), set(b_cols)

    diff = SourceDiff(
        name=name,
        a_rows=int(len(a)),
        b_rows=int(len(b)),
        a_cols=sorted(a_cols),
        b_cols=sorted(b_cols),
        col_a_only=sorted(sa - sb),
        col_b_only=sorted(sb - sa),
        col_shared=sorted(sa & sb),
        applied_renames=applied,
        key_cols=list(key_cols),
    )

    have_keys = all(c in a.columns for c in key_cols) and all(c in b.columns for c in key_cols)
    if not have_keys:
        diff.notes.append(f"key cols {key_cols} not present in both frames; skipped key/value diff")
        return diff

    ka = _key_series(a, key_cols)
    kb = _key_series(b, key_cols)
    diff.a_dupe_keys = int(ka.duplicated().sum())
    diff.b_dupe_keys = int(kb.duplicated().sum())
    set_a, set_b = set(ka), set(kb)
    diff.n_keys_a = len(set_a)
    diff.n_keys_b = len(set_b)
    shared = set_a & set_b
    diff.n_keys_shared = len(shared)
    diff.keys_a_only = len(set_a - set_b)
    diff.keys_b_only = len(set_b - set_a)

    # dtype diffs on shared columns present in the actual (possibly subset) frames
    for c in sorted(set(a.columns) & set(b.columns)):
        da, db = str(a[c].dtype), str(b[c].dtype)
        if da != db:
            diff.dtype_diffs[c] = (da, db)

    if not shared:
        diff.notes.append("no shared join keys; value diff skipped")
        return diff
    if diff.a_dupe_keys or diff.b_dupe_keys:
        diff.notes.append(
            f"duplicate keys (A={diff.a_dupe_keys:,}, B={diff.b_dupe_keys:,}); "
            "kept first per key for value diff"
        )

    a_idx = a.assign(__k=ka).drop_duplicates("__k").set_index("__k")
    b_idx = b.assign(__k=kb).drop_duplicates("__k").set_index("__k")
    shared_list = sorted(shared)
    a_al = a_idx.reindex(shared_list)
    b_al = b_idx.reindex(shared_list)

    for col in value_cols:
        in_a, in_b = col in a.columns, col in b.columns
        if in_a and in_b:
            diff.value_results[col] = _compare_value_column(
                a_al[col], b_al[col], atol, rtol, max_examples
            )
        elif in_a != in_b:
            side = "A-only" if in_a else "B-only"
            diff.value_cols_missing.append(f"{col} ({side})")
        # in neither: not a consumed col here, ignore silently
    return diff


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _diff_to_markdown(d: SourceDiff) -> str:
    lines = [f"# {d.name}", "", f"**Verdict:** `{d.verdict()}`", ""]
    if d.error:
        lines += [f"**ERROR:** {d.error}", ""]
        return "\n".join(lines)
    lines += [
        "## Rows / keys",
        "",
        f"- rows: A={d.a_rows:,}, B={d.b_rows:,}",
        f"- key `{d.key_cols}`: shared={d.n_keys_shared:,}, "
        f"A-only={d.keys_a_only:,}, B-only={d.keys_b_only:,}",
        f"- duplicate keys: A={d.a_dupe_keys:,}, B={d.b_dupe_keys:,}",
        "",
        "## Columns",
        "",
        f"- applied renames (B→A): {d.applied_renames or '—'}",
        f"- only in A ({len(d.col_a_only)}): {d.col_a_only or '—'}",
        f"- only in B ({len(d.col_b_only)}): {d.col_b_only or '—'}",
        "",
    ]
    if d.value_cols_missing:
        lines += [f"- **consumed columns missing on one side:** {d.value_cols_missing}", ""]
    if d.dtype_diffs:
        lines += ["## Dtype diffs (shared columns)", "", "| column | A | B |", "|---|---|---|"]
        lines += [f"| {c} | {a} | {b} |" for c, (a, b) in sorted(d.dtype_diffs.items())]
        lines += [""]
    if d.value_results:
        lines += [
            "## Consumed-column value diff (on shared keys)",
            "",
            "| column | kind | n | mismatch | frac | max|Δ| | mean|Δ| | 1-sided NaN |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for col, r in d.value_results.items():
            lines.append(
                f"| {col} | {r['kind']} | {r['n_compared']:,} | {r['n_mismatch']:,} | "
                f"{r['frac_mismatch']:.4f} | {r.get('max_abs_delta', '—')} | "
                f"{r.get('mean_abs_delta', '—')} | {r.get('n_one_sided_nan', '—')} |"
            )
        lines += [""]
        for col, r in d.value_results.items():
            if r["n_mismatch"] and r["examples"]:
                lines += [
                    f"### Example mismatches: `{col}`",
                    "",
                    "| key | A | B |",
                    "|---|---|---|",
                ]
                lines += [f"| {e['key']} | {e['a']} | {e['b']} |" for e in r["examples"]]
                lines += [""]
    if d.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in d.notes] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Network pulls (only reached from main)
# ---------------------------------------------------------------------------


def _to_pandas(df: object) -> pd.DataFrame:
    """polars -> pandas (nflreadpy returns polars); pass through if already pandas."""
    to_pd = getattr(df, "to_pandas", None)
    return to_pd() if callable(to_pd) else df  # type: ignore[return-value]


@dataclass
class SourceSpec:
    name: str
    fetch_a: Callable[[list[int]], pd.DataFrame]
    fetch_b: Callable[[list[int]], pd.DataFrame]
    key_cols: list[str]
    value_cols: list[str]
    rename_b_to_a: dict[str, str] = field(default_factory=dict)
    clamp: Callable[[list[int]], list[int]] = lambda s: s


def _build_specs() -> dict[str, SourceSpec]:
    import nfl_data_py as nfl
    import nflreadpy as nr

    le2024 = lambda s: [y for y in s if y <= 2024]  # noqa: E731
    ge2012 = lambda s: [y for y in s if y >= 2012]  # noqa: E731

    return {
        "weekly": SourceSpec(
            "weekly",
            lambda s: nfl.import_weekly_data(s),
            lambda s: _to_pandas(nr.load_player_stats(s)),
            ["player_id", "season", "week"],
            _WEEKLY_VALUE,
            WEEKLY_RENAME,
            le2024,  # nfl_data_py weekly caps at 2024
        ),
        "rosters": SourceSpec(
            "rosters",
            lambda s: nfl.import_seasonal_rosters(s),
            lambda s: _to_pandas(nr.load_rosters(s)),
            ["player_id", "season"],
            _ROSTERS_VALUE,
            # nflreadpy rosters key/name cols differ from nfl_data_py.
            {"gsis_id": "player_id", "full_name": "player_name"},
        ),
        "schedules": SourceSpec(
            "schedules",
            lambda s: nfl.import_schedules(s),
            lambda s: _to_pandas(nr.load_schedules(s)),
            ["game_id"],
            _SCHEDULES_VALUE,
        ),
        "snap_counts": SourceSpec(
            "snap_counts",
            lambda s: nfl.import_snap_counts(s),
            lambda s: _to_pandas(nr.load_snap_counts(s)),
            ["pfr_player_id", "season", "week"],
            _SNAP_VALUE,
            clamp=ge2012,
        ),
        "injuries": SourceSpec(
            "injuries",
            lambda s: nfl.import_injuries(s),
            lambda s: _to_pandas(nr.load_injuries(s)),
            ["gsis_id", "season", "week"],
            _INJURY_VALUE,
        ),
        "depth_charts": SourceSpec(
            "depth_charts",
            lambda s: nfl.import_depth_charts(s),
            lambda s: _to_pandas(nr.load_depth_charts(s)),
            ["gsis_id", "season", "week"],
            _DEPTH_VALUE,
            clamp=le2024,  # loader only uses nfl_data_py depth for ≤2024
        ),
        "ids": SourceSpec(
            "ids",
            lambda s: nfl.import_ids(),
            lambda s: _to_pandas(nr.load_ff_playerids()),
            ["gsis_id"],
            _IDS_VALUE,
        ),
        "teams": SourceSpec(
            "teams",
            lambda s: nfl.import_team_desc(),
            lambda s: _to_pandas(nr.load_teams()),
            ["team_abbr"],
            _TEAMS_VALUE,
        ),
    }


def _run_simple(
    spec: SourceSpec, seasons: list[int], atol: float, rtol: float, ex: int
) -> SourceDiff:
    clamped = spec.clamp(seasons)
    try:
        a = spec.fetch_a(clamped)
        b = spec.fetch_b(clamped)
    except Exception as e:  # network / schema / package errors — record, continue
        return SourceDiff(name=spec.name, error=f"{type(e).__name__}: {e}")
    return compare_frames(
        a, b, spec.key_cols, spec.value_cols, spec.rename_b_to_a, spec.name, atol, rtol, ex
    )


def _load_pbp_subset(
    fetch_year: Callable[[int], pd.DataFrame], seasons: list[int], keep: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Pull pbp one season at a time, subsetting to ``keep`` columns to bound
    memory. Returns (concatenated subset, full column list from the first
    successfully-loaded season).
    """
    parts: list[pd.DataFrame] = []
    full_cols: list[str] = []
    for yr in seasons:
        df = fetch_year(yr)
        if not full_cols:
            full_cols = list(df.columns)
        present = [c for c in keep if c in df.columns]
        parts.append(df[present].copy())
        del df
    subset = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return subset, full_cols


def _run_pbp(seasons: list[int], atol: float, rtol: float, ex: int) -> SourceDiff:
    import nfl_data_py as nfl
    import nflreadpy as nr

    key = ["game_id", "play_id"]
    keep = key + _PBP_VALUE
    try:
        a_sub, a_full = _load_pbp_subset(
            lambda yr: nfl.import_pbp_data([yr], downcast=True), seasons, keep
        )
        b_sub, b_full = _load_pbp_subset(lambda yr: _to_pandas(nr.load_pbp([yr])), seasons, keep)
    except Exception as e:
        return SourceDiff(name="pbp", error=f"{type(e).__name__}: {e}")
    return compare_frames(
        a_sub,
        b_sub,
        key,
        _PBP_VALUE,
        {},
        "pbp",
        atol,
        rtol,
        ex,
        a_cols_override=a_full,
        b_cols_override=b_full,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "weekly",
            "rosters",
            "schedules",
            "snap_counts",
            "injuries",
            "depth_charts",
            "ids",
            "teams",
            "pbp",
        ],
        help="Subset of sources to compare (default: all 9).",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    seasons = sorted(args.seasons)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Comparing nflreadpy vs nfl_data_py over seasons "
        f"{seasons[0]}–{seasons[-1]} | sources: {', '.join(args.sources)}\n"
    )

    specs = _build_specs()
    diffs: list[SourceDiff] = []
    for src in args.sources:
        print(f"… {src}", flush=True)
        if src == "pbp":
            d = _run_pbp(seasons, args.atol, args.rtol, args.max_examples)
        elif src in specs:
            d = _run_simple(specs[src], seasons, args.atol, args.rtol, args.max_examples)
        else:
            print(f"  unknown source '{src}', skipping")
            continue
        diffs.append(d)
        print("  " + d.summary_line())
        (OUT_DIR / f"{src}.md").write_text(_diff_to_markdown(d))

    print("\n" + "=" * 78 + "\nVERDICT SUMMARY\n" + "=" * 78)
    summary = [
        "# nflreadpy vs nfl_data_py — verdict summary",
        "",
        f"Seasons {seasons[0]}–{seasons[-1]}.",
        "",
        "| source | verdict | notes |",
        "|---|---|---|",
    ]
    for d in diffs:
        print("  " + d.summary_line())
        note = d.error or "; ".join(d.notes) or ""
        summary.append(f"| {d.name} | `{d.verdict()}` | {note} |")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(f"\n✔ wrote per-source tables + SUMMARY.md to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
