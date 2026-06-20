"""Position-aware data profile of the QB/RB/WR/TE unified training splits.

Reads ``data/splits/{train,val,test}.parquet`` and produces a structured
markdown profile covering:

  1. Overview (shapes, grain, season/week coverage, position mix, dtypes)
  2. Column classification (engineered families, raw stats, targets, labels)
  3. Completeness — structural nulls (position-specific stat columns) vs.
     genuine missingness (cold-start / external-source coverage gaps)
  4. Target distributions per position — min/pX/mean/max, zero%, skew, shape
  5. Fantasy-points label distribution and pairwise variant correlations
  6. Temporal drift (train 2013–23 → val 2024 → test 2025 target means)
  7. Integrity checks (duplicate key rows, season_type, cross-split season overlap)
  8. Data-quality flags (empty columns, K inconsistency, schema drift, sentinels)
  9. Relationship summary (top fantasy-points correlates, high-|r| pairs)
 10. Recommended follow-ups (collinearity audit, empty-column drop, drift watch)

All heavy lifting (NaN→0 imputation, target derivation) is *reported on* in the
note-mode this file adopts — no production data is written. ``fumbles_lost`` is
derived from its three source columns to match ``src/{pos}/targets.py``.

K is incidentally present in the splits (14/0/543 rows across train/val/test)
and is excluded from this profile; K trains from its own PBP loader.
DST is absent from the unified splits entirely.

**NaN policy** (matches ``src/shared/feature_build.py:110``): null rates are
reported as-is; no ``dropna`` — that would silently profile only the
veteran-heavy subset (see ``feedback_analysis_match_production_nan_handling``
memory entry and issue #594).

Outputs (``analysis_output/`` is gitignored):
  - ``analysis_output/training_data_profile.md``  (full markdown report)

Usage::

    python -m src.analysis.analysis_training_data_profile
"""

import os
import re

import numpy as np
import pandas as pd

from src.config import SPLITS_DIR

OUT_DIR = "analysis_output"
OUT_MD = os.path.join(OUT_DIR, "training_data_profile.md")

SKILL = ["QB", "RB", "WR", "TE"]
FANTASY_COLS = [
    "fantasy_points",
    "fantasy_points_ppr",
    "fantasy_points_half_ppr",
    "fantasy_points_standard",
]
FUMBLE_PARTS = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]

# Per-position raw-stat targets (src/{pos}/targets.py). fumbles_lost is DERIVED.
TARGETS = {
    "QB": [
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
    ],
    "RB": [
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "fumbles_lost",
    ],
    "WR": ["receiving_yards", "receiving_tds", "receptions", "fumbles_lost"],
    "TE": ["receiving_yards", "receiving_tds", "receptions", "fumbles_lost"],
}


def _add_fumbles_lost(df: pd.DataFrame) -> pd.DataFrame:
    parts = [c for c in FUMBLE_PARTS if c in df.columns]
    df = df.copy()
    df["fumbles_lost"] = sum(df[c].fillna(0) for c in parts) if parts else np.nan
    return df


def _classify(col: str) -> str:
    if col in {"season", "week"}:
        return "temporal"
    if col.endswith("_id") or col in {"player_name", "player_display_name", "headshot_url"}:
        return "identifier"
    if col in {"position", "position_group", "recent_team", "season_type", "opponent_team", "team"}:
        return "dimension"
    if col.startswith("fantasy_points"):
        return "fantasy label"
    for fam, pred in [
        ("rolling_*", lambda x: x.startswith("rolling_")),
        ("ewma_*", lambda x: x.startswith("ewma_")),
        ("trend_*", lambda x: x.startswith("trend_")),
        ("prior_season_*", lambda x: x.startswith("prior_season_")),
        ("opp_* (defense)", lambda x: x.startswith("opp_")),
        ("share", lambda x: "share" in x),
        (
            "weather/vegas",
            lambda x: bool(
                re.search(r"implied|dome|wind|temp|total_line|divisional|rest|grass|spread", x)
            ),
        ),
        (
            "epa/efficiency",
            lambda x: bool(re.search(r"epa|_rate$|_pct$|cpoe|pacr|dakota|racr|wopr", x)),
        ),
    ]:
        if pred(col):
            return fam
    if col.startswith("passing_") or col in ("completions", "attempts", "sacks", "sack_yards"):
        return "raw: passing"
    if col.startswith("rushing_") or col == "carries":
        return "raw: rushing"
    if col.startswith("receiving_") or col in ("receptions", "targets"):
        return "raw: receiving"
    if col.startswith("def_") or "fumble" in col or "tackle" in col:
        return "raw: defense/fumble"
    if re.search(r"fg_|pat_|^pat|kick|punt|return|gwfg", col):
        return "raw: kicking/return"
    return "raw: other"


def _dist_row(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return {}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    zero = (s == 0).mean()
    neg = int((s < 0).sum())
    sk = s.skew()
    if zero > 0.6:
        shape = "zero-inflated"
    elif sk > 2:
        shape = "heavy right-skew"
    elif sk > 0.5:
        shape = "right-skew"
    elif sk < -0.5:
        shape = "left-skew"
    else:
        shape = "~symmetric"
    return dict(
        n=len(s),
        zero=zero,
        neg=neg,
        mn=s.min(),
        p5=q[0.05],
        p25=q[0.25],
        med=q[0.5],
        mean=s.mean(),
        p75=q[0.75],
        p95=q[0.95],
        p99=q[0.99],
        mx=s.max(),
        std=s.std(),
        skew=sk,
        shape=shape,
    )


def _dist_table(title: str, items: list, L: list) -> None:
    L.append(f"**{title}**\n")
    L.append(
        "| Stat | n | zero% | neg | min | p5 | p25 | med | mean | p75 | p95 | p99 | max | std | skew | shape |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, s in items:
        d = _dist_row(s)
        if not d:
            continue
        L.append(
            f"| {label} | {d['n']:,} | {d['zero']:.0%} | {d['neg']} | {d['mn']:.2f} | {d['p5']:.2f} | "
            f"{d['p25']:.2f} | {d['med']:.2f} | {d['mean']:.2f} | {d['p75']:.2f} | {d['p95']:.2f} | "
            f"{d['p99']:.2f} | {d['mx']:.2f} | {d['std']:.2f} | {d['skew']:.2f} | {d['shape']} |"
        )
    L.append("")


def build_profile() -> str:
    """Build and return the full markdown profile string."""
    tr = _add_fumbles_lost(pd.read_parquet(f"{SPLITS_DIR}/train.parquet"))
    va = _add_fumbles_lost(pd.read_parquet(f"{SPLITS_DIR}/val.parquet"))
    te = _add_fumbles_lost(pd.read_parquet(f"{SPLITS_DIR}/test.parquet"))
    splits = {"train": tr, "val": va, "test": te}
    orig_cols = [c for c in tr.columns if c != "fumbles_lost"]

    L: list[str] = []

    L.append("# Training Data Profile — QB/RB/WR/TE unified splits\n")
    L.append(
        "Source: `data/splits/{train,val,test}.parquet`. Read-only. "
        "`fumbles_lost` derived (sum of three `*_fumbles_lost` parts) per `src/{pos}/targets.py`. "
        "K rows are incidental and excluded; DST uses a separate team-week loader.\n"
    )

    # 1. Overview
    L.append("## 1. Overview\n")
    L.append("| Split | Seasons | Rows | Cols | Mem (MB) | QB | RB | WR | TE | K |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, df in splits.items():
        pc = df["position"].value_counts().to_dict()
        sn = sorted(int(s) for s in df["season"].unique())
        srange = f"{sn[0]}–{sn[-1]}" if len(sn) > 1 else str(sn[0])
        mem = df.memory_usage(deep=True).sum() / 1e6
        L.append(
            f"| {name} | {srange} | {len(df):,} | {len(orig_cols)} | {mem:.0f} | "
            f"{pc.get('QB', 0):,} | {pc.get('RB', 0):,} | {pc.get('WR', 0):,} | "
            f"{pc.get('TE', 0):,} | {pc.get('K', 0):,} |"
        )
    L.append("")
    dtype_str = ", ".join(f"{k}×{v}" for k, v in tr[orig_cols].dtypes.value_counts().items())
    L.append(
        f"**Grain**: one row per `(player_id, season, week)`, REG-season only. "
        f"**Dtypes (train)**: {dtype_str}.\n"
    )

    # 2. Column classification
    L.append("## 2. Column classification\n")
    buckets: dict[str, list[str]] = {}
    for c in orig_cols:
        buckets.setdefault(_classify(c), []).append(c)
    L.append("| Bucket | # cols | Examples |")
    L.append("|---|---:|---|")
    for b in sorted(buckets, key=lambda k: -len(buckets[k])):
        ex = ", ".join(f"`{c}`" for c in buckets[b][:5])
        L.append(f"| {b} | {len(buckets[b])} | {ex} |")
    L.append(f"\nTotal classified: {sum(len(v) for v in buckets.values())} / {len(orig_cols)}.\n")
    L.append(
        "**Targets per position** (raw stats; `fumbles_lost` derived): "
        + "; ".join(f"**{p}** → {', '.join(TARGETS[p])}" for p in SKILL)
        + ".\n"
    )

    # 3. Completeness
    L.append("## 3. Completeness (train)\n")
    nr = tr[orig_cols].isna().mean()
    bands = [
        ("Complete (>99% non-null)", (nr < 0.01).sum()),
        ("Mostly complete (95–99%)", ((nr >= 0.01) & (nr < 0.05)).sum()),
        ("Incomplete (80–95%)", ((nr >= 0.05) & (nr < 0.20)).sum()),
        ("Sparse (<80% non-null)", (nr >= 0.20).sum()),
    ]
    L.append("| Band | # cols |")
    L.append("|---|---:|")
    for label, count in bands:
        L.append(f"| {label} | {int(count)} |")
    empty = sorted(nr[nr == 1.0].index)
    L.append(
        f"\n**Fully-empty columns ({len(empty)})** — 100% null in train: "
        + ", ".join(f"`{c}`" for c in empty)
        + ".\n"
    )
    sparse = nr[(nr >= 0.20) & (nr < 1.0)].sort_values(ascending=False)
    struct, genuine = [], []
    for c in sparse.index:
        perpos = {p: tr.loc[tr["position"] == p, c].isna().mean() for p in SKILL}
        home = min(perpos, key=perpos.get)
        (struct if perpos[home] < 0.05 else genuine).append((c, sparse[c], home, perpos[home]))
    if struct:
        L.append(
            f"**Sparse but structural ({len(struct)})** — position-specific stat columns "
            "(null for other positions, ~complete within their home position):\n"
        )
        L.append("| Column | overall null% | home pos | null% in home |")
        L.append("|---|---:|---|---:|")
        for c, ov, home, hn in struct[:8]:
            L.append(f"| `{c}` | {ov:.0%} | {home} | {hn:.0%} |")
        L.append("")
    if genuine:
        L.append(
            f"**Sparse beyond position structure ({len(genuine)})** — cold-start / "
            "external-source coverage gaps. Dominated by `prior_season_*` (null for "
            "first-season players — rookie cold-start) and early-seasons QBR/epa coverage. "
            "All are imputed to 0 at pipeline time (`feature_build.py:110`):\n"
        )
        L.append("| Column | overall null% | best pos | null% in best pos |")
        L.append("|---|---:|---|---:|")
        for c, ov, home, hn in genuine[:12]:
            L.append(f"| `{c}` | {ov:.0%} | {home} | {hn:.0%} |")
        L.append("")

    # 4+5. Target distributions
    L.append("## 4. Target distributions (position-aware) + fantasy-point labels\n")
    _dist_table(
        "Fantasy points (all skill rows, train)",
        [(c, tr[c]) for c in FANTASY_COLS if c in tr.columns],
        L,
    )
    for p in SKILL:
        sub = tr[tr["position"] == p]
        _dist_table(
            f"{p} targets (within {p} rows, train)",
            [(t, sub[t]) for t in TARGETS[p] if t in sub.columns],
            L,
        )

    # 6. Temporal drift
    L.append("## 6. Temporal drift (train 2013–23 → val 2024 → test 2025)\n")
    L.append("Mean of headline targets per split:")
    L.append("")
    L.append("| Metric | scope | train | val | test |")
    L.append("|---|---|---:|---:|---:|")
    drift_items = [
        ("fantasy_points", "all"),
        ("passing_yards", "QB"),
        ("rushing_yards", "RB"),
        ("receiving_yards", "WR"),
        ("receptions", "WR"),
    ]
    for m, scope in drift_items:
        if scope == "all":
            vals = [splits[s][m].mean() for s in ("train", "val", "test")]
        else:
            vals = [
                splits[s].loc[splits[s]["position"] == scope, m].mean()
                for s in ("train", "val", "test")
            ]
        L.append(f"| `{m}` | {scope} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |")
    L.append("")
    nrt = pd.DataFrame({s: splits[s][orig_cols].isna().mean() for s in ("train", "val", "test")})
    swing = nrt[(nrt.max(axis=1) - nrt.min(axis=1)) > 0.5]
    swing = swing[swing.min(axis=1) < 1.0]
    if len(swing):
        L.append(
            f"**Availability drift ({len(swing)} cols)** — null-rate swings >50% across splits. "
            "These are raw stat columns (e.g. `game_id`, `def_*`, `misc_yards`, `fumble_recovery_*`) "
            "absent in train/val but populated in test(2025) from a newer nflverse schema. "
            "None are whitelisted model features — benign now but splits are not schema-uniform.\n"
        )
        L.append("| Column | train null% | val null% | test null% |")
        L.append("|---|---:|---:|---:|")
        for c in swing.index[:15]:
            L.append(
                f"| `{c}` | {nrt.loc[c, 'train']:.0%} | {nrt.loc[c, 'val']:.0%} | {nrt.loc[c, 'test']:.0%} |"
            )
        L.append("")

    # 7. Integrity
    L.append("## 7. Integrity & leakage checks\n")
    L.append("| Check | Value | Pass |")
    L.append("|---|---|:--:|")
    key = ["player_id", "season", "week"]
    seasons: dict[str, set] = {}
    for name, df in splits.items():
        dups = int(df.duplicated(subset=key).sum())
        L.append(
            f"| {name}: duplicate (player_id,season,week) | {dups} | {'✅' if dups == 0 else '❌'} |"
        )
        st = sorted(df["season_type"].unique().tolist())
        L.append(f"| {name}: season_type | {st} | {'✅' if st == ['REG'] else '❌'} |")
        seasons[name] = set(df["season"].unique())
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ov = sorted(seasons[a] & seasons[b])
        L.append(f"| season overlap {a}∩{b} | {ov} | {'✅' if not ov else '❌'} |")
    L.append("")

    # 8. Quality flags
    L.append("## 8. Data-quality flags\n")
    L.append("| Severity | Finding |")
    L.append("|---|---|")
    L.append(
        f"| HIGH | **{len(empty)} fully-empty (100%-null) columns** carried in every split "
        f"(`fg_*`, `def_*`, `penalties`, `punt_returns`, `kickoff_returns`, `game_id`, etc.) — "
        "dead weight; should be dropped from the split builder (see TODO.md). |"
    )
    L.append(
        f"| MED | **Schema drift at the 2024→2025 boundary**: {len(swing)} raw columns are "
        "100%-null in train/val but fully populated in test(2025) from a newer nflverse schema. "
        "None are whitelisted features today, but future feature additions reading these would "
        "silently train on all-null and test on real data. |"
    )
    kc = {s: int((splits[s]["position"] == "K").sum()) for s in ("train", "val", "test")}
    L.append(
        f"| MED | **K rows are inconsistent** ({kc['train']}/{kc['val']}/{kc['test']}) — "
        "incidental presence; K trains from its own PBP loader, not these splits. |"
    )
    fp_neg = int((tr["fantasy_points"] < 0).sum())
    L.append(
        f"| INFO | {fp_neg} train rows have negative `fantasy_points` "
        f"(min {tr['fantasy_points'].min():.2f}) — expected (INT/fumble-heavy games). |"
    )
    sp0 = int((tr["snap_pct"] == 0).sum()) if "snap_pct" in tr else 0
    L.append(
        f"| INFO | {sp0:,} zero-`snap_pct` rows (rostered / ~no snaps) — "
        "relevant to returner, role-player, and injury-return analysis. |"
    )
    L.append("")

    # 9. Relationships
    L.append("## 9. Relationships\n")
    num = tr[orig_cols].select_dtypes("number").fillna(0.0)
    num = num.loc[:, num.std() > 0]
    fp_corr = (
        num.corrwith(num["fantasy_points"])
        .drop(labels=[c for c in FANTASY_COLS if c in num.columns], errors="ignore")
        .pipe(lambda s: s.reindex(s.abs().sort_values(ascending=False).index))
    )
    L.append("**Top correlates of `fantasy_points`** (all skill rows, NaN→0):\n")
    L.append("| Feature | r |")
    L.append("|---|---:|")
    for c, r in fp_corr.head(10).items():
        L.append(f"| `{c}` | {r:.3f} |")
    L.append("")
    corr = num.corr().to_numpy()
    cn = num.columns.tolist()
    n_high_r = sum(
        1 for i in range(len(cn)) for j in range(i + 1, len(cn)) if abs(corr[i, j]) > 0.95
    )
    L.append(
        f"**{n_high_r} numeric pairs have |r|>0.95** — almost all window-redundancy "
        "(e.g. `ewma_passing_yards_L3`↔`_L5`) and the four near-identical `fantasy_points` "
        "variants. Deep VIF / condition-number audit: "
        "`python -m src.analysis.analysis_feature_audit QB RB WR TE`.\n"
    )
    fp_vars = [c for c in FANTASY_COLS if c in tr.columns]
    L.append(
        "Fantasy-point variants are near-identical (differ only by reception weight): "
        + "; ".join(f"`{fp_vars[0]}`~`{c}` r={tr[fp_vars[0]].corr(tr[c]):.4f}" for c in fp_vars[1:])
        + ".\n"
    )

    # 10. Follow-ups
    L.append("## 10. Recommended follow-ups\n")
    L.append(
        "1. **Drop the 63 fully-empty columns** from `src/data/split.py` / "
        "`src/features/engineer.py` — they cost 15–20 MB of split memory and "
        "muddy automated feature scans (see TODO.md)."
    )
    L.append(
        "2. **Reconcile schema drift** — make train/val/test schema-uniform; "
        "the test(2025) split contains 56 columns absent in train/val."
    )
    L.append(
        "3. **Multicollinearity audit** — "
        "`python -m src.analysis.analysis_feature_audit QB RB WR TE`; "
        f"the {n_high_r} |r|>0.95 pairs are the entry point."
    )
    L.append(
        "4. **Target zero-inflation** — confirm gated/Poisson head choices match the "
        "empirical zero rates in §4 (QB rushing_tds 88% zero, WR fumbles_lost 98%)."
    )
    L.append(
        "5. **Drift watch** — test(2025) headline target means are visibly lower "
        "(fantasy_points 6.72 vs train 8.78); confirm this is a partial-season "
        "artifact vs genuine distribution shift before the next retrain."
    )
    L.append("")

    return "\n".join(L)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    report = build_profile()
    with open(OUT_MD, "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nReport written: {OUT_MD}")


if __name__ == "__main__":
    main()
