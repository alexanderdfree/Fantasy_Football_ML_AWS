"""Per-position calculated-fields audit.

Loads ``data/splits/train.parquet`` (general features already computed by the
SETUP.md first-time data pull), runs each skill position's
``add_*_specific_features`` pipeline, and prints pre-fill NaN/Inf rates plus
basic distribution stats per whitelisted feature.

Special section for ``depth_chart_rank``: distribution per position + a
non-determinism surface-area count on the raw depth-chart cache (the share
of player-week groups with >1 row, which the previous ``agg('last')`` was
non-deterministic over).

K and DST features are pre-computed on dedicated team/kicker datasets rather
than the splits, so this script reports on them only at the depth-chart /
share-of-rows level — running their full pipelines is out of scope for a
read-only audit.

Usage::

    python scripts/audit_features.py [--splits-dir DIR] [--raw-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QB.qb_data import filter_to_qb  # noqa: E402
from QB.qb_features import add_qb_specific_features, get_qb_feature_columns  # noqa: E402
from RB.rb_data import filter_to_rb  # noqa: E402
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns  # noqa: E402
from shared.weather_features import merge_schedule_features  # noqa: E402
from TE.te_data import filter_to_te  # noqa: E402
from TE.te_features import add_te_specific_features, get_te_feature_columns  # noqa: E402
from WR.wr_data import filter_to_wr  # noqa: E402
from WR.wr_features import add_wr_specific_features, get_wr_feature_columns  # noqa: E402


def _column_stats(s: pd.Series) -> dict:
    """NaN%, Inf%, min/max/mean/std on a single column."""
    n = len(s)
    if n == 0 or not pd.api.types.is_numeric_dtype(s):
        return {
            "nan_pct": float("nan"),
            "inf_pct": float("nan"),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    nan_pct = float(s.isna().mean() * 100)
    inf_mask = np.isinf(s.to_numpy(dtype=float, copy=False, na_value=np.nan))
    inf_pct = float(inf_mask.mean() * 100)
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) == 0:
        return {
            "nan_pct": nan_pct,
            "inf_pct": inf_pct,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "nan_pct": nan_pct,
        "inf_pct": inf_pct,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _audit_position(
    name: str,
    filter_fn,
    add_features_fn,
    get_cols_fn,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run a position's add_features pipeline on the train split and return stats."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    pos_train = filter_fn(train_df).copy()
    print(f"  rows: {len(pos_train):,}")

    # Mirror build_position_features's first step so weather/Vegas features
    # are present before add_features_fn runs (otherwise they'd show up as
    # "missing" in the report even though the real pipeline adds them).
    merge_schedule_features(pos_train, label="train")

    # add_*_specific_features expects a (train, val, test) triple. For the
    # audit we only need train; pass empty stubs that share the train schema
    # so any column accesses inside the function don't blow up.
    empty = pos_train.iloc[:0].copy()
    pos_train, _, _ = add_features_fn(pos_train, empty.copy(), empty.copy())

    feature_cols = get_cols_fn()
    missing = [c for c in feature_cols if c not in pos_train.columns]
    present = [c for c in feature_cols if c in pos_train.columns]
    print(f"  features audited: {len(present)} / {len(feature_cols)}")
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"  ⚠ {len(missing)} expected feature(s) MISSING from split: {preview}{suffix}")

    rows = []
    train_means = pos_train[present].mean(numeric_only=True)
    for col in present:
        stats = _column_stats(pos_train[col])
        rows.append(
            {
                "feature": col,
                **stats,
                "train_mean_NaN": pd.isna(train_means.get(col, np.nan)),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    high_nan = summary.nlargest(10, "nan_pct")
    print("\n  Top 10 features by NaN%:")
    print(f"    {'feature':<46s} {'nan%':>6s} {'inf%':>6s} {'mean':>10s}")
    for _, r in high_nan.iterrows():
        flag = "  🚨 train_mean=NaN" if r["train_mean_NaN"] else ""
        mean_s = f"{r['mean']:.3g}" if r["mean"] is not None else "—"
        print(
            f"    {r['feature'][:46]:<46s} {r['nan_pct']:6.2f} {r['inf_pct']:6.2f} {mean_s:>10s}{flag}"
        )

    all_nan = summary[summary["train_mean_NaN"]]
    if not all_nan.empty:
        print(
            f"\n  ⚠ {len(all_nan)} feature(s) entirely NaN in train_df "
            f"(would silently flow as 0 to model after the catch-all fillna):"
        )
        for f in all_nan["feature"]:
            print(f"    - {f}")

    return summary


def _audit_depth_chart(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_dir: Path,
) -> None:
    print(f"\n{'=' * 72}\nDEPTH CHART RANK AUDIT\n{'=' * 72}")
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if "depth_chart_rank" not in full.columns:
        print("  ⚠ depth_chart_rank not in splits — loader merge may have failed")
        return

    print("\n  Distribution per position (% rows at each rank):")
    print(f"    {'pos':<6}{'rank=1':>9}{'rank=2':>9}{'rank=3':>9}{'NaN':>9}{'rows':>12}")
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = full[full["position"] == pos]
        if len(sub) == 0:
            continue
        rank = sub["depth_chart_rank"]
        n = len(rank)
        pct1 = 100.0 * (rank == 1.0).sum() / n
        pct2 = 100.0 * (rank == 2.0).sum() / n
        pct3 = 100.0 * (rank == 3.0).sum() / n
        print(
            f"    {pos:<6}{pct1:8.2f}%{pct2:8.2f}%{pct3:8.2f}%"
            f"{rank.isna().mean() * 100:8.2f}%{n:>12,}"
        )

    print(
        "\n  ↑ rank=3 includes both real third-string AND the loader's "
        "fillna(3) default — high rank=3 % suggests merge misses or "
        "non-numeric depth_team coercion."
    )

    print("\n  Non-determinism surface area on the raw depth_charts cache:")
    chart_files = sorted(raw_dir.glob("depth_charts_*.parquet"))
    if not chart_files:
        print(f"    ⚠ No raw depth_charts cache found in {raw_dir}")
        return

    for f in chart_files:
        depth = pd.read_parquet(f)
        if "formation" not in depth.columns:
            print(f"    {f.name}: no 'formation' column — skipping")
            continue
        off = depth[depth["formation"] == "Offense"].copy()
        if not len(off):
            print(f"    {f.name}: 0 Offense rows")
            continue
        groups = off.groupby(["gsis_id", "season", "week"]).size()
        n_groups = len(groups)
        n_dupes = int((groups > 1).sum())
        max_dupes = int(groups.max()) if n_groups > 0 else 0
        dupe_pct = 100.0 * n_dupes / max(1, n_groups)
        print(
            f"    {f.name}: {n_groups:,} player-week groups; "
            f"{n_dupes:,} ({dupe_pct:.1f}%) have >1 row "
            f"(max {max_dupes}/group) → bug-1 fired on {dupe_pct:.1f}% of groups "
            f"before the agg='min' fix"
        )

        # Show how often the duplicates actually disagree on rank
        if n_dupes > 0:
            # numeric coerce + agg(min/max) per group; disagreement = max != min
            off["depth_team_num"] = pd.to_numeric(off["depth_team"], errors="coerce")
            agg = off.groupby(["gsis_id", "season", "week"])["depth_team_num"].agg(
                ["min", "max", "nunique"]
            )
            disagree = int((agg["min"] != agg["max"]).sum())
            disagree_pct = 100.0 * disagree / max(1, n_groups)
            print(
                f"      └ {disagree:,} ({disagree_pct:.1f}%) of those groups have "
                f"disagreeing depth_team values across rows (the only ones where "
                f"old 'last' vs new 'min' produces a different answer)"
            )


def _load_splits(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run SETUP.md's first-time data pull first."
        )
    train_df = pd.read_parquet(train_path)
    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"
    val_df = pd.read_parquet(val_path) if val_path.exists() else train_df.iloc[:0]
    test_df = pd.read_parquet(test_path) if test_path.exists() else train_df.iloc[:0]
    return train_df, val_df, test_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
        help="Directory containing {train,val,test}.parquet (default: data/splits)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw nflverse caches (default: data/raw)",
    )
    args = parser.parse_args()

    print(f"Loading splits from {args.splits_dir}")
    train_df, val_df, test_df = _load_splits(args.splits_dir)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    _audit_position("QB", filter_to_qb, add_qb_specific_features, get_qb_feature_columns, train_df)
    _audit_position("RB", filter_to_rb, add_rb_specific_features, get_rb_feature_columns, train_df)
    _audit_position("WR", filter_to_wr, add_wr_specific_features, get_wr_feature_columns, train_df)
    _audit_position("TE", filter_to_te, add_te_specific_features, get_te_feature_columns, train_df)

    _audit_depth_chart(train_df, val_df, test_df, args.raw_dir)

    print(f"\n{'=' * 72}")
    print(
        "K, DST: features pre-computed on dedicated kicker/team datasets, not "
        "on these splits. Run their pipelines (run_k_pipeline.py / "
        "run_dst_pipeline.py) to inspect their feature distributions."
    )
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
