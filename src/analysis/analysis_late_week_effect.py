"""Late-season (weeks 17-18) effect analysis.

Question being measured: *should* the season's final week(s) be excluded because
teams rest starters? This script answers it with data instead of intuition. It is
read-only — it touches no pipeline / model / feature / target code, so it triggers
no retrain (``src/analysis/`` is the right home for cohort diagnostics).

Two staged diagnostics, cheapest first:

Stage 1 — raw-label anomaly (NO model runs). Reads the splits directly; needs only
``fantasy_points / week / season / position / player_id`` (all present, none
feature-whitelist-dependent, so split staleness is irrelevant here).
  * "Final week" is **season-aware**: ``max(week)`` per season -> 17 for 2012-2020
    (17-game seasons) and 18 for 2021+ (18-game). A flat ``week <= 16`` filter would
    be era-wrong (it nukes 2012-2020's normal wk17 and post-2021's championship wk17).
  * Table A (composition): per position x {early, penultimate, final} -> n / mean /
    median fantasy points, i.e. the resting drop directly in the target. Pooled over
    all seasons, plus a 2025-test-only view (what the eval metric actually grades).
  * Table B (player-relative): among established players (>= N games that season),
    mean drop of final/penultimate-week points vs the player's *own* early-week
    baseline. Controls for roster composition; the cleaner resting fingerprint.
  * Era contrast: final-week drop split by era, to confirm the effect tracks the
    season-aware final week rather than a fixed integer.

Stage 2 — prediction degradation (model-based; run only if Stage 1 warrants). Uses
**production** predictions (no proxy). Per position: run the pipeline, slice
``result["test_df"]`` (2025) into wk1-16 / wk17 / wk18, and report pooled MAE plus
ranking (top-12 hit rate, Spearman) per model+bucket. This separates "the label is
noisy" from "it actually hurts predictions" — the tuned model may already absorb the
rest via ``snap_pct`` + Huber.

Usage:
    python -m src.analysis.analysis_late_week_effect --stage1
    python -m src.analysis.analysis_late_week_effect --stage1 --splits-dir /abs/path
    python -m src.analysis.analysis_late_week_effect --stage2 --positions RB WR TE QB
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import POSITIONS, SPLITS_DIR

TRUE_COL = "fantasy_points"
EARLY, PENULT, FINAL = "early", "penult", "final"
BUCKET_ORDER = [EARLY, PENULT, FINAL]
DEFAULT_ESTABLISHED_GAMES = 8
DEFAULT_TOP_K = 12

# Stage 1 reads the raw split's `fantasy_points`, which carries skill scoring only
# (passing/rushing/receiving). It is ~0 for K (kicking points live in fg_*/pat_*
# columns, not folded in) and DST is absent from this split entirely (separate data
# path). So the label-only stage is trustworthy for skill positions only; K and DST
# get their correct per-position totals from the pipeline in Stage 2.
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


# --------------------------------------------------------------------------- #
# Shared loading + season-aware bucketing
# --------------------------------------------------------------------------- #
def _load_splits(splits_dir: str | Path, which: tuple[str, ...]) -> pd.DataFrame:
    """Concatenate the requested split parquet(s), REG-only (mirrors the pipeline)."""
    frames = []
    for name in which:
        path = Path(splits_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing split: {path}")
        df = pd.read_parquet(path)
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_final_week_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``final_week`` / ``week_bucket`` / ``era`` columns (season-aware final week)."""
    df = df.copy()
    df["final_week"] = df.groupby("season")["week"].transform("max")
    df["week_bucket"] = np.where(
        df["week"] == df["final_week"],
        FINAL,
        np.where(df["week"] == df["final_week"] - 1, PENULT, EARLY),
    )
    df["era"] = np.where(df["season"] <= 2020, "2012-2020(17wk)", "2021-2025(18wk)")
    return df


def _ordered_positions(df: pd.DataFrame) -> list[str]:
    present = set(df["position"].dropna().unique())
    return [p for p in POSITIONS if p in present]


# --------------------------------------------------------------------------- #
# Stage 1 — label anomaly
# --------------------------------------------------------------------------- #
def _composition_table(df: pd.DataFrame, title: str) -> None:
    """Per position x bucket: n / mean / median fantasy points, with delta vs early."""
    print(f"\n  {title}")
    print(f"    {'pos':<5}{'bucket':<8}{'n':>7}{'mean':>9}{'median':>9}{'dMean':>9}{'dMean%':>8}")
    for pos in ["ALL", *_ordered_positions(df)]:
        sub = df if pos == "ALL" else df[df["position"] == pos]
        early_mean = sub.loc[sub["week_bucket"] == EARLY, TRUE_COL].mean()
        for bucket in BUCKET_ORDER:
            cell = sub[sub["week_bucket"] == bucket]
            if cell.empty:
                continue
            mean = cell[TRUE_COL].mean()
            median = cell[TRUE_COL].median()
            d = mean - early_mean
            dpct = 100.0 * d / early_mean if early_mean else float("nan")
            tag = "*" if bucket == FINAL else ""
            print(
                f"    {pos:<5}{bucket + tag:<8}{len(cell):>7}{mean:>9.2f}"
                f"{median:>9.2f}{d:>9.2f}{dpct:>7.1f}%"
            )


def _player_relative_drop(df: pd.DataFrame, established_games: int) -> pd.DataFrame:
    """Per (player, season) with >= N games: final/penult points minus own early baseline."""
    recs = []
    for (_pid, _season), g in df.groupby(["player_id", "season"], sort=False):
        if len(g) < established_games:
            continue
        fw = int(g["final_week"].iloc[0])
        early = g[g["week"] <= fw - 2]
        if early.empty:
            continue
        base = early[TRUE_COL].mean()
        pos = g["position"].iloc[0]
        fin = g[g["week"] == fw]
        if not fin.empty:
            recs.append((pos, FINAL, fin[TRUE_COL].mean() - base))
        pen = g[g["week"] == fw - 1]
        if not pen.empty:
            recs.append((pos, PENULT, pen[TRUE_COL].mean() - base))
    return pd.DataFrame(recs, columns=["position", "bucket", "drop"])


def _player_relative_table(drops: pd.DataFrame) -> None:
    print(
        "\n  Player-relative drop vs own early-week baseline "
        "(established players; negative = rested/declined)"
    )
    print(f"    {'pos':<5}{'bucket':<8}{'n_players':>10}{'meanDrop':>10}{'medianDrop':>12}")
    order = {b: i for i, b in enumerate(BUCKET_ORDER)}
    grouped = (
        drops.groupby(["position", "bucket"])["drop"].agg(["count", "mean", "median"]).reset_index()
    )
    grouped["pos_rank"] = grouped["position"].map(
        lambda p: POSITIONS.index(p) if p in POSITIONS else 99
    )
    grouped["b_rank"] = grouped["bucket"].map(order)
    for _, r in grouped.sort_values(["pos_rank", "b_rank"]).iterrows():
        print(
            f"    {r['position']:<5}{r['bucket']:<8}{int(r['count']):>10}"
            f"{r['mean']:>10.2f}{r['median']:>12.2f}"
        )


def _era_contrast_table(df: pd.DataFrame) -> None:
    """Final-week delta-mean split by era — validates the season-aware bucketing."""
    print("\n  Era contrast: final-week mean minus early-week mean (per position)")
    eras = sorted(df["era"].unique())
    print(f"    {'pos':<5}" + "".join(f"{e:>20}" for e in eras))
    for pos in _ordered_positions(df):
        sub = df[df["position"] == pos]
        cells = []
        for era in eras:
            e = sub[sub["era"] == era]
            early = e.loc[e["week_bucket"] == EARLY, TRUE_COL].mean()
            fin = e.loc[e["week_bucket"] == FINAL, TRUE_COL].mean()
            cells.append(f"{(fin - early):>20.2f}" if not np.isnan(fin) else f"{'-':>20}")
        print(f"    {pos:<5}" + "".join(cells))


def stage1_label_anomaly(splits_dir: str | Path, established_games: int) -> None:
    print("=" * 78)
    print("STAGE 1 — raw-label anomaly in the season's final weeks (no models)")
    print("  'final' = season-aware last week (17 for 2012-2020, 18 for 2021+).")
    print("  Reminder: wk18 is not a fantasy week; wk17 is the fantasy championship.")
    print("  Scope: QB/RB/WR/TE only (raw-split fantasy_points is skill-only; K≈0,")
    print("  DST absent) — K/DST get correct totals in Stage 2.")
    print("=" * 78)

    all_df = assign_final_week_buckets(_load_splits(splits_dir, ("train", "val", "test")))
    all_df = all_df[all_df["position"].isin(SKILL_POSITIONS)].copy()
    fw_by_season = all_df.groupby("season")["final_week"].first().to_dict()
    print("\n  Sanity — final (max) week by season:")
    print("    " + ", ".join(f"{s}:{w}" for s, w in sorted(fw_by_season.items())))

    _composition_table(all_df, "Table A1 — composition, ALL seasons 2012-2025")

    test_df = all_df[all_df["season"] == all_df["season"].max()]
    _composition_table(
        test_df, f"Table A2 — composition, TEST season {int(test_df['season'].iloc[0])} only"
    )

    _era_contrast_table(all_df)

    drops = _player_relative_drop(all_df, established_games)
    _player_relative_table(drops)


# --------------------------------------------------------------------------- #
# Stage 2 — prediction degradation (needs trained models)
# --------------------------------------------------------------------------- #
def _week_bucket_eval(week: int) -> str:
    if week <= 16:
        return "wk1-16"
    return "wk17" if week == 17 else "wk18"


def _week_hit_spear(wdf: pd.DataFrame, pred_col: str, top_k: int) -> tuple[float, float] | None:
    """Single-week top-k hit rate + Spearman; None if fewer than top_k rows."""
    from scipy.stats import spearmanr

    if len(wdf) < top_k:
        return None
    actual_top = set(wdf.nlargest(top_k, TRUE_COL)["player_id"])
    pred_top = set(wdf.nlargest(top_k, pred_col)["player_id"])
    hit = len(actual_top & pred_top) / top_k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(wdf[pred_col], wdf[TRUE_COL])
    return hit, corr


def _avg_weekly_ranking(df: pd.DataFrame, pred_col: str, top_k: int) -> tuple[float, float]:
    """Mean per-week top-k hit rate + Spearman over all weeks present in ``df``."""
    pairs = [hs for _w, wdf in df.groupby("week") if (hs := _week_hit_spear(wdf, pred_col, top_k))]
    if not pairs:
        return float("nan"), float("nan")
    corrs = [c for _, c in pairs if not np.isnan(c)]
    return (
        float(np.mean([h for h, _ in pairs])),
        float(np.mean(corrs)) if corrs else float("nan"),
    )


def _ranking_by_bucket(test_df: pd.DataFrame, pred_col: str, top_k: int) -> dict[str, dict]:
    """Per-week top-k hit rate + Spearman, averaged within each eval bucket."""
    per: dict[str, list[tuple[float, float]]] = {}
    for week, wdf in test_df.groupby("week"):
        hs = _week_hit_spear(wdf, pred_col, top_k)
        if hs is not None:
            per.setdefault(_week_bucket_eval(int(week)), []).append(hs)
    out = {}
    for bucket, vals in per.items():
        corrs = [c for _, c in vals if not np.isnan(c)]
        out[bucket] = {
            "hit": float(np.mean([h for h, _ in vals])),
            "spear": float(np.mean(corrs)) if corrs else float("nan"),
        }
    return out


def stage2_prediction_degradation(positions: list[str], top_k: int) -> None:
    from src.analysis.significance import pred_columns_from_test_df
    from src.shared.evaluation import compute_metrics
    from src.shared.registry import get_runner

    print("=" * 78)
    print("STAGE 2 — does the final-week anomaly hurt PREDICTIONS? (production models)")
    print("  Buckets: wk1-16 (baseline) | wk17 (championship) | wk18 (dead/rest)")
    print("=" * 78)

    buckets = ["wk1-16", "wk17", "wk18"]
    for pos in positions:
        print(f"\n### {pos}: running pipeline (trains models)…")
        result = get_runner(pos)()
        test_df = result["test_df"].copy()
        if TRUE_COL not in test_df.columns:
            print(f"  ! {pos}: no '{TRUE_COL}' in test_df; skipping.")
            continue
        test_df["wk_bucket"] = test_df["week"].map(_week_bucket_eval)
        pred_cols = pred_columns_from_test_df(test_df)
        if not pred_cols:
            print(f"  ! {pos}: no prediction columns on test_df; skipping.")
            continue

        # Best model = lowest baseline-bucket MAE (used for the per-position takeaway).
        base_mask = test_df["wk_bucket"] == "wk1-16"
        best_model = min(
            pred_cols,
            key=lambda n: compute_metrics(
                test_df.loc[base_mask, TRUE_COL], test_df.loc[base_mask, pred_cols[n]]
            )["mae"],
        )

        print(
            f"  {'model':<14}{'bucket':<8}{'n':>6}{'MAE':>8}{'dMAE':>8}"
            f"{'RMSE':>8}{'R2':>7}{'top' + str(top_k):>7}{'spear':>7}"
        )
        takeaways = {}
        for name, col in pred_cols.items():
            ranking = _ranking_by_bucket(test_df, col, top_k)
            base_mae = compute_metrics(
                test_df.loc[base_mask, TRUE_COL], test_df.loc[base_mask, col]
            )["mae"]
            star = " *" if name == best_model else ""
            for bucket in buckets:
                sub = test_df[test_df["wk_bucket"] == bucket]
                if sub.empty:
                    continue
                m = compute_metrics(sub[TRUE_COL], sub[col])
                rk = ranking.get(bucket, {"hit": float("nan"), "spear": float("nan")})
                print(
                    f"  {name + star:<14}{bucket:<8}{len(sub):>6}{m['mae']:>8.3f}"
                    f"{(m['mae'] - base_mae):>8.3f}{m['rmse']:>8.3f}{m['r2']:>7.2f}"
                    f"{rk['hit']:>7.2f}{rk['spear']:>7.2f}"
                )
                if name == best_model:
                    takeaways[bucket] = (m["mae"] - base_mae, rk["hit"])

        base_hit = takeaways.get("wk1-16", (0.0, float("nan")))[1]
        d17 = takeaways.get("wk17", (float("nan"), float("nan")))
        d18 = takeaways.get("wk18", (float("nan"), float("nan")))
        print(
            f"  -> best={best_model}: ΔMAE wk17={d17[0]:+.3f} wk18={d18[0]:+.3f} | "
            f"top{top_k} wk17={d17[1]:.2f} wk18={d18[1]:.2f} (base {base_hit:.2f})"
        )


# --------------------------------------------------------------------------- #
# Ablation — is the season's final week WORTH KEEPING in the training set?
# --------------------------------------------------------------------------- #
def _drop_final_week(df: pd.DataFrame) -> pd.DataFrame:
    """Drop each season's max week (season-aware: 17 pre-2021, 18 from 2021)."""
    fw = df.groupby("season")["week"].transform("max")
    return df[df["week"] != fw].copy()


def run_ablation(
    positions: list[str], splits_dir: str | Path, top_k: int, eval_max_week: int
) -> None:
    from src.analysis.significance import pred_columns_from_test_df
    from src.shared.evaluation import compute_metrics
    from src.shared.pipeline import _read_split
    from src.shared.registry import get_runner

    print("=" * 78)
    print("ABLATION — does the season's FINAL WEEK in TRAINING help or hurt?")
    print("  KEEP = train on all weeks | CUT = drop each season's final week (train+val).")
    print(f"  Both eval on the SAME test set, weeks 1-{eval_max_week} (deployment-relevant).")
    print("  Same seed. dMAE = CUT - KEEP: positive => cutting HURTS => keep the rows.")
    print("=" * 78)

    train = _read_split(f"{splits_dir}/train.parquet")
    val = _read_split(f"{splits_dir}/val.parquet")
    test = _read_split(f"{splits_dir}/test.parquet")
    train_cut, val_cut = _drop_final_week(train), _drop_final_week(val)
    dropped = len(train) - len(train_cut)
    print(
        f"\n  train rows: keep={len(train)} cut={len(train_cut)} "
        f"(-{dropped}, {100 * dropped / len(train):.1f}%) | val: keep={len(val)} cut={len(val_cut)}"
    )

    for pos in positions:
        runner = get_runner(pos)
        print(f"\n### {pos}: training KEEP (all weeks)…")
        res_keep = runner(train, val, test, seed=42)
        print(f"### {pos}: training CUT (no final week)…")
        res_cut = runner(train_cut, val_cut, test, seed=42)

        tk, tc = res_keep["test_df"], res_cut["test_df"]
        pred_cols = pred_columns_from_test_df(tk)
        sub_k = tk[tk["week"] <= eval_max_week]
        sub_c = tc[tc["week"] <= eval_max_week]
        best = min(
            pred_cols, key=lambda n: compute_metrics(sub_k[TRUE_COL], sub_k[pred_cols[n]])["mae"]
        )
        print(
            f"  eval weeks 1-{eval_max_week}: n_keep={len(sub_k)} n_cut={len(sub_c)} (should match)"
        )
        print(
            f"  {'model':<14}{'MAE_keep':>9}{'MAE_cut':>9}{'dMAE':>8}{'hit_keep':>9}{'hit_cut':>9}"
        )
        for name, col in pred_cols.items():
            mk = compute_metrics(sub_k[TRUE_COL], sub_k[col])["mae"]
            mc = compute_metrics(sub_c[TRUE_COL], sub_c[col])["mae"]
            hk = _avg_weekly_ranking(sub_k, col, top_k)[0]
            hc = _avg_weekly_ranking(sub_c, col, top_k)[0]
            star = " *" if name == best else ""
            print(f"  {name + star:<14}{mk:>9.3f}{mc:>9.3f}{(mc - mk):>+8.3f}{hk:>9.2f}{hc:>9.2f}")
        bk = compute_metrics(sub_k[TRUE_COL], sub_k[pred_cols[best]])["mae"]
        bc = compute_metrics(sub_c[TRUE_COL], sub_c[pred_cols[best]])["mae"]
        verdict = "KEEP better" if bc > bk else ("CUT better" if bc < bk else "tie")
        print(f"  -> best={best}: dMAE={bc - bk:+.4f} on wk1-{eval_max_week}  =>  {verdict}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", action="store_true", help="label anomaly (cheap, no models)")
    ap.add_argument("--stage2", action="store_true", help="prediction degradation (trains models)")
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="train KEEP vs CUT(final week) and compare on deployment weeks (trains 2x/pos)",
    )
    ap.add_argument(
        "--positions",
        nargs="*",
        default=POSITIONS,
        help="positions for stage2/ablation (default: all six). Stage1 always covers all.",
    )
    ap.add_argument("--splits-dir", default=SPLITS_DIR, help="dir with {train,val,test}.parquet")
    ap.add_argument("--established-games", type=int, default=DEFAULT_ESTABLISHED_GAMES)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument(
        "--eval-max-week",
        type=int,
        default=17,
        help="ablation: evaluate on test weeks 1..N (default 17, the fantasy-relevant slice)",
    )
    args = ap.parse_args()

    positions = [p.upper() for p in args.positions]
    # Default to the cheap stage when nothing is requested.
    run_s1 = args.stage1 or not (args.stage2 or args.ablation)
    if run_s1:
        stage1_label_anomaly(args.splits_dir, args.established_games)
    if args.stage2:
        stage2_prediction_degradation(positions, args.top_k)
    if args.ablation:
        run_ablation(positions, args.splits_dir, args.top_k, args.eval_max_week)


if __name__ == "__main__":
    main()
