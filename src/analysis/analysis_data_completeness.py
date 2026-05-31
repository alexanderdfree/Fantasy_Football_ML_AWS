"""Verify training-data completeness for all six positions.

Runs the **real production feature build** for each position and measures NaN
on the exact model inputs (``get_feature_columns()``) and targets
(``POSITION_CONFIG.targets``) — not the raw split columns. This is the
ground-truth completeness check because:

  - ``src/shared/feature_build.py:81`` raises ``KeyError`` if any whitelisted
    feature column is missing, so a successful run proves column coverage.
  - Imputation reliance is measured **after** each position's intentional
    ``fill_nans_fn`` but **before** the catch-all ``fillna(0)`` at
    ``feature_build.py:110``, exposing which features rely on the blunt impute.

**Data-flow families**:
  - QB/RB/WR/TE: read ``data/splits/{train,val,test}.parquet`` → filter →
    ``compute_targets`` → ``build_position_features`` (schedule merge +
    ``add_specific_features`` + ``fill_nans_fn``).
  - K / DST: self-load from ``data/raw`` cache exactly as their
    ``src/{pos}/run_pipeline.py::run()`` does (``load_data``/``build_data`` →
    ``compute_targets`` → ``compute_features``), then the same shared prep.

**NaN policy**: matches ``feature_build.py:110`` — imputing NaN→0 is the
production contract, not a gap. The "imputation reliance" count tells you which
features are structurally zero-filled (cold-start players, coverage gaps) vs.
genuinely observed.

**Important caveat on stale local splits**: reading the on-disk
``data/splits/*.parquet`` against the current ``get_feature_columns()``
whitelist is only valid when the splits are fresh (built from the current
``src/features/engineer.py``). Stale splits flag columns as "missing" that
are actually present in a fresh build — a false positive. Rebuild splits
first: ``python -m src.analysis.analysis_data_completeness --rebuild`` or
run ``temporal_split(build_features(preprocess(load_raw_data())))`` manually.

Outputs (``analysis_output/`` is gitignored):
  - ``analysis_output/data_completeness.json``  (machine-readable per-position)

Usage::

    # profile against current on-disk splits (fast, may flag stale-split gaps)
    python -m src.analysis.analysis_data_completeness

    # rebuild splits first, then profile (slow but authoritative)
    python -m src.analysis.analysis_data_completeness --rebuild
"""

import argparse
import json
import os

import pandas as pd

from src.config import SPLITS_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.shared.registry import get_inference_spec
from src.shared.team_box_score import merge_team_box_score_features
from src.shared.weather_features import merge_schedule_features

OUT_DIR = "analysis_output"
OUT_JSON = os.path.join(OUT_DIR, "data_completeness.json")

ALL_POS = ["QB", "RB", "WR", "TE", "K", "DST"]


def _rebuild_splits() -> None:
    """Rebuild data/splits from data/raw — the production recipe."""
    from src.data.loader import load_raw_data
    from src.data.preprocessing import preprocess
    from src.data.split import temporal_split
    from src.features.engineer import build_features

    print("Rebuilding splits from data/raw (NFLREADPY_CACHE should be off)...")
    df = build_features(preprocess(load_raw_data()))
    temporal_split(df)
    print(f"  splits written: {SPLITS_DIR}/")


def _build_frames(pos: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test frames exactly as each position's run() does."""
    if pos in ("QB", "RB", "WR", "TE"):
        return (
            pd.read_parquet(f"{SPLITS_DIR}/train.parquet"),
            pd.read_parquet(f"{SPLITS_DIR}/val.parquet"),
            pd.read_parquet(f"{SPLITS_DIR}/test.parquet"),
        )
    if pos == "K":
        from src.k.data import load_data, season_split
        from src.k.features import compute_features
        from src.k.targets import compute_targets

        k = load_data()
        k = compute_targets(k)
        compute_features(k)
        return season_split(k)
    if pos == "DST":
        from src.dst.data import build_data
        from src.dst.features import compute_features
        from src.dst.targets import compute_targets

        d = build_data()
        d = compute_targets(d)
        compute_features(d)
        return (
            d[d["season"].isin(TRAIN_SEASONS)].copy(),
            d[d["season"].isin(VAL_SEASONS)].copy(),
            d[d["season"].isin(TEST_SEASONS)].copy(),
        )
    raise ValueError(f"Unknown position: {pos}")


def _prep_preimpute(
    spec: dict,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Reproduce build_position_features up to (not incl.) the catch-all fillna(0)."""
    f = spec["filter_fn"]
    ptr, pva, pte = f(tr), f(va), f(te)
    ct = spec["compute_targets_fn"]
    ptr, pva, pte = ct(ptr), ct(pva), ct(pte)
    fcols = list(spec["get_feature_columns_fn"]())
    for df, lab in [(ptr, "train"), (pva, "val"), (pte, "test")]:
        merge_schedule_features(df, label=lab)
        merge_team_box_score_features(df, label=lab)
    ptr, pva, pte = spec["add_features_fn"](ptr, pva, pte)
    ptr, pva, pte = spec["fill_nans_fn"](ptr, pva, pte, spec["specific_features"])
    return ptr, pva, pte, fcols


def audit_position(pos: str) -> dict:
    """Run the full completeness audit for one position. Returns a result dict."""
    spec = get_inference_spec(pos)
    tr, va, te = _build_frames(pos)
    ptr, pva, pte, fcols = _prep_preimpute(spec, tr, va, te)
    targets = list(spec["targets"])

    present = [c for c in fcols if c in ptr.columns]
    missing = [c for c in fcols if c not in ptr.columns]

    pre = ptr[present].isna().mean().sort_values(ascending=False)
    n_with_nan = int((pre > 0).sum())
    worst = {c: round(float(v), 4) for c, v in pre.head(8).items() if v > 0}

    tgt_present = [t for t in targets if t in ptr.columns]
    tgt_null = ptr[tgt_present].isna().mean()
    tgt_bad = {c: round(float(v), 4) for c, v in tgt_null.items() if v > 0}

    return {
        "pos": pos,
        "rows": {"train": len(ptr), "val": len(pva), "test": len(pte)},
        "features": {"total": len(fcols), "present": len(present), "missing": missing},
        "imputation_reliance": {
            "n_features_pre_null": n_with_nan,
            "worst": worst,
        },
        "targets": {
            "list": targets,
            "max_null": float(tgt_null.max()) if len(tgt_null) else 0.0,
            "null_by_target": tgt_bad,
        },
        "complete": len(missing) == 0 and not tgt_bad,
    }


def main(rebuild: bool = False) -> None:
    if rebuild:
        _rebuild_splits()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    print("=" * 70)
    for pos in ALL_POS:
        try:
            r = audit_position(pos)
            results.append(r)
            verdict = (
                "✅ COMPLETE"
                if r["complete"]
                else f"⚠️  GAPS ({len(r['features']['missing'])} missing)"
            )
            rows = r["rows"]
            print(
                f"[{pos:3s}] {verdict:18s}  "
                f"tr/va/te={rows['train']}/{rows['val']}/{rows['test']}  "
                f"features {r['features']['present']}/{r['features']['total']}  "
                f"| {r['imputation_reliance']['n_features_pre_null']} rely-on-impute  "
                f"| target max-null={r['targets']['max_null']:.4f}"
            )
            if r["features"]["missing"]:
                print(f"        MISSING: {r['features']['missing'][:8]}")
            if r["imputation_reliance"]["worst"]:
                top = list(r["imputation_reliance"]["worst"].items())[:4]
                print(f"        top impute: {top}")
        except Exception as exc:
            import traceback

            results.append({"pos": pos, "error": str(exc)})
            print(f"[{pos:3s}] FAILED: {exc}")
            traceback.print_exc()
        print("-" * 70)

    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"  {r['pos']}: ERROR — {r['error']}")
        else:
            ok = r["complete"]
            print(
                f"  {r['pos']}: {'✅ complete' if ok else '⚠️ gaps'} — "
                f"{r['features']['present']}/{r['features']['total']} feature cols present, "
                f"{r['imputation_reliance']['n_features_pre_null']} rely on impute, "
                f"targets max-null={r['targets']['max_null']:.4f}"
            )

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written: {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training-data completeness audit for all 6 positions."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild data/splits from data/raw before auditing (authoritative but slow).",
    )
    args = parser.parse_args()
    main(rebuild=args.rebuild)
