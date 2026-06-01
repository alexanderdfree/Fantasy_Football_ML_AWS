"""Aggregate per-position ``tune_nn`` results from S3 into one summary.

Each Batch tune job writes
``s3://$S3_BUCKET/tune_nn/{search-space-version}/{pos}/results.json`` with a
single-key dict like ``{"RB": {"best_trial": ..., "best_params": {...}, ...}}``.
This script pulls every per-position file and merges them into a unified
``tune_nn_results.json`` + a markdown summary table suitable for
``$GITHUB_STEP_SUMMARY``. Re-emits ``BEST_PARAMS_JSON_START/END``
markers so the existing log-extraction tooling (modeled on
``src/tuning/tune_lgbm.py``'s convention) keeps working.

Usage:
    python -m src.tuning.aggregate_results                    # uses $S3_BUCKET
    python -m src.tuning.aggregate_results --bucket my-bucket
    python -m src.tuning.aggregate_results --positions QB RB  # subset
    python -m src.tuning.aggregate_results --no-s3 --local-dir ./local-runs
        # for testing without S3 — reads tune_nn_*_results.json files locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning.tune_nn_storage import SEARCH_SPACE_VERSION, s3_key_prefix, s3_prefix  # noqa: E402

ALL_TUNABLE_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def _download_from_s3(
    bucket: str, positions: list[str], dest_dir: str, search_space_version: str
) -> list[str]:
    """Download per-position results.json from S3 into ``dest_dir``.

    Returns the list of local paths actually downloaded (missing positions
    are skipped with a warning, not an error — a single Spot failure
    shouldn't block aggregation of the rest).
    """
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")
    os.makedirs(dest_dir, exist_ok=True)
    downloaded: list[str] = []
    for pos in positions:
        key = f"{s3_key_prefix(pos, search_space_version)}/results.json"
        local_path = os.path.join(dest_dir, f"tune_nn_{pos.lower()}_results.json")
        try:
            s3.download_file(bucket, key, local_path)
            downloaded.append(local_path)
            print(f"[aggregate] downloaded s3://{bucket}/{key}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                print(
                    f"[aggregate] WARNING: no results at s3://{bucket}/{key} (job failed or pending)"
                )
                continue
            raise
    return downloaded


def _collect_local_files(local_dir: str, positions: list[str]) -> list[str]:
    """Find tune_nn_{pos}_results.json files in ``local_dir`` for the given positions."""
    found: list[str] = []
    for pos in positions:
        path = os.path.join(local_dir, f"tune_nn_{pos.lower()}_results.json")
        if os.path.exists(path):
            found.append(path)
        else:
            print(f"[aggregate] WARNING: no local file {path}")
    return found


def _merge_results(paths: list[str]) -> dict:
    """Merge per-position results dicts into one. Later files win on duplicates
    (shouldn't happen — each file is one position by construction).

    Malformed inputs (truncated S3 download, partial write, list-instead-of-
    dict shape) are warned about and skipped rather than aborting the whole
    aggregator. One Spot failure mid-upload shouldn't block the rest of the
    positions' results from making it into the merged JSON / GITHUB_STEP_SUMMARY.
    """
    merged: dict = {}
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[aggregate] WARNING: skipping {path}: invalid JSON ({e})")
            continue
        if not isinstance(data, dict):
            print(f"[aggregate] WARNING: skipping {path}: top-level is not a dict")
            continue
        for pos, entry in data.items():
            if pos in merged:
                print(f"[aggregate] WARNING: duplicate entry for {pos} in {path} — overwriting")
            merged[pos] = entry
    return merged


def _format_markdown_summary(merged: dict) -> str:
    """Markdown table suitable for ``$GITHUB_STEP_SUMMARY``: one row per
    position with best val_loss, trial count, and a compact param fingerprint.
    """
    if not merged:
        return "# Tune NN results\n\n_No per-position results found._\n"

    lines = [
        "# Tune NN results",
        "",
        "| Position | Best val_loss | Trial # | Trials | Elapsed (s) | d_model / n_heads / lr |",
        "|---|---|---|---|---|---|",
    ]
    for pos in sorted(merged):
        entry = merged[pos]
        params = entry.get("best_params", {})
        d_model = params.get("attn_d_model", "?")
        n_heads = params.get("attn_n_heads", "?")
        lr = params.get("attn_lr", "?")
        lr_str = f"{lr:.4g}" if isinstance(lr, (int, float)) else lr
        lines.append(
            f"| {pos} "
            f"| {entry.get('best_val_loss', 0):.4f} "
            f"| {entry.get('best_trial', '?')} "
            f"| {entry.get('n_trials', '?')} "
            f"| {entry.get('elapsed_seconds', '?')} "
            f"| {d_model} / {n_heads} / {lr_str} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=list(ALL_TUNABLE_POSITIONS),
        choices=list(ALL_TUNABLE_POSITIONS),
        help=f"Positions to aggregate (default: all {len(ALL_TUNABLE_POSITIONS)} tunable)",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET", "ff-predictor-training"),
        help=(
            "S3 bucket holding the per-position results.json under "
            f"{s3_prefix(SEARCH_SPACE_VERSION)}/{{pos}}/ "
            "(default: $S3_BUCKET or ff-predictor-training)"
        ),
    )
    parser.add_argument(
        "--search-space-version",
        default=os.environ.get("TUNE_NN_STORAGE_VERSION", SEARCH_SPACE_VERSION),
        help=(
            "Storage namespace under tune_nn/ to aggregate. Batch MPS+graph tuning "
            "passes scheduler_v2_mps_graph."
        ),
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 download; read tune_nn_*_results.json files from --local-dir.",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="Directory to download into (S3 mode) or read from (--no-s3 mode). Default: cwd.",
    )
    parser.add_argument(
        "--output",
        default="tune_nn_results.json",
        help="Path to write the merged JSON. Default: tune_nn_results.json",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path to also write the markdown summary (e.g. $GITHUB_STEP_SUMMARY).",
    )
    args = parser.parse_args()

    positions = [p.upper() for p in args.positions]

    if args.no_s3:
        paths = _collect_local_files(args.local_dir, positions)
    else:
        paths = _download_from_s3(args.bucket, positions, args.local_dir, args.search_space_version)

    merged = _merge_results(paths)

    # Atomic write so a crash mid-render can't leave a half-written file
    # consumed by downstream tooling.
    tmp = f"{args.output}.tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    os.replace(tmp, args.output)
    print(f"[aggregate] wrote merged JSON to {args.output}")

    summary = _format_markdown_summary(merged)
    print("\n" + summary)
    if args.summary_output:
        with open(args.summary_output, "a") as f:
            f.write(summary)
        print(f"[aggregate] appended markdown summary to {args.summary_output}")

    print("\n==== BEST_PARAMS_JSON_START ====")
    print(json.dumps(merged, indent=2, default=str))
    print("==== BEST_PARAMS_JSON_END ====")


if __name__ == "__main__":
    main()
