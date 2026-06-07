"""Merge per-position scheduler-type ablation results from S3 into one report.

Each Batch container (``src/batch/train.py --ablation scheduler-type``) uploads
``s3://$S3_BUCKET/ablate_scheduler/{pos}/result.json`` (the dict returned by
``src.tuning.ablate_scheduler_type.run_position``). This aggregator downloads
them, prints the cross-position rollup, writes a merged JSON, and emits a
Markdown table to ``--summary-output`` (e.g. ``$GITHUB_STEP_SUMMARY``).

Missing per-position files are skipped with a warning (a Spot job may have
failed) so the rest still aggregate — mirrors ``aggregate_results.py``.

Usage:
    python -m src.tuning.aggregate_scheduler \
        --positions QB RB WR TE K DST \
        --bucket ff-predictor-training \
        --output scheduler_ablation_results.json \
        --summary-output "$GITHUB_STEP_SUMMARY"
"""

import argparse
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.tuning.ablate_scheduler_type import (  # noqa: E402
    SCHEDULERS,
    SHORT,
    print_cross_position,
)

DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "ff-predictor-training")
RESULT_PREFIX = "ablate_scheduler"


def _download(bucket: str, pos: str, local_dir: str, s3) -> dict | None:
    key = f"{RESULT_PREFIX}/{pos.upper()}/result.json"
    local = os.path.join(local_dir, f"{pos.upper()}_result.json")
    try:
        s3.download_file(bucket, key, local)
    except Exception as e:  # noqa: BLE001 — network/S3 boundary, skip-with-warning by design
        print(f"[warn] {pos}: could not download s3://{bucket}/{key}: {e}")
        return None
    with open(local) as f:
        return json.load(f)


def _md_table(summaries: list[dict]) -> str:
    head = (
        "| position | prod | "
        + " | ".join(SHORT[s] for s in SCHEDULERS)
        + " | winner | Δ vs prod | sentinel |"
    )
    sep = "|" + "---|" * (4 + len(SCHEDULERS))
    lines = [head, sep]
    for sm in summaries:
        agg = sm["aggregated"]
        cells = []
        for s in SCHEDULERS:
            if s in agg:
                mean = agg[s]["attn_fp_mae_mean"]
                std = agg[s]["attn_fp_mae_std"]
                star = "*" if s == sm["production_type"] else ""
                cells.append(f"{mean:.4f}±{std:.4f}{star}")
            else:
                cells.append("—")
        v = sm["verdict"]
        winner = SHORT.get(v.get("winner"), v.get("winner") or "n/a")
        margin = v.get("margin_vs_production")
        mstr = f"{margin:+.4f}" if margin is not None else "n/a"
        sentinel = "ok" if sm.get("sentinel_ok") else "**FAILED**"
        lines.append(
            f"| {sm['position']} | {SHORT.get(sm['production_type'], sm['production_type'])} | "
            + " | ".join(cells)
            + f" | {winner} | {mstr} | {sentinel} |"
        )
    note = (
        "\n\n_`*` = production type. Δ vs prod = MAE(prod) − MAE(winner); + means the winner "
        "beats production. attn FP MAE mean±std over seeds; eager (FF_CUDA_GRAPH=0) for "
        "bit-comparability. An UNTUNED alternative beating tuned production is the strong signal._"
    )
    return "\n".join(lines) + note


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--local-dir", default="./scheduler_ablation_results")
    p.add_argument("--output", default="scheduler_ablation_results.json")
    p.add_argument(
        "--summary-output", default=None, help="Markdown summary path (e.g. step summary)"
    )
    args = p.parse_args(argv)

    os.makedirs(args.local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    merged: dict[str, dict] = {}
    summaries: list[dict] = []
    for pos in args.positions:
        result = _download(args.bucket, pos, args.local_dir, s3)
        if result is None:
            continue
        merged[pos.upper()] = result
        if "summary" in result:
            summaries.append(result["summary"])

    if not summaries:
        raise SystemExit("No per-position result.json files downloaded — nothing to aggregate.")

    print_cross_position(summaries)

    with open(args.output, "w") as f:
        json.dump({"positions": list(merged), "results": merged}, f, indent=2)
    print(f"\nWrote merged results to {args.output} ({len(summaries)} positions)")

    if args.summary_output:
        with open(args.summary_output, "a") as f:
            f.write("## LR-scheduler-type ablation — cross-position rollup\n\n")
            f.write(_md_table(summaries))
            f.write("\n")
        print(f"Wrote Markdown summary to {args.summary_output}")


if __name__ == "__main__":
    main()
