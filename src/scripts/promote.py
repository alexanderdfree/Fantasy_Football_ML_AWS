"""Manual rollback: rewrite ``models/{POS}/manifest.json`` to point ``current``
at any entry from ``history[]``.

Part A (PR #104) introduced automatic ``current → previous`` fallback in
``src.shared.model_sync._sync_one`` — that covers a single bad ship. This script
is the escape hatch for the "two consecutive bad ships" case documented in
#104: both ``current`` and ``previous`` broken, but an older entry in
``history[]`` is still good. Operator picks one, script atomically promotes
it via a new manifest.json put, and mirrors the legacy ``model.tar.gz`` so
pre-manifest consumers see the same bytes.

Usage:
    python scripts/promote.py --position WR --list
    python scripts/promote.py --position WR --to models/WR/history/...sha7/model.tar.gz
    python scripts/promote.py --position WR --to ... --dry-run

All state lives in ``src.shared.model_sync``'s manifest helpers — producer
(``src/batch/train.py``), consumer (``src/shared/model_sync.py``), and this operator
tool all share one schema. If you're editing the manifest shape, search for
call sites before landing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.model_sync import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    legacy_model_key,
    load_manifest,
    manifest_key,
    write_manifest,
)

_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


class PromotionError(Exception):
    """Raised when a promotion request cannot be satisfied (unknown key,
    missing S3 object, malformed manifest, etc). Tests assert on this type;
    ``main()`` catches it and exits with a human-readable message."""


def list_history(manifest: dict) -> str:
    """Return a human-readable listing of ``history[]`` newest-first,
    annotating which entry is ``current`` vs ``previous``.
    """
    history = manifest.get("history") or []
    cur_key = (manifest.get("current") or {}).get("key")
    prev_key = (manifest.get("previous") or {}).get("key")
    lines = ["history[] (newest-first):"]
    for i, key in enumerate(history):
        flags = []
        if key == cur_key:
            flags.append("← current")
        if key == prev_key:
            flags.append("← previous")
        flag_str = f"  {' '.join(flags)}" if flags else ""
        lines.append(f"  [{i}] {key}{flag_str}")
    if not history:
        lines.append("  (empty)")
    return "\n".join(lines)


def _parse_version_from_key(target_key: str) -> tuple[str, str]:
    """Pull ``uploaded_at`` + ``sha7`` out of a history key path.

    Keys are produced by ``src.shared.model_sync.new_history_key`` and have the
    shape ``{prefix}/{POS}/history/{ts}-{sha7}/model.tar.gz`` — the dir name
    before the filename is the only sha7 source we have post-facto (we can't
    recompute it without re-downloading the tarball).
    """
    parts = target_key.split("/")
    # Expected: [..., "history", "{ts}-{sha7}", "model.tar.gz"]
    if len(parts) < 3 or parts[-1] != "model.tar.gz" or parts[-3] != "history":
        return "", ""
    version_dir = parts[-2]
    if "-" not in version_dir:
        return "", ""
    uploaded_at, sha7 = version_dir.rsplit("-", 1)
    return uploaded_at, sha7


def build_promotion_manifest(
    old_manifest: dict,
    target_key: str,
    bucket: str,
    s3_client,
) -> dict:
    """Compute the new manifest that points ``current`` at ``target_key``.

    Rules:
      - ``target_key`` MUST appear in ``old_manifest["history"]`` (defensive;
        blocks typos and prevents operator from pointing at a random key
        that was never tracked).
      - The S3 object at ``target_key`` MUST exist (head_object). If it's
        been GC'd, refuse to promote — the manifest update would orphan
        the consumer.
      - ``previous`` becomes ``old.current`` so the next automatic fallback
        still has somewhere to go if the promoted artifact itself fails.
      - ``history`` stays unchanged — promotion is a pointer rewrite, not a
        reshuffle of the audit trail.
    """
    from botocore.exceptions import ClientError

    history = old_manifest.get("history") or []
    if target_key not in history:
        raise PromotionError(
            f"Target key not in manifest.history[]: {target_key}\n"
            f"  Available entries:\n    " + "\n    ".join(history or ["(empty)"])
        )
    try:
        head = s3_client.head_object(Bucket=bucket, Key=target_key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            raise PromotionError(
                f"Target key is in history[] but missing from S3 "
                f"(likely GC'd): s3://{bucket}/{target_key}"
            ) from e
        raise

    uploaded_at, sha7 = _parse_version_from_key(target_key)
    return {
        "schema_version": old_manifest.get("schema_version", MANIFEST_SCHEMA_VERSION),
        "current": {
            "key": target_key,
            "sha7": sha7,
            "bytes": head["ContentLength"],
            "uploaded_at": uploaded_at,
        },
        "previous": old_manifest.get("current"),
        "history": history,
    }


def promote(
    s3_client,
    bucket: str,
    prefix: str,
    position: str,
    target_key: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Promote ``target_key`` to ``current`` for ``position``. Returns the
    new manifest dict (whether or not it was actually written).

    On success, writes the new ``manifest.json`` and copies the target bytes
    into the legacy ``{prefix}/{POS}/model.tar.gz`` mirror so pre-manifest
    consumers see the same artifact. A dry-run returns the computed manifest
    without touching S3.
    """
    old = load_manifest(s3_client, bucket, prefix, position)
    if old is None:
        raise PromotionError(
            f"No manifest at s3://{bucket}/{manifest_key(prefix, position)} — "
            f"nothing to promote from. Either the bucket is pre-migration "
            f"(run a training job first) or the prefix is wrong."
        )
    new = build_promotion_manifest(old, target_key, bucket, s3_client)
    if dry_run:
        return new
    write_manifest(s3_client, bucket, prefix, position, new)
    legacy_k = legacy_model_key(prefix, position)
    s3_client.copy_object(
        Bucket=bucket,
        Key=legacy_k,
        CopySource={"Bucket": bucket, "Key": target_key},
    )
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manual rollback: rewrite models/{POS}/manifest.json to point "
            "'current' at any entry from history[]. See docstring in "
            "scripts/promote.py for the when/why."
        )
    )
    parser.add_argument("--position", required=True, choices=_POSITIONS)
    parser.add_argument(
        "--bucket",
        default="ff-predictor-training",
        help="S3 bucket. Defaults to the prod training bucket.",
    )
    parser.add_argument(
        "--prefix",
        default="models",
        help="Prefix under the bucket. Must match FF_MODEL_S3_PREFIX used by the consumer.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List history[] entries and exit.")
    group.add_argument("--to", metavar="KEY", help="Promote this history/ key to 'current'.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the new manifest without writing or copying.",
    )
    args = parser.parse_args(argv)

    import boto3

    s3 = boto3.client("s3")

    try:
        if args.list:
            old = load_manifest(s3, args.bucket, args.prefix, args.position)
            if old is None:
                print(
                    f"No manifest at s3://{args.bucket}/{manifest_key(args.prefix, args.position)}"
                )
                return 1
            print(list_history(old))
            return 0

        new = promote(s3, args.bucket, args.prefix, args.position, args.to, dry_run=args.dry_run)
        if args.dry_run:
            print("[dry-run] Would write manifest:")
            print(json.dumps(new, indent=2, sort_keys=True))
            return 0
        old_cur_key = "null"
        # Fetch-again just to report what the previous was — cheap + clearer log
        # than threading it back through ``promote``.
        latest = load_manifest(s3, args.bucket, args.prefix, args.position)
        if latest and (latest.get("previous") or {}).get("key"):
            old_cur_key = latest["previous"]["key"]
        print(f"Promoted {args.position}: current → {args.to}")
        print(f"  previous now: {old_cur_key}")
        print(
            f"  legacy mirror updated: "
            f"s3://{args.bucket}/{legacy_model_key(args.prefix, args.position)}"
        )
        return 0
    except PromotionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
