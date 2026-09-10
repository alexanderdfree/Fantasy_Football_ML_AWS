"""Roll back a retained model through the protected v3 publication protocol.

Copy an entry from history[] into a fresh releases/history/ key, then conditionally
advance both stable and current in models/{POS}/releases/manifest.json. Preserve
the source high-water mark so queued jobs cannot undo the operator's rollback.
A legacy position must complete a source-registered v3 publication first.

Usage:
    python -m src.scripts.promote --position WR --list
    python -m src.scripts.promote --position WR --to models/WR/releases/history/.../model.tar.gz
    python -m src.scripts.promote --position WR --to ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.artifact_publication import references, snapshot  # noqa: E402
from src.shared.model_sync import (  # noqa: E402
    HISTORY_KEEP_N,
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    manifest_key,
    new_history_key,
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
    shape ``{prefix}/{POS}/releases/history/{ts}-{uuid}-{sha7}/model.tar.gz`` — the dir name
    before the filename is the only sha7 source we have post-facto (we can't
    recompute it without re-downloading the tarball).

    Raises ``PromotionError`` if ``target_key`` doesn't match the expected
    shape. Previously this swallowed malformed keys and returned ``("", "")``
    or other garbage (e.g. ``("no-dashes", "here")`` for a key with a single
    hyphen in the version dir), which then flowed into the new manifest's
    ``current.uploaded_at`` / ``current.sha7`` fields silently. Failing
    loudly keeps the rollback contract honest: the operator promotes a
    real version or sees an error explaining why their input was rejected.
    """
    parts = target_key.split("/")
    # Expected: [..., "history", "{ts}-{sha7}", "model.tar.gz"]
    if len(parts) < 3 or parts[-1] != "model.tar.gz" or parts[-3] != "history":
        raise PromotionError(
            f"Malformed history key: {target_key!r}. Expected shape "
            f"'{{prefix}}/{{POS}}/history/{{ts}}-{{sha7}}/model.tar.gz' "
            f"(use --list to see valid entries)."
        )
    version_dir = parts[-2]
    # The ts portion itself contains hyphens (ISO date), so we need at least
    # one hyphen separating ts from sha7 AND a recognizable ts prefix. Use
    # rsplit so a hyphenated ts doesn't confuse the parse: only the LAST
    # ``-`` separates ts from sha7. Reject keys whose version dir has no
    # hyphen at all (no sha7 present) or whose sha7 portion is empty.
    if "-" not in version_dir:
        raise PromotionError(
            f"Malformed version dir in {target_key!r}: expected '<ts>-<sha7>', got {version_dir!r}."
        )
    uploaded_at, sha7 = version_dir.rsplit("-", 1)
    if not uploaded_at or not sha7:
        raise PromotionError(
            f"Empty ts or sha7 in version dir of {target_key!r}: "
            f"got ts={uploaded_at!r} sha7={sha7!r}."
        )
    # V3 adds a uniqueness token before the content hash. It is not part of
    # the upload timestamp and should not accumulate across manual rollbacks.
    timestamp, separator, token = uploaded_at.rpartition("-")
    if separator and re.fullmatch(r"[0-9a-f]{32}", token):
        uploaded_at = timestamp
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
    entry = {
        "key": target_key,
        "sha7": sha7,
        "bytes": head["ContentLength"],
        "uploaded_at": uploaded_at,
    }
    return {
        "schema_version": old_manifest.get("schema_version", MANIFEST_SCHEMA_VERSION),
        "current": entry,
        "stable": entry,
        "previous": old_manifest.get("current"),
        "history": history,
        "publication_source": old_manifest.get("publication_source"),
        "promotion_mode": "rollback",
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
    """Copy a retained artifact and CAS both serving pointers to the fresh key.

    Preserve the source high-water mark and retry conflicts only while the target
    remains retained. A dry run computes the manifest without writing or copying.
    A failed conditional update leaves the active pointer untouched; its orphaned
    copy is safe to retain for offline cleanup.
    """
    from botocore.exceptions import ClientError

    for _ in range(12):
        old, etag = snapshot(s3_client, bucket, prefix, position)
        if old is None:
            raise PromotionError(
                f"No manifest at s3://{bucket}/{manifest_key(prefix, position)}; "
                "run a source-registered training publication to migrate legacy artifacts first."
            )
        new = build_promotion_manifest(old, target_key, bucket, s3_client)
        if not old.get("publication_source"):
            raise PromotionError(
                "Manifest lacks verified publication source; migrate before rollback"
            )
        entry = new["current"]
        copied_key = new_history_key(prefix, position, entry["uploaded_at"], entry["sha7"])
        new["current"] = new["stable"] = {**entry, "key": copied_key, "rollback_of": target_key}
        new["history"] = [copied_key, *new["history"]][:HISTORY_KEEP_N]
        new["retired"] = sorted(references(old) - references(new))
        if dry_run:
            return new
        try:
            data = s3_client.get_object(Bucket=bucket, Key=target_key)["Body"].read()
            s3_client.put_object(Bucket=bucket, Key=copied_key, Body=data, IfNoneMatch="*")
            write_manifest(s3_client, bucket, prefix, position, new, etag=etag)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "?")
            if code in ("PreconditionFailed", "ConditionalRequestConflict", "409", "412"):
                continue
            raise PromotionError(
                f"Failed to write new manifest: [{code}] {e!s}; safe to retry"
            ) from e
        return new
    raise PromotionError("Concurrent publication prevented rollback; safe to retry")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manual rollback: copy a retained releases/history/ artifact and conditionally "
            "advance stable and current in models/{POS}/releases/manifest.json."
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
    group.add_argument(
        "--to",
        metavar="KEY",
        help="Copy this retained history key and promote it to stable and current.",
    )
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
        # ``promote`` already returns the freshly-written manifest, so read the
        # demoted entry off it directly instead of re-fetching from S3.
        if (new.get("previous") or {}).get("key"):
            old_cur_key = new["previous"]["key"]
        print(f"Promoted {args.position}: stable and current → {new['current']['key']}")
        print(f"  rollback source: {args.to}")
        print(f"  previous now: {old_cur_key}")
        return 0
    except PromotionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
