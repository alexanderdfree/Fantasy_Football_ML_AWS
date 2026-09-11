"""Validate and promote a history artifact into ``current`` and ``stable``.

History includes failed candidates, so even an operator-selected rollback must
pass the runtime smoke test before serving. Publication uses the manifest ETag
captured before validation; a concurrent publisher makes this attempt fail.

Usage:
    python -m src.scripts.promote --position WR --list
    python -m src.scripts.promote --position WR --to models/WR/history/...sha7/model.tar.gz
    python -m src.scripts.promote --position WR --to ... --dry-run

All state lives in ``src.shared.model_sync``'s manifest helpers — producer
(``src/batch/train.py``), consumer (``src/shared/model_sync.py``), and this operator
tool all share one schema. If you're editing the manifest shape, search for
call sites before landing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import uuid
from pathlib import Path

# Allow running as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.model_sync import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    ManifestLockedError,
    _extract_tarball,
    history_prefix,
    load_legacy_manifest,
    load_manifest,
    load_manifest_snapshot,
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
    return uploaded_at, sha7


def build_promotion_manifest(
    old_manifest: dict,
    target_key: str,
    bucket: str,
    s3_client,
    *,
    head_key: str | None = None,
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
        head = s3_client.head_object(Bucket=bucket, Key=head_key or target_key)
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
        **old_manifest,
        "schema_version": MANIFEST_SCHEMA_VERSION,
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
    """Validate and promote ``target_key`` to ``stable`` for ``position``. Returns the
    new manifest dict (whether or not it was actually written).

    On success, writes the new ``manifest.json`` only — the legacy
    ``{prefix}/{POS}/model.tar.gz`` mirror is no longer maintained (removed
    in the parallel-train-batch race fix; see PR #282). All consumers
    (serving, ``benchmark.py``) read the manifest. A dry-run returns the
    validated manifest without writing to S3 (it still downloads the target).

    A ``ClientError`` from ``write_manifest`` is translated to
    ``PromotionError`` so ``main()`` can render a human-friendly error
    instead of a raw boto3 stack trace. The pre-write steps
    (``load_manifest``, ``build_promotion_manifest``) already raise
    ``PromotionError`` on their own failure modes, so the manifest is
    either fully written or fully untouched — no partial state to
    clean up.
    """
    from botocore.exceptions import ClientError

    try:
        old, expected_etag = load_manifest_snapshot(s3_client, bucket, prefix, position)
    except (ClientError, json.JSONDecodeError, ManifestLockedError) as e:
        raise PromotionError(f"Cannot read manifest; no publication attempted: {e}") from e
    legacy_preview_key = None
    if old is None:
        legacy = load_legacy_manifest(s3_client, bucket, prefix, position)
        if legacy is None:
            raise PromotionError(f"No manifest at s3://{bucket}/{manifest_key(prefix, position)}")
        if target_key not in (legacy.get("history") or []):
            raise PromotionError("Target key not in manifest.history[]")
        from src.artifacts.publication import protect_legacy
        from src.artifacts.source import image_source_sha, load_source

        try:
            source = load_source(s3_client, bucket, prefix, image_source_sha())
            old = protect_legacy(
                s3_client, bucket, prefix, position, source, require_lineage=False, dry_run=dry_run
            )
            if dry_run:
                legacy_preview_key = target_key
        except Exception as error:
            raise PromotionError(
                f"Cannot protect legacy artifacts before promotion: {error}"
            ) from error
    target_key = old.get("legacy_keys", {}).get(target_key, target_key)
    if not target_key.startswith(history_prefix(prefix, position)):
        raise PromotionError("Manual promotion must use protected v3 artifact storage")
    if not old.get("source_frontier"):
        raise PromotionError("Protected manifest lacks a verified source frontier")
    new = build_promotion_manifest(old, target_key, bucket, s3_client, head_key=legacy_preview_key)
    # A history entry is not proof of approval. Revalidate even legacy entries
    # against this runtime before making the operator's target serving-eligible.
    from src.shared.smoke_test import run_smoke_test

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=legacy_preview_key or target_key)
        artifact_bytes = obj["Body"].read()
        with tempfile.TemporaryDirectory(prefix="model-promote-") as staging:
            _extract_tarball(artifact_bytes, Path(staging))
            run_smoke_test(position, staging)
    except Exception as e:
        raise PromotionError(f"Target artifact failed validation; manifest unchanged: {e}") from e
    new["current"]["smoke_passed"] = True
    new["current"]["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    new["promotion_mode"] = "rollback"
    new["rollback_epoch"] = uuid.uuid4().hex
    # An explicit upgraded operator action adopts epoch/intent fencing. Merely
    # migrating the predecessor's rollback must keep its same-source barrier.
    new.pop("rollback_source_barrier", None)
    new["stable"] = new["current"]
    if (old.get("stable") or {}).get("key") != target_key:
        new["previous_stable"] = old.get("stable")
    else:
        new["previous_stable"] = old.get("previous_stable")
    if dry_run:
        return new
    if expected_etag is None and old.get("predecessor_manifest_digest"):
        from src.artifacts.publication import _manifest_digest

        if old["predecessor_manifest_digest"] != _manifest_digest(
            load_legacy_manifest(s3_client, bucket, prefix, position)
        ):
            raise PromotionError("Predecessor manifest changed during validation; retry promotion")
    try:
        write_manifest(s3_client, bucket, prefix, position, new, expected_etag=expected_etag)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        raise PromotionError(
            f"Failed to write new manifest to s3://{bucket}/"
            f"{manifest_key(prefix, position)}: [{code}] {e!s}. "
            "No unconditional retry was attempted. Re-read the manifest before retrying; "
            "a concurrent publisher may have advanced it."
        ) from e
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
        # ``promote`` already returns the freshly-written manifest, so read the
        # demoted entry off it directly instead of re-fetching from S3.
        if (new.get("previous") or {}).get("key"):
            old_cur_key = new["previous"]["key"]
        print(f"Promoted {args.position}: current → {args.to}")
        print(f"  previous now: {old_cur_key}")
        return 0
    except PromotionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
