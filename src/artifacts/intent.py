"""Reserve publication intent before training, with atomic retry-safe bindings."""

from __future__ import annotations

import json
import re

from src.artifacts.source import source_key

_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
_CONFLICT = {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"}


def _binding(position, source_sha, dataset_id, run_id, publication_revision=None):
    source_key("models", source_sha)
    if position not in _POSITIONS:
        raise RuntimeError("Publication intent requires a supported position")
    if dataset_id is not None and (
        not isinstance(dataset_id, str) or re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None
    ):
        raise RuntimeError("Publication intent dataset must be an immutable dataset ID or None")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
        raise RuntimeError(
            "Publication intent requires a nonempty run ID of at most 256 characters"
        )
    if publication_revision is not None and (
        not isinstance(publication_revision, str) or not publication_revision
    ):
        raise RuntimeError("Publication revision must be a nonempty revision or None")
    return {
        "source_sha": source_sha,
        "position": position,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "publication_revision": publication_revision,
    }


def intent_key(prefix: str, position: str, source_sha: str) -> str:
    _binding(position, source_sha, None, "key")
    return f"{prefix.strip('/')}/releases/v3/intents/{source_sha}/{position}.json"


def _read(s3, bucket, key, position, source_sha):
    from botocore.exceptions import ClientError

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NotFound"}:
            raise
        return None, None
    ledger = json.loads(response["Body"].read())
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != 1
        or ledger.get("source_sha") != source_sha
        or ledger.get("position") != position
        or type(ledger.get("latest_sequence")) is not int
        or not isinstance(ledger.get("reservations"), dict)
        or ledger["latest_sequence"] != len(ledger["reservations"])
    ):
        raise RuntimeError(f"Malformed publication-intent ledger: {key}")
    sequences = set()
    for run_id, descriptor in ledger["reservations"].items():
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"Malformed publication-intent reservation: {key}")
        binding = _binding(
            position,
            source_sha,
            descriptor.get("dataset_id"),
            run_id,
            descriptor.get("publication_revision"),
        )
        sequence = descriptor.get("sequence")
        if (
            type(sequence) is not int
            or not 1 <= sequence <= ledger["latest_sequence"]
            or sequence in sequences
            or descriptor != {**binding, "sequence": sequence}
        ):
            raise RuntimeError(f"Malformed publication-intent reservation: {key}")
        sequences.add(sequence)
    if not response.get("ETag"):
        raise RuntimeError("Publication-intent reads require an ETag for conditional writes")
    return ledger, response["ETag"]


def reserve_intent(
    s3, bucket, prefix, position, source_sha, dataset_id, run_id, *, publication_revision=None
) -> dict:
    """Reserve once before compute; retries return the same immutable descriptor.

    A single conditional write records both the counter and binding. Separate
    counter/receipt objects would allow a crash to reassign an old run a newer
    sequence. Reservations are retained with their plans; expiration is explicit
    future policy rather than a silent loss of idempotency.

    ``publication_revision`` captures the operator's rollback epoch, not the
    generic CAS revision that also changes during artifact collection.
    """
    from botocore.exceptions import ClientError

    binding = _binding(position, source_sha, dataset_id, run_id, publication_revision)
    key = intent_key(prefix, position, source_sha)
    for _ in range(16):
        ledger, etag = _read(s3, bucket, key, position, source_sha)
        if ledger is None:
            ledger = {
                "schema_version": 1,
                "source_sha": source_sha,
                "position": position,
                "latest_sequence": 0,
                "reservations": {},
            }
        existing = ledger["reservations"].get(run_id)
        if existing is not None:
            expected = {
                **binding,
                "sequence": existing["sequence"],
                "publication_revision": existing["publication_revision"],
            }
            if existing != expected:
                raise RuntimeError("Publication run ID is already bound to different inputs")
            return existing
        sequence = ledger["latest_sequence"] + 1
        descriptor = {**binding, "sequence": sequence}
        ledger["latest_sequence"] = sequence
        ledger["reservations"][run_id] = descriptor
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode(),
                ContentType="application/json",
                **({"IfMatch": etag} if etag is not None else {"IfNoneMatch": "*"}),
            )
            return descriptor
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in _CONFLICT:
                raise
    raise RuntimeError(
        "Publication-intent reservation repeatedly conflicted; retry before training"
    )


def validate_intent(s3, bucket, prefix, descriptor) -> bool:
    """Reject unregistered/forged bindings; report whether this run is still latest."""
    if not isinstance(descriptor, dict):
        raise RuntimeError("Missing publication-intent descriptor")
    if type(descriptor.get("sequence")) is not int or descriptor["sequence"] < 1:
        raise RuntimeError("Publication intent requires a positive integer sequence")
    position, source_sha = descriptor.get("position"), descriptor.get("source_sha")
    binding = _binding(
        position,
        source_sha,
        descriptor.get("dataset_id"),
        descriptor.get("run_id"),
        descriptor.get("publication_revision"),
    )
    key = intent_key(prefix, position, source_sha)
    ledger, _ = _read(s3, bucket, key, position, source_sha)
    if ledger is None or ledger["reservations"].get(binding["run_id"]) != descriptor:
        raise RuntimeError("Publication intent does not match its registered immutable binding")
    return descriptor["sequence"] == ledger["latest_sequence"]
