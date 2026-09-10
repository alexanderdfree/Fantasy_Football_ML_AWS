"""Delete only immutable keys retired by a successful v3 publication CAS.

Never sweep a shared prefix: its unreferenced objects may belong to a publisher
still validating an upload. Delayed cleanup is safe because publication and
explicit rollback always use fresh physical keys, never retired keys.
"""

from src.shared.artifact_publication import references
from src.shared.model_sync import HISTORY_KEEP_N, history_prefix


def prune(s3_client, bucket, prefix, pos, manifest, keep_n=HISTORY_KEEP_N) -> list[str]:
    """Delete this committed manifest's retirement set, never concurrent uploads.

    ``keep_n`` remains accepted for callers; history retention is decided during
    publication. Legacy manifests confer no deletion authority. Abandoned uploads
    require offline coordinated cleanup and are intentionally preserved here.
    """
    if manifest.get("schema_version") != 3:
        return []
    protected = references(manifest)
    candidates = sorted(
        {
            key
            for key in manifest.get("retired", [])
            if key.startswith(history_prefix(prefix, pos)) and key not in protected
        }
    )
    deleted = []
    for start in range(0, len(candidates), 1000):
        keys = candidates[start : start + 1000]
        result = s3_client.delete_objects(
            Bucket=bucket, Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True}
        )
        errors = {entry["Key"] for entry in (result or {}).get("Errors", [])}
        deleted.extend(key for key in keys if key not in errors)
        if errors:
            raise RuntimeError(f"Artifact retention failed for {len(errors)} objects")
    return deleted
