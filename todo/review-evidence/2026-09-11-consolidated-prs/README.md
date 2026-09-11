# Saved consolidated PR review evidence

This folder is the Git-tracked companion to
[the paused review checkpoint](../../consolidated-pr-review-checkpoint-2026-09-11.md).
The review is unfinished, the three findings are unfixed, and no PR was merged.

It preserves all seven review scripts/probes, four patches for the exact tested
source assembly, complete test logs, the failing probes, file inventories,
captured production manifest metadata, archive checksums/ETags, migration and
readiness results, and all six fitted replay reports. Diagnostic logs have only
trailing whitespace normalized; their original bytes remain in the local archive.

The `.py.txt` files are exact source archives of the executed review scripts.
Their suffix prevents the deliberately failing probes from being discovered by
ordinary repository-wide pytest runs. To rerun a probe, copy it to a temporary
`.py` file and invoke pytest explicitly from the reconstructed review checkout
with `PYTHONPATH=.`. Use the pinned interpreter recorded in the checkpoint, or
recreate the environment from the checked-in requirements.

For example, after reconstructing the reviewed source assembly:

```sh
cp todo/review-evidence/2026-09-11-consolidated-prs/test_review_maintenance.py.txt \
  /tmp/test_consolidated_review_maintenance.py
PYTHONPATH=. python -m pytest -n 4 --dist=loadgroup \
  /tmp/test_consolidated_review_maintenance.py
```

The assembly order is #1575, #1534, #1576, #1479 on the recorded #1577 head.
Patch files are stored as `.patch.json` line arrays to preserve unified-diff
context markers without introducing trailing whitespace in the repository.
Decode and verify their exact bytes before use:

```sh
python - <<'PY'
import hashlib, json, tempfile
from pathlib import Path
source = Path('todo/review-evidence/2026-09-11-consolidated-prs')
output = Path(tempfile.mkdtemp(prefix='consolidated-review-patches-'))
for path in source.glob('assemble-*.patch.json'):
    record = json.loads(path.read_text())
    body = ''.join(record['lines']).encode()
    assert hashlib.sha256(body).hexdigest() == record['sha256']
    (output / record['filename']).write_bytes(body)
print(output)
PY
```

Run `git apply --check` separately before applying each decoded patch.
These patches restore previously reviewed PR code; they are not fixes for the
findings. `reviewed-revisions.json` records all eight full base/head SHAs and
checksums of the complete PR diffs. Full diffs are already represented by those
Git objects and can be recreated with `git diff BASE...HEAD`.

Downloaded data, model weights and generated native build outputs remain in:

`/Users/alex/.codex/worktrees/992b/Final-Project/logs/consolidated-review-20260911`

They are deliberately not committed, following `AGENTS.md`'s artifact rule.
The production manifest and archive inventory here identify the exact S3 keys,
sizes and hashes. The original scripts retain their recorded absolute local
paths; adjust those paths if replaying on another machine. S3 inventory/capture
scripts perform read-only cloud requests; the review probes use in-memory stores.

`production-archive-readiness.json` is the verified production result. The
discarded old local-directory `archive-readiness.json` is excluded to avoid
mistaking obsolete duplicate caches for a production defect.

The full-unit log initially reports six failures: five environment restrictions
passed in `retest-network-tests.log`; the remaining failure is the combined
#1479/#1566 package boundary. `confirmed-findings.log` separately records the
canonical artifact-path failure, two missing-lease cases, and that package-boundary
failure, with two passing controls. See the checkpoint for interpretation and
remaining work.
