### [FIXED] CI test scope missed unknown paths mixed with recognized changes

**Files:** `src/scripts/scope_positions.py`,
`tests/scripts/test_scope_positions.py`,
`tests/artifacts/test_deployment_rollout.py`,
`agent-guides/operations.md`. Extracted from PR #1587 at
`994d36ade28328ed600a468383a3d318c29a37fc`.

**What:** Agent configuration changes fell back to all eight test shards.
An unknown path mixed with a recognized path could instead escape that fallback
and receive only the recognized path's test coverage.

**Fix:** Explicit provider configuration and hook paths use the shared shard.
Every unclassified non-documentation path forces the full matrix, even when
mixed with recognized files. Training and benchmark scope stay unchanged.

The extraction also retains regression coverage for rollout behavior already
implemented by #1566 and #1577: an already-ready revision succeeds immediately,
an unchanged readiness response remains acceptable, and a failed readiness
request propagates its failure. These tests use fake AWS and HTTP transports.

Image-cache changes and temporary measurement workflows remain in #1587 and
are outside this extraction.

**Validation:** 240 selected unit tests passed for test scoping, rollout
contracts and agent-context invariants, with no model or scaler fitting.
Full Ruff lint and format checks passed.

**Lesson:** Scope recognized paths explicitly without allowing them to hide
unknown paths. Distinguish new behavior from coverage of an already-landed fix.
