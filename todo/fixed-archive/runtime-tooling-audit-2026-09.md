### [FIXED] Provider hook paths, workflow inputs and infrastructure ownership diverged

**File(s)**: `.claude/settings.json`, `.gemini/settings.json`,
`.claude/hooks/{guard-worktree-path,ruff-format,post-pr-merge,lib}.sh`,
`.gemini/hooks/{guard-worktree-path,ruff-format}.sh`, `.codex/hooks/lib.sh`,
`scripts/agent-hooks-lib.sh`, `scripts/bootstrap-claude-wsl.sh`,
`.github/workflows/{ab-batch,ablate-rb-gate,gemini-scheduled-triage}.yml`,
`infra/batch/teardown.sh`, `infra/aws/bootstrap.sh`.
Defects reproduced against `92be2873` during the 2026-09-10 runtime audit
(`codex/audit-runtime-correctness`, PR #1565). This record covers the
metric-neutral tooling slice extracted from that branch; the model, data-cache,
client and experiment findings have their own records on that branch.

**What**:

- Claude/Gemini configured hook commands split repository paths containing
  spaces, preventing every hook from running even though the scripts handled
  those paths correctly.
- Claude/Gemini and the generated WSL guard compared unnormalized paths, so a
  parent-checkout write reached through `..`, a relative path or a symlinked
  alias passed, and an aliased path into the worktree itself was blocked.
- Claude promoted the worktree's local `data/splits` to the parent after queued
  or mismatched merge events (a `gh pr merge --auto` that only queued, another
  head, a base other than `main`, or `main` already past the PR), and would
  copy loose parquets over a sealed data release.
- The RB-gate ablation seed and the `ab-batch` dispatch inputs were spliced
  unquoted into a remote shell script / a re-parsed argument string.
- Batch teardown retained the CPU queue, compute environment and job definition
  and deleted `ecsTaskExecutionRole`, which the serving task shares. Bootstrap
  requested the portfolio certificate (`alexfree.me` + `www`) and instructed
  apex DNS changes, while deployment and clients require
  `fantasy.alexfree.me`; an existing HTTPS listener kept its old certificate.
- Scheduled issue triage searched `no:label label:"status/needs-triage"`
  (both conditions), selecting neither intended population.

**Fix**: Quote configured hook executable paths while preserving every other
setting. Share Python-based path resolution (`agent_hooks_abs_path`),
current-PR lookup and merged-commit verification across providers in
`scripts/agent-hooks-lib.sh`; require the matching completed merge (`MERGED`
into `main`, head equal to the worktree HEAD, `origin/main` at the merge
commit) before promoting splits, and skip promotion when either side carries a
sealed release marker. Validate the ablation seed and quote it inside the SSM
command; build the `launch_ab` argv as an array under `set -f`. Cover the CPU
teardown resources and keep the shared serving role. Request the application
certificate, reconcile an existing HTTPS listener and render subdomain DNS
instructions within the portfolio's DNS zone. Select the union of the two
triage populations.

**Validation**: `tests/scripts/test_provider_config_paths.py` (every configured
Claude/Gemini command runs from a space-containing checkout),
`test_provider_guard_integrity.py` (guards compare resolved paths; only a
matching completed merge promotes splits), `test_claude_hooks.py` (merge
housekeeping against stubbed PR metadata), `test_ab_workflow_inputs.py`,
`test_benchmark_batch_workflow.py`, `test_gemini_triage_selection.py` and
`test_infra_lifecycle.py` run the exact workflow/CLI fragments against local
command stubs with harmless injection markers and valid-input controls. No live
infrastructure, DNS, label, comment or merge mutation is exercised.

**Lesson**: Verify the complete producer/consumer handoff: quoting at the
configuration layer, normalization at the path-comparison layer, and the remote
state (merge result, sealed data) a hook assumes before it mutates the parent.

**Not in this slice** (still on `codex/audit-runtime-correctness`): the
`launch_ab --only` `action="extend"` parser change and the matching per-variant
`--only=NAME` argv in `ab-batch.yml` (names starting with `-` and containing
`=` remain undispatchable from the workflow), the `retune-nn-batch.yml`
`--batch-cuda-graph` collection namespace (needs `aggregate_results`), and the
rolling-origin, tuning-namespace, scheduler, LightGBM-objective and
resource-probe repairs. The `benchmark-batch.yml` `image_sha` and resolver
hardening shipped separately in #1591.
