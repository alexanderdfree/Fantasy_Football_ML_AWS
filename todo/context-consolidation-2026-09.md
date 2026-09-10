# Context consolidation — preservation ledger

Guidance baseline: `41941966ba59a45b978b09533e684275d9673e80`. Archive baseline, including concurrently merged incidents: `7dd5e923dc7ac27888dfbd688d6a5a1f272fcb01`. This ledger distinguishes relocated constraints from deliberate corrections. Source line numbers refer to the guidance baseline, not the shortened files. The new evaluation-cohort constraint from PR #1541 is retained in both the entrypoint and validation guide.

## Guidance destinations

| Original source | Retained destination | Disposition |
|---|---|---|
| `AGENTS.md 11–41` | [agent-guides/project.md](../agent-guides/project.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 42–70` | [agent-guides/platform.md](../agent-guides/platform.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 71–117` | [agent-guides/modeling.md](../agent-guides/modeling.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 118–134` | [agent-guides/stop-rules.md](../agent-guides/stop-rules.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 135–146` | [agent-guides/experiments.md](../agent-guides/experiments.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 147–154, 205–217` | [agent-guides/operations.md](../agent-guides/operations.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 155–177, 218–234` | [agent-guides/delivery.md](../agent-guides/delivery.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 182–204` | [agent-guides/validation.md](../agent-guides/validation.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 235–253` | [agent-guides/investigation.md](../agent-guides/investigation.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 254–268` | [agent-guides/environment.md](../agent-guides/environment.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 269–272` | [agent-guides/providers/capabilities.md](../agent-guides/providers/capabilities.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `AGENTS.md 273–303` | [Codex](../agent-guides/providers/codex.md), [Gemini](../agent-guides/providers/gemini.md), [Claude](../agent-guides/providers/claude.md) | Retired duplicate parity table and Gemini synopsis. Hook names, PR gates, worktree launcher, legacy prompt bootstrap, audits, labels, CI setup and separate memory-sync prefixes remain in the provider references. Shared workflow contracts remain in agent-workflows/. |
| `CODEX.md` | [agent-guides/providers/codex.md](../agent-guides/providers/codex.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `CLAUDE.md` | [agent-guides/providers/claude.md](../agent-guides/providers/claude.md) | Moved with relative links rebased; corrections below are the only semantic differences. |
| `GEMINI.md` | [agent-guides/providers/gemini.md](../agent-guides/providers/gemini.md) | Moved with relative links rebased; corrections below are the only semantic differences. |

The old AGENTS introduction/orientation (lines 1–10) and section labels (178–181) are replaced by the short router. All referenced overview/setup/decision documents remain. Provider `@AGENTS.md` imports remain at their original root paths. No incident evidence was deleted.

## Deliberate corrections

- `agent-guides/validation.md`: Restore the qualified heuristic; a scalar metric is not an injective representation of the data.
- `agent-guides/platform.md`: Same false equivalence appeared in a second active source; retain the retrain warning and skip discipline.
- `agent-guides/delivery.md`: Preserve incident documentation while replacing the unbounded-growth instruction.
- `agent-guides/delivery.md`: Point the preserved documentation requirement at the granular archive.
- `agent-guides/operations.md`: The old section label no longer exists; clarify freshness without removing observations.
- `agent-guides/operations.md`: Keep the architecture/wheel distinction without an obsolete current-version cue.
- `agent-guides/environment.md`: Retain the environment-isolation lesson while removing an unsupported universal diagnosis.
- `agent-guides/environment.md`: Retain data reuse while removing an unconditional destructive preparation command.
- `SETUP.md`: Repair a moved-section reference.
- `SETUP.md`: Stop directing every durable lesson into the startup file.
- `.codex/config.toml`: Remove stale project model/context overrides; preserve service tier and fast-mode choices and inherit user/runtime settings.
- `agent-guides/providers/codex.md`: Use one scoped source for durable knowledge; preserve all sync mechanics.
- `agent-guides/providers/claude.md`: Replace duplicated parity sections with direct references.
- `agent-guides/providers/claude.md`: Eliminate the explicit duplicate-copy growth rule while preserving durable knowledge.
- `agent-guides/providers/claude.md`: Repair a moved-section reference.
- `agent-guides/providers/gemini.md`: Keep shared facts discoverable without duplicating startup text.
- `agent-guides/providers/gemini.md`: Avoid conflating provider runtime databases with their Markdown recall stores.
- `agent-workflows/post-session-critique/instructions.md`: Route new lessons to their scoped source.
- `agent-workflows/post-session-critique/instructions.md`: Use selective retrieval in the maintenance workflow itself.
- `agent-workflows/post-session-critique/instructions.md`: Prevent re-growth of the entrypoint.
- `agent-workflows/post-session-critique/instructions.md`: Consolidate ownership while preserving provider write restrictions.
- `agent-workflows/post-session-critique/instructions.md`: Remove a second stale destination instruction.
- `agent-workflows/post-session-critique/instructions.md`: Make the example obey the duplicate-check requirement.
- `TODO.md`: Replace the remaining blanket archive-read instruction.
- `agent-guides/stop-rules.md`: Repair a cross-section reference after moving the topic.
- `agent-guides/stop-rules.md`: Repair a cross-section reference after moving the topic.
- `agent-guides/stop-rules.md`: Repair a cross-section reference after moving the topic.
- `agent-guides/stop-rules.md`: Repair a cross-section reference after moving the topic.
- `agent-guides/experiments.md`: Repair a moved operating-lesson reference.
- `agent-guides/delivery.md`: Keep the existing CI exception linked to its source.
- `agent-guides/providers/claude.md`: Clarify that the moved provider reference is not a mandatory full import.
- `agent-guides/providers/gemini.md`: Clarify the moved provider reference.
- `agent-guides/providers/gemini.md`: Remove a second instruction to mirror every discipline through startup text.
- `docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md`: Repair the active ADR reference to the moved platform matrix.
- `docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md`: Repair the ADR reference list.
- `docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md`: Reconcile the same overstated metric-identity claim in the authoritative decision.
- `agent-guides/providers/capabilities.md`: Replace the stale installed-tool inventory with runtime discovery while retaining provider and platform-injection boundaries.
- `.agents/skills/solve-issues/SKILL.md`: Update the shared workflow and provider memory destinations together; preserve approval/write rules.
- `.claude/skills/solve-issues/SKILL.md`: Update the shared workflow and provider memory destinations together; preserve approval/write rules.
- `.codex/prompts/solve-issues.md`: Update the shared workflow and provider memory destinations together; preserve approval/write rules.
- `agent-workflows/solve-issues/instructions.md`: Update the shared workflow and provider memory destinations together; preserve approval/write rules.
- `.agents/skills/post-session-critique/SKILL.md`: Reconcile both Gemini runtime cues with the new shared knowledge destination.
- `agent-workflows/solve-issues/instructions.md`: Repair moved rule references without changing FIX/LEAVE categories, delegation or owner merge sign-off.
- `routines/audit/instructions.md`: Replace repeated worker prompt dumps with scoped rule references and matching-incident reads; retain complete orchestrator dedupe, audit coverage and final verification.
- `routines/tests-audit/instructions.md`: Replace repeated worker prompt dumps with scoped rule references and matching-incident reads; retain complete orchestrator dedupe, audit coverage and final verification.
- `routines/infrastructure-audit/instructions.md`: Replace repeated worker prompt dumps with scoped rule references and matching-incident reads; retain complete orchestrator dedupe, audit coverage and final verification.
- `agent-guides/validation.md`: Preserve the new evaluation-cohort constraint from upstream PR #1541, including matched full-season actuals, the pregame reference, prior-season elite importance and serialization completeness.
- `agent-guides/platform.md`: Respect upstream PR #1542 retiring the training-skipped marker; preserve the qualified docs-only contract without resurrecting deleted tooling.
- `docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md`: Respect upstream PR #1542 retiring the training-skipped marker; preserve the qualified docs-only contract without resurrecting deleted tooling.
- `TODO.md`: replace the superseded CUDA-stream recommendation with the closed Lever B/B′ decisions and the dtype-specific graph caveat; retain the old fragment anchor and link to the full experiment record.

## Incident preservation

All **142 pre-existing incident records** were compared to the baseline after normalizing relative links to repository-root destinations. Every heading, body, number, code fragment and exception matched. The old index retains the original headings, so incoming fragment links still resolve. The new consolidation incident is additional.

All **14 relocated guidance blocks** also matched after reversing only the explicit corrections above. The duplicated provider parity synopsis was checked against the retained provider references separately.

## Retrieval and scope

- README and SETUP retain human-facing content; startup guidance now routes to relevant sections instead of requesting the full bundle.
- The root instruction budget is 8 KiB; each provider root is 4 KiB. Tests enforce these limits and index/record routing.
- Personal Codex memory is outside git. The authorized update uses the supported update-note mechanism; generated summaries and this running task are not claimed to have refreshed.
- Plugins were disabled by the owner and are not modified by this PR.
- Prediction, feature, training and serving implementation code is unchanged. The Codex model/context inheritance change affects future agent configuration only.

## Validation

- Baseline-to-destination preservation comparison: passed as described above.
- Context budgets, routes, historical-path exemptions and provider workflow parity: checked by the targeted test run recorded in the PR.
- Full local gates and GitHub CI: results recorded in the PR rather than frozen here.

## Review and upstream reconciliation

- The post-PR prompt's contradictory admin-bypass option is removed, while its user-confirmed silent-stop exception remains. Merge state must be verified before branch deletion. Critique descriptions now route to scoped guidance.
- The detailed review found locale-dependent Markdown reads on Windows and a cross-incident fragment whose label contains nested brackets. Reads now specify UTF-8, the fragment targets the retained index, and a rendered-link test covers archive routes.
- A separate check caught a literal memory-index example. Its original `slug.md` text is restored. The strict preservation comparison leaves code spans verbatim and handles nested labels. Both bad cases failed the check before correction; all incident records and relocated guidance blocks now pass.
- PR #1542 merged an overlapping operating-lessons extraction during this review. Its complete five-section body was compared with the original guidance after normalizing relative links and matched. Those rules already survive in the scoped guides (with the listed corrections), so `agent-workflows/operating-lessons.md` remains a compact router with its original five headings/anchors. Its path and all constraints remain available without a second copy.
- PR #1542 also retired the `training-skipped:` marker tooling. Current guidance and ADR-0017 now respect that removal and retain only the qualified `[docs-only]` contract. Historical incident evidence is unchanged.
