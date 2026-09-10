# ADR-0025 — Scoped agent guidance and incident retrieval

**Status:** Accepted

## Context

The shared startup file grew to 80,754 bytes. A task with a 65,536-byte instruction
limit received only its prefix, losing later environment and workflow rules.
Provider documents repeated shared mechanics, and a large fixed-issue archive
encouraged reading historical recipes alongside current policy. A stale TODO
summary still recommended an experiment its linked investigation had closed.

## Decision

Keep `AGENTS.md` as an 8 KiB shared router plus essential invariants. Keep provider
root files under 4 KiB each. Specialized rules live in `agent-guides/` and are read
by topic, with provider mechanics under `agent-guides/providers/`. Architectural
decisions remain in their existing ADRs. Human README/setup documentation remains
available without being a mandatory startup reading bundle.

Resolved incidents live in individual `todo/fixed-archive/*.md` files. The original
index path and headings remain for discoverability and fragment-link compatibility.
Incident bodies retain their evidence; only relative link destinations change when
moved. Read matching records and treat their recipes as historical.

Memory stores concise user preferences and scoped retrieval pointers. Operational
settings are verified from current code/configuration and live state. Shared Codex
configuration inherits model/reasoning/context defaults from the user/runtime;
the repository does not pin a historical workstation profile.

## Consequences

- More reference files, but much less startup text and smaller individual reads.
- A topic must be routed from the shared entrypoint/index to stay discoverable.
- Tests enforce entrypoint budgets and archive/index integrity. A consolidation
  still requires a separate evidence-preservation review; size checks cannot
  prove that a constraint survived.
- New lessons amend their existing topic or ADR. They are not copied into each
  provider file and memory store. Provider-specific memory-write rules remain.
- Old archive paths and empirical results stay as historical evidence. Current
  recommendations must reflect superseding decisions.

## Rejected alternatives

- Raising the instruction limit alone: retains the duplication and stale-policy
  problem and still depends on each runtime's context-loading limits.
- Deleting old incident evidence: loses the reasons and conditions behind stop
  rules. Granular retrieval preserves it without preloading it.
- Loading every guide through imports: recreates the original context growth.

## References

[Entrypoint](../../AGENTS.md), [context policy](../../agent-guides/context-maintenance.md),
[guidance index](../../agent-guides/README.md), [incident index](../../todo/fixed-archive.md),
[consolidation ledger](../../todo/context-consolidation-2026-09.md).

## Changelog

- 2026-09-10: Split startup policy, provider references and incident retrieval;
  retain evidence and add context budgets (PR pending).
