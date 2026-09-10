# Context and memory maintenance

## Placement and retrieval

- `AGENTS.md` is the provider-neutral startup router, with an 8 KiB UTF-8 budget.
  Keep each provider root (`CLAUDE.md`, `CODEX.md`, `GEMINI.md`) under 4 KiB.
  Preserve critical constraints in the entrypoint and link specialized rules.
  Compatibility routers such as `agent-workflows/operating-lessons.md` also stay
  under 4 KiB; they preserve old links without copying the full rules again.
  Raising the loader limit is not a substitute for deciding what belongs there.
- Put a detailed operational rule in the matching `agent-guides/` topic. Read
  only that topic's relevant sections. Give a new topic a route in the index.
- Architectural choices belong in their ADR, with an index/changelog update.
  Existing decisions should be amended rather than re-explained in several files.
- Keep active status in TODO and its linked plan. Record each resolved incident
  once under `todo/fixed-archive/`, indexed by its original/descriptive heading.
  Preserve File(s), What, Fix and Lesson plus dated evidence. Historical commands,
  paths and measurements may remain there, clearly labeled as historical.
- Search indexes first; read matching incident records, not the whole archive.
  A closed experiment must not remain recommended by an active TODO summary.

## Evidence and duplication

Before adding a lesson, search the topic guide, ADR, active plan and archive for
the same rule. Update or link the existing source. Keep conditions, exceptions,
approval boundaries and rejected alternatives that explain the decision.
Omit repeated narratives from active guidance, while retaining the evidence in
the incident or investigation record. Do not erase history to make a file small.

When consolidating, map removed sections to their retained destination or explain
why they are superseded. Verify moved relative links, old fragment destinations,
and any tests/CI allowlists that name the old path. A matching metric is evidence,
not proof of data identity; do not strengthen an observation into a universal rule.

Keep operational facts close to their source: code/config for implementations,
ADRs for intended decisions, and live state for deployed flags, quotas, versions
and results. Dated observations must retain their date/regime. Check the current
source when answering a current-state question; reconcile conflicting sources.

## Memory

- Keep durable user preferences and concise retrieval pointers in memory. Scope
  project-specific pointers to the project. Avoid duplicating entire agent guides,
  resolved debugging narratives or configuration snapshots in a global summary.
- Treat settings, model catalogs, quotas, paths and run metrics as dated recall.
  Never restore a historical value merely because it appears in memory. Verify
  the effective user/project/runtime layers before describing or changing it.
- Preserve useful general habits: bounded output, production-faithful validation,
  active-worktree edits, current remote-state checks and respect for user scope.
- Follow the active provider's memory-write rules. Codex only writes an update
  note under `$CODEX_HOME/memories/extensions/ad_hoc/notes/` when explicitly
  authorized; generated indexes/summaries are not edited directly. An update note
  does not prove the generated summary or a running task has refreshed.
- Claude's generated memory index comes from topic `index_line` frontmatter;
  retain that metadata. Gemini's memory remains private local Markdown. Sync
  details are in the provider references and SETUP.md. Never commit personal
  memory or credentials to this repository.

## Validation and budgets

Use `tests/test_agent_context.py` for startup budgets and archive/index integrity.
For a consolidation, also compare moved evidence against the recorded base commit;
budgets alone cannot establish that constraints survived. Existing history can be
large on disk without being preloaded. Plugin catalogs are separate startup inputs:
curate enabled capabilities for the task, and verify the next task's actual loaded
catalog rather than treating a removed plugin's directory as active context.
