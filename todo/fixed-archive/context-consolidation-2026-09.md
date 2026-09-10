### [FIXED] Startup guidance exceeded its load limit and duplicated stale decisions

- **File(s):** `AGENTS.md`, provider root files, `agent-guides/`, `todo/fixed-archive/`, `TODO.md`, `.codex/config.toml`, context/rename checks and ADR-0025 (PR pending).
- **What:** The 80,754-byte shared entrypoint was truncated at 65,536 bytes in an observed task. Later instructions disappeared; a monolithic incident archive and copied policies increased retrieval cost. TODO still proposed closed stream experiments, and matching Ridge MAE had been overstated as proof of identical data.
- **Fix:** Keep bounded startup routers and topic references; move all 142 existing incident records into individually indexed files with preserved headings and rebased links. Reconcile the stale recommendations and metric heuristic, stop copying every lesson into startup files, and inherit personal Codex model/context settings. See the [preservation ledger](../context-consolidation-2026-09.md).
- **Lesson:** Preserve evidence and constraints in scoped sources. Check the actual loaded prompt, reconcile superseded decisions, and validate content preservation as well as byte budgets.
