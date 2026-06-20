# Cross-Model Agent Parity Audit (2026-06-20)

Audit of the repo's three-agent automation machinery — **Claude Code** (`.claude/`), **OpenAI Codex** (`.codex/`), and the **Gemini/Antigravity** family (`.agents/` + `.gemini/`) — for (a) genuine model-agnosticism and (b) Claude→Codex→Gemini feature parity, over the shared provider-neutral brain: `AGENTS.md` + `agent-workflows/*/instructions.md` + `routines/*/instructions.md`.

This is the audit deliverable; the prioritized remediation backlog at the bottom is being worked (see "Remediation status").

> **Update (2026-06-20, follow-up):** backlog P0–P3 have since shipped (#1260, #1281, #1283). The **parity matrix** and findings **F1/F2/F4/F5** below are the *pre-remediation* snapshot — the Gemini ❌ cells for `ruff-format`, `guard-worktree-path`, `pre-pr`, `session-start`, memory-sync, and the cross-provider parity test are now ✅. For current state read "Remediation status", not the matrix. A re-sweep after the Gemini hooks landed found one second-order miss, recorded as **F6** below.

## Local runtimes (what actually reads this config)

- **Claude Code** — `.claude/settings.json` hooks, `.claude/skills/`, `.claude/routines/`; project-scoped memory synced to S3 `claude-memory/`.
- **Codex** — `.codex/hooks.json` + `.codex/hooks/`, user-home `$CODEX_HOME/prompts/` (installed from `.codex/prompts/`), `.codex/automations/`; memory synced to S3 `codex-memory/`.
- **Gemini family** — run two ways, **both reading `.agents/` + `AGENTS.md`**: locally via **Antigravity CLI (`agy`)** (Gemini-CLI lineage — reads skills in `.agents/skills/`, root `AGENTS.md`, and hooks in `.gemini/settings.json` under a `hooks` object with `BeforeTool`/`AfterTool`/`SessionStart`/`SessionEnd` events, regex `matcher` on `tool_name`, block via exit 2 + stderr reason or stdout `{"decision":"deny","reason":...}`); and in CI via the gated `run-gemini-cli` GitHub App (`.github/workflows/gemini-*.yml`).

## Verdict

**The wrapper/shared-instructions design is genuinely model-agnostic and parity at the skill/routine layer is complete.** Every shared workflow and routine has a thin per-provider wrapper that injects `WORKFLOW_*` runtime values over one shared instructions file — adding a provider means adding wrappers, not forking logic. The real gaps are concentrated in deterministic enforcement and supporting tests/docs, almost entirely on the Gemini side.

## Parity matrix

✅ present · ⚠️ partial · ❌ missing.

| Capability | Claude | Codex | Gemini/Antigravity |
|---|---|---|---|
| Skill/prompt — pre-pr-judge | ✅ `.claude/skills/` | ✅ `.codex/prompts/` | ✅ `.agents/skills/` |
| Skill/prompt — post-session-critique | ✅ | ✅ | ✅ |
| Skill/prompt — solve-issues | ✅ | ✅ | ✅ |
| Routine — audit | ✅ (deployed) | ✅ (template) | ✅ (template) |
| Routine — tests-audit | ✅ (template) | ✅ (template) | ✅ (template) |
| Routine — infrastructure-audit | ✅ (template) | ✅ (template) | ✅ (template) |
| Shared `WORKFLOW_*` wrapper contract | ✅ | ✅ | ✅ |
| Hook — ruff-format | ✅ | ✅ | ❌ |
| Hook — guard-worktree-path | ✅ | ✅ | ❌ (prompt-only) |
| Hook — pre-pr gate (ruff+pytest+freshness) | ✅ | ✅ (wraps Claude's) | ❌ (prompt-only) |
| Hook — post-pr-create followup | ✅ | ✅ | ❌ |
| Hook — post-pr-merge (parent ff + splits) | ✅ | ✅ | ❌ |
| Hook — session-start (memory pull + data-link) | ✅ | ⚠️ warn-only (no env persist) | ❌ (`SessionStart` IS supported) |
| Hook — memory-sync on stop | ✅ (Stop) | ✅ (Stop) | ❌ (`SessionEnd` IS supported) |
| Memory S3 sync | ✅ `claude-memory/` | ✅ `codex-memory/` | ❌ local-only Markdown |
| Cross-provider parity test in CI | ⚠️ `test_claude_hooks.py` (self) | ⚠️ `test_codex_hooks.py` (self) | ❌ none |

## Findings

### F1 — Gemini has zero hook enforcement (largest gap)
Every deterministic guardrail (worktree-guard, pre-PR ruff/pytest/freshness gate, ruff-format, post-pr-merge housekeeping, memory sync) is a Claude+Codex hook with **no** Gemini equivalent; `.gemini/settings.json` has no `hooks` key. Gemini's constraints are **prompt-enforced** via `AGENTS.md` MUST-rules only. Antigravity supports the full hook lifecycle (including `SessionStart`/`SessionEnd`, which Codex cannot persist), so Gemini can reach **near-Claude** parity — not just Codex parity.

### F2 — No test pins cross-provider parity
`test_claude_hooks.py` and `test_codex_hooks.py` each test their own provider; nothing asserts that all three providers stay in lockstep when a 4th workflow/routine is added or a shared instructions file is renamed. This is the "a hook file existing ≠ it firing" / by-name-path-allowlist drift trap the project pins elsewhere with regression tests (`test_pr_tokenizer.py`).

### F3 — Shared instruction files leak toward Claude
`agent-workflows/solve-issues/instructions.md` cites **`CLAUDE.md` "Sub-agent contract"** as the authority for the tier-by-risk/worker pattern (≈ lines 79, 192, 286, 291) and enumerates only "Claude `Agent` … ; Codex subagents" for worker spawn (zero Gemini mentions). The provider-neutral source is `AGENTS.md` ("Large (>10-item) parallel cleanups") plus the already-injected `WORKFLOW_SUBAGENTS`. `pre-pr-judge` / `post-session-critique` instructions list only a "Claude wrapper:" pointer.

### F4 — Docs frame Gemini as CI-only (stale)
`AGENTS.md` ("Gemini specifics", "Tool capabilities differ") and `GEMINI.md` describe Gemini as "primarily a GitHub-App CI integration" with "no hooks wired yet." Antigravity (`agy`) is a local interactive runtime reading the same `.agents/` + `AGENTS.md` + `.gemini/settings.json` — the framing should reflect both surfaces.

### F5 — Gemini memory is not synced
Claude and Codex sync incidental memory to per-agent S3 prefixes; Gemini keeps local-only Markdown that never reaches the shared cross-machine store. `scripts/agent-memory-sync.sh` already dispatches `claude`/`codex`/`all` — a `gemini` mode + prefix closes it.

### F6 — Audit routines never updated to scan the new `.gemini/hooks/` (found on re-sweep)
P2 created `.gemini/hooks/` (guard-worktree-path, pre-pr, ruff-format, session-start, session-end), but the shared audit routines that enumerate provider hook dirs as scan scope were not updated to include it: `routines/audit/instructions.md` (ci area-map, Batch+CI auditor, L5 tooling-parity lens) and `routines/infrastructure-audit/instructions.md` (ci area-map, position-scope+hooks auditor) listed only `.claude/hooks` + `.codex/hooks`. A scheduled `audit`/`infrastructure-audit` run would lint Claude's and Codex's deterministic guardrails but silently skip Gemini's. The P0 parity test did not catch it — it asserts each provider's hook *files* exist, not that the routines' scan *scope* names all three dirs. (`tests-audit` names no hook dir and is correctly exempt.) Classic second-order miss: the artifact landed, the consumer that should pick it up didn't.

### Correctly out of scope
- **`worktree-cleanup`** is a plugin/global Claude skill, not project-authored (no file in-repo) and has no shared instructions — stays Claude-only.
- **Codex `session-start` env-persist** is a documented architectural limit of Codex hooks (cannot write `VIRTUAL_ENV`/`PATH`); memory-pull + data context still work. Leave as-is.

## Prioritized remediation backlog

Ordered by risk-adjusted value; each ships as its own PR (tier-by-risk).

- **P0 — Cross-provider parity drift test** (`tests/scripts/test_cross_model_parity.py`, `shared` shard, `-m unit`). For each shared workflow/routine: assert a wrapper exists for all three providers, references its shared instructions path (which exists + is non-empty), and carries the required `WORKFLOW_*` keys; plus neutrality guards encoding P1. Pure addition, lowest risk, backstops everything after it. Fixes F2.
- **P1 — Shared-file neutrality + doc reconciliation.** Re-point `agent-workflows/solve-issues/instructions.md` authority to `AGENTS.md` + `WORKFLOW_SUBAGENTS`; generalize the wrapper-pointer lines; update `AGENTS.md` + `GEMINI.md` to document Antigravity as the local Gemini runtime. Docs/instructions only. Fixes F3, F4. Lands with P0 so the neutrality assertions are green.
- **P2 — Gemini/Antigravity deterministic hooks** (`.gemini/hooks/*` + `.gemini/settings.json` `hooks` + `tests/scripts/test_gemini_hooks.py`). `BeforeTool` → guard-worktree-path (on `write_file|replace`) + pre-pr gate (on `run_shell_command`, delegating to the single-source `.claude/hooks/pre-pr.sh` exactly as Codex does); `AfterTool` → ruff-format; `SessionStart`/`SessionEnd` → memory pull/push. Mirrors the Codex adapter pattern; pinned by synthetic-stdin tests. Fixes F1.
- **P3 — Gemini/Antigravity memory S3 sync.** Extend `scripts/agent-memory-sync.sh` with a `gemini` mode + `gemini-memory/` prefix; add `scripts/gemini-memory-sync.sh`; wire into P2's `SessionStart`/`SessionEnd`. Fixes F5.
- **P4 — (optional, last) De-triplicate hook libs.** After P2 there are three near-identical `lib.sh` copies; extract neutral helpers (gh-pr tokenizer, worktree detection, parent-main ff, splits promotion, venv resolution) into one `scripts/agent-hooks-lib.sh` sourced by thin per-provider shims. Touches the destructive-action gate surface → behind P0+P2 green, backstopped by the per-provider hook tests; warrants an ADR.
- **P5 — Audit-routine scope covers `.gemini/hooks/`** (`routines/audit/instructions.md`, `routines/infrastructure-audit/instructions.md`) + a parity-test guard (`test_routine_hook_scope_covers_all_providers` in `tests/scripts/test_cross_model_parity.py`) asserting any routine that scans `.claude/hooks` + `.codex/hooks` also scans `.gemini/hooks`. Docs/instructions + one test; no retrain. Fixes F6.

## Remediation status

- [x] P0 — parity drift test (`tests/scripts/test_cross_model_parity.py`)
- [x] P1 — shared-file neutrality + Antigravity doc reframe
- [x] P2 — Gemini hooks (`.gemini/hooks/` guard-worktree-path + pre-pr + ruff-format; `tests/scripts/test_gemini_hooks.py`)
- [x] P3 — Gemini memory sync (`agent-memory-sync.sh gemini` + `gemini-memory-sync.sh` + `.gemini/` SessionStart/SessionEnd hooks)
- [x] P4 — hook-lib consolidation (see note below)
- [x] P5 — audit-routine scope covers `.gemini/hooks/` + parity-test guard (`test_routine_hook_scope_covers_all_providers`)

### P4 decision — what was consolidated

The safety-critical, 3×-duplicated, provably byte-identical core moved to one **`scripts/agent-hooks-lib.sh`** (the gh-pr subcommand tokenizer that gates destructive PR/merge actions, plus `find_jq` / `main_worktree` / `abs_path` / `tool_command`). Each provider's `.{claude,codex,gemini}/hooks/lib.sh` now sources it and re-exports the canonical functions under its own prefix (`claude_*` / `codex_*` / `gemini_*`), so every existing hook and test calls the same names while the implementation lives once. Pinned by `tests/scripts/test_cross_model_parity.py` (shared lib exists + each provider sources it) on top of the per-provider tokenizer tests.

**Deliberately left per-provider** (a lower-value, higher-risk follow-up, not done): `refresh_parent_main` / `promote_worktree_splits` (git-mutating, only 2× duplicated, with provider-divergent signatures + call sites and per-provider merge-behavior tests) and the genuinely provider-specific `project_root` / `tool_paths` (apply_patch is Codex-only) / `json_context`.

No `docs/adr/` entry: that set is scoped to ML-system decisions and deploys to the public project wiki; this is agent-tooling, recorded here instead.
