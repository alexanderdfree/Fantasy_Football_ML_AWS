@AGENTS.md

# Codex

Read the relevant provider section before using or changing its mechanism:

- CLI startup: [fresh worktree launcher](agent-guides/providers/codex.md#fresh-worktree-launcher).
- Hook trust, coverage and tool use: [hooks](agent-guides/providers/codex.md#hooks).
- Reusable workflows and legacy aliases: [skills and prompts](agent-guides/providers/codex.md#skills-and-slash-prompts).
- Audit automation: [wrapper](agent-guides/providers/codex.md#audit-automation-wrapper).
- Personal memory: [write policy](agent-guides/context-maintenance.md#memory) and
  [sync mechanics](agent-guides/providers/codex.md#auto-memory).

Shared workflow behavior lives in `agent-workflows/`; provider wrappers supply
runtime values. Current configuration follows the evidence rule in `AGENTS.md`.
