---
description: Triage claude-audit issues into tier-by-risk Codex PRs
argument-hint: [ISSUES="<numbers or range>"] [DRY_RUN=1]
---

Run the Final-Project Codex solve-issues workflow for `claude-audit` findings.

If `ISSUES` is supplied, scope to those issues. Otherwise enumerate the open actionable backlog:

`gh issue list --label claude-audit --state open --json number,title,labels,updatedAt --limit 400`

Workflow:

1. Read `AGENTS.md`, especially stop-rules, worktree workflow, large-cleanup PR partitioning, and the solve-issues discipline.
2. For each finding, read the cited file and parse the machine-readable `claude-audit/v1` JSON block when present. Fall back to body prose only when the JSON is absent or malformed.
3. Classify each finding as FIX, LEAVE, or UNCERTAIN.
   - LEAVE covers false positives, stale claims, and feature-drift suggestions that violate stop-rules.
   - UNCERTAIN needs user input before edits.
4. For FIX findings, partition into file-disjoint bundles and risk tiers:
   - Tier A: localized tests/docs/operator scripts.
   - Tier B: localized production code with small blast radius.
   - Tier C: shared pipeline/model/training/serving behavior or anything needing benchmark evidence.
5. For more than ten fixes, use Codex subagents when available. Workers draft commits only; the orchestrator opens tier PRs. If subagents are unavailable, work tier by tier manually.
6. For each tier branch, run the deterministic gate, then `/prompts:pre-pr-judge`, then open a PR.
7. For Tier C, include benchmark or pipeline evidence in the PR body and require explicit user merge sign-off after green CI.
8. Close LEAVE issues as not planned with a one-line reason and the `leave` label. Fixed issues should close through PR references.

If `DRY_RUN=1`, stop after the classification and tier plan; do not edit, close issues, commit, push, or open PRs.
