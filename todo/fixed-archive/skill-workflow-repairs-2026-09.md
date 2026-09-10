### [FIXED] Skill metadata and workflow assumptions broke loading and safe delivery

**File(s)**: `.claude/skills/solve-issues/SKILL.md`, `.codex/prompts/*.md`,
`.agents/skills/post-session-critique/SKILL.md`, `agent-workflows/{pre-pr-judge,post-session-critique,solve-issues}/instructions.md`,
and `tests/test_agent_context.py`; observed from baseline `ab1b0c84` on 2026-09-10.

**What**: The Claude issue-solving description contained an unquoted colon, and
legacy prompt argument hints parsed as invalid YAML or lists instead of strings.
The scope judge skipped itself after an explicit scope addition. Issue-solving
assumed spawned workers owned isolated checkouts and used PR/log labels as stronger
preservation evidence than they provide. Codex memory writing required a magic
flag even when the user had explicitly authorized the same action in plain language.

**Fix**: Use valid scalar metadata and exercise all checked-in skill/prompt
frontmatter through its YAML loader. Judge the revised task, resolve the current
base, and recheck affected gates after rebasing. Allocate and verify actual worker
worktrees before parallel edits, retain the sequential fallback, and check exact
merged content before branch deletion. Preserve owner sign-off for audit-tier PRs
and accept explicit memory authorization in either supported form.

**Lesson**: A workflow must match the current provider's capabilities and the
user's actual scope. Metadata loading and concrete revision/path evidence require
direct checks; names, old examples, and successful worker creation are not proof.
