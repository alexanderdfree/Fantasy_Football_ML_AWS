@AGENTS.md

# Gemini / Antigravity

Use [the Gemini reference](agent-guides/providers/gemini.md) for local `agy`
hooks, skills, memory paths, CI setup and audit wrappers. Local and CI runtimes
differ; verify the available tools and active configuration.

- Shared skills live in `.agents/skills/` and read `agent-workflows/`. Invoke via
  `activate_skill` when that tool is available, using the Gemini runtime block.
- `.gemini/settings.json` wires local worktree, format and pre-PR guards. CI uses
  a clean checkout. Workflow configuration is in `.github/workflows/gemini-*.yml`;
  verify `GEMINI_ENABLED` and backend setup before assuming it is active.
- Audit wrappers are in `.agents/routines/`; they share the existing audit labels
  and schema. A tracked wrapper is not evidence of an active scheduled task.
- Private memory is a local recall cache. Set `GEMINI_MEMORY_DIR` if the runtime's
  actual path differs from the derived default. Keep durable shared guidance in
  the relevant [agent guide](agent-guides/README.md) or ADR, and keep personal
  material out of git. See [context maintenance](agent-guides/context-maintenance.md).
