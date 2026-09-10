# Agent guidance index

[AGENTS.md](../AGENTS.md) is the shared startup entrypoint and routing table.
These files are read on demand. Search headings or symbols, then read the
relevant section; do not concatenate this directory into the prompt.

| Topic | Reference |
|---|---|
| Repository shape and six-position symmetry | [Project layout](project.md) |
| Features, targets, losses, attention contracts | [Modeling](modeling.md) |
| Rejected approaches and conditions for reconsideration | [Stop rules](stop-rules.md) |
| Experiment/test commands and existing harnesses | [Experiments](experiments.md) |
| Data/model investigation and production-faithful evidence | [Validation](validation.md) |
| Hardware, dtype, GPU and CPU policy | [Platform](platform.md) |
| CI, serving, training orchestration, artifact operations | [Operations](operations.md) |
| Worktree, Git, PR, review and merge workflow | [Delivery](delivery.md) |
| Investigation and communication discipline | [Investigation](investigation.md) |
| Local environment and validation-gate gotchas | [Environment](environment.md) |
| Provider mechanisms | [Capability boundaries](providers/capabilities.md), [Claude](providers/claude.md), [Codex](providers/codex.md), [Gemini](providers/gemini.md) |
| Instruction/memory maintenance | [Context policy](context-maintenance.md) |

The [ADR index](../docs/ARCHITECTURE.md) owns architectural decisions. The
[fixed-issue index](../todo/fixed-archive.md) owns historical incident evidence.
Their details should be linked, not copied into every provider file or memory.
