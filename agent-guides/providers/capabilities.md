# Provider capability boundaries

Read only the sections relevant to the task. [AGENTS.md](../../AGENTS.md) supplies the shared entrypoint; current code/config and linked decisions supply operational state. Dated measurements describe their recorded regime, not a promise about today.

## Tool capabilities differ between agents

No agent should assume another provider's tools, or assume a previously installed plugin remains enabled. Inspect the current callable tools and runtime before choosing a workflow. Claude cloud connectors and Agent/Workflow orchestration, Codex plugins and browser tools, and Gemini/Antigravity skills are runtime-dependent capabilities rather than a shared repository configuration. Local Gemini uses `agy`; its CI surface is the `run-gemini-cli` GitHub App, configured through the [Gemini reference](gemini.md). A capability may be platform-injected rather than repo-configurable: the historical Claude cloud `claude-code-remote` server supplied `send_later`/`list_repos`/`add_repo`; verify availability in that runtime instead of copying local MCP configuration. Name concrete capabilities when coordinating across providers.
