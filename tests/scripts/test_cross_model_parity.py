"""Cross-model agent parity guard.

Pins the invariant that every shared agent workflow and audit routine has a thin
wrapper for ALL THREE providers (Claude Code, Codex, Gemini/Antigravity), that
each wrapper points back at its single shared instructions file, and that the
shared instruction files stay provider-neutral.

Why this exists (see todo/cross-model-parity-audit.md, finding F2): the
wrapper/shared-instructions design is only model-agnostic for as long as the
three wrappers stay in lockstep. Nothing else catches "a 4th shared workflow was
added but a provider wrapper was forgotten" or "the shared instructions file was
renamed and a wrapper's pointer rotted". The workflow/routine name lists are
derived from the filesystem (not hardcoded) so this test self-maintains: drop a
new ``agent-workflows/<name>/instructions.md`` and the parity assertions below
immediately demand wrappers for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit

# Provider -> wrapper path template for a SHARED WORKFLOW (skill/prompt).
# `{n}` is the workflow name (the agent-workflows/<n>/ dir name).
WORKFLOW_WRAPPERS: dict[str, str] = {
    "claude": ".claude/skills/{n}/SKILL.md",
    "codex": ".codex/prompts/{n}.md",
    "gemini": ".agents/skills/{n}/SKILL.md",
}

# Provider -> wrapper path template for a SHARED AUDIT ROUTINE.
# Claude's canonical routine wrapper is shim.md (prompt.md is a compat alias).
ROUTINE_WRAPPERS: dict[str, str] = {
    "claude": ".claude/routines/{n}/shim.md",
    "codex": ".codex/automations/{n}/prompt.md",
    "gemini": ".agents/routines/{n}/prompt.md",
}

# Required runtime-value keys every workflow wrapper must inject over the shared
# instructions, and the audit-routine wrapper keys.
REQUIRED_WORKFLOW_KEYS = (
    "WORKFLOW_PROVIDER",
    "WORKFLOW_ENTRYPOINT",
    "WORKFLOW_SHARED_INSTRUCTIONS",
)
REQUIRED_ROUTINE_KEYS = ("AUDIT_PROVIDER", "AUDIT_LABEL")
ALLOWED_AUDIT_LABELS = {"claude-audit", "codex-audit"}

# Provider -> hooks dir. Each provider wires the same deterministic guardrails as
# parallel per-provider adapter scripts (audit P2); the runtime contract of each
# is pinned by its own tests/scripts/test_{provider}_hooks.py.
HOOK_DIRS = {"claude": ".claude/hooks", "codex": ".codex/hooks", "gemini": ".gemini/hooks"}
GUARDRAIL_HOOKS = ("guard-worktree-path", "pre-pr", "ruff-format")

# Provider -> thin memory-sync wrapper over scripts/agent-memory-sync.sh (audit P3).
MEMORY_SYNC_WRAPPERS = {
    "claude": "scripts/claude-memory-sync.sh",
    "codex": "scripts/codex-memory-sync.sh",
    "gemini": "scripts/gemini-memory-sync.sh",
}

# Provider-specific authority pointers that must NOT appear in the shared,
# provider-neutral instruction files (finding F3). The neutral home for the
# tier-by-risk consolidation pattern and the pre-PR rule is AGENTS.md; the worker
# mechanism is the injected WORKFLOW_SUBAGENTS value.
NEUTRALITY_FORBIDDEN = ("Sub-agent contract", "](../../CLAUDE.md)")


def _names(shared_dir: str) -> list[str]:
    """Workflow/routine names, derived from the shared instructions dirs."""
    return sorted(p.parent.name for p in (PROJECT_ROOT / shared_dir).glob("*/instructions.md"))


WORKFLOW_NAMES = _names("agent-workflows")
ROUTINE_NAMES = _names("routines")

# (provider, name) pairs for parametrization.
HOOK_CELLS = [(prov, h) for h in GUARDRAIL_HOOKS for prov in HOOK_DIRS]
WORKFLOW_CELLS = [(prov, n) for n in WORKFLOW_NAMES for prov in WORKFLOW_WRAPPERS]
ROUTINE_CELLS = [(prov, n) for n in ROUTINE_NAMES for prov in ROUTINE_WRAPPERS]


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


# --- the shared brain exists -------------------------------------------------


def test_expected_workflows_present() -> None:
    # Guards against a broken glob vacuously passing every parametrized test.
    assert {"pre-pr-judge", "post-session-critique", "solve-issues"} <= set(WORKFLOW_NAMES)


def test_expected_routines_present() -> None:
    assert {"audit", "tests-audit", "infrastructure-audit"} <= set(ROUTINE_NAMES)


@pytest.mark.parametrize("name", WORKFLOW_NAMES)
def test_shared_workflow_instructions_nonempty(name: str) -> None:
    p = PROJECT_ROOT / "agent-workflows" / name / "instructions.md"
    assert p.is_file() and p.stat().st_size > 0, f"empty/missing shared instructions: {p}"


@pytest.mark.parametrize("name", ROUTINE_NAMES)
def test_shared_routine_instructions_nonempty(name: str) -> None:
    p = PROJECT_ROOT / "routines" / name / "instructions.md"
    assert p.is_file() and p.stat().st_size > 0, f"empty/missing shared instructions: {p}"


# --- every provider wraps every shared workflow ------------------------------


@pytest.mark.parametrize(("provider", "name"), WORKFLOW_CELLS)
def test_workflow_wrapper_exists(provider: str, name: str) -> None:
    rel = WORKFLOW_WRAPPERS[provider].format(n=name)
    assert (PROJECT_ROOT / rel).is_file(), (
        f"{provider} is missing a wrapper for shared workflow '{name}' (expected {rel}). "
        "Every agent-workflows/<name>/ needs a wrapper for all three providers."
    )


@pytest.mark.parametrize(("provider", "name"), WORKFLOW_CELLS)
def test_workflow_wrapper_points_at_shared_instructions(provider: str, name: str) -> None:
    rel = WORKFLOW_WRAPPERS[provider].format(n=name)
    text = _read(rel)
    target = f"agent-workflows/{name}/instructions.md"
    assert target in text, f"{rel} does not reference its shared instructions ({target})"


@pytest.mark.parametrize(("provider", "name"), WORKFLOW_CELLS)
def test_workflow_wrapper_has_required_keys(provider: str, name: str) -> None:
    rel = WORKFLOW_WRAPPERS[provider].format(n=name)
    text = _read(rel)
    missing = [k for k in REQUIRED_WORKFLOW_KEYS if f"{k}=" not in text]
    assert not missing, f"{rel} missing WORKFLOW_* keys: {missing}"
    assert f"WORKFLOW_SHARED_INSTRUCTIONS=agent-workflows/{name}/instructions.md" in text, (
        f"{rel} WORKFLOW_SHARED_INSTRUCTIONS does not point at agent-workflows/{name}/instructions.md"
    )


# --- every provider wraps every shared routine -------------------------------


@pytest.mark.parametrize(("provider", "name"), ROUTINE_CELLS)
def test_routine_wrapper_exists(provider: str, name: str) -> None:
    rel = ROUTINE_WRAPPERS[provider].format(n=name)
    assert (PROJECT_ROOT / rel).is_file(), (
        f"{provider} is missing a wrapper for shared routine '{name}' (expected {rel})."
    )


@pytest.mark.parametrize(("provider", "name"), ROUTINE_CELLS)
def test_routine_wrapper_points_at_shared_instructions(provider: str, name: str) -> None:
    rel = ROUTINE_WRAPPERS[provider].format(n=name)
    text = _read(rel)
    target = f"routines/{name}/instructions.md"
    assert target in text, f"{rel} does not reference its shared instructions ({target})"


@pytest.mark.parametrize(("provider", "name"), ROUTINE_CELLS)
def test_routine_wrapper_has_audit_keys(provider: str, name: str) -> None:
    rel = ROUTINE_WRAPPERS[provider].format(n=name)
    text = _read(rel)
    missing = [k for k in REQUIRED_ROUTINE_KEYS if f"{k}=" not in text]
    assert not missing, f"{rel} missing AUDIT_* keys: {missing}"
    assert any(f"AUDIT_LABEL={lbl}" in text for lbl in ALLOWED_AUDIT_LABELS), (
        f"{rel} AUDIT_LABEL is not one of {sorted(ALLOWED_AUDIT_LABELS)}"
    )


# --- every provider wires the same deterministic guardrail hooks (P2) ---------


@pytest.mark.parametrize(("provider", "hook"), HOOK_CELLS)
def test_guardrail_hook_exists(provider: str, hook: str) -> None:
    rel = f"{HOOK_DIRS[provider]}/{hook}.sh"
    assert (PROJECT_ROOT / rel).is_file(), (
        f"{provider} is missing the '{hook}' guardrail hook (expected {rel}). "
        "All three providers wire guard-worktree-path + pre-pr + ruff-format."
    )


@pytest.mark.parametrize("provider", sorted(MEMORY_SYNC_WRAPPERS))
def test_memory_sync_wrapper_exists(provider: str) -> None:
    rel = MEMORY_SYNC_WRAPPERS[provider]
    assert (PROJECT_ROOT / rel).is_file(), (
        f"{provider} is missing its memory-sync wrapper (expected {rel})."
    )


def test_shared_hook_lib_exists() -> None:
    # The gh-pr tokenizer + find_jq/main_worktree/abs_path/tool_command live once
    # in this shared lib (audit P4); the per-provider lib.sh files source it.
    assert (PROJECT_ROOT / "scripts/agent-hooks-lib.sh").is_file()


@pytest.mark.parametrize("provider", sorted(HOOK_DIRS))
def test_provider_lib_sources_shared_hook_lib(provider: str) -> None:
    lib = PROJECT_ROOT / HOOK_DIRS[provider] / "lib.sh"
    assert lib.is_file(), f"{provider} has no hooks/lib.sh"
    assert "scripts/agent-hooks-lib.sh" in lib.read_text(encoding="utf-8"), (
        f"{provider} hooks/lib.sh must source the shared scripts/agent-hooks-lib.sh "
        "(don't re-triplicate the gh-pr tokenizer)."
    )


# --- the shared instruction files stay provider-neutral (F3) -----------------


@pytest.mark.parametrize("name", WORKFLOW_NAMES)
def test_shared_workflow_instructions_are_provider_neutral(name: str) -> None:
    """Shared workflow instructions must not cite a provider-specific doc as the
    authority for the work. The tier-by-risk pattern and pre-PR rule live in the
    neutral AGENTS.md; the worker mechanism is the injected WORKFLOW_SUBAGENTS.
    (routines/* legitimately enumerate CLAUDE.md/.claude as audit targets, so
    this neutrality guard is scoped to agent-workflows/* only.)
    """
    text = _read(f"agent-workflows/{name}/instructions.md")
    hits = [needle for needle in NEUTRALITY_FORBIDDEN if needle in text]
    assert not hits, (
        f"agent-workflows/{name}/instructions.md leaks provider-specific authority pointers {hits}; "
        "re-point to AGENTS.md / WORKFLOW_SUBAGENTS so the shared brain stays model-agnostic."
    )
