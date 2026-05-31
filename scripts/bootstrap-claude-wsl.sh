#!/usr/bin/env bash
# Bootstrap the MACHINE-LOCAL Claude Code config on a fresh box (e.g. a new WSL2
# dev env). These are the only Claude Code settings that do NOT travel with the
# git repo, so they must be recreated by hand on each machine:
#
#   1. ~/.claude/settings.json            -- global user settings (effort level,
#                                            experimental agent-teams opt-in, the
#                                            global worktree-parent guard hook,
#                                            theme + notif prefs).
#   2. ~/.claude/hooks/worktree-parent-guard.sh -- the global PreToolUse hook the
#                                            settings above reference.
#
# What this does NOT touch (deliberately):
#   * The PROJECT config -- .claude/settings.json and .claude/hooks/*.sh are
#     tracked in git, so `git clone`/`pull` already brings them. Nothing to do.
#   * .claude/settings.local.json -- gitignored, machine-local permission
#     allowlist; it regenerates itself as you approve tool calls.
#   * The Anthropic server-pushed experiment flags (the `tengu_*` blocks cached
#     in ~/.claude.json) -- those are account/version-tied and refresh on their
#     own. You can't and shouldn't copy them.
#   * The PROJECT's own FF_* / BLAS env (FF_DEVICE, OPENBLAS_NUM_THREADS, ...) --
#     that is a separate concern; `source scripts/wsl-env.sh` before a run.
#
# Idempotent: safe to re-run. If ~/.claude/settings.json already exists it is
# backed up (timestamped) and the desired keys are deep-merged in via jq, so
# unrelated settings you already have are preserved. (Note: jq object-merge
# REPLACES arrays, so the `permissions.allow` / `hooks.PreToolUse` arrays below
# overwrite any existing same-named arrays -- the backup is your safety net.)
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
HOOKS_DIR="${CLAUDE_DIR}/hooks"
SETTINGS="${CLAUDE_DIR}/settings.json"
GUARD="${HOOKS_DIR}/worktree-parent-guard.sh"

mkdir -p "$HOOKS_DIR"

# --- 1. Global hook: worktree-parent-guard.sh --------------------------------
# Embedded verbatim (this script must run on a fresh box where the repo's own
# hooks are the only ones present and this GLOBAL hook does not yet exist).
cat > "$GUARD" <<'GUARD_EOF'
#!/usr/bin/env bash
# PreToolUse guard (Edit|Write|MultiEdit): block writes that target the PARENT
# checkout while the session runs inside a git worktree.
#
# Why this exists: when CLAUDE_PROJECT_DIR is a worktree
# (.../.claude/worktrees/<name>/), absolute file_paths that point at the parent
# checkout (.../<repo>/src/...) silently land on the parent's branch (usually
# main, often carrying unrelated WIP) instead of the feature branch. This bit 5
# PRs in a row (#284/#354/#370/#378/#381) despite the rule living in both
# auto-memory and CLAUDE.md — prose guidance never fired at path-construction
# time. This hook enforces it mechanically.
#
# Contract: exit 0 = allow, exit 2 = block (stderr is fed back to the model).
# Strict no-op when not in a worktree, so it is safe in every project.
set -euo pipefail

# Not in a worktree → nothing to guard.
case "${CLAUDE_PROJECT_DIR:-}" in
  */.claude/worktrees/*) ;;
  *) exit 0 ;;
esac

# Parent checkout root = everything before /.claude/worktrees/.
parent_root="${CLAUDE_PROJECT_DIR%%/.claude/worktrees/*}"

# Extract the target path. jq is already a hook dependency in this repo.
file_path="$(jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"
[ -n "$file_path" ] || exit 0

# Inside the worktree → correct target, allow. (Checked first because the
# worktree path is itself under parent_root.)
case "$file_path" in
  "$CLAUDE_PROJECT_DIR"/*) exit 0 ;;
esac

# Under the parent root but NOT the worktree → a parent-checkout write. Block.
case "$file_path" in
  "$parent_root"/*)
    echo "BLOCKED: '$file_path' targets the PARENT checkout ($parent_root), not this worktree." >&2
    echo "This session runs in the worktree: $CLAUDE_PROJECT_DIR" >&2
    echo "Re-prefix the path to the worktree, e.g.:" >&2
    echo "  ${file_path/#$parent_root/$CLAUDE_PROJECT_DIR}" >&2
    echo "(Absolute paths from sub-agents/plan files are PARENT paths — re-prefix before Edit/Write.)" >&2
    exit 2
    ;;
esac

# Anywhere else (/tmp, ~/.claude/.../memory, relative paths) → allow.
exit 0
GUARD_EOF
chmod +x "$GUARD"
echo "[bootstrap] wrote hook  -> $GUARD"

# --- 2. Global settings.json -------------------------------------------------
# Desired machine-local settings, mirroring the macOS box. The two `env` keys
# are the real behavioral opt-ins:
#   CLAUDE_CODE_EFFORT_LEVEL=max          -> max thinking/effort budget
#   CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 -> experimental agent-teams tooling
read -r -d '' DESIRED <<'JSON_EOF' || true
{
  "env": {
    "CLAUDE_CODE_EFFORT_LEVEL": "max",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "permissions": {
    "allow": [
      "Bash(git fetch:*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/worktree-parent-guard.sh",
            "statusMessage": "Checking worktree path"
          }
        ]
      }
    ]
  },
  "skipWorkflowUsageWarning": true,
  "theme": "dark",
  "inputNeededNotifEnabled": true
}
JSON_EOF

backup_existing() {
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  cp "$SETTINGS" "${SETTINGS}.bak.${stamp}"
  echo "[bootstrap] backed up existing settings -> ${SETTINGS}.bak.${stamp}"
}

if command -v jq >/dev/null 2>&1; then
  if [ -f "$SETTINGS" ]; then
    backup_existing
    tmp="$(mktemp)"
    jq --argjson desired "$DESIRED" '. * $desired' "$SETTINGS" > "$tmp"
    mv "$tmp" "$SETTINGS"
    echo "[bootstrap] merged settings -> $SETTINGS"
  else
    printf '%s\n' "$DESIRED" | jq . > "$SETTINGS"
    echo "[bootstrap] created settings -> $SETTINGS"
  fi
else
  echo "[bootstrap] WARN: jq not found. Writing settings without a merge," >&2
  echo "[bootstrap]       and the guard hook itself NEEDS jq at runtime." >&2
  echo "[bootstrap]       Install it:  sudo apt-get install -y jq" >&2
  [ -f "$SETTINGS" ] && backup_existing
  printf '%s\n' "$DESIRED" > "$SETTINGS"
  echo "[bootstrap] wrote settings (unmerged) -> $SETTINGS"
fi

# --- Summary -----------------------------------------------------------------
cat <<'DONE'

[bootstrap] Done. Machine-local Claude Code config is in place.

  Next, for actually RUNNING this project on WSL (separate from Claude config):
    source scripts/wsl-env.sh     # BLAS thread caps + LightGBM cores + S3 sync

  Verify Claude picked up the settings: open Claude Code and run  /status
  (effort should read "max"; agent-teams tooling should be available).
DONE
