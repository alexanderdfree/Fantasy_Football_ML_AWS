#!/bin/bash
# PreToolUse hook: pre-PR verification gate.
# Inspired by Spotify Honk Part 3 — block `gh pr create` if deterministic
# verifiers fail (B1) or if a pipeline-affecting change lacks a fresh
# benchmark for the affected position(s) (B2).
set -eu

input=$(cat)
cmd=$(printf '%s' "$input" | /usr/bin/jq -r '.tool_input.command // empty')

# Only gate `gh pr create`. Match at word boundaries to avoid false positives
# from substrings inside other commands' arguments.
if ! [[ "$cmd" =~ (^|[[:space:]&|;\(])gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$|[&|;\)]) ]]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Locate tools: prefer project venv, fall back to PATH.
if [ -x ".venv/bin/ruff" ]; then
  ruff=".venv/bin/ruff"
elif command -v ruff >/dev/null 2>&1; then
  ruff="ruff"
else
  ruff=""
fi
if [ -x ".venv/bin/pytest" ]; then
  pytest=".venv/bin/pytest"
elif command -v pytest >/dev/null 2>&1; then
  pytest="pytest"
else
  pytest=""
fi

fail=0

run_and_capture() {
  # $1=label, rest=command
  label="$1"; shift
  tmp=$(mktemp)
  if ! "$@" >"$tmp" 2>&1; then
    echo "----- pre-pr hook: $label FAILED -----" >&2
    cat "$tmp" >&2
    fail=1
  fi
  rm -f "$tmp"
}

# --- B1: deterministic verifiers ---
if [ -n "$ruff" ]; then
  run_and_capture "ruff check ." "$ruff" check .
  run_and_capture "ruff format --check ." "$ruff" format --check .
else
  echo "pre-pr hook: ruff not found (install or activate .venv)" >&2
  fail=1
fi
if [ -n "$pytest" ]; then
  run_and_capture "pytest -m unit" "$pytest" -m unit -q
else
  echo "pre-pr hook: pytest not found (install or activate .venv)" >&2
  fail=1
fi

# --- B2: metric-regression gate ---
# Derive affected positions from committed branch changes. Pick the best
# available base ref (origin/main → main → origin/master → master) and
# diff via merge-base so committed work is detected, not just uncommitted.
base=""
for ref in origin/main main origin/master master; do
  if git rev-parse --verify --quiet "$ref" >/dev/null 2>&1; then
    if base=$(git merge-base "$ref" HEAD 2>/dev/null) && [ -n "$base" ]; then
      break
    fi
    base=""
  fi
done
if [ -n "$base" ]; then
  changed=$(git diff --name-only "$base" HEAD 2>/dev/null || true)
else
  # No upstream ref found — fall back to uncommitted changes only.
  changed=$(git diff --name-only HEAD 2>/dev/null || true)
fi

positions=""
add_pos() { positions="$positions $1"; }
shared_changed=0
for f in $changed; do
  case "$f" in
    src/qb/config.py|src/qb/features.py|src/qb/targets.py|src/qb/run_pipeline.py) add_pos QB ;;
    src/rb/config.py|src/rb/features.py|src/rb/targets.py|src/rb/run_pipeline.py) add_pos RB ;;
    src/wr/config.py|src/wr/features.py|src/wr/targets.py|src/wr/run_pipeline.py) add_pos WR ;;
    src/te/config.py|src/te/features.py|src/te/targets.py|src/te/run_pipeline.py) add_pos TE ;;
    src/k/config.py|src/k/features.py|src/k/targets.py|src/k/run_pipeline.py) add_pos K ;;
    src/dst/config.py|src/dst/features.py|src/dst/targets.py|src/dst/run_pipeline.py) add_pos DST ;;
    src/shared/pipeline.py|src/shared/models.py|src/shared/neural_net.py|src/shared/aggregate_targets.py|src/shared/training.py|src/shared/evaluation.py|src/shared/backtest.py)
      shared_changed=1 ;;
  esac
done
if [ "$shared_changed" -eq 1 ]; then
  positions="$positions QB RB WR TE K DST"
fi

if [ -n "$positions" ]; then
  positions=$(printf '%s\n' $positions | sort -u | tr '\n' ' ')

  # Reference mtime: newest mtime among any pipeline-affecting file in the tree.
  pipeline_files=(
    src/shared/pipeline.py src/shared/models.py src/shared/neural_net.py
    src/shared/aggregate_targets.py src/shared/training.py
    src/shared/evaluation.py src/shared/backtest.py
  )
  for p in qb rb wr te k dst; do
    for s in config features targets run_pipeline; do
      pipeline_files+=("src/$p/$s.py")
    done
  done
  ref_ts=0
  for pf in "${pipeline_files[@]}"; do
    [ -f "$pf" ] || continue
    t=$(stat -f %m "$pf" 2>/dev/null || stat -c %Y "$pf" 2>/dev/null || echo 0)
    if [ "$t" -gt "$ref_ts" ]; then ref_ts="$t"; fi
  done

  missing=""
  for pos in $positions; do
    found=0
    for bf in benchmark_history/*.json; do
      [ -f "$bf" ] || continue
      bts=$(stat -f %m "$bf" 2>/dev/null || stat -c %Y "$bf" 2>/dev/null || echo 0)
      if [ "$bts" -gt "$ref_ts" ]; then
        if /usr/bin/jq -e --arg p "$pos" '.positions | index($p)' "$bf" >/dev/null 2>&1; then
          found=1
          break
        fi
      fi
    done
    if [ "$found" -eq 0 ]; then
      missing="$missing $pos"
    fi
  done

  if [ -n "$missing" ]; then
    echo "----- pre-pr hook: metric-regression gate FAILED -----" >&2
    echo "pipeline files changed but no fresh benchmark_history/ entry covers:$missing" >&2
    echo "run one of:" >&2
    for pos in $missing; do
      lpos=$(printf '%s' "$pos" | tr '[:upper:]' '[:lower:]')
      echo "  python -m src.${lpos}.run_pipeline" >&2
    done
    echo "  python -m src.benchmarking.benchmark$missing" >&2
    fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo "----- pre-pr hook: blocking gh pr create (see errors above) -----" >&2
  exit 2
fi

# B3 nudge: deterministic checks alone don't catch scope drift ("agent did more
# than I asked"). The pre-pr-judge skill spawns a worker subagent for that.
# Belt-and-suspenders to the CLAUDE.md "When making changes" instruction.
echo "pre-pr hook: deterministic checks passed. if pre-pr-judge has not run this session, invoke it before this PR opens (catches scope creep — see .claude/skills/pre-pr-judge/SKILL.md)." >&2

exit 0
