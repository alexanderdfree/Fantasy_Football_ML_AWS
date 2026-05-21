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

# Content-inspector: returns 0 ("safe — skip gate for this file") when the
# diff for FILE between $base and HEAD is additive-only AND free of tokens
# that name model architecture, loss functions, feature lists, or known
# hyperparameter knobs. Returns 1 ("risky — gate") otherwise.
#
# Rationale: the path-only rule used to flag any change to a pipeline file
# as needing a benchmark — including pure-additive edits (new optional
# kwargs with `None` defaults, new dict entries, new helper functions,
# comments, docstrings) that can't possibly change training output. Those
# false-positives forced unnecessary ~2–3hr local retrain sessions for
# changes whose behaviour was provably unchanged.
#
# Two cheap shell checks approximate "could this change model output?":
#   1. Are there removed code lines (excluding blanks/comments/diff headers)?
#      Any deletion of real code may have removed an effect; force benchmark.
#   2. Do added lines reference risky tokens — loss config, feature
#      whitelists, NN architecture pieces, hyperparam keys, scoring/metrics?
#      Touching those names is the strong signal that behaviour can change.
# Either check failing → gate as before. Both passing → skip the gate for
# this file. Conservative direction is "force benchmark"; the false-positive
# mode is recoverable (run benchmark) while the false-negative mode (skip
# benchmark when we shouldn't) is the dangerous one — keep the token list
# generous and prefer to catch suspicious changes here.
_RISKY_TOKENS='(loss_weights|huber_deltas|head_losses|gated_targets|LOSS_WEIGHTS|HUBER_DELTAS|INCLUDE_FEATURES|ATTN_STATIC_FEATURES|ATTN_HISTORY_STATS|attn_d_model|attn_n_heads|nn_backbone_layers|nn_head_hidden|nn_lr|attn_lr|nn_weight_decay|nn_dropout|attn_dropout|nn_epochs|nn_batch_size|attn_batch_size|nn\.Linear|nn\.LayerNorm|nn\.Dropout|nn\.MultiheadAttention|MultiHeadNet|compute_target_metrics|aggregate_fn|predictions_to_fantasy_points|optimizer|learning_rate|criterion|HuberLoss|PoissonNLLLoss|scheduler_type|onecycle_max_lr|cosine_t0)'

is_additive_and_safe() {
  local f="$1"
  local diff_output
  diff_output=$(git diff "$base" HEAD -- "$f" 2>/dev/null || echo "")
  if [ -z "$diff_output" ]; then
    return 0
  fi

  local removed_code
  removed_code=$(printf '%s\n' "$diff_output" \
    | grep -E '^-' \
    | grep -vE '^---|^-[[:space:]]*$|^-[[:space:]]*#' \
    | wc -l | tr -d ' ')
  if [ "$removed_code" -gt 0 ]; then
    return 1
  fi

  local risky_added
  risky_added=$(printf '%s\n' "$diff_output" \
    | grep -E '^\+' \
    | grep -vE '^\+\+\+|^\+[[:space:]]*$|^\+[[:space:]]*#' \
    | grep -cE "$_RISKY_TOKENS" || true)
  if [ "${risky_added:-0}" -gt 0 ]; then
    return 1
  fi

  return 0
}

positions=""
add_pos() { positions="$positions $1"; }
shared_changed=0
skipped_files=""
for f in $changed; do
  case "$f" in
    src/qb/config.py|src/qb/features.py|src/qb/targets.py|src/qb/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos QB; fi ;;
    src/rb/config.py|src/rb/features.py|src/rb/targets.py|src/rb/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos RB; fi ;;
    src/wr/config.py|src/wr/features.py|src/wr/targets.py|src/wr/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos WR; fi ;;
    src/te/config.py|src/te/features.py|src/te/targets.py|src/te/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos TE; fi ;;
    src/k/config.py|src/k/features.py|src/k/targets.py|src/k/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos K; fi ;;
    src/dst/config.py|src/dst/features.py|src/dst/targets.py|src/dst/run_pipeline.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else add_pos DST; fi ;;
    src/shared/pipeline.py|src/shared/models.py|src/shared/neural_net.py|src/shared/aggregate_targets.py|src/shared/training.py|src/shared/evaluation.py|src/shared/backtest.py)
      if is_additive_and_safe "$f"; then skipped_files="$skipped_files $f"; else shared_changed=1; fi ;;
  esac
done
if [ -n "$skipped_files" ]; then
  echo "pre-pr hook: benchmark gate skipped for additive-only files:$skipped_files" >&2
fi
if [ "$shared_changed" -eq 1 ]; then
  positions="$positions QB RB WR TE K DST"
fi

# [docs-only] opt-in: any commit in base..HEAD whose message contains the
# `[docs-only]` literal signals the author asserts the diff has no
# behavioural impact (comment/docstring/import-reorder only). Skips the B2
# benchmark freshness gate. Mirrors `.github/workflows/tests.yml`'s
# detect-job opt-in (same tag, same trust contract). The hook cannot
# verify the assertion — author owns correctness.
if [ -n "$base" ] && [ -n "$positions" ] && git log --format=%B "$base..HEAD" 2>/dev/null | grep -qF '[docs-only]'; then
  positions=""
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
