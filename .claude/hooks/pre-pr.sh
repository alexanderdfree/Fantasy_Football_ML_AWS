#!/bin/bash
# PreToolUse hook: pre-PR verification gate.
# Inspired by Spotify Honk Part 3 — block `gh pr create` if deterministic
# verifiers fail (B1) or if a pipeline-affecting change lacks a fresh
# benchmark for the affected position(s) (B2).
set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/lib.sh
. "$script_dir/lib.sh"

# Resolve jq: prefer PATH, fall back to common absolute install locations so the
# hook works whether or not jq lives at /usr/bin (WSL/dev boxes differ from CI).
jq_bin=""
for _c in jq /usr/bin/jq /usr/local/bin/jq /opt/homebrew/bin/jq /home/linuxbrew/.linuxbrew/bin/jq; do
  if command -v "$_c" >/dev/null 2>&1; then jq_bin="$_c"; break; fi
done
[ -n "$jq_bin" ] || exit 0  # no jq → cannot parse; skip gate rather than block all Bash

input=$(cat)
cmd=$(printf '%s' "$input" | "$jq_bin" -r '.tool_input.command // empty')

# Cheap pre-filter: any real `gh pr create` contains `gh`, so skip the O(n)
# tokenizer for the ~all Bash commands that don't (parity with the post-pr hooks).
case "$cmd" in *gh*) ;; *) exit 0 ;; esac

# Only gate an ACTUAL `gh pr create` invocation. The shell-parser-aware matcher
# strips quotes/heredocs/comments and splits on ; | & before testing each
# segment, so the literal token sequence inside a quoted string or another
# command's args does not trip the gate — a false positive here is a PreToolUse
# blocker (exit 2) that wedges the session (#894).
if ! claude_command_invokes_gh_pr_create "$cmd"; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# --- Early bail: [docs-only] tag opts out of B1 (ruff/pytest) AND B2 (benchmark).
# Mirrors tests.yml's detect job convention and batch-image.yml / _detect-positions.yml
# tag-respect. CI's tests-pass already short-circuits on the same tag. Trust contract:
# the author asserts the change is non-behavioral; CI does not verify. Saves the gate
# from environmental flakes (xdist races, miniforge3 vs .venv) on PRs that don't need
# pytest. Skip if no upstream ref can be resolved — better to run the gate than
# silently skip on a fresh repo.

# Subject-line scan only — keep awk to line 1 + `* `-prefixed bullets (squash-merge
# constituent subjects). Flat grep against full %B false-positived on commits that
# *described* the [docs-only] mechanism in prose body text. Per-commit iteration
# (rather than catting the whole range into one stream) prevents a body line of
# commit N+1 looking like a subject of commit N due to lack of separators in
# `git log --format=%B`.
docs_only_in_range() {
  local range="$1"
  local sha
  for sha in $(git log --format=%H "$range" 2>/dev/null); do
    if git log -1 --format=%B "$sha" 2>/dev/null | awk 'NR==1 || /^\* /' | grep -qF '[docs-only]'; then
      return 0
    fi
  done
  return 1
}

docs_only_base=""
for ref in origin/main main origin/master master; do
  if git rev-parse --verify --quiet "$ref" >/dev/null 2>&1; then
    if docs_only_base=$(git merge-base "$ref" HEAD 2>/dev/null) && [ -n "$docs_only_base" ]; then
      break
    fi
    docs_only_base=""
  fi
done
if [ -n "$docs_only_base" ] && docs_only_in_range "$docs_only_base..HEAD"; then
  echo "pre-pr hook: [docs-only] detected in BASE..HEAD commit subject lines — skipping ruff/pytest/benchmark gates" >&2
  exit 0
fi

# Locate tools: prefer a project venv, fall back to PATH. Handle both the Unix
# venv layout (`.venv/bin/<tool>`) and the Windows layout
# (`.venv/Scripts/<tool>.exe`; pip may not install a standalone pytest.exe, so
# fall back to the venv interpreter's `python.exe -m pytest`). In a git worktree
# the venv is usually created once in the main worktree and not duplicated per
# feature checkout, so after the local `.venv` also probe the main worktree's
# (resolved via `git worktree list`, whose first entry is always the main one).
venv_roots=(".venv")
main_wt=$(git worktree list --porcelain 2>/dev/null \
  | awk 'NR==1 && /^worktree /{print substr($0, 10); exit}' | tr -d '\r')
if [ -n "$main_wt" ] && [ -d "$main_wt/.venv" ]; then
  venv_roots+=("$main_wt/.venv")
fi

# ruff is a standalone executable in every layout, so a single token suffices.
ruff=""
for vr in "${venv_roots[@]}"; do
  if [ -x "$vr/bin/ruff" ]; then ruff="$vr/bin/ruff"; break; fi
  if [ -x "$vr/Scripts/ruff.exe" ]; then ruff="$vr/Scripts/ruff.exe"; break; fi
done
if [ -z "$ruff" ] && command -v ruff >/dev/null 2>&1; then
  ruff="ruff"
fi

# pytest is kept as an ARRAY: the Windows `python.exe -m pytest` fallback is
# multiple argv tokens, and a scalar `$pytest` would be mis-run as a single
# command name containing spaces. Invoked below as `"${pytest_cmd[@]}" -m unit`.
pytest_cmd=()
for vr in "${venv_roots[@]}"; do
  if [ -x "$vr/bin/pytest" ]; then pytest_cmd=("$vr/bin/pytest"); break; fi
  if [ -x "$vr/Scripts/pytest.exe" ]; then pytest_cmd=("$vr/Scripts/pytest.exe"); break; fi
  if [ -x "$vr/Scripts/python.exe" ]; then pytest_cmd=("$vr/Scripts/python.exe" -m pytest); break; fi
done
if [ "${#pytest_cmd[@]}" -eq 0 ] && command -v pytest >/dev/null 2>&1; then
  pytest_cmd=("pytest")
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
  # --no-cache: CI runs on a clean runner with no ruff cache; the local
  # mtime+size cache can false-skip a same-size rewrite landing in the same
  # mtime tick, letting the gate pass what CI would flag.
  run_and_capture "ruff check ." "$ruff" check --no-cache .
  run_and_capture "ruff format --check ." "$ruff" format --check --no-cache .
else
  echo "pre-pr hook: ruff not found (install or activate .venv)" >&2
  fail=1
fi
if [ "${#pytest_cmd[@]}" -gt 0 ]; then
  run_and_capture "pytest -m unit" "${pytest_cmd[@]}" -m unit -q
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
_RISKY_TOKENS='(loss_weights|huber_deltas|head_losses|gated_targets|include_features|specific_features|contextual_features|all_features|attn_static_features|attn_history_stats|attn_kick_stats|opp_attn_history_stats|SPECIFIC_FEATURES|CONTEXTUAL_FEATURES|ALL_FEATURES|ATTN_KICK_STATS|LOSS_WEIGHTS|HUBER_DELTAS|INCLUDE_FEATURES|ATTN_STATIC_FEATURES|ATTN_HISTORY_STATS|TARGETS|attn_d_model|attn_n_heads|nn_backbone_layers|nn_head_hidden|nn_lr|attn_lr|nn_weight_decay|nn_dropout|attn_dropout|nn_epochs|nn_batch_size|attn_batch_size|nn_patience|attn_patience|attn_max_seq_len|attn_weight_decay|nn_non_negative_targets|gate_weight|attn_gate_weight|train_attention_nn|train_lightgbm|train_elasticnet|train_ridge|train_base_nn|attn_history_structure|attn_history_builder_fn|td_model_type|alpha_grids|ridge_alpha_grids|enet_alpha_grids|n_cv_folds|ridge_cv_folds|nn\.Linear|nn\.LayerNorm|nn\.Dropout|nn\.MultiheadAttention|MultiHeadNet|compute_target_metrics|aggregate_fn|aggregate_targets|predictions_to_fantasy_points|optimizer|learning_rate|criterion|HuberLoss|PoissonNLLLoss|scheduler_type|onecycle_max_lr|cosine_t0|cosine_t_mult|cosine_eta_min|np\.clip|torch\.clamp)'

is_additive_and_safe() {
  local f="$1"
  local diff_output
  # `-b` (ignore-space-change) so pure re-indentation (e.g. wrapping a block
  # in a new `if`/`with`) doesn't appear on either side and trip the risky-
  # token check against the re-indented existing code. Without `-b`, wrapping
  # a Ridge block in `if cfg.get("train_ridge", True):` shows every interior
  # line as removed + added, and lines that happen to contain `ridge_alpha_
  # grids` / `n_cv_folds` etc. false-positive the risky-token check even when
  # no semantic change occurred. `-b` ignores changes in the *amount* of
  # whitespace but not whitespace inside strings/comments, so string-content
  # edits are still gated correctly.
  diff_output=$(git diff -b "$base" HEAD -- "$f" 2>/dev/null || echo "")
  if [ -z "$diff_output" ]; then
    return 0
  fi

  # Symmetric token check: a diff is risky if EITHER added OR removed code
  # lines reference loss-config / model-architecture / feature-list /
  # hyperparam names. The previous version flagged *all* removed code,
  # which false-positived on backwards-compatible signature widening
  # (e.g. `def run(seed=42):` -> `def run(seed=42, config=None):` plus
  # `CONFIG` -> `config or CONFIG`) — net behaviour preserved under the
  # default value, no risk to model output. Catching by token instead of
  # by line-count handles that case correctly while still gating real
  # behavioural deletions (e.g. removing an `np.clip` or a `criterion =`).
  local risky_minus risky_plus risky_plus_intro
  risky_minus=$(printf '%s\n' "$diff_output" \
    | grep -E '^-' \
    | grep -vE '^---|^-[[:space:]]*$|^-[[:space:]]*#' \
    | grep -cE "$_RISKY_TOKENS" || true)
  risky_plus=$(printf '%s\n' "$diff_output" \
    | grep -E '^\+' \
    | grep -vE '^\+\+\+|^\+[[:space:]]*$|^\+[[:space:]]*#' \
    | grep -cE "$_RISKY_TOKENS" || true)

  if [ "${risky_minus:-0}" -gt 0 ]; then
    return 1
  fi

  # Introductory default-True `train_*` gate exemption: a `+` line of the
  # form `if cfg.get("train_NAME", True):` is provably behaviour-preserving
  # for the default branch — when the default fires, the gated code runs
  # exactly as before, and the gate-flipping callers (e.g. the tuner setting
  # train_ridge=False per trial) are caught when their cfg edit lands. Without
  # this exemption, the PR that *adds* train_ridge / train_base_nn to
  # _RISKY_TOKENS fails the gate on the gate-definition line itself. Scope
  # restricted to `train_NAME` keys (every existing gate name is a boolean
  # toggle starting with `train_`) so the exemption can't be used to slip
  # through edits like `if cfg.get("nn_lr", True):` that match the same
  # outer shape but touch hyperparam knobs.
  # Ruff format normalises string literals to double quotes in this repo
  # (pyproject.toml's [tool.ruff.format] inherits the default `quote-style =
  # "double"`), so matching only `"train_NAME"` is sufficient — a single-
  # quoted intro would fail `ruff format --check .` upstream in B1 anyway.
  risky_plus_intro=$(printf '%s\n' "$diff_output" \
    | grep -E '^\+' \
    | grep -vE '^\+\+\+|^\+[[:space:]]*$|^\+[[:space:]]*#' \
    | grep -E "$_RISKY_TOKENS" \
    | grep -cE 'if[[:space:]]+cfg\.get\("train_[a-z_]+",[[:space:]]*True\):' || true)

  if [ "${risky_plus:-0}" -gt "${risky_plus_intro:-0}" ]; then
    return 1
  fi

  return 0
}

# The B2 brain lives in src/scripts/pre_pr_bench_check.py (unit-tested Python,
# stdlib-only): scoping is single-sourced in scope_positions.compute_benchmark_scope
# (per-position prefix rule + shared regex + explicit Batch/deps exemptions), and
# evidence is CONTENT-FINGERPRINTED — a position passes when some
# benchmark_history/*.json entry's code_fingerprints[POS] matches this HEAD's
# fingerprint (see src/scripts/bench_fingerprint.py), so rebase/stash/checkout
# mtime churn can no longer invalidate real evidence, and evidence recorded
# against DIFFERENT code can no longer pass. Two legacy mtime tiers remain: a
# self-retiring one for pre-fingerprint history entries and a permanent one for
# bare `run_pipeline` artifacts under {pos}/outputs/models (with a nudge).
# Shared-path changes still require evidence on at least ONE position (shared
# code runs the same path for every position; the risky-token filter below is
# the safety net for partial-effect changes — rationale preserved from the old
# inline implementation).
#
# The hook keeps three cheap file filters before the evaluator:
#   tier 0: AST-inert (comments/docstrings/formatting only — provably inert)
#   tier 1: is_additive_and_safe (additive-only, no risky tokens — see above)
#   exempt-class paths (src/batch/**, requirements.txt) bypass the filters and
#   flow straight to the evaluator so it can REPORT them as exempt-not-gated.
# Evaluator protocol: exit 0 + stdout line 1 PASS|FAIL (+ detail lines). A
# nonzero exit means the evaluator itself broke — warn loudly and FAIL OPEN
# (parity with the missing-jq behavior above; a false block wedges the session).
py_bin=""
for _p in python3 python; do
  if command -v "$_p" >/dev/null 2>&1; then py_bin="$_p"; break; fi
done

candidates=""
for f in $changed; do
  case "$f" in
    src/*|requirements.txt) candidates="$candidates$f"$'\n' ;;
  esac
done

if [ -n "$candidates" ]; then
  if [ -z "$py_bin" ]; then
    echo "pre-pr hook: WARNING — python3 not found; skipping the benchmark gate (fail-open)" >&2
  else
    inert=""
    if [ -n "$base" ]; then
      inert=$(printf '%s' "$candidates" \
        | "$py_bin" -m src.scripts.pre_pr_bench_check inert --base "$base" 2>/dev/null) || inert=""
    fi

    kept=""
    skipped_files=""
    inert_files=""
    for f in $candidates; do
      case "$f" in
        src/batch/*|requirements.txt)
          # Exempt-class: evaluator reports these, never gates on them.
          kept="$kept$f"$'\n'
          continue ;;
      esac
      if [ -n "$inert" ] && printf '%s\n' "$inert" | grep -qFx -- "$f"; then
        inert_files="$inert_files $f"
      elif is_additive_and_safe "$f"; then
        skipped_files="$skipped_files $f"
      else
        kept="$kept$f"$'\n'
      fi
    done
    if [ -n "$inert_files" ]; then
      echo "pre-pr hook: benchmark gate skipped for AST-inert (comment/docstring/format-only) files:$inert_files" >&2
    fi
    if [ -n "$skipped_files" ]; then
      echo "pre-pr hook: benchmark gate skipped for additive-only files:$skipped_files" >&2
    fi

    eval_err=$(mktemp)
    if out=$(printf '%s' "$kept" | "$py_bin" -m src.scripts.pre_pr_bench_check evaluate 2>"$eval_err"); then
      verdict=$(printf '%s\n' "$out" | head -n 1)
      detail=$(printf '%s\n' "$out" | tail -n +2)
      [ -n "$detail" ] && printf '%s\n' "$detail" >&2
      case "$verdict" in
        PASS) ;;
        FAIL)
          echo "----- pre-pr hook: metric-regression gate FAILED (see detail above) -----" >&2
          fail=1 ;;
        *)
          echo "pre-pr hook: WARNING — unexpected benchmark-gate output '$verdict'; failing open" >&2 ;;
      esac
    else
      echo "pre-pr hook: WARNING — benchmark-gate evaluator errored; failing open:" >&2
      cat "$eval_err" >&2
    fi
    rm -f "$eval_err"
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo "----- pre-pr hook: blocking gh pr create (see errors above) -----" >&2
  exit 2
fi

# B3 nudge: deterministic checks alone don't catch scope drift ("agent did more
# than I asked"). The pre-pr-judge skill/prompt checks that layer. Keep the
# deterministic gate single-sourced, but point the nudge at the active agent's
# entrypoint when Codex wraps this hook.
if [ "${CODEX_PRE_PR_WRAPPER:-}" = "1" ]; then
  echo "pre-pr hook: deterministic checks passed. if /prompts:pre-pr-judge has not run this session, invoke it before this PR opens (catches scope creep — wrapper .codex/prompts/pre-pr-judge.md; shared behavior agent-workflows/pre-pr-judge/instructions.md)." >&2
else
  echo "pre-pr hook: deterministic checks passed. if pre-pr-judge has not run this session, invoke it before this PR opens (catches scope creep — wrapper .claude/skills/pre-pr-judge/SKILL.md; shared behavior agent-workflows/pre-pr-judge/instructions.md)." >&2
fi

exit 0
