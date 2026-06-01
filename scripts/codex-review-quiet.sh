#!/usr/bin/env bash
# Run Codex review while filtering known Codex/plugin loader noise from stderr.
set -u

tmpdir="$(mktemp -d)"
stdout_file="$tmpdir/stdout"
stderr_file="$tmpdir/stderr"
filtered_stderr="$tmpdir/stderr.filtered"
trap 'rm -rf "$tmpdir"' EXIT

RUST_LOG="${RUST_LOG:-error}" codex review "$@" >"$stdout_file" 2>"$stderr_file"
rc=$?

cat "$stdout_file"

grep -vE \
  -e 'WARN codex_core_skills::loader: ignoring interface\.icon_(small|large): icon path with' \
  -e 'ERROR codex_core::session::session: failed to load skill .*invalid name: exceeds maximum length' \
  -e 'WARN codex_protocol::openai_models: Model personality requested but model_messages is missing' \
  -e 'WARN codex_analytics::reducer: dropping turn tool count update' \
  "$stderr_file" >"$filtered_stderr" || true

if [ -s "$filtered_stderr" ]; then
  cat "$filtered_stderr" >&2
fi

exit "$rc"
