#!/usr/bin/env bash
# shellcheck shell=bash
#
# pytest-fair.sh — run pytest in its own systemd user scope so concurrent runs
# (e.g. two Claude Code windows) share CPU fairly instead of oversubscribing.
#
# `pytest -n auto` picks a worker count per-invocation and is blind to other
# processes, so two concurrent runs spawn 2x the xdist workers on the same
# cores. Putting each run in a transient cgroup scope with equal CPUWeight lets
# the kernel split CPU ~50/50 under contention while each run still bursts to
# full when the other is idle (work-conserving) — strictly better than guessing
# from a laggy loadavg. cgroups do the arbitration; nothing is added to the test
# path (conftest.py / pyproject.toml are untouched, so plain `pytest` and CI are
# byte-identical).
#
# Portable: falls back to plain `pytest` wherever user scopes aren't usable
# (macOS, native Windows, CI runners, or a WSL2 box without the cgroup v2 `cpu`
# controller delegated to the user slice), so it is safe to use everywhere.
#
#   Usage:  scripts/pytest-fair.sh [pytest args]      # e.g. -m unit -q
#   Env:    PYTEST_FAIR_CPUWEIGHT=100   # scope weight (systemd default is 100)
#           PYTEST_FAIR_NO_SCOPE=1      # force plain pytest (skip the scope)
set -euo pipefail

weight="${PYTEST_FAIR_CPUWEIGHT:-100}"

if [ -z "${PYTEST_FAIR_NO_SCOPE:-}" ] \
  && command -v systemd-run >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1 \
  && grep -qw cpu /sys/fs/cgroup/user.slice/cgroup.subtree_control 2>/dev/null; then
  echo "[pytest-fair] running pytest in a user scope (CPUWeight=${weight})" >&2
  exec systemd-run --user --scope --quiet --collect \
    -p "CPUWeight=${weight}" -- pytest "$@"
fi

echo "[pytest-fair] systemd user cpu scope unavailable — running pytest directly" >&2
exec pytest "$@"
