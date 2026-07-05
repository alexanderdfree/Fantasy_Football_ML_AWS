---
description: Run the repo's deterministic pre-PR checks before gh pr create
argument-hint: [BASE=origin/main]
---

Run the Final-Project deterministic pre-PR gate manually.

Use `BASE` if supplied; otherwise use `origin/main`.

1. Check for `[docs-only]` in commit subjects across `BASE..HEAD`. If present, verify the diff is truly non-behavioral before skipping expensive gates.
2. Run `ruff check .`.
3. Run `ruff format --check .`.
4. Run `pytest -m unit -q`.
5. Scope `git diff --name-only BASE...HEAD` the way the hook does
   (`src/scripts/scope_positions.py::compute_benchmark_scope`; `tests/` paths are ignored):
   - ANY file under `src/{qb,rb,wr,te,k,dst}/` scopes that position (benchmark evidence required for it).
   - `src/shared/**`, `src/data/**`, `src/features/**`, `src/config.py`, `src/__init__.py` = the shared arm (evidence required on at least one position).
   - `src/batch/**` and `requirements.txt` are exempt-but-reported (no benchmark evidence).
6. For each scoped position, require benchmark evidence matched by content fingerprint, not mtime:
   some `benchmark_history/*.json` entry's `code_fingerprints[POS]` must match the branch HEAD's
   fingerprint (`src/scripts/bench_fingerprint.py`) — produce it with
   `python -m src.benchmarking.benchmark <POS ...>`. A bare `python -m src.<pos>.run_pipeline` run
   also passes via the permanent `{pos}/outputs/models` mtime fallback (accepted with a nudge).
   AST-inert edits (a gated file whose ENTIRE BASE..HEAD diff is comments/docstrings/formatting)
   and additive-only-no-risky-token edits bypass this step; `[docs-only]` (step 1) bypasses both gates.
   For shared-arm changes, any one position's evidence is acceptable unless the diff affects
   position-specific behavior (e.g. a K/DST-only branch inside `src/shared/`) — then benchmark
   that position; the fingerprint gate is position-blind for `src/shared/**` and cannot catch this.
7. Run `/prompts:pre-pr-judge` before `gh pr create` unless the change is trivial.

Report the exact commands run, pass/fail status, and any remaining gate friction.
