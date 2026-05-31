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
5. Inspect `git diff --name-only BASE...HEAD` for position pipeline edits:
   - `src/{qb,rb,wr,te,k,dst}/{config,features,targets,run_pipeline}.py`
   - `src/shared/*.py`, `src/data/*.py`, `src/features/*.py`, `src/config.py`
6. If model-affecting pipeline files changed, require fresh evidence newer than those edits:
   - `python -m src.<pos>.run_pipeline` for each affected position, or
   - `python -m src.benchmarking.benchmark <POS ...>` when a benchmark JSON is needed.
   For shared pipeline changes, at least one fresh position run is acceptable unless the diff affects position-specific behavior.
7. Run `/prompts:pre-pr-judge` before `gh pr create` unless the change is trivial.

Report the exact commands run, pass/fail status, and any remaining gate friction.
