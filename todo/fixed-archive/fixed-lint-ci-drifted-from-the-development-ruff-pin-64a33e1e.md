> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Lint CI drifted from the development Ruff pin
- **File(s):** [../.github/workflows/tests.yml](../../.github/workflows/tests.yml), [../tests/test_dependency_pins.py](../../tests/test_dependency_pins.py) (audit #1509; PR pending).
- **What:** The lint job independently pinned Ruff 0.15.16 while development requirements had advanced to 0.15.20, and then 0.16.6. Existing dependency parity tests did not inspect the lint install command.
- **Fix:** Read the exact Ruff requirement from `requirements-dev.txt` in the lint install step. The contract test executes that step with a stub `uv` for both the current requirement and a future synthetic pin; development and GPU pins are checked together.
- **Lesson:** A second hardcoded pin inevitably drifts. Test the consuming install command against the canonical requirement so future dependency bumps reach CI automatically.
