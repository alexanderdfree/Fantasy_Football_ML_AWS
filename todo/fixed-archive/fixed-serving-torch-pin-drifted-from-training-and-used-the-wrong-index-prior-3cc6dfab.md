> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Serving Torch pin drifted from training and used the wrong index priority
- **File(s):** [../Dockerfile](../../Dockerfile), [../requirements-dev.txt](../../requirements-dev.txt), [../requirements-gpu.txt](../../requirements-gpu.txt), [../src/batch/requirements.txt](../../src/batch/requirements.txt) (dependency refresh, PR pending).
- **What:** Serving still requested Torch 2.12.0 while local development and Batch requested 2.12.1. Its uv command also listed PyPI as the extra index, giving it priority over the intended CPU index under uv's first-index strategy.
- **Fix:** Align all four environments on Torch 2.14.0 and give the CPU index priority in the serving install. Resolve CPU and CUDA dependency sets for Python 3.12 before shipping; keep the existing CUDA 13.0 variant.
- **Lesson:** Dependency parity includes Dockerfile install commands and index order, not just requirements files. A matching public Torch version does not prove that serving installs the CPU wheel.
