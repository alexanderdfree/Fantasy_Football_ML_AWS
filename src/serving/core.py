"""Select the artifact reader or the optional local prediction builder."""

import os
import sys

if os.environ.get("FF_ALLOW_RUNTIME_INFERENCE", "1").lower() in {"0", "false", "off"}:
    from src.artifacts import snapshot_runtime as _implementation
else:
    from src.prediction import historical as _implementation

sys.modules[__name__] = _implementation
