"""Compatibility entrypoint; offline builds are owned by src.prediction.upcoming."""

import os
import sys

if __name__ != "__main__" and os.environ.get("FF_ALLOW_RUNTIME_INFERENCE", "1").lower() in {
    "0",
    "false",
    "off",
}:
    from src.artifacts import upcoming as _implementation
else:
    from src.prediction import upcoming as _implementation

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    _implementation.cli()
