"""Compatibility import; implementation is owned by ``src.prediction.build_snapshot``."""

import sys

from src.prediction import build_snapshot as _implementation

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    sys.exit(_implementation.main())
