"""Compatibility import; implementation is owned by ``src.prediction.comparison_snapshot``."""

import sys

from src.prediction import comparison_snapshot as _implementation

sys.modules[__name__] = _implementation
