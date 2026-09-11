"""Compatibility import; implementation is owned by ``src.prediction.comparison``."""

import sys

from src.prediction import comparison as _implementation

sys.modules[__name__] = _implementation
