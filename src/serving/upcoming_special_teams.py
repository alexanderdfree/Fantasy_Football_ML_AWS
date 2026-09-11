"""Compatibility import; implementation is owned by ``src.prediction.upcoming_special_teams``."""

import sys

from src.prediction import upcoming_special_teams as _implementation

sys.modules[__name__] = _implementation
