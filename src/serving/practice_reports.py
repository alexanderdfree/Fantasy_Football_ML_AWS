"""Compatibility import; implementation is owned by ``src.data.practice_reports``."""

import sys

from src.data import practice_reports as _implementation

sys.modules[__name__] = _implementation
