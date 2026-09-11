"""Compatibility import; implementation is owned by ``src.contracts.upcoming_status``."""

import sys

from src.contracts import upcoming_status as _implementation

sys.modules[__name__] = _implementation
