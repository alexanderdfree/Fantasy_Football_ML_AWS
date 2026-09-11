"""Compatibility import; implementation is owned by ``src.data.live_schedule``."""

import sys

from src.data import live_schedule as _implementation

sys.modules[__name__] = _implementation
