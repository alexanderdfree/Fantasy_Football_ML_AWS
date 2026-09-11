"""Compatibility import; implementation is owned by ``src.data.live_build``."""

import sys

from src.data import live_build as _implementation

sys.modules[__name__] = _implementation
