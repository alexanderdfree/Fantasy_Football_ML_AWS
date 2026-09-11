"""Compatibility import; implementation is owned by ``src.data.live_sources``."""

import sys

from src.data import live_sources as _implementation

sys.modules[__name__] = _implementation
