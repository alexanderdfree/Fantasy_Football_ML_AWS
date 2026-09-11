"""Compatibility import; implementation is owned by ``src.data.live_qbr``."""

import sys

from src.data import live_qbr as _implementation

sys.modules[__name__] = _implementation
