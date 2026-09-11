"""Compatibility import; implementation is owned by ``src.data.roster_meta``."""

import sys

from src.data import roster_meta as _implementation

sys.modules[__name__] = _implementation
