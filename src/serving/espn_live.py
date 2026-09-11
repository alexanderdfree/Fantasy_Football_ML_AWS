"""Compatibility import; implementation is owned by ``src.data.espn_live``."""

import sys

from src.data import espn_live as _implementation

sys.modules[__name__] = _implementation
