"""Compatibility import; implementation is owned by ``src.data.espn_projections``."""

import sys

from src.data import espn_projections as _implementation

sys.modules[__name__] = _implementation
