"""Compatibility import; implementation is owned by ``src.data.roster_identity``."""

import sys

from src.data import roster_identity as _implementation

sys.modules[__name__] = _implementation
