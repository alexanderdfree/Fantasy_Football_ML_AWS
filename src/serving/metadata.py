"""Compatibility import; implementation is owned by ``src.artifacts.position_metadata``."""

import sys

from src.artifacts import position_metadata as _implementation

sys.modules[__name__] = _implementation
