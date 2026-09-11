"""Compatibility import for shared upcoming-artifact transfer."""

import sys

from src.artifacts import upcoming_transfer as _implementation

sys.modules[__name__] = _implementation
