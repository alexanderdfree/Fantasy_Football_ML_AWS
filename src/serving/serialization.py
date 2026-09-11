"""Compatibility import; implementation is owned by ``src.contracts.serialization``."""

import sys

from src.contracts import serialization as _implementation

sys.modules[__name__] = _implementation
