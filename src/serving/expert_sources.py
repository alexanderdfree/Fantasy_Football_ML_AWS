"""Compatibility import; implementation is owned by ``src.data.expert_sources``."""

import sys

from src.data import expert_sources as _implementation

sys.modules[__name__] = _implementation
