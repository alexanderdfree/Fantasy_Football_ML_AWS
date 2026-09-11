"""Compatibility alias for :mod:`src.artifacts.model_sync`.

Keep one module object so existing imports and monkeypatches share state.
"""

import sys

from src.artifacts import model_sync as _implementation

sys.modules[__name__] = _implementation
