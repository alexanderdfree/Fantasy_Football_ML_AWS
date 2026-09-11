"""Compatibility alias for :mod:`src.artifacts.artifact_gc`.

Keep one module object so existing imports and monkeypatches share state.
"""

import sys

from src.artifacts import artifact_gc as _implementation

sys.modules[__name__] = _implementation
