"""Compatibility import; implementation is owned by ``src.data.forecast_weather``."""

import sys

from src.data import forecast_weather as _implementation

sys.modules[__name__] = _implementation
