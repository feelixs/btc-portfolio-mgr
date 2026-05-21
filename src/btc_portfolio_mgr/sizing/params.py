"""Sizing parameters for the fractional Kelly + threshold sizer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingParams:
    kelly_fraction: float = 0.20
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    rebalance_threshold: float = 0.02


DEFAULT_PARAMS = SizingParams()
